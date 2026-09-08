#!/usr/bin/env python3
"""Stream Video-MME clips out of HuggingFace instead of downloading the dataset.

Video-MME ships its 900 videos as 20 zips of ~5.2 GB (101 GB total), so
`hf_hub_download` per video is not possible; there are no per-video files.
But zip is a random-access format: the central directory sits at EOF and each
member can be pulled on its own. `HfFileSystem` gives a seekable handle backed
by HTTP range requests, so `zipfile` reads the directory (a few hundred KB) and
then only the bytes of the one member asked for.

Videos land in a disk cache with an LRU cap, and the next clip is prefetched in
the background while the current one is being evaluated.

  src = HFVideoSource(cache_dir="/scratch/me/vmme_cache", max_gb=20)
  src.build_index()                      # ~30s once, then cached to JSON
  path = src.path_for("026dzf-vc5g")     # local mp4, fetched if absent
"""
import json
import os
import threading
import time
import zipfile
from collections import OrderedDict

DEFAULT_REPO = "lmms-eval/Video-MME"


class HFVideoSource:
    def __init__(self, repo_id=DEFAULT_REPO, repo_type="dataset",
                 zip_prefix="videos_chunked_", cache_dir=None, max_gb=20.0,
                 index_path=None, revision=None, token=None, prefetch=True,
                 verbose=True, policy="keep"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.zip_prefix = zip_prefix
        self.cache_dir = os.path.abspath(cache_dir or ".hf_video_cache")
        self.max_bytes = int(max_gb * 1e9)
        self.index_path = index_path or os.path.join(self.cache_dir, "_index.json")
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN")
        self.verbose = verbose
        self.index = None
        # Access pattern here is a repeated sequential scan: every sweep cell walks
        # all 900 videos in the same order. Plain LRU is pathological for that:
        # cell 1 ends holding the tail, cell 2 starts at the head, hit rate ~0, and
        # you re-download the whole 101 GB every cell.
        #   "keep" (default): once the cache is full, new arrivals are marked
        #     transient and are the first evicted, so a stable resident prefix
        #     survives across cells. Later cells hit on that prefix.
        #   "lru": textbook LRU. Correct, but thrashes on this workload.
        if policy not in ("keep", "lru"):
            raise ValueError("policy must be 'keep' or 'lru'")
        self.policy = policy
        self._transient = OrderedDict()

        os.makedirs(self.cache_dir, exist_ok=True)
        self._fs = None
        self._zips = {}                       # zip path -> (handle, ZipFile)
        self._zlock = {}                      # zip path -> Lock (ZipFile isn't reentrant)
        self._lock = threading.Lock()
        self._inuse = set()                   # never evict what a worker holds open
        self._pf_thread = None
        self._pf_queue = OrderedDict()
        self._pf_cv = threading.Condition()
        self._pf_stop = False
        self.stats = {"hits": 0, "fetched": 0, "bytes": 0, "evicted": 0,
                      "fetch_s": 0.0, "prefetch_errors": 0}
        self._last_prefetch_error = None
        if prefetch:
            self._start_prefetch()

    # ---------- remote plumbing ----------

    def _root(self):
        p = f"{self.repo_id}"
        if self.repo_type == "dataset":
            p = f"datasets/{p}"
        return p

    def _filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def _zipfile(self, rel):
        """Cached (handle, ZipFile) for one remote zip. Parsing the central
        directory costs a round trip, so it is done once per zip per process."""
        with self._lock:
            if rel in self._zips:
                return self._zips[rel], self._zlock[rel]
        fs = self._filesystem()
        full = f"{self._root()}/{rel}"
        h = fs.open(full, "rb", revision=self.revision) if self.revision else fs.open(full, "rb")
        zf = zipfile.ZipFile(h)
        with self._lock:
            self._zips[rel] = (h, zf)
            self._zlock[rel] = threading.Lock()
            return self._zips[rel], self._zlock[rel]

    def list_zips(self):
        fs = self._filesystem()
        root = self._root()
        out = []
        for p in fs.ls(root, detail=False):
            b = os.path.basename(p)
            if b.startswith(self.zip_prefix) and b.endswith(".zip"):
                out.append(b)
        return sorted(out)

    # ---------- index ----------

    def build_index(self, force=False, zips=None):
        """video_id -> [zip, member, size]. Reads only central directories."""
        if not force and os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.index = json.load(f)
            if self.verbose:
                print(f"[hf] index: {len(self.index)} videos (cached {self.index_path})",
                      flush=True)
            return self.index
        names = zips if zips is not None else self.list_zips()
        idx = {}
        for i, rel in enumerate(names, 1):
            (_, zf), lk = self._zipfile(rel)
            with lk:
                infos = zf.infolist()
            for info in infos:
                if info.is_dir() or not info.filename.lower().endswith(
                        (".mp4", ".mkv", ".webm", ".mov")):
                    continue
                vid = os.path.splitext(os.path.basename(info.filename))[0]
                idx[vid] = [rel, info.filename, info.file_size]
            if self.verbose:
                print(f"[hf] indexed {rel} ({i}/{len(names)}) -> {len(idx)} videos",
                      flush=True)
        self.index = idx
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(idx, f)
        os.replace(tmp, self.index_path)
        if self.verbose:
            tot = sum(v[2] for v in idx.values())
            print(f"[hf] index built: {len(idx)} videos, {tot/1e9:.1f} GB remote "
                  f"-> {self.index_path}", flush=True)
        return idx

    # ---------- cache ----------

    def _local(self, vid, member):
        return os.path.join(self.cache_dir, vid + os.path.splitext(member)[1])

    def _cache_bytes(self):
        t = 0
        for f in os.scandir(self.cache_dir):
            if f.is_file() and not f.name.startswith("_"):
                t += f.stat().st_size
        return t

    def _candidates(self):
        with self._lock:
            inuse, transient = set(self._inuse), set(self._transient)
        res, tra = [], []
        for f in os.scandir(self.cache_dir):
            if not f.is_file() or f.name.startswith("_") or f.path in inuse:
                continue
            st = f.stat()
            (tra if f.path in transient else res).append((st.st_atime, f.path, st.st_size))
        res.sort(); tra.sort()
        return res, tra

    def _drop(self, path, size, total):
        try:
            os.remove(path)
            self.stats["evicted"] += 1
            with self._lock:
                self._transient.pop(path, None)
            return total - size
        except OSError:
            return total

    def _evict_for(self, need):
        """Free room for `need` bytes; return (bytes_in_cache, admit_as_resident).

        Transients always go first. Under policy="keep" nothing else happens:
        resident files are never evicted, and if the cache is still full the new
        arrival is admitted as transient (so it is first out next time). That
        admission control is what makes repeated scans work. Evicting residents
        to make room turns a sequential scan into a 0% hit rate.
        """
        res, tra = self._candidates()
        total = self._cache_bytes()
        for _, path, size in tra:
            if total + need <= self.max_bytes:
                break
            total = self._drop(path, size, total)
        if total + need <= self.max_bytes:
            return total, True
        if self.policy == "keep":
            return total, False              # don't touch residents; admit transient
        for _, path, size in res:            # policy == "lru"
            if total + need <= self.max_bytes:
                break
            total = self._drop(path, size, total)
        return total, True

    def path_for(self, vid, hold=False):
        """Local path for `vid`, range-fetching from the remote zip if absent."""
        if self.index is None:
            self.build_index()
        ent = self.index.get(vid)
        if ent is None:
            raise KeyError(f"{vid} not in Video-MME index ({len(self.index)} videos). "
                           f"Rebuild with build_index(force=True) if the repo changed.")
        rel, member, size = ent
        dst = self._local(vid, member)

        if os.path.exists(dst) and os.path.getsize(dst) == size:
            os.utime(dst, None)                       # refresh LRU position
            self.stats["hits"] += 1
            if hold:
                with self._lock:
                    self._inuse.add(dst)
            return dst

        if hold:
            with self._lock:
                self._inuse.add(dst)
        _, resident = self._evict_for(size)
        if not resident:
            with self._lock:
                self._transient[dst] = 1
        (_, zf), lk = self._zipfile(rel)
        tmp = dst + f".part{os.getpid()}"
        t0 = time.time()
        with lk:
            with zf.open(member) as fin, open(tmp, "wb") as fout:
                while True:
                    b = fin.read(8 << 20)
                    if not b:
                        break
                    fout.write(b)
        os.replace(tmp, dst)
        dt = time.time() - t0
        self.stats["fetched"] += 1
        self.stats["bytes"] += size
        self.stats["fetch_s"] += dt
        if self.verbose:
            print(f"[hf] {vid} {size/1e6:.1f}MB in {dt:.1f}s "
                  f"({size/1e6/max(dt, .01):.1f}MB/s)", flush=True)
        return dst

    def release(self, vid):
        with self._lock:
            for p in list(self._inuse):
                if os.path.splitext(os.path.basename(p))[0] == vid:
                    self._inuse.discard(p)

    # ---------- prefetch ----------

    def _start_prefetch(self):
        def worker():
            while True:
                with self._pf_cv:
                    while not self._pf_queue and not self._pf_stop:
                        self._pf_cv.wait()
                    if self._pf_stop:
                        return
                    vid, _ = self._pf_queue.popitem(last=False)
                try:
                    self.path_for(vid)
                except Exception as e:
                    # Prefetch is best effort; a failure must not kill the eval,
                    # since path_for() retries synchronously. Record it: a dead
                    # prefetch thread looks like a slow network.
                    with self._lock:
                        self.stats["prefetch_errors"] += 1
                        self._last_prefetch_error = f"{type(e).__name__}: {e}"
        self._pf_thread = threading.Thread(target=worker, daemon=True)
        self._pf_thread.start()

    def queue(self, vids):
        """Hint the next clips. Called with the upcoming slice of the eval order."""
        with self._pf_cv:
            for v in vids:
                self._pf_queue[v] = 1
            self._pf_cv.notify()

    def close(self):
        with self._pf_cv:
            self._pf_stop = True
            self._pf_cv.notify_all()
        with self._lock:
            for h, zf in self._zips.values():
                try:
                    zf.close(); h.close()
                except Exception:
                    pass
            self._zips.clear()

    def summary(self):
        s = self.stats
        n = s["hits"] + s["fetched"]
        mbps = s["bytes"] / 1e6 / max(s["fetch_s"], 0.01)
        out = (f"[hf] {s['fetched']} fetched ({s['bytes']/1e9:.1f} GB @ {mbps:.1f} MB/s), "
               f"{s['hits']} cache hits of {n} lookups, {s['evicted']} evicted")
        if s["prefetch_errors"]:
            out += (f" | {s['prefetch_errors']} prefetch errors "
                    f"(last: {self._last_prefetch_error})")
        return out

    # ---------- integration ----------

    def make_resolver(self, fallback=None):
        """Drop-in replacement for make_fix_path()'s fix_path.

        Takes a dataset-relative path like `videos/<id>.mp4` and returns a local
        file. A local copy under the normal base_path always wins, so a partly
        downloaded dataset is used where it exists and streamed where it isn't.
        """
        def resolve(video_path):
            if fallback is not None:
                p = fallback(video_path)
                if os.path.exists(p):
                    return p
            vid = os.path.splitext(os.path.basename(video_path))[0]
            p = self.path_for(vid)
            if not os.path.exists(p):
                # Sweep cells share one cache dir; another process's eviction can
                # unlink a file between fetch and read. Eviction picks the oldest
                # atime and this file was touched above, so a miss is rare. One
                # retry is cheaper than a crashed 2700-item job.
                p = self.path_for(vid)
            return p
        return resolve
