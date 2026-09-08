#!/usr/bin/env python3
"""Emit the project page's charts as inline SVG instead of PNG.

Inline SVG scales on retina displays, follows dark mode, and keeps text visible to
search and screen readers. The marks are elements and the colours are CSS variables,
so the page can style and hover them.

Marks carry `data-*` and a `<title>`, so the page's own script can attach tooltips without
knowing anything about how the chart was built.

  python3 src/analysis/site/make_svg_charts.py --emit ablation      > /tmp/ablation.svg
  python3 src/analysis/site/make_svg_charts.py --inject motionblind.io/index.html

Colours are never hard-coded here. Every fill is a CSS variable such as `var(--blue)`,
resolved by the page's stylesheet. One SVG is then correct in light and dark mode.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from collections import Counter

W, H = 760, 360          # viewBox units; the SVG scales to its container
# Generous gutters: l holds the rotated axis title and the tick labels without
# them touching, t clears the panel title, b clears the x label under the ticks.
PAD = {"l": 64, "r": 22, "t": 38, "b": 54}


def esc(t):
    return html.escape(str(t), quote=True)


def nice_ticks(ymax):
    """Round tick steps. ymax//5 gives 14/28/42 for a 70% axis, which reads as noise.
    Ticks land on round numbers instead."""
    for step in (5, 10, 20, 25, 50):
        if ymax / step <= 8:
            return list(range(0, int(ymax) + 1, step))
    return [0, int(ymax)]


# ---------------------------------------------------------------- reference-line labels
# A chance/baseline label sits inside the plot, where a series line or a bar runs straight
# through it. A label pinned to one fixed corner can land on a mark ("chance 50%"
# across the HORNet line). Instead: propose a handful of anchor positions, test each
# against the marks already placed, and keep the first that collides with nothing.
CH_W, CH_H = 6.05, 11.0          # avg glyph advance / cap-to-descender at font-size 11px


LINE_H = 12.0                    # baseline-to-baseline for a wrapped label


def _lbox(text, anchor, x, y, pad=2.0):
    """Bounding box of a label whose last line's baseline is at (x, y).

    A label may carry newlines: "I_Acc chance 6.25" is wider than the widest gutter
    between bar groups. A bare 6.25 rule beside 50%-tall Acc bars would not say
    which series it is the floor for."""
    lines = text.split("\n")
    w = max(len(l) for l in lines) * CH_W
    x1 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    top = y - CH_H - LINE_H * (len(lines) - 1) + 1
    return (x1 - pad, top, x1 + w + pad, y + 3)


def _ref_text(text, anchor, x, y):
    """Render a (possibly wrapped) reference label, last line's baseline at (x, y)."""
    lines = text.split("\n")
    y0 = y - LINE_H * (len(lines) - 1)
    spans = ""
    for i, l in enumerate(lines):
        dy = "" if i == 0 else f' dy="{LINE_H:g}"'
        spans += f'<tspan x="{x:.1f}"{dy}>{esc(l)}</tspan>'
    return (f'<text class="cchancet" text-anchor="{anchor}" '
            f'x="{x:.1f}" y="{y0:.1f}">{spans}</text>')


def _rects_hit(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _seg_hits(p, q, r):
    """Exact segment-vs-rect test. Using a segment's bounding box instead would report a
    collision across most of the panel for any diagonal, rejecting every candidate."""
    for pt in (p, q):
        if r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3]:
            return True

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    corners = ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))
    for i in range(4):
        c, d = corners[i], corners[(i + 1) % 4]
        d1, d2 = cr(c, d, p), cr(c, d, q)
        d3, d4 = cr(p, q, c), cr(p, q, d)
        if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
            return True
    return False


def _place_ref_label(text, cands, obstacles, bounds):
    """`obstacles`: ("rect", box) or ("seg", p, q). Returns the first candidate that hits
    nothing, else the one that hits least; never nothing, so the label always renders."""
    best, best_n = None, None
    for anchor, x, y in cands:
        box = _lbox(text, anchor, x, y)
        if box[1] < bounds[1] or box[3] > bounds[3]:
            continue                      # would escape the plot area vertically
        n = sum(1 for ob in obstacles
                if (_rects_hit(box, ob[1]) if ob[0] == "rect"
                    else _seg_hits(ob[1], ob[2], box)))
        if n == 0:
            return anchor, x, y
        if best_n is None or n < best_n:
            best, best_n = (anchor, x, y), n
    return best or cands[0]


def _axes(plot_w, plot_h, ticks, ymax, ylab, x0=0, y0=0):
    """Y gridlines + labels, and the y-axis title. Shared by both chart kinds."""
    out = []
    for v in ticks:
        y = y0 + plot_h - (v / ymax) * plot_h
        out.append(f'<line class="cg" x1="{x0}" y1="{y:.1f}" '
                   f'x2="{x0 + plot_w}" y2="{y:.1f}"/>')
        out.append(f'<text class="ct ctr" x="{x0 - 8}" y="{y + 4:.1f}">{v:g}</text>')
    out.append(f'<text class="cax" transform="translate({x0 - 46},{y0 + plot_h / 2}) '
               f'rotate(-90)">{esc(ylab)}</text>')
    return out


def bar_chart(groups, series, values, ymax, ylab, chance=None, chance_label=None,
              decimals=1, na_label="n/a", classes=None):
    """Grouped bars. `values[series][group]` may hold None -> drawn as an explicit
    'n/a' label, never as a zero bar: a zero bar and an unrun cell look identical and
    mean opposite things."""
    pw = W - PAD["l"] - PAD["r"]
    ph = H - PAD["t"] - PAD["b"]
    x0, y0 = PAD["l"], PAD["t"]
    step = pw / len(groups)
    n = len(series)
    # Colour is a property of the series, named explicitly. The same entity keeps its
    # hue across figures (TimeBlind is green everywhere), not the slot it lands in.
    cls = classes or [f"k{i}" for i in range(n)]
    bw = min(46, (step * 0.72) / n)

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    out += _axes(pw, ph, nice_ticks(ymax), ymax, ylab, x0, y0)
    blocked = []                 # what the chance label must not be printed on top of

    for gi, g in enumerate(groups):
        cx = x0 + step * (gi + 0.5)
        for si, s in enumerate(series):
            v = values[s][gi]
            bx = cx + (si - (n - 1) / 2) * bw - bw / 2
            if v is None:
                out.append(f'<text class="cna" x="{bx + bw / 2:.1f}" '
                           f'y="{y0 + ph - 6}">{na_label}</text>')
                continue
            bh = (v / ymax) * ph
            by = y0 + ph - bh
            out.append(
                f'<rect class="cbar {cls[si]}" x="{bx:.1f}" y="{by:.1f}" '
                f'width="{bw:.1f}" height="{max(bh, 0.6):.1f}" rx="2" '
                f'data-series="{esc(s)}" data-group="{esc(g)}" data-panel="" '
                f'data-unit="{esc(ylab)}" data-value="{v:.{decimals}f}">'
                f'<title>{esc(s)} · {esc(g)}: {v:.{decimals}f}</title></rect>')
            vt = f'{v:.{decimals}f}'
            out.append(f'<text class="cval" x="{bx + bw / 2:.1f}" '
                       f'y="{by - 5:.1f}">{vt}</text>')
            blocked.append(("rect", (bx, by, bx + bw, y0 + ph)))
            blocked.append(("rect", _lbox(vt, "middle", bx + bw / 2, by - 5)))
        out.append(f'<text class="ct" x="{cx:.1f}" y="{y0 + ph + 20}">{esc(g)}</text>')

    if chance is not None:
        cy = y0 + ph - (chance / ymax) * ph
        out.append(f'<line class="cchance" x1="{x0}" y1="{cy:.1f}" '
                   f'x2="{x0 + pw}" y2="{cy:.1f}"/>')
        txt = chance_label or "chance"
        # Above/below the rule at either end first (the conventional spots), then centred
        # in the gaps between bar groups. On a 0-70 axis a 6.25% rule sits deep inside
        # every bar, so the inter-group gutters are often the only clear space.
        cands = [("start", x0 + 6, cy - 7), ("end", x0 + pw - 4, cy - 7)]
        cands += [("middle", x0 + step * (gi + 1), cy - 7)
                  for gi in range(len(groups) - 1)]
        cands += [("start", x0 + 6, cy + 15), ("end", x0 + pw - 4, cy + 15)]
        cands += [("middle", x0 + step * (gi + 1), cy + 15)
                  for gi in range(len(groups) - 1)]
        anc, lx, ly = _place_ref_label(txt, cands, blocked, (x0, y0, x0 + pw, y0 + ph))
        out.append(_ref_text(txt, anc, lx, ly))
    out.append("</svg>")
    return "\n".join(out)


def bar_panels(panels, groups, series, ymax, ylab, chance=None, classes=None,
               na_label="not run"):
    """Grouped bars as small multiples, shared y range across panels.

    Shared on purpose: the panels hold the same measure (I_Acc) for two models.
    Per-panel scaling would redraw Molmo2's flat 1.7 as a dramatic spread."""
    cols = len(panels)
    gap = 54
    pw = (W - PAD["l"] * cols - PAD["r"] - gap * (cols - 1)) / cols
    ph = H - PAD["t"] - PAD["b"]
    y0 = PAD["t"]
    n = len(series)
    cls = classes or [f"k{i}" for i in range(n)]
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    for pi, (title, vals) in enumerate(panels):
        x0 = PAD["l"] + pi * (pw + PAD["l"] + gap)
        out += _axes(pw, ph, nice_ticks(ymax), ymax, ylab, x0, y0)
        out.append(f'<text class="cpt" x="{x0 + pw / 2:.1f}" y="{y0 - 12}">'
                   f'{esc(title)}</text>')
        step = pw / len(groups)
        bw = min(26, (step * 0.7) / n)
        blocked = []
        for gi, g in enumerate(groups):
            cx = x0 + step * (gi + 0.5)
            got = vals.get(g)
            if got is None or all(v is None for v in got):
                # The whole cell is missing, so it gets one mark. One dash per absent
                # series would print two glyphs a few px apart and read as a smudge.
                out.append(f'<text class="cna" x="{cx:.1f}" y="{y0 + ph - 6}">'
                           f'not run</text>')
                out.append(f'<text class="ct" x="{cx:.1f}" y="{y0 + ph + 19}">'
                           f'{esc(g.replace(" (GRPO)", ""))}</text>')
                continue
            for si, sname in enumerate(series):
                v = None if got is None else got[si]
                bx = cx + (si - (n - 1) / 2) * bw - bw / 2
                if v is None:
                    # A missing cell is drawn as an explicit label. A zero bar would say
                    # "the model scored nothing", which is the opposite of "not run".
                    out.append(f'<text class="cna" x="{bx + bw / 2:.1f}" '
                               f'y="{y0 + ph - 6}">&#8211;</text>')
                    continue
                bh = (v / ymax) * ph
                by = y0 + ph - bh
                out.append(
                    f'<rect class="cbar {cls[si]}" x="{bx:.1f}" y="{by:.1f}" '
                    f'width="{bw:.1f}" height="{max(bh, 0.6):.1f}" rx="2" '
                    f'data-series="{esc(sname)}" data-group="{esc(g)}" '
                    f'data-panel="{esc(title)}" data-unit="{esc(ylab)}" '
                    f'data-value="{v:.1f}"><title>{esc(sname)} &#183; {esc(g)}: '
                    f'{v:.1f}</title></rect>')
                out.append(f'<text class="cval" x="{bx + bw / 2:.1f}" '
                           f'y="{by - 5:.1f}">{v:.1f}</text>')
                blocked.append(("rect", (bx, by, bx + bw, y0 + ph)))
                blocked.append(("rect", _lbox(f"{v:.1f}", "middle",
                                              bx + bw / 2, by - 5)))
            lab = g.replace(" (GRPO)", "")
            out.append(f'<text class="ct" x="{cx:.1f}" y="{y0 + ph + 19}">'
                       f'{esc(lab)}</text>')
        if chance is not None:
            cy = y0 + ph - (chance / ymax) * ph
            out.append(f'<line class="cchance" x1="{x0}" y1="{cy:.1f}" '
                       f'x2="{x0 + pw}" y2="{cy:.1f}"/>')
            txt = "chance 6.25"
            cands = [("start", x0 + 6, cy - 7), ("end", x0 + pw - 4, cy - 7),
                     ("start", x0 + 6, cy + 15), ("end", x0 + pw - 4, cy + 15)]
            cands += [("middle", x0 + step * (gi + 1), cy - 7)
                      for gi in range(len(groups) - 1)]
            anc, lx, ly = _place_ref_label(txt, cands, blocked,
                                           (x0, y0, x0 + pw, y0 + ph))
            out.append(_ref_text(txt, anc, lx, ly))
    out.append("</svg>")
    return "\n".join(out)

def line_panels(panels, xs, ylab_of, ymax, chance_of, chance_label_of,
                classes=None):
    """`ymax` may be a scalar (one shared range) or a list (one per panel).

    A shared range fits panels that show the same measure on different data;
    auto-scaling each would inflate whichever has the smaller spread. Per-panel
    ranges fit panels that show different measures (I_Acc vs yes-rate), where a
    common axis would flatten the smaller one into a flat line."""
    cols = len(panels)
    gap = 46
    pw = (W - PAD["l"] * cols - PAD["r"] - gap * (cols - 1)) / cols
    ph = H - PAD["t"] - PAD["b"]
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    lo, hi = min(xs), max(xs)
    ymaxes = ymax if isinstance(ymax, (list, tuple)) else [ymax] * cols

    def px(v, x0):
        import math
        a, b = math.log2(lo), math.log2(hi)
        return x0 + (0 if b == a else (math.log2(v) - a) / (b - a)) * pw

    for pi, (title, series) in enumerate(panels):
        x0 = PAD["l"] + pi * (pw + PAD["l"] + gap)
        y0 = PAD["t"]
        ym = ymaxes[pi]
        out += _axes(pw, ph, nice_ticks(ym), ym, ylab_of[pi], x0, y0)
        out.append(f'<text class="cpt" x="{x0 + pw / 2:.1f}" y="{y0 - 10}">{esc(title)}</text>')

        # The rule is drawn under the data. Its label is placed last, after the line
        # positions are known, and painted on top.
        ch = chance_of[pi]
        cy = None
        if ch is not None:
            cy = y0 + ph - (ch / ym) * ph
            out.append(f'<line class="cchance" x1="{x0}" y1="{cy:.1f}" '
                       f'x2="{x0 + pw}" y2="{cy:.1f}"/>')

        blocked = []
        cls = classes or [f'k{i}' for i in range(len(series))]
        for si, (name, pts) in enumerate(series):
            xy = [(px(x, x0), y0 + ph - (pts[x] / ym) * ph) for x in sorted(pts)]
            d = " ".join(f'{"M" if k == 0 else "L"}{cxp:.1f},{cyp:.1f}'
                         for k, (cxp, cyp) in enumerate(xy))
            out.append(f'<path class="cline {cls[si]}" d="{d}"/>')
            blocked += [("seg", xy[k], xy[k + 1]) for k in range(len(xy) - 1)]
            for (cxp, cyp), x in zip(xy, sorted(pts)):
                blocked.append(("rect", (cxp - 5, cyp - 5, cxp + 5, cyp + 5)))
                out.append(
                    f'<circle class="cdot {cls[si]}" cx="{cxp:.1f}" cy="{cyp:.1f}" r="3.6" '
                    f'data-series="{esc(name)}" data-group="{x}" data-panel="{esc(title)}" '
                    f'data-unit="{esc(ylab_of[pi])}" data-value="{pts[x]:.1f}">'
                    f'<title>{esc(name)} · {x} frames: {pts[x]:.1f}</title></circle>')

        if cy is not None:
            txt = chance_label_of[pi]
            cands = [("start", x0 + 6, cy - 7), ("end", x0 + pw - 4, cy - 7),
                     ("start", x0 + 6, cy + 15), ("end", x0 + pw - 4, cy + 15),
                     ("middle", x0 + pw / 2, cy - 7), ("middle", x0 + pw / 2, cy + 15)]
            anc, lx, ly = _place_ref_label(txt, cands, blocked,
                                           (x0, y0, x0 + pw, y0 + ph))
            out.append(_ref_text(txt, anc, lx, ly))

        for x in xs:
            out.append(f'<text class="ct" x="{px(x, x0):.1f}" y="{y0 + ph + 20}">{x}</text>')
        out.append(f'<text class="cax cxl" x="{x0 + pw / 2:.1f}" '
                   f'y="{y0 + ph + 36}">frames sent</text>')
    out.append("</svg>")
    return "\n".join(out)



# ================================================================= horizontal forms
# Model names are long ("Gemma-4-12B-it", "Gemini 3.1 Pro"). On a vertical bar chart they
# either overlap or have to be rotated; read left-to-right in a gutter they need no
# tricks, and the bars then run in the direction the eye already scans for magnitude.
def hbar_chart(rows, xmax, xlab, chance=None, chance_label=None, gutter=118,
               rowh=30, decimals=1, note_of=None, mark_of=None):
    """`rows` = [(label, value, class, group), ...]. One bar per row, direct-labelled.

    `note_of` appends to the value label, `mark_of` to the row name. A caveat marker
    belongs on the name. The value label of a short bar sits near the axis. A trailing
    mark there lands on the chance rule.
    """
    ph = rowh * len(rows)
    H2 = PAD["t"] + ph + 46
    x0, y0 = gutter, PAD["t"]
    pw = W - gutter - PAD["r"] - 26
    out = [f'<svg class="chart" viewBox="0 0 {W} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    for v in nice_ticks(xmax):
        x = x0 + (v / xmax) * pw
        out.append(f'<line class="cg" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+ph}"/>')
        out.append(f'<text class="ct" x="{x:.1f}" y="{y0 + ph + 18}">{v:g}</text>')
    out.append(f'<text class="cax cxl" x="{x0 + pw / 2:.1f}" '
               f'y="{y0 + ph + 36}">{esc(xlab)}</text>')
    blocked = []
    for i, (label, v, cls, grp) in enumerate(rows):
        cy = y0 + rowh * i + rowh / 2
        bh = min(rowh - 11, 18)
        bw = (v / xmax) * pw
        rowlab = esc(label) + ("" if not mark_of else esc(mark_of.get(label, "")))
        out.append(f'<text class="ct ctr crow" x="{x0 - 10}" '
                   f'y="{cy + 4:.1f}">{rowlab}</text>')
        out.append(
            f'<rect class="cbar {cls}" x="{x0}" y="{cy - bh / 2:.1f}" '
            f'width="{max(bw, 0.6):.1f}" height="{bh}" rx="2" '
            f'data-series="{esc(grp)}" data-group="{esc(label)}" data-panel="" '
            f'data-unit="{esc(xlab)}" data-value="{v:.{decimals}f}">'
            f'<title>{esc(label)} &#183; {esc(grp)}: {v:.{decimals}f}</title></rect>')
        txt = f"{v:.{decimals}f}" + ("" if not note_of else note_of.get(label, ""))
        out.append(f'<text class="cval cl" x="{x0 + bw + 6:.1f}" '
                   f'y="{cy + 4:.1f}">{esc(txt)}</text>')
        blocked.append(("rect", (x0, cy - bh / 2, x0 + bw + 8 + len(txt) * 6.3,
                                 cy + bh / 2)))
    if chance is not None:
        cx = x0 + (chance / xmax) * pw
        out.append(f'<line class="cchance" x1="{cx:.1f}" y1="{y0}" '
                   f'x2="{cx:.1f}" y2="{y0 + ph}"/>')
        txt = chance_label or "chance"
        # Above the first row, in the top padding. Every bar in a horizontal chart starts
        # at the same x, so a 6.25%-of-100 rule crosses all of them. No clear space for
        # the label exists inside the plot.
        cands = [("start", cx + 6, y0 - 9), ("end", cx - 6, y0 - 9),
                 ("start", cx + 6, y0 + 12), ("end", cx - 6, y0 + 12),
                 ("start", cx + 6, y0 + ph - 4), ("end", cx - 6, y0 + ph - 4)]
        anc, lx, ly = _place_ref_label(txt, cands, blocked, (x0, 8, x0 + pw, y0 + ph))
        out.append(_ref_text(txt, anc, lx, ly))
    out.append("</svg>")
    return "\n".join(out)


def hbar_panels(panels, labels, xmax, xlab, chance=None, gutter=112, rowh=26,
                classes=None):
    """Small multiples of horizontal bars sharing one row axis, drawn once on the left.

    `panels` = [(title, {label: value}), ...]. Colour here carries the model group (open
    vs frontier), not identity; identity is the row label. Five models then never need
    five hues, and the palette never stretches past a validated pair."""
    cols = len(panels)
    gap = 18
    pw = (W - gutter - PAD["r"] - gap * (cols - 1)) / cols
    ph = rowh * len(labels)
    H2 = PAD["t"] + ph + 50
    y0 = PAD["t"]
    out = [f'<svg class="chart" viewBox="0 0 {W} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    for i, lab in enumerate(labels):
        cy = y0 + rowh * i + rowh / 2
        out.append(f'<text class="ct ctr crow" x="{gutter - 10}" y="{cy + 4:.1f}">'
                   f'{esc(lab)}</text>')
    for pi, (title, vals) in enumerate(panels):
        x0 = gutter + pi * (pw + gap)
        out.append(f'<text class="cpt" x="{x0 + pw / 2:.1f}" y="{y0 - 12}">'
                   f'{esc(title)}</text>')
        for v in nice_ticks(xmax):
            x = x0 + (v / xmax) * pw
            out.append(f'<line class="cg" x1="{x:.1f}" y1="{y0}" '
                       f'x2="{x:.1f}" y2="{y0 + ph}"/>')
            out.append(f'<text class="ct" x="{x:.1f}" y="{y0 + ph + 17}">{v:g}</text>')
        if chance is not None:
            cx = x0 + (chance / xmax) * pw
            out.append(f'<line class="cchance" x1="{cx:.1f}" y1="{y0}" '
                       f'x2="{cx:.1f}" y2="{y0 + ph}"/>')
        for i, lab in enumerate(labels):
            v = vals.get(lab)
            if v is None:
                continue
            cy = y0 + rowh * i + rowh / 2
            bh = min(rowh - 10, 16)
            bw = (v / xmax) * pw
            cls = (classes or {}).get(lab, "k-blue")
            out.append(
                f'<rect class="cbar {cls}" x="{x0}" y="{cy - bh / 2:.1f}" '
                f'width="{max(bw, 0.6):.1f}" height="{bh}" rx="2" '
                f'data-series="{esc(lab)}" data-group="{esc(title)}" '
                f'data-panel="{esc(title)}" data-unit="{esc(xlab)}" '
                f'data-value="{v:.1f}"><title>{esc(lab)} &#183; {esc(title)}: '
                f'{v:.1f}</title></rect>')
    out.append(f'<text class="cax cxl" x="{gutter + (W - gutter - PAD["r"]) / 2:.1f}" '
               f'y="{y0 + ph + 36}">{esc(xlab)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def donut(segs, center_lines, unit):
    """Paper Figure 2a: the benchmark's make-up as a ring, each slice carrying its count
    inside and its name outside on a leader line, with a representative frame.

    Every slice is named and numbered in place, so colour here is redundant rather
    than load bearing. The two direction subtypes can then share one hue in two steps
    (the paper does the same). They are still stepped wide enough to clear the CVD
    floor: OKLab dE 13.5 deutan in light mode, 9.4 in dark."""
    import math
    W2, H2 = 760, 430
    cx, cy = 300, 218
    rout, rin = 92, 58
    tot = sum(v for _, v, _, _ in segs)
    out = [f'<svg class="chart dnut" viewBox="0 0 {W2} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    out.append('<defs>')
    for i, _ in enumerate(segs):
        out.append(f'<clipPath id="thumb{i}"><circle cx="0" cy="0" r="34"/></clipPath>')
    out.append('</defs>')

    ang = -90.0                      # start at 12 o'clock, sweep clockwise
    for i, (name, v, sub, thumb) in enumerate(segs):
        sweep = 360.0 * v / tot
        a0, a1 = math.radians(ang), math.radians(ang + sweep)
        big = 1 if sweep > 180 else 0
        p = (f'M{cx + rout * math.cos(a0):.1f},{cy + rout * math.sin(a0):.1f} '
             f'A{rout},{rout} 0 {big} 1 '
             f'{cx + rout * math.cos(a1):.1f},{cy + rout * math.sin(a1):.1f} '
             f'L{cx + rin * math.cos(a1):.1f},{cy + rin * math.sin(a1):.1f} '
             f'A{rin},{rin} 0 {big} 0 '
             f'{cx + rin * math.cos(a0):.1f},{cy + rin * math.sin(a0):.1f} Z')
        out.append(
            f'<path class="arc a{i}" d="{p}" data-series="{esc(name)}" '
            f'data-group="{esc(name)}" data-panel="" data-unit="{esc(unit)}" '
            f'data-value="{v}" data-vlabel="{v} ({100 * v / tot:.0f}%)">'
            f'<title>{esc(name)}: {v} {esc(unit)} ({100 * v / tot:.0f}%) \u00b7 '
            f'{esc(sub)}</title></path>')

        mid = math.radians(ang + sweep / 2)
        mc, ms = math.cos(mid), math.sin(mid)
        out.append(f'<text class="arcn" x="{cx + (rout + rin) / 2 * mc:.1f}" '
                   f'y="{cy + (rout + rin) / 2 * ms + 5:.1f}">{v}</text>')

        # Thumbnail sits on the slice's own bearing, then the name goes further out on the
        # same line; so a label can never be traced back to the wrong slice.
        tx, ty = cx + (rout + 52) * mc, cy + (rout + 52) * ms
        out.append(f'<line class="lead" x1="{cx + (rout + 3) * mc:.1f}" '
                   f'y1="{cy + (rout + 3) * ms:.1f}" x2="{tx - 34 * mc:.1f}" '
                   f'y2="{ty - 34 * ms:.1f}"/>')
        out.append(f'<g transform="translate({tx:.1f},{ty:.1f})">'
                   f'<image href="{esc(thumb)}" x="-34" y="-34" width="68" height="68" '
                   f'preserveAspectRatio="xMidYMid slice" clip-path="url(#thumb{i})"/>'
                   f'<circle class="thring t{i}" cx="0" cy="0" r="34"/></g>')
        lx, ly = cx + (rout + 96) * mc, cy + (rout + 96) * ms
        anc = "start" if mc > 0.12 else ("end" if mc < -0.12 else "middle")
        lines = name.split(" (")
        head = lines[0]
        tail = "(" + lines[1] if len(lines) > 1 else ""
        out.append(f'<text class="arclab" text-anchor="{anc}" x="{lx:.1f}" '
                   f'y="{ly - (6 if tail else 0):.1f}">{esc(head)}</text>')
        if tail:
            out.append(f'<text class="arclab" text-anchor="{anc}" x="{lx:.1f}" '
                       f'y="{ly + 9:.1f}">{esc(tail)}</text>')
        ang += sweep

    for j, (txt, cls) in enumerate(center_lines):
        out.append(f'<text class="{cls}" x="{cx}" y="{cy - 16 + j * 17}">{esc(txt)}</text>')
    out.append("</svg>")
    return "\n".join(out)



def wordcloud(words, W2=760, H2=430):
    """Paper Figure 2b: words sized by how strongly they belong to one benchmark.

    `words` = [(word, weight, side), ...]. Placement is a deterministic Archimedean
    spiral with rectangle collision, largest first. The same input always produces the
    same picture.
    """
    import math
    cx, cy = W2 / 2, H2 / 2
    # Size is dominance within a benchmark, so the two sides are comparable. On one
    # shared scale every TimeBlind word comes out tiny: its 30k-token corpus makes the
    # log-odds z of a word merely absent from MotionBlind bottom out near 3. A word
    # common in MotionBlind's 1.5k tokens reaches 19. That is a property of the corpora,
    # not of how characteristic the word is. It would falsely read as "TimeBlind has
    # no distinctive vocabulary".
    span = {}
    for side in {sd for _, _, sd in words}:
        ws = [w for _, w, sd in words if sd == side]
        span[side] = (min(ws), max(ws))
    placed, out = [], []

    def norm(w, side):
        lo, hi = span[side]
        return (w - lo) / (hi - lo) if hi > lo else 1.0

    def size_of(w, side):
        # sqrt, not linear: glyph area grows with the square of the font size, so a linear
        # map would make the top word look several times more dominant than it is.
        return 13.0 + math.sqrt(norm(w, side)) * 44.0

    ordered = sorted(words, key=lambda t: -norm(t[1], t[2]))

    def hits(b):
        return any(not (b[2] <= o[0] or o[2] <= b[0] or b[3] <= o[1] or o[3] <= b[1])
                   for o in placed)

    for word, weight, side in ordered:
        fs = size_of(weight, side)
        w_ = len(word) * fs * 0.54 + 6
        h_ = fs * 1.02
        pos = None
        # Wider than tall, so the spiral is stretched to match the canvas instead of
        # packing a circle into a rectangle and wasting the corners.
        for i in range(0, 60000):
            a = i * 0.22
            r = 2.4 * a
            x, y = cx + r * math.cos(a) * 1.55, cy + r * math.sin(a) * 0.72
            b = (x - w_ / 2, y - h_ / 2, x + w_ / 2, y + h_ / 2)
            if b[0] < 4 or b[1] < 4 or b[2] > W2 - 4 or b[3] > H2 - 4:
                if r > max(W2, H2) * 1.4:
                    break
                continue
            if not hits(b):
                pos = (x, y, b)
                break
        if pos is None:
            continue
        placed.append(pos[2])
        cls = "k-orange" if side == "MotionBlind" else "k-green"
        # No numeric attributes: the cloud shows which benchmark a word belongs to and
        # roughly how strongly, and hovering says which one. It does not publish a score.
        out.append(
            f'<text class="wc {cls}" x="{pos[0]:.1f}" y="{pos[1] + fs * 0.34:.1f}" '
            f'font-size="{fs:.1f}" data-series="{side}" data-group="{esc(word)}" '
            f'data-panel=""><title>{esc(word)} &#183; {side}</title>{esc(word)}</text>')
    return (f'<svg class="chart cloud" viewBox="0 0 {W2} {H2}" role="img" '
            f'preserveAspectRatio="xMidYMid meet">\n' + "\n".join(out) + "\n</svg>")


def butterfly(rows, left_label, right_label, xlab, xmax=None):
    """`rows` = [(word, left_rate, right_rate), ...]. Each word gets a bar on both sides.

    Bar lengths are rates per 1,000 tokens, not the weighted log-odds z. MotionBlind's
    questions are 1.5k tokens against TimeBlind's 30k. z runs to +19 for a word common
    in MotionBlind but only near -3 for one merely absent; 1476 tokens give no
    confidence about a word never seen. Rates per 1,000 tokens are symmetric, directly
    readable, and show the contrast as the shape of the figure. The z-score still
    picks which words appear; it no longer sets the bar lengths."""
    rowh, gut, pad = 19, 52, 22
    ph = rowh * len(rows)
    H2 = 66 + ph + 48
    y0, mid = 66, W / 2
    half = mid - gut - pad
    xmax = xmax or max(max(l, r) for _, l, r in rows)
    ticks = nice_ticks(xmax)
    out = [f'<svg class="chart bfly" viewBox="0 0 {W} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    out.append(f'<text class="cpt" x="{mid - gut - half / 2:.1f}" y="34">'
               f'{esc(left_label)}</text>')
    out.append(f'<text class="cpt" x="{mid + gut + half / 2:.1f}" y="34">'
               f'{esc(right_label)}</text>')
    for t in ticks:
        for sgn in (-1, 1):
            if t == 0 and sgn < 0:
                continue
            x = mid + sgn * (gut + t / xmax * half)
            out.append(f'<line class="cg" x1="{x:.1f}" y1="{y0 - 10}" x2="{x:.1f}" '
                       f'y2="{y0 + ph + 4}"/>')
            out.append(f'<text class="ct" x="{x:.1f}" y="{y0 + ph + 21}">{t:g}</text>')
    for i, (word, lv, rv) in enumerate(rows):
        cy = y0 + rowh * i + rowh / 2
        out.append(f'<text class="ct cbword" x="{mid:.1f}" y="{cy + 4:.1f}">'
                   f'{esc(word)}</text>')
        for side, v, cls, name in ((-1, lv, "k-green", left_label),
                                   (1, rv, "k-orange", right_label)):
            w_ = v / xmax * half
            bx = mid + gut if side > 0 else mid - gut - w_
            out.append(
                f'<rect class="cbar {cls}" x="{bx:.1f}" y="{cy - 6:.1f}" '
                f'width="{max(w_, 0.8):.1f}" height="12" rx="2" '
                f'data-series="{esc(name)}" data-group="{esc(word)}" data-panel="" '
                f'data-unit="{esc(xlab)}" data-value="{v:.1f}" '
                f'data-vlabel="{v:.1f}"><title>{esc(word)} &#183; {esc(name)}: '
                f'{v:.1f} {esc(xlab)}</title></rect>')
    out.append(f'<text class="cax cxl" x="{mid:.1f}" y="{y0 + ph + 40}">'
               f'{esc(xlab)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------- the paper's figures & tables
def _paper(name):
    return list(csv.DictReader(open(os.path.join("results/paper", name), encoding="utf-8")))


GROUP_CLS = {"open": "k-blue", "frontier": "k-orange", "human": "k-green"}


def chart_leaderboard(a):
    """Paper Table 1, as a chart. One bar clears the floor, one clears the benchmark,
    and the human row sits above both."""
    rows = _paper("table1_leaderboard.csv")
    bars, notes, marks = [], {}, {}
    for r in rows:
        bars.append((r["model"], float(r["MB_IAcc"]), GROUP_CLS[r["group"]], "I_Acc"))
        if r["group"] == "human":
            notes[r["model"]] = "  (5 annotators)"
        elif r["n_instances"] != "60":
            # On the name, not the value: Gemma's bar is short, so anything trailing its
            # value label lands on the dashed chance rule at 6.25.
            marks[r["model"]] = " *"
    return hbar_chart(bars, 100, "MotionBlind I_Acc (%)", chance=6.25,
                      chance_label="chance 6.25", note_of=notes, mark_of=marks)


def chart_percat(a):
    """Paper Figure 4. One panel per motion category, models sharing a single row axis."""
    rows = _paper("by_category.csv")
    show = ["Gemini 3.1 Pro", "Eagle2.5-8B", "Qwen3-VL-4B", "Motion-o (7B)", "Molmo2-8B"]
    cats = ["speed", "magnitude", "direction (translational)",
            "direction (rotational)"]
    lb = {r["model"]: r for r in _paper("table1_leaderboard.csv")}
    panels = [(c.replace("direction (", "dir. ").replace(")", ""),
               {r["model"]: float(r["I_Acc"]) for r in rows
                if r["category"] == c and r["model"] in show})
              for c in cats]
    cls = {m: GROUP_CLS[lb[m]["group"]] for m in show}
    return hbar_panels(panels, show, 80, "I_Acc (%) at uniform-16",
                       chance=6.25, classes=cls)


def chart_budget(a):
    """Paper Figure 5. Open models only, on their own range.

    Gemini and the human ceiling are annotated rather than plotted: at 60 and 91 they
    would compress every open model into one flat line at the axis. That would hide
    the rise-then-plateau result. The truncation is stated on the chart."""
    rows = [r for r in _paper("sweep_motionblind.csv") if r["sampler"] == "uniform"]
    show = ["Eagle2.5-8B", "Qwen3-VL-4B", "Motion-o (7B)", "Molmo2-8B"]
    xs = [1, 4, 8, 16, 24]
    series = [(m, {int(r["num_frames"]): float(r["I_Acc"]) for r in rows
                   if r["model"] == m}) for m in show]
    svg = line_panels([("", series)], xs, ["I_Acc (%)"], 15, [6.25],
                      ["chance 6.25"],
                      classes=["k-blue", "k-red", "k-green", "k-amber"])
    off = ('<text class="coff" x="{:.0f}" y="26">Gemini 3.1 Pro 60.0 and Human 91.3 are '
           'off this scale &#8593;</text>'.format(PAD["l"]))
    return svg.replace("</svg>", off + "\n</svg>")


def chart_selection(a):
    """Paper Table 3. Both panels share one 0-25 range: they are the same measure on two
    suites. Giving each its own range would make Molmo2's flat 1.7 look like a
    spread."""
    mb = {(r["model"], r["sampler"]): float(r["I_Acc"])
          for r in _paper("sweep_motionblind.csv") if r["num_frames"] == "16"}
    tb = {(r["model"], r["sampler"]): float(r["I_Acc"])
          for r in _paper("sweep_timeblind.csv") if r["num_frames"] == "16"}
    NAME_OF = [("Uniform", "uniform"), ("Random", "random"),
               ("HORNet (GRPO)", "hornet"), ("Frame2Clip", "f2cfull")]
    samplers = [d for d, _ in NAME_OF]
    panels = []
    for m in ("Eagle2.5-8B", "Molmo2-8B"):
        vals = {d: (mb.get((m, s)), tb.get((m, s))) for d, s in NAME_OF}
        panels.append((m, vals))
    return bar_panels(panels, samplers, ["MotionBlind", "TimeBlind"], 25,
                      "I_Acc (%) at 16 frames", chance=6.25,
                      classes=["k-orange", "k-green"])


def chart_makeup(a):
    # Dataset composition, computed from the benchmark file itself: the taxonomy id
    # is the leading field of the clip filename (see make_tables.py).
    cats = {**{i: "speed" for i in range(1, 8)}, **{i: "magnitude" for i in range(8, 14)},
            14: "direction (translational)", 15: "direction (rotational)"}
    inst, clips_of = {}, {}
    for line in open("data/data.jsonl", encoding="utf-8"):
        o = json.loads(line)
        cid = int(re.search(r"(\d+)_\d+_\d+\.mp4$", o["video_path"]).group(1))
        c = cats[cid]
        inst[o["index"] // 4] = c
        clips_of.setdefault(c, set()).add(o["video_path"])
    cnt = Counter(inst.values())
    rows = [{"category": c, "instances": cnt[c], "clips": len(clips_of[c]),
             "items": cnt[c] * 4}
            for c in ("speed", "magnitude",
                      "direction (translational)", "direction (rotational)")]
    # One representative, face-blurred frame per category, taken from the frame tiles the
    # browser further down already ships; no new assets.
    # The clip that actually shows the characteristic, not just any clip in the category:
    # swirling for speed, the toy car for magnitude, the cup sliding toward the camera for
    # translation, the bottle rolling for rotation.
    THUMB = {"speed": "figures/frames/02_00_0/293.jpg",
             "magnitude": "figures/frames/09_02_0/55.jpg",
             "direction (translational)": "figures/frames/14_01_0/53.jpg",
             "direction (rotational)": "figures/frames/15_01_1/61.jpg"}
    segs = [(r["category"], int(r["instances"]),
             f'{r["clips"]} clips, {r["items"]} items', THUMB[r["category"]])
            for r in rows]
    tot = sum(s[1] for s in segs)
    clips = sum(int(r["clips"]) for r in rows)
    return donut(segs, [("MotionBlind", "cmid b"), (f"{tot} instances", "cmid"),
                        (f"{clips} clips", "cmid"), (f"{tot * 4} items", "cmid")],
                 "instances")


def chart_words(a):
    """Paper Figure 2b, transcribed. The words, their benchmark and their relative size are
    the published figure's, read off it and kept in src/analysis/site/data/figure2b_words.csv; not
    rescored from the repo, which would draw a different (and differently worded) cloud."""
    rows = list(csv.DictReader(open(a.words_csv)))
    return wordcloud([(r["word"], float(r["dominance"]), r["side"]) for r in rows])


def _two_lobes(pts, aspect_w=0.41, iters=60):
    """Lloyd's k=2 on a point cloud -> [((cx, cy), members), ...] or None.

    y is down-weighted because the plot box is ~2.4x wider than tall. A gap that looks
    square on screen is then not square in data units. Returns None when the two centres
    land close together: a cloud with no real split must not get an annotation marking
    half of it as a distinct group.
    """
    if len(pts) < 20:
        return None
    C = [(0.15, 0.75), (0.55, 0.45)]
    groups = None
    for _ in range(iters):
        groups = [[], []]
        for x, y in pts:
            d0 = (x - C[0][0]) ** 2 + ((y - C[0][1]) * aspect_w) ** 2
            d1 = (x - C[1][0]) ** 2 + ((y - C[1][1]) * aspect_w) ** 2
            groups[0 if d0 < d1 else 1].append((x, y))
        C = [(sum(a for a, _ in g) / len(g), sum(b for _, b in g) / len(g)) if g else C[i]
             for i, g in enumerate(groups)]
    if min(len(g) for g in groups) < 0.15 * len(pts):
        return None
    sep = ((C[0][0] - C[1][0]) ** 2 + ((C[0][1] - C[1][1]) * aspect_w) ** 2) ** 0.5
    if sep < 0.18:
        return None
    return [(C[i], groups[i]) for i in (0, 1)]


def scatter_projection(pts, xlab, title, auc, aspect=2.44, region_note=None):
    """Paper Figure 2c, at the paper's own coordinates.

    The points are not recomputed; they are lifted straight out of the published figure,
    which is vector, so each dot's centre is exact rather than digitised off a raster. The
    plot box keeps the paper's width:height ratio, so the cloud has the shape it has in the
    paper instead of one stretched to fit whatever box the page gives it."""
    x0, y0 = PAD["l"] - 22, 50
    pw = W - x0 - PAD["r"] - 6
    ph = pw / aspect
    H2 = int(y0 + ph + 46)
    ins = 0.035                       # breathing room between the cloud and the frame

    def px(v):
        return x0 + (ins + v * (1 - 2 * ins)) * pw

    def py(v):
        return y0 + (ins + v * (1 - 2 * ins)) * ph

    out = [f'<svg class="chart scat" viewBox="0 0 {W} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    out.append(f'<text class="cpt cl" x="{x0}" y="30">{esc(title)} '
               f'<tspan class="sauc">(AUC {esc(auc)})</tspan></text>')
    out.append(f'<rect class="sframe" x="{x0}" y="{y0}" width="{pw:.0f}" '
               f'height="{ph:.0f}" rx="3"/>')
    # Direct labels go in the plot, so identity survives without a round trip to the
    # legend. But a class centroid here is the densest part of its own cloud. A label
    # there would sit on the points it names. Search for open space instead: maximise
    # distance to the nearest dot. Lightly prefer x near the class centroid, so the
    # label still reads as part of that cloud. y is weighted down because the box is
    # ~2.4x wider than tall; a gap square in data units is not square on screen.
    occupied = [(x, y) for _, _, x, y, _ in pts]
    # The legend block is drawn later at the top-left; treat it as occupied or the
    # search puts a label underneath it.
    occupied += [(0.02 + 0.01 * i, 0.02 + 0.03 * j) for i in range(9) for j in range(3)]

    def label_spot(cx):
        # Search only a band around the class's own centroid. Clearance alone can put
        # "MotionBlind" in the empty gap between the two clouds, closer to the green
        # cloud than the orange one it names.
        best = None
        for gi in range(3, 38):
            gx = gi / 40
            if abs(gx - cx) > 0.18:
                continue
            for gj in range(3, 38):
                gy = gj / 40
                d = min((gx - ox) ** 2 + ((gy - oy) * 0.42) ** 2 for ox, oy in occupied)
                if best is None or d > best[0]:
                    best = (d, gx, gy)
        return best[1], best[2]

    def spot_near(cx, cy, reach=0.30):
        """Clearest gap within `reach` of a point; used to park a region annotation
        beside the lobe it describes without landing on it."""
        best = None
        for gi in range(2, 39):
            for gj in range(2, 39):
                gx, gy = gi / 40, gj / 40
                if ((gx - cx) ** 2 + ((gy - cy) * 0.42) ** 2) ** 0.5 > reach:
                    continue
                d = min((gx - ox) ** 2 + ((gy - oy) * 0.42) ** 2 for ox, oy in occupied)
                if best is None or d > best[0]:
                    best = (d, gx, gy)
        return (best[1], best[2]) if best else (cx, cy)

    # TimeBlind first: it is 15x the larger cloud and would otherwise bury the other.
    # data-i is the point's row index in the array the figure was plotted from, so a
    # labels file keyed on it lights up the actual question without moving any dot.
    # Each class is wrapped in a <g data-series>. The filter above the figure can then
    # hide a whole cloud with one attribute rather than walking 1,259 circles.
    for name, cls in (("TimeBlind", "k-green"), ("MotionBlind", "k-orange")):
        mine = [(x, y) for bench, _, x, y, _ in pts if bench == name]
        out.append(f'<g class="sgroup" data-series="{name}">')
        for bench, i, x, y, lab in pts:
            if bench != name:
                continue
            t = f'<title>{esc(lab)}</title>' if lab else ""
            out.append(f'<circle class="spt {cls}" cx="{px(x):.1f}" cy="{py(y):.1f}" '
                       f'r="2.7" data-series="{name}" data-i="{i}" '
                       f'data-ax="{x * 100:.1f}"'
                       + (f' data-q="{esc(lab)}"' if lab else "")
                       + (f'>{t}</circle>' if t else '/>'))
        if mine:
            cx = sum(x for x, _ in mine) / len(mine)
            lx, ly = label_spot(cx)
            # Dot carries the colour, text stays in ink. Coloured label text over a
            # coloured cloud fights the dots behind it and stops reading as text.
            out.append(f'<circle class="slabeld {cls}" cx="{px(lx):.1f}" '
                       f'cy="{py(ly) - 4:.1f}" r="4"/>')
            out.append(f'<text class="slabel" x="{px(lx) + 11:.1f}" y="{py(ly):.1f}">'
                       f'{esc(name)}</text>')
        out.append('</g>')
    # Region annotation. This is an author's claim about a lobe, not a per-point
    # attribute: the points file carries only (benchmark, i, x, y), so nothing here knows
    # which questions were web-collected. It is drawn with a leader to the lobe it
    # describes. It is not styled like the series labels, so it does not read as a
    # third class.
    if region_note:
        rn_series, rn_text = region_note
        own = [(x, y) for bench, _, x, y, _ in pts if bench == rn_series]
        lobes = _two_lobes(own)
        if lobes:
            left = min(lobes, key=lambda c: c[0][0])       # the lower-x lobe
            (lcx, lcy), members = left
            ax_, ay_ = spot_near(lcx, lcy)
            out.append(f'<line class="sleader" x1="{px(ax_):.1f}" y1="{py(ay_) + 4:.1f}" '
                       f'x2="{px(lcx):.1f}" y2="{py(lcy):.1f}"/>')
            out.append(f'<text class="sannot" x="{px(ax_):.1f}" y="{py(ay_):.1f}">'
                       f'{esc(rn_text)}</text>')

    for i, (name, cls) in enumerate((("TimeBlind", "k-green"),
                                     ("MotionBlind", "k-orange"))):
        ly = y0 + 22 + i * 19
        out.append(f'<circle class="spt lg {cls}" cx="{x0 + 20}" cy="{ly - 4:.0f}" r="4"/>')
        out.append(f'<text class="ct cl slg" x="{x0 + 32}" y="{ly}">{esc(name)}</text>')
    out.append(f'<text class="cax cxl" x="{x0 + pw / 2:.1f}" y="{y0 + ph + 28}">'
               f'{esc(xlab)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def chart_embed(a):
    rows = list(csv.DictReader(open(a.points_csv)))
    # Optional: src/analysis/site/data/figure2c_labels.csv with (benchmark, i, question)
    # binds each dot to the sample it stands for. The published figure carries geometry
    # only. Without that file the hover reports only the benchmark and the axis position.
    lab = {}
    if os.path.exists(a.labels_csv):
        for r in csv.DictReader(open(a.labels_csv)):
            lab[(r["benchmark"], int(r["i"]))] = r.get("question", "")
        print(f"  figure2c: joined {len(lab)} labels")
    pts = [(r["benchmark"], int(r["i"]), float(r["x"]), float(r["y"]),
            lab.get((r["benchmark"], int(r["i"])), "")) for r in rows]
    # The lower-left TimeBlind lobe is that benchmark's web-collected subset. Stated here
    # rather than derived: figure2c_points.csv carries geometry only, so this is an
    # author's annotation on a region, and the caption says as much.
    return scatter_projection(pts, "MotionBlind \u2212 TimeBlind axis",
                              "Question embedding (Qwen2.5-VL-7B \u00b7 class-mean axis)",
                              "0.97",
                              region_note=("TimeBlind", "web-collected subset"))


def chart_ablation(a):
    rows = list(csv.DictReader(open(a.ablation_csv, encoding="utf-8")))
    order = ["ordered", "reversed", "shuffled", "no-video"]
    by = {r["arm"]: r for r in rows if r.get("model", "Qwen3-VL-4B") == "Qwen3-VL-4B"}
    groups = [o.replace("-", "‑") for o in order]
    vals = {m: [float(by[o][m]) if o in by else None for o in order]
            for m in ("I_Acc", "Acc")}
    return bar_chart(groups, ["I_Acc", "Acc"], vals, 70,
                     "percent", chance=6.25, chance_label="I_Acc\nchance 6.25",
                     classes=["k-blue", "k-green"])


def _sweep(path, metric):
    """-> {sampler: {N: value}} from a curated *_sweep.csv."""
    out = {}
    for r in csv.DictReader(open(path)):
        v = (r.get(metric) or "").strip()
        if not v:
            continue
        out.setdefault(r["sampler"], {})[int(r["num_frames"])] = float(v)
    return out


def chart_luna_sweep(a):
    """One panel per metric, one line per sampler; the frame-budget sweep."""
    metrics = [("I_Acc", "I_Acc (%)", 20, 6.25, "chance 6.25%"),
               ("Acc", "Acc (%)", 80, 50.0, "chance 50%")]
    panels, ylabs, ymaxes, chances, clabels = [], [], [], [], []
    xs = set()
    for m, ylab, ym, ch, cl in metrics:
        cur = _sweep(a.luna_hires_csv, m)
        if not cur:
            continue
        series = [(s, cur[s]) for s in ("uniform", "random", "hornet") if s in cur]
        for _, pts in series:
            xs |= set(pts)
        panels.append((ylab.replace(" (%)", ""), series))
        ylabs.append(ylab); ymaxes.append(ym); chances.append(ch); clabels.append(cl)
    return line_panels(panels, sorted(xs), ylabs, ymaxes, chances, clabels,
                       classes=["k-blue", "k-red", "k-green"])



def chart_strategies(a):
    """Paper Table 3 (right): the four selection strategies on a toy 12-frame clip.

    This one is a schematic, as it is in the paper; it shows the shape of each policy,
    not a run. Every Frame2Clip cell in `results/` was recorded with a uniform scale
    of 1.000. The reduced-resolution trade (many frames at lower resolution for the
    same native-frame budget) never appears in the logs. It cannot be plotted from
    them.
    """
    N = 12
    cell, gap = 46, 6
    rows = [
        ("Uniform",   "k-uniform", [0, 3, 6, 9],  [], []),
        ("Random",    "k-random",  [1, 2, 7, 10], [], []),
        ("HORNet",    "k-hornet",  [4, 5, 7, 8],  [], []),
        # one anchor grown into a contiguous clip, the extra frames paid for by resolution
        ("Frame2Clip", "k-f2c",    [4, 5, 6, 7, 8, 9], [4], [6, 7, 8, 9]),
    ]
    TAPS = [(4, 6), (7, 9)]                      # the two events the question turns on
    x0, y0 = 118, 74
    pw = N * cell + (N - 1) * gap
    H2 = y0 + 30 + len(rows) * 34 + 46
    out = [f'<svg class="chart strat" viewBox="0 0 {x0 + pw + 16} {H2}" role="img" '
           f'preserveAspectRatio="xMidYMid meet">']
    out.append('<defs><pattern id="mosaic" width="6" height="6" '
               'patternUnits="userSpaceOnUse">'
               '<rect width="6" height="6" fill="var(--surface)"/>'
               '<rect width="3" height="3" fill="currentColor" opacity=".55"/>'
               '<rect x="3" y="3" width="3" height="3" fill="currentColor" opacity=".55"/>'
               '</pattern></defs>')

    def cx(i):
        return x0 + i * (cell + gap)

    for lo, hi in TAPS:
        out.append(f'<rect class="tapband" x="{cx(lo) - 3:.0f}" y="{y0 - 30}" '
                   f'width="{cx(hi) + cell - cx(lo) + 6:.0f}" height="{H2 - y0 + 8}" rx="4"/>')
    out.append(f'<text class="ct ctr crow" x="{x0 - 12}" y="{y0 - 8}">video</text>')
    for i in range(N):
        out.append(f'<rect class="fcell" x="{cx(i):.0f}" y="{y0 - 24}" '
                   f'width="{cell}" height="22" rx="2"/>')
        out.append(f'<text class="ct fnum" x="{cx(i) + cell / 2:.0f}" '
                   f'y="{y0 + 8}">{i + 1}</text>')
    for k, (lo, hi) in enumerate(TAPS):
        mid = (cx(lo) + cx(hi) + cell) / 2
        out.append(f'<text class="tapt" x="{mid:.0f}" y="{y0 - 34}">tap {k + 1}</text>')

    for r, (name, cls, picks, anchors, lowres) in enumerate(rows):
        ry = y0 + 26 + r * 34
        out.append(f'<text class="ct ctr crow" x="{x0 - 12}" y="{ry + 15}">{esc(name)}</text>')
        for i in range(N):
            on = i in picks
            fill = ' fill="url(#mosaic)"' if i in lowres else ''
            out.append(
                f'<rect class="pcell {cls}{" on" if on else ""}" x="{cx(i):.0f}" '
                f'y="{ry:.0f}" width="{cell}" height="22" rx="2"{fill} '
                f'data-series="{esc(name)}" data-group="frame {i + 1}" data-panel="" '
                f'data-unit="" data-value="{1 if on else 0}" '
                f'data-vlabel="{"selected" if on else "&#8212;"}">'
                f'<title>{esc(name)} &#183; frame {i + 1}: '
                f'{"selected" if on else "not selected"}'
                f'{" (reduced resolution)" if i in lowres else ""}</title></rect>')
            if i in anchors:
                out.append(f'<circle class="anchor" cx="{cx(i) + cell / 2:.0f}" '
                           f'cy="{ry + 11:.0f}" r="4"/>')
    out.append("</svg>")
    return "\n".join(out)



# --------------------------------------------------------------------------------------
# Tables. Same marker mechanism as the charts: a results table authored by hand drifts
# from results/paper/table1_leaderboard.csv (wrong cells, missing cells). Generating it
# removes that class of bug.
# --------------------------------------------------------------------------------------
NBH = "\u2011"          # non-breaking hyphen, so model names never wrap mid-token

# Footnote markers, keyed by model. Kept here beside the generator rather than in the
# page, so a row and its marker cannot get out of step.
FOOT = {"Gemma-4-12B-it": "&lowast;", "Inkling-Small": "&dagger;",
        "Motion-o (7B)": "&Dagger;", "GPT-5.6": "&para;", "Human": "&#8214;"}


def table_leaderboard(a):
    rows = list(csv.DictReader(open(os.path.join("results/paper", "table1_leaderboard.csv"), encoding="utf-8")))
    cols = ["MB_IAcc", "MB_Acc", "TB_IAcc", "TB_Acc", "VMME_Acc"]

    # Bold the best value per column among the models only; the human row is a ceiling,
    # not a competitor, and bolding it would read as "best model".
    comp = [r for r in rows if r["group"] != "human"]
    best = {}
    for c in cols:
        vals = [float(r[c]) for r in comp if r[c]]
        best[c] = max(vals) if vals else None

    def cell(r, c):
        v = r[c]
        if not v:
            return '<td class="num">&ndash;</td>'
        mark = ""
        # Gemini's TimeBlind I_Acc is TimeBlind's own published figure, not this repo's run.
        if r["model"] == "Gemini 3.1 Pro" and c == "TB_IAcc":
            mark = ' <sup>&sect;</sup>'
        body = f"<strong>{v}</strong>" if best[c] is not None and abs(float(v) - best[c]) < 1e-9 else v
        return f'<td class="num">{body}{mark}</td>'

    def name(r):
        n = r["model"].replace("-", NBH)
        f = FOOT.get(r["model"])
        n = f"<em>{n}</em>" if r["group"] == "human" else n
        if r["model"] == "Gemini 3.1 Pro":
            n = f"<strong>{n}</strong>"
        return n + (f' <sup>{f}</sup>' if f else "")

    out = ["""    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="2" class="num">MotionBlind</th>
        <th colspan="2" class="num">TimeBlind</th>
        <th rowspan="2" class="num">Video&#8209;MME<br><span class="thsub">Acc &uarr;</span></th>
      </tr>
      <tr><th class="num">I_Acc &uarr;</th><th class="num">Acc &uarr;</th>
          <th class="num">I_Acc &uarr;</th><th class="num">Acc &uarr;</th></tr>
    </thead>
    <tbody>
      <tr class="chance"><td class="chance"><em>Chance</em></td>"""
           + "".join(f'<td class="num chance">{v}</td>'
                     for v in ("6.25", "50.0", "6.25", "50.0", "25.0"))
           + "</tr>"]
    for grp, label in (("open", "Open Video&#8209;LLMs"), ("frontier", "Frontier proprietary")):
        out.append(f'      <tr class="grouphead"><td colspan="6"><em>{label}</em></td></tr>')
        for r in rows:
            if r["group"] != grp:
                continue
            out.append(f'      <tr><td>{name(r)}</td>'
                       + "".join(cell(r, c) for c in cols) + "</tr>")
    for r in rows:
        if r["group"] == "human":
            out.append(f'      <tr class="humanrow"><td>{name(r)}</td>'
                       + "".join(cell(r, c) for c in cols) + "</tr>")
    out.append("    </tbody>")
    return "\n".join(out)


TABLES = {"leaderboard": table_leaderboard}

CHARTS = {"leaderboard": chart_leaderboard, "makeup": chart_makeup,
          "words": chart_words, "embed": chart_embed, "ablation": chart_ablation,
          "percat": chart_percat, "budget": chart_budget, "selection": chart_selection,
          "strategies": chart_strategies, "lunasweep": chart_luna_sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=sorted(CHARTS))
    ap.add_argument("--inject", metavar="HTML", nargs="+",
                    help="rewrite the charts inside <!-- chart:NAME --> markers. Accepts "
                         "several files: the page is authored as motionblind.io/sections/*.html "
                         "partials, so the 11 markers are spread across 7 of them.")
    ap.add_argument("--all-to", default=None, metavar="DIR",
                    help="write <name>.svg for every chart")
    ap.add_argument("--words-csv", default="src/analysis/site/data/figure2b_words.csv")
    ap.add_argument("--points-csv", default="src/analysis/site/data/figure2c_points.csv")
    ap.add_argument("--labels-csv", default="src/analysis/site/data/figure2c_labels.csv")
    ap.add_argument("--ablation-csv", default="results/paper/table2_probes.csv")
    ap.add_argument("--luna-hires-csv",
                    default="results/models/GPT-5.6-Luna-720x1280/gpt_5_6_luna_hires_mbhuman_sweep.csv")
    a = ap.parse_args()

    if a.inject:
        # Replace between markers, so re-running after new results refreshes the charts
        # in place instead of appending a second copy.
        #
        # A chart lives in exactly one file, so a missing marker is only worth reporting
        # once all the targets have been searched; otherwise splitting the page into
        # partials would print ten "skip" lines per file and bury a real miss.
        placed = set()
        for target in a.inject:
            page = open(target, encoding="utf-8").read()
            touched = False
            for kind, reg in (("chart", CHARTS), ("table", TABLES)):
              for name, fn in reg.items():
                start = f'<!-- {kind}:{name} -->'
                end = f'<!-- /{kind}:{name} -->'
                if start not in page:
                    continue
                svg = fn(a)
                if svg is None:
                    print(f"  skip {name}: source not available")
                    continue
                i, j = page.index(start) + len(start), page.index(end)
                page = page[:i] + "\n" + svg + "\n    " + page[j:]
                placed.add(name)
                touched = True
                print(f"  injected {kind} {name}  ({os.path.basename(target)})")
            if touched:
                open(target, "w", encoding="utf-8").write(page)
        for name in list(CHARTS) + list(TABLES):
            if name not in placed:
                print(f"  SKIPPED {name}: no <!-- chart:{name} --> marker in any target")
        return

    if a.all_to:
        os.makedirs(a.all_to, exist_ok=True)
        for name, fn in CHARTS.items():
            svg = fn(a)
            if svg is None:
                print(f"  skip {name}: source not available")
                continue
            open(os.path.join(a.all_to, f"{name}.svg"), "w", encoding="utf-8").write(svg)
            print(f"wrote {a.all_to}/{name}.svg  ({len(svg)} bytes)")
    elif a.emit:
        print(CHARTS[a.emit](a))
    else:
        ap.error("pass --emit NAME, --all-to DIR, or --inject HTML")


if __name__ == "__main__":
    main()
