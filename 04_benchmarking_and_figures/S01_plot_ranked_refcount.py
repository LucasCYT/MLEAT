#!/usr/bin/env python3
"""
RefCount histogram along a ranked SNP list, with a moving-window overlay.

X  : rank position in the input list order (1 = top of list)
Y1 : moving sum (or mean) of RefCount   -- LEFT axis, coloured line
Y2 : RefCount per SNP                   -- RIGHT axis, bars
Optional: mark a set of rsIDs (dot on the bar) from a text file.

Give -i more than once to stack several ranked lists as panels of one figure
(-t once per panel, in the same order).

Examples
--------
    python plot_ranked_refcount.py -i Female_50wAddone_rankedList_20260903.csv

    python plot_ranked_refcount.py -i data.csv -s 1:3000 -L TWBMList.txt -r 16:5

    python plot_ranked_refcount.py \
        -i Male_50wAddone_rankedList_20260903.csv \
        -i Female_50wAddone_rankedList_20260903.csv \
        -t "The top 3000 ranked male SNPs and known references" \
        -t "The top 3000 ranked female SNPs and known references" \
        -s 1:3000 -L TWBMList.txt -r 16:7 --ylim-left 300 --ylim-right 18 \
        --panel-label A -o Figure4A.png
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# per-panel colours: (bars, moving-window line, marked rsIDs)
PALETTES = [
    ("#4472c4", "#548235", "#c00000"),   # blue bars  / green line  / red dots
    ("#ffc000", "#843c0c", "#c00000"),   # gold bars  / brown line  / red dots
    ("#7f7f7f", "#1f6fd0", "#c00000"),   # grey bars  / blue line   / red dots
]
SINGLE_PALETTE = ("#9e9e9e", "#1f6fd0", "#d1495b")   # used when there is one panel
PRESETS = {"male": PALETTES[0], "female": PALETTES[1], "grey": PALETTES[2],
           "gray": PALETTES[2], "default": SINGLE_PALETTE}


def parse_colors(spec):
    """'male' / 'female' / 'grey' or 'bar,line,mark' hex triple -> tuple."""
    key = str(spec).strip().lower()
    if key in PRESETS:
        return PRESETS[key]
    parts = [c.strip() for c in str(spec).split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--colors takes a preset name or 3 comma-separated "
                         f"colours (bar,line,mark), got {spec!r}")
    return tuple(parts)


# --------------------------------------------------------------------------- #
# argument helpers
# --------------------------------------------------------------------------- #
def parse_ratio(text):
    """'16:2' / '16x2' / '16,2' -> (16.0, 2.0) == (length, width) in inches."""
    parts = re.split(r"[:x,\s/]+", str(text).strip().lower())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"ratio must look like L:W (e.g. 16:2), got {text!r}")
    try:
        L, W = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"ratio must be numeric, got {text!r}")
    if L <= 0 or W <= 0:
        raise argparse.ArgumentTypeError("ratio values must be > 0")
    return L, W


def parse_span(text):
    """'1:3000' / '1-3000' -> (1, 3000); 'all' -> None (the whole list)."""
    if text is None or str(text).strip().lower() in ("all", "full", "whole"):
        return None
    parts = re.split(r"[:\-,\s]+", str(text).strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"span must look like START:END (e.g. 1:3000), got {text!r}")
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"span must be integers, got {text!r}")
    if a < 1 or b < a:
        raise argparse.ArgumentTypeError("span must satisfy 1 <= START <= END")
    return a, b


def read_rsid_list(path):
    """Pull rsIDs out of a text file.

    Accepts one-per-line / comma / whitespace separated plain text, and also
    survives files saved straight from the macOS clipboard (binary plist),
    because it just regex-scans the bytes for rs-numbers. Order is preserved
    and duplicates dropped.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    found = re.findall(rb"\brs\d{2,}\b", raw)
    return list(dict.fromkeys(i.decode("ascii") for i in found))


# --------------------------------------------------------------------------- #
# computation
# --------------------------------------------------------------------------- #
def moving_stat(y, window, mode="sum"):
    """Centred moving sum or mean; edges use the partial window (no NaN pad).

    mode='sum'  -> total RefCount inside each window (e.g. 243 refs / 100 SNPs)
    mode='mean' -> that total divided by the number of SNPs in the window
    """
    w = max(1, int(window))
    csum = np.concatenate(([0.0], np.cumsum(y, dtype=float)))
    n = len(y)
    half = w // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) - half + w, 0, n)
    total = csum[hi] - csum[lo]
    return total if mode == "sum" else total / (hi - lo)


def load_panel(path, args, wanted, span):
    """Read one ranked list and slice it to the requested rank span."""
    cols = [args.column] + ([args.id_column] if wanted else [])
    df = pd.read_csv(path, usecols=cols)
    y_all = df[args.column].fillna(0).to_numpy(dtype=float)
    n_all = len(y_all)
    ma_all = moving_stat(y_all, args.window, args.mode)   # over the FULL list
    x_all = np.arange(1, n_all + 1)

    lo, hi = (1, n_all) if span is None else span
    hi = min(hi, n_all)
    if lo > n_all:
        raise SystemExit(f"{path}: span starts at {lo} but the list has {n_all} rows")
    sl = slice(lo - 1, hi)

    marks = []                       # (rank, rsid, value) inside the span
    if wanted:
        rank_of = {r: i + 1 for i, r in enumerate(df[args.id_column].astype(str))}
        for rsid in wanted:
            rk = rank_of.get(rsid)
            if rk is not None and lo <= rk <= hi:
                marks.append((rk, rsid, y_all[rk - 1]))
        marks.sort()

    return dict(path=path, x=x_all[sl], y=y_all[sl], ma=ma_all[sl],
                lo=lo, hi=hi, n_all=n_all, marks=marks)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw_panel(ax1, d, args, colours, base, stat_name, n_requested, show_xticks):
    bar_c, line_c, mark_c = colours
    x, y, ma = d["x"], d["y"], d["ma"]

    # LEFT axis: moving window line
    ax1.plot(x, ma, color=line_c, linewidth=1.4, zorder=3,
             label=f"{stat_name} (window = {args.window:,} SNPs)")
    ax1.set_ylabel(args.ylabel_left, color=line_c)
    ax1.tick_params(axis="y", labelcolor=line_c)
    ax1.set_xlim(d["lo"] - 1, d["hi"])
    ax1.set_ylim(0, args.ylim_left or max(ma.max() * 1.10, 1e-9))

    # RIGHT axis: RefCount bars
    ax2 = ax1.twinx()
    nz = y > 0                      # zero-height bars are invisible; skip them
    ax2.vlines(x[nz], 0, y[nz], color=bar_c, linewidth=0.6, zorder=1,
               label=f"{args.ylabel_right} per SNP")
    ax2.set_ylabel(args.ylabel_right, color="#4d4d4d")
    ax2.tick_params(axis="y", labelcolor="#4d4d4d")
    ax2.set_ylim(0, args.ylim_right or max(y.max() * 1.08, 1))

    nbins = int(np.clip(round(ax1.get_figure().get_figheight() * 1.6), 3, 8))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=True))

    # marked rsIDs: a dot on top of the bar
    if d["marks"]:
        mx = np.array([m[0] for m in d["marks"]], dtype=float)
        my = np.array([m[2] for m in d["marks"]], dtype=float)
        ax2.scatter(mx, my, s=base * 2.2, color=mark_c, zorder=5, marker="o",
                    linewidths=0, clip_on=False,
                    label=f"{args.mark_label} ({len(d['marks'])}/{n_requested})")

    ax1.set_zorder(ax2.get_zorder() + 1)      # keep the line on top of the bars
    ax1.patch.set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    if not show_xticks:
        ax1.tick_params(axis="x", labelbottom=False)

    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    return ax2, handles, labels


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="RefCount histogram + moving window line over ranked SNP lists.")
    p.add_argument("-i", "--input", required=True, action="append", metavar="CSV",
                   help="input CSV; repeat for one stacked panel per file")
    p.add_argument("-t", "--title", action="append", default=None,
                   help="panel title; repeat once per -i")
    p.add_argument("-r", "--ratio", type=parse_ratio, default="16:5", metavar="L:W",
                   help="whole-figure length x width in inches, e.g. 16:2")
    p.add_argument("-s", "--span", type=parse_span, action="append", default=None,
                   metavar="START:END",
                   help="rank range to plot, e.g. 1:3000 or 'all'; give once for "
                        "every panel, or once per -i")
    p.add_argument("-C", "--colors", action="append", default=None, metavar="SPEC",
                   help="panel colours: a preset (male/female/grey) or "
                        "bar,line,mark hex triple; once for all panels or one per -i")
    p.add_argument("-L", "--mark-list", default=None, metavar="FILE",
                   help="text file of rsIDs to mark with a dot in the plot")
    p.add_argument("-w", "--window", type=int, default=100,
                   help="moving window size, in number of SNPs")
    p.add_argument("-m", "--mode", choices=["sum", "mean"], default="sum",
                   help="moving window statistic: total in the window, or its mean")
    p.add_argument("-c", "--column", default="RefCount", help="value column to plot")
    p.add_argument("--xlabel", default="", help="x-axis label [default: none]")
    p.add_argument("--ylabel-left", default="MA of Refs", help="left y-axis label")
    p.add_argument("--ylabel-right", default="Refs", help="right y-axis label")
    p.add_argument("--ylim-left", type=float, default=None,
                   help="fix the left axis maximum (keeps panels comparable)")
    p.add_argument("--ylim-right", type=float, default=None,
                   help="fix the right axis maximum")
    p.add_argument("--mark-label", default="TWB biomarker",
                   help="legend text for the marked rsIDs")
    p.add_argument("--panel-label", action="append", default=None,
                   help="figure label, e.g. A. Give once for a whole-figure label, "
                        "or once per -i (empty string for panels without one) to "
                        "letter groups of panels")
    p.add_argument("--panel-label-pos", choices=["top", "bottom"], default="bottom",
                   help="corner for --panel-label")
    p.add_argument("--legend", choices=["auto", "inside", "below"], default="auto",
                   help="legend inside each panel, or once below the figure "
                        "(auto: below for a single panel, inside when stacked)")
    p.add_argument("--id-column", default="rsID", help="rsID column, used with -L")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiplier applied to the ratio (figsize = L*scale x W*scale)")
    p.add_argument("-o", "--out", default=None,
                   help="output image [default: <first input>_<column>[_span].png]")
    p.add_argument("-d", "--dpi", type=int, default=300, help="raster output DPI")
    p.add_argument("--no-pdf", action="store_true",
                   help="skip writing the companion vector PDF")
    args = p.parse_args()

    titles = args.title or []
    if titles and len(titles) != len(args.input):
        raise SystemExit(f"got {len(args.input)} -i but {len(titles)} -t; "
                         "give one title per input (or none)")

    n_in = len(args.input)

    def per_panel(values, what):
        if not values:
            return [None] * n_in
        if len(values) == 1:
            return values * n_in
        if len(values) != n_in:
            raise SystemExit(f"got {n_in} -i but {len(values)} {what}; "
                             "give one for every panel, or exactly one")
        return values

    spans = per_panel(args.span, "-s")
    colour_specs = per_panel(args.colors, "-C")

    wanted = read_rsid_list(args.mark_list) if args.mark_list else []
    panels = [load_panel(path, args, wanted, sp)
              for path, sp in zip(args.input, spans)]
    n_panels = len(panels)
    lo, hi = panels[0]["lo"], panels[0]["hi"]

    out = args.out or (os.path.splitext(os.path.basename(args.input[0]))[0]
                       + f"_{args.column}"
                       + (f"_{lo}-{hi}" if spans[0] else "") + ".png")

    # ---- figure geometry / type scale -------------------------------------- #
    L, W = args.ratio
    fig_w, fig_h = L * args.scale, W * args.scale
    panel_h = fig_h / n_panels
    base = float(np.clip(3.0 + 1.6 * panel_h, 6.0, 11.0))
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base * 0.9, "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9,
    })
    stat_name = "Moving sum" if args.mode == "sum" else "Moving average"
    inside = args.legend == "inside" or (args.legend == "auto" and n_panels > 1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    handles = labels = None
    for k, (ax, d) in enumerate(zip(axes, panels)):
        if colour_specs[k] is not None:
            colours = parse_colors(colour_specs[k])
        else:
            colours = SINGLE_PALETTE if n_panels == 1 else PALETTES[k % len(PALETTES)]
        # every panel keeps its own x tick labels: stacked panels may cover
        # different rank ranges, so a shared axis would misread the top one
        _, handles, labels = draw_panel(
            ax, d, args, colours, base, stat_name, len(wanted), show_xticks=True)
        if titles:
            ax.set_title(titles[k], fontsize=base * 1.25, pad=5)
        if k == n_panels - 1:
            ax.set_xlabel(args.xlabel)
        if inside:
            ax.legend(handles, labels, loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor="#cccccc", fancybox=False,
                      borderpad=0.4, handlelength=1.6)

    bottom = 0.0
    if not inside:                    # one shared legend under the figure
        leg = fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                         frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.canvas.draw()
        bottom = min(leg.get_window_extent().height / fig.dpi / fig_h * 1.15, 0.45)

    left, top = 0.0, 1.0
    labels_in = args.panel_label or []
    if len(labels_in) == 1:           # one label for the whole figure
        left = min((base * 1.9 / 72.0) / fig_w, 0.10)
        if args.panel_label_pos == "top":
            top = 1.0 - min((base * 2.4 / 72.0) / fig_h, 0.10)
            fig.text(left * 0.35, 0.995, labels_in[0], fontsize=base * 1.9,
                     fontweight="bold", ha="left", va="top")
        else:
            fig.text(left * 0.35, 0.012, labels_in[0], fontsize=base * 1.9,
                     fontweight="bold", ha="left", va="bottom")
    elif labels_in:                   # one per panel: letters groups of panels
        if len(labels_in) != n_panels:
            raise SystemExit(f"got {n_panels} -i but {len(labels_in)} --panel-label; "
                             "give one per panel (empty string for none) or just one")
        left = min((base * 1.9 / 72.0) / fig_w, 0.10)

    h_pad = 3.0 if len(labels_in) > 1 else 1.2
    if len(labels_in) > 1:            # reserve a strip for the group letters
        top = min(top, 1.0 - (base * 2.2 / 72.0) / fig_h)
    fig.tight_layout(rect=[left, bottom, 1, top], h_pad=h_pad)

    if len(labels_in) > 1:            # after layout: above each labelled panel
        for ax, text in zip(axes, labels_in):
            if text:
                pos = ax.get_position()
                fig.text(0.008, pos.y1 + (base * 1.5 / 72.0) / fig_h, text,
                         fontsize=base * 1.9, fontweight="bold",
                         ha="left", va="bottom")
    fig.savefig(out, dpi=args.dpi)
    if not args.no_pdf:
        fig.savefig(os.path.splitext(out)[0] + ".pdf")

    for d in panels:
        name = os.path.basename(d["path"])
        print(f"{name}: ranks {d['lo']}-{d['hi']} of {d['n_all']:,} | "
              f"max {args.column} = {d['y'].max():g} | "
              f"{stat_name.lower()} max = {d['ma'].max():g}"
              + (f" | marked {len(d['marks'])}/{len(wanted)}" if wanted else ""))
    print(f"figsize = {fig_w:g} x {fig_h:g} in ({n_panels} panel"
          f"{'s' if n_panels > 1 else ''})")
    print(f"saved: {out}" + ("" if args.no_pdf else f" (+ {os.path.splitext(out)[0]}.pdf)"))


if __name__ == "__main__":
    main()
