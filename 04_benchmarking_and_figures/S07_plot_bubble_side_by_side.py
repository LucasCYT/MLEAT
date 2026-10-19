#!/usr/bin/env python3
"""
Rank-correlation bubble plot of ML vs GWAS ranks, male and female side by side.

Same figure as the notebook cell that produced bubble_male_female_side_by_side.svg,
but the top-N threshold is a parameter (the notebook had X = 3000 hard-coded while
the titles said "Top 1000").

    python3 plot_bubble_side_by_side.py -n 1000
    python3 plot_bubble_side_by_side.py -n 3000 -o bubble_top3000.svg

Input: MaleCompare.txt / FemaleCompare.txt (columns Feature, ML_Rank, GWAS_Rank, Ref)
or the pre-filtered bubble_input_*_top3000.tsv files.
"""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORSCALE = [
    [0.0, "#ffffff"],     # Ref = 0  (white)
    [0.3, "#fee0d2"],
    [0.6, "#fc9272"],
    [1.0, "#de2d26"],     # Ref max  (deep red)
]
BORDER_COLOR = "gray"
BORDER_WIDTH = 0.6
SIZE_LEGEND = [2, 5, 10, 15]


def bubble_size(ref):
    """Notebook's sizing rule: (Ref * 1.5 + 6) scaled to two thirds."""
    return (ref * 1.5 + 6) * 2 / 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--top", type=int, default=1000,
                   help="rank threshold shown on both axes")
    p.add_argument("--male", default="MaleCompare.txt")
    p.add_argument("--female", default="FemaleCompare.txt")
    p.add_argument("-o", "--out", default=None,
                   help="output .svg [default: bubble_male_female_side_by_side_topN.svg]")
    p.add_argument("--html", action="store_true", help="also write an interactive .html")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=800)
    args = p.parse_args()

    X = args.top
    out = args.out or f"bubble_male_female_side_by_side_top{X}.svg"

    male = pd.read_csv(args.male, sep="\t")
    female = pd.read_csv(args.female, sep="\t")
    max_ref = max(male["Ref"].max(), female["Ref"].max())

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=(f"Male (Top {X:,})", f"Female (Top {X:,})"))

    for name, df, col in (("Male", male, 1), ("Female", female, 2)):
        sel = df[(df["ML_Rank"] <= X) | (df["GWAS_Rank"] <= X)].copy()

        known = sel[sel["Ref"] > 0]
        fig.add_trace(
            go.Scatter(
                x=known["ML_Rank"], y=known["GWAS_Rank"], mode="markers",
                marker=dict(
                    size=bubble_size(known["Ref"]),
                    color=known["Ref"], colorscale=COLORSCALE,
                    cmin=0, cmax=max_ref,
                    colorbar=dict(title="Ref (Color)", thickness=15, len=0.45,
                                  x=1.02, y=0.75, yanchor="middle") if col == 2 else None,
                    showscale=(col == 2), opacity=0.8,
                    line=dict(width=BORDER_WIDTH, color=BORDER_COLOR)),
                customdata=known[["Feature", "Ref"]],
                hovertemplate=("<b>%{customdata[0]}</b><br>ML Rank: %{x}"
                               "<br>GWAS Rank: %{y}<br>Ref: %{customdata[1]}<extra></extra>"),
                name=f"{name} (Known)", showlegend=False),
            row=1, col=col)

        novel = sel[sel["Ref"] == 0]
        fig.add_trace(
            go.Scatter(
                x=novel["ML_Rank"], y=novel["GWAS_Rank"], mode="markers",
                marker=dict(size=5 * 2 / 3, color="lightblue", symbol="circle",
                            line=dict(width=BORDER_WIDTH, color=BORDER_COLOR)),
                customdata=novel[["Feature", "Ref"]],
                hovertemplate=("<b>%{customdata[0]}</b> (Novel)<br>ML Rank: %{x}"
                               "<br>GWAS Rank: %{y}<br>Ref: 0<extra></extra>"),
                name="Novel point (Ref=0)", showlegend=(col == 1)),
            row=1, col=col)

        print(f"{name}: {len(sel):,} points in view "
              f"({len(known):,} with Ref>0, {len(novel):,} novel)")

    for i, ref_val in enumerate(SIZE_LEGEND):        # bubble-size legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=bubble_size(ref_val), color="rgba(222, 45, 38, 0.4)",
                        line=dict(width=BORDER_WIDTH, color=BORDER_COLOR)),
            name=f"Ref = {ref_val}", legendgroup="size_legend",
            legendgrouptitle_text="Bubble Size" if i == 0 else None))

    fig.update_layout(
        title_text=f"Rank Correlation of ML vs. GWAS (Male & Female, Top {X:,} Bubble Plot)",
        title_x=0.5, width=args.width, height=args.height, template="plotly_white",
        showlegend=True,
        legend=dict(x=1.02, y=0.25, yanchor="middle", bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="lightgray", borderwidth=1))

    for col in (1, 2):
        fig.update_xaxes(title_text="ML Rank", range=[0, X], row=1, col=col)
        fig.update_yaxes(title_text="GWAS Rank", range=[0, X], row=1, col=col)

    fig.write_image(out)
    print(f"saved: {out}")
    if args.html:
        fig.write_html(os.path.splitext(out)[0] + ".html")
        print(f"saved: {os.path.splitext(out)[0]}.html")


if __name__ == "__main__":
    main()
