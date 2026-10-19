#!/usr/bin/env python3
"""
Figure 4A -- one figure, two panels:
    top    : the top 3000 ranked MALE   SNPs and known references
    bottom : the top 3000 ranked FEMALE SNPs and known references

Run it from the folder that holds the CSVs (or edit the paths below):

    python make_Fig4A.py

Everything is drawn by plot_ranked_refcount.py, which must sit next to this
file. Edit the constants below to change inputs, size or scales.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_ranked_refcount as engine          # noqa: E402

# --------------------------------------------------------------------------- #
MALE = "Male_50wAddone_rankedList_20260903.csv"
FEMALE = "Female_50wAddone_rankedList_20260903.csv"
TWB_LIST = "TWBMList.txt"                      # rsIDs to mark with a red dot

SPAN = "1:3000"                                # rank range shown in both panels
WINDOW = 100                                   # moving-window size, in SNPs
RATIO = "16:7"                                 # whole-figure length:width, inches
YLIM_LEFT = 300                                # MA of Refs axis maximum
YLIM_RIGHT = 18                                # Refs axis maximum
PANEL_LABEL = "A"                              # bottom-left figure label
OUT = "Figure4A_top3000_male_female.png"       # a matching .pdf is written too
# --------------------------------------------------------------------------- #


def resolve(name):
    """Look for the file in the working directory first, then next to this script."""
    for cand in (Path(name), HERE / name):
        if cand.exists():
            return str(cand)
    raise SystemExit(f"cannot find {name} in {Path.cwd()} or {HERE}")


ARGV = [
    "-i", resolve(MALE),
    "-i", resolve(FEMALE),
    "-t", "The top 3000 ranked male SNPs and known references",
    "-t", "The top 3000 ranked female SNPs and known references",
    "-C", "male",
    "-C", "female",
    "-s", SPAN,
    "-L", resolve(TWB_LIST),
    "-w", str(WINDOW),
    "-r", RATIO,
    "--ylim-left", str(YLIM_LEFT),
    "--ylim-right", str(YLIM_RIGHT),
    "--panel-label", PANEL_LABEL,
    "--panel-label-pos", "bottom",
    "-o", OUT,
]

if __name__ == "__main__":
    sys.argv = ["plot_ranked_refcount.py"] + ARGV + sys.argv[1:]
    engine.main()
