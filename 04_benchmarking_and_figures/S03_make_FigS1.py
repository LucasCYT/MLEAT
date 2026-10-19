#!/usr/bin/env python3
"""
Figure S1 -- one figure, four panels in two lettered groups:

    A   the ranked MALE   SNPs and known references      (whole list)
        the top 3000 ranked MALE SNPs and known references
    B   the ranked FEMALE SNPs and known references      (whole list)
        the top 3000 ranked FEMALE SNPs and known references

Run it from the folder that holds the CSVs (or edit the paths below):

    python make_FigS1.py

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

TOP_SPAN = "1:3000"                            # the zoomed-in panel of each group
WINDOW = 100                                   # moving-window size, in SNPs
RATIO = "16:13"                                # whole-figure length:width, inches
YLIM_LEFT = 300                                # MA of Refs axis maximum
YLIM_RIGHT = 18                                # Refs axis maximum
OUT = "FigureS1_male_female_all_top3000.png"   # a matching .pdf is written too
# --------------------------------------------------------------------------- #


def resolve(name):
    """Look for the file in the working directory first, then next to this script."""
    for cand in (Path(name), HERE / name):
        if cand.exists():
            return str(cand)
    raise SystemExit(f"cannot find {name} in {Path.cwd()} or {HERE}")


male, female, twb = resolve(MALE), resolve(FEMALE), resolve(TWB_LIST)

ARGV = [
    # -- group A: male ---------------------------------------------------------
    "-i", male, "-t", "The ranked male SNPs and known references",
    "-s", "all", "-C", "male", "--panel-label", "A",
    "-i", male, "-t", "The top 3000 ranked male SNPs and known references",
    "-s", TOP_SPAN, "-C", "male", "--panel-label", "",
    # -- group B: female -------------------------------------------------------
    "-i", female, "-t", "The ranked female SNPs and known references",
    "-s", "all", "-C", "female", "--panel-label", "B",
    "-i", female, "-t", "The top 3000 ranked female SNPs and known references",
    "-s", TOP_SPAN, "-C", "female", "--panel-label", "",
    # -- shared settings -------------------------------------------------------
    "-L", twb,
    "-w", str(WINDOW),
    "-r", RATIO,
    "--ylim-left", str(YLIM_LEFT),
    "--ylim-right", str(YLIM_RIGHT),
    "-o", OUT,
]

if __name__ == "__main__":
    sys.argv = ["plot_ranked_refcount.py"] + ARGV + sys.argv[1:]
    engine.main()
