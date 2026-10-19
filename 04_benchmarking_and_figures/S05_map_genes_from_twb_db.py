#!/usr/bin/env python3
"""
Map rsIDs to genes offline, using the TWB chip annotation already in
TWBV2_20240515.db (table TWBv2_Probeinfor, column AssociatedGene).

No internet needed -- everything comes from the local database.

    python3 map_genes_from_twb_db.py                      # both top-3000 lists
    python3 map_genes_from_twb_db.py -i my_rsids.txt

Per rsID it writes
------------------
    chr, pos_hg38, pos_hg19          from the probe annotation
    genes_in        genes the SNP sits inside (intron / exon / UTR / splice)
    location        the relationships seen for those genes
    nearest_gene    if the SNP is intergenic: the closest gene ...
    nearest_distance_bp             ... and how far away it is
    all_genes       every gene symbol the probe annotation mentions
    entrez_ids      matching Entrez gene ids

AssociatedGene packs one record per transcript, fields separated by ' // '
and records by ' /// ':

    transcript // relation // distance // unigene // symbol // entrez // name

so a SNP inside a gene has relation intron/exon/UTR and distance 0, while an
intergenic one only has upstream/downstream records with a real distance.
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
from collections import OrderedDict

DEFAULT_DB = "TWBV2_20240515.db"
DEFAULT_INPUTS = ["ML_male_top3000_rsID.txt", "ML_female_top3000_rsID.txt"]
INSIDE = {"intron", "exon", "utr-3", "utr-5", "cds", "splice-site", "utr"}
NULL = {"---", "", "None", None}


def read_rsids(path):
    """rsIDs from a one-per-line list or from the rsID column of a CSV."""
    ids = []
    if path.lower().endswith((".csv", ".tsv")):
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            col = next((c for c in reader.fieldnames
                        if c.lower() in ("rsid", "snp", "id")), None)
            if col is None:
                sys.exit(f"no rsID column in {path}: {reader.fieldnames}")
            ids = [row[col] for row in reader]
    else:
        ids = [ln.strip() for ln in open(path)]
    seen = OrderedDict()
    for i in ids:
        if re.fullmatch(r"rs\d+", str(i).strip()):
            seen[i.strip()] = None
    return list(seen)


def parse_associated_gene(text):
    """-> (inside, nearby, entrez) ; inside/nearby map symbol -> info."""
    inside, nearby, entrez = OrderedDict(), OrderedDict(), OrderedDict()
    if text in NULL:
        return inside, nearby, entrez
    for record in str(text).split("///"):
        parts = [p.strip() for p in record.split("//")]
        if len(parts) < 6:
            continue
        relation, distance, symbol, gene_id = parts[1].lower(), parts[2], parts[4], parts[5]
        if symbol in NULL:
            continue
        if gene_id not in NULL:
            entrez[symbol] = gene_id
        try:
            dist = int(distance)
        except ValueError:
            dist = 0
        if relation in INSIDE or dist == 0:
            inside.setdefault(symbol, set()).add(relation)
        else:
            if symbol not in nearby or dist < nearby[symbol][1]:
                nearby[symbol] = (relation, dist)
    return inside, nearby, entrez


def annotate(db, rsids):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows, missing = [], 0
    for rsid in rsids:
        # dbSNPRSID is '---' for a good number of probes; those carry the id in
        # ExtendedRSID instead, so fall back to it before giving up
        sql = ("select Chromosome, PhysicalPosition, hg19position, AssociatedGene "
               "from TWBv2_Probeinfor where ")
        probes = con.execute(sql + "dbSNPRSID = ?", (rsid,)).fetchall()
        if not probes:
            probes = con.execute(sql + "ExtendedRSID = ?", (rsid,)).fetchall()
        if not probes:
            missing += 1
            rows.append(dict(rsID=rsid, chr="", pos_hg38="", pos_hg19="",
                             genes_in="", location="", nearest_gene="",
                             nearest_distance_bp="", all_genes="", entrez_ids="",
                             note="no probe in TWBv2_Probeinfor"))
            continue
        # a rsID can carry several probes; merge what they say
        chrom = pos38 = pos19 = ""
        inside_all, nearby_all, entrez_all = OrderedDict(), OrderedDict(), OrderedDict()
        for chrom_, pos38_, pos19_, assoc in probes:
            chrom = chrom or chrom_
            pos38 = pos38 or pos38_
            pos19 = pos19 or pos19_
            ins, near, ent = parse_associated_gene(assoc)
            for sym, rels in ins.items():
                inside_all.setdefault(sym, set()).update(rels)
            for sym, val in near.items():
                if sym not in nearby_all or val[1] < nearby_all[sym][1]:
                    nearby_all[sym] = val
            entrez_all.update(ent)

        nearest, nearest_d = "", ""
        if not inside_all and nearby_all:
            nearest = min(nearby_all, key=lambda s: nearby_all[s][1])
            nearest_d = nearby_all[nearest][1]

        all_genes = list(inside_all) + [g for g in nearby_all if g not in inside_all]
        rows.append(dict(
            rsID=rsid, chr=chrom, pos_hg38=pos38, pos_hg19=pos19,
            genes_in=";".join(inside_all),
            location=";".join(sorted({r for rels in inside_all.values() for r in rels})),
            nearest_gene=nearest, nearest_distance_bp=nearest_d,
            all_genes=";".join(all_genes),
            entrez_ids=";".join(entrez_all.get(g, "") for g in all_genes),
            note=""))
    return rows, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", action="append", default=None,
                   help="rsID list or CSV; repeat for several files")
    p.add_argument("--db", default=DEFAULT_DB, help="TWB sqlite database")
    p.add_argument("-o", "--out", action="append", default=None,
                   help="output CSV per input [default: <input>_genes.csv]")
    p.add_argument("--no-summary", action="store_true",
                   help="skip the per-gene summary file")
    args = p.parse_args()

    inputs = args.input or DEFAULT_INPUTS
    outs = args.out or [os.path.splitext(os.path.basename(i))[0].replace("_rsID", "")
                        + "_genes.csv" for i in inputs]

    for src, dst in zip(inputs, outs):
        rsids = read_rsids(src)
        rows, missing = annotate(args.db, rsids)
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        inside = sum(1 for r in rows if r["genes_in"])
        near = sum(1 for r in rows if r["nearest_gene"])
        genes = {g for r in rows for g in r["all_genes"].split(";") if g}
        in_genes = {g for r in rows for g in r["genes_in"].split(";") if g}
        print(f"{src}: {len(rsids):,} rsIDs")
        print(f"   inside a gene      {inside:,} ({inside/len(rsids)*100:.1f}%)")
        print(f"   intergenic, nearest{near:>7,} ({near/len(rsids)*100:.1f}%)")
        print(f"   no annotation      {len(rsids)-inside-near:,} "
              f"(not on chip: {missing:,})")
        print(f"   unique genes: {len(genes):,} mentioned, {len(in_genes):,} containing a SNP")
        print(f"   saved: {dst}")

        if not args.no_summary:
            summary = OrderedDict()          # gene -> counts, best rank, snps
            for rank, r in enumerate(rows, 1):
                for sym in [g for g in r["genes_in"].split(";") if g]:
                    e = summary.setdefault(sym, dict(gene=sym, entrez="", n_snps_inside=0,
                                                     n_snps_nearest=0, best_rank=rank,
                                                     rsIDs=[]))
                    e["n_snps_inside"] += 1
                    e["best_rank"] = min(e["best_rank"], rank)
                    e["rsIDs"].append(r["rsID"])
                if r["nearest_gene"]:
                    sym = r["nearest_gene"]
                    e = summary.setdefault(sym, dict(gene=sym, entrez="", n_snps_inside=0,
                                                     n_snps_nearest=0, best_rank=rank,
                                                     rsIDs=[]))
                    e["n_snps_nearest"] += 1
                    e["best_rank"] = min(e["best_rank"], rank)
                    e["rsIDs"].append(r["rsID"])
                for sym, eid in zip([g for g in r["all_genes"].split(";") if g],
                                    r["entrez_ids"].split(";")):
                    if sym in summary and eid:
                        summary[sym]["entrez"] = eid
            sfile = dst.replace("_genes.csv", "_geneSummary.csv")
            ordered = sorted(summary.values(),
                             key=lambda e: (-(e["n_snps_inside"] + e["n_snps_nearest"]),
                                            e["best_rank"]))
            with open(sfile, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["gene", "entrez", "n_snps_inside", "n_snps_nearest",
                            "n_snps_total", "best_rank", "rsIDs"])
                for e in ordered:
                    w.writerow([e["gene"], e["entrez"], e["n_snps_inside"],
                                e["n_snps_nearest"],
                                e["n_snps_inside"] + e["n_snps_nearest"],
                                e["best_rank"], ";".join(e["rsIDs"])])
            with open(sfile.replace("_geneSummary.csv", "_geneList.txt"), "w") as fh:
                fh.write("\n".join(e["gene"] for e in ordered) + "\n")
            print(f"   saved: {sfile}  (+ gene list .txt for enrichment tools)")
        print()


if __name__ == "__main__":
    main()
