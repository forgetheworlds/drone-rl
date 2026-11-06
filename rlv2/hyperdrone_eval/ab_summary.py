#!/usr/bin/env python3
"""Round-7 A/B summary from grade.cpp RLV2_DIAG side files.
Usage: ab_summary.py DIR NAME... ; reads DIR/NAME_SET.csv.diag for SET in orig m3d pt1 h26.
Per set: completion, contact flights, timeouts, seconds per waypoint and path length per waypoint over completed flights (take-off
scan included), path length per reached waypoint, top causes."""
import csv, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from failure_breakdown import cause

d, names = sys.argv[1], sys.argv[2:]
sets = ["orig", "m3d", "pt1", "h26"]
print("| config | set | flights | completion | contacts | timeouts | s / waypoint | m / waypoint | top causes |")
print("|---|---|---:|---:|---:|---:|---:|---:|---|")
for s in sets:
    for n in names:
        fn = f"{d}/{n}_{s}.csv.diag"
        if not os.path.exists(fn):
            continue
        rows = list(csv.DictReader(open(fn)))
        N = len(rows)
        done = sum(r["end"] == "complete" for r in rows)
        con = sum(r["end"] == "contact" for r in rows)
        to = sum(r["end"] == "timeout" for r in rows)
        reached = [r for r in rows if r["end"] == "complete"]  # time / path over completed flights
        wp = sum(int(r["goal"]) for r in reached)
        t = sum(float(r["seconds"]) for r in reached) / max(1, wp)
        L = sum(float(r["path_len"]) for r in reached) / max(1, wp)
        c = Counter(cause(r).split("/")[1] for r in rows if r["end"] != "complete")
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(4))
        print(f"| {n} | {s} | {N} | {done / N:.3f} | {con} | {to} | {t:.1f} | {L:.2f} | {top} |")
