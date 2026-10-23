#!/usr/bin/env python3
"""Plot per-mission outcomes and flight times from results/*/summary.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TITLES = {"source": "Generated indoor missions", "public": "Moving AI Warframe maps", "fresh": "Fresh generated layouts"}
COLORS = {"success": "#2a9d8f", "waypoint_timeout": "#e9c46a", "requires_supervisor": "#e76f51"}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), gridspec_kw={"width_ratios": [32, 16, 15]})
    for ax, portfolio in zip(axes, TITLES):
        rows = list(csv.DictReader((ROOT / "results" / portfolio / "summary.csv").open()))
        names = [r["mission"].replace("public-visible-", "") for r in rows]
        times = [float(r["time_s"]) for r in rows]
        ax.barh(names, times, color=[COLORS[r["outcome"]] for r in rows])
        arrived = sum(r["outcome"] == "success" for r in rows)
        contacts = sum(int(r["contact"]) for r in rows)
        ax.set_title(f"{TITLES[portfolio]}\n{arrived}/{len(rows)} arrived, {contacts} contacts", fontsize=11)
        ax.set_xlabel("mission time (s)")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, ["arrived", "timed out", "handed to supervisor"], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output = ROOT / "media/results.png"
    fig.savefig(output, dpi=110)
    print(output)


if __name__ == "__main__":
    main()
