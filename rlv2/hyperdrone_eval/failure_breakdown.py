#!/usr/bin/env python3
"""Classify failed flights from grade.cpp RLV2_DIAG=1 side files (CSV.diag).

Usage: failure_breakdown.py OUT.md OUT.csv LABEL=FILE.diag[,FILE.diag...] ...
Causes (first match):
  contact/floor-ceiling, contact/moving-box (a moving box within 0.25 m), contact/diagonal-thin (no
  static surface within 0.3 m on the 72 horizontal + up/down rays: hit diagonally above/below or a
  thin edge), contact/side-behind (nearest obstacle >45 deg off the
  heading), contact/in-view-seen (obstacle ahead and stereo had >=50% of its <0.8 m rays within
  tolerance in the last frame, or the local map held it along the velocity), contact/in-view-hole
  (ahead but neither), timeout/stopped-short (<1 m from the waypoint, never settled),
  timeout/stopped-at-obstacle (moved <0.5 m in the last 10 s with an obstacle <0.6 m: the shield or
  memory guard holding it), timeout/hover-open (moved <0.5 m, nothing within 0.6 m),
  timeout/no-route (still moving, never got halfway along the leg: did not find the
  gap / way over / under), timeout/wandering (moving, got past halfway but not in).
The grader has no out-of-bounds end: a flight ends on contact, completion or 40 s on one waypoint.
Clean pass: completed and never closer than 0.25 m (body radius 0.18 m + 7 cm) to a static obstacle
(16 horizontal + up/down rays every 4 ticks)."""
import csv, re, sys
from collections import Counter, defaultdict

CAUSES = ["contact/in-view-seen", "contact/in-view-hole", "contact/side-behind", "contact/floor-ceiling",
          "contact/moving-box", "contact/diagonal-thin", "timeout/stopped-short", "timeout/stopped-at-obstacle", "timeout/hover-open",
          "timeout/no-route", "timeout/wandering"]


def kind(name):
    if name.startswith("pt-"):
        return name.split("-")[1]
    if name.startswith("g3d-"):
        return name.split("-")[1]
    return re.sub(r"-[0-9]+$", "", name).replace("public-visible-", "public-")


def cause(r):
    if r["end"] == "contact":
        k = int(r["contact_kind"])
        if k in (3, 4):
            return "contact/floor-ceiling"
        if k == 5:
            return "contact/moving-box" if float(r.get("mover_gap", 9)) < .25 else "contact/diagonal-thin"
        if k in (1, 2):
            return "contact/side-behind"
        n, s = int(r["near_rays"]), int(r["near_seen"])
        if (n and s / n >= .5) or r["map_ahead"] == "1":
            return "contact/in-view-seen"
        return "contact/in-view-hole"
    d = {k: float(r[k]) for k in ("end_dist", "disp10", "clear10", "min_dist", "leg_dist")}
    if d["end_dist"] < 1.0:
        return "timeout/stopped-short"
    if d["disp10"] < .5:
        return "timeout/stopped-at-obstacle" if d["clear10"] < .6 else "timeout/hover-open"
    if d["leg_dist"] > 0 and d["min_dist"] > .5 * d["leg_dist"]:
        return "timeout/no-route"
    return "timeout/wandering"


def main():
    out_md, out_csv, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    md, rows = [], []
    summary = []
    for spec in specs:
        label, files = spec.split("=", 1)
        table = defaultdict(Counter)
        flights, done, clean = Counter(), Counter(), Counter()
        for fn in files.split(","):
            for r in csv.DictReader(open(fn)):
                k = kind(r["mission"])
                flights[k] += 1
                if r["end"] == "complete":
                    done[k] += 1
                    clean[k] += float(r["min_clear"]) >= .25
                    continue
                c = cause(r)
                table[k][c] += 1
                rows.append([label, r["mission"], r["seed"], k, c])
        kinds = sorted(flights)
        used = [c for c in CAUSES if any(table[k][c] for k in kinds)]
        md.append(f"\n### {label}\n")
        md.append("| type | flights | done | clean | " + " | ".join(c.split("/")[1] for c in used) + " |")
        md.append("|---|---:|---:|---:|" + "---:|" * len(used))
        for k in kinds + ["all"]:
            ks = kinds if k == "all" else [k]
            f_, d_, c_ = sum(flights[x] for x in ks), sum(done[x] for x in ks), sum(clean[x] for x in ks)
            cells = [str(sum(table[x][c] for x in ks) or "") for c in used]
            b = "**" if k == "all" else ""
            md.append(f"| {b}{k}{b} | {f_} | {d_} | {c_} | " + " | ".join(cells) + " |")
        tot = Counter()
        for k in kinds:
            tot.update(table[k])
        summary.append((label, sum(flights.values()), sum(done.values()), sum(clean.values()), tot))
    head = ["\n### Totals by cause\n", "| set | flights | done | clean | " + " | ".join(CAUSES) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(CAUSES)]
    for label, f_, d_, c_, tot in summary:
        head.append(f"| {label} | {f_} | {d_} | {c_} | " + " | ".join(str(tot[c] or "") for c in CAUSES) + " |")
    open(out_md, "w").write("\n".join(head + md) + "\n")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "mission", "seed", "type", "cause"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
