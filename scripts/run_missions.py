#!/usr/bin/env python3
"""Fly every mission in a portfolio and summarise arrivals, contacts and flight time."""
import argparse
import concurrent.futures
import os
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIOS = ("source", "public", "fresh")

# Navigator configuration used for every reported result.
NAVIGATOR_FLAGS = [
    "1", "0", "1", "9",
    "--standoff-legs", "2",
    "--reacquire", "0",
    "--camera-pitch", "0.0",
    "--ray-slope", "0.75",
    "--attention-motion",
    "--attention-window", "0.2",
    "--attention-floor", "0.35",
    "--ring-retreat",
    "--ring-extra", "0.02",
]


def fly(binary, task, output):
    report = output / f"{task.stem}.json"
    command = [str(binary), str(task), str(ROOT / "assets/raptor.bin"), str(report), *NAVIGATOR_FLAGS]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"{task.stem} failed:\n{process.stderr[-2000:]}")
    return json.loads(report.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolios", nargs="*", choices=PORTFOLIOS,
                        help="portfolios to fly (default: all)")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--only", help="fly a single mission by name")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="missions flown in parallel (use 1 for clean planner timings)")
    parser.add_argument("--require-success", action="store_true",
                        help="exit with an error unless every flown mission succeeds")
    args = parser.parse_args()
    args.portfolios = args.portfolios or list(PORTFOLIOS)

    totals = []
    for portfolio in args.portfolios:
        output = args.output / portfolio
        output.mkdir(parents=True, exist_ok=True)
        tasks = sorted((ROOT / "missions" / portfolio).glob("*.task"))
        if args.only:
            tasks = [task for task in tasks if task.stem == args.only]
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(lambda task: fly(args.binary, task, output), tasks))
        rows = []
        for task, result in zip(tasks, results):
            rows.append({
                "mission": task.stem,
                "outcome": result["outcome"],
                "waypoints": f"{result['goals_completed']}/{result['goals_requested']}",
                "contact": result["contact"],
                "time_s": result["time_s"],
                "path_m": round(result["path_m"], 2),
                "planner_mean_ms": round(result["planner_mean_us"] / 1000, 2),
            })
            row = rows[-1]
            print(f"{portfolio:7s} {row['mission']:28s} {row['outcome']:20s} "
                  f"{row['waypoints']:>4s}  {row['time_s']:6.1f} s  {row['path_m']:6.2f} m", flush=True)
        if rows:
            with (output / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        arrived = sum(row["outcome"] == "success" for row in rows)
        contacts = sum(row["contact"] for row in rows)
        totals.append((portfolio, arrived, len(rows), contacts))

    print()
    for portfolio, arrived, count, contacts in totals:
        print(f"{portfolio:7s} arrived {arrived}/{count}  contacts {contacts}")
    if args.require_success and any(arrived != count or contacts for _, arrived, count, contacts in totals):
        sys.exit(1)


if __name__ == "__main__":
    main()
