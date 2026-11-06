"""Per mission kind table from run_closed_loop.sh CSVs: completion / contact flights / s per waypoint.
  python closed_loop_table.py OUTDIR NAME..."""
import csv, sys, os, json
out = sys.argv[1]
kinds = ["open", "gap", "vertical", "low_block"]
print("| run | all | " + " | ".join(kinds) + " | rays: hole / phantom / false-free / k |")
print("|---" * (len(kinds) + 3) + "|")
for name in sys.argv[2:]:
    rows = list(csv.DictReader(open(os.path.join(out, name + ".csv"))))
    def cell(rs):
        n = len(rs); c = sum(int(r["complete"]) for r in rs); k = sum(int(r["contact"]) > 0 for r in rs)
        done = [float(r["waypoint_s"]) for r in rs if int(r["complete"])]
        return f"{c / n:.3f}/{k}/{(sum(done) / len(done) if done else 0):.1f}"
    per = [cell([r for r in rows if r["mission"].split("-")[1] == k]) for k in kinds]
    rj = os.path.join(out, name + "_rays.json")
    rays = ""
    if os.path.exists(rj):
        j = json.load(open(rj)); rays = f'{j["desp_hole"]:.3f} / {j["desp_phantom"]:.3f} / {j["desp_false_free"]:.3f} / {j["desp_k"]:.3f}'
    print(f"| {name} | {cell(rows)} | " + " | ".join(per) + f" | {rays} |")
