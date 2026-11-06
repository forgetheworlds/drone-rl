"""Small dev-split parameter sweep for the filtered SGBM / BM families (never touches test)."""
import itertools, json, sys
import methods, run_bench

def make(base, **kw):
    name = base.name + "_" + "_".join(f"{k}{v}" for k, v in kw.items())
    cls = type(name, (base,), dict(name=name, **kw))
    methods.REGISTRY[name] = cls
    return name

grid = []
for tex, uniq, block, sc in itertools.product((4, 8, 12), (5, 15), (5, 9), (1.0, 0.5)):
    nd = 64 if sc == 1.0 else 32
    grid.append(make(methods.SGBMFiltered320, tex=float(tex), uniq=uniq, block=block if sc == 1.0 else max(3, block - 2), scale=sc, nd=nd))
for n in grid:
    r = run_bench.run(n, "dev", sys.argv[1] if len(sys.argv) > 1 else "ov2640", max_seqs=3)
    print(json.dumps({k: r[k] for k in ("method", "planner_cost", "ray_false_free_near", "ray_false_occ", "ray_unknown", "free_coverage", "ms_box")}), flush=True)
