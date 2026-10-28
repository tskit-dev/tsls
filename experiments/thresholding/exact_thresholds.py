"""
Compare tsls Viterbi on sc2ts sub-ARG queries: exact, sc2ts's floor at
0.99 mu^k, and the exact prune threshold for k mismatches
(tsls.model.path_threshold). Reports the time, the mean traceback size,
and the numbers of paths that are certified (odds >= the threshold, so
provably optimal) and optimal.

Run from benchmarking/, which has the data:

    cd ../../benchmarking
    uv run python ../experiments/thresholding/exact_thresholds.py 100000
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarking"))
import engines  # noqa: E402
import queries  # noqa: E402
import run as run_module  # noqa: E402
import tsls  # noqa: E402

size = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
query_ids = range(0, 110, 11)
ts = run_module.load_ts(f"data/subargs/sc2ts_bench_n{size}.trees")
H, _, _ = queries.load_queries("data/queries_scaling")
tables = ts.dump_tables()
tables.edges.keep_rows(tables.edges.parent != 0)
ref = tsls.TreeSequence(tables.tree_sequence(), alleles=engines.ALLELES)
n = ts.num_nodes
m = ts.num_sites
mu, q = engines.solve_num_mismatches(3)
r = np.full(m, tsls.model.recombination_probability(q, n))
mu_a = np.full(m, mu)
A = len(engines.ALLELES)
log_e = np.log(tsls.model.mismatch_odds(mu, A))
log_q = np.log(q)
print(f"n={n} m={m} q={q:.3g} r={r[0]:.3g}")


def run(h, start, end, **kwargs):
    am = tsls.Hmm(ref, r, mu_a, **kwargs)
    matched = np.zeros(m, dtype=np.int8)
    before = time.perf_counter()
    left, _, _ = am.find_path(h, start, end, matched)
    t = time.perf_counter() - before
    hs, ms = h[start:end], matched[start:end]
    k = int(np.sum((hs != -1) & (hs != ms)))
    s = len(left) - 1
    return t, am.mean_traceback_size, k * log_e + s * log_q, k, s


for k in [0, 1, 2, 3]:
    t_exact = tsls.model.path_threshold(k, mu, A)
    settings = {
        "exact": {},
        "floor 0.99mu^k": {"likelihood_threshold": 0.99 * mu**k},
        "prune e^k": {"prune_threshold": t_exact},
    }
    totals = {name: [0.0, 0.0, 0, 0, 0] for name in settings}
    for qid in query_ids:
        start, end = queries.query_window(H, qid)
        h = H[qid].astype(np.int8)
        res = {name: run(h, start, end, **kw) for name, kw in settings.items()}
        best = res["exact"][2]
        for name, (t, tb, lo, _, _) in res.items():
            tot = totals[name]
            tot[0] += t
            tot[1] += tb
            certified = lo >= np.log(t_exact)
            optimal = abs(lo - best) <= 1e-9 * abs(best) + 1e-12
            tot[2] += certified
            tot[3] += optimal
            tot[4] += certified and not optimal
    print(f"k={k} t={t_exact:.4g}")
    for name, (t, tb, c, o, bad) in totals.items():
        print(
            f"  {name:16s} time={t:7.3f}s mean_tb={tb / len(query_ids):9.1f} "
            f"certified={c}/{len(query_ids)} optimal={o}/{len(query_ids)} "
            f"certified_but_suboptimal={bad}"
        )
