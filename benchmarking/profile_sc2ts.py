"""
Run tsls Viterbi on sc2ts queries, for profiling and comparing builds.

The queries are those of paired.py's sc2ts workloads (a spread of the
110 scaling queries), matched against an sc2ts ARG with the sc2ts model
(num_mismatches = 3, weight_by_n=False) and optionally the k-mismatch
threshold. Prints, for each query, the time, a hash of the path, the
path's model log likelihood and the engine's statistics, as JSON.

    uv run python profile_sc2ts.py data/subargs/sc2ts_bench_n100000.trees --k 1
"""

import hashlib
import json
import pathlib
import time

import click
import numpy as np

import engines
import paired


@click.command()
@click.argument("ts_path")
@click.option("--k", "mismatch_threshold", type=int, default=None)
@click.option("--queries", "query_ids", default=None)
@click.option("--sparse/--no-sparse", default=True)
@click.option("--repeats", type=int, default=1)
def main(ts_path, mismatch_threshold, query_ids, sparse, repeats):
    path = pathlib.Path(ts_path)
    workload = paired.Workload(
        "sc2ts",
        str(path.relative_to("data")),
        "queries_scaling",
        10,
        mismatch_threshold=mismatch_threshold,
        sc2ts=True,
    )
    ts, H, ids, model = paired.load_workload(pathlib.Path("data"), workload)
    if query_ids is not None:
        ids = [int(q) for q in query_ids.split(",")]
    engine = engines.TslsEngine(
        ts,
        mu=model["mu"],
        rho=model["rho"],
        weight_by_n=model["weight_by_n"],
        alleles=model["alleles"],
        mismatch_threshold=mismatch_threshold,
    )
    num_alleles = engines.site_num_alleles(ts, model["alleles"])
    mu = np.full(ts.num_sites, model["mu"])
    for qid in ids:
        h = H[qid].astype(np.int8)
        start, end = paired.queries.query_window(H, qid)
        for _ in range(repeats):
            hmm = engine.make_matcher() if sparse else engines.tsls.Hmm(
                engine.reference,
                engine.recombination,
                engine.mismatch,
                prune_threshold=engine.prune_threshold,
                sparse=False,
            )
            matched = np.zeros(ts.num_sites, dtype=np.int8)
            before = time.perf_counter()
            left, right, parent = hmm.find_path(h, start, end, matched)
            seconds = time.perf_counter() - before
            digest = hashlib.md5()
            for array in [left, right, parent, matched]:
                digest.update(np.ascontiguousarray(array).tobytes())
            log_likelihood = engines.model_path_log_likelihood(
                h,
                matched,
                left[::-1],
                ts.sites_position,
                start,
                end,
                rho=engine.recombination,
                mu=mu,
                n=ts.num_nodes,
                num_alleles=num_alleles,
            )
            row = {
                "query": qid,
                "time": round(seconds, 4),
                "segments": len(left),
                "path": digest.hexdigest()[:12],
                "log_likelihood": log_likelihood,
                "sparse_sites": hmm.num_sparse_sites,
                "mean_values": round(hmm.mean_num_values, 1),
                "updated": round(hmm.mean_num_updated, 1),
                "delta": round(getattr(hmm, "mean_delta_size", float("nan")), 2),
            }
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
