"""
Run tsls Viterbi on chr20p sample queries, for profiling and step counts.

The reference is the samples workload's reference.trees with the vestigial
root edges from node 0 removed (as engines.TslsEngine does), and the model
is tsinfer's: r = 0.01 or per-site probabilities from a genetic map, with
weight_by_n=True (tsls's model). Prints the time, a hash of the path, the
path's model log likelihood and the engine's statistics for each query, and the step counters if the extension was
built with TSLS_STEP_COUNTERS.

    uv run python profile_samples.py --map data/genetic_maps/chr20_HapMapII_GRCh38.txt
"""

import hashlib
import json
import time

import click
import numpy as np
import tskit
import tsls

import engines
import queries

WORKLOAD = "data/tgp_chr20p/samples"


def load_reference(path):
    ts = tskit.load(path)
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    return tables.tree_sequence()


@click.command()
@click.option("--map", "genetic_map", default=None, help="HapMap genetic map")
@click.option("--mu", type=float, default=1e-20)
@click.option("--queries", "query_ids", default="0,40,80,120,160,199")
@click.option("--missing", is_flag=True, help="Use all-missing haplotypes")
@click.option("--repeats", type=int, default=1)
def main(genetic_map, mu, query_ids, missing, repeats):
    ts = load_reference(f"{WORKLOAD}/reference.trees")
    H, positions, _ = queries.load_queries(WORKLOAD)
    m = ts.num_sites
    if genetic_map is None:
        r = np.full(m, 0.01)
    else:
        r = tsls.model.recombination_probabilities(genetic_map, positions)
    reference = tsls.TreeSequence(ts)
    num_alleles = engines.site_num_alleles(ts, None)
    mu_array = np.full(m, mu)
    for qid in [int(q) for q in query_ids.split(",")]:
        h = H[qid].astype(np.int8)
        start, end = queries.query_window(H, qid)
        if missing:
            h[start:end] = -1
        for _ in range(repeats):
            hmm = tsls.Hmm(reference, r, np.full(m, mu))
            matched = np.zeros(m, dtype=np.int8)
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
                rho=r,
                mu=mu_array,
                n=ts.num_nodes,
                num_alleles=num_alleles,
            )
            row = {
                "query": qid,
                "time": round(seconds, 4),
                "segments": len(left),
                "path": digest.hexdigest()[:12],
                "log_likelihood": log_likelihood,
                "mean_values": round(hmm.mean_num_values, 1),
                "traceback": round(hmm.mean_traceback_size, 1),
                "delta": round(getattr(hmm, "mean_delta_size", float("nan")), 2),
            }
            counts = getattr(hmm, "step_counts", None)
            if counts is not None:
                row.update(counts)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
