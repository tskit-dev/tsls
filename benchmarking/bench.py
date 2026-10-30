"""
Time tsls on the benchmark workloads.

Each query is matched several times and the best time is kept, so that
drift in machine speed affects the result less. Run this on an otherwise
idle machine, pinned to one core (taskset -c N).

There are two kinds of workload:

- tsinfer: the chr20p ancestor groups and samples made by tsinfer_data.py,
  with the model in their model.json. The "samples_map" workloads replace
  its constant rho with per-site probabilities from the HapMap II chr20 map
  (genetic_map.py), and "_mu1e-8" also replaces its mu. For these we also
  count the queries for which tsls finds the path that tsinfer found.
- sc2ts: queries from data/queries_scaling matched against the sc2ts
  sub-ARGs or the full benchmark ARG, with the sc2ts model
  (num_mismatches = 3, weight_by_n=False). sc2ts matches every sample first
  with a threshold of k = 0 and then k = 1 mismatches, and reruns those that
  fail exactly. The "_k1" workloads use the k = 1 threshold, and the others
  are exact.

    taskset -c 3 uv run python bench.py
    taskset -c 3 uv run python bench.py -w samples -w sc2ts_n100k
"""

import dataclasses
import pathlib
import time

import click
import numpy as np
import tsls

import matcher
import queries
import run as run_module

# A spread of copies and recombinants from the 110 scaling queries
SC2TS_QUERIES = list(range(0, 110, 11))


@dataclasses.dataclass
class Workload:
    name: str
    reference: str
    query_dir: str
    num_queries: int
    # None for tsinfer workloads, which use their model.json
    mismatch_threshold: float = None
    sc2ts: bool = False
    # A genetic map (relative to the data directory) and mu to use in place
    # of model.json's rho and mu
    genetic_map: str = None
    mu: float = None


GENETIC_MAP = "genetic_maps/chr20_HapMapII_GRCh38.txt"


def tsinfer_workload(name, num_queries, directory=None, **kwargs):
    path = f"tgp_chr20p/{name if directory is None else directory}"
    return Workload(name, f"{path}/reference.trees", path, num_queries, **kwargs)


def sc2ts_workload(name, reference, num_queries, mismatch_threshold=None):
    return Workload(
        name,
        reference,
        "queries_scaling",
        num_queries,
        mismatch_threshold=mismatch_threshold,
        sc2ts=True,
    )


WORKLOADS = [
    tsinfer_workload("ancestors_g84", 50),
    tsinfer_workload("ancestors_g1000", 50),
    tsinfer_workload("ancestors_g1483", 100),
    tsinfer_workload("ancestors_g1484", 50),
    tsinfer_workload("ancestors_g1485", 100),
    tsinfer_workload("samples", 6),
    tsinfer_workload("samples_map", 6, "samples", genetic_map=GENETIC_MAP),
    tsinfer_workload(
        "samples_map_mu1e-8", 6, "samples", genetic_map=GENETIC_MAP, mu=1e-8
    ),
    sc2ts_workload(
        "sc2ts_n100k_k1", "subargs/sc2ts_bench_n100000.trees", 10, mismatch_threshold=1
    ),
    sc2ts_workload("sc2ts_n100k", "subargs/sc2ts_bench_n100000.trees", 10),
    sc2ts_workload("sc2ts_n300k", "subargs/sc2ts_bench_n300000.trees", 4),
    sc2ts_workload("sc2ts_full_k1", "sc2ts_bench.trees", 3, mismatch_threshold=1),
]

# A smaller set for checking changes, which takes a few minutes
QUICK_WORKLOADS = ["ancestors_g84", "ancestors_g1485", "samples", "sc2ts_n100k"]


def load_workload(data_dir, workload):
    """
    Return the (ts, H, df, query_ids, model) tuple for the workload.
    """
    ts = run_module.load_ts(data_dir / workload.reference)
    H, _, df = queries.load_queries(data_dir / workload.query_dir)
    if workload.sc2ts:
        mu, rho = matcher.solve_num_mismatches(3)
        model = {"mu": mu, "rho": rho, "weight_by_n": False, "alleles": matcher.ALLELES}
        query_ids = SC2TS_QUERIES[: workload.num_queries]
    else:
        model = queries.load_model(data_dir / workload.query_dir)
        if workload.genetic_map is not None:
            model["rho"] = tsls.recombination_probabilities(
                data_dir / workload.genetic_map, ts.sites_position
            )
        if workload.mu is not None:
            model["mu"] = workload.mu
        query_ids = list(df.query_id[: workload.num_queries])
    return ts, H, df, query_ids, model


def make_matcher(ts, model, mismatch_threshold=None):
    return matcher.Matcher(
        ts,
        mu=model["mu"],
        rho=model["rho"],
        weight_by_n=model["weight_by_n"],
        alleles=model["alleles"],
        mismatch_threshold=mismatch_threshold,
    )


def same_as_tsinfer(df, query_id, result):
    """
    Return True if the path is the one tsinfer found for the query.
    """
    left, _, parent, _, _ = result
    row = df[df.query_id == query_id].iloc[0]
    truth_parents = queries.parse_list(row.truth_parents)
    truth_breakpoints = queries.parse_list(row.truth_breakpoints)
    return list(parent) == truth_parents and list(left[1:]) == truth_breakpoints


def time_workload(data_dir, workload, repeats):
    """
    Return a dict summarising the workload: the best time for each query,
    the largest memory reported, the mean traceback size, and for tsinfer
    workloads the number of queries with tsinfer's path.
    """
    ts, H, df, query_ids, model = load_workload(data_dir, workload)
    before = time.perf_counter()
    m = make_matcher(ts, model, mismatch_threshold=workload.mismatch_threshold)
    setup_time = time.perf_counter() - before
    best = np.full(len(query_ids), np.inf)
    memory = np.zeros(len(query_ids))
    traceback = np.zeros(len(query_ids))
    num_same = 0
    for j, query_id in enumerate(query_ids):
        start, end = queries.query_window(H, query_id)
        h = H[query_id]
        for _ in range(repeats):
            before = time.perf_counter()
            result = m.match(h, start, end)
            best[j] = min(best[j], time.perf_counter() - before)
        stats = result[4]
        memory[j] = stats["total_memory"]
        traceback[j] = stats["mean_traceback_size"]
        if not workload.sc2ts:
            num_same += same_as_tsinfer(df, query_id, result)
    return {
        "times": best,
        "setup_time": setup_time,
        "max_memory": float(np.max(memory)),
        "mean_traceback": float(np.mean(traceback)),
        "num_same": None if workload.sc2ts else num_same,
        "num_nodes": ts.num_nodes,
    }


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data",
)
@click.option("-w", "--workload", "workload_names", multiple=True)
@click.option("-r", "--repeats", type=int, default=3)
@click.option("--quick", is_flag=True, help="Use the smaller set of workloads")
def main(data_dir, workload_names, repeats, quick):
    """
    Print a markdown table of tsls's times and memory for each workload.
    """
    if quick:
        workload_names += tuple(QUICK_WORKLOADS)
    workloads = WORKLOADS
    if len(workload_names) > 0:
        workloads = [w for w in WORKLOADS if w.name in workload_names]
    click.echo(
        "| workload | nodes | queries | setup (s) | total (s) | mean (s) | max (s) "
        "| max memory (MiB) | mean traceback | tsinfer's path |"
    )
    click.echo("|---|---|---|---|---|---|---|---|---|---|")
    for workload in workloads:
        r = time_workload(pathlib.Path(data_dir), workload, repeats)
        times = r["times"]
        same = "" if r["num_same"] is None else f"{r['num_same']}/{len(times)}"
        click.echo(
            f"| {workload.name} | {r['num_nodes']:,} | {len(times)} "
            f"| {r['setup_time']:.2f} | {np.sum(times):.2f} | {np.mean(times):.4f} "
            f"| {np.max(times):.4f} | {r['max_memory'] / 2**20:.0f} "
            f"| {r['mean_traceback']:.3g} | {same} |"
        )


if __name__ == "__main__":
    main()
