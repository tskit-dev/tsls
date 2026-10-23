"""
Compare the wall time of tsls and tsinfer-hmm on the tsinfer workloads with
paired runs.

run.py runs all the queries with one engine and then all with the next, so
drift in machine speed shows up as a difference between the engines. Here
each query is matched several times with each engine, alternating between
them, and the best time for each engine is kept. The ratio for a workload is
the total tsls time over the total tsinfer-hmm time. Run this on an
otherwise idle machine.
"""

import pathlib
import time

import click
import numpy as np

import callgrind
import queries
import run as run_module

ENGINES = ["tsinfer", "tsls"]

DEFAULT_WORKLOADS = [
    ("ancestors_g84", 50),
    ("ancestors_g1000", 50),
    ("ancestors_g1483", 100),
    ("ancestors_g1484", 50),
    ("ancestors_g1485", 100),
    ("samples", 6),
]

# A smaller set for checking changes, which takes about two minutes
QUICK_WORKLOADS = [
    ("ancestors_g84", 50),
    ("ancestors_g1485", 30),
    ("samples", 2),
]


def time_workload(workload_dir, num_queries, repeats):
    """
    Return the (best_times, same_paths) tuple for the first num_queries
    queries, where best_times[engine] is an array of the best time for each
    query, and same_paths is True if the engines found the same paths.
    """
    model = queries.load_model(workload_dir)
    ts = run_module.load_ts(workload_dir / "reference.trees")
    H, _, df = queries.load_queries(workload_dir)
    engines = {name: callgrind.make_engine(name, ts, model) for name in ENGINES}
    query_ids = list(df.query_id[:num_queries])
    best = {name: np.full(len(query_ids), np.inf) for name in ENGINES}
    same_paths = True
    for j, query_id in enumerate(query_ids):
        start, end = queries.query_window(H, query_id)
        h = H[query_id]
        parents = {}
        for rep in range(repeats):
            # Alternate which engine goes first
            order = ENGINES if rep % 2 == 0 else ENGINES[::-1]
            for name in order:
                before = time.perf_counter()
                _, _, parent, _, _ = engines[name].match(h, start, end)
                elapsed = time.perf_counter() - before
                best[name][j] = min(best[name][j], elapsed)
                parents[name] = list(parent)
        same_paths = same_paths and parents["tsinfer"] == parents["tsls"]
    return best, same_paths


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/tgp_chr20p",
)
@click.option("-w", "--workload", "workload_names", multiple=True)
@click.option("-r", "--repeats", type=int, default=3)
@click.option("--quick", is_flag=True, help="Use the smaller set of workloads")
def main(data_dir, workload_names, repeats, quick):
    """
    Print a markdown table of paired wall times for each workload.
    """
    workloads = QUICK_WORKLOADS if quick else DEFAULT_WORKLOADS
    if len(workload_names) > 0:
        workloads = [w for w in DEFAULT_WORKLOADS if w[0] in workload_names]
    click.echo(
        "| workload | queries | tsinfer total (s) | tsls total (s) | ratio "
        "| median per-query ratio |"
    )
    click.echo("|---|---|---|---|---|---|")
    for workload, num_queries in workloads:
        best, same_paths = time_workload(
            pathlib.Path(data_dir) / workload, num_queries, repeats
        )
        total = {name: float(np.sum(best[name])) for name in ENGINES}
        median = float(np.median(best["tsls"] / best["tsinfer"]))
        click.echo(
            f"| {workload} | {num_queries} | {total['tsinfer']:.2f} | "
            f"{total['tsls']:.2f} | {total['tsls'] / total['tsinfer']:.3f} "
            f"| {median:.3f} |",
        )
        if not same_paths:
            click.echo(f"WARNING: engines found different paths for {workload}")


if __name__ == "__main__":
    main()
