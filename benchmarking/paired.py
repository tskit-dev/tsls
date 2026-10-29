"""
Compare the wall time of tsls and tsinfer-hmm with paired runs.

run.py runs all the queries with one engine and then all with the next, so
drift in machine speed shows up as a difference between the engines. Here
each query is matched several times with each engine, alternating between
them, and the best time for each engine is kept. The ratio for a workload is
the total tsls time over the total tsinfer-hmm time. Run this on an
otherwise idle machine.

There are two kinds of workload:

- tsinfer: the chr20p ancestor groups and samples made by tsinfer_data.py,
  with the model in their model.json. The "samples_map" workloads replace
  its constant rho with per-site probabilities from the HapMap II chr20 map
  (genetic_map.py), and "_mu1e-8" also replaces its mu.
- sc2ts: queries from data/queries_scaling matched against the sc2ts
  sub-ARGs, with the sc2ts model (num_mismatches = 3, weight_by_n=False).
  sc2ts matches every sample first with a likelihood threshold of k = 0
  and then k = 1 mismatches, and reruns those that fail at high precision.
  The "_k1" workloads use the k = 1 threshold, and the others are exact.

Paths are checked against tsinfer-hmm's: a different path must have the
same log likelihood under the model (a tie). With a mismatch threshold
(the "_k1" workloads), tsinfer-hmm optimises floored values, in which
nodes with different numbers of mismatches can tie. tsls prunes at the
exact threshold instead (see engines.AncestorMatcherEngine), so it may
report a better path.
"""

import dataclasses
import pathlib
import time

import click
import numpy as np
import tsls

import callgrind
import engines
import queries
import run as run_module

ENGINES = ["tsinfer", "tsls"]

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


def sc2ts_workload(name, num_samples, num_queries, mismatch_threshold=None):
    return Workload(
        name,
        f"subargs/sc2ts_bench_n{num_samples}.trees",
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
    sc2ts_workload("sc2ts_n100k_k1", 100000, 10, mismatch_threshold=1),
    sc2ts_workload("sc2ts_n100k", 100000, 10),
    sc2ts_workload("sc2ts_n300k", 300000, 4),
]

TSINFER_WORKLOADS = [w.name for w in WORKLOADS if not w.sc2ts]
SC2TS_WORKLOADS = [w.name for w in WORKLOADS if w.sc2ts]

# A smaller set for checking changes, which takes a few minutes
QUICK_WORKLOADS = [
    tsinfer_workload("ancestors_g84", 50),
    tsinfer_workload("ancestors_g1485", 30),
    tsinfer_workload("samples", 2),
    sc2ts_workload("sc2ts_n100k", 100000, 3),
]


def load_workload(data_dir, workload):
    """
    Return the (ts, H, query_ids, engine_args) tuple for the workload.
    """
    ts = run_module.load_ts(data_dir / workload.reference)
    H, _, df = queries.load_queries(data_dir / workload.query_dir)
    if workload.sc2ts:
        mu, rho = engines.solve_num_mismatches(3)
        model = {"mu": mu, "rho": rho, "weight_by_n": False, "alleles": engines.ALLELES}
        query_ids = SC2TS_QUERIES[: workload.num_queries]
    else:
        model = queries.load_model(data_dir / workload.query_dir)
        if workload.genetic_map is not None:
            model["rho"] = tsls.model.recombination_probabilities(
                data_dir / workload.genetic_map, ts.sites_position
            )
        if workload.mu is not None:
            model["mu"] = workload.mu
        query_ids = list(df.query_id[: workload.num_queries])
    return ts, H, query_ids, model


def time_workload(data_dir, workload, repeats):
    """
    Return the (best_times, max_memory, path_counts) tuple for the workload,
    where best_times[engine] is an array of the best time for each query,
    max_memory[engine] the largest total_memory reported, and path_counts
    the numbers of queries for which tsls found the same path as
    tsinfer-hmm, a different path with the same likelihood, and a different
    path that is better or worse.
    """
    ts, H, query_ids, model = load_workload(data_dir, workload)
    engine_objs = {
        name: callgrind.make_engine(
            name, ts, model, mismatch_threshold=workload.mismatch_threshold
        )
        for name in ENGINES
    }
    n = ts.num_nodes if model["weight_by_n"] else 1
    num_alleles = engines.site_num_alleles(ts, model["alleles"])
    rates = {
        "rho": np.broadcast_to(model["rho"], ts.num_sites),
        "mu": np.broadcast_to(model["mu"], ts.num_sites),
    }
    best = {name: np.full(len(query_ids), np.inf) for name in ENGINES}
    max_memory = {name: 0 for name in ENGINES}
    path_counts = {"same": 0, "tie": 0, "better": 0, "worse": 0}
    for j, query_id in enumerate(query_ids):
        start, end = queries.query_window(H, query_id)
        h = H[query_id]
        results = {}
        for rep in range(repeats):
            # Alternate which engine goes first
            order = ENGINES if rep % 2 == 0 else ENGINES[::-1]
            for name in order:
                before = time.perf_counter()
                result = engine_objs[name].match(h, start, end)
                elapsed = time.perf_counter() - before
                best[name][j] = min(best[name][j], elapsed)
                results[name] = result
        for name in ENGINES:
            max_memory[name] = max(max_memory[name], results[name][4]["total_memory"])
        if list(results["tsinfer"][2]) == list(results["tsls"][2]):
            path_counts["same"] += 1
            continue
        ll = {}
        for name in ENGINES:
            left, _, _, matched, _ = results[name]
            ll[name] = engines.model_path_log_likelihood(
                h,
                matched,
                left,
                ts.sites_position,
                start,
                end,
                n=n,
                num_alleles=num_alleles,
                **rates,
            )
        diff = ll["tsls"] - ll["tsinfer"]
        if abs(diff) <= 1e-9 * abs(ll["tsinfer"]):
            path_counts["tie"] += 1
        elif diff > 0:
            path_counts["better"] += 1
        else:
            path_counts["worse"] += 1
    return best, max_memory, path_counts


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data",
)
@click.option("-w", "--workload", "workload_names", multiple=True)
@click.option("--tsinfer", "only_tsinfer", is_flag=True, help="tsinfer workloads only")
@click.option("--sc2ts", "only_sc2ts", is_flag=True, help="sc2ts workloads only")
@click.option("-r", "--repeats", type=int, default=3)
@click.option("--quick", is_flag=True, help="Use the smaller set of workloads")
def main(data_dir, workload_names, only_tsinfer, only_sc2ts, repeats, quick):
    """
    Print a markdown table of paired wall times for each workload.
    """
    workloads = QUICK_WORKLOADS if quick else WORKLOADS
    if only_tsinfer:
        workload_names += tuple(TSINFER_WORKLOADS)
    if only_sc2ts:
        workload_names += tuple(SC2TS_WORKLOADS)
    if len(workload_names) > 0:
        workloads = [w for w in WORKLOADS if w.name in workload_names]
    click.echo(
        "| workload | queries | tsinfer total (s) | tsls total (s) | ratio "
        "| median per-query ratio | tsinfer memory (MiB) | tsls memory (MiB) "
        "| paths same/tie/better/worse |"
    )
    click.echo("|---|---|---|---|---|---|---|---|---|")
    for workload in workloads:
        best, memory, paths = time_workload(pathlib.Path(data_dir), workload, repeats)
        total = {name: float(np.sum(best[name])) for name in ENGINES}
        median = float(np.median(best["tsls"] / best["tsinfer"]))
        click.echo(
            f"| {workload.name} | {len(best['tsls'])} | {total['tsinfer']:.2f} | "
            f"{total['tsls']:.2f} | {total['tsls'] / total['tsinfer']:.3f} "
            f"| {median:.3f} | {memory['tsinfer'] / 2**20:.0f} "
            f"| {memory['tsls'] / 2**20:.0f} "
            f"| {paths['same']}/{paths['tie']}/{paths['better']}/{paths['worse']} |",
        )
        if paths["worse"] > 0 or paths["better"] > 0:
            click.echo(f"WARNING: paths with different likelihoods in {workload.name}")


if __name__ == "__main__":
    main()
