"""
What do we lose, and gain, by coarsening the recombination map or the
Viterbi likelihood values?

Under a recombination map the switch odds differ at every site, so the
Viterbi values of nodes that switched at different sites are never equal,
and compression (dropping a node's value when it equals its parent's)
works less well. On chr20p sample matching the map gave 8x the traceback
of tsinfer's constant rate (benchmarking/README.md). This runs Viterbi on
chr20p samples under the HapMap II chr20 map, and under coarsened
versions of it:

Input coarsening (per-site recombination probabilities r):

- map: the map itself (the reference);
- constant: tsinfer's r = 0.01 at every site;
- constant_mean: the map's mean r at every site, which has the map's
  genetic length and scale but no variation at all;
- uniform: a constant rate per bp with the map's total genetic length,
  so r still varies with the distance between sites;
- smooth_W: the map averaged over windows of W bp;
- round_F: each r rounded to the nearest power of F (in log space).

Value coarsening (Hmm(log_value_step=d)): the map, with each normalised
Viterbi value rounded to the nearest exp(-k d).

Every path is scored under the map: loss is the map's optimal log odds
minus the log odds of the path, so 0 means optimal under the map.

    uv run python coarsening.py -t 2            # about 10 minutes
    uv run python coarsening.py --analyse-only
"""

import argparse
import concurrent.futures
import functools
import logging
import os
import pathlib
import resource
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import tskit
import tqdm

import tsls

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parents[1] / "benchmarking" / "data"
SAMPLES = DATA / "tgp_chr20p" / "samples"
GENETIC_MAP = DATA / "genetic_maps" / "chr20_HapMapII_GRCh38.txt"
RESULTS = HERE / "results"

MEMORY_LIMIT = 5 * 1024**3
MISSING = -1
NUM_SAMPLES = 6
MISMATCH = [1e-20, 1e-8]
TGP_RHO = 0.01
SMOOTH_WINDOWS = [100_000, 1_000_000]
ROUND_FACTORS = [2, 10]
LOG_VALUE_STEPS = [0.1, 0.5, 1, float(np.log(10))]


def load_reference():
    """
    The samples reference, with the vestigial root edges from node 0
    removed (as in benchmarking/engines.py).
    """
    ts = tskit.load(SAMPLES / "reference.trees")
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    return tables.tree_sequence()


def load_queries():
    """
    Return the (get, num_queries) tuple, where get(j) is the (h, start, end)
    tuple of query j, from the windowed query format.
    """
    data = np.load(SAMPLES / "queries.npz")
    num_sites = len(data["positions"])
    start = data["start"].astype(int)
    end = data["end"].astype(int)
    offsets = np.concatenate([[0], np.cumsum(end - start)])
    values = data["values"]

    def get(j):
        h = np.full(num_sites, MISSING, dtype=np.int8)
        h[start[j] : end[j]] = values[offsets[j] : offsets[j + 1]]
        return h, int(start[j]), int(end[j])

    return get, len(start)


def haldane(distance):
    """Per-site recombination probabilities from genetic distances (Morgans)."""
    r = np.zeros(len(distance) + 1)
    r[1:] = (1 - np.exp(-2 * distance)) / 2
    return r


def round_log(x, factor):
    """
    Round positive values to the nearest power of factor in log space,
    leaving zeros.
    """
    out = np.zeros_like(x)
    positive = x > 0
    step = np.log(factor)
    out[positive] = np.exp(step * np.round(np.log(x[positive]) / step))
    return out


def recombination_variants(positions):
    """
    Return the dict of named per-site recombination probabilities.
    """
    genetic = tsls.model.read_genetic_map(GENETIC_MAP, positions)
    r_map = haldane(np.diff(genetic))
    variants = {"map": r_map, "constant": np.full(len(positions), TGP_RHO)}
    variants["constant_mean"] = np.full(len(positions), np.mean(r_map[1:]))
    span = positions[-1] - positions[0]
    rate = (genetic[-1] - genetic[0]) / span
    variants["uniform"] = haldane(np.diff(positions) * rate)
    for window in SMOOTH_WINDOWS:
        grid = np.arange(0, positions[-1] + window, window, dtype=float)
        smoothed = np.interp(
            positions, grid, tsls.model.read_genetic_map(GENETIC_MAP, grid)
        )
        variants[f"smooth_{window // 1000}kb"] = haldane(np.diff(smoothed))
    for factor in ROUND_FACTORS:
        variants[f"round_{factor}"] = round_log(r_map, factor)
    return variants


def strategies():
    """
    Return the list of (strategy, parameter, recombination variant, Hmm
    kwargs) tuples.
    """
    out = [("exact", "", "map", {})]
    for name in ["constant", "constant_mean", "uniform"]:
        out.append(("input", "", name, {}))
    for window in SMOOTH_WINDOWS:
        out.append(("input", "", f"smooth_{window // 1000}kb", {}))
    for factor in ROUND_FACTORS:
        out.append(("input", "", f"round_{factor}", {}))
    for step in LOG_VALUE_STEPS:
        out.append(("value_step", f"{step:.3g}", "map", {"log_value_step": step}))
    out.append(("both", "2.3", "round_10", {"log_value_step": float(np.log(10))}))
    return out


@functools.lru_cache(maxsize=1)
def load_workload():
    ts = load_reference()
    get, _ = load_queries()
    num_alleles = np.full(ts.num_sites, 2)
    return (
        ts,
        tsls.TreeSequence(ts),
        get,
        recombination_variants(ts.sites_position),
        (num_alleles),
    )


def path_log_odds(h, matched, left, positions, start, end, log_q, log_e):
    """
    Return the (log odds, num_mismatches, num_switches, switch sites) tuple of
    a path relative to staying on a node that matches at every site.
    """
    sites = np.arange(start, end)
    mismatch = (h[sites] != MISSING) & (h[sites] != matched[sites])
    switch_sites = np.searchsorted(positions, np.sort(left)[1:])
    value = np.sum(log_e[sites][mismatch]) + np.sum(log_q[switch_sites])
    return float(value), int(np.sum(mismatch)), len(switch_sites), switch_sites


def run_task(task):
    query_id, mu_value = task
    ts, tree_sequence, get, variants, num_alleles = load_workload()
    h, start, end = get(query_id)
    m = ts.num_sites
    n = ts.num_nodes
    positions = ts.sites_position
    mu = np.full(m, mu_value)
    with np.errstate(divide="ignore"):
        log_q = np.log(tsls.model.switch_odds(variants["map"], n))
    log_e = np.log(tsls.model.mismatch_odds(mu, num_alleles))
    rows = []
    exact = None
    for strategy, parameter, variant, kwargs in strategies():
        r = variants[variant]
        am = tsls.Hmm(tree_sequence, r, mu, **kwargs)
        matched = np.zeros(m, dtype=np.int8)
        before = time.perf_counter()
        left, _, _ = am.find_path(h, start, end, matched)
        seconds = time.perf_counter() - before
        odds, k, s, switch_sites = path_log_odds(
            h, matched, left, positions, start, end, log_q, log_e
        )
        if exact is None:
            exact = (odds, set(switch_sites), seconds, am.mean_traceback_size)
        rows.append(
            {
                "query": query_id,
                "mu": mu_value,
                "strategy": strategy,
                "parameter": parameter,
                "variant": variant,
                "time": seconds,
                "mean_values": am.mean_num_values,
                "max_values": am.max_num_values,
                "traceback_size": am.mean_traceback_size,
                "total_memory": am.total_memory,
                "num_mismatches": k,
                "num_switches": s,
                "map_log_odds": odds,
                "loss": exact[0] - odds,
                "breakpoints_kept": len(exact[1] & set(switch_sites))
                / max(1, len(exact[1])),
                "time_ratio": seconds / exact[2],
                "traceback_ratio": am.mean_traceback_size / exact[3],
            }
        )
    return rows


def limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


def make_tasks(num_samples):
    _, num_queries = load_queries()
    queries = np.unique(np.linspace(0, num_queries - 1, num_samples).astype(int))
    return [(int(q), mu) for mu in MISMATCH for q in queries]


def run(num_samples, num_processes, results):
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        num_processes, initializer=limit_memory
    ) as executor:
        futures = [executor.submit(run_task, t) for t in make_tasks(num_samples)]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            rows.extend(future.result())
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / "viterbi.csv", index=False)


def describe_variants(results):
    """
    Summarise the per-site recombination probabilities of each variant: the
    number of distinct values and their range.
    """
    ts = load_reference()
    rows = []
    for name, r in recombination_variants(ts.sites_position).items():
        r = r[1:]
        rows.append(
            {
                "variant": name,
                "distinct r": len(np.unique(r)),
                "min r": r.min(),
                "median r": np.median(r),
                "max r": r.max(),
                "total": r.sum(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(results / "variants.csv", index=False)
    return df


def analyse(results):
    df = pd.read_csv(results / "viterbi.csv", keep_default_na=False)
    lines = ["# Coarsening evaluation: summary", ""]
    lines += ["## Recombination variants", ""]
    lines += [describe_variants(results).to_markdown(index=False, floatfmt=".3g"), ""]
    lines += ["## Viterbi", ""]
    for mu, group in df.groupby("mu"):
        g = group.groupby(["strategy", "parameter", "variant"], sort=False)
        summary = pd.DataFrame(
            {
                "queries": g.size(),
                "time (s)": g.time.mean(),
                "time ratio": g.time_ratio.mean(),
                "traceback": g.traceback_size.mean(),
                "traceback ratio": g.traceback_ratio.mean(),
                "max values": g.max_values.max(),
                "switches": g.num_switches.mean(),
                "mismatches": g.num_mismatches.mean(),
                "optimal": g.loss.apply(lambda x: np.mean(np.abs(x) < 1e-6)),
                "mean loss": g.loss.mean(),
                "max loss": g.loss.max(),
                "breakpoints kept": g.breakpoints_kept.mean(),
            }
        ).reset_index()
        lines += [f"### mu = {mu:g}", ""]
        lines += [summary.to_markdown(index=False, floatfmt=".3g"), ""]
    (results / "summary.md").write_text("\n".join(lines))
    logger.info(f"Wrote {results / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-processes", "-t", type=int, default=1)
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.analyse_only:
        run(args.num_samples, args.num_processes, args.results)
    analyse(args.results)


if __name__ == "__main__":
    main()
