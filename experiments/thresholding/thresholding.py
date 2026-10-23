"""
Experiments on the effectiveness of thresholding and pruning in the tsls
engine.

After normalisation at each site, each method reduces the number of stored
likelihood values in a different way:

- floor: values below t are set to t (the likelihood_threshold option);
- relative: nodes with values below f times the maximum are pruned;
- top_n: all but the n nodes with the largest values are pruned;
- lossless_floor: a floor at t* = (rho / n) / (1 - rho + rho / n). In Viterbi,
  any node below t* is beaten by recombination at the next site, so this
  floor does not change the Viterbi likelihoods.

Pruned nodes get the value likelihood_threshold (by default the smallest
positive double), so they can then only be reached by recombination.

Scenarios:

- threading: match a held-out sample from an msprime simulation, as in
  verification.py (HeldOutThreading), at several sample sizes;
- calibration: queries simulated from the HMM, as in verification.py
  (ModelCalibration), to see whether pruning biases the posteriors;
- sc2ts: queries from benchmarking/ on sub-ARGs of the sc2ts ARG.

Run with --quick for a small version. Results and plots are written to
results/.
"""

import argparse
import concurrent.futures
import functools
import logging
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verification"))
import tsls
import verification as v

logger = logging.getLogger(__name__)

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SC2TS_DATA = ROOT / "benchmarking" / "data"

FLOOR_GRID = [1e-30, 1e-20, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
TOP_N_GRID = [10, 30, 100, 300, 1000, 3000]
QUICK_FLOOR_GRID = [1e-20, 1e-8, 1e-4, 1e-2]
QUICK_TOP_N_GRID = [10, 100, 1000]
METHODS = ["floor", "relative", "top_n"]
CALIBRATION_BINS = np.linspace(0, 1, 21)


def method_grid(quick, rho, n, viterbi=True):
    """
    Return the list of (method, parameter, matcher kwargs) tuples to run.
    """
    floors = QUICK_FLOOR_GRID if quick else FLOOR_GRID
    top_ns = QUICK_TOP_N_GRID if quick else TOP_N_GRID
    grid = [("exact", np.nan, {})]
    grid += [("floor", t, {"likelihood_threshold": t}) for t in floors]
    grid += [("relative", f, {"pruning": ("relative", f)}) for f in floors]
    grid += [("top_n", k, {"pruning": ("top_n", k)}) for k in top_ns]
    if viterbi:
        t_star = (rho / n) / (1 - rho + rho / n)
        # Just below t*, so that pruned nodes lose the tie with recombination
        grid.append(("lossless_floor", t_star, {"likelihood_threshold": t_star * 0.999}))
    return grid


def path_log_likelihood(h, num_segments, matched, mu, rho):
    """
    Return the log likelihood of a Viterbi path with weight_by_n=False (so
    staying has probability 1 and each switch rho), from the engine's output.
    """
    non_missing = h != -1
    num_mismatches = np.sum(h[non_missing] != matched[non_missing])
    return (
        (num_segments - 1) * np.log(rho)
        + num_mismatches * np.log(mu)
        + (len(h) - num_mismatches) * np.log(1 - (v.NUM_ALLELES - 1) * mu)
    )


def run_viterbi(am, ts, h):
    """
    Return the (left, right, parent, path, matched, seconds) tuple, with the
    segments in left-to-right order and path the node copied at each site.
    """
    m = ts.num_sites
    matched = np.zeros(m, dtype=np.int8)
    before = time.perf_counter()
    left, right, parent = am.find_path(h, 0, m, matched)
    seconds = time.perf_counter() - before
    order = np.argsort(left)
    left, right, parent = left[order], right[order], parent[order]
    segment = np.searchsorted(left, ts.sites_position, side="right") - 1
    return left, right, parent, parent[segment], matched, seconds


def mean_distinct_values(cm):
    counts = [
        len(np.unique(cm.values[cm.offsets[j] : cm.offsets[j + 1]]))
        for j in range(cm.num_sites)
    ]
    return float(np.mean(counts))


def run_forward_backward(am, h):
    """
    Return the (fm, bm, seconds) tuple.
    """
    before = time.perf_counter()
    fm = am.forward_matrix(h)
    bm = am.backward_matrix(h, fm.normalisation_factor)
    return fm, bm, time.perf_counter() - before


def normalise_rows(P):
    return P / np.sum(P, axis=1)[:, None]


def total_variation(P, Q):
    """
    Mean total variation distance between the rows of P and Q.
    """
    return 0.5 * float(np.mean(np.sum(np.abs(P - Q), axis=1)))


# ---------------------------------------------------------------------------
# Scenario 1: threading a held-out sample
# ---------------------------------------------------------------------------


def threading_case(num_samples, seed):
    """
    Return (ts, h, true_parent, near) for a held-out sample, as in
    verification.threading_replicate: ts is the ARG of the other samples
    with the same node IDs, true_parent[j] is the held-out sample's parent at
    site j and near[j] the nodes within 1 edge of it in ts.
    """
    params = dict(v.THREADING_DEFAULTS)
    full_ts = v.simulate_arg(
        num_samples + 1,
        params["sequence_length"],
        params["recombination_rate"],
        params["mutation_rate"],
        seed,
    )
    samples = full_ts.samples()
    query = samples[-1]
    ts = full_ts.simplify(
        samples[:-1],
        keep_unary=True,
        filter_nodes=False,
        filter_sites=False,
        update_sample_flags=False,
    )
    h = full_ts.genotype_matrix(
        samples=[query], isolated_as_missing=False, alleles=v.ALLELES
    )[:, 0].astype(np.int8)
    true_parent = np.zeros(ts.num_sites, dtype=np.int32)
    for tree in full_ts.trees():
        for site in tree.sites():
            true_parent[site.id] = tree.parent(query)
    near = [None] * ts.num_sites
    for tree in ts.trees():
        for site in tree.sites():
            p = true_parent[site.id]
            nodes = [p, *tree.children(p)]
            if tree.parent(p) != -1:
                nodes.append(tree.parent(p))
            near[site.id] = nodes
    return ts, h, true_parent, near


def mean_breakpoint_distance(true_switch, inferred_switch):
    if len(true_switch) == 0 or len(inferred_switch) == 0:
        return np.nan
    return float(np.mean(v.nearest_distances(true_switch, inferred_switch)))


def threading_distances(ts, path, true_parent):
    distance = np.zeros(ts.num_sites, dtype=int)
    for tree in ts.trees():
        parent = tree.parent_array
        for site in tree.sites():
            distance[site.id] = v.tree_distance(
                parent, path[site.id], true_parent[site.id]
            )
    return distance


def run_threading_task(num_samples, seed, quick):
    ts, h, true_parent, near = threading_case(num_samples, seed)
    m = ts.num_sites
    mu, rho = v.hmm_params(v.THREADING_DEFAULTS["k"])
    mi = tsls.TreeSequence(ts)
    true_switch = np.where(true_parent[1:] != true_parent[:-1])[0] + 1
    rows = []
    exact_posterior = None
    exact = None
    for method, parameter, kwargs in method_grid(quick, rho, 1):
        am = v.make_matcher(
            ts,
            mu,
            rho,
            weight_by_n=False,
            count_distinct=True,
            tree_sequence=mi,
            **kwargs,
        )
        left, right, parent, path, matched, seconds = run_viterbi(am, ts, h)
        distance = threading_distances(ts, path, true_parent)
        inferred_switch = np.where(path[1:] != path[:-1])[0] + 1
        row = {
            "num_samples": num_samples,
            "case": seed,
            "method": method,
            "parameter": parameter,
            "num_sites": m,
            "num_nodes": ts.num_nodes,
            "viterbi_time": seconds,
            "traceback_size": am.mean_traceback_size,
            "distinct_values": am.mean_num_distinct_values,
            "total_memory": am.total_memory,
            "log_likelihood": path_log_likelihood(h, len(parent), matched, mu, rho),
            "exact_parent": np.mean(path == true_parent),
            "within_2": np.mean((distance >= 0) & (distance <= 2)),
            "breakpoint_distance": mean_breakpoint_distance(
                true_switch, inferred_switch
            ),
        }
        if exact is None:
            exact = (left, right, parent)
        row["identical_path"] = all(
            np.array_equal(x, y) for x, y in zip(exact, (left, right, parent))
        )
        if method != "lossless_floor":
            fm, bm, seconds = run_forward_backward(am, h)
            P = fm.decode() * bm.decode()
            mass = np.sum(P, axis=1)
            P = normalise_rows(P)
            if exact_posterior is None:
                exact_posterior = (P, fm.log_likelihood)
            row.update(
                {
                    "fb_time": seconds,
                    "forward_values": len(fm.values) / m,
                    "forward_distinct_values": mean_distinct_values(fm),
                    "forward_log_likelihood": fm.log_likelihood,
                    "posterior_mass_error": float(np.max(np.abs(mass - 1))),
                    "posterior_tv": total_variation(P, exact_posterior[0]),
                    "posterior_true": float(np.mean(P[np.arange(m), true_parent])),
                    "posterior_near": float(
                        np.mean([np.sum(P[j, near[j]]) for j in range(m)])
                    ),
                }
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scenario 2: calibration under the model
# ---------------------------------------------------------------------------

CALIBRATION_SETTINGS = {"default": (5, 0.01), "high_recombination": (25, 0.01)}


def run_calibration_task(setting, seed, quick):
    """
    Simulate an ARG and several paths from the model's prior, and return
    per-method sums for the reliability diagram and 90% credible sets.
    """
    switches, mu = CALIBRATION_SETTINGS[setting]
    rng = np.random.default_rng(seed)
    ts = v.simulate_arg(50, 5e5, 1e-8, 1e-8, seed)
    A, present, continuous = v.node_states(ts)
    m = ts.num_sites
    n = ts.num_nodes
    rho = switches / m
    mi = tsls.TreeSequence(ts)
    queries = []
    for _ in range(3 if quick else 5):
        path = v.sample_prior_path(present, continuous, np.full(m, rho), n, rng)
        queries.append((path, v.mutate_along_path(A, path, mu, rng)))
    rows = []
    for method, parameter, kwargs in method_grid(quick, rho, n, viterbi=False):
        am = v.make_matcher(ts, mu, rho, weight_by_n=True, tree_sequence=mi, **kwargs)
        num_bins = len(CALIBRATION_BINS) - 1
        count = np.zeros(num_bins)
        total_p = np.zeros(num_bins)
        num_true = np.zeros(num_bins)
        covered = 0
        set_mass = 0
        values = 0
        for path, h in queries:
            fm, bm, _ = run_forward_backward(am, h)
            values += len(fm.values) / m
            P = normalise_rows(fm.decode() * bm.decode())
            is_true = np.zeros_like(P, dtype=bool)
            is_true[np.arange(m), path] = True
            keep = P > 1e-4
            index = np.clip(np.digitize(P[keep], CALIBRATION_BINS) - 1, 0, num_bins - 1)
            count += np.bincount(index, minlength=num_bins)
            total_p += np.bincount(index, weights=P[keep], minlength=num_bins)
            num_true += np.bincount(index, weights=is_true[keep], minlength=num_bins)
            cumulative = np.cumsum(-np.sort(-P, axis=1), axis=1)
            p_true = P[np.arange(m), path][:, None]
            tied = np.abs(P - p_true) <= 1e-9 * p_true
            num_greater = np.sum((P > p_true) & ~tied, axis=1)
            size = np.sum(cumulative < 0.9, axis=1) + 1
            covered += np.sum(np.clip((size - num_greater) / np.sum(tied, axis=1), 0, 1))
            set_mass += np.sum(cumulative[np.arange(m), size - 1])
        row = {
            "setting": setting,
            "case": seed,
            "method": method,
            "parameter": parameter,
            "num_site_queries": m * len(queries),
            "covered_90": covered,
            "set_mass_90": set_mass,
            "forward_values": values / len(queries),
        }
        for j in range(num_bins):
            row[f"count_{j}"] = count[j]
            row[f"total_p_{j}"] = total_p[j]
            row[f"num_true_{j}"] = num_true[j]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scenario 3: sc2ts sub-ARGs
# ---------------------------------------------------------------------------

SC2TS_SIZES = {10000: 30, 30000: 30, 100000: 10}
SC2TS_SAMPLED_SITES = 100


@functools.lru_cache(maxsize=1)
def load_sc2ts(size):
    """
    Return the (ts, tree_sequence, H, kind, k) tuple for the sc2ts sub-ARG
    of the specified size, with the vestigial 0 -> 1 edge removed so that
    node 0 is isolated (as in benchmarking/engines.py).
    """
    import tskit

    ts = tskit.load(SC2TS_DATA / "subargs" / f"sc2ts_bench_n{size}.trees")
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    ts = tables.tree_sequence()
    mi = tsls.TreeSequence(ts)
    query_dir = SC2TS_DATA / "queries_scaling"
    H = np.load(query_dir / "queries.npz")["H"]
    df = pd.read_csv(query_dir / "queries.csv")
    return ts, mi, H, df.kind.values, df.k.values


def decode_at_sites(cm, sites):
    """
    Return the dense (len(sites), num_nodes) array of values at the
    specified sorted sites, decoding only those sites.
    """
    ts = cm.ts
    N = ts.num_nodes
    D = np.zeros((len(sites), N))
    position = ts.sites_position[sites]
    for tree in ts.trees():
        start, stop = np.searchsorted(position, tree.interval)
        if start == stop:
            continue
        parent = tree.parent_array
        in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
        for k in range(start, stop):
            site_nodes, site_values = cm.get_site(sites[k])
            value = np.full(N + 1, np.nan)
            value[site_nodes] = site_values
            ancestor = parent.copy()
            unset = np.where(in_tree & np.isnan(value[:N]))[0]
            while len(unset) > 0:
                value[unset] = value[ancestor[unset]]
                ancestor[unset] = parent[ancestor[unset]]
                unset = unset[np.isnan(value[unset])]
            D[k] = np.where(in_tree, value[:N], 0)
    return D


def sc2ts_query_ids(size, quick):
    num_queries = 3 if quick else SC2TS_SIZES[size]
    return np.unique(np.linspace(0, 109, num_queries).astype(int))


def run_sc2ts_task(size, query_id, quick):
    ts, mi, H, kind, k = load_sc2ts(size)
    h = H[query_id]
    m = ts.num_sites
    mu, rho = v.hmm_params(3)
    with_forward = size == min(SC2TS_SIZES)
    sites = np.unique(np.linspace(0, m - 1, SC2TS_SAMPLED_SITES).astype(int))
    rows = []
    exact = None
    exact_posterior = None
    for method, parameter, kwargs in method_grid(quick, rho, 1):
        am = v.make_matcher(
            ts,
            mu,
            rho,
            weight_by_n=False,
            count_distinct=True,
            tree_sequence=mi,
            **kwargs,
        )
        left, right, parent, _, matched, seconds = run_viterbi(am, ts, h)
        if exact is None:
            exact = (left, right, parent)
        row = {
            "size": size,
            "case": query_id,
            "kind": kind[query_id],
            "k": k[query_id],
            "method": method,
            "parameter": parameter,
            "viterbi_time": seconds,
            "traceback_size": am.mean_traceback_size,
            "distinct_values": am.mean_num_distinct_values,
            "total_memory": am.total_memory,
            "log_likelihood": path_log_likelihood(h, len(parent), matched, mu, rho),
            "num_mismatches": int(np.sum(matched != h)),
            "identical_path": all(
                np.array_equal(x, y) for x, y in zip(exact, (left, right, parent))
            ),
        }
        if with_forward and method != "lossless_floor":
            fm, bm, seconds = run_forward_backward(am, h)
            P = normalise_rows(decode_at_sites(fm, sites) * decode_at_sites(bm, sites))
            if exact_posterior is None:
                exact_posterior = P
            row.update(
                {
                    "fb_time": seconds,
                    "forward_values": len(fm.values) / m,
                    "forward_log_likelihood": fm.log_likelihood,
                    "posterior_tv": total_variation(P, exact_posterior),
                }
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Running and analysis
# ---------------------------------------------------------------------------


def make_tasks(scenarios, quick):
    tasks = []
    if "threading" in scenarios:
        sizes = [50] if quick else [50, 200, 1000]
        replicates = 3 if quick else 10
        for n in sizes:
            for seed in range(1, replicates + 1):
                tasks.append(("threading", run_threading_task, (n, seed, quick)))
    if "calibration" in scenarios:
        for setting in CALIBRATION_SETTINGS:
            for seed in range(1, (3 if quick else 10) + 1):
                tasks.append(
                    ("calibration", run_calibration_task, (setting, seed, quick))
                )
    if "sc2ts" in scenarios:
        if not (SC2TS_DATA / "subargs").exists():
            logger.warning(f"No sc2ts data in {SC2TS_DATA}: skipping sc2ts scenario")
        else:
            sizes = [10000] if quick else list(SC2TS_SIZES)
            for size in sizes:
                for query_id in sc2ts_query_ids(size, quick):
                    tasks.append(("sc2ts", run_sc2ts_task, (size, int(query_id), quick)))
    return tasks


def run_task(task):
    scenario, func, args = task
    return scenario, func(*args)


def run(scenarios, quick, num_processes):
    tasks = make_tasks(scenarios, quick)
    results = {scenario: [] for scenario in scenarios}
    progress = tqdm.tqdm(total=len(tasks))
    if num_processes <= 1:
        for task in tasks:
            scenario, rows = run_task(task)
            results[scenario].extend(rows)
            progress.update()
    else:
        with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                scenario, rows = future.result()
                results[scenario].extend(rows)
                progress.update()
    progress.close()
    RESULTS.mkdir(exist_ok=True)
    for scenario, rows in results.items():
        if len(rows) > 0:
            pd.DataFrame(rows).to_csv(RESULTS / f"{scenario}.csv", index=False)


def add_relative_columns(df, group, columns):
    """
    Add, for each column, a column of its value relative to the exact method
    in the same group and case (ratio_<col>), and the difference
    (diff_<col> = method - exact).
    """
    exact = df[df.method == "exact"].set_index([group, "case"])
    for column in columns:
        if column not in df:
            continue
        base = exact[column].reindex(pd.MultiIndex.from_frame(df[[group, "case"]]))
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"ratio_{column}"] = df[column].values / base.values
        df[f"diff_{column}"] = df[column].values - base.values
    return df


def summarise(df, group, x, losses):
    """
    Return the mean of x and the losses for each (group, method, parameter).
    """
    columns = [x, *losses]
    return (
        df.groupby([group, "method", "parameter"], dropna=False)[columns]
        .mean()
        .reset_index()
    )


def plot_frontiers(summary, group, x, losses, xlabel, name):
    """
    For each value of group, plot each loss against x (relative storage),
    with one curve per method across its parameters.
    """
    groups = sorted(summary[group].unique())
    fig, axes = plt.subplots(
        len(losses),
        len(groups),
        figsize=(4.5 * len(groups), 3.5 * len(losses)),
        squeeze=False,
    )
    for col, g in enumerate(groups):
        sub = summary[summary[group] == g]
        for row, (loss, label) in enumerate(losses):
            ax = axes[row, col]
            for method in METHODS:
                d = sub[sub.method == method].sort_values(x)
                ax.plot(d[x], d[loss], marker="o", label=method)
                for _, r in d.iterrows():
                    ax.annotate(
                        f"{r.parameter:g}", (r[x], r[loss]), fontsize=6, alpha=0.6
                    )
            lossless = sub[sub.method == "lossless_floor"]
            if len(lossless) > 0 and not np.all(np.isnan(lossless[loss])):
                ax.plot(
                    lossless[x],
                    lossless[loss],
                    marker="*",
                    markersize=12,
                    linestyle="none",
                    label="lossless_floor",
                )
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.set_title(f"{group}={g}")
            ax.axvline(1, color="grey", linestyle=":")
        axes[0, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=100)
    plt.close(fig)


def best_within_budget(summary, group, x, budgets):
    """
    For each group, method and budget, the smallest storage ratio among the
    parameters that meet the budget.
    """
    rows = []
    for (g, method), sub in summary.groupby([group, "method"]):
        if method == "exact":
            continue
        row = {group: g, "method": method}
        for name, condition in budgets.items():
            ok = sub[condition(sub)]
            if len(ok) == 0:
                row[name] = "-"
            else:
                best = ok.loc[ok[x].idxmin()]
                row[name] = f"{best[x]:.3f} ({best.parameter:g})"
        rows.append(row)
    return pd.DataFrame(rows)


def analyse_threading(lines):
    df = pd.read_csv(RESULTS / "threading.csv")
    df = add_relative_columns(
        df,
        "num_samples",
        [
            "traceback_size",
            "viterbi_time",
            "log_likelihood",
            "exact_parent",
            "within_2",
            "forward_values",
            "fb_time",
            "posterior_true",
            "forward_log_likelihood",
        ],
    )
    df["log_likelihood_loss"] = -df.diff_log_likelihood
    df["accuracy_drop"] = -df.diff_exact_parent
    df["within_2_drop"] = -df.diff_within_2
    df["posterior_true_drop"] = -df.diff_posterior_true
    df["identical_path"] = df.identical_path.astype(float)
    viterbi = summarise(
        df,
        "num_samples",
        "ratio_traceback_size",
        [
            "log_likelihood_loss",
            "accuracy_drop",
            "within_2_drop",
            "identical_path",
            "ratio_viterbi_time",
            "breakpoint_distance",
            "traceback_size",
            "distinct_values",
        ],
    )
    viterbi.to_csv(RESULTS / "threading_viterbi_summary.csv", index=False)
    plot_frontiers(
        viterbi,
        "num_samples",
        "ratio_traceback_size",
        [
            ("log_likelihood_loss", "Viterbi path log-likelihood loss"),
            ("accuracy_drop", "Drop in fraction of sites copying parent"),
            ("ratio_viterbi_time", "Viterbi time relative to exact"),
        ],
        "Mean traceback size relative to exact",
        "threading_viterbi",
    )
    fb = summarise(
        df[df.method != "lossless_floor"],
        "num_samples",
        "ratio_forward_values",
        ["posterior_tv", "posterior_true_drop", "posterior_mass_error", "ratio_fb_time"],
    )
    fb.to_csv(RESULTS / "threading_fb_summary.csv", index=False)
    plot_frontiers(
        fb,
        "num_samples",
        "ratio_forward_values",
        [
            ("posterior_tv", "Posterior total variation from exact"),
            ("posterior_true_drop", "Drop in posterior on the true parent"),
            ("ratio_fb_time", "Forward-backward time relative to exact"),
        ],
        "Mean forward values per site relative to exact",
        "threading_forward_backward",
    )
    budgets = {
        "no loss": lambda s: (s.log_likelihood_loss < 1e-9) & (s.identical_path == 1),
        "loss < 0.1": lambda s: s.log_likelihood_loss < 0.1,
        "accuracy drop < 0.01": lambda s: s.accuracy_drop < 0.01,
    }
    lines += [
        "## Threading a held-out sample: Viterbi",
        "",
        (
            "Smallest mean traceback size relative to exact (parameter) within "
            "each loss budget."
        ),
        "",
        best_within_budget(
            viterbi, "num_samples", "ratio_traceback_size", budgets
        ).to_markdown(index=False),
        "",
        "Exact traceback size and distinct values per site:",
        "",
        viterbi[viterbi.method == "exact"][
            ["num_samples", "traceback_size", "distinct_values"]
        ].to_markdown(index=False),
        "",
    ]
    budgets = {
        "TV < 0.001": lambda s: s.posterior_tv < 0.001,
        "TV < 0.01": lambda s: s.posterior_tv < 0.01,
        "TV < 0.05": lambda s: s.posterior_tv < 0.05,
    }
    lines += [
        "## Threading a held-out sample: forward-backward",
        "",
        (
            "Smallest mean number of forward values per site relative to exact "
            "(parameter) within each budget on the posterior total variation "
            "distance from exact."
        ),
        "",
        best_within_budget(
            fb, "num_samples", "ratio_forward_values", budgets
        ).to_markdown(index=False),
        "",
    ]
    exact = df[df.method == "exact"]
    lines += [
        "Exact forward values and distinct values per site:",
        "",
        exact.groupby("num_samples")[["forward_values", "forward_distinct_values"]]
        .mean()
        .reset_index()
        .to_markdown(index=False),
        "",
    ]


def analyse_calibration(lines):
    df = pd.read_csv(RESULTS / "calibration.csv")
    num_bins = len(CALIBRATION_BINS) - 1
    rows = []
    for (setting, method, parameter), sub in df.groupby(
        ["setting", "method", "parameter"], dropna=False
    ):
        count = np.array([sub[f"count_{j}"].sum() for j in range(num_bins)])
        total_p = np.array([sub[f"total_p_{j}"].sum() for j in range(num_bins)])
        num_true = np.array([sub[f"num_true_{j}"].sum() for j in range(num_bins)])
        nonempty = count > 0
        error = np.abs(num_true[nonempty] - total_p[nonempty]) / count[nonempty]
        sites = sub.num_site_queries.sum()
        rows.append(
            {
                "setting": setting,
                "method": method,
                "parameter": parameter,
                "reliability_error": np.sum(error * count[nonempty])
                / np.sum(count[nonempty]),
                "coverage_90": sub.covered_90.sum() / sites,
                "expected_coverage_90": sub.set_mass_90.sum() / sites,
                "forward_values": sub.forward_values.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["coverage_excess"] = summary.coverage_90 - summary.expected_coverage_90
    exact = summary[summary.method == "exact"].set_index("setting")
    summary["ratio_forward_values"] = (
        summary.forward_values / exact.forward_values.reindex(summary.setting).values
    )
    summary.to_csv(RESULTS / "calibration_summary.csv", index=False)
    plot_frontiers(
        summary,
        "setting",
        "ratio_forward_values",
        [
            ("reliability_error", "Mean |observed - predicted| (reliability)"),
            ("coverage_excess", "90% set coverage - expected"),
        ],
        "Mean forward values per site relative to exact",
        "calibration",
    )
    lines += [
        "## Calibration under the model",
        "",
        (
            "Mean absolute calibration error (weighted over reliability bins) and "
            "90% credible-set coverage minus expected coverage. Exact values give "
            "the noise level."
        ),
        "",
        summary[
            [
                "setting",
                "method",
                "parameter",
                "ratio_forward_values",
                "reliability_error",
                "coverage_excess",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
    ]


def analyse_sc2ts(lines):
    df = pd.read_csv(RESULTS / "sc2ts.csv")
    df = add_relative_columns(
        df,
        "size",
        [
            "traceback_size",
            "viterbi_time",
            "total_memory",
            "log_likelihood",
            "forward_values",
            "fb_time",
            "forward_log_likelihood",
        ],
    )
    df["log_likelihood_loss"] = -df.diff_log_likelihood
    df["identical_path"] = df.identical_path.astype(float)
    exact_copy = (df.kind == "copy") & (df.k == 0)
    df["exact_copy_failed"] = np.where(
        exact_copy, (df.num_mismatches > 0).astype(float), np.nan
    )
    viterbi = summarise(
        df,
        "size",
        "ratio_traceback_size",
        [
            "log_likelihood_loss",
            "identical_path",
            "ratio_viterbi_time",
            "ratio_total_memory",
            "exact_copy_failed",
            "traceback_size",
            "distinct_values",
            "viterbi_time",
        ],
    )
    viterbi.to_csv(RESULTS / "sc2ts_viterbi_summary.csv", index=False)
    plot_frontiers(
        viterbi,
        "size",
        "ratio_traceback_size",
        [
            ("log_likelihood_loss", "Viterbi path log-likelihood loss"),
            ("identical_path", "Fraction of paths identical to exact"),
            ("ratio_viterbi_time", "Viterbi time relative to exact"),
        ],
        "Mean traceback size relative to exact",
        "sc2ts_viterbi",
    )
    budgets = {
        "no loss": lambda s: (s.log_likelihood_loss < 1e-9) & (s.identical_path == 1),
        "loss < 0.1": lambda s: s.log_likelihood_loss < 0.1,
        "loss < 1": lambda s: s.log_likelihood_loss < 1,
    }
    lines += [
        "## sc2ts sub-ARGs: Viterbi",
        "",
        (
            "Smallest mean traceback size relative to exact (parameter) within "
            "each loss budget."
        ),
        "",
        best_within_budget(viterbi, "size", "ratio_traceback_size", budgets).to_markdown(
            index=False
        ),
        "",
        "Exact traceback size, distinct values per site and time:",
        "",
        viterbi[viterbi.method == "exact"][
            ["size", "traceback_size", "distinct_values", "viterbi_time"]
        ].to_markdown(index=False),
        "",
        "Time relative to exact for each method's settings:",
        "",
        viterbi[viterbi.method != "exact"][
            [
                "size",
                "method",
                "parameter",
                "ratio_traceback_size",
                "ratio_viterbi_time",
                "log_likelihood_loss",
                "identical_path",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    if "posterior_tv" in df:
        fb = summarise(
            df[df.method != "lossless_floor"].dropna(subset=["posterior_tv"]),
            "size",
            "ratio_forward_values",
            ["posterior_tv", "ratio_fb_time"],
        )
        fb.to_csv(RESULTS / "sc2ts_fb_summary.csv", index=False)
        lines += [
            "## sc2ts sub-ARGs: forward-backward (10k)",
            "",
            fb.to_markdown(index=False, floatfmt=".4g"),
            "",
        ]


def analyse(scenarios):
    lines = ["# Thresholding experiments: summary", ""]
    for scenario, func in [
        ("threading", analyse_threading),
        ("calibration", analyse_calibration),
        ("sc2ts", analyse_sc2ts),
    ]:
        if scenario in scenarios and (RESULTS / f"{scenario}.csv").exists():
            func(lines)
    (RESULTS / "summary.md").write_text("\n".join(lines))
    logger.info(f"Wrote {RESULTS / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenario",
        "-s",
        action="append",
        choices=["threading", "calibration", "sc2ts"],
        help="Scenarios to run (default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small version")
    parser.add_argument("--num-processes", "-t", type=int, default=1)
    parser.add_argument(
        "--analyse-only", action="store_true", help="Only analyse existing results"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    scenarios = args.scenario
    if scenarios is None:
        scenarios = ["threading", "calibration", "sc2ts"]
    if not args.analyse_only:
        run(scenarios, args.quick, args.num_processes)
    analyse(scenarios)


if __name__ == "__main__":
    main()
