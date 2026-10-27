"""
Evaluate the thresholding strategies of the tsls engine on real workloads,
under the unified likelihood model (docs/likelihood_model.md).

For each query we run every strategy and compare it with the exact run:

- Viterbi over the query's whole window, with the loss in path log odds,
  whether the path is certified optimal, the nodes tracked and updated per
  site, and the time;
- forward and backward over a window of WINDOW sites in the middle of the
  query (the whole sequence needs too much memory on the tsinfer data),
  with the error in the log likelihood and in the posteriors at
  NUM_POSTERIOR_SITES sites, the values stored, and the time.

Workloads (from benchmarking/data):

- sc2ts: queries_scaling on the 10k and 100k sub-ARGs, with the sc2ts model
  (one recombination costs about 3 mismatches);
- tgp_ancestors: ancestor matching for chr20p groups 84, 1000 and 1485,
  with tsinfer's rho = 0.01 and mu = 1e-20;
- tgp_samples: sample matching for chr20p, with rho = 0.01 and mu such that
  k mismatches cost the same as one switch, for k = 1, 2, 4, and with
  tsinfer's mu = 1e-20.

Each task (one query) runs in a worker process with its address space
limited to MEMORY_LIMIT, and results go to results/viterbi.csv and
results/forward_backward.csv. Then summary.md is written.

    uv run python evaluate.py -t 2           # about an hour
    uv run python evaluate.py --quick -t 2   # a few minutes
    uv run python evaluate.py --analyse-only
"""

import argparse
import concurrent.futures
import functools
import logging
import os
import pathlib
import resource
import time

# Threaded BLAS reserves address space, which counts against MEMORY_LIMIT
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
RESULTS = HERE / "results"

WINDOW = 5000
NUM_POSTERIOR_SITES = 50
MEMORY_LIMIT = 5 * 1024**3
MISSING = -1
SC2TS_ALLELES = ("A", "C", "G", "T")
TGP_RHO = 0.01
SAMPLE_MISMATCH_K = [1, 2, 4]

FLOORS = [1e-8, 1e-4, 1e-2]
PRUNE_K = [0, 1, 2, 3]
ESCALATION_K = [0, 1, 2, 4]


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def load_reference(path):
    """
    Return the tree sequence at path with the vestigial root edges from
    node 0 removed, as in benchmarking/engines.py.
    """
    ts = tskit.load(path)
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    return tables.tree_sequence()


def load_queries(query_dir):
    """
    Return the (get, num_queries) tuple for the query set, where get(j) is
    the (h, start, end) tuple of query j.
    """
    data = np.load(query_dir / "queries.npz")
    num_sites = len(data["positions"])
    if "values" in data:
        start = data["start"].astype(int)
        end = data["end"].astype(int)
        offsets = np.concatenate([[0], np.cumsum(end - start)])
        values = data["values"]

        def get(j):
            h = np.full(num_sites, MISSING, dtype=np.int8)
            h[start[j] : end[j]] = values[offsets[j] : offsets[j + 1]]
            return h, int(start[j]), int(end[j])

        return get, len(start)
    H = data["H"]
    return (lambda j: (H[j].astype(np.int8), 0, num_sites)), len(H)


def site_num_alleles(ts, alleles):
    """
    The number of alleles at each site as tsls counts them.
    """
    if alleles is not None:
        return np.full(ts.num_sites, len(alleles))
    num = np.zeros(ts.num_sites, dtype=int)
    for site in ts.sites():
        states = {site.ancestral_state} | {m.derived_state for m in site.mutations}
        num[site.id] = max(2, len(states))
    return num


def sc2ts_mismatch_and_odds(k=3, num_alleles=4):
    """
    sc2ts's (mu, q): one recombination costs about k mismatches.
    """
    mu = 0.0125
    q = mu**k / (mu**k + (1 - (num_alleles - 1) * mu) ** k) * 1.1
    return mu, q


def sample_mismatch(k, n):
    """
    The biallelic mu for which k mismatches cost the same as one switch.
    """
    e = float(tsls.model.switch_odds(TGP_RHO, n)) ** (1 / k)
    return e / (1 + e)


@functools.lru_cache(maxsize=1)
def load_workload(name, reference):
    """
    Return the (ts, tree_sequence, alleles, num_alleles, get, num_queries)
    tuple for the workload, cached for the tasks in a worker.
    """
    if name == "sc2ts":
        ts = load_reference(DATA / "subargs" / f"sc2ts_bench_n{reference}.trees")
        alleles = SC2TS_ALLELES
        get, num_queries = load_queries(DATA / "queries_scaling")
    else:
        directory = DATA / "tgp_chr20p" / reference
        ts = load_reference(directory / "reference.trees")
        alleles = None
        get, num_queries = load_queries(directory)
    return (
        ts,
        tsls.TreeSequence(ts, alleles=alleles),
        alleles,
        site_num_alleles(ts, alleles),
        get,
        num_queries,
    )


def model_rates(name, ts, mu_parameter):
    """
    Return the per-site (r, mu) arrays for the workload.
    """
    m = ts.num_sites
    if name == "sc2ts":
        mu, q = sc2ts_mismatch_and_odds()
        r = tsls.model.recombination_probability(q, ts.num_nodes)
        return np.full(m, r), np.full(m, mu)
    return np.full(m, TGP_RHO), np.full(m, mu_parameter)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def viterbi_strategies(r, mu, num_alleles, n, quick):
    """
    Return the list of (strategy, parameter, Hmm kwargs, certificate
    threshold) tuples for Viterbi. A path with odds of at least the
    certificate threshold is claimed optimal (NaN: no claim).
    """
    floors = FLOORS[1:] if quick else FLOORS
    out = [("exact", np.nan, {}, np.nan)]
    for t in floors:
        out.append(("floor", t, {"likelihood_threshold": t}, t))
    t = tsls.model.lossless_threshold(r, n)
    out.append(("lossless_prune", t, {"prune_threshold": t}, np.nan))
    for k in PRUNE_K[:2] if quick else PRUNE_K:
        t = tsls.model.path_threshold(k, mu, num_alleles)
        out.append(("prune_k", k, {"prune_threshold": t}, t))
    for n_top in [10, 100]:
        out.append(("top_n", n_top, {"pruning": ("top_n", n_top)}, np.nan))
    return out


def fb_strategies(r, n, quick):
    """
    Return the list of (strategy, parameter, Hmm kwargs) tuples for forward
    and backward. relative_q drops nodes below c q times the maximum, where
    q is the smallest switch odds: such a node has less mass than
    recombination alone gives every node (as a fraction of the maximum).
    """
    floors = FLOORS[1:] if quick else FLOORS
    q = float(np.min(tsls.model.switch_odds(r, n)))
    out = [("exact", np.nan, {})]
    out += [("floor", t, {"likelihood_threshold": t}) for t in floors]
    out += [("relative", t, {"pruning": ("relative", t)}) for t in floors]
    out += [("relative_q", c, {"pruning": ("relative", c * q)}) for c in [0.01, 1]]
    out += [("top_n", k, {"pruning": ("top_n", k)}) for k in [10, 100, 1000]]
    return out


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------


def path_log_odds(h, matched, left, positions, start, end, log_q, log_e):
    """
    Return the (log odds, num_mismatches, num_switches) tuple of a path,
    relative to staying on a node that matches at every site.
    """
    sites = np.arange(start, end)
    mismatch = (h[sites] != MISSING) & (h[sites] != matched[sites])
    switch_sites = np.searchsorted(positions, np.sort(left)[1:])
    value = np.sum(log_e[sites][mismatch]) + np.sum(log_q[switch_sites])
    return float(value), int(np.sum(mismatch)), len(switch_sites)


def run_viterbi(tree_sequence, r, mu, kwargs, h, start, end):
    am = tsls.Hmm(tree_sequence, r, mu, **kwargs)
    m = len(r)
    matched = np.zeros(m, dtype=np.int8)
    before = time.perf_counter()
    left, _, parent = am.find_path(h, start, end, matched)
    seconds = time.perf_counter() - before
    stats = {
        "time": seconds,
        "mean_values": am.mean_num_values,
        "max_values": am.max_num_values,
        "mean_updated": am.mean_num_updated,
        "sparse_sites": am.num_sparse_sites / (end - start),
        "traceback_size": am.mean_traceback_size,
        "traceback_memory": am.total_memory,
    }
    return left, parent, matched, stats


def path_by_site(left, parent, positions, start, end):
    order = np.argsort(left)
    segment = np.searchsorted(left[order], positions[start:end], side="right") - 1
    return parent[order][segment]


def viterbi_rows(base, tree_sequence, ts, r, mu, num_alleles, h, start, end, quick):
    n = ts.num_nodes
    positions = ts.sites_position
    log_q = np.log(tsls.model.switch_odds(r, n))
    log_e = np.log(tsls.model.mismatch_odds(mu, num_alleles))
    rows = []
    exact = None
    for strategy, parameter, kwargs, t_cert in viterbi_strategies(
        r, mu, num_alleles, n, quick
    ):
        left, parent, matched, stats = run_viterbi(
            tree_sequence, r, mu, kwargs, h, start, end
        )
        odds, k, s = path_log_odds(h, matched, left, positions, start, end, log_q, log_e)
        path = path_by_site(left, parent, positions, start, end)
        if exact is None:
            exact = (odds, path, stats)
        loss = exact[0] - odds
        certified = bool(odds >= np.log(t_cert)) if not np.isnan(t_cert) else np.nan
        rows.append(
            {
                **base,
                "strategy": strategy,
                "parameter": parameter,
                "log_odds": odds,
                "num_mismatches": k,
                "num_switches": s,
                "loss": loss,
                "optimal": bool(abs(loss) <= 1e-9 * max(1, abs(exact[0]))),
                "same_path": bool(np.array_equal(path, exact[1])),
                "certified": certified,
                **stats,
                "time_ratio": stats["time"] / exact[2]["time"],
                "values_ratio": stats["mean_values"] / exact[2]["mean_values"],
            }
        )
    # Escalation: prune at k = 0, 1, ... until the path is certified, and
    # run exact if none is.
    total = 0
    level = "exact"
    for k in ESCALATION_K:
        t = tsls.model.path_threshold(k, mu, num_alleles)
        left, _, matched, stats = run_viterbi(
            tree_sequence, r, mu, {"prune_threshold": t}, h, start, end
        )
        total += stats["time"]
        odds, _, _ = path_log_odds(h, matched, left, positions, start, end, log_q, log_e)
        if odds >= np.log(t):
            level = k
            break
    else:
        left, _, matched, stats = run_viterbi(tree_sequence, r, mu, {}, h, start, end)
        total += stats["time"]
        odds, _, _ = path_log_odds(h, matched, left, positions, start, end, log_q, log_e)
    loss = exact[0] - odds
    rows.append(
        {
            **base,
            "strategy": "escalation",
            "parameter": level,
            "log_odds": odds,
            "loss": loss,
            "optimal": bool(abs(loss) <= 1e-9 * max(1, abs(exact[0]))),
            "certified": True,
            "time": total,
            "time_ratio": total / exact[2]["time"],
        }
    )
    return rows, exact[1]


# ---------------------------------------------------------------------------
# Forward and backward
# ---------------------------------------------------------------------------


def window_reference(ts, alleles, start, end):
    """
    Return the (window ts, tree_sequence, a, b) tuple for a window of up to
    WINDOW sites in the middle of [start, end). Node IDs are kept, and the
    window has sites a to b of the reference.
    """
    a = max(start, (start + end) // 2 - WINDOW // 2)
    b = min(end, a + WINDOW)
    position = ts.sites_position
    right = position[b] if b < ts.num_sites else ts.sequence_length
    wts = ts.keep_intervals([[position[a], right]], simplify=False)
    wts = wts.delete_sites(np.where(wts.sites_position < position[a])[0])
    assert wts.num_sites == b - a
    return wts, tsls.TreeSequence(wts, alleles=alleles), a, b


def decode_site(cm, tree, site):
    """
    Return the values of all nodes at the site (0 for nodes not in the
    tree), where tree is the (parent array, in_tree) tuple of its tree.
    """
    N = cm.ts.num_nodes
    parent, in_tree = tree
    nodes, values = cm.get_site(site)
    value = np.full(N + 1, np.nan)
    value[nodes] = values
    ancestor = parent.copy()
    unset = np.where(in_tree & np.isnan(value[:N]))[0]
    while len(unset) > 0:
        value[unset] = value[ancestor[unset]]
        ancestor[unset] = parent[ancestor[unset]]
        unset = unset[np.isnan(value[unset])]
    return np.where(in_tree, value[:N], 0)


def posterior_sites(wts):
    """
    Return the (sites, trees) of the sites at which posteriors are compared.
    """
    sites = np.unique(np.linspace(0, wts.num_sites - 1, NUM_POSTERIOR_SITES).astype(int))
    N = wts.num_nodes
    trees = []
    tree = wts.first()
    for j in sites:
        tree.seek(wts.sites_position[j])
        parent = tree.parent_array.copy()
        in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
        trees.append((parent, in_tree))
    return sites, trees


def posteriors(fm, bm, sites, trees):
    """
    Return the list of (unnormalised mass, normalised posterior) at the sites.
    If forward and backward have no mass on any common node (pruning can do
    this), the posterior is None.
    """
    out = []
    for site, tree in zip(sites, trees):
        p = decode_site(fm, tree, site) * decode_site(bm, tree, site)
        mass = np.sum(p)
        out.append((mass, p / mass if mass > 0 and np.isfinite(mass) else None))
    return out


def compare_posterior(p, q):
    """
    Return the (TV, max abs error, argmax agrees) of p against the exact q.
    A missing posterior counts as wholly wrong.
    """
    if p is None:
        return 1.0, float(np.max(q)), False
    return 0.5 * float(np.sum(np.abs(p - q))), float(np.max(np.abs(p - q))), bool(
        np.argmax(p) == np.argmax(q)
    )


def fb_rows(base, ts, alleles, r, mu, h, start, end, viterbi_path, quick):
    wts, wtree_sequence, a, b = window_reference(ts, alleles, start, end)
    hw = h[a:b]
    rw, muw = r[a:b], mu[a:b]
    sites, trees = posterior_sites(wts)
    viterbi_node = viterbi_path[a - start + sites]
    rows = []
    exact = None
    for strategy, parameter, kwargs in fb_strategies(rw, ts.num_nodes, quick):
        am = tsls.Hmm(wtree_sequence, rw, muw, **kwargs)
        before = time.perf_counter()
        fm = am.forward_matrix(hw)
        forward_time = time.perf_counter() - before
        forward_stats = (am.mean_num_values, am.max_num_values, am.mean_num_updated)
        before = time.perf_counter()
        bm = am.backward_matrix(hw, fm.normalisation_factor)
        backward_time = time.perf_counter() - before
        post = posteriors(fm, bm, sites, trees)
        stored_f = np.diff(fm.offsets)
        stored_b = np.diff(bm.offsets)
        if exact is None:
            exact = (fm.log_likelihood, post, forward_time + backward_time,
                     np.mean(stored_f) + np.mean(stored_b))
        compared = [compare_posterior(p, q) for (_, p), (_, q) in zip(post, exact[1])]
        tv = [c[0] for c in compared]
        max_abs = [c[1] for c in compared]
        argmax = [c[2] for c in compared]
        no_mass = sum(p is None for _, p in post)
        rows.append(
            {
                **base,
                "window_start": a,
                "window_sites": b - a,
                "strategy": strategy,
                "parameter": parameter,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "time_ratio": (forward_time + backward_time) / exact[2],
                "forward_mean_values": forward_stats[0],
                "forward_max_values": forward_stats[1],
                "forward_mean_updated": forward_stats[2],
                "backward_mean_values": am.mean_num_values,
                "backward_max_values": am.max_num_values,
                "stored_forward_mean": float(np.mean(stored_f)),
                "stored_forward_max": int(np.max(stored_f)),
                "stored_backward_mean": float(np.mean(stored_b)),
                "values_ratio": (np.mean(stored_f) + np.mean(stored_b)) / exact[3],
                "log_likelihood_error": fm.log_likelihood - exact[0],
                "mass_error": float(max(abs(mass - 1) for mass, _ in post)),
                "no_mass_sites": no_mass,
                "tv_mean": float(np.mean(tv)),
                "tv_max": float(np.max(tv)),
                "max_abs_error": float(np.max(max_abs)),
                "argmax_agreement": float(np.mean(argmax)),
                "posterior_viterbi_node": float(
                    np.mean(
                        [0 if p is None else p[u] for (_, p), u in zip(post, viterbi_node)]
                    )
                ),
            }
        )
        del fm, bm
    return rows


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def run_task(task):
    name, reference, mu_label, mu_parameter, query_id, quick = task
    ts, tree_sequence, alleles, num_alleles, get, _ = load_workload(name, reference)
    r, mu = model_rates(name, ts, mu_parameter)
    h, start, end = get(query_id)
    base = {
        "workload": name,
        "reference": reference,
        "mu_label": mu_label,
        "mu": mu[0],
        "r": r[0],
        "num_nodes": ts.num_nodes,
        "query": query_id,
        "num_sites": end - start,
    }
    try:
        vrows, path = viterbi_rows(
            base, tree_sequence, ts, r, mu, num_alleles, h, start, end, quick
        )
        frows = fb_rows(base, ts, alleles, r, mu, h, start, end, path, quick)
    except (MemoryError, tsls.LibraryError) as error:
        if isinstance(error, tsls.LibraryError) and "memory" not in str(error).lower():
            raise
        logger.warning(f"Out of memory: {base}")
        return [{**base, "strategy": "out_of_memory"}], []
    return vrows, frows


def make_tasks(quick):
    tasks = []
    sc2ts_sizes = [10000] if quick else [10000, 100000]
    num_sc2ts = 3 if quick else 10
    for size in sc2ts_sizes:
        for q in np.unique(np.linspace(0, 109, num_sc2ts).astype(int)):
            tasks.append(("sc2ts", size, "sc2ts", np.nan, int(q), quick))
    groups = ["ancestors_g84"] if quick else ["ancestors_g84", "ancestors_g1000",
                                              "ancestors_g1485"]
    num_ancestors = 2 if quick else 6
    for group in groups:
        _, num_queries = load_queries(DATA / "tgp_chr20p" / group)
        for q in np.unique(np.linspace(0, num_queries - 1, num_ancestors).astype(int)):
            tasks.append(("tgp_ancestors", group, "1e-20", 1e-20, int(q), quick))
    n = tskit.load(DATA / "tgp_chr20p" / "samples" / "reference.trees").num_nodes
    mus = [("1e-20", 1e-20)] + [
        (f"k={k}", sample_mismatch(k, n)) for k in SAMPLE_MISMATCH_K
    ]
    if quick:
        mus = mus[::3]
    _, num_queries = load_queries(DATA / "tgp_chr20p" / "samples")
    num_samples = 1 if quick else 8
    for label, mu in mus:
        for q in np.unique(np.linspace(0, num_queries - 1, num_samples).astype(int)):
            tasks.append(("tgp_samples", "samples", label, mu, int(q), quick))
    return tasks


def limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


def run(quick, num_processes):
    tasks = make_tasks(quick)
    viterbi, fb = [], []
    # Tasks for the same workload are submitted together so that a worker's
    # cached workload is reused; maxtasksperchild isn't available here, so
    # load_workload keeps one workload at a time.
    with concurrent.futures.ProcessPoolExecutor(
        num_processes, initializer=limit_memory
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            vrows, frows = future.result()
            viterbi.extend(vrows)
            fb.extend(frows)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(viterbi).to_csv(RESULTS / "viterbi.csv", index=False)
    pd.DataFrame(fb).to_csv(RESULTS / "forward_backward.csv", index=False)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def workload_label(df):
    label = df.workload + " " + df.reference.astype(str)
    return np.where(df.workload == "tgp_samples", label + " mu " + df.mu_label, label)


def fmt(x):
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
            return f"{x:.2g}"
        return f"{x:.3g}"
    return str(x)


def table(df, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    return lines


def analyse():
    v = pd.read_csv(RESULTS / "viterbi.csv")
    f = pd.read_csv(RESULTS / "forward_backward.csv")
    v["label"] = workload_label(v)
    f["label"] = workload_label(f)
    lines = ["# Thresholding evaluation: summary", ""]

    lines += ["## Exact runs: nodes tracked per site", ""]
    ve = v[v.strategy == "exact"].groupby("label")
    fe = f[f.strategy == "exact"].groupby("label")
    exact = pd.DataFrame(
        {
            "queries": ve.size(),
            "nodes": ve.num_nodes.first(),
            "sites": ve.num_sites.mean(),
            "mismatches": ve.num_mismatches.mean(),
            "switches": ve.num_switches.mean(),
            "V time (s)": ve.time.mean(),
            "V values mean": ve.mean_values.mean(),
            "V values max": ve.max_values.max(),
            "V updated": ve.mean_updated.mean(),
            "V sparse sites": ve.sparse_sites.mean(),
            "F values mean": fe.forward_mean_values.mean(),
            "F values max": fe.forward_max_values.max(),
            "F stored": fe.stored_forward_mean.mean(),
            "B stored": fe.stored_backward_mean.mean(),
            "FB time (s)": (fe.forward_time.mean() + fe.backward_time.mean()),
        }
    ).reset_index()
    lines += table(exact, list(exact.columns)) + [""]

    lines += ["## Viterbi escalation", ""]
    esc = v[v.strategy == "escalation"].groupby("label")
    summary = pd.DataFrame(
        {
            "queries": esc.size(),
            "time ratio": esc.time_ratio.mean(),
            "certified at k": esc.parameter.apply(
                lambda x: ", ".join(f"{k}: {c}" for k, c in x.value_counts().items())
            ),
            "optimal": esc.optimal.mean(),
        }
    ).reset_index()
    lines += table(summary, list(summary.columns)) + [""]
    lines += ["## Viterbi strategies", ""]
    for label, group in v[~v.strategy.isin(["exact", "escalation"])].groupby("label"):
        g = group.groupby(["strategy", "parameter"], dropna=False)
        summary = pd.DataFrame(
            {
                "time ratio": g.time_ratio.mean(),
                "values ratio": g.values_ratio.mean(),
                "max values": g.max_values.max(),
                "sparse sites": g.sparse_sites.mean(),
                "optimal": g.optimal.mean(),
                "same path": g.same_path.mean(),
                "mean loss": g.loss.mean(),
                "max loss": g.loss.max(),
                "certified": g.certified.mean(),
                "certified suboptimal": g.apply(
                    lambda x: int(np.sum((x.certified == True) & ~x.optimal)),  # noqa
                    include_groups=False,
                ),
            }
        ).reset_index()
        lines += [f"### {label}", ""] + table(summary, list(summary.columns)) + [""]

    lines += ["## Forward-backward strategies", ""]
    for label, group in f[f.strategy != "exact"].groupby("label"):
        g = group.groupby(["strategy", "parameter"])
        summary = pd.DataFrame(
            {
                "time ratio": g.time_ratio.mean(),
                "values ratio": g.values_ratio.mean(),
                "mean abs logL error": g.log_likelihood_error.apply(
                    lambda x: np.mean(np.abs(x))
                ),
                "max abs logL error": g.log_likelihood_error.apply(
                    lambda x: np.max(np.abs(x))
                ),
                "mean TV": g.tv_mean.mean(),
                "max TV": g.tv_max.max(),
                "max abs error": g.max_abs_error.max(),
                "mass error": g.mass_error.max(),
                "argmax agree": g.argmax_agreement.mean(),
                "no-mass sites": g.no_mass_sites.sum(),
            }
        ).reset_index()
        lines += [f"### {label}", ""] + table(summary, list(summary.columns)) + [""]
    (RESULTS / "summary.md").write_text("\n".join(lines))
    logger.info(f"Wrote {RESULTS / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-processes", "-t", type=int, default=1)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.analyse_only:
        run(args.quick, args.num_processes)
    analyse()


if __name__ == "__main__":
    main()
