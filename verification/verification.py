"""
Statistical validation of the tsls Li and Stephens HMM engine.

This is not a test suite: it runs empirical validations on simulated ARGs
and writes plots that help a user get a feel for the statistical
performance of the engine, rather than making hard assertions. It is
modelled on msprime's verification.py.

Tests are organised into classes of similar tests, and each test is
usually a call to a general method with different parameters. Tests must
be independent and not depend on any shared state within the class, other
than self.output_dir, which is guaranteed to exist when the test method is
called. The output directory is <output-dir>/<class name>/<test name>.
Each test writes one or more plots and a CSV file of the numbers behind
them. Test method names must be unique across all classes.

To list the available tests:

    uv run python verification.py --list

Run with no arguments to run all tests, with test names to run specific
tests, or with -c to run all the tests in a class. Use -t to run tests in
parallel processes.
"""

import argparse
import concurrent.futures
import inspect
import logging
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")

import msprime
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

import tsls

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent / "tsls" / "tests")
)
import dense

logger = logging.getLogger(__name__)

ALLELES = ("A", "C", "G", "T")
NUM_ALLELES = len(ALLELES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simulate_arg(
    num_samples,
    sequence_length,
    recombination_rate,
    mutation_rate,
    seed,
    population_size=1e4,
):
    """
    Simulate a haploid ARG with JC69 mutations (so sites can have several
    mutations and alleles).
    """
    ts = msprime.sim_ancestry(
        num_samples,
        ploidy=1,
        population_size=population_size,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed,
    )
    return msprime.sim_mutations(
        ts, rate=mutation_rate, model=msprime.JC69(), random_seed=seed
    )


def node_states(ts):
    """
    Return the (A, present, continuous) tuple of (num_sites, num_nodes)
    arrays, where A[j, u] is the allele index of node u at site j,
    present[j, u] is True if node u is in the tree (has a parent or a child)
    at site j, and continuous[j, u] is True if node u is in every tree
    between site j - 1 and site j. This is the vectorised equivalent of
    dense.node_states.
    """
    N = ts.num_nodes
    A = ts.genotype_matrix(
        samples=np.arange(N, dtype=np.int32), isolated_as_missing=False, alleles=ALLELES
    ).astype(np.int8)
    present = np.zeros((ts.num_sites, N), dtype=bool)
    continuous = np.zeros((ts.num_sites, N), dtype=bool)
    position = ts.sites_position
    # Nodes that have been in every tree since the last site
    since_last_site = np.ones(N, dtype=bool)
    for tree in ts.trees():
        start, stop = np.searchsorted(position, tree.interval)
        in_tree = (tree.parent_array[:N] != -1) | (tree.num_children_array[:N] > 0)
        since_last_site &= in_tree
        if stop > start:
            present[start:stop] = in_tree
            continuous[start] = since_last_site
            continuous[start + 1 : stop] = in_tree
            since_last_site = in_tree.copy()
    if ts.num_sites > 0:
        continuous[0] = present[0]
    return A, present, continuous


def hmm_params(k, num_alleles=NUM_ALLELES):
    """
    Return the (mu, rho) tuple for which k mismatches have about the same
    likelihood as one recombination, with weight_by_n=False (as in sc2ts and
    benchmarking/engines.py).
    """
    mu = 0.0125
    denom = mu**k + (1 - (num_alleles - 1) * mu) ** k
    rho = mu**k / denom
    rho += rho * 0.1
    return mu, rho


def n_weighted_rho(rho, n):
    """
    Return the rho for weight_by_n=True with the same ratio of switching to a
    given node to staying as rho with weight_by_n=False.
    """
    return rho / (rho + (1 - rho) / n)


def make_matcher(
    ts,
    mu,
    rho,
    *,
    weight_by_n=True,
    likelihood_threshold=None,
    pruning=None,
    count_distinct=False,
    tree_sequence=None,
):
    m = ts.num_sites
    if tree_sequence is None:
        tree_sequence = tsls.TreeSequence(ts, alleles=ALLELES)
    return tsls.Hmm(
        tree_sequence,
        np.full(m, rho),
        np.full(m, mu),
        weight_by_n=weight_by_n,
        likelihood_threshold=likelihood_threshold,
        pruning=pruning,
        count_distinct=count_distinct,
    )


def viterbi(am, ts, h):
    """
    Return the (path, matched) tuple, where path[j] is the node copied at
    site j.
    """
    m = ts.num_sites
    matched = np.zeros(m, dtype=np.int8)
    left, _, parent = am.find_path(h.astype(np.int8), 0, m, matched)
    index = np.argsort(left)
    segment = np.searchsorted(left[index], ts.sites_position, side="right") - 1
    return parent[index][segment], matched


def posterior(am, h):
    """
    Return the (P, log_likelihood) tuple, where P is the dense
    (num_sites, num_nodes) matrix of posterior probabilities.
    """
    fm = am.forward_matrix(h.astype(np.int8))
    bm = am.backward_matrix(h.astype(np.int8), fm.normalisation_factor)
    return fm.decode() * bm.decode(), fm.log_likelihood


def iter_decoded_sites(cm):
    """
    Yield the (site, values) of the specified CompressedMatrix one site at a
    time, without building the dense matrix (for long sequences). Uses the
    same approach as CompressedMatrix.decode.
    """
    ts = cm.ts
    N = ts.num_nodes
    for tree in ts.trees():
        parent = tree.parent_array
        in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
        for site in tree.sites():
            site_nodes, site_values = cm.get_site(site.id)
            value = np.full(N + 1, np.nan)
            value[site_nodes] = site_values
            ancestor = parent.copy()
            unset = np.where(in_tree & np.isnan(value[:N]))[0]
            while len(unset) > 0:
                value[unset] = value[ancestor[unset]]
                ancestor[unset] = parent[ancestor[unset]]
                unset = unset[np.isnan(value[unset])]
            yield site.id, np.where(in_tree, value[:N], 0)


def sample_prior_path(present, continuous, rho, n, rng):
    """
    Sample a copying path exactly from the engine's prior over paths,
    p(path) proportional to the product of A_j(u_{j-1}, u_j), where
    A_j(u, v) = present_j(v) [(1 - rho_j) [u = v] continuous_j(u) + rho_j / n].
    We first run a backward pass with no data and then sample forwards.
    """
    m, N = present.shape
    beta = np.zeros((m, N))
    b = present[m - 1].astype(float)
    beta[m - 1] = b / np.sum(b)
    for j in range(m - 1, 0, -1):
        b_next = beta[j]
        b = present[j - 1] * (
            (1 - rho[j]) * continuous[j] * b_next + rho[j] / n * np.sum(b_next)
        )
        beta[j - 1] = b / np.sum(b)
    path = np.zeros(m, dtype=np.int32)
    w = present[0] * beta[0]
    path[0] = rng.choice(N, p=w / np.sum(w))
    for j in range(1, m):
        u = path[j - 1]
        w = present[j] * rho[j] / n * beta[j]
        if continuous[j, u]:
            w[u] += (1 - rho[j]) * beta[j, u]
        path[j] = rng.choice(N, p=w / np.sum(w))
    return path


def mutate_along_path(A, path, mu, rng):
    """
    Return the haplotype obtained by copying the alleles along the path,
    changing each to each of the other alleles with probability mu (the
    engine's emission model).
    """
    m = len(path)
    h = A[np.arange(m), path].copy()
    change = rng.random(m) < (NUM_ALLELES - 1) * mu
    offset = rng.integers(1, NUM_ALLELES, size=m)
    h[change] = (h[change] + offset[change]) % NUM_ALLELES
    return h.astype(np.int8)


def tree_distance(parent, u, v):
    """
    Return the number of edges between u and v in the tree with the specified
    parent array, or -1 if they are not connected.
    """
    depth = {}
    x, d = u, 0
    while x != -1:
        depth[x] = d
        x = parent[x]
        d += 1
    x, d = v, 0
    while x != -1:
        if x in depth:
            return depth[x] + d
        x = parent[x]
        d += 1
    return -1


def num_switches(path):
    return int(np.sum(path[1:] != path[:-1]))


def nearest_distances(true_sites, inferred_sites):
    """
    Return the distance (in sites) from each of the true switch sites to the
    nearest inferred switch site, or NaN if there are none.
    """
    if len(inferred_sites) == 0:
        return np.full(len(true_sites), np.nan)
    inferred_sites = np.sort(inferred_sites)
    distances = []
    for s in true_sites:
        k = np.searchsorted(inferred_sites, s)
        candidates = inferred_sites[max(k - 1, 0) : k + 1]
        distances.append(np.min(np.abs(candidates - s)))
    return np.array(distances, dtype=float)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class Test:
    """
    The superclass of all test classes. self.output_dir is set and exists
    when a test method is called.
    """

    output_dir = None

    def savefig(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {path}")

    def save_csv(self, df, name):
        path = self.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Wrote {path}")


def plot_sweep(ax, df, x, y, label, logx=False):
    """
    Plot the mean and standard deviation of y over replicates for each x.
    """
    summary = df.groupby(x)[y].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        summary[x],
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=3,
        label=label,
    )
    if logx:
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xticks(summary[x], [f"{v:g}" for v in summary[x]])


# ---------------------------------------------------------------------------
# Calibration under the model
# ---------------------------------------------------------------------------


class ModelCalibration(Test):
    """
    Queries are simulated from the HMM itself: we sample a copying path
    exactly from the engine's prior on an msprime ARG and add mutations with
    the engine's emission model. Forward-backward posteriors should then be
    calibrated: of the (site, node) pairs given posterior p, a fraction p
    should be the true node. With weight_by_n=False the prior is an
    unnormalised measure over paths, but we sample from it exactly, so the
    posteriors should still be calibrated.
    """

    num_args = 20
    paths_per_arg = 10
    num_samples = 50
    sequence_length = 5e5
    levels = (0.5, 0.8, 0.9, 0.95, 0.99)

    def simulate_queries(
        self, switches_per_sequence, mu, weight_by_n, seed, num_args, paths_per_arg
    ):
        """
        Yield (ts, am, A, path, h) for queries simulated from the model, with
        about the specified expected number of switches along the sequence.
        """
        rng = np.random.default_rng(seed)
        for j in range(num_args):
            ts = simulate_arg(
                self.num_samples, self.sequence_length, 1e-8, 1e-8, seed + j + 1
            )
            A, present, continuous = node_states(ts)
            n = ts.num_nodes if weight_by_n else 1
            rho = switches_per_sequence / ts.num_sites
            # With weight_by_n=False use the same probability of switching to
            # a given node.
            engine_rho = rho if weight_by_n else rho / ts.num_nodes
            am = make_matcher(ts, mu, engine_rho, weight_by_n=weight_by_n)
            rho_array = np.full(ts.num_sites, engine_rho)
            for _ in range(paths_per_arg):
                path = sample_prior_path(present, continuous, rho_array, n, rng)
                h = mutate_along_path(A, path, mu, rng)
                yield ts, am, A, path, h

    def _run_calibration(self, switches_per_sequence, mu, weight_by_n=True, seed=1):
        bins = np.linspace(0, 1, 21)
        num_bins = len(bins) - 1
        # Per-query counts in each bin, so that we can bootstrap over queries
        # (sites within a query are strongly correlated).
        query_count = []
        query_total_p = []
        query_num_true = []
        covered = {level: [] for level in self.levels}
        set_size = {level: [] for level in self.levels}
        set_mass = {level: [] for level in self.levels}
        for ts, am, _, path, h in self.simulate_queries(
            switches_per_sequence,
            mu,
            weight_by_n,
            seed,
            self.num_args,
            self.paths_per_arg,
        ):
            P, _ = posterior(am, h)
            m = ts.num_sites
            is_true = np.zeros_like(P, dtype=bool)
            is_true[np.arange(m), path] = True
            keep = P > 1e-4
            index = np.clip(np.digitize(P[keep], bins) - 1, 0, num_bins - 1)
            query_count.append(np.bincount(index, minlength=num_bins))
            query_total_p.append(np.bincount(index, weights=P[keep], minlength=num_bins))
            query_num_true.append(
                np.bincount(index, weights=is_true[keep], minlength=num_bins)
            )
            # Credible sets: the smallest set of nodes with total posterior at
            # least the level, at each site. Nodes with identical haplotypes
            # have exactly equal posteriors, so we break ties at random: if
            # the true node is tied with g nodes, r nodes have a larger
            # posterior and the set has size s, the probability that the set
            # contains the true node is (s - r) / g, clipped to [0, 1].
            sorted_P = -np.sort(-P, axis=1)
            cumulative = np.cumsum(sorted_P, axis=1)
            p_true = P[np.arange(m), path][:, None]
            tied = np.abs(P - p_true) <= 1e-9 * p_true
            num_greater = np.sum((P > p_true) & ~tied, axis=1)
            num_tied = np.sum(tied, axis=1)
            for level in self.levels:
                size = np.sum(cumulative < level, axis=1) + 1
                covered[level].extend(np.clip((size - num_greater) / num_tied, 0, 1))
                set_size[level].extend(size)
                set_mass[level].extend(cumulative[np.arange(m), size - 1])

        query_count = np.array(query_count)
        query_total_p = np.array(query_total_p)
        query_num_true = np.array(query_num_true)
        count = np.sum(query_count, axis=0)
        nonempty = count > 0
        mean_p = np.sum(query_total_p, axis=0)[nonempty] / count[nonempty]
        frequency = np.sum(query_num_true, axis=0)[nonempty] / count[nonempty]
        # Bootstrap over queries for the uncertainty in the observed frequency
        rng = np.random.default_rng(seed)
        replicates = []
        for _ in range(500):
            sample = rng.integers(0, len(query_count), size=len(query_count))
            c = np.sum(query_count[sample], axis=0)[nonempty]
            t = np.sum(query_num_true[sample], axis=0)[nonempty]
            replicates.append(np.where(c > 0, t / np.maximum(c, 1), np.nan))
        lower, upper = np.nanpercentile(replicates, [2.5, 97.5], axis=0)
        df = pd.DataFrame(
            {
                "mean_posterior": mean_p,
                "observed_frequency": frequency,
                "lower": lower,
                "upper": upper,
                "count": count[nonempty],
            }
        )
        self.save_csv(df, "reliability")
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 8), height_ratios=[3, 1], sharex=True
        )
        ax1.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax1.errorbar(
            mean_p,
            frequency,
            yerr=[frequency - lower, upper - frequency],
            marker="o",
            capsize=3,
        )
        ax1.set_ylabel("Fraction that are the true node (95% bootstrap CI)")
        ax1.set_title("Reliability of posterior P(copy node u at site j)")
        ax2.bar(
            bins[:-1][nonempty], count[nonempty], width=np.diff(bins)[0], align="edge"
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("Posterior probability")
        ax2.set_ylabel("Count")
        self.savefig(fig, "reliability")

        rows = []
        for level in self.levels:
            rows.append(
                {
                    "level": level,
                    "coverage": np.mean(covered[level]),
                    "mean_set_mass": np.mean(set_mass[level]),
                    "mean_size": np.mean(set_size[level]),
                    "median_size": np.median(set_size[level]),
                }
            )
        df = pd.DataFrame(rows)
        self.save_csv(df, "credible_sets")
        logger.info(f"\n{df}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot([0.4, 1], [0.4, 1], color="grey", linestyle="--")
        ax1.plot(df.level, df.coverage, marker="o", label="Coverage")
        ax1.plot(
            df.level,
            df.mean_set_mass,
            marker="x",
            linestyle=":",
            label="Mean posterior mass in set (expected coverage)",
        )
        ax1.legend()
        ax1.set_xlabel("Nominal level")
        ax1.set_ylabel("Fraction of sites where the set contains the true node")
        ax1.set_title("Credible set coverage")
        ax2.boxplot([set_size[level] for level in self.levels], showfliers=False)
        ax2.set_xticks(range(1, len(self.levels) + 1), [str(x) for x in self.levels])
        ax2.set_xlabel("Nominal level")
        ax2.set_ylabel("Number of nodes in set")
        ax2.set_title("Credible set size")
        self.savefig(fig, "credible_sets")

    def test_calibration_default(self):
        self._run_calibration(switches_per_sequence=5, mu=0.01)

    def test_calibration_high_recombination(self):
        self._run_calibration(switches_per_sequence=25, mu=0.01)

    def test_calibration_high_mutation(self):
        self._run_calibration(switches_per_sequence=5, mu=0.05)

    def test_calibration_no_weight_by_n(self):
        self._run_calibration(switches_per_sequence=5, mu=0.01, weight_by_n=False)

    def _run_decoding(self, name, values, make_params):
        rows = []
        for value in values:
            switches, mu = make_params(value)
            for ts, am, _, path, h in self.simulate_queries(
                switches, mu, True, 2, num_args=10, paths_per_arg=5
            ):
                vpath, _ = viterbi(am, ts, h)
                P, _ = posterior(am, h)
                m = ts.num_sites
                rows.append(
                    {
                        name: value,
                        "viterbi_accuracy": np.mean(vpath == path),
                        "posterior_max_accuracy": np.mean(np.argmax(P, axis=1) == path),
                        "mean_posterior_true": np.mean(P[np.arange(m), path]),
                        "true_switches": num_switches(path),
                        "viterbi_switches": num_switches(vpath),
                    }
                )
        df = pd.DataFrame(rows)
        self.save_csv(df, "decoding")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        plot_sweep(ax1, df, name, "viterbi_accuracy", "Viterbi", logx=True)
        plot_sweep(ax1, df, name, "posterior_max_accuracy", "Posterior max", logx=True)
        plot_sweep(
            ax1, df, name, "mean_posterior_true", "Mean posterior of truth", logx=True
        )
        ax1.set_xlabel(name)
        ax1.set_ylabel("Fraction of sites")
        ax1.set_title("Decoding the true node")
        ax1.legend()
        for value, group in df.groupby(name):
            ax2.scatter(
                group.true_switches,
                group.viterbi_switches,
                label=f"{value:g}",
                alpha=0.6,
            )
        top = max(df.true_switches.max(), df.viterbi_switches.max())
        ax2.plot([0, top], [0, top], color="grey", linestyle="--")
        ax2.set_xlabel("True number of switches")
        ax2.set_ylabel("Viterbi number of switches")
        ax2.legend(title=name)
        self.savefig(fig, "decoding")

    def test_decoding_accuracy_vs_mutation(self):
        self._run_decoding("mu", [0.001, 0.003, 0.01, 0.03, 0.1], lambda mu: (5, mu))

    def test_decoding_accuracy_vs_recombination(self):
        self._run_decoding(
            "switches_per_sequence", [1, 3, 10, 30, 100], lambda s: (s, 0.01)
        )


# ---------------------------------------------------------------------------
# Held-out sample threading
# ---------------------------------------------------------------------------


def threading_replicate(
    num_samples,
    sequence_length,
    recombination_rate,
    mutation_rate,
    k,
    seed,
    *,
    weight_by_n=False,
    likelihood_threshold=None,
    compute_posterior=True,
):
    """
    Simulate num_samples + 1 samples, hold out the last one, and match its
    haplotype against the ARG of the others. Return the (metrics, details)
    tuple, where metrics is a dict of per-replicate summaries.
    """
    full_ts = simulate_arg(
        num_samples + 1, sequence_length, recombination_rate, mutation_rate, seed
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
    m = ts.num_sites
    h = full_ts.genotype_matrix(
        samples=[query], isolated_as_missing=False, alleles=ALLELES
    )[:, 0].astype(np.int8)

    # The true copying target at each site is the query's parent
    true_parent = np.zeros(m, dtype=np.int32)
    for tree in full_ts.trees():
        for site in tree.sites():
            true_parent[site.id] = tree.parent(query)

    mu, rho = hmm_params(k)
    if weight_by_n:
        rho = n_weighted_rho(rho, ts.num_nodes)
    am = make_matcher(
        ts, mu, rho, weight_by_n=weight_by_n, likelihood_threshold=likelihood_threshold
    )
    before = time.perf_counter()
    path, _ = viterbi(am, ts, h)
    viterbi_time = time.perf_counter() - before

    distance = np.zeros(m, dtype=int)
    near = []
    time_error = np.abs(ts.nodes_time[path] - ts.nodes_time[true_parent])
    for tree in ts.trees():
        parent = tree.parent_array
        for site in tree.sites():
            p = true_parent[site.id]
            distance[site.id] = tree_distance(parent, path[site.id], p)
            near.append(
                [p, *tree.children(p)] + ([parent[p]] if parent[p] != -1 else [])
            )

    metrics = {
        "exact": np.mean(path == true_parent),
        "within_2": np.mean((distance >= 0) & (distance <= 2)),
        "mean_distance": np.mean(distance[distance >= 0]),
        "median_time_error": np.median(time_error),
        "viterbi_time": viterbi_time,
        "mean_traceback_size": am.mean_traceback_size,
        "num_nodes": ts.num_nodes,
        "num_sites": m,
    }
    P = None
    if compute_posterior:
        P, _ = posterior(am, h)
        metrics["posterior_true"] = np.mean(P[np.arange(m), true_parent])
        metrics["posterior_near"] = np.mean([np.sum(P[j, near[j]]) for j in range(m)])
    true_switch = np.where(true_parent[1:] != true_parent[:-1])[0] + 1
    inferred_switch = np.where(path[1:] != path[:-1])[0] + 1
    metrics["true_switches"] = len(true_switch)
    metrics["inferred_switches"] = len(inferred_switch)
    details = {
        "breakpoint_distance": nearest_distances(true_switch, inferred_switch),
        "path": path,
        "posterior": P,
        "ts": ts,
        "h": h,
        "mu": mu,
        "rho": rho,
    }
    return metrics, details


THREADING_DEFAULTS = {
    "num_samples": 50,
    "sequence_length": 5e5,
    "recombination_rate": 1e-8,
    "mutation_rate": 1e-8,
    "k": 3,
}


class HeldOutThreading(Test):
    """
    Match a held-out sample from an msprime simulation against the ARG of the
    other samples. The truth at each site is the held-out sample's parent in
    the full ARG: the node where its lineage joins the others, which is the
    best node to copy. This needs no assumptions about the HMM.
    """

    num_replicates = 20

    def _run(self, name, values, logx=True):
        rows = []
        breakpoint_rows = []
        for value in values:
            params = dict(THREADING_DEFAULTS)
            params[name] = value
            for j in range(self.num_replicates):
                metrics, details = threading_replicate(**params, seed=j + 1)
                metrics[name] = value
                rows.append(metrics)
                for d in details["breakpoint_distance"]:
                    breakpoint_rows.append({name: value, "distance": d})
        df = pd.DataFrame(rows)
        self.save_csv(df, "threading")
        logger.info(f"\n{df.groupby(name).mean()}")

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax = axes[0, 0]
        plot_sweep(ax, df, name, "exact", "Viterbi node is the parent", logx)
        plot_sweep(ax, df, name, "within_2", "Within 2 edges", logx)
        ax.set_ylabel("Fraction of sites")
        ax.set_title("Viterbi copying target")
        ax.legend()
        ax = axes[0, 1]
        plot_sweep(ax, df, name, "posterior_true", "On the parent", logx)
        plot_sweep(ax, df, name, "posterior_near", "Within 1 edge", logx)
        ax.set_ylabel("Mean posterior mass")
        ax.set_title("Posterior mass on the true parent")
        ax.legend()
        ax = axes[1, 0]
        plot_sweep(ax, df, name, "median_time_error", "", logx)
        ax.set_yscale("log")
        ax.set_ylabel("Median |t(Viterbi node) - t(parent)| (generations)")
        ax = axes[1, 1]
        plot_sweep(ax, df, name, "true_switches", "True", logx)
        plot_sweep(ax, df, name, "inferred_switches", "Viterbi", logx)
        ax.set_ylabel("Number of switches")
        ax.legend()
        for ax in axes.flat:
            ax.set_xlabel(name)
        self.savefig(fig, "threading")

        bp = pd.DataFrame(breakpoint_rows)
        self.save_csv(bp, "breakpoints")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        groups = [g.distance.dropna().values for _, g in bp.groupby(name)]
        labels = [f"{v:g}" for v, _ in bp.groupby(name)]
        ax1.boxplot(groups, showfliers=False)
        ax1.set_xticks(range(1, len(labels) + 1), labels)
        ax1.set_xlabel(name)
        ax1.set_ylabel("Sites to the nearest Viterbi switch")
        ax1.set_title("Distance from each true switch to the nearest inferred")
        for value, group in df.groupby(name):
            ax2.scatter(
                group.true_switches,
                group.inferred_switches,
                label=f"{value:g}",
                alpha=0.6,
            )
        top = max(df.true_switches.max(), df.inferred_switches.max())
        ax2.plot([0, top], [0, top], color="grey", linestyle="--")
        ax2.set_xlabel("True switches")
        ax2.set_ylabel("Viterbi switches")
        ax2.legend(title=name)
        self.savefig(fig, "breakpoints")

    def test_threading_vs_mutation_rate(self):
        self._run("mutation_rate", [2.5e-9, 5e-9, 1e-8, 2e-8, 4e-8])

    def test_threading_vs_recombination_rate(self):
        self._run("recombination_rate", [2.5e-9, 5e-9, 1e-8, 2e-8, 4e-8])

    def test_threading_vs_sample_size(self):
        self._run("num_samples", [10, 20, 50, 100, 200])

    def test_threading_vs_hmm_rho(self):
        self._run("k", [2, 3, 4, 5, 6], logx=False)


# ---------------------------------------------------------------------------
# Numerics and approximations
# ---------------------------------------------------------------------------


class Numerics(Test):
    """
    The numerical behaviour of the engine and the effect of its
    approximations.
    """

    def test_likelihood_threshold(self):
        """
        The likelihood threshold sets normalised values below it to the
        threshold, which bounds the number of distinct values (and so time and
        memory) at the cost of exactness.
        """
        thresholds = [None, 1e-30, 1e-20, 1e-10, 1e-6, 1e-3]
        rows = []
        for j in range(20):
            reference = None
            for threshold in thresholds:
                metrics, details = threading_replicate(
                    **THREADING_DEFAULTS,
                    seed=j + 1,
                    likelihood_threshold=threshold,
                )
                ts = details["ts"]
                A, present, continuous = node_states(ts)
                m = ts.num_sites
                value = dense.path_log_likelihood(
                    A,
                    present,
                    continuous,
                    details["h"],
                    np.full(m, details["rho"]),
                    np.full(m, details["mu"]),
                    NUM_ALLELES,
                    1,
                    details["path"],
                )
                P = details["posterior"]
                if reference is None:
                    reference = (value, P)
                metrics["threshold"] = "none" if threshold is None else f"{threshold:g}"
                metrics["log_likelihood_loss"] = reference[0] - value
                metrics["posterior_tv"] = 0.5 * np.mean(
                    np.sum(np.abs(P - reference[1]), axis=1)
                )
                rows.append(metrics)
        df = pd.DataFrame(rows)
        self.save_csv(df, "threshold")
        labels = ["none"] + [f"{t:g}" for t in thresholds[1:]]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, column, ylabel in [
            (axes[0, 0], "log_likelihood_loss", "Loss in Viterbi path log likelihood"),
            (axes[0, 1], "posterior_tv", "Mean total variation from exact posterior"),
            (axes[1, 0], "mean_traceback_size", "Mean traceback size"),
            (axes[1, 1], "viterbi_time", "Viterbi time (s)"),
        ]:
            groups = [df[df.threshold == label][column].values for label in labels]
            ax.boxplot(groups)
            ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.set_xlabel("likelihood_threshold")
            ax.set_ylabel(ylabel)
        self.savefig(fig, "threshold")
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = [df[df.threshold == label].exact.values for label in labels]
        ax.boxplot(groups)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_xlabel("likelihood_threshold")
        ax.set_ylabel("Fraction of sites where Viterbi copies the parent")
        self.savefig(fig, "threshold_accuracy")

    def test_posterior_normalisation(self):
        """
        With the forward normalisation factors, sum_u F(u) B(u) = 1 at every
        site in exact arithmetic. Check for numerical drift along long
        sequences.
        """
        rows = []
        for sequence_length in [1e5, 3e5, 1e6, 3e6, 1e7, 2e7]:
            for weight_by_n in [True, False]:
                for j in range(3):
                    ts = simulate_arg(20, sequence_length, 1e-8, 1e-8, j + 1)
                    rng = np.random.default_rng(j)
                    h = ts.genotype_matrix(
                        samples=[0], isolated_as_missing=False, alleles=ALLELES
                    )[:, 0]
                    noise = rng.random(ts.num_sites) < 0.01
                    h[noise] = rng.integers(0, NUM_ALLELES, size=np.sum(noise))
                    mu, rho = hmm_params(3)
                    if weight_by_n:
                        rho = n_weighted_rho(rho, ts.num_nodes)
                    am = make_matcher(ts, mu, rho, weight_by_n=weight_by_n)
                    fm = am.forward_matrix(h.astype(np.int8))
                    bm = am.backward_matrix(h.astype(np.int8), fm.normalisation_factor)
                    max_error = 0
                    for (_, f), (_, b) in zip(
                        iter_decoded_sites(fm), iter_decoded_sites(bm)
                    ):
                        max_error = max(max_error, abs(np.sum(f * b) - 1))
                    rows.append(
                        {
                            "num_sites": ts.num_sites,
                            "num_nodes": ts.num_nodes,
                            "weight_by_n": weight_by_n,
                            "max_error": max_error,
                            "log_likelihood_per_site": fm.log_likelihood / ts.num_sites,
                        }
                    )
        df = pd.DataFrame(rows)
        self.save_csv(df, "normalisation")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for weight_by_n, group in df.groupby("weight_by_n"):
            label = f"weight_by_n={weight_by_n}"
            ax1.scatter(group.num_sites, np.maximum(group.max_error, 1e-17), label=label)
            ax2.scatter(group.num_sites, group.log_likelihood_per_site, label=label)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Number of sites")
        ax1.set_ylabel("max over sites of |sum F*B - 1|")
        ax1.legend()
        ax2.set_xscale("log")
        ax2.set_xlabel("Number of sites")
        ax2.set_ylabel("Forward log likelihood per site")
        ax2.legend()
        self.savefig(fig, "normalisation")

    def test_dense_agreement_random_args(self):
        """
        Compare the engine with the dense implementation used in the unit tests
        on many small random ARGs with random parameters.
        """
        rng = np.random.default_rng(1234)
        rows = []
        for j in range(200):
            ts = simulate_arg(
                int(rng.integers(3, 11)),
                int(rng.integers(20, 61)),
                rng.uniform(0, 0.05),
                0.3,
                j + 1,
                population_size=1,
            )
            if ts.num_sites == 0:
                continue
            A, present, continuous = dense.node_states(ts, list(ALLELES))
            rho = 10 ** rng.uniform(-4, -0.5)
            mu = 10 ** rng.uniform(-3, np.log10(0.2))
            weight_by_n = bool(rng.integers(0, 2))
            threshold = [None, 1e-6, 1e-3][rng.integers(0, 3)]
            am = make_matcher(
                ts, mu, rho, weight_by_n=weight_by_n, likelihood_threshold=threshold
            )
            n = ts.num_nodes if weight_by_n else 1
            m = ts.num_sites
            rho_a = np.full(m, rho)
            mu_a = np.full(m, mu)
            dense_threshold = 0 if threshold is None else threshold
            u = int(rng.integers(0, ts.num_samples))
            h_random = rng.integers(0, NUM_ALLELES, size=m).astype(np.int8)
            h_missing = A[:, u].copy()
            h_missing[rng.random(m) < 0.2] = -1
            for h in [A[:, u].copy(), h_random, h_missing]:
                fm = am.forward_matrix(h)
                bm = am.backward_matrix(h, fm.normalisation_factor)
                F, c = dense.forward(
                    A,
                    present,
                    continuous,
                    h,
                    rho_a,
                    mu_a,
                    NUM_ALLELES,
                    n,
                    dense_threshold,
                )
                B = dense.backward(
                    A,
                    present,
                    continuous,
                    h,
                    rho_a,
                    mu_a,
                    NUM_ALLELES,
                    n,
                    c,
                    dense_threshold,
                )

                def relative_error(x, y):
                    return np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-200))

                path, _ = viterbi(am, ts, h)
                optimum = dense.viterbi_log_likelihood(
                    A, present, continuous, h, rho_a, mu_a, NUM_ALLELES, n
                )
                value = dense.path_log_likelihood(
                    A, present, continuous, h, rho_a, mu_a, NUM_ALLELES, n, path
                )
                rows.append(
                    {
                        "forward_error": relative_error(fm.decode(), F),
                        "backward_error": relative_error(bm.decode(), B),
                        "viterbi_gap": optimum - value if threshold is None else np.nan,
                    }
                )
        df = pd.DataFrame(rows)
        self.save_csv(df, "dense_agreement")
        logger.info(f"\n{df.describe()}")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, column in zip(axes, ["forward_error", "backward_error", "viterbi_gap"]):
            values = np.abs(df[column].dropna().values)
            ax.hist(np.log10(np.maximum(values, 1e-18)), bins=30)
            ax.set_xlabel(f"log10 {column} (floored at -18)")
            ax.set_ylabel("Count")
        axes[2].set_title("Viterbi optimum - path (no threshold)")
        self.savefig(fig, "dense_agreement")

    def test_weight_by_n(self):
        """
        Compare threading with weight_by_n on and off, with the same ratio of
        switching to a given node to staying (so the Viterbi paths are the
        same up to ties); the posteriors differ because the models do.
        """
        rows = []
        for k in [2, 3, 4, 5, 6]:
            for weight_by_n in [True, False]:
                for j in range(20):
                    params = dict(THREADING_DEFAULTS)
                    params["k"] = k
                    metrics, _ = threading_replicate(
                        **params, seed=j + 1, weight_by_n=weight_by_n
                    )
                    metrics["k"] = k
                    metrics["weight_by_n"] = weight_by_n
                    rows.append(metrics)
        df = pd.DataFrame(rows)
        self.save_csv(df, "weight_by_n")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for weight_by_n, group in df.groupby("weight_by_n"):
            label = f"weight_by_n={weight_by_n}"
            plot_sweep(ax1, group, "k", "exact", label)
            plot_sweep(ax2, group, "k", "posterior_true", label)
        ax1.set_xlabel("k (mismatches per recombination)")
        ax1.set_ylabel("Fraction of sites where Viterbi copies the parent")
        ax1.legend()
        ax2.set_xlabel("k (mismatches per recombination)")
        ax2.set_ylabel("Mean posterior on the parent")
        ax2.legend()
        self.savefig(fig, "weight_by_n")


# ---------------------------------------------------------------------------
# Running tests
# ---------------------------------------------------------------------------


class TestInstance:
    def __init__(self, test_class, method_name):
        self.test_class = test_class
        self.method_name = method_name

    def run(self, basedir):
        cls = globals()[self.test_class]
        instance = cls()
        instance.output_dir = pathlib.Path(basedir) / self.test_class / self.method_name
        instance.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running {self.method_name}")
        getattr(instance, self.method_name)()


def make_suite():
    """
    Return a dict mapping test method names to TestInstances.
    """
    tests = {}
    for class_name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(cls, Test) and cls is not Test:
            for name, _ in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith("test_"):
                    if name in tests:
                        raise ValueError(f"Duplicate test name {name}")
                    tests[name] = TestInstance(class_name, name)
    return tests


def run_tests(tests, basedir, num_processes, show_progress):
    progress = tqdm.tqdm(total=len(tests), disable=not show_progress)
    if num_processes <= 1:
        for test in tests:
            test.run(basedir)
            progress.update()
    else:
        with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
            futures = [executor.submit(test.run, basedir) for test in tests]
            for future in concurrent.futures.as_completed(futures):
                exception = future.exception()
                if exception is not None:
                    logger.error(f"EXCEPTION: {exception}")
                    for f in futures:
                        f.cancel()
                    raise exception
                progress.update()
    progress.close()


def main():
    suite = make_suite()
    classes = sorted({test.test_class for test in suite.values()})
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tests", nargs="*", help="Run specific tests (see --list)")
    parser.add_argument(
        "--test-class", "-c", choices=classes, help="Run all the tests in this class"
    )
    parser.add_argument(
        "--output-dir", "-d", default="tmp__NOBACKUP__", help="Base output directory"
    )
    parser.add_argument(
        "--num-processes", "-t", type=int, default=1, help="Number of processes"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quiet", "-q", action="store_true", help="Only log warnings")
    group.add_argument("--debug", "-D", action="store_true", help="Debug logging")
    parser.add_argument(
        "--no-progress", "-n", action="store_true", help="Do not show a progress bar"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List the tests")
    args = parser.parse_args()

    level = logging.INFO
    if args.quiet:
        level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("msprime").setLevel(logging.WARNING)

    if args.list:
        for name, test in suite.items():
            print(test.test_class, name, sep="\t")
        return
    if len(args.tests) > 0:
        unknown = [name for name in args.tests if name not in suite]
        if len(unknown) > 0:
            parser.error(f"Unknown tests: {unknown}")
        tests = [suite[name] for name in args.tests]
    elif args.test_class is not None:
        tests = [t for t in suite.values() if t.test_class == args.test_class]
    else:
        tests = list(suite.values())
    run_tests(tests, args.output_dir, args.num_processes, not args.no_progress)


if __name__ == "__main__":
    main()
