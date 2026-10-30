"""
Run tsls's Viterbi algorithm on the benchmark workloads.

A Matcher is set up once for a tree sequence and model parameters, and its
match(h, start, end) method runs Viterbi for one query haplotype over sites
[start, end), returning the (left, right, parent, matched, stats) tuple:

- left, right, parent: the copying path as arrays of genome coordinates
  and node IDs, in left-to-right order;
- matched: the haplotype of the path at each site;
- stats: the HMM's statistics (memory, traceback and list sizes).

Model parameters
----------------
The model dicts give rho and weight_by_n as tsinfer's engine took them.
With weight_by_n=False, as in sc2ts, rho is the odds of switching to a
given node, and tsls is given r = tsls.recombination_probability(rho, n),
which has the same Viterbi paths (see docs/likelihood_model.md). rho can
also be an array of per-site probabilities (tsls.recombination_probabilities),
as tsinfer computes from a recombination rate or genetic map. We choose
rho as sc2ts does (solve_num_mismatches), so that k mismatches cost about
the same as one recombination.

tsinfer and sc2ts trees have a vestigial root, node 0, above node 1. We
remove node 0's edges so that it is never copied, which makes node IDs the
same as in the paths tsinfer found.
"""

import numpy as np
import tsls

ALLELES = ("A", "C", "G", "T")
MISSING = -1


def solve_num_mismatches(k, num_alleles=4):
    """
    Return the (mu, rho) tuple such that k mismatches have about the same
    likelihood as a single recombination, with recombination probabilities
    not weighted by n. Copied from sc2ts.inference.solve_num_mismatches.
    """
    assert k > 1
    assert num_alleles > 1
    mu = 0.0125
    denom = mu**k + (1 - (num_alleles - 1) * mu) ** k
    rho = mu**k / denom
    # Add a tiny bit of extra mass so that we deterministically recombine
    rho += rho * 0.1
    rho = max(rho, 1e-200)
    return mu, rho


def path_log_likelihood(h, num_segments, matched, mu, rho):
    """
    Return the log likelihood of a path with the specified number of segments
    and matched haplotype under the unweighted model, dropping the constant
    factors for staying on the same node. Missing sites count as matches, as
    in the engines.
    """
    num_alleles = len(ALLELES)
    non_missing = h != MISSING
    num_mismatches = np.sum(h[non_missing] != matched[non_missing])
    return (
        (num_segments - 1) * np.log(rho)
        + num_mismatches * np.log(mu)
        + (len(h) - num_mismatches) * np.log(1 - (num_alleles - 1) * mu)
    )


def site_num_alleles(ts, alleles):
    """
    Return the number of alleles at each site, as the engines count them for
    the emission probabilities: the number of alleles if a list is given,
    and otherwise the distinct states at the site, with at least 2.
    """
    if alleles is not None:
        return np.full(ts.num_sites, len(alleles))
    num_alleles = np.zeros(ts.num_sites, dtype=int)
    for site in ts.sites():
        states = {site.ancestral_state} | {m.derived_state for m in site.mutations}
        num_alleles[site.id] = max(2, len(states))
    return num_alleles


def model_path_log_likelihood(
    h, matched, left, positions, start, end, *, rho, mu, n, num_alleles
):
    """
    Return the log likelihood of a Viterbi path over sites [start, end)
    under the engines' model, with per-site rates: a switch at a site has
    probability rho / n, staying has probability 1 - rho + rho / n, and
    the emission is 1 - (A - 1) mu for a match (or missing data) and mu for
    a mismatch, where A is the number of alleles at the site. left holds the
    left coordinates of the path's segments; each one after the first is
    the position of a site at which the path switches. The initial
    probability is the same for every path and is left out.
    """
    sites = np.arange(start, end)
    switch = np.isin(positions[sites], np.asarray(left)[1:])
    switch[0] = False
    rho = rho[sites]
    stay = np.log(1 - rho + rho / n)
    transition = np.where(switch, np.log(rho / n), stay)
    transition[0] = 0
    mu = mu[sites]
    hs = h[sites]
    match = (hs == MISSING) | (hs == matched[sites])
    emission = np.where(match, np.log(1 - (num_alleles[sites] - 1) * mu), np.log(mu))
    return float(np.sum(transition) + np.sum(emission))


def path_haplotype(ts, left, right, parent, alleles=ALLELES):
    """
    Return the haplotype obtained by copying the specified path through the
    specified tree sequence, with states encoded as indexes into alleles, or
    as tskit numbers them if alleles is None.
    """
    nodes, index = np.unique(parent, return_inverse=True)
    G = ts.genotype_matrix(samples=nodes, isolated_as_missing=False, alleles=alleles)
    segment = np.searchsorted(left, ts.sites_position, side="right") - 1
    assert np.all(ts.sites_position < right[segment])
    return G[np.arange(ts.num_sites), index[segment]].astype(np.int8)


class Matcher:
    """
    tsls with the specified model. If mismatch_threshold is specified, nodes
    are pruned below the exact threshold for that many mismatches
    (tsls.path_threshold), which finds an optimal path if one has at most k
    mismatches and no switches, as sc2ts's first matching passes do.

    By default, states are encoded as indexes into ALLELES. If alleles is
    None, they are numbered at each site as tskit does, which for tsinfer's
    data is ancestral 0 and derived 1.
    """

    def __init__(
        self,
        ts,
        *,
        mu,
        rho,
        mismatch_threshold=None,
        weight_by_n=False,
        alleles=ALLELES,
    ):
        self.num_sites = ts.num_sites
        self.mu = mu
        self.rho = rho
        # rho can be a per-site array, as from a recombination map
        recombination = np.array(
            np.broadcast_to(np.asarray(rho, dtype=float), (ts.num_sites,))
        )
        if not weight_by_n:
            recombination = tsls.recombination_probability(recombination, ts.num_nodes)
        self.recombination = recombination
        self.mismatch = np.full(ts.num_sites, mu)
        tables = ts.dump_tables()
        tables.edges.keep_rows(tables.edges.parent != 0)
        self.reference = tsls.TreeSequence(tables.tree_sequence(), alleles=alleles)
        self.prune_threshold = None
        if mismatch_threshold is not None:
            self.prune_threshold = tsls.path_threshold(
                mismatch_threshold, self.mismatch, site_num_alleles(ts, alleles)
            )

    def make_hmm(self, **kwargs):
        return tsls.Hmm(
            self.reference,
            self.recombination,
            self.mismatch,
            prune_threshold=self.prune_threshold,
            **kwargs,
        )

    def match(self, h, start=0, end=None):
        """
        Match the specified haplotype over the sites from start to end
        (exclusive). The path covers only these sites.
        """
        if end is None:
            end = self.num_sites
        hmm = self.make_hmm()
        matched = np.zeros(self.num_sites, dtype=np.int8)
        left, right, parent = hmm.find_path(h.astype(np.int8), start, end, matched)
        stats = {
            "total_memory": hmm.total_memory,
            "mean_traceback_size": hmm.mean_traceback_size,
            "mean_num_values": hmm.mean_num_values,
            "num_sparse_sites": hmm.num_sparse_sites,
        }
        return left[::-1].copy(), right[::-1].copy(), parent[::-1].copy(), matched, stats
