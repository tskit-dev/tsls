"""
Uniform adapters over the HMM engines.

Each engine is set up once for a tree sequence and model parameters, and
its match(h) method runs the Viterbi algorithm for one query haplotype,
returning the (left, right, parent, matched, stats) tuple:

- left, right, parent: the copying path as arrays of genome coordinates
  and node IDs, in left-to-right order, covering the whole sequence;
- matched: the haplotype of the path at each site, or None if the engine
  does not compute it (see path_haplotype);
- stats: a dict of engine-specific statistics.

Model parameters
----------------
Both engines use the same Viterbi recursion: at each site the likelihood
of staying on a node is p_last * (1 - rho + rho / n), and of switching
is rho / n. tsinfer-hmm with weight_by_n=False uses n = 1, as sc2ts does,
while tskit-hmm uses n = num_samples. Because the likelihoods are
renormalised at every site, only the ratio of switching to staying
matters, so the tskit rho that gives the same model as a tsinfer rho r
is r / (r + (1 - r) / n). We choose r as sc2ts does, so that k mismatches
cost about the same as one recombination.

The engines still differ in what they can copy from: tsinfer-hmm can copy
any node, while tskit-hmm can only copy samples.
"""

import _tskit_hmm
import numpy as np
import tsinfer_hmm
import tskit_hmm
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


def n_weighted_rho(rho, n):
    """
    Return the rho for an engine that divides rho by n that gives the same
    Viterbi path as the specified rho in an engine that does not.
    """
    return rho / (rho + (1 - rho) / n)


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


def path_to_segments(path, positions, sequence_length):
    """
    Convert a per-site array of node IDs into the (left, right, parent)
    tuple of segments covering the whole sequence.
    """
    starts = np.concatenate([[0], np.where(path[1:] != path[:-1])[0] + 1])
    left = positions[starts].copy()
    left[0] = 0
    right = np.append(left[1:], sequence_length)
    parent = path[starts].astype(np.int32)
    return left, right, parent


def path_haplotype(ts, left, right, parent):
    """
    Return the haplotype obtained by copying the specified path through the
    specified tree sequence.
    """
    nodes, index = np.unique(parent, return_inverse=True)
    G = ts.genotype_matrix(samples=nodes, isolated_as_missing=False, alleles=ALLELES)
    segment = np.searchsorted(left, ts.sites_position, side="right") - 1
    assert np.all(ts.sites_position < right[segment])
    return G[np.arange(ts.num_sites), index[segment]].astype(np.int8)


class AncestorMatcherEngine:
    """
    Base class for the engines with tsinfer's AncestorMatcher interface,
    which differ only in how the shared reference (tsinfer's MatcherIndexes
    or a tsls TreeSequence) and the per-query matcher are made.

    If mismatch_threshold is specified, normalised likelihoods are not
    allowed to drop below that of mismatch_threshold mismatches, as in
    sc2ts. This bounds the number of distinct likelihood values, and so
    the time and memory needed, at the cost of exactness.

    By default, states are encoded as indexes into ALLELES. If alleles is
    None, the ancestral state of every site is 0 and every mutation has
    derived state 1, as in tsinfer (tsls numbers the alleles at each site
    as tskit does, which is the same for tsinfer's data). weight_by_n is
    False as in sc2ts; tsinfer uses True.
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
        self.weight_by_n = weight_by_n
        self.likelihood_threshold = None
        if mismatch_threshold is not None:
            self.likelihood_threshold = mu**mismatch_threshold * 0.99
        self.recombination = np.full(ts.num_sites, rho)
        self.mismatch = np.full(ts.num_sites, mu)
        self.alleles = alleles
        self.reference = self.make_reference(ts, alleles)

    def make_reference(self, ts, alleles):
        raise NotImplementedError

    def make_matcher(self):
        raise NotImplementedError

    def match(self, h, start=0, end=None):
        """
        Match the specified haplotype over the sites from start to end
        (exclusive). The path covers only these sites.
        """
        if end is None:
            end = self.num_sites
        matcher = self.make_matcher()
        matched = np.zeros(self.num_sites, dtype=np.int8)
        left, right, parent = matcher.find_path(h.astype(np.int8), start, end, matched)
        stats = {
            "total_memory": matcher.total_memory,
            "mean_traceback_size": matcher.mean_traceback_size,
        }
        return (
            left[::-1].astype(np.float64),
            right[::-1].astype(np.float64),
            parent[::-1].copy(),
            matched,
            stats,
        )


class TsinferEngine(AncestorMatcherEngine):
    name = "tsinfer"

    def make_reference(self, ts, alleles):
        return tsinfer_hmm.MatcherIndexes(ts, vestigial_root=False, alleles=alleles)

    def make_matcher(self):
        return tsinfer_hmm.AncestorMatcher(
            self.reference,
            self.recombination,
            self.mismatch,
            likelihood_threshold=self.likelihood_threshold,
            weight_by_n=self.weight_by_n,
        )


class TskitEngine:
    name = "tskit"

    def __init__(self, ts, *, mu, rho):
        self.positions = ts.sites_position.copy()
        self.sequence_length = ts.sequence_length
        self.num_samples = ts.num_samples
        self.mu = mu
        self.unweighted_rho = rho
        self.rho = n_weighted_rho(rho, ts.num_samples)
        self.ll_ts = tskit_hmm.ll_tree_sequence(ts)
        self.ls_hmm = _tskit_hmm.LsHmm(
            self.ll_ts,
            np.full(ts.num_sites, self.rho),
            np.full(ts.num_sites, mu),
            acgt_alleles=1,
        )

    def match(self, h):
        vm = _tskit_hmm.ViterbiMatrix(self.ll_ts)
        self.ls_hmm.viterbi_matrix(h.astype(np.int32), vm)
        path = vm.traceback()
        num_transitions = vm.num_transitions
        stats = {
            "total_transitions": int(np.sum(num_transitions)),
            "max_transitions": int(np.max(num_transitions)),
            "optimum_log_likelihood": self.optimum_log_likelihood(
                vm.normalisation_factor
            ),
        }
        left, right, parent = path_to_segments(
            path, self.positions, self.sequence_length
        )
        return left, right, parent, None, stats

    def optimum_log_likelihood(self, normalisation_factor):
        """
        Return the log likelihood of the optimal path found in the forward
        pass, in the same units as path_log_likelihood. Because values are
        normalised by the maximum at each site, the product of the
        normalisation factors is the likelihood of the best path, which
        includes an initial factor of 1 / n and a factor of
        (1 - rho + rho / n) for staying on the same node at every site.
        Comparing this with the likelihood of the traceback path detects
        suboptimal tracebacks.
        """
        n = self.num_samples
        m = len(normalisation_factor)
        stay = 1 - self.rho + self.rho / n
        return float(np.sum(np.log(normalisation_factor)) + np.log(n) - m * np.log(stay))


class TslsEngine(AncestorMatcherEngine):
    """
    The merged tsls engine, with the same model and options as tsinfer. In
    tsls every node in the tree is a state, so we remove the vestigial root
    edge 0 -> 1 (added by prepare.py, and present in tsinfer's trees) to
    make node 0 isolated, as in sc2ts.
    """

    name = "tsls"

    def make_reference(self, ts, alleles):
        tables = ts.dump_tables()
        tables.edges.keep_rows(tables.edges.parent != 0)
        return tsls.TreeSequence(tables.tree_sequence())

    def make_matcher(self):
        return tsls.Hmm(
            self.reference,
            self.recombination,
            self.mismatch,
            alleles=self.alleles,
            likelihood_threshold=self.likelihood_threshold,
            weight_by_n=self.weight_by_n,
        )


ENGINES = {engine.name: engine for engine in [TsinferEngine, TskitEngine, TslsEngine]}
