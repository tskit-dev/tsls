"""
Tests for the likelihood model (docs/likelihood_model.md): the conversions
in tsls.model, the odds form of Viterbi, the equivalence with the model
that does not divide by n, and the Viterbi prune thresholds.
"""

import numpy as np
import pytest
import tskit
from test_matching import (
    ALLELES,
    EXAMPLES,
    finalise,
    noisy_queries,
    path_by_site,
    queries,
    wright_fisher_ts,
)

import dense
import tsls

A_NUM = len(ALLELES)


def viterbi_costs(A, present, continuous, h, rho, mu, num_alleles, n):
    """
    Return the (num_sites, num_nodes) array of the cost of the best path to
    each node at each site, computed from the odds alone: a switch costs
    -log q and a mismatch -log e. Nodes not in the tree have infinite cost.
    """
    m, N = A.shape
    log_q = np.log(tsls.model.switch_odds(rho, n))
    log_e = np.log(tsls.model.mismatch_odds(mu, num_alleles))
    C = np.full((m, N), np.inf)
    c = np.where(present[0], 0.0, np.inf)
    for j in range(m):
        if j > 0:
            stay = np.where(continuous[j], c, np.inf)
            c = np.minimum(stay, np.min(c) - log_q[j])
        match = (A[j] == h[j]) | (h[j] == -1)
        c = c - np.where(match, 0, log_e[j])
        c = np.where(present[j], c, np.inf)
        C[j] = c
    return C


def path_log_odds(continuous, h, matched, path, rho, mu, num_alleles, n):
    """
    Return the log of the odds of a path relative to staying on a node that
    matches at every site, and its numbers of mismatches and switches.
    """
    m = len(path)
    switch = np.zeros(m, dtype=bool)
    switch[1:] = (path[1:] != path[:-1]) | ~continuous[np.arange(1, m), path[1:]]
    mismatch = (h != -1) & (h != matched)
    log_q = np.log(tsls.model.switch_odds(rho, n))
    log_e = np.log(tsls.model.mismatch_odds(mu, num_alleles))
    value = np.sum(log_q[switch]) + np.sum(log_e[mismatch])
    return value, int(np.sum(mismatch)), int(np.sum(switch))


class TestConversions:
    @pytest.mark.parametrize("q", [0, 1e-200, 1e-6, 0.3, 1])
    @pytest.mark.parametrize("n", [1, 7, 10**7])
    def test_round_trip(self, q, n):
        r = tsls.model.recombination_probability(q, n)
        assert 0 <= r <= 1
        # 1 - r loses precision as r approaches 1
        rel = 1e-12 + 1e-15 / (1 - r) if r < 1 else 1e-12
        assert tsls.model.switch_odds(r, n) == pytest.approx(q, rel=rel)

    def test_n_one(self):
        # With n = 1, the odds are the probability
        assert tsls.model.recombination_probability(0.01, 1) == pytest.approx(0.01)

    @pytest.mark.parametrize("q", [-0.1, 1.5])
    def test_bad_odds(self, q):
        with pytest.raises(ValueError, match="between 0 and 1"):
            tsls.model.recombination_probability(q, 10)

    def test_mismatch_odds(self):
        assert tsls.model.mismatch_odds(0.01, 5) == pytest.approx(0.01 / 0.96)

    def test_path_threshold_constant(self):
        e = tsls.model.mismatch_odds(0.01, 5)
        q = tsls.model.switch_odds(0.02, 100)
        tol = tsls.model.THRESHOLD_TOLERANCE
        assert tsls.model.path_threshold(0, 0.01, 5) == 1 - tol
        assert tsls.model.path_threshold(3, 0.01, 5) == pytest.approx(
            e**3 * (1 - tol), rel=1e-14
        )
        t = tsls.model.path_threshold(
            2, 0.01, 5, num_switches=1, recombination=0.02, num_nodes=100
        )
        assert t == pytest.approx(e**2 * q * (1 - tol), rel=1e-14)

    def test_path_threshold_per_site(self):
        # The k most costly sites: the smallest odds
        mu = np.array([0.01, 0.001, 0.02, 0.005])
        num_alleles = np.array([2, 5, 5, 3])
        e = tsls.model.mismatch_odds(mu, num_alleles)
        t = tsls.model.path_threshold(2, mu, num_alleles, tolerance=0)
        assert t == pytest.approx(np.prod(np.sort(e)[:2]))

    def test_path_threshold_errors(self):
        with pytest.raises(ValueError, match="recombination"):
            tsls.model.path_threshold(1, 0.01, 5, num_switches=1)
        with pytest.raises(ValueError, match="More events"):
            tsls.model.path_threshold(3, [0.01, 0.01], 5)
        with pytest.raises(ValueError, match=">= 0"):
            tsls.model.path_threshold(-1, 0.01, 5)


@pytest.mark.parametrize("name", list(EXAMPLES))
@pytest.mark.parametrize("q", [1e-3, 0.1])
class TestModelEquivalence:
    """
    The model that does not divide by n (n = 1), with recombination q, is the
    model with r = recombination_probability(q, n) up to a constant factor
    at each site.
    """

    def test_dense(self, name, q):
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        n = ts.num_nodes
        q_a = np.full(m, q)
        r_a = tsls.model.recombination_probability(q_a, n)
        mu = np.full(m, 0.02)
        for h in queries(ts, A, present):
            args = (A, present, continuous, h)
            F1, c1 = dense.forward(*args, q_a, mu, A_NUM, 1)
            F, c = dense.forward(*args, r_a, mu, A_NUM, n)
            np.testing.assert_allclose(F1, F, rtol=1e-10, atol=1e-300)
            # The factor is (1 - q) / (1 - r) at each site, and n at the
            # first site for the uniform start.
            factor = (1 - q_a) / (1 - r_a)
            factor[0] *= n
            np.testing.assert_allclose(c1 / c, factor, rtol=1e-10)
            B1 = dense.backward(*args, q_a, mu, A_NUM, 1, c1)
            B = dense.backward(*args, r_a, mu, A_NUM, n, c)
            np.testing.assert_allclose(B1, B, rtol=1e-10, atol=1e-300)
            V1, _ = dense.viterbi_matrix(*args, q_a, mu, A_NUM, 1)
            V, _ = dense.viterbi_matrix(*args, r_a, mu, A_NUM, n)
            np.testing.assert_allclose(V1, V, rtol=1e-10, atol=1e-300)

    def test_engine_paths(self, name, q):
        # The engine with r finds paths that are optimal in the n = 1 model
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        q_a = np.full(m, q)
        mu = np.full(m, 0.02)
        r = tsls.model.recombination_probability(q_a, ts.num_nodes)
        am = tsls.Hmm(ts, r, mu, alleles=ALLELES, extended_checks=True)
        for h in queries(ts, A, present):
            match = np.zeros(m, dtype=np.int8)
            path = path_by_site(ts, *am.find_path(h, 0, m, match))
            args = (A, present, continuous, h, q_a, mu, A_NUM, 1)
            value = dense.path_log_likelihood(*args, path)
            assert value == pytest.approx(dense.viterbi_log_likelihood(*args), rel=1e-9)


@pytest.mark.parametrize("name", list(EXAMPLES))
@pytest.mark.parametrize(("rho", "mu"), [(0.05, 0.02), (1e-4, 1e-2)])
def test_viterbi_values_are_odds(name, rho, mu):
    # The normalised Viterbi value of each node is exp(-(C - min C)), where C
    # is the cost of the best path to it in the odds
    ts = EXAMPLES[name]()
    A, present, continuous = dense.node_states(ts, ALLELES)
    m = ts.num_sites
    n = ts.num_nodes
    rho_a = np.full(m, rho)
    mu_a = np.full(m, mu)
    for h in queries(ts, A, present):
        args = (A, present, continuous, h, rho_a, mu_a, A_NUM, n)
        V, _ = dense.viterbi_matrix(*args)
        C = viterbi_costs(*args)
        expected = np.exp(-(C - np.min(C, axis=1)[:, None]))
        np.testing.assert_allclose(V, expected, rtol=1e-9, atol=1e-300)


class TestPruneThreshold:
    def run(self, ts, h, rho, mu, threshold, sparse=True):
        m = ts.num_sites
        am = tsls.Hmm(
            ts,
            rho,
            mu,
            alleles=ALLELES,
            prune_threshold=threshold,
            extended_checks=True,
            sparse=sparse,
        )
        match = np.zeros(m, dtype=np.int8)
        path = path_by_site(ts, *am.find_path(h, 0, m, match))
        return path, match, am

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("num_trees_hint", [0, 3])
    @pytest.mark.parametrize("sparse", [True, False])
    def test_path_threshold_is_exact(self, seed, num_trees_hint, sparse):
        # The threshold for the numbers of mismatches and switches of an
        # optimal path finds an optimal path.
        ts = wright_fisher_ts(seed, num_trees_hint=num_trees_hint)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        n = ts.num_nodes
        rho = tsls.model.recombination_probability(np.full(m, 1e-5), n)
        mu = np.full(m, 0.01)
        num_sparse_sites = 0
        for h in noisy_queries(ts, A, present, seed):
            args = (A, present, continuous, h, rho, mu, A_NUM, n)
            optimum = dense.viterbi_log_likelihood(*args)
            path, match, _ = self.run(ts, h, rho, mu, 0, sparse)
            _, k, s = path_log_odds(continuous, h, match, path, rho, mu, A_NUM, n)
            t = tsls.model.path_threshold(
                k, mu, A_NUM, num_switches=s, recombination=rho, num_nodes=n
            )
            path, match, am = self.run(ts, h, rho, mu, t, sparse)
            num_sparse_sites += am.num_sparse_sites
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            value = dense.path_log_likelihood(*args, path)
            assert value == pytest.approx(optimum, rel=1e-9)
        if sparse and num_trees_hint == 0:
            assert num_sparse_sites > 0

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("sparse", [True, False])
    def test_certificate(self, seed, sparse):
        # Whatever the threshold, if the path found has odds of at least the
        # threshold, it is optimal.
        ts = wright_fisher_ts(seed, num_trees_hint=2)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        n = ts.num_nodes
        rho = tsls.model.recombination_probability(np.full(m, 1e-4), n)
        mu = np.full(m, 0.01)
        num_certified = 0
        for h in noisy_queries(ts, A, present, seed):
            args = (A, present, continuous, h, rho, mu, A_NUM, n)
            optimum = dense.viterbi_log_likelihood(*args)
            for k in range(6):
                t = tsls.model.path_threshold(k, mu, A_NUM)
                path, match, _ = self.run(ts, h, rho, mu, t, sparse)
                log_odds, _, _ = path_log_odds(
                    continuous, h, match, path, rho, mu, A_NUM, n
                )
                if log_odds >= np.log(t):
                    num_certified += 1
                    value = dense.path_log_likelihood(*args, path)
                    assert value == pytest.approx(optimum, rel=1e-9)
        assert num_certified > 0

    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("sparse", [True, False])
    def test_threshold_is_tight(self, k, sparse):
        # Node 0 mismatches at the first k sites, where node 1 matches, and
        # then matches at every site, where node 1 and the root mismatch. So
        # the optimal path copies node 0 throughout, and falls k mismatches
        # behind the best node at site k - 1. Any larger threshold prunes it,
        # and switching to node 0 costs more than k mismatches.
        m = 10
        tables = tskit.TableCollection(m)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.edges.add_row(0, m, 2, 0)
        tables.edges.add_row(0, m, 2, 1)
        for j in range(m):
            tables.sites.add_row(position=j, ancestral_state="A")
            tables.mutations.add_row(site=j, node=1 if j < k else 0, derived_state="C")
        ts = finalise(tables)
        h = np.full(m, ALLELES.index("C"), dtype=np.int8)
        n = ts.num_nodes
        mu = np.full(m, 0.01)
        e = tsls.model.mismatch_odds(0.01, A_NUM)
        # A switch costs one more mismatch than k
        rho = tsls.model.recombination_probability(np.full(m, e ** (k + 1)), n)

        t = tsls.model.path_threshold(k, mu, A_NUM)
        path, match, _ = self.run(ts, h, rho, mu, t, sparse)
        assert list(path) == [0] * m
        path, match, _ = self.run(ts, h, rho, mu, e**k * (1 + 1e-6), sparse)
        assert list(path) != [0] * m

    def test_zero_is_unpruned(self):
        ts = wright_fisher_ts(0, num_trees_hint=2)
        A, present, _ = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = np.full(m, 0.02)
        mu = np.full(m, 0.01)
        for h in noisy_queries(ts, A, present, 0):
            path1, match1, _ = self.run(ts, h, rho, mu, 0)
            am = tsls.Hmm(ts, rho, mu, alleles=ALLELES)
            match2 = np.zeros(m, dtype=np.int8)
            path2 = path_by_site(ts, *am.find_path(h, 0, m, match2))
            np.testing.assert_array_equal(path1, path2)
            np.testing.assert_array_equal(match1, match2)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan])
    def test_bad_threshold(self, threshold):
        ts = EXAMPLES["single_tree"]()
        m = ts.num_sites
        with pytest.raises(ValueError, match="prune threshold"):
            tsls.Hmm(
                ts,
                np.full(m, 0.01),
                np.full(m, 0.01),
                alleles=ALLELES,
                prune_threshold=threshold,
            )


def test_weight_by_n_removed():
    ts = EXAMPLES["single_tree"]()
    m = ts.num_sites
    with pytest.raises(ValueError, match="recombination_probability"):
        tsls.Hmm(ts, np.full(m, 0.01), np.full(m, 0.01), weight_by_n=False)
