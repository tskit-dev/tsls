"""
Tests for the tsls HMM engine on small hand-built examples, compared with
the dense implementation in dense.py.
"""

import msprime
import numpy as np
import pytest
import tskit

import dense
import tsls

ACGT = ["A", "C", "G", "T"]


def finalise(tables):
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def add_mutations(tables, mutations):
    """
    Add sites and mutations from a list of (position, ancestral_state,
    [(node, derived_state), ...]) tuples, with mutations on the same node
    listed oldest first.
    """
    for position, ancestral_state, site_mutations in mutations:
        site = tables.sites.add_row(position=position, ancestral_state=ancestral_state)
        for node, derived_state in site_mutations:
            tables.mutations.add_row(site=site, node=node, derived_state=derived_state)


def single_tree_ts():
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         8
    tables = tskit.Tree.generate_balanced(4, span=8).tree_sequence.dump_tables()
    add_mutations(
        tables,
        [
            # Recurrent mutations and a non-zero ancestral state
            (1, "G", [(0, "T"), (3, "T")]),
            # A back mutation
            (3, "A", [(4, "C"), (1, "A")]),
            # Three different derived states
            (5, "T", [(0, "A"), (2, "C"), (3, "G")]),
            # Stacked mutations on the same node, and one on the root
            (7, "C", [(6, "G"), (2, "A"), (2, "T")]),
        ],
    )
    return finalise(tables)


def two_tree_ts():
    # Node 4 is only in the left tree and node 7 only in the right tree.
    #
    # 3.00┊     6   ┊    6    ┊
    #     ┊   ┏━┻┓  ┊   ┏┻━┓  ┊
    # 2.00┊   5  ┃  ┊   5  ┃  ┊
    #     ┊  ┏┻┓ ┃  ┊ ┏━┻┓ ┃  ┊
    # 1.50┊  ┃ ┃ ┃  ┊ ┃  7 ┃  ┊
    #     ┊  ┃ ┃ ┃  ┊ ┃ ┏┻┓┃  ┊
    # 1.00┊  4 ┃ ┃  ┊ ┃ ┃ ┃┃  ┊
    #     ┊ ┏┻┓┃ ┃  ┊ ┃ ┃ ┃┃  ┊
    # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊
    #     0        10        20
    tables = tskit.TableCollection(sequence_length=20)
    for t in [0, 0, 0, 0, 1, 2, 3, 1.5]:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if t == 0 else 0, time=t)
    for left, right, parent, child in [
        (0, 10, 4, 0),
        (0, 10, 4, 1),
        (0, 10, 5, 4),
        (0, 10, 5, 2),
        (0, 20, 6, 5),
        (0, 20, 6, 3),
        (10, 20, 7, 1),
        (10, 20, 7, 2),
        (10, 20, 5, 0),
        (10, 20, 5, 7),
    ]:
        tables.edges.add_row(left, right, parent, child)
    add_mutations(
        tables,
        [
            (2, "A", [(4, "C")]),
            (5, "A", [(0, "G"), (2, "T")]),
            (8, "A", [(5, "C"), (1, "A")]),
            (12, "T", [(7, "G")]),
            (15, "C", [(0, "T"), (3, "T")]),
            (18, "G", [(6, "C")]),
        ],
    )
    return finalise(tables)


def forest_ts():
    # Two roots on [0, 5), joined by node 6 on [5, 10). Node 4 is a unary
    # root on [0, 5), node 5 is isolated, and node 6 is only in the right
    # tree.
    #
    # 3.00┊         ┊    6    ┊
    #     ┊         ┊  ┏━┻━┓  ┊
    # 2.00┊       4 ┊  ┃   4  ┊
    #     ┊       ┃ ┊  ┃   ┃  ┊
    # 1.00┊   3   ┃ ┊  3   ┃  ┊
    #     ┊  ┏┻┓  ┃ ┊ ┏┻┓  ┃  ┊
    # 0.00┊  0 1  2 ┊ 0 1  2  ┊
    #     0         5        10
    tables = tskit.TableCollection(sequence_length=10)
    for t in [0, 0, 0, 1, 2, 4, 3]:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if t == 0 else 0, time=t)
    for left, right, parent, child in [
        (0, 10, 3, 0),
        (0, 10, 3, 1),
        (0, 10, 4, 2),
        (5, 10, 6, 3),
        (5, 10, 6, 4),
    ]:
        tables.edges.add_row(left, right, parent, child)
    add_mutations(
        tables,
        [
            (1, "A", [(3, "C"), (2, "G")]),
            (3, "C", [(4, "A")]),
            (4, "T", []),
            (6, "G", [(0, "A"), (4, "T")]),
            (8, "A", [(6, "T"), (1, "A")]),
        ],
    )
    return finalise(tables)


def isolated_root_ts():
    # The sc2ts convention: node 0 is an isolated vestigial root above the
    # root node 1 (in tsinfer, node 0 is joined to node 1 by an edge, which
    # is removed before matching). Nodes 1, 2 and 4 are internal samples.
    #
    # 3.00┊    1    ┊    1    ┊
    #     ┊  ┏━┻━┓  ┊  ┏━╋━┓  ┊
    # 2.00┊  2   ┃  ┊  2 ┃ ┃  ┊
    #     ┊ ┏┻┓  ┃  ┊  ┃ ┃ ┃  ┊
    # 1.00┊ 3 4  ┃  ┊  3 4 ┃  ┊
    #     ┊   ┃  ┃  ┊    ┃ ┃  ┊
    # 0.00┊   5  6  ┊    5 6  ┊
    #     0         6        12
    tables = tskit.TableCollection(sequence_length=12)
    flags = tskit.NODE_IS_SAMPLE
    for f, t in [(0, 4), (flags, 3), (flags, 2), (flags, 1), (flags, 1)]:
        tables.nodes.add_row(flags=f, time=t)
    tables.nodes.add_row(flags=flags, time=0)
    tables.nodes.add_row(flags=flags, time=0)
    for left, right, parent, child in [
        (0, 12, 1, 2),
        (0, 12, 1, 6),
        (0, 12, 2, 3),
        (0, 6, 2, 4),
        (6, 12, 1, 4),
        (0, 12, 4, 5),
    ]:
        tables.edges.add_row(left, right, parent, child)
    add_mutations(
        tables,
        [
            (1, "A", [(2, "C")]),
            (3, "G", [(4, "T"), (6, "T")]),
            (5, "C", [(3, "A")]),
            (7, "T", [(5, "G")]),
            (9, "A", [(4, "-")]),
            (11, "C", [(1, "G"), (6, "C")]),
        ],
    )
    return finalise(tables)


def gap_ts():
    # Node 3 is in the left and right trees but not in the middle tree, which
    # has no sites, so copying node 3 across the middle tree requires a
    # recombination. Node 4 is only in the middle tree, and so is never a
    # state at a site.
    #
    # 2.00┊    5    ┊    5    ┊    5    ┊
    #     ┊  ┏━┻┓   ┊  ┏━┻┓   ┊  ┏━┻┓   ┊
    # 1.50┊  ┃  ┃   ┊  4  ┃   ┊  ┃  ┃   ┊
    #     ┊  ┃  ┃   ┊ ┏┻┓ ┃   ┊  ┃  ┃   ┊
    # 1.00┊  3  ┃   ┊ ┃ ┃ ┃   ┊  3  ┃   ┊
    #     ┊ ┏┻┓ ┃   ┊ ┃ ┃ ┃   ┊ ┏┻┓ ┃   ┊
    # 0.00┊ 0 1 2   ┊ 0 1 2   ┊ 0 1 2   ┊
    #     0         4         8        12
    tables = tskit.TableCollection(sequence_length=12)
    for t in [0, 0, 0, 1, 1.5, 2]:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if t == 0 else 0, time=t)
    for left, right, parent, child in [
        (0, 4, 3, 0),
        (0, 4, 3, 1),
        (8, 12, 3, 0),
        (8, 12, 3, 1),
        (4, 8, 4, 0),
        (4, 8, 4, 1),
        (0, 4, 5, 3),
        (8, 12, 5, 3),
        (4, 8, 5, 4),
        (0, 12, 5, 2),
    ]:
        tables.edges.add_row(left, right, parent, child)
    add_mutations(
        tables,
        [
            (1, "A", [(3, "C")]),
            (3, "G", [(0, "T"), (2, "A")]),
            (9, "T", [(3, "G"), (1, "A")]),
            (11, "C", [(5, "A"), (0, "G")]),
        ],
    )
    return finalise(tables)


EXAMPLES = {
    "single_tree": single_tree_ts,
    "two_trees": two_tree_ts,
    "forest": forest_ts,
    "isolated_root": isolated_root_ts,
    "gap": gap_ts,
}
ALLELES = ACGT + ["-"]
PARAMS = [(1e-2, 1e-2), (0.2, 0.05), (1e-4, 1e-3)]
# How the recombination parameter is given: as the probability r, or as the
# switch odds q (the recombination of an engine that does not divide by n,
# as sc2ts uses)
RATES = ["probability", "odds"]


def model_rho(ts, rho, rate):
    """
    Return the recombination probability for the specified parameter.
    """
    if rate == "probability":
        return rho
    return tsls.model.recombination_probability(rho, ts.num_nodes)


def queries(ts, A, present):
    """
    Return a list of query haplotypes: the haplotypes of all nodes that are
    in every tree, plus some random haplotypes, with and without missing
    data.
    """
    rng = np.random.default_rng(42)
    H = [A[:, u] for u in range(ts.num_nodes) if np.all(present[:, u])]
    num_alleles = len(ALLELES)
    for _ in range(3):
        H.append(rng.integers(0, num_alleles, size=ts.num_sites).astype(np.int8))
    for h in list(H[:3]):
        h = h.copy()
        h[rng.integers(0, ts.num_sites)] = -1
        H.append(h)
    h = H[0].copy()
    h[0] = -1
    h[-1] = -1
    H.append(h)
    return [np.asarray(h, dtype=np.int8) for h in H]


def make_matcher(ts, rho, mu, likelihood_threshold=None, prune_threshold=None):
    m = ts.num_sites
    am = tsls.Hmm(
        ts,
        np.full(m, rho),
        np.full(m, mu),
        alleles=ALLELES,
        likelihood_threshold=likelihood_threshold,
        prune_threshold=prune_threshold,
        extended_checks=True,
    )
    return am, ts.num_nodes


def path_by_site(ts, left, right, parent, start=0, end=None):
    """
    Return the node copied at each site in [start, end), from a path in any
    order.
    """
    if end is None:
        end = ts.num_sites
    index = np.argsort(left)
    left, right, parent = left[index], right[index], parent[index]
    position = ts.sites_position
    assert left[0] == (0 if start == 0 else position[start])
    assert right[-1] == (ts.sequence_length if end == ts.num_sites else position[end])
    np.testing.assert_array_equal(left[1:], right[:-1])
    segment = np.searchsorted(left, position[start:end], side="right") - 1
    return parent[segment]


@pytest.fixture(scope="module", params=list(EXAMPLES))
def example(request):
    ts = EXAMPLES[request.param]()
    A, present, continuous = dense.node_states(ts, ALLELES)
    return ts, A, present, continuous


class TestExamples:
    def test_node_states(self, example):
        ts, A, present, continuous = example
        G = ts.genotype_matrix(isolated_as_missing=False, alleles=tuple(ALLELES))
        np.testing.assert_array_equal(A[:, ts.samples()], G)
        assert np.any(np.bincount(ts.mutations_site) > 1)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize(("rho", "mu"), PARAMS)
class TestAgainstDense:
    def test_viterbi(self, example, rate, rho, mu):
        ts, A, present, continuous = example
        rho = model_rho(ts, rho, rate)
        am, n = make_matcher(ts, rho, mu)
        m = ts.num_sites
        rho_a = np.full(m, rho)
        mu_a = np.full(m, mu)
        for h in queries(ts, A, present):
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            assert np.all(present[np.arange(m), path])
            value = dense.path_log_likelihood(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n, path
            )
            optimum = dense.viterbi_log_likelihood(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n
            )
            assert value == pytest.approx(optimum, rel=1e-9)

    def test_forward_backward(self, example, rate, rho, mu):
        ts, A, present, continuous = example
        rho = model_rho(ts, rho, rate)
        am, n = make_matcher(ts, rho, mu)
        m = ts.num_sites
        rho_a = np.full(m, rho)
        mu_a = np.full(m, mu)
        for h in queries(ts, A, present):
            fm = am.forward_matrix(h)
            F, c = dense.forward(A, present, continuous, h, rho_a, mu_a, len(ALLELES), n)
            np.testing.assert_allclose(fm.normalisation_factor, c, rtol=1e-12)
            np.testing.assert_allclose(fm.decode(), F, rtol=1e-12, atol=1e-300)
            assert fm.log_likelihood == pytest.approx(np.sum(np.log(c)))

            bm = am.backward_matrix(h, fm.normalisation_factor)
            B = dense.backward(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n, c
            )
            np.testing.assert_allclose(bm.decode(), B, rtol=1e-12, atol=1e-300)
            np.testing.assert_allclose(np.sum(F * B, axis=1), 1)

    def test_likelihood_threshold(self, example, rate, rho, mu):
        ts, A, present, continuous = example
        threshold = 1e-3
        rho = model_rho(ts, rho, rate)
        am, n = make_matcher(ts, rho, mu, likelihood_threshold=threshold)
        m = ts.num_sites
        rho_a = np.full(m, rho)
        mu_a = np.full(m, mu)
        for h in queries(ts, A, present):
            fm = am.forward_matrix(h)
            F, c = dense.forward(
                A,
                present,
                continuous,
                h,
                rho_a,
                mu_a,
                len(ALLELES),
                n,
                threshold=threshold,
            )
            np.testing.assert_allclose(fm.decode(), F, rtol=1e-12)
            bm = am.backward_matrix(h, fm.normalisation_factor)
            B = dense.backward(
                A,
                present,
                continuous,
                h,
                rho_a,
                mu_a,
                len(ALLELES),
                n,
                c,
                threshold=threshold,
            )
            np.testing.assert_allclose(bm.decode(), B, rtol=1e-12)
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])


class TestViterbiPaths:
    def test_gap_requires_recombination(self):
        # Copying node 3 across the middle tree, where it is not in the tree,
        # needs two switches, so a node in every tree is preferred.
        ts = gap_ts()
        A, _, _ = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 1e-4, 1e-2)
        h = A[:, 3]
        match = np.zeros(ts.num_sites, dtype=np.int8)
        _, _, parent = am.find_path(h, 0, ts.num_sites, match)
        assert 3 not in parent
        # With a high recombination rate, we copy node 3 in two segments.
        am, _ = make_matcher(ts, 0.3, 1e-3)
        left, right, parent = am.find_path(h, 0, ts.num_sites, match)
        # Breakpoints are at site positions: the first site after the gap.
        assert list(parent) == [3, 3]
        assert list(left) == [9, 0]
        np.testing.assert_array_equal(match, h)

    @pytest.mark.parametrize("name", list(EXAMPLES))
    def test_exact_copy(self, name):
        # Each node that is in every tree and has a unique haplotype is
        # copied exactly with no recombination.
        ts = EXAMPLES[name]()
        A, _, continuous = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 1e-4, 1e-3)
        m = ts.num_sites
        always = np.all(continuous, axis=0)
        for u in range(ts.num_nodes):
            h = A[:, u]
            if not always[u]:
                continue
            num_same = np.sum(np.all(A == h[:, None], axis=0) & always)
            if num_same > 1:
                continue
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            assert list(parent) == [u]
            assert list(left) == [0]
            assert list(right) == [ts.sequence_length]
            np.testing.assert_array_equal(match, h)

    def test_two_tree_recombinant(self):
        # Copy node 4 (only in the left tree) then node 7 (only in the right)
        ts = two_tree_ts()
        A, _, _ = dense.node_states(ts, ALLELES)
        h = np.concatenate([A[:3, 4], A[3:, 7]])
        am, _ = make_matcher(ts, 1e-2, 1e-3)
        match = np.zeros(ts.num_sites, dtype=np.int8)
        left, right, parent = am.find_path(h, 0, ts.num_sites, match)
        np.testing.assert_array_equal(match, h)
        # The path is returned right-to-left
        assert list(parent) == [7, 4]
        assert list(right) == [20, 12]
        assert list(left) == [12, 0]

    def test_isolated_node_never_copied(self):
        ts = isolated_root_ts()
        A, present, _ = dense.node_states(ts, ALLELES)
        assert not np.any(present[:, 0])
        am, _ = make_matcher(ts, 0.2, 0.05)
        for h in queries(ts, A, present):
            match = np.zeros(ts.num_sites, dtype=np.int8)
            _, _, parent = am.find_path(h, 0, ts.num_sites, match)
            assert 0 not in parent

    def test_start_end(self):
        ts = two_tree_ts()
        A, _, _ = dense.node_states(ts, ALLELES)
        h = A[:, 0].copy()
        h[0] = -1
        h[-1] = -1
        am, _ = make_matcher(ts, 1e-4, 1e-3)
        match = np.zeros(ts.num_sites, dtype=np.int8)
        left, right, parent = am.find_path(h, 1, ts.num_sites - 1, match)
        assert list(parent) == [0]
        assert list(left) == [ts.sites_position[1]]
        assert list(right) == [ts.sites_position[-1]]
        np.testing.assert_array_equal(match[1:-1], h[1:-1])

    @pytest.mark.parametrize("name", list(EXAMPLES))
    def test_windows(self, name):
        # The first tree of a window is found by seeking from whichever end
        # of the sequence is closer, so check windows starting and ending
        # throughout the sequence against the dense optimum for the window.
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        rho, mu = 0.05, 0.02
        am, n = make_matcher(ts, rho, mu)
        m = ts.num_sites
        for h in queries(ts, A, present)[:4]:
            for start in range(m):
                for end in range(start + 1, m + 1):
                    match = np.zeros(m, dtype=np.int8)
                    left, right, parent = am.find_path(h, start, end, match)
                    path = path_by_site(ts, left, right, parent, start, end)
                    sites = np.arange(start, end)
                    np.testing.assert_array_equal(match[start:end], A[sites, path])
                    window = (A[start:end], present[start:end])
                    cont = continuous[start:end].copy()
                    cont[0] = present[start]
                    rho_a = np.full(end - start, rho)
                    mu_a = np.full(end - start, mu)
                    args = (*window, cont, h[start:end], rho_a, mu_a, len(ALLELES), n)
                    value = dense.path_log_likelihood(*args, path)
                    optimum = dense.viterbi_log_likelihood(*args)
                    assert value == pytest.approx(optimum, rel=1e-9)


class TestInterface:
    def test_unknown_allele(self):
        ts = single_tree_ts()
        m = ts.num_sites
        with pytest.raises(tsls.LibraryError, match="not in the list"):
            tsls.Hmm(ts, np.zeros(m), np.zeros(m), alleles=["A", "C"])

    def test_duplicate_alleles(self):
        ts = single_tree_ts()
        m = ts.num_sites
        with pytest.raises(ValueError, match="unique"):
            tsls.Hmm(ts, np.zeros(m), np.zeros(m), alleles=["A", "A"])

    def test_default_binary_alleles(self):
        tables = tskit.Tree.generate_balanced(2, span=4).tree_sequence.dump_tables()
        add_mutations(tables, [(1, "0", [(0, "1")])])
        ts = finalise(tables)
        am = tsls.Hmm(ts, np.full(1, 1e-3), np.full(1, 1e-3))
        match = np.zeros(1, dtype=np.int8)
        _, _, parent = am.find_path(np.array([1], dtype=np.int8), 0, 1, match)
        assert list(parent) == [0]

    @pytest.mark.parametrize("name", list(EXAMPLES))
    def test_per_site_alleles(self, name):
        # By default the alleles are numbered at each site as tskit does, so
        # the matched haplotypes are tskit's genotypes.
        ts = EXAMPLES[name]()
        m = ts.num_sites
        am = tsls.Hmm(ts, np.full(m, 1e-4), np.full(m, 1e-3), extended_checks=True)
        G = ts.genotype_matrix(isolated_as_missing=False)
        for j in range(ts.num_samples):
            h = G[:, j].astype(np.int8)
            match = np.zeros(m, dtype=np.int8)
            am.find_path(h, 0, m, match)
            np.testing.assert_array_equal(match, h)

    def test_site_without_mutations(self):
        # As in tsinfer's ancestor matching, a haplotype can carry an allele
        # at a site that has no mutations in the tree sequence.
        tables = tskit.Tree.generate_balanced(2, span=4).tree_sequence.dump_tables()
        add_mutations(tables, [(1, "0", [(0, "1")]), (2, "0", [])])
        ts = finalise(tables)
        am = tsls.Hmm(ts, np.full(2, 1e-3), np.full(2, 1e-3))
        match = np.zeros(2, dtype=np.int8)
        _, _, parent = am.find_path(np.array([1, 1], dtype=np.int8), 0, 2, match)
        assert list(parent) == [0]
        assert list(match) == [1, 0]
        with pytest.raises(tsls.LibraryError, match="Haplotype allele"):
            am.find_path(np.array([1, 2], dtype=np.int8), 0, 2, match)

    def test_shared_tree_sequence(self):
        ts = two_tree_ts()
        tts = tsls.TreeSequence(ts, alleles=ALLELES)
        assert tts.num_nodes == ts.num_nodes
        assert tts.num_sites == ts.num_sites
        m = ts.num_sites
        with pytest.raises(ValueError, match="making the tsls.TreeSequence"):
            tsls.Hmm(tts, np.full(m, 0.01), np.full(m, 0.01), alleles=ALLELES)
        am1 = tsls.Hmm(tts, np.full(m, 0.01), np.full(m, 0.01))
        am2 = tsls.Hmm(ts, np.full(m, 0.01), np.full(m, 0.01), alleles=ALLELES)
        assert am1.tree_sequence is tts
        h = np.zeros(m, dtype=np.int8)
        for x, y in zip(
            am1.find_path(h, 0, m, np.zeros(m, dtype=np.int8)),
            am2.find_path(h, 0, m, np.zeros(m, dtype=np.int8)),
        ):
            np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("state", [5, -2])
    def test_bad_haplotype_allele(self, state):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 1e-2, 1e-2)
        h = np.zeros(ts.num_sites, dtype=np.int8)
        h[1] = state
        match = np.zeros(ts.num_sites, dtype=np.int8)
        with pytest.raises(tsls.LibraryError, match="Haplotype allele"):
            am.find_path(h, 0, ts.num_sites, match)
        with pytest.raises(tsls.LibraryError, match="Haplotype allele"):
            am.forward_matrix(h)

    @pytest.mark.parametrize(("start", "end"), [(0, 0), (-1, 2), (2, 1), (0, 5)])
    def test_bad_site_range(self, start, end):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 1e-2, 1e-2)
        h = np.zeros(ts.num_sites, dtype=np.int8)
        match = np.zeros(ts.num_sites, dtype=np.int8)
        with pytest.raises(tsls.LibraryError, match="site range"):
            am.find_path(h, start, end, match)

    def test_bad_lengths(self):
        ts = tsls.TreeSequence(single_tree_ts())
        with pytest.raises(ValueError, match="recombination"):
            tsls.Hmm(ts, np.zeros(1), np.zeros(ts.num_sites))
        with pytest.raises(ValueError, match="mismatch"):
            tsls.Hmm(ts, np.zeros(ts.num_sites), np.zeros(1))
        am, _ = make_matcher(single_tree_ts(), 1e-2, 1e-2)
        with pytest.raises(ValueError):
            am.forward_matrix(np.zeros(1, dtype=np.int8))
        h = np.zeros(ts.num_sites, dtype=np.int8)
        with pytest.raises(ValueError, match="normalisation_factor"):
            am.backward_matrix(h, np.ones(1))

    def test_bad_normalisation_factor(self):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 1e-2, 1e-2)
        h = np.zeros(ts.num_sites, dtype=np.int8)
        with pytest.raises(tsls.LibraryError, match="normalisation"):
            am.backward_matrix(h, np.zeros(ts.num_sites))

    def test_match_impossible(self):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 0, 0)
        h = np.full(ts.num_sites, 4, dtype=np.int8)
        match = np.zeros(ts.num_sites, dtype=np.int8)
        with pytest.raises(tsls.MatchImpossible):
            am.find_path(h, 0, ts.num_sites, match)
        with pytest.raises(tsls.MatchImpossible):
            am.forward_matrix(h)

    def test_non_integer_coordinates(self):
        # Node 0 on [0, 1.5), and node 1 on [1.5, 3.25)
        tables = tskit.Tree.generate_balanced(2, span=3.25).tree_sequence.dump_tables()
        add_mutations(tables, [(0.5, "0", [(0, "1")]), (2.75, "0", [(1, "1")])])
        ts = finalise(tables)
        am = tsls.Hmm(ts, np.full(2, 0.4), np.full(2, 1e-3))
        match = np.zeros(2, dtype=np.int8)
        left, right, parent = am.find_path(np.array([1, 1], dtype=np.int8), 0, 2, match)
        assert left.dtype == np.float64
        assert list(parent) == [1, 0]
        assert list(left) == [2.75, 0]
        assert list(right) == [3.25, 2.75]

    def test_memory_stats(self):
        ts = two_tree_ts()
        am, _ = make_matcher(ts, 1e-2, 1e-2)
        assert am.total_memory > 0
        h = np.zeros(ts.num_sites, dtype=np.int8)
        match = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites, match)
        assert am.mean_traceback_size > 0

    def test_reuse_matcher(self):
        # Running the algorithms repeatedly on the same matcher gives the
        # same results.
        ts = two_tree_ts()
        A, present, _ = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 0.2, 0.05)
        h = queries(ts, A, present)[-1]
        m = ts.num_sites
        results = []
        for _ in range(2):
            match = np.zeros(m, dtype=np.int8)
            path = am.find_path(h, 0, m, match)
            fm = am.forward_matrix(h)
            bm = am.backward_matrix(h, fm.normalisation_factor)
            results.append((path, match, fm.decode(), bm.decode()))
        for x, y in zip(results[0][0], results[1][0]):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(results[0][1:], results[1][1:]):
            np.testing.assert_array_equal(x, y)

    def test_compressed_matrix(self):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 1e-2, 1e-2)
        fm = am.forward_matrix(np.zeros(ts.num_sites, dtype=np.int8))
        assert fm.num_sites == ts.num_sites
        nodes, values = fm.get_site(0)
        assert len(nodes) == len(values) > 0
        D = fm.decode([0, 6])
        assert D.shape == (ts.num_sites, 2)


class TestPruning:
    def run_all(self, ts, h, **kwargs):
        m = ts.num_sites
        am = tsls.Hmm(
            ts,
            np.full(m, 0.05),
            np.full(m, 0.02),
            alleles=ALLELES,
            extended_checks=True,
            **kwargs,
        )
        match = np.zeros(m, dtype=np.int8)
        path = am.find_path(h, 0, m, match)
        fm = am.forward_matrix(h)
        bm = am.backward_matrix(h, fm.normalisation_factor)
        return am, path, match, fm.decode(), bm.decode()

    @pytest.mark.parametrize("name", list(EXAMPLES))
    def test_weak_pruning_is_exact(self, name):
        ts = EXAMPLES[name]()
        A, present, _ = dense.node_states(ts, ALLELES)
        for h in queries(ts, A, present):
            # Pruning processes every Viterbi site in full, which may choose
            # another path among equally likely ones than sparse sites do.
            _, path1, match1, F1, B1 = self.run_all(ts, h, sparse=False)
            _, path2, match2, F2, B2 = self.run_all(
                ts, h, relative_prune_threshold=1e-300
            )
            for x, y in zip(path1, path2):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(match1, match2)
            np.testing.assert_array_equal(F1, F2)
            np.testing.assert_array_equal(B1, B2)

    def test_strong_pruning_reduces_traceback(self):
        ts = two_tree_ts()
        A, present, _ = dense.node_states(ts, ALLELES)
        h = queries(ts, A, present)[-2]
        am1, *_ = self.run_all(ts, h)
        size1 = am1.mean_traceback_size
        am2, *_ = self.run_all(ts, h, relative_prune_threshold=0.5)
        assert am2.mean_traceback_size < size1

    @pytest.mark.parametrize("name", list(EXAMPLES))
    @pytest.mark.parametrize("rate", RATES)
    @pytest.mark.parametrize("kind", ["floor", "prune"])
    def test_lossless_threshold(self, name, rate, kind):
        # Values below the switch odds q = (rho / n) / (1 - rho + rho / n)
        # are beaten by recombination at the next site, so flooring or
        # pruning at q does not change the Viterbi likelihood.
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        rho, mu = model_rho(ts, 0.05, rate), 0.02
        n = ts.num_nodes
        threshold = tsls.model.lossless_threshold(rho, n)
        m = ts.num_sites
        if kind == "floor":
            am, _ = make_matcher(ts, rho, mu, likelihood_threshold=threshold)
        else:
            am, _ = make_matcher(ts, rho, mu, prune_threshold=threshold)
        rho_a = np.full(m, rho)
        mu_a = np.full(m, mu)
        for h in queries(ts, A, present):
            match = np.zeros(m, dtype=np.int8)
            path = path_by_site(ts, *am.find_path(h, 0, m, match))
            value = dense.path_log_likelihood(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n, path
            )
            optimum = dense.viterbi_log_likelihood(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n
            )
            assert value == pytest.approx(optimum, rel=1e-9)

    def test_count_distinct(self):
        ts = two_tree_ts()
        A, present, _ = dense.node_states(ts, ALLELES)
        h = queries(ts, A, present)[-2]
        am, *_ = self.run_all(ts, h, count_distinct=True)
        assert 0 < am.mean_num_distinct_values <= am.mean_traceback_size
        am, *_ = self.run_all(ts, h)
        assert am.mean_num_distinct_values == 0

    @pytest.mark.parametrize("threshold", [2, -1, np.nan])
    def test_bad_relative_prune_threshold(self, threshold):
        ts = single_tree_ts()
        m = ts.num_sites
        with pytest.raises(tsls.LibraryError, match="pruning"):
            tsls.Hmm(
                ts,
                np.zeros(m),
                np.zeros(m),
                alleles=ALLELES,
                relative_prune_threshold=threshold,
            )

    def test_pruning_removed(self):
        ts = single_tree_ts()
        m = ts.num_sites
        with pytest.raises(TypeError, match="pruning"):
            tsls.Hmm(ts, np.zeros(m), np.zeros(m), pruning=("top_n", 1))

    def test_zero_relative_prune_threshold_keeps_sparse(self):
        ts = wright_fisher_ts(0, num_trees_hint=0)
        m = ts.num_sites
        h = np.zeros(m, dtype=np.int8)
        am = tsls.Hmm(
            ts,
            np.full(m, 0.02),
            np.full(m, 0.01),
            alleles=ALLELES,
            relative_prune_threshold=0,
        )
        am.find_path(h, 0, m, np.zeros(m, dtype=np.int8))
        assert am.num_sparse_sites > 0


class TestViterbiMatrix:
    @pytest.mark.parametrize("name", list(EXAMPLES))
    @pytest.mark.parametrize("rate", RATES)
    def test_against_dense(self, name, rate):
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        rho, mu = model_rho(ts, 0.05, rate), 0.02
        am, n = make_matcher(ts, rho, mu)
        m = ts.num_sites
        rho_a = np.full(m, rho)
        mu_a = np.full(m, mu)
        for h in queries(ts, A, present):
            path, matched, vm = am.viterbi_matrix(h)
            V, c = dense.viterbi_matrix(
                A, present, continuous, h, rho_a, mu_a, len(ALLELES), n
            )
            np.testing.assert_allclose(vm.normalisation_factor, c, rtol=1e-12)
            np.testing.assert_allclose(vm.decode(), V, rtol=1e-12, atol=1e-300)
            # The same path as find_path processing every site in full, as
            # viterbi_matrix does
            full = tsls.Hmm(ts, rho_a, mu_a, alleles=ALLELES, sparse=False)
            match = np.zeros(m, dtype=np.int8)
            expected = full.find_path(h, 0, m, match)
            for x, y in zip(path, expected):
                np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(matched, match)

    def test_traceback_lists(self):
        ts = two_tree_ts()
        A, present, _ = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 0.05, 0.02)
        for h in queries(ts, A, present):
            _, _, vm = am.viterbi_matrix(h)
            sizes = np.diff(vm.offsets)
            assert np.mean(sizes) == pytest.approx(am.mean_traceback_size)
            assert len(vm.flags) == len(vm.nodes)
            assert set(np.unique(vm.flags)) <= {0, 1}

    def test_start_end(self):
        ts = two_tree_ts()
        A, _, _ = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 0.05, 0.02)
        _, _, vm = am.viterbi_matrix(A[:, 0], 1, ts.num_sites - 1)
        sizes = np.diff(vm.offsets)
        assert sizes[0] == 0
        assert sizes[-1] == 0
        assert np.all(sizes[1:-1] > 0)

    def test_forward_has_no_flags(self):
        ts = single_tree_ts()
        am, _ = make_matcher(ts, 0.05, 0.02)
        fm = am.forward_matrix(np.zeros(ts.num_sites, dtype=np.int8))
        assert fm.flags is None

    @pytest.mark.parametrize("algorithm", ["viterbi", "forward"])
    def test_dump_load(self, tmp_path, algorithm):
        ts = two_tree_ts()
        A, _, _ = dense.node_states(ts, ALLELES)
        am, _ = make_matcher(ts, 0.05, 0.02)
        h = A[:, 1]
        if algorithm == "viterbi":
            _, _, cm = am.viterbi_matrix(h)
        else:
            cm = am.forward_matrix(h)
        path = tmp_path / "matrix.npz"
        cm.dump(path)
        loaded = tsls.load_compressed_matrix(path, ts)
        for attr in ["normalisation_factor", "offsets", "nodes", "values"]:
            np.testing.assert_array_equal(getattr(cm, attr), getattr(loaded, attr))
        if algorithm == "viterbi":
            np.testing.assert_array_equal(cm.flags, loaded.flags)
        else:
            assert loaded.flags is None
        np.testing.assert_array_equal(cm.decode(), loaded.decode())
        with pytest.raises(ValueError, match="does not match"):
            tsls.load_compressed_matrix(path, single_tree_ts())


def wright_fisher_ts(seed, *, num_trees_hint, population_size=15, generations=25):
    """
    Return a random tree sequence from a haploid Wright-Fisher simulation
    with 60 sites, whose samples are the last generation and some internal
    nodes. Small populations give deep trees with polytomies and unary
    nodes, as in sc2ts ARGs. Each child recombines with probability
    num_trees_hint / (population_size * generations), and some sites have
    several mutations (on the same or different nodes).
    """
    rng = np.random.default_rng(seed)
    L = 100
    tables = tskit.TableCollection(L)
    current = [tables.nodes.add_row(time=generations) for _ in range(population_size)]
    recombination = num_trees_hint / (population_size * generations)
    for g in range(generations - 1, -1, -1):
        parents = current
        current = []
        for _ in range(population_size):
            child = tables.nodes.add_row(time=g)
            p1, p2 = rng.choice(parents, size=2)
            if rng.random() < recombination:
                x = int(rng.integers(1, L))
                tables.edges.add_row(0, x, p1, child)
                tables.edges.add_row(x, L, p2, child)
            else:
                tables.edges.add_row(0, L, p1, child)
            current.append(child)
    internal = rng.choice(
        np.arange(population_size, tables.nodes.num_rows - population_size),
        size=5,
        replace=False,
    )
    tables.sort()
    tables.simplify(np.concatenate([current, internal]).astype(np.int32))
    ts = tables.tree_sequence()
    tables = ts.dump_tables()
    positions = np.sort(rng.choice(np.arange(1, 10 * L), size=60, replace=False)) / 10
    for x in positions:
        tree = ts.at(x)
        in_tree = [u for u in tree.nodes() if tree.parent(u) != -1 or tree.num_children(u)]
        ancestral = ACGT[rng.integers(4)]
        site = tables.sites.add_row(position=x, ancestral_state=ancestral)
        num_mutations = rng.choice([1, 1, 1, 2, 3])
        nodes = rng.choice(in_tree, size=num_mutations)
        # Older mutations first, as tskit requires
        for u in sorted(nodes, key=lambda u: -ts.nodes_time[u]):
            tables.mutations.add_row(site=site, node=u, derived_state=ACGT[rng.integers(4)])
    return finalise(tables)


def noisy_queries(ts, A, present, seed):
    """
    Return copies and two-parent recombinants of nodes in every tree, with
    some sites changed and some missing.
    """
    rng = np.random.default_rng(seed)
    always = np.where(np.all(present, axis=0))[0]
    m = ts.num_sites
    H = []
    for _ in range(6):
        u, v = rng.choice(always, size=2)
        x = int(rng.integers(m))
        h = np.concatenate([A[:x, u], A[x:, v]])
        k = int(rng.integers(4))
        sites = rng.choice(m, size=k, replace=False)
        h[sites] = rng.integers(len(ALLELES), size=k)
        h[rng.choice(m, size=int(rng.integers(3)), replace=False)] = -1
        H.append(h.astype(np.int8))
    return H


class TestSparse:
    """
    Viterbi processes most sites sparsely when trees span many sites; check
    that this finds paths with the same likelihood as processing every site
    in full.
    """

    def matchers(self, ts, rho, mu, likelihood_threshold=None, prune_threshold=None):
        kwargs = dict(
            alleles=ALLELES,
            likelihood_threshold=likelihood_threshold,
            prune_threshold=prune_threshold,
            extended_checks=True,
        )
        return (
            tsls.Hmm(ts, rho, mu, **kwargs),
            tsls.Hmm(ts, rho, mu, sparse=False, **kwargs),
        )

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("num_trees_hint", [0, 2, 10])
    @pytest.mark.parametrize("rate", RATES)
    @pytest.mark.parametrize("likelihood_threshold", [None, 1e-3])
    def test_against_full(self, seed, num_trees_hint, rate, likelihood_threshold):
        ts = wright_fisher_ts(seed, num_trees_hint=num_trees_hint)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = model_rho(ts, np.full(m, 0.02), rate)
        mu = np.full(m, 0.01)
        n = ts.num_nodes
        sparse, full = self.matchers(ts, rho, mu, likelihood_threshold)
        num_sparse_sites = 0
        for h in noisy_queries(ts, A, present, seed):
            values = []
            for am in [sparse, full]:
                match = np.zeros(m, dtype=np.int8)
                left, right, parent = am.find_path(h, 0, m, match)
                path = path_by_site(ts, left, right, parent)
                np.testing.assert_array_equal(match, A[np.arange(m), path])
                values.append(
                    dense.path_log_likelihood(
                        A, present, continuous, h, rho, mu, len(ALLELES), n, path
                    )
                )
            num_sparse_sites += sparse.num_sparse_sites
            assert full.num_sparse_sites == 0
            assert values[0] == pytest.approx(values[1], rel=1e-9)
            if likelihood_threshold is None:
                optimum = dense.viterbi_log_likelihood(
                    A, present, continuous, h, rho, mu, len(ALLELES), n
                )
                assert values[0] == pytest.approx(optimum, rel=1e-9)
        if num_trees_hint == 0:
            assert num_sparse_sites > 0

    @pytest.mark.parametrize("seed", range(3))
    def test_windows(self, seed):
        ts = wright_fisher_ts(seed, num_trees_hint=2)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = model_rho(ts, np.full(m, 0.02), "odds")
        mu = np.full(m, 0.01)
        sparse, full = self.matchers(ts, rho, mu)
        for h in noisy_queries(ts, A, present, seed)[:2]:
            for start, end in [(0, 30), (10, 50), (25, 60), (59, 60)]:
                values = []
                for am in [sparse, full]:
                    match = np.zeros(m, dtype=np.int8)
                    left, right, parent = am.find_path(h, start, end, match)
                    path = path_by_site(ts, left, right, parent, start, end)
                    window = (A[start:end], present[start:end])
                    cont = continuous[start:end].copy()
                    cont[0] = present[start]
                    args = (*window, cont, h[start:end], rho[start:end], mu[start:end])
                    values.append(
                        dense.path_log_likelihood(*args, len(ALLELES), ts.num_nodes, path)
                    )
                assert values[0] == pytest.approx(values[1], rel=1e-9)

    def test_many_mutations(self):
        # More mutations at a site than sparse sites used to handle
        ts = wright_fisher_ts(3, num_trees_hint=0)
        tables = ts.dump_tables()
        rng = np.random.default_rng(3)
        site = ts.num_sites // 2
        # Several mutations on some nodes, as recurrent mutations
        nodes = rng.choice(ts.num_nodes - 1, size=80)
        for u in nodes:
            tables.mutations.add_row(
                site=site, node=int(u), derived_state=ALLELES[int(rng.integers(4))]
            )
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = np.full(m, 0.02)
        mu = np.full(m, 0.01)
        am = tsls.Hmm(ts, rho, mu, alleles=ALLELES, extended_checks=True)
        num_sparse_sites = 0
        for h in noisy_queries(ts, A, present, 3):
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            args = (A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes)
            value = dense.path_log_likelihood(*args, path)
            assert value == pytest.approx(dense.viterbi_log_likelihood(*args), rel=1e-9)
            num_sparse_sites += am.num_sparse_sites
        assert num_sparse_sites > 0

    def test_sites_without_mutations(self):
        # Trees whose remaining sites have no mutations are numbered too
        ts = wright_fisher_ts(2, num_trees_hint=3)
        tables = ts.dump_tables()
        m = ts.num_sites
        tables.mutations.keep_rows(tables.mutations.site < m // 2)
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        A, present, continuous = dense.node_states(ts, ALLELES)
        rho = np.full(m, 0.02)
        mu = np.full(m, 0.01)
        for h in noisy_queries(ts, A, present, 2):
            for start, end in [(0, m), (m // 2, m), (m // 2 + 3, m - 2)]:
                # A new HMM, so the first tree numbered is one of these
                am = tsls.Hmm(
                    ts, rho, mu, alleles=ALLELES, extended_checks=True, index_min_sites=1
                )
                match = np.zeros(m, dtype=np.int8)
                left, right, parent = am.find_path(h, start, end, match)
                path = path_by_site(ts, left, right, parent, start, end)
                np.testing.assert_array_equal(
                    match[start:end], A[np.arange(start, end), path]
                )

    def test_rates_change(self):
        # Rates that change between sites don't stop sites being sparse
        ts = wright_fisher_ts(1, num_trees_hint=0)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = np.repeat([0.01, 0.05, 0.02], [20, 20, m - 40])
        mu = np.repeat([0.01, 0.02], [30, m - 30])
        sparse, full = self.matchers(ts, rho, mu)
        for h in noisy_queries(ts, A, present, 1):
            values = []
            for am in [sparse, full]:
                match = np.zeros(m, dtype=np.int8)
                left, right, parent = am.find_path(h, 0, m, match)
                path = path_by_site(ts, left, right, parent)
                values.append(
                    dense.path_log_likelihood(
                        A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes, path
                    )
                )
            assert values[0] == pytest.approx(values[1], rel=1e-9)
            optimum = dense.viterbi_log_likelihood(
                A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes
            )
            assert values[0] == pytest.approx(optimum, rel=1e-9)

    def test_matrix_is_full(self):
        # viterbi_matrix stores every value, so it processes every site
        ts = wright_fisher_ts(0, num_trees_hint=0)
        m = ts.num_sites
        h = np.zeros(m, dtype=np.int8)
        am = tsls.Hmm(ts, np.full(m, 0.02), np.full(m, 0.01), alleles=ALLELES)
        am.viterbi_matrix(h)
        assert am.num_sparse_sites == 0

    def test_relative_pruning_is_sparse(self):
        # In Viterbi, relative pruning is a prune threshold
        ts = wright_fisher_ts(0, num_trees_hint=0)
        m = ts.num_sites
        h = np.zeros(m, dtype=np.int8)
        am = tsls.Hmm(
            ts,
            np.full(m, 0.02),
            np.full(m, 0.01),
            alleles=ALLELES,
            relative_prune_threshold=1e-3,
        )
        am.find_path(h, 0, m, np.zeros(m, dtype=np.int8))
        assert am.num_sparse_sites > 0

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("num_trees_hint", [0, 3])
    def test_every_tree_indexed(self, seed, num_trees_hint):
        # Index every tree with a site left after its first, so that the
        # indexed sites meet tree transitions often
        ts = wright_fisher_ts(seed, num_trees_hint=num_trees_hint)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho = np.full(m, 0.05)
        mu = np.full(m, 0.01)
        am = tsls.Hmm(
            ts, rho, mu, alleles=ALLELES, extended_checks=True, index_min_sites=1
        )
        num_sparse_sites = 0
        for h in noisy_queries(ts, A, present, seed):
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            args = (A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes)
            value = dense.path_log_likelihood(*args, path)
            assert value == pytest.approx(dense.viterbi_log_likelihood(*args), rel=1e-9)
            num_sparse_sites += am.num_sparse_sites
        assert num_sparse_sites > 0


class TestPerSiteRates:
    """
    Viterbi with rates that vary at every site, as with a genetic map, so
    that the cost of switching changes from site to site.
    """

    def rates(self, ts, seed):
        rng = np.random.default_rng(seed)
        m = ts.num_sites
        rho = 10 ** rng.uniform(-6, -0.3, size=m)
        mu = 10 ** rng.uniform(-4, -1, size=m)
        return rho, mu

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("num_trees_hint", [0, 2, 10])
    @pytest.mark.parametrize("sparse", [True, False])
    def test_optimal(self, seed, num_trees_hint, sparse):
        ts = wright_fisher_ts(seed, num_trees_hint=num_trees_hint)
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho, mu = self.rates(ts, seed)
        am = tsls.Hmm(
            ts, rho, mu, alleles=ALLELES, extended_checks=True, sparse=sparse
        )
        num_sparse_sites = 0
        for h in noisy_queries(ts, A, present, seed):
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            args = (A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes)
            value = dense.path_log_likelihood(*args, path)
            optimum = dense.viterbi_log_likelihood(*args)
            assert value == pytest.approx(optimum, rel=1e-9)
            num_sparse_sites += am.num_sparse_sites
        # Sparse sites don't depend on the rates being the same at every site
        if sparse and num_trees_hint == 0:
            assert num_sparse_sites > 0

    @pytest.mark.parametrize("name", list(EXAMPLES))
    @pytest.mark.parametrize("seed", range(3))
    def test_matrix(self, name, seed):
        ts = EXAMPLES[name]()
        A, present, continuous = dense.node_states(ts, ALLELES)
        rho, mu = self.rates(ts, seed)
        am = tsls.Hmm(ts, rho, mu, alleles=ALLELES, extended_checks=True)
        for h in queries(ts, A, present):
            _, _, vm = am.viterbi_matrix(h)
            V, c = dense.viterbi_matrix(
                A, present, continuous, h, rho, mu, len(ALLELES), ts.num_nodes
            )
            np.testing.assert_allclose(vm.normalisation_factor, c, rtol=1e-12)
            np.testing.assert_allclose(vm.decode(), V, rtol=1e-12, atol=1e-300)

    @pytest.mark.parametrize("rho", [0.5, 0.9])
    def test_reentry(self, rho):
        # Nodes leave and re-enter the tree; with frequent recombination
        # they can be the best nodes soon after.
        ts = gap_ts()
        A, present, continuous = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        rho_a = np.full(m, rho)
        mu_a = np.full(m, 0.05)
        am = tsls.Hmm(ts, rho_a, mu_a, alleles=ALLELES, extended_checks=True)
        for h in queries(ts, A, present):
            match = np.zeros(m, dtype=np.int8)
            left, right, parent = am.find_path(h, 0, m, match)
            path = path_by_site(ts, left, right, parent)
            np.testing.assert_array_equal(match, A[np.arange(m), path])
            args = (A, present, continuous, h, rho_a, mu_a, len(ALLELES), ts.num_nodes)
            value = dense.path_log_likelihood(*args, path)
            assert value == pytest.approx(dense.viterbi_log_likelihood(*args), rel=1e-9)


class TestNodeStatistics:
    def test_forward_backward_updated_match_stored(self):
        ts = wright_fisher_ts(0, num_trees_hint=3)
        A, present, _ = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        am = tsls.Hmm(ts, np.full(m, 0.02), np.full(m, 0.01), alleles=ALLELES)
        for h in noisy_queries(ts, A, present, 0):
            fm = am.forward_matrix(h)
            # Each site's values are stored before they are coalesced
            assert am.mean_num_updated == pytest.approx(np.mean(np.diff(fm.offsets)))
            assert 0 < am.mean_num_values <= am.max_num_values
            assert am.max_num_values <= np.max(np.diff(fm.offsets))
            bm = am.backward_matrix(h, fm.normalisation_factor)
            assert am.mean_num_updated == pytest.approx(np.mean(np.diff(bm.offsets)))

    def test_viterbi_sparse_updates_fewer(self):
        ts = wright_fisher_ts(1, num_trees_hint=0)
        A, present, _ = dense.node_states(ts, ALLELES)
        m = ts.num_sites
        for h in noisy_queries(ts, A, present, 1):
            stats = {}
            for sparse in [True, False]:
                am = tsls.Hmm(
                    ts, np.full(m, 0.02), np.full(m, 0.01), alleles=ALLELES, sparse=sparse
                )
                am.find_path(h, 0, m, np.zeros(m, dtype=np.int8))
                stats[sparse] = (am.mean_num_updated, am.mean_num_values, am.num_sparse_sites)
            if stats[True][2] > 0:
                assert stats[True][0] < stats[False][0]
            assert stats[False][0] >= stats[False][1]

    def test_window_counts_sites(self):
        ts = wright_fisher_ts(2, num_trees_hint=2)
        m = ts.num_sites
        am = tsls.Hmm(ts, np.full(m, 0.02), np.full(m, 0.01), alleles=ALLELES)
        assert am.mean_num_values == 0
        h = np.zeros(m, dtype=np.int8)
        am.find_path(h, 10, 20, np.zeros(m, dtype=np.int8))
        assert am.max_num_values >= am.mean_num_values > 0


# The following tests are adapted from tskit's haplotype matching tests
# (python/tests/test_haplotype_matching.py and c/tests/test_haplotype_matching.c),
# from before the LsHmm engine was removed from tskit.


def check_against_dense(ts, h, rho, mu, *, rtol=1e-12):
    """
    Check the Viterbi path and the forward and backward matrices for the
    specified haplotype against the dense implementation, with constant
    rates rho and mu, and return the path by site.
    """
    A, present, continuous = dense.node_states(ts, ALLELES)
    m = ts.num_sites
    n = ts.num_nodes
    rho_a = np.full(m, rho)
    mu_a = np.full(m, mu)
    am, _ = make_matcher(ts, rho, mu)
    match = np.zeros(m, dtype=np.int8)
    left, right, parent = am.find_path(h, 0, m, match)
    path = path_by_site(ts, left, right, parent)
    np.testing.assert_array_equal(match, A[np.arange(m), path])
    value = dense.path_log_likelihood(
        A, present, continuous, h, rho_a, mu_a, len(ALLELES), n, path
    )
    optimum = dense.viterbi_log_likelihood(
        A, present, continuous, h, rho_a, mu_a, len(ALLELES), n
    )
    assert value == pytest.approx(optimum, rel=1e-9)

    fm = am.forward_matrix(h)
    F, c = dense.forward(A, present, continuous, h, rho_a, mu_a, len(ALLELES), n)
    np.testing.assert_allclose(fm.normalisation_factor, c, rtol=rtol)
    np.testing.assert_allclose(fm.decode(), F, rtol=rtol, atol=1e-300)
    bm = am.backward_matrix(h, fm.normalisation_factor)
    B = dense.backward(A, present, continuous, h, rho_a, mu_a, len(ALLELES), n, c)
    np.testing.assert_allclose(bm.decode(), B, rtol=rtol, atol=1e-300)
    np.testing.assert_allclose(np.sum(F * B, axis=1), 1)
    return path


def unique_mutations_ts():
    # tskit's balanced tree example, with one mutation on each sample: the
    # mutation on sample j is at site j.
    #
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         8
    tables = tskit.Tree.generate_balanced(4, span=8).tree_sequence.dump_tables()
    add_mutations(tables, [(2 * j + 1, "A", [(j, "C")]) for j in range(4)])
    return finalise(tables)


class TestUniqueMutationPaths:
    """
    Exact Viterbi paths on a balanced tree with a unique mutation on each
    sample, from tskit's TestSingleBalancedTreeExample. Recombination is
    much more likely than a mismatch, so the best path switches to avoid
    every mismatch.
    """

    rho = 0.1
    mu = 1e-4

    def path(self, h):
        ts = unique_mutations_ts()
        h = np.array(h, dtype=np.int8)
        return list(check_against_dense(ts, h, self.rho, self.mu))

    @pytest.mark.parametrize("j", range(4))
    def test_single_sample(self, j):
        h = [0, 0, 0, 0]
        h[j] = 1
        assert self.path(h) == [j] * 4

    @pytest.mark.parametrize("j", [1, 2])
    def test_single_sample_missing_flanks(self, j):
        h = [-1, 0, 0, -1]
        h[j] = 1
        assert self.path(h) == [j] * 4

    def test_switch_at_every_site(self):
        assert self.path([1, 1, 1, 1]) == [0, 1, 2, 3]

    def test_missing_flanks(self):
        # Sites 0 and 3 are copied from the samples of sites 1 and 2, since
        # copying anything else there needs another switch.
        assert self.path([-1, 1, 1, -1]) == [1, 1, 2, 2]

    def test_missing_middle(self):
        # One switch from sample 0 to sample 3, which can be at any of the
        # sites 1 to 3: we only require one switch.
        path = self.path([1, -1, -1, 1])
        assert path[0] == 0
        assert path[-1] == 3
        assert np.sum(np.diff(path) != 0) == 1


def mirror(ts):
    """
    Return ts with the coordinates reversed: a site at position x moves to
    L - 1 - x (so integer positions stay in [0, L)), and an edge [l, r)
    moves to [L - r, L - l). Sites stay at integer positions only if the
    original positions are integers.
    """
    L = ts.sequence_length
    tables = ts.dump_tables()
    edges = tables.edges
    left = edges.left.copy()
    edges.left = L - edges.right
    edges.right = L - left
    sites = tables.sites
    sites.position = L - 1 - sites.position
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


class TestMirroring:
    """
    The HMM is reversible: matching the reversed haplotype against the
    mirrored tree sequence gives the same Viterbi optimum and posteriors
    (reversed). From tskit's TestMirroringHap.

    The likelihoods are the same apart from the initial probability, which
    depends on the first site: the HMM starts from the uniform distribution
    over all n nodes, and recombines at the first site too, so each node in
    the tree there starts with probability (1 - r + r P / n) / n, where P
    is the number of nodes in the tree at the first site.
    """

    def examples(self):
        for name in ["single_tree", "two_trees", "forest", "isolated_root"]:
            ts = EXAMPLES[name]()
            if np.all(ts.sites_position == np.floor(ts.sites_position)):
                yield ts

    @pytest.mark.parametrize(("rho", "mu"), PARAMS)
    def test_forward_backward(self, rho, mu):
        for ts in self.examples():
            mts = mirror(ts)
            A, present, _ = dense.node_states(ts, ALLELES)
            m = ts.num_sites
            n = ts.num_nodes
            for h in queries(ts, A, present):
                posteriors = []
                log_likelihoods = []
                for x, hx, P in [
                    (ts, h, np.sum(present[0])),
                    (mts, h[::-1].copy(), np.sum(present[-1])),
                ]:
                    am, _ = make_matcher(x, rho, mu)
                    fm = am.forward_matrix(hx)
                    bm = am.backward_matrix(hx, fm.normalisation_factor)
                    initial = np.log(1 - rho + rho * P / n)
                    log_likelihoods.append(fm.log_likelihood - initial)
                    posteriors.append(fm.decode() * bm.decode())
                assert log_likelihoods[0] == pytest.approx(log_likelihoods[1], rel=1e-12)
                np.testing.assert_allclose(
                    posteriors[0], posteriors[1][::-1], rtol=1e-9, atol=1e-15
                )
                assert m == mts.num_sites

    @pytest.mark.parametrize(("rho", "mu"), PARAMS)
    def test_viterbi(self, rho, mu):
        for ts in self.examples():
            mts = mirror(ts)
            A, present, continuous = dense.node_states(ts, ALLELES)
            mA, mpresent, mcontinuous = dense.node_states(mts, ALLELES)
            m = ts.num_sites
            n = ts.num_nodes
            rho_a = np.full(m, rho)
            mu_a = np.full(m, mu)
            for h in queries(ts, A, present):
                values = []
                for x, hx, states in [
                    (ts, h, (A, present, continuous)),
                    (mts, h[::-1].copy(), (mA, mpresent, mcontinuous)),
                ]:
                    am, _ = make_matcher(x, rho, mu)
                    match = np.zeros(m, dtype=np.int8)
                    left, right, parent = am.find_path(hx, 0, m, match)
                    path = path_by_site(x, left, right, parent)
                    values.append(
                        dense.path_log_likelihood(
                            *states, hx, rho_a, mu_a, len(ALLELES), n, path
                        )
                    )
                assert values[0] == pytest.approx(values[1], rel=1e-9)


@pytest.mark.parametrize(("rho", "mu"), PARAMS)
def test_all_missing(example, rho, mu):
    # From tskit's LSBase queries, which include a haplotype with every
    # site missing.
    ts, _, _, _ = example
    h = np.full(ts.num_sites, -1, dtype=np.int8)
    check_against_dense(ts, h, rho, mu)


class TestExtremeRates:
    """
    Recombination and mismatch probabilities near their limits, from the
    extreme values in tskit's tests.
    """

    # With 5 alleles, the mismatch probability must be less than 1 / 5 for
    # a match to be more likely than a mismatch.
    @pytest.mark.parametrize("rho", [0.999, 1e-6])
    @pytest.mark.parametrize("mu", [1e-6, 0.19])
    def test_against_dense(self, example, rho, mu):
        ts, A, present, _ = example
        for h in queries(ts, A, present):
            check_against_dense(ts, h, rho, mu, rtol=1e-10)


def comb_ts(n, num_sites, seed):
    """
    Return a comb (caterpillar) tree of n samples with random mutations,
    which gives many distinct values at each site (as tskit's
    test_caterpillar_tree_many_values).
    """
    rng = np.random.default_rng(seed)
    ts = tskit.Tree.generate_comb(n, span=num_sites).tree_sequence
    tables = ts.dump_tables()
    nodes = np.arange(ts.num_nodes - 1)
    mutations = []
    for j in range(num_sites):
        site_nodes = rng.choice(nodes, size=rng.integers(1, 4), replace=False)
        site_nodes = site_nodes[np.argsort(-ts.nodes_time[site_nodes])]
        states = rng.choice(ALLELES[1:], size=len(site_nodes))
        mutations.append((j, "A", list(zip(site_nodes, states))))
    add_mutations(tables, mutations)
    return finalise(tables)


@pytest.mark.parametrize("n", [8, 16, 32, 64])
def test_comb_many_values(n):
    ts = comb_ts(n, 12, seed=n)
    A, present, _ = dense.node_states(ts, ALLELES)
    rng = np.random.default_rng(1)
    for u in rng.choice(ts.samples(), size=3, replace=False):
        h = A[:, u].copy()
        h[rng.integers(0, ts.num_sites)] = 2
        check_against_dense(ts, h, 0.05, 0.01, rtol=1e-10)
    fm = make_matcher(ts, 0.05, 0.01)[0].forward_matrix(A[:, 0])
    # The forward values aren't all compressed into a few nodes
    assert np.max(np.diff(fm.offsets)) > 4


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [10, 50])
def test_msprime_against_dense(seed, num_samples):
    # As tskit's TestSimulationExamples, with sequences as long as needed
    # for a few trees and sites.
    ts = msprime.sim_ancestry(
        num_samples,
        ploidy=1,
        sequence_length=100,
        recombination_rate=0.01,
        population_size=1,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=seed)
    assert ts.num_trees > 1
    assert ts.num_sites > 5
    A, present, _ = dense.node_states(ts, ALLELES)
    rng = np.random.default_rng(seed)
    held_out = rng.integers(0, ts.num_nodes, size=2)
    H = [A[:, u] for u in held_out]
    H.append(rng.integers(0, 4, size=ts.num_sites).astype(np.int8))
    for h in H:
        check_against_dense(ts, np.asarray(h, dtype=np.int8), 0.01, 0.01, rtol=1e-10)
