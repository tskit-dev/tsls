import msprime
import numpy as np
import pytest
import tsls

import matcher
import prepare
import queries
import subargs


def simulate_sc2ts_like(seed, samples=30):
    """
    Simulate a small haploid ARG with JC69 mutations, with the sc2ts root
    structure: an isolated vestigial node 0 and node 1 as the root of every
    tree.
    """
    ts = msprime.sim_ancestry(
        samples,
        ploidy=1,
        sequence_length=200,
        recombination_rate=1e-3,
        random_seed=seed,
    )
    # Add a new root (node 1) above all trees and a vestigial node 0.
    tables = ts.dump_tables()
    max_time = np.max(tables.nodes.time)
    tables.nodes.clear()
    tables.nodes.add_row(time=max_time + 2)
    tables.nodes.add_row(time=max_time + 1)
    for node in ts.tables.nodes:
        tables.nodes.append(node)
    tables.edges.parent += 2
    tables.edges.child += 2
    for tree in ts.trees():
        for root in tree.roots:
            tables.edges.add_row(*tree.interval, parent=1, child=root + 2)
    tables.edges.squash()
    tables.sort()
    ts = tables.tree_sequence()
    ts = msprime.sim_mutations(ts, rate=0.02, model=msprime.JC69(), random_seed=seed)
    assert np.any(np.bincount(ts.mutations_site) > 1)
    return ts


@pytest.fixture(scope="module")
def ts():
    ts, deleted_sites = prepare.prepare(simulate_sc2ts_like(1234))
    assert len(deleted_sites) == 0
    return ts


@pytest.fixture(scope="module")
def query_set(ts):
    return queries.make_queries(
        ts,
        seed=5,
        num_nodes=5,
        divergence=(0, 1, 3),
        num_recombinants=4,
        recombinant_divergence=(0, 2),
    )


class TestPrepare:
    def test_vestigial_root(self, ts):
        for tree in ts.trees():
            assert tree.root == 0
            assert list(tree.children(0)) == [1]

    def test_non_acgt_sites_deleted(self):
        tables = simulate_sc2ts_like(2).dump_tables()
        tables.mutations[0] = tables.mutations[0].replace(derived_state="-")
        ts, deleted_sites = prepare.prepare(tables.tree_sequence())
        assert list(deleted_sites) == [tables.mutations[0].site]
        assert ts.num_sites == len(tables.sites) - 1

    def test_node_0_in_edge(self, ts):
        with pytest.raises(ValueError, match="Node 0"):
            prepare.add_vestigial_root_edge(ts)


class TestQueries:
    def test_counts(self, query_set):
        df = query_set.dataframe()
        assert np.sum(df.kind == "copy") == 5 * 3
        assert np.sum(df.kind == "recombinant") == 2 * 4 * 2
        assert list(df.query_id) == list(range(len(df)))

    def test_copy_divergence(self, ts, query_set):
        df = query_set.dataframe()
        for row in df[df.kind == "copy"].itertuples():
            (node,) = queries.parse_list(row.nodes)
            source = queries.node_haplotypes(ts, [node])[0]
            h = query_set.haplotypes[row.query_id]
            diff = np.where(h != source)[0]
            assert len(diff) == row.k
            assert list(diff) == queries.parse_list(row.mutated_sites)

    def test_recombinant_segments(self, ts, query_set):
        df = query_set.dataframe()
        for row in df[df.kind == "recombinant"].itertuples():
            nodes = queries.parse_list(row.nodes)
            parents = queries.node_haplotypes(ts, nodes)
            breakpoints = queries.parse_list(row.breakpoints)
            segment_diffs = queries.parse_list(row.segment_diffs)
            assert len(segment_diffs) == len(breakpoints) + 1
            h = query_set.haplotypes[row.query_id].copy()
            mutated = queries.parse_list(row.mutated_sites)
            assert len(mutated) == row.k
            bounds = [0, *breakpoints, ts.num_sites]
            for j in range(len(bounds) - 1):
                parent = parents[j % 2]
                segment = np.arange(bounds[j], bounds[j + 1])
                unmutated = np.setdiff1d(segment, mutated)
                np.testing.assert_array_equal(h[unmutated], parent[unmutated])
                num_diffs = np.sum(parents[0][segment] != parents[1][segment])
                assert num_diffs == segment_diffs[j]
                assert num_diffs >= 2

    def test_breakpoint_intervals(self, query_set):
        df = query_set.dataframe()
        for row in df[df.kind == "recombinant"].itertuples():
            breakpoints = queries.parse_list(row.breakpoints)
            intervals = np.reshape(queries.parse_list(row.breakpoint_intervals), (-1, 2))
            for b, (lo, hi) in zip(breakpoints, intervals):
                assert lo < b <= hi

    def test_save_load(self, query_set, tmp_path):
        query_set.save(tmp_path)
        H, positions, df = queries.load_queries(tmp_path)
        np.testing.assert_array_equal(H, np.array(query_set.haplotypes))
        np.testing.assert_array_equal(positions, query_set.positions)
        assert len(df) == len(query_set.rows)
        row = df.iloc[0]
        assert queries.parse_list(row.mutated_sites) == []

    def test_choose_breakpoints_too_few_diffs(self):
        rng = np.random.default_rng(1)
        assert queries.choose_breakpoints(np.array([1, 5, 7]), 1, 2, rng) is None

    def test_mutate_skips_missing(self):
        rng = np.random.default_rng(1)
        h = np.array([0, -1, 1, -1, 2], dtype=np.int8)
        hq, sites = queries.mutate(h, 3, rng)
        assert list(sites) == [0, 2, 4]
        assert np.all(hq[sites] != h[sites])
        assert list(hq[[1, 3]]) == [-1, -1]


@pytest.fixture(scope="module", params=[None, 1])
def tsls_matcher(request, ts):
    mu, rho = matcher.solve_num_mismatches(3)
    return matcher.Matcher(ts, mu=mu, rho=rho, mismatch_threshold=request.param)


class TestMatcher:
    def check_path(self, ts, result):
        left, right, parent, _, _ = result
        assert left[0] == 0
        assert right[-1] == ts.sequence_length
        np.testing.assert_array_equal(left[1:], right[:-1])
        assert np.all(left < right)
        assert np.all(parent[1:] != parent[:-1])
        # The vestigial root is never copied
        assert 0 not in parent

    def test_exact_copy(self, ts, tsls_matcher, query_set):
        df = query_set.dataframe()
        for row in df[(df.kind == "copy") & (df.k == 0)].itertuples():
            h = query_set.haplotypes[row.query_id]
            result = tsls_matcher.match(h)
            self.check_path(ts, result)
            left, right, parent, matched, _ = result
            np.testing.assert_array_equal(matched, h)
            assert len(parent) == 1

    def test_all_queries(self, ts, tsls_matcher, query_set):
        for h in query_set.haplotypes:
            result = tsls_matcher.match(h)
            self.check_path(ts, result)
            left, right, parent, matched, stats = result
            path_h = matcher.path_haplotype(ts, left, right, parent)
            np.testing.assert_array_equal(matched, path_h)
            assert stats["total_memory"] > 0

    def test_copy_divergence(self, ts, query_set):
        # A copy with k mismatches has a path with at most k mismatches,
        # and k mismatches cost less than one recombination with k = 3.
        mu, rho = matcher.solve_num_mismatches(3)
        m = matcher.Matcher(ts, mu=mu, rho=rho)
        df = query_set.dataframe()
        for row in df[df.kind == "copy"].itertuples():
            h = query_set.haplotypes[row.query_id]
            _, _, parent, matched, _ = m.match(h)
            num_mismatches = np.sum(h != matched)
            cost = 3 * (len(parent) - 1) + num_mismatches
            assert cost <= row.k

    def test_weight_by_n(self, ts):
        # With weight_by_n=False, rho is the switch odds.
        mu, rho = matcher.solve_num_mismatches(3)
        m = matcher.Matcher(ts, mu=mu, rho=rho)
        q = tsls.switch_odds(m.recombination, ts.num_nodes)
        np.testing.assert_allclose(q, rho)
        m = matcher.Matcher(ts, mu=mu, rho=rho, weight_by_n=True)
        np.testing.assert_allclose(m.recombination, rho)


@pytest.fixture(scope="module")
def serial_ts():
    """
    A benchmark ARG with samples taken at several times, like sc2ts.
    """
    samples = [msprime.SampleSet(8, ploidy=1, time=t) for t in [0, 0.5, 1, 1.5]]
    ts, _ = prepare.prepare(simulate_sc2ts_like(42, samples=samples))
    return ts


class TestSubArgs:
    @pytest.mark.parametrize("num_samples", [1, 8, 16, 24, 32])
    def test_time_cutoff(self, serial_ts, num_samples):
        min_time = subargs.time_cutoff(serial_ts, num_samples)
        sub_ts, _ = subargs.sub_arg(serial_ts, min_time)
        assert sub_ts.num_samples >= num_samples
        assert np.all(sub_ts.nodes_time >= min_time)

    def test_too_many_samples(self, serial_ts):
        with pytest.raises(ValueError, match="Only"):
            subargs.time_cutoff(serial_ts, serial_ts.num_samples + 1)

    @pytest.mark.parametrize("min_time", [0, 0.5, 1, 1.5])
    def test_haplotypes_unchanged(self, serial_ts, min_time):
        sub_ts, node_ids = subargs.sub_arg(serial_ts, min_time)
        assert sub_ts.num_sites == serial_ts.num_sites
        np.testing.assert_array_equal(serial_ts.nodes_time[node_ids], sub_ts.nodes_time)
        np.testing.assert_array_equal(
            serial_ts.nodes_flags[node_ids], sub_ts.nodes_flags
        )
        # Non-sample nodes lose mutations in regions that are no longer
        # ancestral to any sample, so only sample haplotypes are unchanged.
        samples = sub_ts.samples()
        assert np.all(serial_ts.nodes_time[serial_ts.samples()] >= min_time) == (
            len(samples) == serial_ts.num_samples
        )
        np.testing.assert_array_equal(
            queries.node_haplotypes(sub_ts, samples),
            queries.node_haplotypes(serial_ts, node_ids[samples]),
        )
        for tree in sub_ts.trees():
            assert tree.root == 0
            assert list(tree.children(0)) == [1]
            # Every node in a tree has samples below it
            for u in tree.nodes():
                assert tree.num_samples(u) > 0

    def test_full_cutoff_is_identity(self, serial_ts):
        sub_ts, node_ids = subargs.sub_arg(serial_ts, 0)
        np.testing.assert_array_equal(node_ids, np.arange(serial_ts.num_nodes))
        sub_ts.tables.assert_equals(serial_ts.tables, ignore_provenance=True)

    def test_node_ids_round_trip(self, serial_ts, tmp_path):
        sub_ts, node_ids = subargs.sub_arg(serial_ts, 1)
        path = tmp_path / "sub.trees"
        sub_ts.dump(path)
        np.testing.assert_array_equal(
            subargs.load_node_ids(path, sub_ts), np.arange(sub_ts.num_nodes)
        )
        np.save(subargs.node_ids_path(path), node_ids)
        np.testing.assert_array_equal(subargs.load_node_ids(path, sub_ts), node_ids)

    def test_queries_report_original_ids(self, serial_ts):
        sub_ts, node_ids = subargs.sub_arg(serial_ts, 1)
        query_set = queries.make_queries(
            sub_ts,
            seed=1,
            num_nodes=4,
            divergence=(0,),
            num_recombinants=2,
            recombinant_divergence=(0,),
            node_ids=node_ids,
        )
        for row, h in zip(query_set.rows, query_set.haplotypes):
            nodes = queries.parse_list(row["nodes"])
            source = queries.node_haplotypes(serial_ts, nodes)
            if row["kind"] == "copy":
                np.testing.assert_array_equal(h, source[0])

    def test_matcher_runs(self, serial_ts):
        sub_ts, _ = subargs.sub_arg(serial_ts, 1)
        mu, rho = matcher.solve_num_mismatches(3)
        m = matcher.Matcher(sub_ts, mu=mu, rho=rho)
        for h in queries.node_haplotypes(sub_ts, sub_ts.samples()):
            _, _, _, matched, _ = m.match(h)
            np.testing.assert_array_equal(matched, h)


class TestParameters:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_solve_num_mismatches(self, k):
        # A recombination is slightly more likely than k mismatches.
        mu, rho = matcher.solve_num_mismatches(k)
        e = mu / (1 - 3 * mu)
        assert rho > e**k
        assert rho < e ** (k - 1)
