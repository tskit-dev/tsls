import json
import pathlib

import msprime
import numpy as np
import pandas as pd
import pytest
import tskit
import tsls
from click.testing import CliRunner

import matcher
import queries
import run
import tsinfer_data

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "tgp_chr20p"
TSINFER_MODEL = {"rho": 0.01, "mu": 1e-20, "weight_by_n": True, "alleles": None}


def simulate_tsinfer_like(seed, samples=20):
    """
    Simulate a haploid ARG with the structure of a tsinfer ancestors tree
    sequence: nodes in decreasing time order, starting with the ultimate
    root 0 and the virtual root 1, which is the root of every tree, and
    biallelic sites with a single 0 -> 1 mutation each.
    """
    ts = msprime.sim_ancestry(
        samples,
        ploidy=1,
        sequence_length=1000,
        recombination_rate=1e-3,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(
        ts,
        rate=5e-3,
        model=msprime.BinaryMutationModel(),
        random_seed=seed,
    )
    tables = ts.dump_tables()
    num_mutations = np.bincount(tables.mutations.site, minlength=ts.num_sites)
    tables.delete_sites(np.where(num_mutations != 1)[0])
    max_time = np.max(tables.nodes.time)
    tables.nodes.clear()
    tables.nodes.add_row(time=max_time + 2)
    tables.nodes.add_row(time=max_time + 1)
    for node in ts.tables.nodes:
        tables.nodes.append(node.replace(flags=0))
    tables.edges.parent += 2
    tables.edges.child += 2
    tables.mutations.node += 2
    for tree in ts.trees():
        for root in tree.roots:
            tables.edges.add_row(*tree.interval, parent=1, child=root + 2)
    tables.edges.add_row(0, ts.sequence_length, parent=0, child=1)
    tables.edges.squash()
    tables.sort()
    ts = tables.tree_sequence()
    order = np.argsort(-ts.nodes_time, kind="stable")
    ts = ts.subset(order)
    assert ts.num_sites > 20
    assert np.all(np.diff(ts.nodes_time) <= 0)
    return ts


@pytest.fixture(scope="module")
def ts():
    return simulate_tsinfer_like(42)


def node_haplotypes(ts, nodes):
    return ts.genotype_matrix(samples=nodes).T.astype(np.int8)


def write_genetic_map(path, rows):
    """Write a HapMap file from (position, map cM) rows."""
    lines = ["Chromosome Position(bp) Rate(cM/Mb) Map(cM)"]
    lines += [f"chr1 {pos} 0 {cm}" for pos, cm in rows]
    path.write_text("\n".join(lines) + "\n")


# A map with a recombination hotspot in the middle of the 1000 bp sequence
# and no recombination at the end.
MAP_ROWS = [(0, 0), (400, 0.1), (500, 5), (800, 5.5), (900, 6), (1000, 6)]


class TestCutBefore:
    @pytest.mark.parametrize("cut", [2, 10, 25])
    def test_haplotypes_unchanged(self, ts, cut):
        sub = tsinfer_data.cut_before(ts, cut)
        assert sub.num_nodes == cut
        assert sub.num_sites == ts.num_sites
        nodes = np.arange(cut)
        np.testing.assert_array_equal(
            node_haplotypes(sub, nodes), node_haplotypes(ts, nodes)
        )

    def test_full_cut_is_identity(self, ts):
        sub = tsinfer_data.cut_before(ts, ts.num_nodes)
        sub.tables.assert_equals(ts.tables)

    def test_later_parent_fails(self, ts):
        # Youngest nodes first, so children come before their parents.
        reversed_ts = ts.subset(np.arange(ts.num_nodes)[::-1])
        with pytest.raises(ValueError, match="later node"):
            tsinfer_data.cut_before(reversed_ts, ts.num_nodes // 2)


class TestTruthPaths:
    def test_paths_cover_edges(self, ts):
        nodes = np.arange(2, ts.num_nodes)
        paths = tsinfer_data.truth_paths(ts, nodes)
        for node, (left, parent) in zip(nodes, paths):
            edges = ts.tables.edges
            keep = edges.child == node
            assert len(left) == np.sum(keep)
            assert np.all(np.diff(left) > 0)
            assert set(zip(left, parent)) == set(
                zip(edges.left[keep], edges.parent[keep])
            )

    def test_columns(self):
        paths = [(np.array([0.0, 5.0]), np.array([3, 4])), (np.array([2.0]), [1])]
        columns = tsinfer_data.truth_columns(paths)
        assert columns["truth_num_segments"] == [2, 1]
        assert columns["truth_parents"] == ["3 4", "1"]
        assert columns["truth_breakpoints"] == ["5", ""]


class TestWindowedHaplotypes:
    def make(self):
        windows = [np.array([0, 1, 1]), np.array([], dtype=np.int8), np.array([1])]
        return queries.WindowedHaplotypes.from_windows(6, [1, 0, 5], windows)

    def test_getitem(self):
        H = self.make()
        assert len(H) == 3
        np.testing.assert_array_equal(H[0], [-1, 0, 1, 1, -1, -1])
        np.testing.assert_array_equal(H[1], [-1] * 6)
        np.testing.assert_array_equal(H[2], [-1, -1, -1, -1, -1, 1])
        assert queries.query_window(H, 0) == (1, 4)
        assert queries.query_window(H, 2) == (5, 6)

    def test_dense_window(self):
        H = np.zeros((2, 7), dtype=np.int8)
        assert queries.query_window(H, 1) == (0, 7)

    def test_save_load(self, tmp_path):
        H = self.make()
        df = pd.DataFrame({"query_id": np.arange(3)})
        H.save(tmp_path, np.arange(6) * 10, df)
        H2, positions, df2 = queries.load_queries(tmp_path)
        np.testing.assert_array_equal(positions, np.arange(6) * 10)
        for j in range(3):
            np.testing.assert_array_equal(H[j], H2[j])
        assert len(df2) == 3

    def test_non_missing_window(self):
        assert tsinfer_data.non_missing_window(np.array([-1, 0, -1, 1, -1])) == (1, 4)
        assert tsinfer_data.non_missing_window(np.array([-1, -1])) == (0, 2)


class TestTsinferModel:
    def make_matcher(self, ts):
        return matcher.Matcher(
            ts,
            mu=TSINFER_MODEL["mu"],
            rho=TSINFER_MODEL["rho"],
            weight_by_n=True,
            alleles=None,
        )

    @pytest.mark.parametrize("window", [(0, None), (5, 15), (10, 11)])
    def test_exact_copy(self, ts, window):
        m = self.make_matcher(ts)
        start, end = window
        if end is None:
            end = ts.num_sites
        positions = ts.sites_position
        for node in range(ts.num_nodes - 5, ts.num_nodes):
            h = np.full(ts.num_sites, -1, dtype=np.int8)
            h[start:end] = node_haplotypes(ts, [node])[0, start:end]
            left, right, _, matched, _ = m.match(h, start, end)
            # As in tsinfer, a path starting at the first site starts at 0.
            assert left[0] == (0 if start == 0 else positions[start])
            if end < ts.num_sites:
                assert right[-1] == positions[end]
            else:
                assert right[-1] == ts.sequence_length
            np.testing.assert_array_equal(left[1:], right[:-1])
            assert np.all(matched[start:end] == h[start:end])

    def test_genetic_map(self, ts, tmp_path):
        path = tmp_path / "map.txt"
        write_genetic_map(path, MAP_ROWS)
        rho = tsls.recombination_probabilities(path, ts.sites_position)
        m = matcher.Matcher(ts, mu=1e-3, rho=rho, weight_by_n=True, alleles=None)
        np.testing.assert_array_equal(m.recombination, rho)
        rng = np.random.default_rng(2)
        for _ in range(5):
            a, b = rng.choice(np.arange(ts.num_nodes - 10, ts.num_nodes), 2)
            G = node_haplotypes(ts, [a, b])
            breakpoint = rng.integers(1, ts.num_sites)
            h = np.concatenate([G[0, :breakpoint], G[1, breakpoint:]])
            left, right, parent, matched, _ = m.match(h)
            path_h = matcher.path_haplotype(ts, left, right, parent, alleles=None)
            np.testing.assert_array_equal(matched, path_h)

    def test_mismatches(self, ts):
        # With mu = 1e-20, three flipped sites are explained by switches or
        # mismatches, and the matched haplotype is the path's.
        m = self.make_matcher(ts)
        rng = np.random.default_rng(1)
        for node in range(ts.num_nodes - 5, ts.num_nodes):
            h = node_haplotypes(ts, [node])[0]
            flip = rng.choice(ts.num_sites, size=3, replace=False)
            h[flip] = 1 - h[flip]
            left, right, parent, matched, _ = m.match(h)
            path_h = matcher.path_haplotype(ts, left, right, parent, alleles=None)
            np.testing.assert_array_equal(matched, path_h)
            assert np.sum(matched != h) <= 3


class TestRunWithModel:
    @pytest.fixture
    def workload(self, ts, tmp_path):
        reference = tsinfer_data.cut_before(ts, ts.num_nodes - 5)
        nodes = np.arange(ts.num_nodes - 5, ts.num_nodes)
        G = node_haplotypes(ts, nodes)
        starts = [0, 3, 10, 0, 7]
        windows = [G[j, s : s + 10] for j, s in enumerate(starts)]
        H = queries.WindowedHaplotypes.from_windows(ts.num_sites, starts, windows)
        df = pd.DataFrame(
            {
                "query_id": np.arange(len(nodes)),
                "kind": "ancestor",
                "sample_id": [f"a{u}" for u in nodes],
                "ploidy_index": 0,
                "node": nodes,
                "group": 0,
                "start": H.start,
                "end": H.end,
                **tsinfer_data.truth_columns(tsinfer_data.truth_paths(ts, nodes)),
            }
        )
        resources = {"group": 0, "num_haplotypes": 5, "user_time": 1.0}
        tsinfer_data.save_workload(
            tmp_path / "w", reference, H, df, TSINFER_MODEL, resources
        )
        return tmp_path / "w"

    def test_run(self, workload, tmp_path):
        out = tmp_path / "results.csv"
        result = CliRunner().invoke(
            run.main, [str(workload / "reference.trees"), str(workload), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert len(df) == 5
        assert np.all(df.status == "ok")
        assert np.all(df.rho == 0.01)
        assert np.all(df.log_likelihood.isna())
        H, positions, _ = queries.load_queries(workload)
        for row in df.itertuples():
            start, end = queries.query_window(H, row.query_id)
            breakpoints = queries.parse_list(row.breakpoints)
            assert len(breakpoints) == row.num_segments - 1
            assert np.all(np.isin(breakpoints, positions[start + 1 : end]))

    def run(self, workload, out, *args):
        return CliRunner().invoke(
            run.main,
            [str(workload / "reference.trees"), str(workload), "-o", str(out), *args],
        )

    def test_constant_recombination_rate(self, workload, tmp_path):
        out = tmp_path / "results.csv"
        result = self.run(
            workload, out, "--recombination-rate", "1e-4", "--mismatch", "1e-3"
        )
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert np.all(df.status == "ok")
        assert np.all(df.recombination_rate == 1e-4)
        assert np.all(df.mu == 1e-3)
        positions = tskit.load(workload / "reference.trees").sites_position
        rho = tsls.recombination_probabilities(1e-4, positions)
        np.testing.assert_allclose(df.rho, np.mean(rho[1:]))

    def test_genetic_map(self, workload, tmp_path):
        path = tmp_path / "map.txt"
        write_genetic_map(path, MAP_ROWS)
        out = tmp_path / "results.csv"
        result = self.run(workload, out, "--recombination-rate", str(path))
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert np.all(df.status == "ok")
        assert np.all(df.recombination_rate == str(path))
        assert np.all(df.mu == 1e-20)

    def test_recombination_rate_in_model(self, workload, tmp_path):
        path = tmp_path / "map.txt"
        write_genetic_map(path, MAP_ROWS)
        with open(workload / "model.json") as f:
            model = json.load(f)
        model["rho"] = None
        model["recombination_rate"] = str(path)
        with open(workload / "model.json", "w") as f:
            json.dump(model, f)
        out = tmp_path / "results.csv"
        result = self.run(workload, out)
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert np.all(df.recombination_rate == str(path))
        expected = tmp_path / "expected.csv"
        model["recombination_rate"] = None
        model["rho"] = 0.01
        with open(workload / "model.json", "w") as f:
            json.dump(model, f)
        result = self.run(workload, expected, "--recombination-rate", str(path))
        assert result.exit_code == 0, result.output
        other = pd.read_csv(expected)
        assert list(df.parents) == list(other.parents)
        np.testing.assert_array_equal(df.rho, other.rho)

    def test_bad_recombination_rate(self, workload, tmp_path):
        out = tmp_path / "results.csv"
        result = self.run(workload, out, "--recombination-rate", "nofile")
        assert result.exit_code != 0
        assert "not a number or a file" in result.output

    def test_old_results_rejected(self, workload, tmp_path):
        out = tmp_path / "results.csv"
        columns = [c for c in run.RESULT_COLUMNS if c != "recombination_rate"]
        pd.DataFrame({c: [0] for c in columns}).to_csv(out, index=False)
        result = self.run(workload, out)
        assert result.exit_code != 0
        assert "older run.py" in result.output

    def test_resume(self, workload, tmp_path):
        out = tmp_path / "results.csv"
        result = self.run(workload, out, "--limit", "2")
        assert result.exit_code == 0, result.output
        assert len(pd.read_csv(out)) == 2
        result = self.run(workload, out)
        assert result.exit_code == 0, result.output
        df = pd.read_csv(out)
        assert list(df.query_id) == [0, 1, 2, 3, 4]

    def test_num_mismatches_rejected(self, workload, tmp_path):
        result = CliRunner().invoke(
            run.main,
            [
                str(workload / "reference.trees"),
                str(workload),
                "-o",
                str(tmp_path / "out.csv"),
                "--num-mismatches",
                "4",
            ],
        )
        assert result.exit_code != 0

    def test_load_model_recombination_rate(self, tmp_path):
        (tmp_path / tsinfer_data.CONFIG).write_text(
            "[match.sources.samples]\nrecombination_rate = 'map.txt'\nmismatch = 1e-4\n"
        )
        model = tsinfer_data.load_model(tmp_path, "samples")
        assert model["rho"] is None
        assert model["recombination_rate"] == "map.txt"
        assert model["mu"] == 1e-4

    def test_load_model_both_recombination(self, tmp_path):
        (tmp_path / tsinfer_data.CONFIG).write_text(
            "[match.sources.samples]\nrecombination = 0.01\nrecombination_rate = 1e-8\n"
        )
        with pytest.raises(ValueError, match="both"):
            tsinfer_data.load_model(tmp_path, "samples")

    def test_load_model(self, workload, tmp_path):
        assert queries.load_model(workload) == TSINFER_MODEL
        assert queries.load_model(tmp_path) is None
        with open(workload / "tsinfer_group.json") as f:
            assert json.load(f)["group"] == 0


needs_data = pytest.mark.skipif(
    not (DATA_DIR / tsinfer_data.MATCH_JOBS).exists(),
    reason="Needs data/tgp_chr20p with match_jobs.csv (see README)",
)


@pytest.fixture(scope="module")
def workload_g84():
    return tsinfer_data.make_ancestor_workload(DATA_DIR, 84, num_queries=10)


@needs_data
class TestTgpChr20p:
    """
    Checks against the partial ancestor match supplied with the data: the
    tree sequence before group 84 and the paths that tsinfer found for it.
    """

    @pytest.fixture
    def workload(self, workload_g84):
        return workload_g84

    def test_reference_is_before(self, workload):
        reference = workload[0]
        before = tskit.load(DATA_DIR / "partial_ancestor_match" / "before.trees")
        reference.tables.assert_equals(before.tables, ignore_provenance=True)

    def test_resources(self, workload):
        resources = workload[3]
        assert resources["group"] == 84
        assert resources["num_haplotypes"] == 144

    def test_paths_match_tsinfer(self, workload):
        reference, H, df, _ = workload
        after = tskit.load(DATA_DIR / "partial_ancestor_match" / "after.trees")
        m = matcher.Matcher(
            reference, mu=1e-20, rho=0.01, weight_by_n=True, alleles=None
        )
        for row in df.itertuples():
            assert after.node(row.node).metadata["sample_id"] == row.sample_id
            start, end = queries.query_window(H, row.query_id)
            left, _, parent, _, _ = m.match(H[row.query_id], start, end)
            ((truth_left, truth_parent),) = tsinfer_data.truth_paths(after, [row.node])
            np.testing.assert_array_equal(left, truth_left)
            np.testing.assert_array_equal(parent, truth_parent)
