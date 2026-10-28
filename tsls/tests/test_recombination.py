"""
Tests for computing per-site recombination probabilities from a constant
rate or a genetic map.
"""

import msprime
import numpy as np
import pytest
import tskit

import tsls

_HAPMAP_HEADER = "Chromosome Position(bp) Rate(cM/Mb) Map(cM)"


def _write_map_rows(path, rows, header=_HAPMAP_HEADER):
    """Write a HapMap file from (position, rate cM/Mb, map cM) rows."""
    lines = [header]
    for pos, rate, cm in rows:
        lines.append(f"chr1 {pos} {rate!r} {cm!r}")
    path.write_text("\n".join(lines) + "\n")


def _write_hapmap(path, rate_map, omit_origin=False):
    """
    Write an msprime RateMap (rates per bp per generation) as a HapMap file.
    A leading missing (NaN) interval is written by starting the map at the
    end of that interval with 0 cM, which is how HapMap files represent it.
    If omit_origin is True, the row at position 0 is left out, so that the
    map starts at a non-zero genetic position and the rate over the first
    interval is implied.
    """
    left = rate_map.left
    rate = rate_map.rate
    cumulative_cm = rate_map.get_cumulative_mass(left) * 100
    rows = []
    for j in range(len(left)):
        if np.isnan(rate[j]) or (omit_origin and j == 0):
            continue
        rows.append((int(left[j]), float(rate[j] * 1e8), float(cumulative_cm[j])))
    final_cm = float(rate_map.get_cumulative_mass([rate_map.sequence_length])[0] * 100)
    rows.append((int(rate_map.sequence_length), 0.0, final_cm))
    _write_map_rows(path, rows)


# How a random map is laid out in the HapMap file: starting at position 0;
# starting later with 0 cM (missing data before it); or starting later at a
# non-zero genetic position (uniform rate from position 0 implied).
_MAP_LAYOUTS = ["origin", "missing_start", "omit_origin"]


def _random_rate_map(
    seed, sequence_length=100_000, num_intervals=10, missing_start=False
):
    """A random RateMap with integer breakpoints, including zero-rate intervals."""
    rng = np.random.default_rng(seed)
    breaks = rng.choice(np.arange(1, sequence_length), num_intervals - 1, replace=False)
    position = np.concatenate([[0], np.sort(breaks), [sequence_length]])
    rate = rng.exponential(1e-6, num_intervals)
    rate[rng.random(num_intervals) < 0.3] = 0
    if missing_start:
        rate[0] = np.nan
    return msprime.RateMap(position=position, rate=rate)


def _random_sites(seed, sequence_length, rate_map=None, num_sites=50):
    """Sorted unique site positions, including the map breakpoints if given."""
    rng = np.random.default_rng(seed + 1000)
    sites = rng.integers(0, sequence_length, num_sites)
    if rate_map is not None:
        sites = np.concatenate([sites, rate_map.left.astype(int)])
    return np.unique(sites).astype(np.int32)


def _haldane(distance):
    return (1 - np.exp(-2 * np.asarray(distance))) / 2


class TestReadGeneticMap:
    """Tests for tsls.model.read_genetic_map."""

    def _read(self, tmp_path, rows, positions):
        path = tmp_path / "map.txt"
        _write_map_rows(path, rows)
        return tsls.model.read_genetic_map(path, np.asarray(positions))

    # Nominal cases

    def test_values_at_map_entries(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        result = self._read(tmp_path, rows, [0, 1000, 3000])
        np.testing.assert_allclose(result, [0, 0.001, 0.005])

    def test_linear_interpolation(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        result = self._read(tmp_path, rows, [500, 2000, 2500])
        np.testing.assert_allclose(result, [0.0005, 0.003, 0.004])

    def test_returns_morgans(self, tmp_path):
        rows = [(0, 1e6, 0.0), (100, 0.0, 100.0)]
        result = self._read(tmp_path, rows, [100])
        np.testing.assert_allclose(result, [1.0])

    def test_rate_column_ignored(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 2.0, 0.1), (3000, 0.0, 0.5)]
        garbage = [(pos, -99.0, cm) for pos, _, cm in rows]
        positions = [0, 500, 2000, 3000]
        expected = self._read(tmp_path, rows, positions)
        result = self._read(tmp_path, garbage, positions)
        np.testing.assert_array_equal(result, expected)

    def test_path_as_string(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (100, 0.0, 1.0)])
        result = tsls.model.read_genetic_map(str(path), np.array([50]))
        np.testing.assert_allclose(result, [0.005])

    # Corner cases

    def test_zero_rate_interval_is_flat(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 0.0, 1.0), (200, 1.0, 1.0), (300, 0.0, 2.0)]
        result = self._read(tmp_path, rows, [100, 150, 200])
        np.testing.assert_allclose(result, [0.01, 0.01, 0.01])

    def test_flat_before_map_starting_at_zero_cm(self, tmp_path):
        rows = [(1000, 1.0, 0.0), (2000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000, 1500])
        np.testing.assert_allclose(result, [0, 0, 0, 0.005])

    def test_linear_from_origin_before_map_with_offset(self, tmp_path):
        # A map starting at a non-zero genetic position implies a uniform
        # rate from position 0, as in msprime.
        rows = [(1000, 1.0, 2.0), (2000, 0.0, 3.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000, 1500])
        np.testing.assert_allclose(result, [0, 0.01, 0.02, 0.025])

    def test_offset_at_position_zero_preserved(self, tmp_path):
        # Only differences in genetic position matter, so an offset at
        # position zero is harmless.
        rows = [(0, 1.0, 5.0), (1000, 0.0, 6.0)]
        result = self._read(tmp_path, rows, [0, 500, 1000])
        np.testing.assert_allclose(result, [0.05, 0.055, 0.06])

    def test_flat_beyond_last_entry(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, [1000, 1001, 10**9])
        np.testing.assert_allclose(result, [0.01, 0.01, 0.01])

    def test_single_row_at_origin(self, tmp_path):
        result = self._read(tmp_path, [(0, 0.0, 0.0)], [0, 100, 10**6])
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_single_row_after_origin(self, tmp_path):
        result = self._read(tmp_path, [(100, 0.0, 1.0)], [0, 50, 100, 200])
        np.testing.assert_allclose(result, [0, 0.005, 0.01, 0.01])

    def test_no_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (1000, 0.0, 1.0)]
        result = self._read(tmp_path, rows, np.array([], dtype=np.int32))
        assert result.shape == (0,)

    def test_large_int32_positions(self, tmp_path):
        top = 2**31 - 1
        rows = [(0, 1.0, 0.0), (top, 0.0, 100.0)]
        positions = np.array([top - 1000, top], dtype=np.int32)
        result = self._read(tmp_path, rows, positions)
        np.testing.assert_allclose(np.diff(result), [1000 / top])

    def test_tab_separated_extra_columns(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text(
            "chr\tpos\trate\tcm\textra\nchr1\t0\t1.0\t0.0\tx\nchr1\t100\t0.0\t1.0\ty\n"
        )
        result = tsls.model.read_genetic_map(path, np.array([50]))
        np.testing.assert_allclose(result, [0.005])

    # Errors

    @pytest.mark.filterwarnings("ignore:loadtxt. input contained no data")
    def test_empty_map(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text(_HAPMAP_HEADER + "\n")
        with pytest.raises(ValueError, match="empty"):
            tsls.model.read_genetic_map(path, np.array([10]))

    def test_unsorted_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (200, 1.0, 2.0), (100, 0.0, 1.0)]
        with pytest.raises(ValueError, match="not strictly increasing"):
            self._read(tmp_path, rows, [50])

    def test_duplicate_positions(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 1.0, 1.0), (100, 0.0, 1.0)]
        with pytest.raises(ValueError, match="not strictly increasing"):
            self._read(tmp_path, rows, [50])

    def test_decreasing_map(self, tmp_path):
        rows = [(0, 1.0, 0.0), (100, 1.0, 2.0), (200, 0.0, 1.0)]
        with pytest.raises(ValueError, match="decrease"):
            self._read(tmp_path, rows, [50])

    def test_missing_map_column(self, tmp_path):
        path = tmp_path / "map.txt"
        path.write_text("chr pos rate\nchr1 0 1.0\nchr1 100 0.0\n")
        with pytest.raises(ValueError, match="column"):
            tsls.model.read_genetic_map(path, np.array([50]))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tsls.model.read_genetic_map(tmp_path / "nope.txt", np.array([50]))

    # msprime oracle

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_rate_map(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        sites = _random_sites(seed, rate_map.sequence_length, rate_map)
        result = tsls.model.read_genetic_map(path, sites)
        expected = rate_map.get_cumulative_mass(sites)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_msprime_read_hapmap(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        oracle = msprime.RateMap.read_hapmap(path, map_col=3)
        sites = _random_sites(seed, oracle.sequence_length, oracle)
        result = tsls.model.read_genetic_map(path, sites)
        np.testing.assert_allclose(
            result, oracle.get_cumulative_mass(sites), rtol=1e-12, atol=1e-15
        )

    def test_matches_msprime_beyond_last_entry(self, tmp_path):
        rows = [(100, 2.0, 0.0), (5000, 0.0, 0.0098)]
        path = tmp_path / "map.txt"
        _write_map_rows(path, rows)
        oracle = msprime.RateMap.read_hapmap(path, map_col=3, sequence_length=20_000)
        sites = np.array([0, 50, 100, 2500, 5000, 12_000, 20_000], dtype=np.int32)
        result = tsls.model.read_genetic_map(path, sites)
        np.testing.assert_allclose(result, oracle.get_cumulative_mass(sites))


class TestRecombinationProbabilities:
    """Tests for tsls.model.recombination_probabilities."""

    positions = np.array([10, 200, 250, 1000, 5000, 9000], dtype=np.int32)

    # Nominal cases

    def test_constant_rate(self):
        rate = 1e-5
        probs = tsls.model.recombination_probabilities(rate, self.positions)
        assert probs.shape == (6,)
        assert probs[0] == 0
        np.testing.assert_allclose(probs[1:], _haldane(np.diff(self.positions) * rate))

    def test_small_distances_approximately_linear(self):
        probs = tsls.model.recombination_probabilities(1e-8, self.positions)
        np.testing.assert_allclose(probs[1:], np.diff(self.positions) * 1e-8, rtol=1e-4)

    def test_known_value(self):
        # 0.5 Morgans between the sites
        probs = tsls.model.recombination_probabilities(0.005, np.array([0, 100]))
        np.testing.assert_allclose(probs, [0, (1 - np.exp(-1)) / 2])

    def test_proportional_to_spacing_for_small_rate(self):
        positions = np.array([0, 100, 300, 600])
        probs = tsls.model.recombination_probabilities(1e-10, positions)
        np.testing.assert_allclose(probs[2] / probs[1], 2, rtol=1e-6)
        np.testing.assert_allclose(probs[3] / probs[1], 3, rtol=1e-6)

    def test_genetic_map(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (1000, 3.0, 0.1), (3000, 0.0, 0.7)])
        positions = np.array([0, 500, 1000, 2000, 3000])
        probs = tsls.model.recombination_probabilities(path, positions)
        distance = np.array([0.0005, 0.0005, 0.003, 0.003])
        np.testing.assert_allclose(probs, np.concatenate([[0], _haldane(distance)]))

    def test_genetic_map_path_as_string(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (10_000, 0.0, 1.0)])
        from_str = tsls.model.recombination_probabilities(str(path), self.positions)
        from_path = tsls.model.recombination_probabilities(path, self.positions)
        np.testing.assert_array_equal(from_str, from_path)
        np.testing.assert_allclose(
            from_path[1:], _haldane(np.diff(self.positions) * 1e-6)
        )

    def test_uniform_map_equals_constant_rate(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 2.0, 0.0), (10_000, 0.0, 0.02)])
        from_map = tsls.model.recombination_probabilities(path, self.positions)
        from_rate = tsls.model.recombination_probabilities(2e-8, self.positions)
        np.testing.assert_allclose(from_map, from_rate)

    # Corner cases

    def test_zero_rate(self):
        probs = tsls.model.recombination_probabilities(0.0, self.positions)
        np.testing.assert_array_equal(probs, np.zeros(6))

    def test_negative_rate(self):
        with pytest.raises(ValueError, match="non-negative"):
            tsls.model.recombination_probabilities(-1e-8, self.positions)

    def test_single_site(self):
        probs = tsls.model.recombination_probabilities(1e-8, np.array([100]))
        np.testing.assert_array_equal(probs, [0])

    def test_no_sites(self):
        probs = tsls.model.recombination_probabilities(
            1e-8, np.array([], dtype=np.int32)
        )
        assert probs.shape == (0,)

    def test_saturates_at_half(self):
        probs = tsls.model.recombination_probabilities(1.0, self.positions)
        assert np.all(probs <= 0.5)
        np.testing.assert_allclose(probs[1:], 0.5)

    def test_bounded_and_monotonic_in_distance(self):
        positions = np.cumsum(np.arange(1, 50)) * 1000
        probs = tsls.model.recombination_probabilities(1e-5, positions)
        assert np.all(probs >= 0)
        assert np.all(probs <= 0.5)
        assert np.all(np.diff(probs[1:]) > 0)

    def test_large_int32_positions_do_not_overflow(self):
        positions = np.array([0, 2**31 - 1], dtype=np.int32)
        probs = tsls.model.recombination_probabilities(1e-10, positions)
        np.testing.assert_allclose(probs[1], _haldane((2**31 - 1) * 1e-10))

    def test_sites_in_zero_rate_region(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 0.0, 0.0), (5000, 1.0, 0.0), (6000, 0.0, 0.1)])
        probs = tsls.model.recombination_probabilities(path, np.array([10, 2000, 4999]))
        np.testing.assert_array_equal(probs, [0, 0, 0])

    def test_sites_beyond_map(self, tmp_path):
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1.0, 0.0), (1000, 0.0, 1.0)])
        positions = np.array([500, 1500, 2500])
        probs = tsls.model.recombination_probabilities(path, positions)
        np.testing.assert_allclose(probs, [0, _haldane(0.005), 0])

    # msprime oracle

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("layout", _MAP_LAYOUTS)
    def test_matches_rate_map(self, tmp_path, seed, layout):
        rate_map = _random_rate_map(seed, missing_start=layout == "missing_start")
        path = tmp_path / "map.txt"
        _write_hapmap(path, rate_map, omit_origin=layout == "omit_origin")
        sites = _random_sites(seed, rate_map.sequence_length)
        probs = tsls.model.recombination_probabilities(path, sites)
        distance = np.diff(rate_map.get_cumulative_mass(sites))
        assert probs[0] == 0
        np.testing.assert_allclose(probs[1:], _haldane(distance), rtol=1e-9, atol=1e-15)

    @pytest.mark.parametrize("rate", [0.0, 1e-9, 1e-8, 1e-6])
    def test_constant_rate_matches_rate_map(self, rate):
        sites = _random_sites(0, 1_000_000)
        rate_map = msprime.RateMap.uniform(1_000_000, rate)
        probs = tsls.model.recombination_probabilities(rate, sites)
        # The oracle differences cumulative masses of up to ~1, which loses
        # ~1e-16 in absolute terms, so compare with an absolute tolerance too.
        distance = np.diff(rate_map.get_cumulative_mass(sites))
        np.testing.assert_allclose(probs[1:], _haldane(distance), rtol=1e-12, atol=1e-15)


class TestHmmWithRecombinationRate:
    """
    Per-site probabilities from a rate change the path tsls.Hmm finds. The
    recombinant [1, 1, 1] is matched against A, carrying the derived allele at
    the first two sites, and B at the third.
    """

    positions = np.array([10, 20, 30])

    def reference_ts(self):
        tables = tskit.TableCollection(sequence_length=100)
        root = tables.nodes.add_row(time=1)
        a = tables.nodes.add_row(time=0)
        b = tables.nodes.add_row(time=0)
        for child in [a, b]:
            tables.edges.add_row(left=0, right=100, parent=root, child=child)
        for position in self.positions:
            tables.sites.add_row(position=position, ancestral_state="A")
        for site, node in [(0, a), (1, a), (2, b)]:
            tables.mutations.add_row(site=site, node=node, derived_state="T")
        tables.sort()
        return tables.tree_sequence(), a, b

    def find_path(self, recombination, mismatch):
        ts, _, _ = self.reference_ts()
        hmm = tsls.Hmm(ts, recombination, np.full(3, mismatch))
        matched = np.zeros(3, dtype=np.int8)
        h = np.array([1, 1, 1], dtype=np.int8)
        left, _, parent = hmm.find_path(h, 0, 3, matched)
        return left[::-1], parent[::-1], matched

    def test_high_recombination_prefers_switch(self):
        _, a, b = self.reference_ts()
        r = tsls.model.recombination_probabilities(1e-3, self.positions)
        left, parent, matched = self.find_path(r, 1e-20)
        assert list(parent) == [a, b]
        assert list(left) == [0, 30]
        np.testing.assert_array_equal(matched, [1, 1, 1])

    def test_low_recombination_rate_prefers_mismatch(self):
        _, a, _ = self.reference_ts()
        r = tsls.model.recombination_probabilities(1e-12, self.positions)
        _, parent, matched = self.find_path(r, 1e-2)
        assert list(parent) == [a]
        np.testing.assert_array_equal(matched, [1, 1, 0])

    def test_genetic_map(self, tmp_path):
        # The map has no recombination between the second and third sites,
        # so the switch must happen between the first and second.
        _, a, b = self.reference_ts()
        path = tmp_path / "map.txt"
        _write_map_rows(path, [(0, 1e4, 0.0), (15, 0.0, 10.0), (100, 0.0, 10.0)])
        r = tsls.model.recombination_probabilities(path, self.positions)
        assert r[2] == 0
        # Make B also carry the derived allele at the second site.
        ts, _, _ = self.reference_ts()
        tables = ts.dump_tables()
        tables.mutations.add_row(site=1, node=b, derived_state="T")
        tables.sort()
        hmm = tsls.Hmm(tables.tree_sequence(), r, np.full(3, 1e-20))
        matched = np.zeros(3, dtype=np.int8)
        h = np.array([1, 1, 1], dtype=np.int8)
        left, _, parent = hmm.find_path(h, 0, 3, matched)
        assert list(parent[::-1]) == [a, b]
        assert list(left[::-1]) == [0, 20]
