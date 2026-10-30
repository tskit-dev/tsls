"""
Regression tests on cases captured from the benchmark workloads when the
tsinfer-hmm engine was still in this repository (the capture script,
benchmarking/capture_regression.py, and the benchmarks are in the git
history).

Each case in data/regression has a gzipped reference tree sequence, query
haplotypes with their windows, the per-site recombination and mismatch
probabilities, and the outputs of tsls: the Viterbi path, the sites where
it mismatches, its log likelihood, and for some cases the forward log
likelihood and posteriors. When the cases were captured, tsinfer-hmm found
paths with the same likelihood for every query, and the same path except
where same_as_tsinfer is False (ties between equally likely paths). We
require the likelihood and number of mismatches to be the same, and the
path to be the same wherever the engines agreed.
"""

import gzip
import json
import pathlib
import shutil

import numpy as np
import pytest
import tskit

import tsls

DATA_DIR = pathlib.Path(__file__).parent / "data" / "regression"
CASES = sorted(path.name for path in DATA_DIR.iterdir())


class Case:
    def __init__(self, name, tmp_dir):
        case_dir = DATA_DIR / name
        trees_path = tmp_dir / f"{name}.trees"
        gz_path = case_dir / "reference.trees.gz"
        with gzip.open(gz_path, "rb") as f_in, open(trees_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        self.ts = tskit.load(trees_path)
        with open(case_dir / "case.json") as f:
            self.alleles = json.load(f)["alleles"]
        with np.load(case_dir / "case.npz") as data:
            self.data = dict(data)
        self.reference = tsls.TreeSequence(self.ts, alleles=self.alleles)

    @property
    def num_queries(self):
        return len(self.data["haplotypes"])

    def query(self, j):
        d = self.data
        return d["haplotypes"][j], int(d["start"][j]), int(d["end"][j])

    def path(self, j):
        d = self.data
        a, b = d["path_offset"][j], d["path_offset"][j + 1]
        return d["path_left"][a:b], d["path_right"][a:b], d["path_parent"][a:b]

    def mismatch_sites(self, j):
        d = self.data
        a, b = d["mismatch_offset"][j], d["mismatch_offset"][j + 1]
        return d["mismatch_site"][a:b]

    def hmm(self):
        return tsls.Hmm(
            self.reference, self.data["recombination"], self.data["mismatch"]
        )

    def num_alleles(self):
        """
        The number of alleles at each site, as tsls counts them: the length
        of the allele list, or the number of distinct states, at least 2.
        """
        if self.alleles is not None:
            return np.full(self.ts.num_sites, len(self.alleles))
        num_alleles = np.zeros(self.ts.num_sites, dtype=int)
        for site in self.ts.sites():
            states = {site.ancestral_state}
            states |= {m.derived_state for m in site.mutations}
            num_alleles[site.id] = max(2, len(states))
        return num_alleles


@pytest.fixture(scope="module", params=CASES)
def case(request, tmp_path_factory):
    return Case(request.param, tmp_path_factory.mktemp("regression"))


def path_log_likelihood(case, h, start, end, path, matched):
    """
    Return the log likelihood of the specified path (in left-to-right
    order) over sites [start, end), leaving out the initial probability,
    which is the same for every path.
    """
    left = path[0]
    rho = case.data["recombination"]
    mu = case.data["mismatch"]
    n = case.ts.num_nodes
    num_alleles = case.num_alleles()
    sites = np.arange(start, end)
    switch = np.isin(case.ts.sites_position[sites], left[1:])
    switch[0] = False
    r = rho[sites]
    transition = np.where(switch, np.log(r / n), np.log(1 - r + r / n))
    transition[0] = 0
    hs = h[sites]
    match = (hs == -1) | (hs == matched[sites])
    m = mu[sites]
    emission = np.where(match, np.log(1 - (num_alleles[sites] - 1) * m), np.log(m))
    return np.sum(transition) + np.sum(emission)


def path_haplotypes(case, paths):
    """
    Return the (num_sites, len(paths)) array of the alleles of the nodes on
    each of the specified paths.
    """
    ts = case.ts
    nodes = np.unique(np.concatenate([parent for _, _, parent in paths]))
    alleles = None if case.alleles is None else tuple(case.alleles)
    G = ts.genotype_matrix(samples=nodes, alleles=alleles, isolated_as_missing=False)
    positions = ts.sites_position
    H = np.zeros((ts.num_sites, len(paths)), dtype=np.int8)
    for k, (left, _, parent) in enumerate(paths):
        segment = np.searchsorted(left, positions, side="right") - 1
        segment = np.clip(segment, 0, len(parent) - 1)
        column = np.searchsorted(nodes, parent[segment])
        H[:, k] = G[np.arange(ts.num_sites), column]
    return H


def find_path(case, j):
    h, start, end = case.query(j)
    matched = np.zeros(case.ts.num_sites, dtype=np.int8)
    left, right, parent = case.hmm().find_path(h, start, end, matched)
    path = left[::-1], right[::-1], parent[::-1]
    return path, matched


class TestViterbi:
    def test_log_likelihood(self, case):
        for j in range(case.num_queries):
            h, start, end = case.query(j)
            path, matched = find_path(case, j)
            ll = path_log_likelihood(case, h, start, end, path, matched)
            expected = case.data["log_likelihood"][j]
            assert ll == pytest.approx(expected, rel=1e-9)

    def test_num_mismatches(self, case):
        for j in range(case.num_queries):
            h, start, end = case.query(j)
            _, matched = find_path(case, j)
            sites = np.arange(start, end)
            observed = sites[h[sites] != -1]
            num_mismatches = np.sum(h[observed] != matched[observed])
            assert num_mismatches == case.data["num_mismatches"][j]

    def test_paths(self, case):
        num_compared = 0
        for j in range(case.num_queries):
            if not case.data["same_as_tsinfer"][j]:
                continue
            h, start, end = case.query(j)
            path, matched = find_path(case, j)
            expected = case.path(j)
            for a, b in zip(path, expected):
                np.testing.assert_array_equal(a, b)
            sites = np.arange(start, end)
            observed = sites[h[sites] != -1]
            mismatches = observed[h[observed] != matched[observed]]
            np.testing.assert_array_equal(mismatches, case.mismatch_sites(j))
            num_compared += 1
        assert num_compared > 0

    def test_matched_haplotype_is_path_haplotype(self, case):
        results = [find_path(case, j) for j in range(case.num_queries)]
        H = path_haplotypes(case, [path for path, _ in results])
        for j, (_, matched) in enumerate(results):
            _, start, end = case.query(j)
            np.testing.assert_array_equal(matched[start:end], H[start:end, j])

    def test_path_covers_window(self, case):
        positions = case.ts.sites_position
        for j in range(case.num_queries):
            _, start, end = case.query(j)
            (left, right, _), _ = find_path(case, j)
            assert left[0] <= positions[start]
            assert right[-1] > positions[end - 1]
            np.testing.assert_array_equal(left[1:], right[:-1])


class TestForwardBackward:
    def test_forward_log_likelihood(self, case):
        expected = case.data["forward_log_likelihood"]
        if np.all(np.isnan(expected)):
            pytest.skip("No forward likelihoods for this case")
        # The sc2ts queries are long, so check a few of them.
        for j in range(0, case.num_queries, 4 if case.ts.num_sites > 10000 else 1):
            h, _, _ = case.query(j)
            F = case.hmm().forward_matrix(h)
            assert F.log_likelihood == pytest.approx(expected[j], rel=1e-10)

    def test_posteriors(self, case):
        if "posterior" not in case.data:
            pytest.skip("No posteriors for this case")
        for j in range(case.num_queries):
            h, _, _ = case.query(j)
            hmm = case.hmm()
            F = hmm.forward_matrix(h)
            B = hmm.backward_matrix(h, F.normalisation_factor)
            P = F.decode() * B.decode()
            P /= np.sum(P, axis=1)[:, np.newaxis]
            if j == 0:
                np.testing.assert_allclose(P, case.data["posterior"], atol=1e-10)
            np.testing.assert_allclose(
                np.max(P, axis=1), case.data["posterior_max"][j], atol=1e-10
            )
            # The most likely node is only compared where it's clear.
            top2 = np.sort(P, axis=1)[:, -2:]
            clear = top2[:, 1] - top2[:, 0] > 1e-6
            np.testing.assert_array_equal(
                np.argmax(P, axis=1)[clear], case.data["posterior_max_node"][j][clear]
            )
