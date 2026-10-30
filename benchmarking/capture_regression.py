"""
Capture regression cases for tsls from the benchmark workloads, before the
tsinfer-hmm and tskit-hmm engines are removed from this repository.

Each case is a small tree sequence derived from one of the benchmark inputs
(the sc2ts ARG, a tsinfer 1.0 inference of 1000 Genomes chr20p, and an
msprime simulation with multiallelic sites), a set of query haplotypes and
the model parameters. For each query we record the tsls Viterbi path, the
sites where its matched haplotype mismatches, and its log likelihood, and (for
queries over all sites) the forward log likelihood. We also run tsinfer-hmm
on the same inputs and check that its path has the same likelihood,
recording whether the paths are identical (they differ only among equally
likely paths). tskit-hmm only copies from samples, so its paths aren't
comparable and it isn't used.

The cases are written to tsls/tests/data/regression, where
tests/test_regression.py checks that tsls reproduces them.

    uv run python capture_regression.py
"""

import gzip
import json
import pathlib
import shutil

import msprime
import numpy as np
import tsinfer_hmm
import tskit
import tsls

import engines
import queries
import subargs

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT.parent / "tsls" / "tests" / "data" / "regression"

ACGT = list(engines.ALLELES)


def strip(tables):
    """
    Remove the metadata, provenances, individuals and populations, which
    the HMM doesn't use, to keep the files small.
    """
    for table in [tables.nodes, tables.sites, tables.mutations, tables.edges]:
        table.packset_metadata([b""] * table.num_rows)
        table.metadata_schema = tskit.MetadataSchema(None)
    tables.nodes.individual = np.full(tables.nodes.num_rows, -1, dtype=np.int32)
    tables.nodes.population = np.full(tables.nodes.num_rows, -1, dtype=np.int32)
    tables.individuals.clear()
    tables.populations.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""


def drop_root_edges(ts):
    """
    Return ts without the edges of node 0, tsinfer's ultimate root, so that
    tsls never copies it.
    """
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    tables.build_index()
    return tables.tree_sequence()


def site_window(ts, start, stop):
    """
    Return the tree sequence made of the sites start to stop (exclusive) of
    ts, over the interval from the first of these sites to the next site,
    and with nodes that have no edges there removed (keeping node 0 and the
    order of the other nodes).
    """
    positions = ts.sites_position
    left = positions[start]
    right = positions[stop] if stop < ts.num_sites else ts.sequence_length
    tables = ts.dump_tables()
    strip(tables)
    tables.keep_intervals([[left, right]], simplify=False)
    keep = np.zeros(tables.nodes.num_rows, dtype=bool)
    keep[0] = True
    keep[tables.edges.parent] = True
    keep[tables.edges.child] = True
    node_map = tables.nodes.keep_rows(keep)
    tables.edges.parent = node_map[tables.edges.parent]
    tables.edges.child = node_map[tables.edges.child]
    tables.mutations.keep_rows(keep[tables.mutations.node])
    tables.mutations.node = node_map[tables.mutations.node]
    tables.sort()
    tables.build_index()
    return tables.tree_sequence()


def path_log_likelihood(ts, h, start, end, path, matched, model):
    """
    Return the log likelihood of the specified path (left, right, parent in
    left-to-right order) under the tsls model.
    """
    left = path[0]
    return engines.model_path_log_likelihood(
        h,
        matched,
        left,
        ts.sites_position,
        start,
        end,
        rho=model["recombination"],
        mu=model["mismatch"],
        n=ts.num_nodes,
        num_alleles=model["num_alleles"],
    )


def run_tsls(ts, h, start, end, model):
    hmm = tsls.Hmm(model["reference"], model["recombination"], model["mismatch"])
    matched = np.zeros(ts.num_sites, dtype=np.int8)
    left, right, parent = hmm.find_path(h, start, end, matched)
    path = left[::-1].copy(), right[::-1].copy(), parent[::-1].copy()
    return path, matched


def run_tsinfer(ts_tsinfer, h, start, end, model):
    """
    Run tsinfer-hmm on the tree sequence with node 0's edges, with tsinfer's
    weight_by_n model and recombination probabilities, which is the same
    model as tsls's with model["recombination"].
    """
    matcher = tsinfer_hmm.AncestorMatcher(
        model["tsinfer_reference"],
        model["tsinfer_recombination"],
        model["mismatch"],
        weight_by_n=model["weight_by_n"],
    )
    matched = np.zeros(ts_tsinfer.num_sites, dtype=np.int8)
    left, right, parent = matcher.find_path(h, start, end, matched)
    path = (
        left[::-1].astype(np.float64),
        right[::-1].astype(np.float64),
        parent[::-1].copy(),
    )
    return path, matched


def capture(
    name,
    description,
    ts_tsinfer,
    H,
    starts,
    ends,
    *,
    rho,
    mu,
    alleles,
    weight_by_n,
    forward=False,
    posterior=False,
):
    """
    Run both engines on the specified queries and write the case. ts_tsinfer
    has tsinfer's ultimate root, node 0, and tsls is run on it without node
    0's edges. rho is the recombination probability as tsinfer takes it,
    with the specified weight_by_n.
    """
    ts = drop_root_edges(ts_tsinfer)
    num_sites = ts.num_sites
    rho = np.array(np.broadcast_to(np.asarray(rho, dtype=float), (num_sites,)))
    mismatch = np.full(num_sites, mu)
    recombination = rho
    if not weight_by_n:
        recombination = tsls.model.recombination_probability(rho, ts.num_nodes)
    model = {
        "reference": tsls.TreeSequence(ts, alleles=alleles),
        "tsinfer_reference": tsinfer_hmm.MatcherIndexes(
            ts_tsinfer, vestigial_root=False, alleles=alleles
        ),
        "recombination": recombination,
        "tsinfer_recombination": rho,
        "mismatch": mismatch,
        "weight_by_n": weight_by_n,
        "num_alleles": engines.site_num_alleles(ts, alleles),
    }
    H = np.asarray(H, dtype=np.int8)
    rows = []
    path_offset = [0]
    lefts, rights, parents = [], [], []
    mismatch_offset = [0]
    mismatch_sites = []
    for j, (h, start, end) in enumerate(zip(H, starts, ends)):
        path, matched = run_tsls(ts, h, start, end, model)
        ll = path_log_likelihood(ts, h, start, end, path, matched, model)
        t_path, t_matched = run_tsinfer(ts_tsinfer, h, start, end, model)
        t_ll = path_log_likelihood(ts, h, start, end, t_path, t_matched, model)
        if not np.isclose(ll, t_ll, rtol=1e-9, atol=0):
            raise ValueError(f"{name} query {j}: tsls {ll} != tsinfer {t_ll}")
        same = all(np.array_equal(a, b) for a, b in zip(path, t_path))
        sites = np.arange(start, end)
        observed = sites[h[sites] != queries.MISSING]
        sites_mismatched = observed[h[observed] != matched[observed]]
        num_mismatches = len(sites_mismatched)
        forward_ll = np.nan
        if forward:
            hmm = tsls.Hmm(model["reference"], recombination, mismatch)
            forward_ll = hmm.forward_matrix(h).log_likelihood
        rows.append((ll, num_mismatches, same, forward_ll))
        mismatch_sites.append(sites_mismatched)
        mismatch_offset.append(mismatch_offset[-1] + num_mismatches)
        lefts.append(path[0])
        rights.append(path[1])
        parents.append(path[2])
        path_offset.append(path_offset[-1] + len(path[2]))

    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compressed, since tree sequence files of many sites are mostly offsets
    trees_path = out_dir / "reference.trees"
    ts.dump(trees_path)
    gz_path = out_dir / "reference.trees.gz"
    with (
        open(trees_path, "rb") as f_in,
        gzip.open(gz_path, "wb", compresslevel=9) as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)
    trees_path.unlink()
    arrays = {
        "haplotypes": H,
        "start": np.asarray(starts, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "recombination": recombination,
        "mismatch": mismatch,
        "path_offset": np.array(path_offset, dtype=np.int64),
        "path_left": np.concatenate(lefts),
        "path_right": np.concatenate(rights),
        "path_parent": np.concatenate(parents).astype(np.int32),
        "mismatch_offset": np.array(mismatch_offset, dtype=np.int64),
        "mismatch_site": np.concatenate(mismatch_sites).astype(np.int64),
        "log_likelihood": np.array([r[0] for r in rows]),
        "num_mismatches": np.array([r[1] for r in rows], dtype=np.int64),
        "same_as_tsinfer": np.array([r[2] for r in rows]),
        "forward_log_likelihood": np.array([r[3] for r in rows]),
    }
    if posterior:
        # The full matrix for the first query, and the most likely node
        # and its probability at each site for all of them
        P = [posterior_matrix(model["reference"], recombination, mismatch, h) for h in H]
        arrays["posterior"] = P[0]
        arrays["posterior_max_node"] = np.array([np.argmax(p, axis=1) for p in P])
        arrays["posterior_max"] = np.array([np.max(p, axis=1) for p in P])
    np.savez_compressed(out_dir / "case.npz", **arrays)
    with open(out_dir / "case.json", "w") as f:
        json.dump({"description": description, "alleles": alleles}, f, indent=2)
    num_same = int(np.sum(arrays["same_as_tsinfer"]))
    print(
        f"{name}: {len(H)} queries, {ts.num_nodes} nodes, {ts.num_trees} trees, "
        f"{num_sites} sites; paths identical to tsinfer-hmm for {num_same}; "
        f"mean segments {np.mean(np.diff(path_offset)):.2f}, "
        f"mean mismatches {np.mean(arrays['num_mismatches']):.2f}"
    )


def posterior_matrix(reference, recombination, mismatch, h):
    """
    Return the (num_sites, num_nodes) posterior probabilities of copying
    each node, from the forward and backward matrices.
    """
    hmm = tsls.Hmm(reference, recombination, mismatch)
    F = hmm.forward_matrix(h)
    B = hmm.backward_matrix(h, F.normalisation_factor)
    P = F.decode() * B.decode()
    return P / np.sum(P, axis=1)[:, np.newaxis]


def add_missing(H, rng, fraction=0.05):
    """
    Make a block of about the specified fraction of sites missing in every
    other query, as in real sequences with runs of Ns.
    """
    H = H.copy()
    num_sites = H.shape[1]
    length = int(fraction * num_sites)
    for j in range(0, len(H), 2):
        start = rng.integers(0, num_sites - length)
        H[j, start : start + length] = queries.MISSING
    return H


def sc2ts_case():
    """
    The ARG of the 2000 oldest samples of the sc2ts benchmark ARG, with
    copy and recombinant queries made from its samples, and sc2ts's model
    with k = 3 (weight_by_n=False).
    """
    full = tskit.load(DATA / "sc2ts_bench.trees")
    ts, _ = subargs.sub_arg(full, subargs.time_cutoff(full, 2000))
    tables = ts.dump_tables()
    strip(tables)
    ts = tables.tree_sequence()
    rng = np.random.default_rng(1234)
    query_set = queries.QuerySet(ts.sites_position)
    queries.add_copy_queries(query_set, ts, rng, num_nodes=8, divergence=[0, 2, 5])
    for num_breakpoints in [1, 2]:
        queries.add_recombinant_queries(
            query_set,
            ts,
            rng,
            num_recombinants=4,
            num_breakpoints=num_breakpoints,
            divergence=[0, 2],
            min_segment_diffs=4,
        )
    H = add_missing(np.array(query_set.haplotypes), rng)
    mu, rho = engines.solve_num_mismatches(3)
    num_sites = ts.num_sites
    capture(
        "sc2ts",
        "The 2000 oldest samples of the sc2ts ARG (ACGT sites only), with "
        "copy and recombinant queries of its samples, and sc2ts's model for "
        "k = 3 mismatches per recombination.",
        ts,
        H,
        [0] * len(H),
        [num_sites] * len(H),
        rho=rho,
        mu=mu,
        alleles=ACGT,
        weight_by_n=False,
        forward=True,
    )


def tgp_case(group, start, stop, mu, name):
    """
    The ancestors of a tsinfer chr20p match group that overlap the
    specified sites, matched against the tree sequence of the earlier
    groups over those sites.
    """
    workload = DATA / "tgp_chr20p" / f"ancestors_g{group}"
    full = tskit.load(workload / "reference.trees")
    ts = site_window(full, start, stop)
    H_full, _, _ = queries.load_queries(workload)
    with open(workload / "model.json") as f:
        model = json.load(f)
    H, starts, ends = [], [], []
    for j in range(len(H_full)):
        lo = max(int(H_full.start[j]), start)
        hi = min(int(H_full.end[j]), stop)
        if hi - lo < 50:
            continue
        h = np.full(ts.num_sites, queries.MISSING, dtype=np.int8)
        h[lo - start : hi - start] = H_full[j][lo:hi]
        observed = np.where(h != queries.MISSING)[0]
        first = int(observed[0])
        last = int(observed[-1]) + 1
        # The whole window, as tsinfer matches it, and its middle half
        quarter = (last - first) // 4
        for a, b in [(first, last), (first + quarter, last - quarter)]:
            H.append(h)
            starts.append(a)
            ends.append(b)
    capture(
        name,
        f"Ancestors of tsinfer group {group} of 1000 Genomes chr20p over "
        f"sites {start} to {stop}, matched against the earlier groups with "
        f"tsinfer's model (recombination {model['rho']}, mismatch {mu}).",
        ts,
        H,
        starts,
        ends,
        rho=model["rho"],
        mu=mu,
        alleles=None,
        weight_by_n=True,
    )


def msprime_case():
    """
    An msprime simulation with multiallelic sites (JC69, high mutation rate),
    with held-out haplotypes as queries. Every node is a state, and node 0
    is a sample, so no edges are dropped.
    """
    ts = msprime.sim_ancestry(
        24,
        ploidy=1,
        sequence_length=2e4,
        recombination_rate=1e-7,
        population_size=1e4,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=2e-7, random_seed=43)
    held_out = np.arange(18, 24)
    ref = ts.simplify(np.arange(18), filter_sites=False, keep_unary=False)
    G = ts.genotype_matrix(samples=held_out, alleles=tuple(ACGT)).T.astype(np.int8)
    tables = ref.dump_tables()
    strip(tables)
    # Put node 0 first so that tsinfer-hmm sees it as its ultimate root:
    # add a new root above everything, whose edges drop_root_edges removes.
    tables.nodes.clear()
    tables.nodes.add_row(time=ref.max_root_time + 1)
    for node in ref.tables.nodes:
        tables.nodes.append(node.replace(metadata=b"", population=-1, individual=-1))
    tables.edges.parent += 1
    tables.edges.child += 1
    tables.mutations.node += 1
    for tree in ref.trees():
        for root in tree.roots:
            tables.edges.add_row(tree.interval.left, tree.interval.right, 0, root + 1)
    tables.sort()
    tables.edges.squash()
    tables.sort()
    ts_tsinfer = tables.tree_sequence()
    rng = np.random.default_rng(5)
    H = [G[j] for j in range(len(G))]
    for j in range(len(G)):
        h, _ = queries.mutate(G[j], 3, rng)
        H.append(h)
    H = add_missing(np.array(H), rng, fraction=0.1)
    num_sites = ts_tsinfer.num_sites
    capture(
        "msprime",
        "An msprime simulation of 18 haploid samples with multiallelic JC69 "
        "sites, with 6 held-out samples (and the same with 3 mutations) as "
        "queries.",
        ts_tsinfer,
        H,
        [0] * len(H),
        [num_sites] * len(H),
        rho=1e-2,
        mu=1e-2,
        alleles=ACGT,
        weight_by_n=True,
        forward=True,
        posterior=True,
    )


def main():
    sc2ts_case()
    tgp_case(84, 398000, 403000, 1e-20, "tgp_g84")
    tgp_case(1000, 14000, 19000, 1e-3, "tgp_g1000")
    msprime_case()


if __name__ == "__main__":
    main()
