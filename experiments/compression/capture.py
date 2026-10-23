"""
Capture the exact per-site Viterbi and forward values (as tsls
CompressedMatrix files) for queries on sc2ts sub-ARGs, simulated ARGs and
tsinfer workloads, so that compression can be analysed and prototyped
offline.

Writes results/<name>.trees (the tree sequence the matrices refer to),
results/<name>_<query>_<algorithm>.npz and results/manifest.csv. Capturing
some of the cases replaces their rows in an existing manifest and keeps the
others.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import tskit

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "thresholding"))
sys.path.insert(0, str(ROOT / "benchmarking"))
import queries
import thresholding as th
import tsls

v = th.v
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SC2TS_DATA = ROOT / "benchmarking" / "data"
TSINFER_DATA = ROOT / "benchmarking" / "data" / "tgp_chr20p"
TSINFER_WORKLOADS = ["ancestors_g1483", "ancestors_g1484", "ancestors_g1485", "samples"]
CASES = ["msprime", "sc2ts", "tsinfer"]


def sc2ts_queries():
    """
    Return the list of (query_id, label) for the queries we capture: two each
    of copies with k = 0, 2 and 10 mutations, and one- and two-breakpoint
    recombinants with k = 0 and 5.
    """
    df = pd.read_csv(SC2TS_DATA / "queries_scaling" / "queries.csv")
    num_breakpoints = df.breakpoints.fillna("").astype(str).str.split().str.len()
    chosen = []
    for kind, k, nb in [
        ("copy", 0, 0),
        ("copy", 2, 0),
        ("copy", 10, 0),
        ("recombinant", 0, 1),
        ("recombinant", 5, 1),
        ("recombinant", 0, 2),
    ]:
        ids = df[(df.kind == kind) & (df.k == k) & (num_breakpoints == nb)].query_id
        for query_id in ids[:2]:
            label = f"{kind}{nb if kind == 'recombinant' else ''}_k{k}"
            chosen.append((int(query_id), label))
    return chosen


def capture(name, ts, query_list, mu, rho, weight_by_n, rows):
    ts_path = RESULTS / f"{name}.trees"
    ts.dump(ts_path)
    mi = tsls.TreeSequence(ts)
    for query_id, label, h in query_list:
        am = v.make_matcher(
            ts, mu, rho, weight_by_n=weight_by_n, count_distinct=True, tree_sequence=mi
        )
        (_, _, parent), matched, vm = am.viterbi_matrix(h)
        fm = am.forward_matrix(h)
        for algorithm, cm in [("viterbi", vm), ("forward", fm)]:
            path = RESULTS / f"{name}_{query_id}_{algorithm}.npz"
            cm.dump(path)
            rows.append(
                {
                    "arg": name,
                    "trees": ts_path.name,
                    "query": query_id,
                    "label": label,
                    "algorithm": algorithm,
                    "matrix": path.name,
                    "num_nodes": ts.num_nodes,
                    "num_sites": ts.num_sites,
                    "num_samples": ts.num_samples,
                    "mu": mu,
                    "rho": rho,
                    "weight_by_n": weight_by_n,
                    "num_alleles": len(v.ALLELES),
                    "num_segments": len(parent),
                    "num_mismatches": int(np.sum((matched != h) & (h != -1))),
                    "mean_traceback_size": am.mean_traceback_size,
                    "mean_num_distinct_values": am.mean_num_distinct_values,
                }
            )
        np.save(RESULTS / f"{name}_{query_id}_query.npy", h)
        print(f"Captured {name} query {query_id} ({label})", flush=True)


def drop_root_edges(ts):
    """
    Return the tree sequence with the edges of node 0 removed, so that the
    ultimate root of a tsinfer or sc2ts ARG is isolated and not a state.
    """
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.parent != 0)
    return tables.tree_sequence()


def window_ts(ts, start, end):
    """
    Return the tree sequence restricted to the sites from start to end
    (exclusive). Nodes are kept, so node IDs and the number of nodes (n with
    weight_by_n) are unchanged, and running over all its sites is the same
    as running over [start, end) of the original.
    """
    position = ts.sites_position
    right = position[end] if end < ts.num_sites else ts.sequence_length
    tables = ts.dump_tables()
    keep = np.zeros(ts.num_sites, dtype=bool)
    keep[start:end] = True
    tables.delete_sites(np.where(~keep)[0])
    tables.keep_intervals([[position[start], right]], simplify=False)
    return tables.tree_sequence()


def choose_tsinfer_queries(df, kind, num_queries, seed):
    """
    Return the query IDs to capture from a tsinfer workload: ancestors at
    spread quantiles of window length, and random sample haplotypes.
    """
    if kind == "ancestor":
        quantiles = [0.1, 0.4, 0.7, 0.95][:num_queries]
        order = np.argsort((df.end - df.start).to_numpy(), kind="stable")
        index = np.round(np.array(quantiles) * (len(df) - 1)).astype(int)
        return [int(df.query_id.iloc[order[j]]) for j in index]
    rng = np.random.default_rng(seed)
    return sorted(int(q) for q in rng.choice(df.query_id, num_queries, replace=False))


def format_path(left, parent):
    return queries.format_list(parent), queries.format_list(left[1:])


def check_window(mi, ts, h, start, end, vm, rho, mu, path):
    """
    Check that Viterbi on the windowed tree sequence gives the same path and
    stored values as running over [start, end) of the full tree sequence,
    which validates the windowing used for forward.
    """
    wts = window_ts(ts, start, end)
    am = tsls.Hmm(
        wts,
        np.full(wts.num_sites, rho),
        np.full(wts.num_sites, mu),
    )
    (left, _, parent), _, wvm = am.viterbi_matrix(h[start:end])
    assert np.array_equal(parent, path[2])
    assert np.array_equal(left[:-1], path[0][:-1])
    s = slice(vm.offsets[start], vm.offsets[end])
    assert np.array_equal(wvm.nodes, vm.nodes[s])
    assert np.array_equal(wvm.values, vm.values[s])
    print("  windowed Viterbi matches the full tree sequence", flush=True)


def capture_tsinfer(workload, *, num_queries, forward_sites, seed, check, rows):
    """
    Capture Viterbi over each chosen query's window, and forward over the
    same window (ancestors) or a fixed window in the middle (samples), with
    the parameters tsinfer used.
    """
    query_dir = TSINFER_DATA / workload
    with open(query_dir / "model.json") as f:
        model = json.load(f)
    assert model["weight_by_n"] and model["alleles"] is None
    mu, rho = model["mu"], model["rho"]
    H, _, df = queries.load_queries(query_dir)
    kind = df.kind.iloc[0]
    name = f"tsinfer_{workload}"
    ts = drop_root_edges(tskit.load(query_dir / "reference.trees"))
    ts_path = RESULTS / f"{name}.trees"
    ts.dump(ts_path)
    mi = tsls.TreeSequence(ts)
    for query_id in choose_tsinfer_queries(df, kind, num_queries, seed):
        query = df.iloc[query_id]
        assert query.query_id == query_id
        h = H[query_id]
        start, end = queries.query_window(H, query_id)
        am = tsls.Hmm(mi, np.full(ts.num_sites, rho), np.full(ts.num_sites, mu))
        path, matched, vm = am.viterbi_matrix(h, start, end)
        left, _, parent = path
        truth = (str(query.truth_parents), str(query.truth_breakpoints))
        truth = tuple("" if x == "nan" else x for x in truth)
        if format_path(left[::-1], parent[::-1]) != truth:
            raise ValueError(f"{name} query {query_id}: path differs from tsinfer")
        if check:
            check_window(mi, ts, h, start, end, vm, rho, mu, path)
        common = {
            "arg": name,
            "case": "tsinfer",
            "group": int(query.group),
            "query": query_id,
            "label": f"{kind}_{query.sample_id}",
            "mu": mu,
            "rho": rho,
            "weight_by_n": True,
            "num_alleles": 2,
            "num_segments": len(parent),
            "num_mismatches": int(np.sum((matched != h) & (h != -1))),
        }
        matrix = RESULTS / f"{name}_{query_id}_viterbi.npz"
        vm.dump(matrix)
        rows.append(
            {
                **common,
                "trees": ts_path.name,
                "algorithm": "viterbi",
                "matrix": matrix.name,
                "num_nodes": ts.num_nodes,
                "num_sites": ts.num_sites,
                "num_samples": ts.num_samples,
                "start": start,
                "end": end,
                "mean_traceback_size": am.mean_traceback_size,
            }
        )

        if kind == "sample":
            middle = (start + end) // 2
            fstart = max(start, middle - forward_sites // 2)
            fend = min(end, fstart + forward_sites)
        else:
            fstart, fend = start, end
        wts = window_ts(ts, fstart, fend)
        wts_path = RESULTS / f"{name}_{query_id}_window.trees"
        wts.dump(wts_path)
        fam = tsls.Hmm(
            wts,
            np.full(wts.num_sites, rho),
            np.full(wts.num_sites, mu),
        )
        fm = fam.forward_matrix(h[fstart:fend])
        matrix = RESULTS / f"{name}_{query_id}_forward.npz"
        fm.dump(matrix)
        rows.append(
            {
                **common,
                "trees": wts_path.name,
                "algorithm": "forward",
                "matrix": matrix.name,
                "num_nodes": wts.num_nodes,
                "num_sites": wts.num_sites,
                "num_samples": wts.num_samples,
                "start": fstart,
                "end": fend,
            }
        )
        np.save(RESULTS / f"{name}_{query_id}_query.npy", h)
        print(
            f"Captured {name} query {query_id} ({kind}, {end - start} sites, "
            f"forward over {fend - fstart})",
            flush=True,
        )


def case_of(arg):
    return arg.split("_")[0]


def write_manifest(rows, cases):
    """
    Write the manifest, replacing the rows of the specified cases in any
    existing manifest.
    """
    path = RESULTS / "manifest.csv"
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        old["case"] = old.arg.map(case_of)
        df = pd.concat([old[~old.case.isin(cases)], df], ignore_index=True)
    df["case"] = df.arg.map(case_of)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help=f"Cases to capture (comma separated, from {', '.join(CASES)})",
    )
    parser.add_argument(
        "--sizes", default="10000,30000", help="sc2ts sub-ARG sizes (comma separated)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One query of each kind, and check the forward windowing",
    )
    parser.add_argument(
        "--tsinfer-queries",
        type=int,
        default=4,
        help="Queries per tsinfer workload",
    )
    parser.add_argument(
        "--sample-forward-sites",
        type=int,
        default=20000,
        help="Sites in the forward window for tsinfer samples",
    )
    args = parser.parse_args()
    cases = args.cases.split(",")
    for case in cases:
        if case not in CASES:
            parser.error(f"Unknown case {case!r}")
    RESULTS.mkdir(exist_ok=True)
    rows = []
    mu, rho = v.hmm_params(3)

    if "msprime" in cases:
        # Simulated ARGs: held-out samples, as in the verification suite
        for num_samples in [50, 1000]:
            for seed in [1] if args.quick else [1, 2]:
                ts, h, _, _ = th.threading_case(num_samples, seed)
                capture(
                    f"msprime_n{num_samples}_s{seed}",
                    ts,
                    [(0, "held_out", h)],
                    mu,
                    rho,
                    False,
                    rows,
                )

    if "sc2ts" in cases:
        if (SC2TS_DATA / "subargs").exists():
            H = np.load(SC2TS_DATA / "queries_scaling" / "queries.npz")["H"]
            query_list = sc2ts_queries()
            if args.quick:
                seen = set()
                query_list = [
                    q for q in query_list if not (q[1] in seen or seen.add(q[1]))
                ]
            for size in [int(x) for x in args.sizes.split(",")]:
                ts = tskit.load(SC2TS_DATA / "subargs" / f"sc2ts_bench_n{size}.trees")
                capture(
                    f"sc2ts_n{size}",
                    drop_root_edges(ts),
                    [(query_id, label, H[query_id]) for query_id, label in query_list],
                    mu,
                    rho,
                    False,
                    rows,
                )
        else:
            print(f"No sc2ts data in {SC2TS_DATA}: skipping")

    if "tsinfer" in cases:
        missing = [w for w in TSINFER_WORKLOADS if not (TSINFER_DATA / w).exists()]
        if len(missing) > 0:
            print(
                f"Missing tsinfer workloads {missing} in {TSINFER_DATA} (see "
                "benchmarking/README.md): skipping"
            )
        else:
            for workload in TSINFER_WORKLOADS:
                capture_tsinfer(
                    workload,
                    num_queries=1 if args.quick else args.tsinfer_queries,
                    forward_sites=args.sample_forward_sites,
                    seed=1,
                    check=args.quick,
                    rows=rows,
                )
    write_manifest(rows, cases)


if __name__ == "__main__":
    main()
