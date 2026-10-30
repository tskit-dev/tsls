"""
Make benchmark workloads from a tsinfer 1.0 inference.

The data directory holds the inputs and outputs of a tsinfer run (see
data/tgp_chr20p/README.md), plus the match_jobs.csv file written by
tsinfer_groups.py. There are two kinds of workload, matching the two
phases of tsinfer's matching:

- ancestors: the ancestors of one match group, matched against the tree
  sequence made from all earlier groups;
- samples: sample haplotypes, matched against the full ancestors tree
  sequence.

Each workload is a directory that run.py can use, containing:

- reference.trees: the tree sequence to match against;
- queries.npz, queries.csv: the query haplotypes, in the windowed format
  (see queries.WindowedHaplotypes), and their metadata, including the path
  that tsinfer found (truth_parents and truth_breakpoints);
- model.json: the HMM parameters used by tsinfer;
- tsinfer_group.json: the resources tsinfer recorded for matching the
  whole group (elapsed and CPU times are for all threads).

Haplotypes are encoded as in tsinfer: 0 for the ancestral allele, 1 for
the derived allele and -1 for missing data.
"""

import json
import pathlib
import tomllib

import click
import numpy as np
import pandas as pd
import tskit
import zarr

import queries

ANCESTORS_TS = "tgp_chr20p_inferred_ancestors.trees"
ANCESTORS_ZARR = "tgp_chr20p_inferred_ancestors.zarr"
SAMPLES_TS = "tgp_chr20p_inferred_samples_raw.trees"
SAMPLES_ZARR = "tgp_chr20p_samples_masked.zarr"
CONFIG = "full_inference.toml"
MATCH_JOBS = "match_jobs.csv"

# Nodes 0 and 1 of a tsinfer tree sequence are the ultimate and virtual
# roots, so the node of the jth matched haplotype is j + 2.
NUM_ROOTS = 2

QUERY_COLUMNS = [
    "query_id",
    "kind",
    "sample_id",
    "ploidy_index",
    "node",
    "group",
    "start",
    "end",
    "truth_num_segments",
    "truth_parents",
    "truth_breakpoints",
]


def cut_before(ts, node):
    """
    Return the tree sequence made of the nodes before the specified node ID,
    with their edges and mutations. Because tsinfer adds the nodes of each
    group in order, and a node can only copy from earlier groups, cutting
    before the first node of a group gives the tree sequence that the group
    was matched against.
    """
    tables = ts.dump_tables()
    tables.edges.keep_rows(tables.edges.child < node)
    if np.any(tables.edges.parent >= node):
        raise ValueError("Edge from a later node to an earlier one")
    tables.mutations.keep_rows(tables.mutations.node < node)
    tables.nodes.truncate(node)
    tables.build_index()
    return tables.tree_sequence()


def truth_paths(ts, nodes):
    """
    Return the list of (left, parent) tuples giving the path of each of the
    specified nodes, as arrays in left-to-right order.
    """
    edges = ts.tables.edges
    order = np.lexsort((edges.left, edges.child))
    child = edges.child[order]
    lo = np.searchsorted(child, nodes, side="left")
    hi = np.searchsorted(child, nodes, side="right")
    return [(edges.left[order[a:b]], edges.parent[order[a:b]]) for a, b in zip(lo, hi)]


def truth_columns(paths):
    return {
        "truth_num_segments": [len(parent) for _, parent in paths],
        "truth_parents": [queries.format_list(parent) for _, parent in paths],
        "truth_breakpoints": [queries.format_list(left[1:]) for left, _ in paths],
    }


def non_missing_window(h):
    """
    Return the (start, end) tuple of the first and one past the last
    non-missing site, as tsinfer matches over this range.
    """
    non_missing = np.where(h != queries.MISSING)[0]
    if len(non_missing) == 0:
        return 0, len(h)
    return int(non_missing[0]), int(non_missing[-1]) + 1


def load_model(data_dir, source):
    """
    Return the HMM parameters that tsinfer used for the specified source,
    with tsinfer's defaults where the config does not set them.
    """
    with open(data_dir / CONFIG, "rb") as f:
        raw = tomllib.load(f)
    params = raw["match"]["sources"][source]
    model = {
        "rho": params.get("recombination", 1e-2),
        "mu": params.get("mismatch", 1e-20),
        "weight_by_n": True,
        "alleles": None,
    }
    # A recombination rate (per bp, or the path to a genetic map) gives
    # per-site probabilities, which run.py computes from the sites.
    if "recombination_rate" in params:
        if "recombination" in params:
            raise ValueError(
                "Cannot specify both 'recombination' and 'recombination_rate'"
            )
        model["rho"] = None
        model["recombination_rate"] = params["recombination_rate"]
    return model


def group_resources(ts, group, group_size):
    """
    Return the resources that tsinfer recorded in the provenance for
    matching the specified group, with the number of haplotypes in it.
    """
    for provenance in ts.provenances():
        record = json.loads(provenance.record)
        parameters = record.get("parameters", {})
        if parameters.get("command") == "match" and parameters.get("group") == group:
            return {"group": group, "num_haplotypes": group_size, **record["resources"]}
    raise ValueError(f"No provenance for group {group}")


def save_workload(out_dir, reference, H, df, model, resources):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference.dump(out_dir / "reference.trees")
    H.save(out_dir, reference.sites_position, df[QUERY_COLUMNS])
    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)
    with open(out_dir / "tsinfer_group.json", "w") as f:
        json.dump(resources, f, indent=2)


def choose(rng, n, num_queries):
    """
    Return the sorted indexes of num_queries of n items chosen at random, or
    all of them if num_queries is None.
    """
    if num_queries is None or num_queries >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=num_queries, replace=False))


def load_jobs(data_dir):
    jobs = pd.read_csv(data_dir / MATCH_JOBS)
    # The nodes are added in the order of the jobs.
    jobs["node"] = np.arange(len(jobs)) + NUM_ROOTS
    return jobs


def make_ancestor_workload(data_dir, group, *, num_queries=None, seed=42):
    """
    Return the (reference, H, df, resources) tuple for the ancestors of the
    specified group.
    """
    jobs = load_jobs(data_dir)
    jobs = jobs[jobs.source == "ancestors"]
    group_jobs = jobs[jobs.group == group]
    if len(group_jobs) == 0:
        raise ValueError(f"No ancestors in group {group}")
    full = tskit.load(data_dir / ANCESTORS_TS)
    reference = cut_before(full, group_jobs.node.min())
    resources = group_resources(full, group, len(group_jobs))

    rng = np.random.default_rng(seed)
    group_jobs = group_jobs.iloc[choose(rng, len(group_jobs), num_queries)]
    store = zarr.open_group(data_dir / ANCESTORS_ZARR, mode="r")
    positions = store["variant_position"][:]
    assert np.array_equal(positions, full.sites_position)
    call_genotype = store["call_genotype"]

    starts = []
    windows = []
    for job in group_jobs.itertuples():
        # The ancestor zarr has one column per ancestor, in job order.
        assert job.sample_id == f"a{job.haplotype_index}"
        lo = np.searchsorted(positions, job.start_position, side="left")
        hi = np.searchsorted(positions, job.end_position, side="left")
        h = call_genotype[lo:hi, job.haplotype_index, 0].astype(np.int8)
        start, end = non_missing_window(h)
        starts.append(lo + start)
        windows.append(h[start:end])
    H = queries.WindowedHaplotypes.from_windows(len(positions), starts, windows)

    nodes = group_jobs.node.to_numpy()
    df = pd.DataFrame(
        {
            "query_id": np.arange(len(group_jobs)),
            "kind": "ancestor",
            "sample_id": group_jobs.sample_id.to_numpy(),
            "ploidy_index": group_jobs.ploidy_index.to_numpy(),
            "node": nodes,
            "group": group,
            "start": H.start,
            "end": H.end,
            **truth_columns(truth_paths(full, nodes)),
        }
    )
    return reference, H, df, resources


def inference_sites(samples_store, ancestor_positions):
    """
    Return the indexes of the inference sites in the samples store, and
    their ancestral allele indexes. The sites are those selected by the
    include expression in the config,

        (POS < 28100000) & ((INFO/match_eval_inference_site_mask == 0)
        & (INFO/match_eval_derived_ac != 1))

    that were used for ancestor generation, which drops some more sites.
    The samples store has duplicate positions, but none of them survive
    the selection.
    """
    position = samples_store["variant_position"][:]
    include = (
        (position < 28100000)
        & (samples_store["variant_match_eval_inference_site_mask"][:] == 0)
        & (samples_store["variant_match_eval_derived_ac"][:] != 1)
    )
    selected = np.where(include)[0]
    selected = selected[np.isin(position[selected], ancestor_positions)]
    if not np.array_equal(position[selected], ancestor_positions):
        raise ValueError("Inference sites do not match the ancestor sites")
    alleles = samples_store["variant_allele"].get_orthogonal_selection((selected,))
    ancestral = samples_store["variant_ancestral_allele"].get_orthogonal_selection(
        (selected,)
    )
    ancestral_index = np.full(len(selected), -1, dtype=np.int8)
    for j in range(alleles.shape[1]):
        ancestral_index[alleles[:, j] == ancestral] = j
    if np.any(ancestral_index < 0):
        raise ValueError("Ancestral allele not found")
    return selected, ancestral_index


def make_sample_workload(data_dir, *, num_queries=200, seed=42):
    """
    Return the (reference, H, df, resources) tuple for a random subset of
    the sample haplotypes.
    """
    jobs = load_jobs(data_dir)
    jobs = jobs[jobs.source == "samples"]
    raw = tskit.load(data_dir / SAMPLES_TS)
    (group,) = np.unique(jobs.group)
    resources = group_resources(raw, int(group), len(jobs))
    assert np.array_equal(raw.samples(), jobs.node.to_numpy())
    rng = np.random.default_rng(seed)
    jobs = jobs.iloc[choose(rng, len(jobs), num_queries)]
    for job in jobs.itertuples():
        metadata = raw.node(job.node).metadata
        assert metadata["sample_id"] == job.sample_id
        assert metadata["ploidy_index"] == job.ploidy_index

    reference = tskit.load(data_dir / ANCESTORS_TS)
    store = zarr.open_group(data_dir / SAMPLES_ZARR, mode="r")
    sites, ancestral_index = inference_sites(store, reference.sites_position)
    sample_ids = list(store["sample_id"][:])
    individuals = np.array([sample_ids.index(sid) for sid in jobs.sample_id])
    unique_individuals, inverse = np.unique(individuals, return_inverse=True)
    G = store["call_genotype"].get_orthogonal_selection(
        (sites, unique_individuals, slice(None))
    )
    mask = store["call_genotype_mask"].get_orthogonal_selection(
        (sites, unique_individuals, slice(None))
    )
    G = G[:, inverse, jobs.ploidy_index.to_numpy()]
    mask = mask[:, inverse, jobs.ploidy_index.to_numpy()]
    H = (G != ancestral_index[:, np.newaxis]).astype(np.int8)
    H[(G < 0) | mask] = queries.MISSING
    H = H.T

    # The matched tree sequence has no mutations above the samples, so the
    # sample genotypes are the input haplotypes at non-missing sites.
    nodes = jobs.node.to_numpy()
    expected = raw.genotype_matrix(samples=nodes).T
    non_missing = H != queries.MISSING
    if not np.array_equal(H[non_missing], expected[non_missing]):
        raise ValueError("Sample haplotypes do not match the matched tree sequence")

    starts = []
    windows = []
    for h in H:
        start, end = non_missing_window(h)
        starts.append(start)
        windows.append(h[start:end])
    H = queries.WindowedHaplotypes.from_windows(reference.num_sites, starts, windows)
    df = pd.DataFrame(
        {
            "query_id": np.arange(len(jobs)),
            "kind": "sample",
            "sample_id": jobs.sample_id.to_numpy(),
            "ploidy_index": jobs.ploidy_index.to_numpy(),
            "node": nodes,
            "group": jobs.group.to_numpy(),
            "start": H.start,
            "end": H.end,
            **truth_columns(truth_paths(raw, nodes)),
        }
    )
    return reference, H, df, resources


@click.group()
def cli():
    pass


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("group", type=int)
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--num-queries", type=int, default=None, help="[default: all]")
@click.option("--seed", type=int, default=42, show_default=True)
def ancestors(data_dir, group, out_dir, num_queries, seed):
    """
    Make the workload for the ancestors in GROUP.
    """
    data_dir = pathlib.Path(data_dir)
    reference, H, df, resources = make_ancestor_workload(
        data_dir, group, num_queries=num_queries, seed=seed
    )
    model = load_model(data_dir, "ancestors")
    save_workload(out_dir, reference, H, df, model, resources)
    click.echo(
        f"Wrote {len(df)} ancestors (mean {np.mean(df.end - df.start):.0f} sites) "
        f"and a reference with {reference.num_nodes} nodes to {out_dir}"
    )


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--num-queries", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def samples(data_dir, out_dir, num_queries, seed):
    """
    Make the workload for a random subset of the sample haplotypes.
    """
    data_dir = pathlib.Path(data_dir)
    reference, H, df, resources = make_sample_workload(
        data_dir, num_queries=num_queries, seed=seed
    )
    model = load_model(data_dir, "samples")
    save_workload(out_dir, reference, H, df, model, resources)
    click.echo(f"Wrote {len(df)} sample haplotypes to {out_dir}")


if __name__ == "__main__":
    cli()
