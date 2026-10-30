"""
Convert an sc2ts ARG into the form used for benchmarking.

The queries encode every allele as one of A, C, G and T, so we delete any
site with another state (in practice, deletions coded as "-"). We also add
the 0 -> 1 edge between the vestigial root and the reference, if it's
missing, as in tsinfer's trees. (This was needed by the tskit and tsinfer
engines that tsls was compared with; the benchmark ARG is kept the same.)
"""

import collections

import click
import numpy as np
import tszip

ALLELES = ("A", "C", "G", "T")


def non_acgt_sites(ts):
    """
    Return the IDs of sites that have a non-ACGT ancestral or derived state.
    """
    bad_ancestral = ~np.isin(np.asarray(ts.sites_ancestral_state), ALLELES)
    bad_derived = ~np.isin(np.asarray(ts.mutations_derived_state), ALLELES)
    bad_sites = np.union1d(np.where(bad_ancestral)[0], ts.mutations_site[bad_derived])
    return bad_sites.astype(np.int32)


def add_vestigial_root_edge(ts):
    """
    Add an edge from node 0 to the root node 1 across the whole sequence.
    """
    if ts.nodes_flags[0] & 1:
        raise ValueError("Node 0 must not be a sample")
    if np.any(ts.edges_parent == 0) or np.any(ts.edges_child == 0):
        raise ValueError("Node 0 must not be in any edges")
    if ts.nodes_time[0] <= ts.nodes_time[1]:
        raise ValueError("Node 0 must be older than node 1")
    for tree in ts.trees():
        if tree.num_roots != 1 or tree.root != 1:
            raise ValueError(f"Node 1 must be the only root in tree {tree.index}")
    tables = ts.dump_tables()
    tables.edges.add_row(left=0, right=ts.sequence_length, parent=0, child=1)
    tables.sort()
    tables.build_index()
    return tables.tree_sequence()


def prepare(ts):
    """
    Return the (benchmark_ts, deleted_sites) tuple for the specified sc2ts ARG.
    """
    deleted_sites = non_acgt_sites(ts)
    ts = ts.delete_sites(deleted_sites)
    ts = add_vestigial_root_edge(ts)
    return ts, deleted_sites


@click.command()
@click.argument("in_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path(dir_okay=False))
def main(in_path, out_path):
    """
    Convert the sc2ts ARG at IN_PATH (.trees or .tsz) to the benchmark ARG
    at OUT_PATH.
    """
    ts = tszip.load(in_path)
    click.echo(
        f"Loaded {in_path}: {ts.num_samples} samples, {ts.num_nodes} nodes, "
        f"{ts.num_trees} trees, {ts.num_sites} sites, {ts.num_mutations} mutations"
    )
    ts, deleted_sites = prepare(ts)
    click.echo(f"Deleted {len(deleted_sites)} sites with non-ACGT states")
    counts = np.bincount(ts.mutations_site, minlength=ts.num_sites)
    histogram = collections.Counter(np.minimum(counts, 10).tolist())
    click.echo("Mutations per site (10 means >= 10):")
    for k in sorted(histogram):
        click.echo(f"  {k}\t{histogram[k]}")
    ts.dump(out_path)
    click.echo(
        f"Wrote {out_path}: {ts.num_sites} sites, {ts.num_mutations} mutations, "
        f"{ts.num_edges} edges"
    )


if __name__ == "__main__":
    main()
