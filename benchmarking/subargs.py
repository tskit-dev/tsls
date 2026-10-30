"""
Make a series of nested sub-ARGs of a benchmark ARG, for measuring how
tsls scales with ARG size.

sc2ts adds samples day by day, so the natural sub-ARG is the ARG "as of"
an earlier date: all nodes at least as old as a time cutoff, and the
edges and mutations between them. Every parent is older than its child,
so this is always a valid ARG with the same root structure. All sites are
kept, so a set of queries can be run against any of the sub-ARGs.

Node IDs change when nodes are removed, so the original node ID of each
node is written alongside each sub-ARG (see load_node_ids). Results are
reported in original node IDs so that they can be compared across
sub-ARGs.
"""

import pathlib

import click
import numpy as np
import tskit


def node_ids_path(ts_path):
    ts_path = pathlib.Path(ts_path)
    return ts_path.with_name(ts_path.name + ".node_ids.npy")


def load_node_ids(ts_path, ts):
    """
    Return the original node ID of each node in the tree sequence loaded from
    the specified path. This is the identity unless the tree sequence is a
    sub-ARG.
    """
    path = node_ids_path(ts_path)
    if path.exists():
        node_ids = np.load(path)
        assert len(node_ids) == ts.num_nodes
        return node_ids
    return np.arange(ts.num_nodes, dtype=np.int32)


def time_cutoff(ts, num_samples):
    """
    Return the time such that keeping nodes at least this old keeps (at least)
    the specified number of the oldest samples.
    """
    sample_times = np.sort(ts.nodes_time[ts.samples()])[::-1]
    if num_samples > len(sample_times):
        raise ValueError(f"Only {len(sample_times)} samples")
    return sample_times[num_samples - 1]


def sub_arg(ts, min_time):
    """
    Return the (sub_ts, node_ids) tuple for the ARG made from the nodes with
    time >= min_time, where node_ids gives the ID in ts of each node in
    sub_ts.
    """
    keep = ts.nodes_time >= min_time
    if not keep[0] or not keep[1]:
        raise ValueError("Nodes 0 and 1 must be kept")
    tables = ts.dump_tables()
    tables.edges.keep_rows(keep[tables.edges.child])
    assert np.all(keep[tables.edges.parent])
    # This also remaps the mutation parent column
    tables.mutations.keep_rows(keep[tables.mutations.node])
    node_map = tables.nodes.keep_rows(keep)
    tables.edges.parent = node_map[tables.edges.parent]
    tables.edges.child = node_map[tables.edges.child]
    tables.mutations.node = node_map[tables.mutations.node]
    tables.sort()
    node_ids = np.where(keep)[0].astype(np.int32)

    # Remove the parts of the ARG that are not ancestral to the remaining
    # samples. These are ancestors that sc2ts added later along with their
    # descendants (and tskit's engine, which tsls was compared with, required
    # every node in a tree to have samples below it). Simplifying without
    # filtering nodes keeps node IDs
    # (and so the vestigial root 0 above node 1), and the unary nodes
    # keep the sc2ts topology.
    tables.simplify(
        np.where(tables.nodes.flags & tskit.NODE_IS_SAMPLE)[0],
        keep_unary=True,
        keep_input_roots=True,
        filter_nodes=False,
        filter_sites=False,
        filter_populations=False,
        filter_individuals=False,
        update_sample_flags=False,
        record_provenance=False,
    )
    # Then remove the nodes that are now isolated.
    in_edge = np.zeros(len(tables.nodes), dtype=bool)
    in_edge[tables.edges.parent] = True
    in_edge[tables.edges.child] = True
    keep = in_edge | ((tables.nodes.flags & tskit.NODE_IS_SAMPLE) != 0)
    node_map = tables.nodes.keep_rows(keep)
    tables.edges.parent = node_map[tables.edges.parent]
    tables.edges.child = node_map[tables.edges.child]
    tables.mutations.node = node_map[tables.mutations.node]
    node_ids = node_ids[keep]
    tables.sort()
    tables.build_index()
    return tables.tree_sequence(), node_ids


@click.command()
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option(
    "--num-samples",
    type=str,
    default="10000,30000,100000,300000,1000000",
    show_default=True,
    help="Comma separated numbers of samples (oldest first) in the sub-ARGs",
)
def main(ts_path, out_dir, num_samples):
    """
    Write sub-ARGs of the benchmark ARG at TS_PATH to OUT_DIR.
    """
    ts = tskit.load(ts_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(ts_path).name.removesuffix(".trees")
    for n in [int(x) for x in num_samples.split(",")]:
        min_time = time_cutoff(ts, n)
        sub_ts, node_ids = sub_arg(ts, min_time)
        out_path = out_dir / f"{stem}_n{n}.trees"
        sub_ts.dump(out_path)
        np.save(node_ids_path(out_path), node_ids)
        click.echo(
            f"Wrote {out_path}: time >= {min_time:g}, {sub_ts.num_samples} samples, "
            f"{sub_ts.num_nodes} nodes, {sub_ts.num_edges} edges, "
            f"{sub_ts.num_trees} trees, {sub_ts.num_mutations} mutations"
        )


if __name__ == "__main__":
    main()
