"""
Generate query haplotypes for benchmarking tsls.

Queries are haplotypes over the sites of a benchmark ARG, encoded as
int8 indexes into ALLELES (-1 is missing data). There are two kinds:

- copy: the haplotype of a randomly chosen sample node, with k random
  mutations applied;
- recombinant: a mosaic of two randomly chosen sample nodes A and B with
  one (A|B) or two (A|B|A) breakpoints, with k random mutations applied.

Each query has a row of metadata recording how it was made, so that
we can compare the paths with the truth. Node IDs in the
metadata are original IDs if the ARG is a sub-ARG (see subargs.py).
"""

import dataclasses
import json
import pathlib

import click
import numpy as np
import pandas as pd
import tskit

import subargs

ALLELES = ("A", "C", "G", "T")
MISSING = -1

QUERY_COLUMNS = [
    "query_id",
    "kind",
    "nodes",
    "k",
    "mutated_sites",
    "breakpoints",
    "breakpoint_intervals",
    "parent_diff",
    "segment_diffs",
]


def format_list(values):
    return " ".join(str(int(x)) for x in values)


def parse_list(value):
    """
    Parse a space separated list of ints written by format_list, as read
    back by pandas (which gives NaN for empty strings and an int for
    single values).
    """
    if isinstance(value, float) and np.isnan(value):
        return []
    return [int(x) for x in str(value).split()]


def node_haplotypes(ts, nodes):
    """
    Return the haplotypes of the specified nodes as a (len(nodes), num_sites)
    int8 array of allele indexes.
    """
    G = ts.genotype_matrix(samples=nodes, isolated_as_missing=False, alleles=ALLELES)
    return G.T.astype(np.int8)


def mutate(h, k, rng):
    """
    Return the (haplotype, sites) tuple resulting from changing the allele
    at k distinct randomly chosen sites of the specified haplotype to a
    different allele chosen uniformly at random. Missing sites are not
    mutated.
    """
    h = h.copy()
    candidates = np.where(h != MISSING)[0]
    sites = np.sort(rng.choice(candidates, size=k, replace=False))
    offset = rng.integers(1, len(ALLELES), size=k)
    h[sites] = (h[sites] + offset) % len(ALLELES)
    return h, sites


def choose_breakpoints(diff_sites, num_breakpoints, min_segment_diffs, rng):
    """
    Choose breakpoints for a recombinant between two parents that differ at
    the specified (sorted) sites, such that each of the num_breakpoints + 1
    segments contains at least min_segment_diffs differing sites.

    Return the (breakpoints, intervals, segment_diffs) tuple. Breakpoints are
    site indexes: a segment starting at breakpoint b includes site b. Any
    breakpoint in the interval (lo, hi] gives the same haplotype, where lo
    and hi are the site indexes of the flanking differing sites. Return None
    if the parents do not differ at enough sites.
    """
    num_segments = num_breakpoints + 1
    extra = len(diff_sites) - num_segments * min_segment_diffs
    if extra < 0:
        return None
    # Distribute the extra differing sites uniformly among segments.
    cuts = np.sort(rng.integers(0, extra + 1, size=num_segments - 1))
    sizes = np.diff(np.concatenate([[0], cuts, [extra]])) + min_segment_diffs
    assert np.sum(sizes) == len(diff_sites)
    cut_index = np.cumsum(sizes)[:-1]
    lo = diff_sites[cut_index - 1]
    hi = diff_sites[cut_index]
    breakpoints = rng.integers(lo + 1, hi + 1)
    intervals = np.stack([lo, hi], axis=1)
    return breakpoints, intervals, sizes


def make_recombinant(parent_haplotypes, breakpoints):
    """
    Return the haplotype made by alternating between the two specified parent
    haplotypes at the specified breakpoint sites, starting with the first.
    """
    a, b = parent_haplotypes
    h = a.copy()
    bounds = np.concatenate([[0], breakpoints, [len(a)]])
    for j in range(1, len(bounds) - 1, 2):
        h[bounds[j] : bounds[j + 1]] = b[bounds[j] : bounds[j + 1]]
    return h


@dataclasses.dataclass
class QuerySet:
    """
    Accumulates query haplotypes and their metadata.
    """

    positions: np.ndarray
    # The original node ID of each node, if the ARG is a sub-ARG
    node_ids: np.ndarray = None
    haplotypes: list = dataclasses.field(default_factory=list)
    rows: list = dataclasses.field(default_factory=list)

    def format_nodes(self, nodes):
        nodes = np.asarray(nodes)
        if self.node_ids is not None:
            nodes = self.node_ids[nodes]
        return format_list(nodes)

    def add(self, h, **metadata):
        row = {column: "" for column in QUERY_COLUMNS}
        row.update(metadata)
        row["query_id"] = len(self.rows)
        self.rows.append(row)
        self.haplotypes.append(h)

    def dataframe(self):
        return pd.DataFrame(self.rows, columns=QUERY_COLUMNS)

    def save(self, out_dir):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        H = np.array(self.haplotypes, dtype=np.int8).reshape(
            (len(self.haplotypes), len(self.positions))
        )
        np.savez_compressed(out_dir / "queries.npz", H=H, positions=self.positions)
        self.dataframe().to_csv(out_dir / "queries.csv", index=False)


class WindowedHaplotypes:
    """
    Query haplotypes that are only defined over a window of sites, such as
    tsinfer ancestors. Query j has the specified values at sites start[j] to
    end[j] (exclusive) and is missing elsewhere. The values of all queries
    are concatenated in a single array, so that short queries over many sites
    do not need much space.
    """

    def __init__(self, num_sites, start, end, values):
        self.num_sites = num_sites
        self.start = np.asarray(start, dtype=np.int64)
        self.end = np.asarray(end, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.int8)
        lengths = self.end - self.start
        assert np.all(lengths >= 0)
        assert np.all(self.end <= num_sites)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        assert self.offsets[-1] == len(self.values)

    @staticmethod
    def from_windows(num_sites, start, windows):
        """
        Return the WindowedHaplotypes made from the specified start sites and
        list of window value arrays.
        """
        start = np.asarray(start, dtype=np.int64)
        end = start + np.array([len(w) for w in windows], dtype=np.int64)
        values = np.concatenate([np.zeros(0, dtype=np.int8), *windows])
        return WindowedHaplotypes(num_sites, start, end, values)

    def __len__(self):
        return len(self.start)

    def window(self, j):
        return self.values[self.offsets[j] : self.offsets[j + 1]]

    def __getitem__(self, j):
        h = np.full(self.num_sites, MISSING, dtype=np.int8)
        h[self.start[j] : self.end[j]] = self.window(j)
        return h

    def save(self, out_dir, positions, df):
        assert len(df) == len(self)
        assert len(positions) == self.num_sites
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "queries.npz",
            positions=positions,
            start=self.start,
            end=self.end,
            values=self.values,
        )
        df.to_csv(out_dir / "queries.csv", index=False)


def load_queries(query_dir):
    """
    Return the (H, positions, metadata) tuple for the query set in the
    specified directory. H is either an array with one row per query, or
    a WindowedHaplotypes; both give the haplotype of query j as H[j].
    """
    query_dir = pathlib.Path(query_dir)
    data = np.load(query_dir / "queries.npz")
    df = pd.read_csv(query_dir / "queries.csv")
    positions = data["positions"]
    if "values" in data:
        H = WindowedHaplotypes(
            len(positions), data["start"], data["end"], data["values"]
        )
    else:
        H = data["H"]
    assert len(df) == len(H)
    return H, positions, df


def query_window(H, query_id):
    """
    Return the (start, end) tuple of the sites to match for the specified
    query.
    """
    if isinstance(H, WindowedHaplotypes):
        return int(H.start[query_id]), int(H.end[query_id])
    return 0, H.shape[1]


def load_model(query_dir):
    """
    Return the model parameters stored with the query set in the specified
    directory, or None if there are none. Query sets made from tsinfer runs
    store the parameters tsinfer used, which take the place of the sc2ts
    style parameters chosen by run.py.
    """
    path = pathlib.Path(query_dir) / "model.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def add_copy_queries(query_set, ts, rng, *, num_nodes, divergence):
    nodes = rng.choice(ts.samples(), size=num_nodes, replace=False)
    H = node_haplotypes(ts, nodes)
    for node, h in zip(nodes, H):
        for k in divergence:
            hq, sites = mutate(h, k, rng)
            query_set.add(
                hq,
                kind="copy",
                nodes=query_set.format_nodes([node]),
                k=k,
                mutated_sites=format_list(sites),
            )


def add_recombinant_queries(
    query_set,
    ts,
    rng,
    *,
    num_recombinants,
    num_breakpoints,
    divergence,
    min_segment_diffs,
):
    # Draw more candidate pairs than we need so that we can reject pairs that
    # are too similar, and decode all their haplotypes in one pass.
    num_candidates = 4 * num_recombinants
    pairs = np.array(
        [rng.choice(ts.samples(), size=2, replace=False) for _ in range(num_candidates)]
    )
    nodes, index = np.unique(pairs, return_inverse=True)
    index = index.reshape(pairs.shape)
    H = node_haplotypes(ts, nodes)
    num_made = 0
    for pair, (i, j) in zip(pairs, index):
        if num_made == num_recombinants:
            break
        parents = H[i], H[j]
        diff_sites = np.where(parents[0] != parents[1])[0]
        result = choose_breakpoints(diff_sites, num_breakpoints, min_segment_diffs, rng)
        if result is None:
            continue
        breakpoints, intervals, segment_diffs = result
        h = make_recombinant(parents, breakpoints)
        for k in divergence:
            hq, sites = mutate(h, k, rng)
            query_set.add(
                hq,
                kind="recombinant",
                nodes=query_set.format_nodes(pair),
                k=k,
                mutated_sites=format_list(sites),
                breakpoints=format_list(breakpoints),
                breakpoint_intervals=format_list(intervals.ravel()),
                parent_diff=len(diff_sites),
                segment_diffs=format_list(segment_diffs),
            )
        num_made += 1
    if num_made < num_recombinants:
        raise ValueError(
            f"Only found {num_made} of {num_recombinants} parent pairs with "
            f"enough differing sites"
        )


def make_queries(
    ts,
    *,
    seed,
    num_nodes=50,
    divergence=tuple(range(11)),
    num_recombinants=50,
    num_breakpoints=(1, 2),
    recombinant_divergence=(0, 2, 5),
    min_segment_diffs=2,
    node_ids=None,
):
    rng = np.random.default_rng(seed)
    query_set = QuerySet(positions=ts.sites_position.copy(), node_ids=node_ids)
    add_copy_queries(query_set, ts, rng, num_nodes=num_nodes, divergence=divergence)
    for nb in num_breakpoints:
        add_recombinant_queries(
            query_set,
            ts,
            rng,
            num_recombinants=num_recombinants,
            num_breakpoints=nb,
            divergence=recombinant_divergence,
            min_segment_diffs=min_segment_diffs,
        )
    return query_set


class IntList(click.ParamType):
    """
    A comma separated list of ints, where a-b is the inclusive range.
    """

    name = "int_list"

    def convert(self, value, param, ctx):
        if isinstance(value, tuple):
            return value
        values = []
        try:
            for part in value.split(","):
                if "-" in part:
                    start, stop = part.split("-")
                    values.extend(range(int(start), int(stop) + 1))
                else:
                    values.append(int(part))
        except ValueError:
            self.fail(f"{value!r} is not a list of ints", param, ctx)
        return tuple(values)


@click.command()
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--num-nodes",
    type=int,
    default=50,
    show_default=True,
    help="Number of sample nodes to make copy queries from",
)
@click.option(
    "--divergence",
    type=IntList(),
    default="0-10",
    show_default=True,
    help="Numbers of mutations to add to copy queries",
)
@click.option(
    "--num-recombinants",
    type=int,
    default=50,
    show_default=True,
    help="Number of recombinants for each number of breakpoints",
)
@click.option(
    "--num-breakpoints",
    type=IntList(),
    default="1,2",
    show_default=True,
    help="Numbers of breakpoints in recombinants",
)
@click.option(
    "--recombinant-divergence",
    type=IntList(),
    default="0,2,5",
    show_default=True,
    help="Numbers of mutations to add to recombinant queries",
)
@click.option(
    "--min-segment-diffs",
    type=int,
    default=2,
    show_default=True,
    help="Minimum number of sites distinguishing the parents in each segment",
)
def main(
    ts_path,
    out_dir,
    seed,
    num_nodes,
    divergence,
    num_recombinants,
    num_breakpoints,
    recombinant_divergence,
    min_segment_diffs,
):
    """
    Generate queries from the benchmark ARG at TS_PATH and write them to
    OUT_DIR.
    """
    ts = tskit.load(ts_path)
    node_ids = subargs.load_node_ids(ts_path, ts)
    query_set = make_queries(
        ts,
        seed=seed,
        num_nodes=num_nodes,
        divergence=divergence,
        num_recombinants=num_recombinants,
        num_breakpoints=num_breakpoints,
        recombinant_divergence=recombinant_divergence,
        min_segment_diffs=min_segment_diffs,
        node_ids=node_ids,
    )
    query_set.save(out_dir)
    df = query_set.dataframe()
    click.echo(f"Wrote {len(df)} queries to {out_dir}")
    click.echo(df.groupby(["kind", "k"]).size().to_string())


if __name__ == "__main__":
    main()
