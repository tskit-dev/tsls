"""
Compute tsinfer's match groups for the 1000 Genomes chr20p data.

The ancestors tree sequence does not record which nodes were matched in
which group, so we recompute the groups with tsinfer's own grouping code.
This script must be run in the tsinfer environment, which keeps tsinfer
(GPL) out of the benchmarking dependencies:

    uv run --project ../../tsinfer python tsinfer_groups.py data/tgp_chr20p

It writes match_jobs.csv to the data directory, with one row per matched
haplotype (ancestors then samples), ordered by group then haplotype index
as in tsinfer. The order is checked against the node order of the
ancestors tree sequence, and group 84 against the group.json file from
the partial ancestor match.
"""

import json
import pathlib
import sys
import tomllib

import numpy as np
import pandas as pd
import tskit
from tsinfer import config, pipeline

COLUMNS = [
    "haplotype_index",
    "source",
    "sample_id",
    "ploidy_index",
    "time",
    "start_position",
    "end_position",
    "group",
]


def load_config(data_dir):
    """
    Return the tsinfer Config for the inference, with paths pointing to the
    local copies of the stores. The toml file has absolute paths on the
    cluster where it was run, and tsinfer's config has no API for changing
    them, so we change the raw dict and use the private parser.
    """
    with open(data_dir / "full_inference.toml", "rb") as f:
        raw = tomllib.load(f)
    samples_path = str(data_dir / "tgp_chr20p_samples_masked.zarr")
    ancestors_path = str(data_dir / "tgp_chr20p_inferred_ancestors.zarr")
    for source in raw["source"]:
        assert source["name"] == "samples"
        source["path"] = samples_path
    raw["ancestral_state"]["path"] = samples_path
    for ancestors in raw["ancestors"]:
        ancestors["path"] = ancestors_path
    raw["match"]["output"] = str(data_dir / "unused.trees")
    return config._parse_config(raw)


def check_ancestor_order(jobs, ts):
    """
    The nodes of the ancestors tree sequence are the two roots followed by
    the ancestors in match order, so their sample IDs must be in the same
    order as the ancestor jobs.
    """
    ancestors = jobs[jobs.source == "ancestors"]
    assert ts.num_nodes == len(ancestors) + 2
    node_ids = [ts.node(u).metadata["sample_id"] for u in range(2, ts.num_nodes)]
    if node_ids != list(ancestors.sample_id):
        raise ValueError("Ancestor job order does not match tree sequence node order")


def check_group(jobs, group_json_path):
    with open(group_json_path) as f:
        group = json.load(f)
    expected = [job["haplotype_index"] for job in group["jobs"]]
    actual = list(jobs.haplotype_index[jobs.group == group["group"]])
    if actual != expected:
        raise ValueError(f"Group {group['group']} does not match {group_json_path}")


def main(data_dir):
    data_dir = pathlib.Path(data_dir).resolve()
    cfg = load_config(data_dir)
    jobs = pipeline._build_jobs_with_metadata(cfg)[0]
    df = pd.DataFrame(
        {column: [getattr(job, column) for job in jobs] for column in COLUMNS}
    )
    ts = tskit.load(data_dir / "tgp_chr20p_inferred_ancestors.trees")
    check_ancestor_order(df, ts)
    check_group(df, data_dir / "partial_ancestor_match" / "group.json")
    sample_groups = np.unique(df.group[df.source == "samples"])
    assert len(sample_groups) == 1
    out_path = data_dir / "match_jobs.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} jobs in {df.group.nunique()} groups to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
