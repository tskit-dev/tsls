"""
Summarise run.py results on workloads made by tsinfer_data.py.

For each workload, report how often tsls found the same path as tsinfer,
along with its time and memory, and the time tsinfer
recorded for matching the whole group. The model parameters are reported
too, so that runs of the same workload with different parameters (such as
a recombination map) can be compared.
"""

import json
import pathlib

import click
import numpy as np
import pandas as pd


def parse_stats(stats):
    """
    Return the dict of tsls statistics in a stats column value.
    """
    items = (item.split("=") for item in str(stats).split())
    return {key: float(value) for key, value in items}


def join_truth(results, query_dir):
    """
    Return the results joined with the query metadata, with a same_path
    column saying whether the path is the one tsinfer found.
    """
    df = pd.read_csv(query_dir / "queries.csv")
    df = results.merge(df, on="query_id", how="left", validate="many_to_one")

    def column(name):
        return df[name].fillna("").astype(str)

    df["same_path"] = (column("parents") == column("truth_parents")) & (
        column("breakpoints") == column("truth_breakpoints")
    )
    stats = pd.DataFrame([parse_stats(s) for s in df.stats], index=df.index)
    return pd.concat([df, stats], axis=1)


def recombination_label(d):
    """
    Return a label for the recombination parameters of the results: the
    rate or genetic map file name if run.py was given one, otherwise rho.
    """
    rates = d.get("recombination_rate", pd.Series(dtype=object)).dropna()
    if len(rates) > 0:
        return " ".join(sorted({pathlib.Path(str(r)).name for r in rates}))
    return " ".join(f"{rho:.3g}" for rho in sorted(d.rho.unique()))


def summarise(results_path, query_dir):
    query_dir = pathlib.Path(query_dir)
    df = join_truth(pd.read_csv(results_path), query_dir)
    with open(query_dir / "tsinfer_group.json") as f:
        group = json.load(f)
    ok = df[df.status == "ok"]
    row = {
        "workload": query_dir.name,
        "recombination": recombination_label(df),
        "mu": " ".join(f"{mu:.3g}" for mu in sorted(df.mu.unique())),
        "queries": len(df),
        "failed": int(np.sum(df.status != "ok")),
        "mean sites": np.mean(ok.end - ok.start),
        "same path": np.mean(ok.same_path),
        "segment diff": np.mean(ok.num_segments - ok.truth_num_segments),
        "mean mismatches": np.mean(ok.num_mismatches),
        "mean time (s)": np.mean(ok.wall_time),
        "max time (s)": np.max(ok.wall_time),
        "mean memory (MiB)": np.mean(ok.total_memory) / 2**20,
        "mean traceback": np.mean(ok.mean_traceback_size),
        "group size": group["num_haplotypes"],
        "tsinfer CPU time per haplotype (s)": group["user_time"]
        / group["num_haplotypes"],
    }
    return pd.DataFrame([row])


@click.command()
@click.argument(
    "pairs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True),
    metavar="RESULTS QUERY_DIR ...",
)
def main(pairs):
    """
    Summarise each RESULTS file from run.py on the workload in QUERY_DIR,
    as a markdown table.
    """
    if len(pairs) % 2 != 0:
        raise click.UsageError("Specify RESULTS QUERY_DIR pairs")
    tables = [
        summarise(results_path, query_dir)
        for results_path, query_dir in zip(pairs[::2], pairs[1::2])
    ]
    click.echo(pd.concat(tables).to_markdown(index=False, floatfmt=".3g"))


if __name__ == "__main__":
    main()
