"""
Compare the wall time of two or more builds of the tsls extension on chr20p
sample matching, with paired runs.

Each build is a directory holding a _tsls extension module, put first on
PYTHONPATH ("." means the in-tree build). For each round, every build runs
profile_samples.py in its own process, pinned to one core, in turn; the
best time for each (build, query) over the rounds is kept. Reports the
total time of each build over the first, and classifies each build's paths
against the first build's: the same path, a different path with the same
model log likelihood (a tie, within a relative 1e-9), or a better or worse
one.

    uv run python ab.py base=/path/to/build new=. --map data/genetic_maps/... -r 3
    uv run python ab.py base=... new=. --sc2ts data/subargs/sc2ts_bench_n100000.trees
"""

import collections
import json
import os
import subprocess
import sys

import click


def run_build(path, script, args, core):
    env = dict(os.environ)
    if path != ".":
        env["PYTHONPATH"] = path
    command = ["taskset", "-c", str(core), sys.executable, script, *args]
    out = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


@click.command()
@click.argument("builds", nargs=-1, required=True)
@click.option("--map", "genetic_map", default=None)
@click.option(
    "--sc2ts",
    "sc2ts_path",
    default=None,
    help="Time sc2ts queries against this ARG (profile_sc2ts.py) instead",
)
@click.option("--k", "mismatch_threshold", type=int, default=None)
@click.option("--index-min-sites", type=int, default=None)
@click.option("--mu", type=float, default=1e-20)
@click.option("--queries", "query_ids", default=None)
@click.option("-r", "--rounds", type=int, default=3)
@click.option("--core", type=int, default=2)
def main(
    builds,
    genetic_map,
    sc2ts_path,
    mismatch_threshold,
    index_min_sites,
    mu,
    query_ids,
    rounds,
    core,
):
    named = [b.split("=", 1) for b in builds]
    if sc2ts_path is None:
        script = "profile_samples.py"
        args = ["--mu", str(mu), "--queries", query_ids or "0,40,80,120,160,199"]
        if genetic_map is not None:
            args += ["--map", genetic_map]
    else:
        script = "profile_sc2ts.py"
        args = [sc2ts_path]
        if query_ids is not None:
            args += ["--queries", query_ids]
        if mismatch_threshold is not None:
            args += ["--k", str(mismatch_threshold)]
    if index_min_sites is not None:
        args += ["--index-min-sites", str(index_min_sites)]
    best = collections.defaultdict(dict)
    paths = collections.defaultdict(dict)
    log_likelihoods = collections.defaultdict(dict)
    for round_ in range(rounds):
        for name, path in named:
            for row in run_build(path, script, args, core):
                q = row["query"]
                best[name][q] = min(best[name].get(q, float("inf")), row["time"])
                paths[name][q] = row["path"]
                log_likelihoods[name][q] = row["log_likelihood"]
        totals = {name: sum(best[name].values()) for name, _ in named}
        base = named[0][0]
        click.echo(
            f"round {round_ + 1}: "
            + ", ".join(
                f"{name} {totals[name]:.2f}s ({totals[name] / totals[base]:.3f})"
                for name, _ in named
            )
        )
    base = named[0][0]
    for name, _ in named[1:]:
        counts = collections.Counter()
        for q, path in paths[name].items():
            x, y = log_likelihoods[name][q], log_likelihoods[base][q]
            if path == paths[base][q]:
                counts["same"] += 1
            elif abs(x - y) <= 1e-9 * abs(y):
                counts["tie"] += 1
            elif x > y:
                counts["better"] += 1
            else:
                counts["worse"] += 1
        summary = "/".join(str(counts[k]) for k in ["same", "tie", "better", "worse"])
        click.echo(f"{name}: paths same/tie/better/worse {summary}")


if __name__ == "__main__":
    main()
