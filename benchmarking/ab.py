"""
Compare the wall time of two or more builds of the tsls extension on chr20p
sample matching, with paired runs.

Each build is a directory holding a _tsls extension module, put first on
PYTHONPATH ("." means the in-tree build). For each round, every build runs
profile_samples.py in its own process, pinned to one core, in turn; the
best time for each (build, query) over the rounds is kept. Reports the
total time of each build over the first, and checks that every build finds
the same paths.

    uv run python ab.py base=/path/to/build new=. --map data/genetic_maps/... -r 3
"""

import collections
import json
import os
import subprocess
import sys

import click


def run_build(path, args, core):
    env = dict(os.environ)
    if path != ".":
        env["PYTHONPATH"] = path
    command = ["taskset", "-c", str(core), sys.executable, "profile_samples.py", *args]
    out = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


@click.command()
@click.argument("builds", nargs=-1, required=True)
@click.option("--map", "genetic_map", default=None)
@click.option("--mu", type=float, default=1e-20)
@click.option("--queries", "query_ids", default="0,40,80,120,160,199")
@click.option("-r", "--rounds", type=int, default=3)
@click.option("--core", type=int, default=2)
def main(builds, genetic_map, mu, query_ids, rounds, core):
    named = [b.split("=", 1) for b in builds]
    args = ["--mu", str(mu), "--queries", query_ids]
    if genetic_map is not None:
        args += ["--map", genetic_map]
    best = collections.defaultdict(dict)
    paths = collections.defaultdict(dict)
    for round_ in range(rounds):
        for name, path in named:
            for row in run_build(path, args, core):
                q = row["query"]
                best[name][q] = min(best[name].get(q, float("inf")), row["time"])
                paths[name][q] = row["path"]
        totals = {name: sum(best[name].values()) for name, _ in named}
        base = named[0][0]
        click.echo(
            f"round {round_ + 1}: "
            + ", ".join(
                f"{name} {totals[name]:.2f}s ({totals[name] / totals[base]:.3f})"
                for name, _ in named
            )
        )
    for name, _ in named[1:]:
        same = paths[name] == paths[named[0][0]]
        click.echo(f"{name}: paths {'identical' if same else 'DIFFER'}")


if __name__ == "__main__":
    main()
