"""
Count the instructions the HMM engines execute on the tsinfer workloads.

``run`` matches the first queries of a workload with one engine. ``measure``
runs ``run`` under callgrind, with collection limited to the engine's
find_path call, so that the counts are deterministic and leave out setup
(loading the data and building the engine). It prints a markdown table of
total instructions (Ir) for each workload and engine.
"""

import concurrent.futures
import pathlib
import subprocess
import sys

import click
import numpy as np

import engines
import queries
import run as run_module

# The C function that runs one Viterbi match in each engine. Collection is
# switched on only inside these.
FIND_PATH_FUNCTIONS = {
    "tsinfer": ["ancestor_matcher_find_path"],
    "tsls": ["tsls_hmm_find_path", "ancestor_matcher_find_path"],
}

DEFAULT_WORKLOADS = [
    ("ancestors_g84", 20),
    ("ancestors_g1000", 20),
    ("ancestors_g1483", 20),
    ("ancestors_g1484", 20),
    ("ancestors_g1485", 20),
    ("samples", 1),
]


def make_engine(name, ts, model):
    return engines.ENGINES[name](
        ts,
        mu=model["mu"],
        rho=model["rho"],
        weight_by_n=model["weight_by_n"],
        alleles=model["alleles"],
    )


@click.group()
def cli():
    pass


@cli.command()
@click.argument("workload_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("engine_name", type=click.Choice(list(FIND_PATH_FUNCTIONS)))
@click.option("-n", "--num-queries", type=int, default=20)
def run(workload_dir, engine_name, num_queries):
    """
    Match the first NUM_QUERIES queries in WORKLOAD_DIR with the specified
    engine, printing the parents of each path.
    """
    workload_dir = pathlib.Path(workload_dir)
    model = queries.load_model(workload_dir)
    ts = run_module.load_ts(workload_dir / "reference.trees")
    H, _, df = queries.load_queries(workload_dir)
    engine = make_engine(engine_name, ts, model)
    for query_id in df.query_id[:num_queries]:
        start, end = queries.query_window(H, query_id)
        _, _, parent, _, _ = engine.match(H[query_id], start, end)
        print(query_id, " ".join(str(p) for p in np.asarray(parent)))


def read_total_ir(path):
    with open(path) as f:
        for line in f:
            if line.startswith("totals:") or line.startswith("summary:"):
                return int(line.split()[1])
    raise ValueError(f"No totals in {path}")


def callgrind(workload_dir, engine_name, num_queries, out_dir):
    workload = pathlib.Path(workload_dir).name
    out_file = pathlib.Path(out_dir) / f"callgrind.{workload}.{engine_name}.out"
    toggles = [f"--toggle-collect={f}" for f in FIND_PATH_FUNCTIONS[engine_name]]
    cmd = [
        "valgrind",
        "--tool=callgrind",
        "--collect-atstart=no",
        *toggles,
        f"--callgrind-out-file={out_file}",
        sys.executable,
        __file__,
        "run",
        str(workload_dir),
        engine_name,
        "-n",
        str(num_queries),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr[-2000:]}")
    return read_total_ir(out_file), result.stdout


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/tgp_chr20p",
)
@click.option("-w", "--workload", "workload_names", multiple=True)
@click.option(
    "-e",
    "--engine",
    "engine_names",
    multiple=True,
    type=click.Choice(list(FIND_PATH_FUNCTIONS)),
)
@click.option("-o", "--out-dir", type=click.Path(file_okay=False), default="callgrind")
@click.option("-j", "--jobs", type=int, default=3)
def measure(data_dir, workload_names, engine_names, out_dir, jobs):
    """
    Count the instructions in find_path for each workload and engine. The
    callgrind output files are kept in OUT_DIR for callgrind_annotate.
    """
    workloads = DEFAULT_WORKLOADS
    if len(workload_names) > 0:
        workloads = [w for w in DEFAULT_WORKLOADS if w[0] in workload_names]
    if len(engine_names) == 0:
        engine_names = list(FIND_PATH_FUNCTIONS)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    tasks = [
        (workload, num_queries, engine)
        for workload, num_queries in workloads
        for engine in engine_names
    ]
    with concurrent.futures.ThreadPoolExecutor(jobs) as executor:
        futures = {
            task: executor.submit(
                callgrind, pathlib.Path(data_dir) / task[0], task[2], task[1], out_dir
            )
            for task in tasks
        }
        results = {task: future.result() for task, future in futures.items()}

    header = "| workload | queries | " + " | ".join(engine_names) + " |"
    if len(engine_names) == 2:
        header += " ratio |"
    click.echo(header)
    click.echo("|---" * (header.count("|") - 1) + "|")
    for workload, num_queries in workloads:
        counts = [results[(workload, num_queries, e)][0] for e in engine_names]
        row = f"| {workload} | {num_queries} | "
        row += " | ".join(f"{c / 1e9:.3f} G" for c in counts) + " |"
        if len(engine_names) == 2:
            row += f" {counts[1] / counts[0]:.3f} |"
        click.echo(row)
        paths = {results[(workload, num_queries, e)][1] for e in engine_names}
        if len(paths) > 1:
            click.echo(f"WARNING: engines found different paths for {workload}")


if __name__ == "__main__":
    cli()
