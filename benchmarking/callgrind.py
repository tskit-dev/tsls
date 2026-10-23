"""
Count the instructions the HMM engines execute on the tsinfer workloads.

``run`` matches the first queries of a workload with one engine. ``measure``
runs ``run`` under callgrind for each workload, with collection limited to
the engine's find_path call, so that the counts are deterministic and leave
out setup (loading the data and building the engine). Setup also runs
uninstrumented, which makes it many times faster. It prints a markdown
table of total instructions (Ir) for each workload, comparing tsls with
tsinfer-hmm.

The workloads are sized so that measuring tsls takes a few minutes. The
tsinfer-hmm counts do not change, so they are measured once and reused from
the baseline directory.
"""

import concurrent.futures
import os
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
    "tsinfer": "ancestor_matcher_find_path",
    "tsls": "tsls_hmm_find_path",
}

# (workload, number of queries, maximum number of sites per query). Each is
# about 2G instructions in tsinfer-hmm. A whole sample is about 10G, so the
# sample is restricted to the start of its window: like a whole sample, its
# forward sweep then starts at the first site, without a long seek.
WORKLOADS = [
    ("ancestors_g84", 20, None),
    ("ancestors_g1000", 20, None),
    ("ancestors_g1483", 10, None),
    ("ancestors_g1484", 3, None),
    ("ancestors_g1485", 5, None),
    ("samples", 1, 100_000),
]


def make_engine(name, ts, model):
    return engines.ENGINES[name](
        ts,
        mu=model["mu"],
        rho=model["rho"],
        weight_by_n=model["weight_by_n"],
        alleles=model["alleles"],
    )


def clip_window(start, end, max_sites):
    """
    Return the first max_sites sites of the window [start, end).
    """
    if max_sites is None:
        return start, end
    return start, min(end, start + max_sites)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("workload_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("engine_name", type=click.Choice(list(FIND_PATH_FUNCTIONS)))
@click.option("-n", "--num-queries", type=int, default=20)
@click.option("--max-sites", type=int, default=None)
@click.option(
    "--instrument",
    is_flag=True,
    help="Switch on callgrind instrumentation after setup",
)
def run(workload_dir, engine_name, num_queries, max_sites, instrument):
    """
    Match the first NUM_QUERIES queries in WORKLOAD_DIR with the specified
    engine, each over at most MAX_SITES sites at the start of its window,
    printing the parents of each path.
    """
    workload_dir = pathlib.Path(workload_dir)
    model = queries.load_model(workload_dir)
    ts = run_module.load_ts(workload_dir / "reference.trees")
    H, _, df = queries.load_queries(workload_dir)
    engine = make_engine(engine_name, ts, model)
    if instrument:
        subprocess.run(
            ["callgrind_control", "-i", "on", str(os.getpid())],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for query_id in df.query_id[:num_queries]:
        start, end = queries.query_window(H, query_id)
        start, end = clip_window(start, end, max_sites)
        _, _, parent, _, _ = engine.match(H[query_id], start, end)
        print(query_id, " ".join(str(p) for p in np.asarray(parent)))


def read_total_ir(path):
    with open(path) as f:
        for line in f:
            if line.startswith(("totals:", "summary:")):
                return int(line.split()[1])
    raise ValueError(f"No totals in {path}")


def out_path(out_dir, workload, num_queries, max_sites, engine_name):
    sites = "" if max_sites is None else f".first{max_sites}"
    return pathlib.Path(out_dir) / (
        f"callgrind.{workload}.n{num_queries}{sites}.{engine_name}.out"
    )


def callgrind(data_dir, workload, num_queries, max_sites, engine_name, out_file):
    cmd = [
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        "--collect-atstart=no",
        f"--toggle-collect={FIND_PATH_FUNCTIONS[engine_name]}",
        f"--callgrind-out-file={out_file}",
        sys.executable,
        __file__,
        "run",
        str(pathlib.Path(data_dir) / workload),
        engine_name,
        "-n",
        str(num_queries),
        "--instrument",
    ]
    if max_sites is not None:
        cmd += ["--max-sites", str(max_sites)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr[-2000:]}")
    paths_file = out_file.with_suffix(".paths")
    paths_file.write_text(result.stdout)
    return read_total_ir(out_file), result.stdout


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/tgp_chr20p",
)
@click.option("-w", "--workload", "workload_names", multiple=True)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(file_okay=False),
    default="callgrind",
    help="Directory for the tsls callgrind output files",
)
@click.option(
    "--baseline-dir",
    type=click.Path(file_okay=False),
    default="callgrind/tsinfer",
    help="Directory for the tsinfer-hmm callgrind output files, which are reused",
)
@click.option("-j", "--jobs", type=int, default=3)
def measure(data_dir, workload_names, out_dir, baseline_dir, jobs):
    """
    Count the instructions in find_path for each workload, for tsls and
    (unless already measured) tsinfer-hmm. The callgrind output files are
    kept for callgrind_annotate.
    """
    workloads = WORKLOADS
    if len(workload_names) > 0:
        workloads = [w for w in WORKLOADS if w[0] in workload_names]
    tasks = []
    for workload, num_queries, max_sites in workloads:
        for engine_name, directory in [("tsinfer", baseline_dir), ("tsls", out_dir)]:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            path = out_path(directory, workload, num_queries, max_sites, engine_name)
            if engine_name == "tsinfer" and path.exists():
                continue
            tasks.append((workload, num_queries, max_sites, engine_name, path))
    # Run the largest workloads first, so that they don't finish last
    tasks.sort(key=lambda task: task[0] not in ("samples", "ancestors_g1485"))
    with concurrent.futures.ThreadPoolExecutor(jobs) as executor:
        futures = [executor.submit(callgrind, data_dir, *task) for task in tasks]
        for future in futures:
            future.result()

    click.echo("| workload | queries | tsinfer | tsls | ratio |")
    click.echo("|---|---|---|---|---|")
    for workload, num_queries, max_sites in workloads:
        counts = []
        paths = []
        for engine_name, directory in [("tsinfer", baseline_dir), ("tsls", out_dir)]:
            path = out_path(directory, workload, num_queries, max_sites, engine_name)
            counts.append(read_total_ir(path))
            paths.append(path.with_suffix(".paths").read_text())
        label = (
            f"{num_queries}"
            if max_sites is None
            else f"{num_queries} ({max_sites} sites)"
        )
        click.echo(
            f"| {workload} | {label} | {counts[0] / 1e9:.3f} G | "
            f"{counts[1] / 1e9:.3f} G | {counts[1] / counts[0]:.3f} |"
        )
        if paths[0] != paths[1]:
            click.echo(f"WARNING: engines found different paths for {workload}")


if __name__ == "__main__":
    cli()
