"""
Count the instructions tsls executes on the tsinfer workloads.

``run`` matches the first queries of a workload. ``measure`` runs ``run``
under callgrind for each workload, with collection limited to tsls's
find_path call, so that the counts are deterministic and leave out setup
(loading the data and building the HMM). Setup also runs uninstrumented,
which makes it many times faster. It prints a markdown table of total
instructions (Ir) for each workload, and, if a baseline directory of
earlier counts is given, the ratio to them.

The workloads are sized so that measuring takes a few minutes.
"""

import concurrent.futures
import os
import pathlib
import subprocess
import sys

import click
import numpy as np

import matcher
import queries
import run as run_module

# The C function that runs one Viterbi match. Collection is switched on
# only inside it.
FIND_PATH_FUNCTION = "tsls_hmm_find_path"

# (workload, number of queries, number of sites). Each was about 2G
# instructions in tsinfer-hmm. For the sample, the reference is cut at the
# specified number of sites, and the sample is matched over all of them,
# rather than matching just the first sites of the full reference.
WORKLOADS = [
    ("ancestors_g84", 20, None),
    ("ancestors_g1000", 20, None),
    ("ancestors_g1483", 10, None),
    ("ancestors_g1484", 3, None),
    ("ancestors_g1485", 5, None),
    ("samples", 1, 100_000),
]


def make_matcher(ts, model, mismatch_threshold=None):
    return matcher.Matcher(
        ts,
        mu=model["mu"],
        rho=model["rho"],
        weight_by_n=model["weight_by_n"],
        alleles=model["alleles"],
        mismatch_threshold=mismatch_threshold,
    )


def cut_reference(ts, num_sites):
    """
    Return the tree sequence cut just before the specified site, so that it
    has only the first num_sites sites. Node IDs are unchanged.
    """
    cut = ts.sites_position[num_sites]
    return ts.keep_intervals([[0, cut]], simplify=False).rtrim()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("workload_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-n", "--num-queries", type=int, default=20)
@click.option("--num-sites", type=int, default=None)
@click.option(
    "--instrument",
    is_flag=True,
    help="Switch on callgrind instrumentation after setup",
)
def run(workload_dir, num_queries, num_sites, instrument):
    """
    Match the first NUM_QUERIES queries in WORKLOAD_DIR, printing the
    parents of each path. If NUM_SITES is specified,
    the reference is cut after that many sites, and the queries are matched
    over their windows within these sites.
    """
    workload_dir = pathlib.Path(workload_dir)
    model = queries.load_model(workload_dir)
    ts = run_module.load_ts(workload_dir / "reference.trees")
    H, _, df = queries.load_queries(workload_dir)
    if num_sites is not None:
        ts = cut_reference(ts, num_sites)
    m = make_matcher(ts, model)
    if instrument:
        subprocess.run(
            ["callgrind_control", "-i", "on", str(os.getpid())],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for query_id in df.query_id[:num_queries]:
        start, end = queries.query_window(H, query_id)
        h = H[query_id][: ts.num_sites]
        end = min(end, ts.num_sites)
        _, _, parent, _, _ = m.match(h, start, end)
        print(query_id, " ".join(str(p) for p in np.asarray(parent)))


def read_total_ir(path):
    with open(path) as f:
        for line in f:
            if line.startswith(("totals:", "summary:")):
                return int(line.split()[1])
    raise ValueError(f"No totals in {path}")


def out_path(out_dir, workload, num_queries, num_sites):
    sites = "" if num_sites is None else f".cut{num_sites}"
    return pathlib.Path(out_dir) / (
        f"callgrind.{workload}.n{num_queries}{sites}.tsls.out"
    )


def callgrind(data_dir, workload, num_queries, num_sites, out_file):
    cmd = [
        "valgrind",
        "--tool=callgrind",
        "--instr-atstart=no",
        "--collect-atstart=no",
        f"--toggle-collect={FIND_PATH_FUNCTION}",
        f"--callgrind-out-file={out_file}",
        sys.executable,
        __file__,
        "run",
        str(pathlib.Path(data_dir) / workload),
        "-n",
        str(num_queries),
        "--instrument",
    ]
    if num_sites is not None:
        cmd += ["--num-sites", str(num_sites)]
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
    default=None,
    help="Directory of callgrind output files from another build, to compare with",
)
@click.option("-j", "--jobs", type=int, default=3)
def measure(data_dir, workload_names, out_dir, baseline_dir, jobs):
    """
    Count the instructions in find_path for each workload. The callgrind
    output files are kept for callgrind_annotate.
    """
    workloads = WORKLOADS
    if len(workload_names) > 0:
        workloads = [w for w in WORKLOADS if w[0] in workload_names]
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for workload, num_queries, num_sites in workloads:
        path = out_path(out_dir, workload, num_queries, num_sites)
        tasks.append((workload, num_queries, num_sites, path))
    # Run the largest workloads first, so that they don't finish last
    tasks.sort(key=lambda task: task[0] not in ("samples", "ancestors_g1485"))
    with concurrent.futures.ThreadPoolExecutor(jobs) as executor:
        futures = [executor.submit(callgrind, data_dir, *task) for task in tasks]
        for future in futures:
            future.result()

    click.echo("| workload | queries | tsls | baseline | ratio |")
    click.echo("|---|---|---|---|---|")
    for workload, num_queries, num_sites in workloads:
        path = out_path(out_dir, workload, num_queries, num_sites)
        count = read_total_ir(path)
        baseline = ""
        ratio = ""
        if baseline_dir is not None:
            base_path = out_path(baseline_dir, workload, num_queries, num_sites)
            base_count = read_total_ir(base_path)
            baseline = f"{base_count / 1e9:.3f} G"
            ratio = f"{count / base_count:.3f}"
            if base_path.with_suffix(".paths").read_text() != (
                path.with_suffix(".paths").read_text()
            ):
                click.echo(f"WARNING: the builds found different paths for {workload}")
        label = (
            f"{num_queries}"
            if num_sites is None
            else f"{num_queries} ({num_sites} sites)"
        )
        click.echo(
            f"| {workload} | {label} | {count / 1e9:.3f} G | {baseline} | {ratio} |"
        )


if __name__ == "__main__":
    cli()
