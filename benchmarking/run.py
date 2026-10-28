"""
Run the HMM engines over a set of queries and record the results.

Writes one CSV row per (query, engine). Rows are flushed as they are
written, and an existing output file is resumed by skipping the
(query, engine) pairs it already contains.
"""

import csv
import pathlib
import time

import click
import numpy as np
import pandas as pd
import tskit
import tsls
import tszip
from tqdm import tqdm

import engines
import queries
import subargs

RESULT_COLUMNS = [
    "query_id",
    "engine",
    "num_mismatches_param",
    "mismatch_threshold",
    "mu",
    "rho",
    "recombination_rate",
    "status",
    "setup_time",
    "wall_time",
    "cpu_time",
    "num_segments",
    "parents",
    "breakpoints",
    "num_mismatches",
    "log_likelihood",
    "stats",
]


def load_ts(path):
    if str(path).endswith(".tsz"):
        return tszip.load(path)
    return tskit.load(path)


def completed_pairs(out_path):
    """
    Return the set of (query_id, engine) pairs in an existing results file.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(out_path)
    if list(df.columns) != RESULT_COLUMNS:
        raise click.ClickException(
            f"{out_path} has different columns; it was made by an older run.py"
        )
    return set(zip(df.query_id, df.engine))


def parse_recombination_rate(value):
    """
    Return a recombination rate given as a number (per unit of sequence
    length) or the path to a HapMap genetic map, as in tsinfer's config.
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        path = pathlib.Path(value)
        if not path.exists():
            raise click.BadParameter(f"{value} is not a number or a file") from None
        return path


def rho_summary(rho):
    """
    Return the recombination probability to record for a run: rho itself if
    it is constant, or its mean over all but the first site if it is per-site.
    """
    if np.ndim(rho) == 0:
        return rho
    return float(np.mean(rho[1:])) if len(rho) > 1 else 0.0


def summarise_match(ts, node_ids, h, result, mu, rho, *, log_likelihood=True):
    """
    Return the result row fields describing the specified match. The log
    likelihood is computed under the unweighted model with the specified
    parameters, so that it is comparable between engines; it is left empty
    if log_likelihood is False. Parents are reported as original node IDs,
    so that they are comparable between sub-ARGs.
    """
    left, right, parent, matched, stats = result
    if matched is None:
        matched = engines.path_haplotype(ts, left, right, parent)
    non_missing = h != queries.MISSING
    row = {
        "num_segments": len(parent),
        "parents": queries.format_list(node_ids[parent]),
        "breakpoints": queries.format_list(left[1:]),
        "num_mismatches": int(np.sum(h[non_missing] != matched[non_missing])),
        "log_likelihood": "",
        "stats": " ".join(f"{key}={value}" for key, value in stats.items()),
    }
    if log_likelihood:
        row["log_likelihood"] = engines.path_log_likelihood(
            h, len(parent), matched, mu, rho
        )
    return row


@click.command()
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("query_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-e",
    "--engine",
    "engine_names",
    type=click.Choice(list(engines.ENGINES)),
    multiple=True,
    help="Engines to run (default: all, or tsinfer and tsls with a model.json)",
)
@click.option("-o", "--output", type=click.Path(dir_okay=False), required=True)
@click.option(
    "--num-mismatches",
    type=int,
    default=None,
    help="Number of mismatches equivalent to one recombination [default: 3]",
)
@click.option(
    "--mismatch-threshold",
    type=int,
    default=None,
    help="Likelihood threshold for tsinfer, in numbers of mismatches (as sc2ts)",
)
@click.option(
    "--recombination-rate",
    default=None,
    help=(
        "Recombination rate per bp, or the path to a HapMap genetic map, "
        "replacing the model.json rho with per-site probabilities (as tsinfer)"
    ),
)
@click.option(
    "--mismatch",
    type=float,
    default=None,
    help="Mismatch probability, replacing the model.json mu",
)
@click.option("--kind", type=click.Choice(["copy", "recombinant"]), default=None)
@click.option("--limit", type=int, default=None, help="Run at most this many queries")
def main(
    ts_path,
    query_dir,
    engine_names,
    output,
    num_mismatches,
    mismatch_threshold,
    recombination_rate,
    mismatch,
    kind,
    limit,
):
    """
    Run the HMM engines on the queries in QUERY_DIR against the benchmark ARG
    at TS_PATH.

    If QUERY_DIR contains a model.json (as made by tsinfer_data.py), its
    parameters are used instead of --num-mismatches, and the tskit engine,
    which cannot use them, is not available. --recombination-rate and
    --mismatch override its parameters as tsinfer's recombination_rate and
    mismatch would: the rate or genetic map gives per-site recombination
    probabilities (tsls.model.recombination_probabilities).
    """
    model = queries.load_model(query_dir)
    recombination_rate = parse_recombination_rate(recombination_rate)
    engine_options = {}
    if model is None:
        if recombination_rate is not None or mismatch is not None:
            raise click.UsageError(
                "--recombination-rate and --mismatch need a model.json"
            )
        if len(engine_names) == 0:
            engine_names = list(engines.ENGINES)
        if num_mismatches is None:
            num_mismatches = 3
        mu, rho = engines.solve_num_mismatches(num_mismatches)
    else:
        if len(engine_names) == 0:
            engine_names = ["tsinfer", "tsls"]
        if "tskit" in engine_names:
            raise click.UsageError("The tskit engine does not support model.json")
        if num_mismatches is not None:
            raise click.UsageError("--num-mismatches cannot be used with model.json")
        mu = model["mu"] if mismatch is None else mismatch
        rho = model["rho"]
        if recombination_rate is None:
            recombination_rate = parse_recombination_rate(
                model.get("recombination_rate")
            )
        engine_options = {
            "weight_by_n": model["weight_by_n"],
            "alleles": model["alleles"],
        }
    ts = load_ts(ts_path)
    node_ids = subargs.load_node_ids(ts_path, ts)
    if recombination_rate is not None:
        rho = tsls.model.recombination_probabilities(
            recombination_rate, ts.sites_position
        )
        click.echo(
            f"Recombination probabilities from {recombination_rate}: "
            f"min={np.min(rho[1:]):.3g} mean={np.mean(rho[1:]):.3g} "
            f"max={np.max(rho[1:]):.3g}"
        )
    H, positions, df = queries.load_queries(query_dir)
    if not np.array_equal(positions, ts.sites_position):
        raise ValueError("Query sites do not match the tree sequence")
    if kind is not None:
        df = df[df.kind == kind]
    if limit is not None:
        df = df.head(limit)

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pairs(out_path)
    write_header = len(done) == 0

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for name in engine_names:
            todo = [qid for qid in df.query_id if (qid, name) not in done]
            if len(todo) == 0:
                continue
            before = time.perf_counter()
            options = {}
            if name in ("tsinfer", "tsls"):
                options["mismatch_threshold"] = mismatch_threshold
                options.update(engine_options)
            engine = engines.ENGINES[name](ts, mu=mu, rho=rho, **options)
            setup_time = time.perf_counter() - before
            click.echo(f"Set up {name} in {setup_time:.2f}s")
            for query_id in tqdm(todo, desc=name):
                h = H[query_id]
                start, end = queries.query_window(H, query_id)
                row = {
                    "query_id": query_id,
                    "engine": name,
                    "num_mismatches_param": num_mismatches,
                    "mismatch_threshold": options.get("mismatch_threshold"),
                    "mu": engine.mu,
                    "rho": rho_summary(engine.rho),
                    "recombination_rate": (
                        "" if recombination_rate is None else recombination_rate
                    ),
                    "setup_time": setup_time,
                }
                wall_before = time.perf_counter()
                cpu_before = time.thread_time()
                try:
                    if name == "tskit":
                        result = engine.match(h)
                    else:
                        result = engine.match(h, start, end)
                except Exception as e:  # noqa: BLE001 (record any failure)
                    row["status"] = f"{type(e).__name__}: {e}"
                else:
                    row["status"] = "ok"
                row["wall_time"] = time.perf_counter() - wall_before
                row["cpu_time"] = time.thread_time() - cpu_before
                if row["status"] == "ok":
                    row.update(
                        summarise_match(
                            ts,
                            node_ids,
                            h,
                            result,
                            mu,
                            rho,
                            log_likelihood=model is None,
                        )
                    )
                writer.writerow(row)
                f.flush()


if __name__ == "__main__":
    main()
