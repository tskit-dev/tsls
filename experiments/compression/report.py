"""
Analyse the matrices written by capture.py and write results/report.md with
tables and plots.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tskit
from matplotlib import pyplot as plt

import analysis
import tsls

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def summarise_matrix(row, num_sites, parsimony_sites):
    """
    Return the (summary dict, list measures, decoded measures, value histogram)
    tuple for one captured matrix.
    """
    ts = tskit.load(RESULTS / row.trees)
    cm = tsls.load_compressed_matrix(RESULTS / row.matrix, ts)
    num_alleles = getattr(row, "num_alleles", np.nan)
    if np.isnan(num_alleles):
        # Captured before num_alleles was recorded: sc2ts and msprime use ACGT
        num_alleles = 4
    site_map = analysis.SiteMap(
        row.algorithm,
        rho=row.rho,
        mu=row.mu,
        n=row.num_nodes if row.weight_by_n else 1,
        num_alleles=num_alleles,
    )
    lists = analysis.list_measures(cm, site_map)
    decoded = analysis.decoded_measures(
        cm, num_sites=num_sites, parsimony_sites=parsimony_sites, site_map=site_map
    )
    nonempty = lists[lists.stored > 0]
    # Encoding measures compare each site with the previous non-empty one
    carried = nonempty.iloc[1:]
    churn = carried.entered + carried.left
    first = nonempty.stored.iloc[0]
    summary = {
        "case": row.case,
        "arg": row.arg,
        "label": row.label,
        "query": row.query,
        "algorithm": row.algorithm,
        "num_nodes": row.num_nodes,
        "num_samples": row.num_samples,
        "num_segments": row.num_segments,
        "num_mismatches": row.num_mismatches,
        "stored": nonempty.stored.mean(),
        "change_points": decoded.change_points.mean(),
        "redundant": decoded.redundant.mean(),
        "distinct": nonempty.distinct.mean(),
        "median_clade_size": decoded.median_clade_size.mean(),
        "mean_clade_size": decoded.mean_clade_size.mean(),
        "samples_only": decoded.samples_only.mean()
        if "samples_only" in decoded
        else np.nan,
        "jaccard": nonempty.jaccard.mean(),
        "delta_per_site": (nonempty.added + nonempty.removed).mean(),
        "mutation_nodes": nonempty.mutation_nodes.mean(),
        "churn": churn.mean(),
        "changed": carried.changed.mean(),
        "changed_tol": carried.changed_tol.mean(),
        "affine_residual": carried.affine_residual.mean(),
        "affine_fit": carried.affine_fit.mean(),
        "affine_work": (churn + carried.affine_residual).mean(),
        "total_stored": nonempty.stored.sum(),
        "delta_values": first + (churn + carried.changed).sum(),
        "affine_delta_values": first
        + (churn + carried.affine_residual).sum()
        + 2 * len(nonempty),
        # The same, on the minimal (change point) lists at the decoded sites
        "cp_churn": (decoded.cp_entered + decoded.cp_left).mean(),
        "cp_changed": decoded.cp_changed.mean(),
        "cp_affine_residual": decoded.cp_affine_residual.mean(),
    }
    summary["cp_affine_work"] = summary["cp_churn"] + summary["cp_affine_residual"]
    for digits in [1, 2, 4]:
        summary[f"quantised_{digits}"] = decoded[f"quantised_{digits}"].mean()
    if row.algorithm == "viterbi":
        summary["recombination_flags"] = nonempty.recombination_flags.mean()
        summary["new_list_fraction"] = nonempty.new_list.mean()
        # Entries stored in the traceback by the engine (lists are reused only
        # if identical) against a delta encoding of the changes
        summary["traceback_entries"] = nonempty.stored[nonempty.new_list].sum()
        summary["delta_entries"] = (nonempty.added + nonempty.removed).sum() + (
            nonempty.stored.iloc[0]
        )
        # The traceback stores nodes and flags, not values
        summary["flag_changed"] = carried.flag_changed.mean()
        summary["delta_traceback"] = first + (churn + carried.flag_changed).sum()
    return summary, lists, decoded, analysis.value_histogram(cm, top=6)


def plot_along_genome(lists, decoded, title, path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    nonempty = lists[lists.stored > 0]
    ax.plot(nonempty.site, nonempty.stored, label="stored", linewidth=0.8)
    if "affine_residual" in nonempty:
        work = nonempty.entered + nonempty.left + nonempty.affine_residual
        ax.plot(
            nonempty.site,
            work.rolling(25, min_periods=1).mean(),
            label="entered + left + affine residual (rolling mean)",
            linewidth=0.8,
        )
    ax.plot(
        decoded.site,
        decoded.change_points,
        "o",
        markersize=3,
        label="minimum (change points)",
    )
    if "quantised_1" in decoded:
        ax.plot(
            decoded.site,
            decoded.quantised_1,
            "x",
            markersize=3,
            label="change points, log10 values rounded",
        )
    ax.set_yscale("symlog")
    ax.set_xlabel("Site")
    ax.set_ylabel("Nodes")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_encodings(summary, path):
    viterbi = summary[summary.algorithm == "viterbi"]
    labels = [f"{r.arg}\n{r.label}" for r in viterbi.itertuples()]
    x = np.arange(len(viterbi))
    fig, ax = plt.subplots(figsize=(max(8, len(viterbi) * 0.8), 4))
    width = 0.35
    ax.bar(x - width / 2, viterbi.traceback_entries, width, label="engine traceback")
    ax.bar(x + width / 2, viterbi.delta_entries, width, label="delta encoding")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.set_ylabel("Stored (node, value) entries over all sites")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_work(summary, path):
    """
    Plot the per-site work proxy of the current engine (the stored list)
    against an affine engine (entries entering or leaving the list, and
    values not following the global map), for each matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, algorithm in zip(axes, ["viterbi", "forward"]):
        d = summary[summary.algorithm == algorithm]
        labels = [f"{r.arg}\n{r.label}" for r in d.itertuples()]
        x = np.arange(len(d))
        width = 0.35
        ax.bar(x - width / 2, d.stored, width, label="stored (current engine)")
        ax.bar(x + width / 2, d.affine_work, width, label="churn + affine residual")
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=90, fontsize=5)
        ax.set_ylabel("Entries per site")
        ax.set_title(algorithm)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def encodings_table(summary, algorithm):
    """
    Return the markdown table of per-site encoding measures and encoding
    sizes for the specified algorithm, with one row per ARG (means over its
    queries).
    """
    d = summary[summary.algorithm == algorithm].copy()
    d["values_delta_saving"] = d.total_stored / d.delta_values
    d["values_affine_saving"] = d.total_stored / d.affine_delta_values
    d["work_saving"] = d.stored / d.affine_work
    columns = [
        "stored",
        "churn",
        "changed",
        "changed_tol",
        "affine_residual",
        "affine_fit",
        "values_delta_saving",
        "values_affine_saving",
        "work_saving",
    ]
    if algorithm == "viterbi":
        d["traceback_delta_saving"] = d.traceback_entries / d.delta_traceback
        columns += ["flag_changed", "traceback_delta_saving"]
    return grouped_table(d, columns)


def minimal_table(summary, algorithm):
    """
    Return the markdown table of encoding measures on the minimal (change
    point) lists, at the decoded sites.
    """
    d = summary[summary.algorithm == algorithm].copy()
    d["cp_delta_work_saving"] = d.change_points / (d.cp_churn + d.cp_changed)
    d["cp_affine_work_saving"] = d.change_points / d.cp_affine_work
    columns = [
        "stored",
        "change_points",
        "cp_churn",
        "cp_changed",
        "cp_affine_residual",
        "cp_delta_work_saving",
        "cp_affine_work_saving",
    ]
    return grouped_table(d, columns)


def grouped_table(d, columns):
    table = d.groupby(["case", "arg"], sort=False)[columns].mean().reset_index()
    counts = d.groupby(["case", "arg"], sort=False).size().to_numpy()
    table.insert(2, "queries", counts)
    return table.to_markdown(index=False, floatfmt=".3g")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--num-sites", type=int, default=200, help="Sites to decode per matrix"
    )
    parser.add_argument(
        "--parsimony-sites",
        type=int,
        default=10,
        help="Sites for the samples-only parsimony",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Only analyse these cases (comma separated), for quick iteration",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    if "case" not in manifest:
        manifest["case"] = manifest.arg.str.split("_").str[0]
    if args.cases is not None:
        manifest = manifest[manifest.case.isin(args.cases.split(","))]
    rows = []
    plots = RESULTS / "plots"
    plots.mkdir(exist_ok=True)
    histograms = []
    for row in manifest.itertuples():
        summary, lists, decoded, histogram = summarise_matrix(
            row, args.num_sites, args.parsimony_sites
        )
        rows.append(summary)
        name = f"{row.arg}_{row.query}_{row.algorithm}"
        lists.to_csv(RESULTS / f"{name}_lists.csv", index=False)
        decoded.to_csv(RESULTS / f"{name}_decoded.csv", index=False)
        plot_along_genome(
            lists,
            decoded,
            f"{row.arg} {row.label} {row.algorithm}",
            plots / f"{name}.png",
        )
        if row.algorithm == "viterbi":
            histogram.insert(0, "case", f"{row.arg} {row.label}")
            histograms.append(histogram)
        print(f"Analysed {name}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    plot_encodings(summary, plots / "encodings.png")
    plot_work(summary, plots / "affine_work.png")

    viterbi = summary[summary.algorithm == "viterbi"].copy()
    viterbi["delta_saving"] = viterbi.traceback_entries / viterbi.delta_entries
    forward = summary[summary.algorithm == "forward"]
    lines = [
        "# Likelihood compression report",
        "",
        "Per-site means. `stored` is the engine's list; `change_points` is the",
        "minimum for the nearest-ancestor encoding; `samples_only` is the",
        "(approximate) minimum if only samples had to be represented.",
        "",
        "## Viterbi",
        "",
        viterbi[
            [
                "arg",
                "label",
                "num_segments",
                "num_mismatches",
                "stored",
                "change_points",
                "redundant",
                "distinct",
                "samples_only",
                "median_clade_size",
                "jaccard",
                "delta_per_site",
                "recombination_flags",
                "new_list_fraction",
            ]
        ].to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Viterbi traceback memory: engine against a delta encoding",
        "",
        "Entries stored over all sites by the engine (which reuses a list only",
        "if it is identical to the previous site's), against a delta encoding",
        "that stores the first list and then only the entries added and removed.",
        "",
        viterbi[
            ["arg", "label", "traceback_entries", "delta_entries", "delta_saving"]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Most common Viterbi values (fraction of stored entries)",
        "",
        pd.concat(histograms).to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Forward",
        "",
        forward[
            [
                "arg",
                "label",
                "stored",
                "change_points",
                "distinct",
                "samples_only",
                "quantised_1",
                "quantised_2",
                "quantised_4",
                "jaccard",
                "delta_per_site",
            ]
        ].to_markdown(index=False, floatfmt=".3g"),
        "",
        "`quantised_d`: change points after rounding log10 values to 1/d.",
        "",
        "## Encodings",
        "",
        "Per-site means, comparing each site's stored list with the previous",
        "site's, then averaged over the queries of each ARG. `churn` is the",
        "number of nodes entering or leaving the list. `changed` counts nodes",
        "in both lists whose value changed (`changed_tol`: by a relative",
        "difference of more than 1e-9). `affine_residual` counts those whose",
        "value is not predicted by the site's global map (Viterbi:",
        "max(v stay, rho/n) e / c; forward: a v + b, with b fitted), and",
        "`affine_fit` is the fraction that are.",
        "",
        "Savings are ratios of entries stored over all sites. For values: the",
        "full lists against a delta encoding (first list, then churn and",
        "changed values), and against an affine + delta encoding (churn and",
        "affine residuals, plus two map parameters per site). `work_saving`",
        "is the list size against churn + affine residual per site: the",
        "values an affine engine would update explicitly. For the Viterbi",
        "traceback, which stores nodes and flags but not values, the engine's",
        "lists against a delta encoding of nodes and flags.",
        "",
        "### Viterbi",
        "",
        encodings_table(summary, "viterbi"),
        "",
        "### Forward",
        "",
        encodings_table(summary, "forward"),
        "",
        "### On the minimal lists",
        "",
        "The engine stores its lists before removing values equal to the",
        "parent's, and gives nodes their own values when edges change, so",
        "the stored lists churn more than the values do. These are the same",
        "measures on the change points (the minimal lists), comparing each",
        "decoded site with the previous one: the entries an engine that kept",
        "only change points would update with a delta encoding (churn +",
        "changed) or an affine one (churn + affine residual), and the",
        "corresponding savings against the change points.",
        "",
        "#### Viterbi",
        "",
        minimal_table(summary, "viterbi"),
        "",
        "#### Forward",
        "",
        minimal_table(summary, "forward"),
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
