# tsls benchmarks

Benchmarks for tsls's Viterbi algorithm on two kinds of workload:

- **sc2ts:** query haplotypes matched against the sc2ts SARS-CoV-2 ARG and
  sub-ARGs of it, with sc2ts's model.
- **tsinfer:** the ancestor and sample matching of a tsinfer 1.0 inference
  of 1000 Genomes chr20p, with tsinfer's model, replayed one HMM call at a
  time. The paths tsinfer found are the ground truth.

tsls replaced the Li and Stephens engines in tsinfer ("tsinfer-hmm") and
tskit ("tskit-hmm"), which were developed and compared in this directory
until October 2026. The final comparison with tsinfer-hmm is
[below](#final-comparison-with-tsinfer-hmm-october-2026). The engines, the
comparison scripts and the full record of how tsls got there (the previous
version of this README) are in the git history, at commit `1d70724`.

## Setup

```
uv sync --group test
uv run pytest
```

Input data goes in `data/`, which is not tracked by git. Results go in
`results/` and callgrind output in `callgrind/`, which aren't tracked
either.

## Tools

- `bench.py`: the main benchmark. It matches the queries of each workload
  (listed in `bench.WORKLOADS`) several times, keeps the best time for each,
  and prints a markdown table of the time, the HMM's memory and traceback
  size, and for the tsinfer workloads the number of queries for which tsls
  found tsinfer's path. Run it on an otherwise idle machine, pinned to one
  core:

  ```
  taskset -c 3 uv run python bench.py
  taskset -c 3 uv run python bench.py --quick
  ```

- `run.py`: matches a set of queries, writing one CSV row per query with
  the path, mismatches, times and statistics. `tsinfer_summary.py`
  summarises results on the tsinfer workloads against the paths tsinfer
  found.
- `ab.py`: compares builds of the extension (each a directory holding
  `_tsls*.so`, put first on `PYTHONPATH`), with paired runs of
  `profile_samples.py` (chr20p samples) or `profile_sc2ts.py` (sc2ts
  queries), pinned to one core, keeping the best time per query, and
  checks that the paths are the same.
- `callgrind.py measure`: counts the instructions in `find_path` on the
  tsinfer workloads, with setup run uninstrumented, optionally against the
  counts of another build (`--baseline-dir`).

`matcher.Matcher` sets tsls up for a workload's model (see its docstring).

## sc2ts workloads

### 1. Prepare the ARG

The sc2ts ARG is `data/sc2ts_viridian_v2.2.trees.tsz`.

```
uv run python prepare.py data/sc2ts_viridian_v2.2.trees.tsz data/sc2ts_bench.trees
```

This deletes the 75 sites with non-ACGT states (deletions coded as `-`),
and adds the vestigial root edge 0 → 1. (Both were needed by the engines
tsls was compared with, and are kept so that results stay comparable.) The
result has 3.4M nodes, 510 trees and 29,565 sites.

### 2. Make sub-ARGs

```
uv run python subargs.py data/sc2ts_bench.trees data/subargs
```

This writes nested sub-ARGs for measuring how tsls scales with ARG size,
by default with 10k, 30k, 100k, 300k and 1M samples. Each sub-ARG is the
ARG "as of" an earlier date: all nodes at least as old as a time cutoff.

- Parts of the ARG that are no longer ancestral to any remaining sample
  are pruned. These are ancestors that sc2ts only added later, along with
  their descendants.
- All sites are kept, so any query set can be run against any sub-ARG.
- Node IDs change, so each sub-ARG has a `.node_ids.npy` file giving each
  node's ID in the full ARG. `queries.py` and `run.py` report node IDs
  (`nodes`, `parents`) as full-ARG IDs, so they are comparable across
  sub-ARGs.

### 3. Generate queries

To measure scaling with a fixed set of queries, generate them from the
smallest sub-ARG. Its samples are in every larger sub-ARG, so every query
is valid for all of them:

```
uv run python queries.py data/subargs/sc2ts_bench_n10000.trees data/queries_scaling \
    --num-nodes 10 --divergence 0,1,2,5,10 --num-recombinants 10
```

Queries are written to `queries.npz`, which holds the haplotype matrix `H`
(int8, A=0 C=1 G=2 T=3, -1 missing) and the site positions. `queries.csv`
holds one row of metadata per query. There are two kinds of query:

- `copy`: the haplotype of a random sample node with `k` random mutations
  added. The source node stays in the ARG, so a k=0 query has an exact
  match.
- `recombinant`: a mosaic of two random sample nodes, A|B (one breakpoint)
  or A|B|A (two breakpoints), with k mutations added. Each segment has at
  least 2 sites at which the parents differ.

Metadata columns: `nodes` (the source or parents), `k`, `mutated_sites`,
`breakpoints` (site indexes), `breakpoint_intervals`, `parent_diff` and
`segment_diffs`. A breakpoint `b` in the interval `(lo, hi]` gives the same
haplotype as any other breakpoint in that interval.

### 4. Model

The model is sc2ts's, with `--num-mismatches` (default 3) mismatches
costing about the same as one recombination (`matcher.solve_num_mismatches`,
copied from sc2ts). sc2ts gives the odds of switching to a given node,
rho, and tsls is given `r = tsls.recombination_probability(rho, n)`, which
has the same Viterbi paths. sc2ts matches each sample first with a
threshold of k = 0 and then k = 1 mismatches, and reruns those that fail
exactly. With `--mismatch-threshold k`, tsls prunes at the exact threshold
for k mismatches (`tsls.path_threshold`). `bench.py`'s `_k1` workloads use
k = 1, and the others are exact.

```
uv run python run.py data/subargs/sc2ts_bench_n100000.trees data/queries_scaling \
    -o results/sc2ts_n100k.csv
```

## tsinfer workloads: 1000 Genomes chr20p

### Data

A colleague ran tsinfer at commit `cb5017e` on the 1000 Genomes high
coverage chr20p (3,202 individuals, 580,909 inference sites), with the
parameters in `full_inference.toml`: `rho = 0.01` and `mu = 1e-20` for both
ancestors and samples, no path compression, and tsinfer's defaults of
`weight_by_n=True` and no likelihood threshold. The data are in
`data/tgp_chr20p/` (see `data/tgp_chr20p/README.md`):

- `tgp_chr20p_inferred_ancestors.zarr`: the 432,851 ancestors.
- `tgp_chr20p_inferred_ancestors.trees`: the ancestors tree sequence
  (433k nodes, 4.2M edges).
- `tgp_chr20p_samples_masked.zarr`: the sample genotypes.
- `tgp_chr20p_inferred_samples_raw.trees`: the result of matching the
  6,404 sample haplotypes, before post-processing.
- `partial_ancestor_match/`: the tree sequence before group 84
  (`before.trees`), the group's ancestors and the result (`after.trees`).

### 1. Compute the match groups

The tree sequences do not record which nodes were matched in which group,
so we recompute the groups with tsinfer's own code. This needs a tsinfer
checkout at `../../tsinfer`, and runs in its environment so that tsinfer is
not a dependency of the benchmarks:

```
uv run --project ../../tsinfer python tsinfer_groups.py data/tgp_chr20p
```

This writes `data/tgp_chr20p/match_jobs.csv`: 439,255 jobs in 1,487 groups
(groups 0–1485 for the ancestors, and group 1486 for the samples).

### 2. Make workloads

```
uv run python tsinfer_data.py ancestors data/tgp_chr20p 1485 data/tgp_chr20p/ancestors_g1485 --num-queries 500
uv run python tsinfer_data.py samples data/tgp_chr20p data/tgp_chr20p/samples --num-queries 200
```

- `ancestors GROUP`: the ancestors of one group (all of them by default),
  matched against the tree sequence before the group: the ancestors tree
  sequence cut at the group's first node.
- `samples`: a random subset of the sample haplotypes, matched against the
  full ancestors tree sequence.

Each workload directory holds `reference.trees`, the queries, a
`model.json` with tsinfer's parameters, and `tsinfer_group.json` with the
time and memory tsinfer recorded for the whole group. Haplotypes are
encoded as in tsinfer (0 ancestral, 1 derived, -1 missing), and each query
is matched over its window of non-missing sites. `queries.csv` has the path
tsinfer found (`truth_parents`, `truth_breakpoints`).

### 3. Run and summarise

```
uv run python run.py data/tgp_chr20p/ancestors_g1485/reference.trees \
    data/tgp_chr20p/ancestors_g1485 -o results/tgp_chr20p_ancestors_g1485.csv
uv run python tsinfer_summary.py \
    results/tgp_chr20p_ancestors_g1485.csv data/tgp_chr20p/ancestors_g1485
```

### Recombination maps

tsinfer can set a source's `recombination_rate` to a rate per bp or a
HapMap genetic map, giving per-site recombination probabilities with
Haldane's map function (`tsls.recombination_probabilities`).
`run.py --recombination-rate RATE_OR_MAP` (with an optional `--mismatch`)
reruns a workload with these probabilities in place of the `model.json`
rho. To get the HapMap II GRCh38 chr20 map:

```
uv run --with stdpopsim python genetic_map.py data/genetic_maps/chr20_HapMapII_GRCh38.txt
```

`bench.py`'s `samples_map` workloads use this map.

## Final comparison with tsinfer-hmm (October 2026)

The last comparison of tsls (`e4d99c8`) with tsinfer-hmm, before
tsinfer-hmm was removed. It was run with `paired.py` from commit `1d70724`,
with the tsls build installed in place of that commit's: each query is
matched three times with each engine, alternating between them, and the
best time for each engine is kept. One process, pinned to one core of a
4-core VM with 15 GB of RAM (other work ran on the other cores). The
workloads are those of `bench.py`; the times include building the HMM for
each query. Mean times are per query, and the ratio is total tsls time
over total tsinfer-hmm time. Memory is the largest `total_memory` either
engine reported for a query (mostly the Viterbi traceback), not the
process's memory. Paths are classed against tsinfer-hmm's: the same path,
a different path with the same model log likelihood (a tie), or a better
or worse one.

| workload | queries | tsinfer-hmm mean time | tsls mean time | time ratio | tsinfer-hmm memory (MiB) | tsls memory (MiB) | paths same/tie/better/worse |
|---|---|---|---|---|---|---|---|
| ancestors, group 84 | 50 | 4.2 ms | 2.8 ms | 0.703 | 64 | 4 | 49/1/0/0 |
| ancestors, group 1000 | 50 | 11.2 ms | 6.4 ms | 0.572 | 64 | 4 | 48/2/0/0 |
| ancestors, group 1483 | 100 | 49.8 ms | 26.8 ms | 0.539 | 64 | 9 | 100/0/0/0 |
| ancestors, group 1484 | 50 | 123 ms | 58.8 ms | 0.477 | 64 | 10 | 49/1/0/0 |
| ancestors, group 1485 | 100 | 144 ms | 81.8 ms | 0.566 | 64 | 11 | 99/1/0/0 |
| samples, constant r | 6 | 4.4 s | 1.39 s | 0.315 | 128 | 13 | 6/0/0/0 |
| samples, map, mu = 1e-20 | 6 | 14.2 s | 2.28 s | 0.161 | 384 | 53 | 6/0/0/0 |
| samples, map, mu = 1e-8 | 6 | 13 s | 1.98 s | 0.152 | 448 | 53 | 1/5/0/0 |
| sc2ts 100k, k = 1 | 10 | 3.63 s | 19 ms | 0.005 | 640 | 4 | 4/2/3/1 |

The sc2ts 100k (exact) and 300k workloads and the full sc2ts ARG are still
running, and will be added when they finish.

- On the tsinfer workloads, tsls takes 0.48-0.70 of tsinfer-hmm's time on
  ancestors, 0.32 on samples with constant recombination, and 0.15-0.16
  under the HapMap II recombination map, with 6-16 times less memory. The
  paths are tsinfer-hmm's apart from a few ties.
- On sc2ts, tsls is about 190 times faster on the 100k sub-ARG with the
  k = 1 threshold (19 ms against 3.6 s per query), with 4 MiB of memory
  against 640 MiB. With a threshold, neither engine is guaranteed to find
  the optimal path when it costs more than the threshold allows:
  tsinfer-hmm floors the normalised values, so nodes with different
  numbers of mismatches can tie (its path was worse for 3 queries), and
  tsls prunes them (its path was worse for 1). sc2ts reruns such queries
  without a threshold.
