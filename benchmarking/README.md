# HMM benchmarks

Benchmarks for the Li and Stephens HMM engines in tsls:

- `tsinfer`: tsinfer-hmm's `AncestorMatcher`
- `tskit`: tskit-hmm's `LsHmm` Viterbi

Each engine runs Viterbi matching over a set of query haplotypes with
known ground truth, and the results are recorded so the engines can be
characterised across a range of queries. There are two targets: the
sc2ts SARS-CoV-2 ARG, and the ancestor and sample matching phases of a
tsinfer 1.0 inference of 1000 Genomes chr20p (see
[below](#tsinfer-10-1000-genomes-chr20p)).

## Setup

```
uv sync --group test
uv run pytest
```

Input data goes in `data/`, which is not tracked by git. The sc2ts ARG
is `data/sc2ts_viridian_v2.2.trees.tsz`.

## Workflow

### 1. Prepare the ARG

```
uv run python prepare.py data/sc2ts_viridian_v2.2.trees.tsz data/sc2ts_bench.trees
```

This step makes the ARG usable by both engines:

- It deletes the 75 sites with non-ACGT states (deletions coded as `-`),
  because tskit-hmm only supports ACGT.
- It adds the vestigial root edge 0 → 1, as sc2ts does before matching.
  tsinfer-hmm then uses the ARG as is, and node IDs are the same for
  both engines.

### 1b. Make sub-ARGs (optional)

```
uv run python subargs.py data/sc2ts_bench.trees data/subargs
```

This writes a series of nested sub-ARGs for measuring how the engines
scale with ARG size. The default sizes are 10k, 30k, 100k, 300k and 1M
samples. Each sub-ARG is the ARG "as of" an earlier date: all nodes at
least as old as a time cutoff.

- Parts of the ARG that are no longer ancestral to any remaining sample
  are pruned. These are ancestors that sc2ts only added later, along
  with their descendants. tskit-hmm also requires every node in a tree
  to have a sample below it.
- All sites are kept, so any query set can be run against any sub-ARG.
- Node IDs change, so each sub-ARG has a `.node_ids.npy` file giving
  each node's ID in the full ARG. `queries.py` and `run.py` report node
  IDs (`nodes`, `parents`) as full-ARG IDs, so they are comparable
  across sub-ARGs.

To measure scaling with a fixed set of queries, generate the queries
from the smallest sub-ARG. Its samples are in every larger sub-ARG, so
every query is valid for all of them:

```
uv run python queries.py data/subargs/sc2ts_bench_n10000.trees data/queries_scaling \
    --num-nodes 10 --divergence 0,1,2,5,10 --num-recombinants 10
for n in 10000 30000 100000 300000 1000000; do
    uv run python run.py data/subargs/sc2ts_bench_n$n.trees data/queries_scaling \
        -o results/scaling_n$n.csv
done
```

### 2. Generate queries

```
uv run python queries.py data/sc2ts_bench.trees data/queries
```

Queries are written to `queries.npz`, which holds the haplotype matrix
`H` (int8, A=0 C=1 G=2 T=3, -1 missing) and the site positions.
`queries.csv` holds one row of metadata per query. There are two kinds
of query:

- `copy`: the haplotype of a random sample node with `k` random
  mutations added. By default there are 50 nodes and k = 0..10. The
  source node stays in the ARG, so a k=0 query has an exact match.
- `recombinant`: a mosaic of two random sample nodes, A|B (one
  breakpoint) or A|B|A (two breakpoints), with k = 0, 2 or 5 mutations
  added. Each segment has at least 2 sites at which the parents differ.

Metadata columns: `nodes` (the source or parents), `k`, `mutated_sites`,
`breakpoints` (site indexes), `breakpoint_intervals`, `parent_diff` and
`segment_diffs`. A breakpoint `b` in the interval `(lo, hi]` gives the
same haplotype as any other breakpoint in that interval. Lists are
space-separated.

### 3. Run

```
uv run python run.py data/sc2ts_bench.trees data/queries -o results/sc2ts.csv
```

Options:

- `-e/--engine` (repeatable) runs a subset of the engines.
- `--kind` and `--limit` select a subset of the queries.
- `--num-mismatches` sets the model parameters.
- `--mismatch-threshold` sets tsinfer's likelihood threshold, as sc2ts does.

Rows are written and flushed one at a time. Rerunning the same command
resumes, skipping any (query, engine) pairs already in the output, so
use a separate output file for each parameter set.

Each row of the results has these columns:

- `status`: `ok` or the error message
- `setup_time`, `wall_time`, `cpu_time`: setup and per-query times in seconds
- `num_segments`, `parents`, `breakpoints`: the path (breakpoints are
  genome coordinates)
- `num_mismatches`
- `log_likelihood`: the path log likelihood under the shared model (see below)
- `stats`: engine-specific statistics
  - tsinfer: `total_memory`, `mean_traceback_size`
  - tskit: `total_transitions`, `max_transitions`, `optimum_log_likelihood`

## Model parameters

Both engines use the same Viterbi recursion. At each site, staying on a
node has likelihood `p_last * (1 - rho + rho / n)` and switching has
likelihood `rho / n`. The engines differ in what `n` is:

- tsinfer-hmm, run with `weight_by_n=False` as in sc2ts, uses `n = 1`.
- tskit-hmm uses `n = num_samples`.

Likelihoods are renormalised at every site, so only the ratio of
switching to staying matters. A tsinfer `rho` of `r` is therefore
equivalent to a tskit `rho` of `r / (r + (1 - r) / n)`. With this
conversion, both engines optimise the same objective.

`mu` and `r` are chosen as sc2ts does (`solve_num_mismatches`), so that
`--num-mismatches` mismatches (default 3) cost about the same as one
recombination.

`log_likelihood` is computed in the same units for both engines:
`(segments - 1) log r + mismatches log mu + matches log(1 - 3 mu)`.
For tskit, `optimum_log_likelihood` is the optimum found by the forward
pass, derived from the normalisation factors. If it differs from
`log_likelihood`, the traceback returned a suboptimal path.

## Caveats

- tsinfer-hmm can copy from any node; tskit-hmm can copy only from
  samples. Their paths are not directly comparable, but `parents` uses
  the same node IDs for both.
- tskit-hmm gives suboptimal results when samples are internal nodes,
  which they often are in sc2ts. Both its forward pass optimum and its
  traceback can be wrong. See the xfail test
  `test_agree_internal_samples`.
- Both packages build with `NDEBUG` undefined, so C assertions are
  active during timing.
- Matching against the full sc2ts ARG is slow; see the findings below.
  Use `--limit` and the sub-ARGs to try things out.

## Findings so far (October 2026)

These are partial, exploratory results. The runs shared a 4-core machine
with other jobs, so treat absolute times as rough. The scaling queries
are the 110 queries from the sub-ARG section (10 nodes, k = 0, 1, 2, 5,
10, and 10 recombinants of each type with k = 0, 2, 5), run with
`--num-mismatches 3`.

| engine | samples | queries run | mean time | max time | mean memory | mean traceback size |
|---|---|---|---|---|---|---|
| tsinfer | 10k | 110 | 0.56s | 1.1s | 64 MB | 908 |
| tsinfer | 30k | 110 | 2.8s | 5.1s | 204 MB | 2,813 |
| tsinfer | 100k | 52 | 7.1s | 12.6s | 560 MB | 5,106 |
| tsinfer | 3.1M (full) | 4 | 130–157s | 201s | 3.2–6.3 GB | 22k–45k |
| tskit | 10k | 26 | 46s | 69s | | |
| tskit | 3.1M (full) | 0 | none finished in 15 minutes | | | |

### tsinfer-hmm

- Time grows faster than linearly with ARG size: about 5× from 10k to
  30k samples, and 2.5× from 30k to 100k. Memory and traceback size grow
  roughly in step with time.
- Time hardly depends on the query. Divergence (k = 0–10) makes no
  difference, and recombinants take only about 10% longer than copies.
  The cost is set by the ARG, not by the query.
- On the full ARG, tsinfer 0.5.1 (as used by sc2ts) chooses the same
  parents, with the same traceback size and memory, and takes 175–180s
  per query. The cost is therefore inherent to the algorithm, not
  something introduced by the multiallelic extension.
- An sc2ts-style likelihood threshold of 5 mismatches reduced time only
  modestly (to about 125–140s per query) and did not reduce memory.
- The paths are as expected. Copy queries give one segment with exactly
  k mismatches. Recombinants average 1.0–1.5 more mismatches than k and
  mostly a single segment: with `--num-mismatches 3`, a segment with
  only 2–3 informative sites is cheaper to explain with mismatches than
  with a recombination.

### tskit-hmm

- At 10k samples it is about 80× slower than tsinfer, and on the full
  ARG it did not finish a single query.
- It gives suboptimal results on sc2ts ARGs, where about 11% of samples
  are internal nodes. At 10k samples it was worse than tsinfer on 10 of
  the 26 queries compared, always by exactly one mismatch, including
  exact copies (k = 0). It was never better. There are two separate
  failures:
  - Forward pass. When the source is an internal sample (for example
    query 0), the optimum implied by the normalisation factors is
    already one mismatch worse than an exact copy that is available.
  - Traceback. When the source is a leaf (for example query 20), the
    forward pass optimum is correct, but the traceback returns a path
    one mismatch worse. This happened on 5 of the 26 queries.

  When all samples are leaves, the two engines agree exactly. The
  failures are reproduced by `test_agree_internal_samples`.
- It aborts with a C assertion (`lib/haplotype_matching.c:349`) if a tree
  contains a node with no samples below it. `subargs.py` prunes such
  ancestry.

### Implications

- tskit-hmm is not usable as a reference on sc2ts-style ARGs until the
  internal-sample bugs are fixed.
- For tsinfer-hmm, the main targets are memory and traceback size on
  large ARGs.
- For clean timings, run sub-ARGs up to about 300k samples one at a time.
  Restrict tskit to the smallest sub-ARGs with `--limit`.

## tsinfer 1.0: 1000 Genomes chr20p

These benchmarks replay the HMM calls of a real tsinfer 1.0 inference, so
that the engines can be compared with tsinfer on the workloads it
actually runs. The paths tsinfer found are the ground truth.

### Data

A colleague ran tsinfer at commit `cb5017e` on the 1000 Genomes high
coverage chr20p (3,202 individuals, 580,909 inference sites), with the
parameters in `full_inference.toml`: `rho = 0.01` and `mu = 1e-20` for
both ancestors and samples, no path compression, and tsinfer's defaults
of `weight_by_n=True` and no likelihood threshold. The data are copied
from `tmp/for_jerome` to `data/tgp_chr20p/`, leaving out macOS metadata:

```
rsync -a --exclude '._*' --exclude .DS_Store ../tmp/for_jerome/ data/tgp_chr20p/
```

- `tgp_chr20p_inferred_ancestors.zarr`: the 432,851 ancestors.
- `tgp_chr20p_inferred_ancestors.trees`: the ancestors tree sequence
  (433k nodes, 4.2M edges).
- `tgp_chr20p_samples_masked.zarr`: the sample genotypes.
- `tgp_chr20p_inferred_samples_raw.trees`: the result of matching the
  6,404 sample haplotypes, before post-processing.
- `partial_ancestor_match/`: the tree sequence before group 84
  (`before.trees`), the group's ancestors (`group.zarr`, `group.json`) and
  the result (`after.trees`).

See `data/tgp_chr20p/README.md` for details.

### Workflow

#### 1. Compute the match groups

The tree sequences do not record which nodes were matched in which
group, so we recompute the groups with tsinfer's own code. This step
needs a tsinfer checkout at `../../tsinfer`, and runs in its environment
so that tsinfer is not a dependency of the benchmarks:

```
uv run --project ../../tsinfer python tsinfer_groups.py data/tgp_chr20p
```

This writes `data/tgp_chr20p/match_jobs.csv`: 439,255 jobs in 1,487
groups (groups 0–1485 for the ancestors, and group 1486 for the samples).
It checks that the order of the ancestor jobs is the node order of the
ancestors tree sequence, and that group 84 is the one in `group.json`.

#### 2. Make workloads

```
uv run python tsinfer_data.py ancestors data/tgp_chr20p 1485 data/tgp_chr20p/ancestors_g1485 --num-queries 500
uv run python tsinfer_data.py samples data/tgp_chr20p data/tgp_chr20p/samples --num-queries 200
```

- `ancestors GROUP`: the ancestors of one group (all of them by default),
  matched against the tree sequence before the group. This is the
  ancestors tree sequence cut at the group's first node: tsinfer adds
  each group's nodes in order, and a node only copies from earlier
  groups. For group 84 the result is identical to `before.trees`.
- `samples`: a random subset of the sample haplotypes, matched against
  the full ancestors tree sequence. The inference sites are selected from
  the samples zarr with the config's include expression, then restricted
  to the ancestor sites. Haplotypes are checked against the sample
  genotypes in the matched tree sequence.

Each workload directory holds `reference.trees`, the queries, a
`model.json` with tsinfer's parameters, and `tsinfer_group.json` with the
time and memory tsinfer recorded for the whole group. Haplotypes are
encoded as in tsinfer (0 ancestral, 1 derived, -1 missing), and each
query is matched over its window of non-missing sites, as tsinfer does.
Queries are stored in a windowed format (`queries.WindowedHaplotypes`),
because ancestors cover only a short stretch of the genome.
`queries.csv` has the path tsinfer found (`truth_parents`,
`truth_breakpoints`), taken from the edges of the query's node.

#### 3. Run and summarise

```
uv run python run.py data/tgp_chr20p/ancestors_g1485/reference.trees \
    data/tgp_chr20p/ancestors_g1485 -o results/tgp_chr20p_ancestors_g1485.csv
uv run python tsinfer_summary.py \
    results/tgp_chr20p_ancestors_g1485.csv data/tgp_chr20p/ancestors_g1485
```

When a workload has a `model.json`, `run.py` uses its parameters, runs
the `tsinfer` and `tsls` engines by default, and leaves `log_likelihood`
empty (the sc2ts formula does not apply with `weight_by_n=True`).
`tsinfer_summary.py` takes any number of `RESULTS QUERY_DIR` pairs, and
prints a markdown table.

### Caveats

- tskit-hmm cannot be used: the ancestors tree sequence has no samples,
  and tskit-hmm can only copy from samples.
- tsinfer's recorded times are for whole groups, run with many threads
  on a 196-core node. We report its CPU time per haplotype, which includes
  reading genotypes and building the tree sequence, not just the HMM.
- `tsinfer_groups.py` uses tsinfer's private `config._parse_config` and
  `pipeline._build_jobs_with_metadata`, because the config has no API for
  changing its (absolute, cluster) paths. tsinfer's `match` with
  `reference_ts` cannot be used to restart from a given tree sequence
  ([tsinfer#1185](https://github.com/tskit-dev/tsinfer/issues/1185)),
  which is why the workloads call the HMM directly.
- The local tsinfer checkout used for grouping was at `186e1af`
  (a descendant of `cb5017e`); the groups are checked against the data
  as described above.


### Findings so far (October 2026)

Run one query at a time on a 4-core machine with 15 GB of RAM. Ancestor
groups 1483–1485 are random subsets of 500. tsinfer's time is its CPU time
for the whole group divided by the number of haplotypes.

| workload | queries | mean sites | same path as tsinfer | tsinfer-hmm mean time | tsls mean time | tsinfer recorded |
|---|---|---|---|---|---|---|
| ancestors, group 84 | 144 of 144 | 1,390 | 100% (both) | 4.3 ms | 5.2 ms | 9.0 ms |
| ancestors, group 1000 | 70 of 70 | 2,950 | 100% (both) | 9.9 ms | 10.9 ms | 13.6 ms |
| ancestors, group 1483 | 500 of 25,935 | 12,200 | 100% (both) | 68 ms | 73 ms | 60 ms |
| ancestors, group 1484 | 500 of 40,684 | 62,000 | 100% (both) | 182 ms | 189 ms | 214 ms |
| ancestors, group 1485 | 500 of 79,127 | 40,200 | 100% (both) | 154 ms | 170 ms | 166 ms |
| samples | 200 of 6,404 | 580,909 | 100% (both) | 2.53 s | 2.80 s | 3.54 s |

- Both engines find exactly the path tsinfer found for every query, so
  they are drop-in replacements for tsinfer's matcher on this data.
  Sample paths have no mismatches. Ancestor paths have one or two
  mismatches on average: the ancestor's focal sites, whose mutations are
  not yet in the tree sequence.
- tsls is 3–10% slower than tsinfer-hmm in mean time per query, with the
  same matcher memory (64 MiB for ancestors, about 90 MiB for samples)
  and traceback sizes. (These figures predate the engine changes
  described below, and `run.py` timings are biased by run order; see
  [tsls against tsinfer-hmm](#tsls-against-tsinfer-hmm-where-the-time-goes).)
  At the time, building tsls's `MatcherIndexes` took several
  times longer (3.6 s against 0.4 s for the full ancestors tree
  sequence); tsls now uses a `tsls.TreeSequence` (a `tsk_treeseq_t`)
  instead, built once per group.
- Per-query times are close to tsinfer's per-haplotype CPU time, so the
  HMM accounts for most of tsinfer's matching cost.
- The cost is in the last few ancestor groups and the samples. Matching
  all 6,404 samples takes about 4.5 CPU hours with either engine.

### tsls against tsinfer-hmm: where the time goes

This section describes the engine before the `tsls_hmm_t` refactor (when
it was `ancestor_matcher_t` in `matcher.c`); see
[Instruction counts after the refactor](#instruction-counts-after-the-tsls_hmm_t-refactor)
for the current state.

tsls now stores a smaller Viterbi traceback than tsinfer-hmm (13 against
21 entries per site for samples; see `experiments/compression/README.md`),
but it isn't faster, because storing the traceback is under 1% of the
time in both engines. Both engines update the same likelihood lists at
each site, since tsinfer-hmm coalesces them right after storing. The time
goes into walking up the tree, to find each listed node's allele and the
value it inherits, and into moving from tree to tree. Building the first
tree of a query means processing most of the edges before its window,
which matters for late ancestors.

Instruction counts (callgrind, collection limited to
`ancestor_matcher_find_path`, 20 group 1485 ancestors and 1 sample) showed
where tsls did more work:

| | tsinfer-hmm | tsls before | tsls after |
|---|---|---|---|
| 20 ancestors, group 1485 | 6.65 G | 9.89 G | 7.44 G |
| 1 sample | 9.92 G | 12.50 G | 10.96 G |

Two changes to `tsls/lib/matcher.c` account for the difference between
"before" and "after":

1. **Allele lookup.** tsinfer-hmm's trees have the single root 0, whose
   allele it sets to the ancestral state, so the walk up the tree only
   checks for a set allele. tsls allows several roots, so
   `get_allelic_state` also checked for the root at each step. tsls now
   sets every root's allele to the ancestral state before the site's
   update (the roots all have likelihoods, so they are found in the
   likelihood list), and the three likelihood loops use
   `get_marked_allelic_state`, which walks without the check. The walk
   deliberately has no `assert`: the Python module is compiled with
   asserts on, and an `assert` there cost as much as the check it
   replaced (12.71 G on the sample, against 12.50 G before). The traceback
   keeps the general `get_allelic_state`, since its tree has no
   likelihoods.
2. **Building the first tree.** For Viterbi, `init_sweep` now builds the
   first tree with plain edge insertions and removals
   (`ancestor_matcher_next_tree_topology`), as tsinfer-hmm does, instead
   of passing every edge through the general wrappers that handle
   likelihoods, subtree counts and isolated nodes. It also no longer
   initialises `num_below`, which only forward and backward use. Forward
   and backward still build it the general way, because they track
   `num_below`.

Wall times, measured in pairs: each query is run three times with each
engine, alternating, and the best time is kept. Ratios are total tsls time
over total tsinfer-hmm time (100 group 1485 ancestors, 6 samples):

| | tsls before ÷ tsinfer-hmm | tsls after ÷ tsinfer-hmm |
|---|---|---|
| ancestors, group 1485 | 1.35 | 1.10 |
| samples | 1.06 | 1.06 |

- Change 2 is a clear win on late ancestors.
- Change 1 removes about 1.5 G instructions per sample but no wall time.
  The walk up the tree is limited by memory access (each step loads
  `allelic_state` and `parent` for a node that is usually not in cache),
  so the extra compare was nearly free.
- What is left is mostly the same kind of overhead in the coalescing and
  cache code: `ancestor_matcher_get_inherited_likelihood` and
  `ancestor_matcher_reset_likelihood_cache` have asserts and root checks
  inside their walks, which tsinfer-hmm's equivalents don't. By the
  result for change 1, these would also save instructions more than time.
- Timing caveat: `run.py` runs all queries for one engine, then all for
  the next, so drift in machine speed (large on this VM: the same query
  can vary by 20% between passes) shows up as a difference between
  engines. Ratios from `run.py` ranged from 1.09 to 1.25 on the same
  builds. Use paired measurements, as above, to compare engines.

### Instruction counts after the `tsls_hmm_t` refactor

The engine is now `tsls_hmm_t` (`tsls/lib/hmm.c`). It works directly on a
`tsk_treeseq_t`, moves between trees with tskit's `tsk_tree_position_t`,
and finds the first tree of a window by seeking from a null position
instead of applying every edge before it. The git history from `7277b87`
records each optimisation step and the dead ends, with instruction counts.

`callgrind.py measure` counts instructions in `find_path` only, with setup
run uninstrumented, on a profiling set sized to run in a few minutes
(tsinfer-hmm's counts are cached, so only tsls is remeasured):

```
uv run python callgrind.py measure -j 4
```

The sample in that set is matched against its reference cut at site
100,000, because matching only the first sites of the full reference makes
tsinfer-hmm's traceback walk in from the end of the sequence, which a whole
sample doesn't. On the original set (20 ancestors per group, 1 whole
sample), before the refactor and after the optimisations:

| workload | tsinfer-hmm | tsls before | tsls after | ratio after |
|---|---|---|---|---|
| ancestors, group 84 | 0.296 G | 0.608 G | 0.154 G | 0.52 |
| ancestors, group 1000 | 0.621 G | 0.975 G | 0.320 G | 0.52 |
| ancestors, group 1483 | 2.388 G | 2.845 G | 1.813 G | 0.76 |
| ancestors, group 1484 | 9.911 G | 11.137 G | 9.046 G | 0.91 |
| ancestors, group 1485 | 6.653 G | 7.445 G | 5.679 G | 0.85 |
| 1 sample | 9.922 G | 10.955 G | 9.633 G | 0.97 |

tsls finds tsinfer's path for all 1,914 queries in the six workloads, and
its traceback sizes are unchanged.

### Wall time after the refactor

Fewer instructions did not mean less time. `paired.py` times each query
three times with each engine, alternating, and keeps the best time for
each (on an idle machine; the ratio is total tsls time over total
tsinfer-hmm time, including building the matcher for each query, as
`engines.py` does):

| workload | queries | tsinfer-hmm (s) | tsls (s) | ratio |
|---|---|---|---|---|
| ancestors, group 84 | 50 | 0.21 | 0.30 | 1.43 |
| ancestors, group 1000 | 50 | 0.46 | 0.55 | 1.19 |
| ancestors, group 1483 | 100 | 4.76 | 5.67 | 1.19 |
| ancestors, group 1484 | 50 | 5.23 | 6.42 | 1.23 |
| ancestors, group 1485 | 100 | 12.98 | 16.20 | 1.25 |
| samples | 6 | 17.69 | 20.22 | 1.14 |

Two things account for this:

- Building a `tsls.Hmm` takes 4-8 ms against 1.5 ms for tsinfer-hmm's
  `AncestorMatcher`, because it encodes the alleles of every site from the
  tables each time. This dominates for early ancestors.
- `find_path` itself is slower (1.10 on group 1485, 1.18 on samples, timed
  separately) despite executing fewer instructions, because it misses the
  cache more. Callgrind's cache simulation on the profiling sample: tsls
  has 2% fewer instructions but 25% more L1 read misses, 2.2 times the L1
  write misses and 35% more last-level read misses than tsinfer-hmm. The
  last-level misses are mostly in tskit's tree position scans, which read
  edge coordinates through the index arrays (`left[order[j]]`) where
  tsinfer-hmm scans a contiguous array of sorted edges, and in reading
  mutations from `tsk_mutation_t` structs. The extra L1 write misses come
  from two of the optimisations, which write per-edge marks into
  `allelic_state` and `likelihood_cache`.
