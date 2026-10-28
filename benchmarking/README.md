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
- tsls always uses `n = num_nodes` (see `docs/likelihood_model.md`).

Likelihoods are renormalised at every site, so only the ratio of
switching to staying matters. A tsinfer `rho` of `r` is therefore
equivalent to a `rho` of `r / (r + (1 - r) / n)` in an engine that divides
by `n` (`tsls.model.recombination_probability(r, n)`). With this
conversion, the engines optimise the same objective. The model dicts give
`rho` and `weight_by_n` as tsinfer-hmm takes them, and the tsls engine
converts them.

With a mismatch threshold of k, tsinfer-hmm floors the normalised values
at `0.99 mu^k`, as sc2ts does. tsls instead prunes nodes below the exact
threshold for k mismatches (`tsls.model.path_threshold`). This is
guaranteed to find an optimal path whenever an optimal path has at most k
mismatches and no switches.

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

### Recombination maps

tsinfer can set a source's `recombination_rate` to a rate per bp or a
HapMap genetic map, giving per-site recombination probabilities with
Haldane's map function (`tsls.model.recombination_probabilities`, copied
from tsinfer). `run.py --recombination-rate RATE_OR_MAP` (with an optional
`--mismatch`) reruns a workload with these probabilities in place of the
`model.json` rho, so the paths can be compared with the ones tsinfer found
with constant `rho = 0.01`. A config with `recombination_rate` is also read
into `model.json`. To get the HapMap II GRCh38 chr20 map:

```
uv run --with stdpopsim python genetic_map.py data/genetic_maps/chr20_HapMapII_GRCh38.txt
uv run python run.py data/tgp_chr20p/samples/reference.trees data/tgp_chr20p/samples \
    --recombination-rate data/genetic_maps/chr20_HapMapII_GRCh38.txt \
    -o results/recomb_samples_map_mu1e-20.csv
```

#### Findings (October 2026)

Sample matching on chr20p (the `samples` workload), run 4 at a time on
the 4-core machine. On these sites the map gives 0.508 Morgans in total,
with per-site probabilities from 3e-12 to 6e-4 (median 8.4e-8), against
tsinfer's constant 0.01. With n = 432,853 nodes the median switch odds
fall from 2.3e-8 to 1.9e-13. tsls's probabilities are identical to
tsinfer's.

| recombination | mu | engine | queries | segments | mismatches | tsinfer breakpoints kept | mean time (s) | mean memory (MiB) | mean traceback |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 (tsinfer) | 1e-20 | tsls | 200 | 244 | 0 | 100% | 2.2 | 12 | 13 |
| map | 1e-20 | tsinfer | 33 | 159 | 20.7 | | 12.2 | 349 | 113 |
| map | 1e-20 | tsls | 30 | 166 | 21.6 | 3.1% | 4.5 | 36 | 106 |
| map | 1e-8 | tsls | 200 | 130 | 86.8 | 2.0% | 4.6 | 37 | 114 |
| map | 1e-4 | tsls | 200 | 108 | 130 | 1.7% | 4.9 | 38 | 134 |

(Segments for the tsinfer row are over its 33 queries; on the 30 queries
both engines ran they agree exactly.)

- The map changes sample paths completely. No query has tsinfer's path,
  and only 2-3% of tsinfer's breakpoints are kept: breakpoints move into
  the map's hotspots.
- There are mismatches even with `mu = 1e-20`. With constant rho, a
  difference at one site is explained by switching away and back, at odds
  about q^2 = 5e-16, far more than e = 1e-20. Under the map q^2 is about
  4e-26 at the median, so one mismatch is cheaper than two switches. Larger
  mu trades segments for mismatches, as expected.
- The map makes matching 2-5 times slower, with 3 times the memory and 8
  times the traceback: with smaller switch odds, likelihoods are less often
  equal, so fewer are merged. tsinfer-hmm's memory grows much more than
  tsls's (349 against 36 MiB).
- tsinfer-hmm and tsls find the same paths under the map, as with
  constant rho.

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

Fewer instructions did not at first mean less time, so wall time was then
optimised directly. `paired.py` matches each query three times with each
engine, alternating between them, keeps the best time for each, and
reports total tsls time over total tsinfer-hmm time for each workload
(the per-query median ratio is also printed). The times include building
the matcher for each query, as `engines.py` does. `paired.py --quick`
runs a smaller set, but it is too noisy on this VM to compare builds (see
below).

```
uv run python paired.py
```

#### Current result

Two full runs of the current build (`832fc5f`), tsls / tsinfer-hmm:

| workload | queries | round 1 | round 2 |
|---|---|---|---|
| ancestors, group 84 | 50 | 0.80 | 0.81 |
| ancestors, group 1000 | 50 | 0.82 | 0.82 |
| ancestors, group 1483 | 100 | 0.78 | 0.78 |
| ancestors, group 1484 | 50 | 0.84 | 0.82 |
| ancestors, group 1485 | 100 | 0.90 | 0.87 |
| samples | 6 | 0.79 | 0.82 |

tsls is now 10-22% faster than tsinfer-hmm on every workload, with the
same paths and traceback sizes. It executes more instructions than
tsinfer-hmm on samples again (1.13 times, on the profiling sample), but
fewer on all the ancestor groups.

#### How it got there

| workload | after the refactor | + shared index | + huge pages | + two node records |
|---|---|---|---|---|
| ancestors, group 84 | 1.43 | 0.73-0.77 | 0.72-0.76 | 0.80-0.81 |
| ancestors, group 1000 | 1.19 | 0.75-0.76 | 0.74-0.76 | 0.82 |
| ancestors, group 1483 | 1.19 | 0.88-0.92 | 0.80-0.96 | 0.78 |
| ancestors, group 1484 | 1.23 | 0.99-1.00 | 0.79-0.96 | 0.82-0.84 |
| ancestors, group 1485 | 1.25 | 0.97-1.06 | 0.95-1.00 | 0.87-0.90 |
| samples | 1.14 | 0.99-1.17 | 0.81-1.11 | 0.79-0.82 |

(Ranges are over four runs each of the shared index and huge pages
builds, and two of the current build, run alternately with other
builds.)

- **After the refactor**, tsls was slower despite executing fewer
  instructions. Building a `tsls.Hmm` took 4-8 ms against 1.5 ms for
  tsinfer-hmm's `AncestorMatcher`, because each one encoded the alleles
  of every site from the tables, which dominated for early ancestors.
  And `find_path` missed the cache more: callgrind's cache simulation on
  the profiling sample showed 2% fewer instructions but 25% more L1 read
  misses and 35% more last-level read misses than tsinfer-hmm, mostly from
  reading edges through tskit's order arrays and mutations from
  `tsk_mutation_t` structs.
- **Shared index** (`92d471b`, `ebf50a2`): a `tsls_index_t`, built once
  per tree sequence and shared by HMMs, holds the allele encoding, compact
  mutation arrays, and the edges copied in tskit's insertion and removal
  orders, which the ranges of a `tsk_tree_position_t` index directly.
  Late ancestors gain most, because finding their first tree scans most of
  the edges, now sequentially.
- **Huge pages** (`9806f55`): the likelihood walks jump around per-node
  arrays of several MB each (433k nodes), so with 4 KB pages most steps
  miss the TLB. The large arrays are now mapped directly, aligned to
  2 MiB, with `madvise(MADV_HUGEPAGE)` (Linux). They have to be mapped
  directly: glibc reuses memory already faulted in with small pages, which
  `madvise` doesn't change, so `posix_memalign` + `madvise` gave no huge
  pages at all.
- **Two node records** (`832fc5f`): the per-node fields the walks read are
  kept in an 8-byte record {parent, allelic state, recombination flag and
  its cache} and a 16-byte record {likelihood, its cache}, so the allele
  walk reads one record per step and the likelihood cache walks two,
  instead of two and four arrays. This helps the late groups and samples
  and costs a little on the early groups, whose queries take only 4-8 ms.
  It also costs instructions (the profiling sample goes from 1.52G to
  1.76G; GCC forms a pointer to the record at each step of the walk), but
  each step is still one memory access.

#### Measuring on this machine

- Single runs mislead. tsls's time for the same build varied by up to 20%
  between processes (6 samples: 16.4 to 19.9 s) while tsinfer-hmm's was
  steady, which fits physical page placement varying between runs. With
  huge pages alone it still varied (14.2 to 18.0 s over four runs); with
  the two node records, the two runs gave 14.1 and 14.5 s. Another session
  on the machine may also compete for memory bandwidth at times. Builds
  were therefore compared with full runs repeated in alternation (build A,
  build B, A, B), and a single `--quick` run once showed 1.36 for group
  1485 where full runs of the same build gave 0.85-0.92.
- Callgrind's cache simulation is useful for finding misses but not for
  judging changes: it showed the edge copies increasing last-level misses
  (1.31M to 1.83M on the profiling sample, since tskit's scans still read
  the edge table and the copies add a second stream), yet they were much
  faster, because the simulator doesn't model the hardware prefetching
  that hides the cost of a sequential stream.
- Instruction counts and wall time can move in opposite directions:
  reverting two changes that reduced instructions by adding per-edge
  writes (the cache clear in `remove_edge`, and keeping roots marked
  through the sweep) made no measurable difference to the samples.

#### Node layouts

A `perf` profile of the shared index build put the samples' remaining
time in the likelihood cache walks, which read several per-node arrays at
each step. Two layouts that put these fields together were tried, each in
two full runs alternating with the then-current build:

- One 24-byte record per node (likelihood, its cache, parent, allelic
  state, recombination flag and its cache): worse everywhere, before huge
  pages (samples 1.52 and 1.52; ancestors up to 1.64). Indexing 24-byte
  records also costs an address computation at every step of the walks:
  instructions on the profiling sample rose from 1.52G to 1.82G.
- The two records now used: no consistent difference from separate arrays
  before huge pages, but faster on the late groups and samples with them
  (above).

## Algorithmic changes: memoised walks, delta traceback, sparse sites

After the refactor, tsls was level with or slower than tsinfer-hmm on
sc2ts. `paired.py` now also times sc2ts sub-ARGs (see its docstring):
`sc2ts_n100k_k1` (10 queries with sc2ts's k = 1 likelihood threshold, as in
its first matching passes), `sc2ts_n100k` (the same queries, exact) and
`sc2ts_n300k` (4 queries, exact). A path that differs from tsinfer-hmm's
is classed by its model log likelihood as a tie, better or worse.

Three changes, each recorded in its commit with measurements and dead
ends:

1. **Memoised allele walk** (`e399d67`). Finding each likelihood node's
   allele walked up to a mutation or root from every node, so the cost
   was the sum of the path lengths: about 40% of `find_path` on sc2ts and
   58% on a tsinfer sample. Walks now leave the state on the nodes they
   pass, so the cost is the size of the union of the paths.
2. **Delta traceback** (`ce0334a`). Each site stores only the nodes whose
   recombination flag changed, or that entered or left the list, with
   their previous flags; the traceback undoes them right to left.
3. **Sparse sites** (`c81ac3b`). On sc2ts only about 6 of hundreds to
   thousands of stored values change per site. Once the previous site's
   maximum likelihood node matches the query, every matching node goes
   through the same map as before, so only nodes that mismatch, or changed
   at the previous site, need recomputing. When a tree spans enough
   sites, its nodes are numbered in preorder, so the nodes under each
   mutation are a range of the (sorted) likelihood list. Other sites, and
   tsinfer, whose trees change about every second site, are processed in
   full. `Hmm(sparse=False)` turns this off.

Paired wall time, tsls / tsinfer-hmm, two full runs of each build in
alternation:

| workload | before (3868de8) | + memoised walk | + delta traceback | + sparse sites |
|---|---|---|---|---|
| ancestors, group 84 | 0.76, 0.75 | 0.78, 0.78 | 0.78, 0.77 | 0.78, 0.75 |
| ancestors, group 1000 | 0.82, 0.74 | 0.71, 0.71 | 0.72, 0.70 | 0.73, 0.74 |
| ancestors, group 1483 | 0.91, 0.76 | 0.81, 0.68 | 0.73, 0.68 | 0.73, 0.72 |
| ancestors, group 1484 | 0.91, 0.74 | 0.75, 0.68 | 0.71, 0.67 | 0.73, 0.69 |
| ancestors, group 1485 | 0.98, 0.90 | 0.78, 0.69 | 0.84, 0.84 | 0.89, 0.72 |
| samples | 1.14, 0.80 | 0.76, 0.57 | 0.55, 0.55 | 0.58, 0.56 |
| sc2ts 100k, k = 1 | 1.05, 1.02 | 0.87, 0.85 | 0.84, 0.81 | 0.017, 0.017 |
| sc2ts 100k, exact | 1.09, 1.06 | 0.88, 0.86 | 0.82, 0.80 | 0.012, 0.012 |
| sc2ts 300k, exact | 1.00, 0.95 | 0.79, 0.73 | 0.69, 0.69 | 0.051, 0.048 |

(The tsinfer rows of the last column are from the final build, run
alternately with the delta build; the delta column there is from the
four-build runs.)

- On sc2ts, tsls is now 20-80 times faster than tsinfer-hmm: the 10 exact
  100k queries take 0.75 s against 63 s.
- On tsinfer, the memoised walk gives most of the gain; tsls takes
  0.56-0.75 of tsinfer-hmm's time on the late groups and samples.
- Traceback memory falls from 64 MiB (tsinfer) and up to 1.8 GB (sc2ts
  300k) to 4-12 MiB, the allocator's smallest block.
- Paths are identical to tsinfer-hmm's on every tsinfer query. On sc2ts,
  sparse sites can break ties differently: 5 of 24 queries took a
  different path with the same likelihood, and one k = 1 query took a
  path with 9 more mismatches, which ties in the floored values (the
  floor erases how many mismatches a node has made; see `c81ac3b`).

### Full sc2ts ARG

One run each, k = 1 threshold, on the full benchmark ARG (3.4M nodes,
510 trees, 29,565 sites):

| query | tsls (s) | tsinfer-hmm (s) | ratio | sparse sites | traceback memory (MiB), tsls / tsinfer-hmm |
|---|---|---|---|---|---|
| 0 | 4.3 | 8.3 | 0.51 | 1,229 | 26 / 320 |
| 11 | 27.3 | 104.1 | 0.26 | 4,091 | 26 / 2,240 |
| 22 | 33.6 | 106.7 | 0.31 | 4,093 | 26 / 2,368 |

Paths are the same as tsinfer-hmm's. The gain is much smaller than on
the sub-ARGs because few sites are sparse: trees span about 58 sites
here, against about 1,100 on the 300k sub-ARG, and numbering a tree costs
time in proportion to all 3.4M nodes, so most trees aren't numbered.
Numbering 16 times more eagerly made more sites sparse (6,490 and 9,780
for queries 11 and 22) but was slower (35 and 40 s), so numbering whole
trees is the limit at this scale.
