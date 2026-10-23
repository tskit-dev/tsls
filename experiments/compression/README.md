# Likelihood compression

This directory captures the exact per-site Viterbi and forward likelihoods
computed by tsls and analyses how they are compressed. The aim is to have
fixed, real data for developing better representations and algorithms,
without changing the engine.

```
uv sync
uv run python capture.py --sizes 10000,30000,100000   # about 35 minutes
uv run python report.py                               # results/report.md
```

`capture.py --cases` captures a subset of the cases (`msprime`, `sc2ts`,
`tsinfer`), replacing their rows in an existing manifest and keeping the
others. `report.py --cases` analyses a subset, for quick iteration.

`capture.py` uses `Hmm.viterbi_matrix` and `forward_matrix`. It
saves each result as a `CompressedMatrix` with `CompressedMatrix.dump`, next
to the tree sequence it refers to. Reload a matrix with
`tsls.load_compressed_matrix(path, ts)`.

The Viterbi matrix records, at every site, the list of (node, value) pairs
stored in the traceback, with each node's `recombination_required` flag.
The values are normalised by the maximum, and the maximum is stored as the
site's normalisation factor. Results are written to `results/`, which is not
tracked by git. The sc2ts part needs the sub-ARGs and queries in
`benchmarking/data`.

The cases captured are:
- **sc2ts sub-ARGs** with 10k, 30k and 100k samples (11k, 33k and 111k
  nodes), using the benchmark settings (k = 3, `weight_by_n=False`). The
  queries are two each of copies with k = 0, 2 and 10 extra mutations, and
  recombinants with one or two breakpoints.
- **msprime ARGs**: held-out samples matched against 50 or 1000 other
  samples, with the same HMM settings.
- **tsinfer workloads**: the late ancestor matching groups 1483, 1484 and
  1485, and sample matching, from the tsinfer 1.0 inference of 1000 Genomes
  chr20p (580,909 sites). These need the workloads made by
  `benchmarking/tsinfer_data.py` (see `benchmarking/README.md`). They use
  tsinfer's settings (ρ = 0.01, μ = 1e-20, `weight_by_n=True`, two
  alleles). For each group there are 4 ancestors at spread quantiles of
  window length, and there are 4 random sample haplotypes. Each path is
  checked against the one tsinfer found.
  - Viterbi runs over each query's window: the whole genome for samples.
  - `forward_matrix` has no start and end, so forward runs on the tree
    sequence restricted to the window's sites (`window_ts`), which keeps the
    node IDs and n. For samples the forward window is the middle 20,000
    sites (`--sample-forward-sites`). `capture.py --quick` checks the
    windowing by comparing windowed Viterbi with Viterbi over
    `[start, end)`.

`analysis.py` computes the measures:
- From the stored lists, at every site: the list size, distinct values,
  overlap with the previous site's list, the entries added and removed,
  recombination flags, and whether the engine stores a new traceback list.
- By decoding every node's value, at 200 sampled sites:
  - the minimum number of stored nodes ("change points");
  - redundant stored nodes;
  - clade sizes;
  - change points after rounding the values;
  - an approximate minimum if only samples had to be represented, as in
    tskit (10 sites).
- Encoding measures, comparing each site with the previous one:
  - churn: nodes entering or leaving the stored list;
  - changed values of nodes in both lists (a delta encoding);
  - affine residuals: values of nodes in both lists that the site's global
    map does not predict. For Viterbi the map is
    `f(v) = max(v·stay, ρ/n)·e/c`, with `stay = 1 − ρ + ρ/n`, `e` the
    matching emission and `c` the normalisation factor; this is exact. For
    forward it is `f(v) = a·v + b`, with `a = (1 − ρ)·e/c` and `b` fitted.
  - the same on the minimal (change point) lists at the decoded sites,
    whose previous sites are decoded too.

## The representation

At each site the engine stores a list of (node, value) pairs, and every
node in the tree takes the value of its nearest ancestor in the list.

In tsls every node in the tree is a state with its own value. So the
smallest list that represents the values exactly is the set of **change
points**: nodes whose value differs from their parent's, plus the roots.

Merging siblings with equal values isn't possible, because their parent has
its own, different value. This corrects an earlier suggestion in
`experiments/thresholding`.

## Findings

Ranges are over the captured queries. "Change points" and samples-only
figures come from 200 and 10 sampled sites respectively.

| | sc2ts 10k | sc2ts 30k | sc2ts 100k | msprime 50 / 1000 |
|---|---|---|---|---|
| Viterbi stored / site | 312–1,242 | 515–4,088 | 666–7,598 | 9–19 |
| Viterbi change points / site | 314–1,279 | 519–4,181 | 676–7,582 | 6–16 |
| Viterbi distinct values / site | 4.1–6.3 | 4.3–6.8 | 4.5–7.1 | 3–5 |
| Viterbi samples-only / site | 299–1,302 | 458–4,395 | 465–7,285 | 4–11 |
| Viterbi median change-point clade size (samples) | 1.0 | 1.0 | 1.0–1.1 | 3–105 |
| Viterbi list overlap with previous site (Jaccard) | ≥ 0.999 | ≥ 0.997 | ≥ 0.991 | 0.45–0.64 |
| Viterbi entries added + removed / site | 1.0–1.3 | 3.3–4.5 | 10–13 | 8–12 |
| Sites where the engine stores a new list | 21–24% | 44–49% | 74–77% | 98–100% |
| Traceback entries: engine ÷ delta encoding | 74–230× | 77–429× | 50–428× | 1.1–1.6× |
| Forward stored / site | 255–257 | 320–321 | 355–356 | 35–163 |
| Forward distinct values / site | 180–182 | 148 | 91 | 30–122 |
| Forward change points, log10 values rounded to integers | 4.2 | 2.4–2.6 | 5.7 | 10–24 |

### 1. The stored lists are already minimal

For both Viterbi and forward, the stored list equals the minimum number of
change points, give or take sampling noise, with almost no redundant
entries. On sc2ts there are under 6 redundant entries per site. On the
small simulated ARGs there are 2–5: these are mutation nodes inserted at
the site that end up equal to their parent.

Representing only the samples, as tskit does, would save at most 5–30% on
sc2ts. So the all-nodes model is not what makes the lists large.

This does not hold on tsinfer workloads, where the trees change much more
often. There, a quarter to a third of the Viterbi entries were redundant
until the engine was changed to coalesce before storing. See
[Affine and delta encodings on tsinfer workloads](#affine-and-delta-encodings-on-tsinfer-workloads).

### 2. Why Viterbi stores so many nodes with so few values on sc2ts

The Viterbi values are almost entirely four exact classes:
- 1: matches the query since the last switch;
- μ/(1 − 3μ): one mismatch;
- (μ/(1 − 3μ))²: two mismatches;
- ρ: recombined.

With `weight_by_n=False`, the probability of staying on a node is 1, and
normalisation divides out the match probability. So a node's value is
constant while it matches the query. A third mismatch takes it below ρ,
where it is replaced by the recombination value.

The stored nodes are therefore the boundary of the query's "at most two
mismatches" neighbourhood in the ARG. On sc2ts, nearly all of them are
single samples (median clade size 1): samples with private mutations under
the very large polytomies of the sc2ts trees.

The list size depends on where the query sits in the ARG, not on how
divergent it is. Copies of the same source with k = 0, 2 or 10 extra
mutations give the same counts. Different sources differ by up to 11×.

### 3. The Viterbi traceback is dominated by repeats

The lists hardly change along the sequence. On sc2ts, about 1–13 entries
are added or removed per site, against hundreds or thousands stored. But
the engine reuses the previous list only if it is identical, so it stores
a full copy at 21–77% of sites.

A **delta encoding** stores the first list, then only the entries added and
removed. It would reduce traceback memory by about 50–430× on sc2ts. On the
full ARG, traceback memory reached 3–7 GB per query. On simulated ARGs the
lists change almost completely at every site, so the gain there is small
(1.1–1.6×).

### 4. Forward has many distinct values but very little structure

The forward lists are also minimal (about 255–355 values per site on sc2ts),
but every value changes at every site. That's because the recombination
term ρ/n·S is added to every node, so a delta encoding of values doesn't
help.

The values cluster tightly, though. Rounding log10 values to integers
leaves only about 2–6 change points per site on sc2ts, which is 50–140×
fewer. Approximate compression, such as quantising values or a floor, is
very effective for forward/backward. That fits the thresholding results,
where `floor = 1e-4` cut forward storage by about 200×.

## Implications

- **Viterbi memory:** delta-encode the traceback. Store each node's value
  and flag only when it changes, and walk the changes backwards in the
  traceback. This is exact.
- **Viterbi time:** delta encoding alone doesn't reduce the number of
  tracked nodes, which is already minimal, or the per-site loop over them.
  But only about 1–13 values change per site on sc2ts, because a uniform
  match factor cancels in normalisation. An engine that keeps log-values
  with a global offset could update only the nodes under the site's
  mutations, plus those crossing the recombination level. The per-site work
  would then scale with the number of changes rather than with the list
  size. This needs prototyping and profiling.
- **Forward/backward time:** apart from nodes under disagreeing mutations,
  each site's update is the same scale-and-shift for every node. A global
  transform plus explicit updates for those nodes would give forward the
  same structure. Separately, quantising values is a large and effective
  approximation.
- The captured matrices (`results/*.npz`) can be used to size and check
  any new encoding offline before it is implemented in C.

## Affine and delta encodings on tsinfer workloads

Means over 4 queries per workload, for late ancestor matching (groups
1483–1485) and sample matching on 1000 Genomes chr20p. Viterbi covers each
query's whole window; forward covers ancestor windows and 20,000 sites for
samples.

### Before coalescing the traceback

| | ancestors g1483 | g1484 | g1485 | samples |
|---|---|---|---|---|
| Viterbi stored / site | 12.2 | 11.5 | 20.1 | 26.9 |
| Viterbi change points / site | 9.1 | 8.3 | 13.2 | 19.8 |
| Viterbi churn (entering + leaving) / site | 6.0 | 6.3 | 13.8 | 16.1 |
| Viterbi values changed / site | 0.73 | 0.76 | 1.13 | 1.25 |
| Viterbi affine residuals / site | 0.22 | 0.19 | 0.32 | 0.28 |
| Traceback: engine ÷ delta encoding | 1.6× | 1.6× | 1.1× | 1.7× |
| Forward stored / site | 73 | 80 | 139 | 185 |
| Forward values changed / site | 70 | 77 | 132 | 176 |
| Forward affine residuals / site | 1.6 | 1.6 | 2.8 | 3.1 |
| Forward: stored ÷ affine + delta | 7.8× | 8.5× | 8.0× | 8.5× |

- Viterbi lists are small (12–27 per site, against hundreds to thousands
  on sc2ts), and their values hardly change: about one per site, and only
  0.2–0.3 per site that the global map `max(v·stay, ρ/n)·e/c` doesn't
  predict. `weight_by_n=True` doesn't change this, because the uniform stay
  factor cancels in normalisation. Exact and tolerant comparisons agree, so
  rounding doesn't break a delta encoding.
- But the lists churn: half or more of the entries enter or leave at each
  site, because the tree changes about every second site. So delta
  encoding the traceback saved only 1.1–1.7×, against 50–430× on sc2ts.
- Most of the churn was the engine's bookkeeping, not structure in the
  likelihoods. When edges change, nodes get explicit copies of their
  inherited values, and the engine stored the traceback before removing
  values equal to their parent's. A quarter to a third of the stored
  entries were redundant (7 of 27 for samples), against almost none on
  sc2ts. On the
  minimal lists the churn was only 1.3–2.2 per site.
- Forward values all change at every site, so delta encoding doesn't help
  (0.95×), but the affine map `a·v + b` predicts all but 1.6–3.1 values per
  site. Forward has the same redundant entries (3–9 per site), because it
  also stores before coalescing.

### Coalescing before storing the traceback

The fix is to coalesce before storing. The catch is the traceback: it
stores only `(node, recombination_required)`, and during traceback a node
not in the list takes the flag of its nearest ancestor in the list. A node
with the same value as its parent but a different flag can't be removed
without changing the path. So the coalescing pass, when run before storing
in Viterbi, removes a node only if both its value and its flag equal the
ones it inherits. Nodes kept because of their flag are removed at a later
site, once their flag is recomputed.

What was tried, in order:

1. **Flag-aware coalescing before storing, plus the old pass after
   storing.** The stored lists became exactly the change points, and the
   paths were unchanged. But it ran two coalescing passes per site, and tsls
   went from about 1.1× to 1.36–1.48× tsinfer-hmm's time.
2. **Flag-aware coalescing before storing only (one pass per site).** The
   stored lists are the same as in 1. The nearest-ancestor walk for the
   flag cost 2–4% more instructions than the old engine (table below). Its
   time relative to tsinfer-hmm, measured in separate runs, looked like
   1.13–1.22× (against 1.09–1.10× before). This turned out to be mostly
   noise in comparing ratios across runs.
3. **As 2, with the inherited flag cached** (landed). The flag is cached
   along tree paths with the inherited likelihood
   (`ancestor_matcher_get_inherited_likelihood_and_flag`, which uses the new
   `recombination_required_cache` array), so the walk is shared. It costs
   no more than the old engine.

Instruction counts, from callgrind with collection limited to
`ancestor_matcher_find_path`, so they are deterministic:

| | old engine | 2: flag lookup uncached | 3: flag lookup cached |
|---|---|---|---|
| 20 ancestors, group 1485 | 9.92 G | 10.13 G (+2.2%) | 9.89 G (−0.2%) |
| 1 sample | 12.54 G | 13.01 G (+3.8%) | 12.50 G (−0.3%) |

Head-to-head wall times of the old engine and 3 agree within noise
(4 samples: 11.2 vs 11.3 s; 100 group 1485 ancestors: 14.1 vs 14.4 s, best
of several runs, in alternating order). The 1.36–1.48× for version 1 is
also a ratio to tsinfer-hmm from a separate run. It is well outside the
noise, but not as precise as the counts above.

The matrices below were captured with version 1; versions 2 and 3 store
the same lists.

| | ancestors g1483 | g1484 | g1485 | samples |
|---|---|---|---|---|
| Viterbi stored / site | 9.2 | 8.3 | 13.1 | 18.8 |
| Viterbi churn / site | 1.3 | 1.3 | 1.8 | 2.2 |
| Viterbi values changed / site | 0.23 | 0.21 | 0.32 | 0.26 |
| Engine traceback entries (change) | −35% | −35% | −47% | −31% |
| Traceback: engine ÷ delta encoding | 5.1× | 5.0× | 5.2× | 8.4× |
| Traceback: old engine ÷ delta encoding | 7.8× | 7.7× | 9.7× | 12× |

- The stored lists are now exactly the change points, and churn falls about
  5–8×. Paths are unchanged: tsls still finds tsinfer's path for every
  query in all six benchmark workloads (2,014 queries).
- The engine's traceback shrinks by a third to a half, and a delta encoding
  on top of it saves another 5–8×. In the benchmarks, mean traceback size
  for samples fell from 21.6 to 13.4, and the matcher's memory from 91 to
  64–71 MiB.
- Not changed: forward and backward still coalesce after storing (values
  only; they have no flags), so their stored lists keep the redundant
  entries described above.
- The sc2ts and msprime results earlier in this README were captured with
  the old engine. They had almost no redundant entries, so they shouldn't
  change much.

### Implications for tsinfer

- Coalesce before storing (done for Viterbi) before trying a delta encoding.
  Delta encoding then cuts traceback memory by 5–8×, though tsinfer's
  traceback is already small (13 entries per site for samples).
- In an affine Viterbi engine, the per-site work would be dominated by
  churn from edge insertions and removals (1–2 per site), not by value
  updates (0.1–0.3 per site).
- For forward and backward, the affine update is the clear win: about 8×
  fewer explicit updates on tsinfer workloads, and 50–100× on sc2ts.
  Forward should also coalesce before storing.
