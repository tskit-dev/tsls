# Thresholding experiments

`thresholding.py` measures how well different ways of limiting the number of
stored likelihood values work in the tsls engine. It also measures what each
one costs on the tasks from `verification/`.

```
uv sync
uv run python thresholding.py --quick -t 4    # about 2 minutes
uv run python thresholding.py -t 4            # full grid; about an hour
uv run python thresholding.py --analyse-only  # redo plots and summary
```

Results, plots and `summary.md` are written to `results/`, which is not
tracked by git. The sc2ts scenario needs the sub-ARGs and queries from
`benchmarking/data`; if they are missing, the scenario is skipped.

## Methods

Every method is applied after the values are normalised at each site:
by the maximum for Viterbi, and by the sum for forward/backward. A "pruned"
node is given the value `likelihood_threshold`, which by default is the
smallest positive double. It then merges into its parent, and can only be
reached again by recombination.

- **`floor`** is the existing `likelihood_threshold`: any value below `t` is
  set to `t`.
- **`relative`** prunes nodes with a value below `f` times the maximum. Use
  `pruning=("relative", f)` (experimental option).
- **`top_n`** prunes all but the `n` nodes with the largest values. Nodes tied
  with the n-th value are kept. Use `pruning=("top_n", n)` (experimental option).
- **`lossless_floor`** is `floor` at `t* = (ρ/n) / (1 − ρ + ρ/n)`. In Viterbi,
  any node below `t*` loses to recombination at the next site, so this floor
  changes no Viterbi likelihoods. With `weight_by_n=False`, as sc2ts uses,
  `t* = ρ`.

`Hmm(count_distinct=True)` also records the mean number of
**distinct** values per site in Viterbi (`mean_num_distinct_values`). This
is compared with the number of stored nodes (`mean_traceback_size`).

## Scenarios

- **threading**: a held-out sample from an msprime simulation, matched as in
  `verification.py` (`HeldOutThreading`). The HMM uses sc2ts parameters
  (k = 3, `weight_by_n=False`). For Viterbi we measure the loss in path
  log-likelihood relative to exact, how often the path is unchanged, and the
  accuracy of copying the true parent. For forward/backward we measure the
  posterior total-variation distance from exact and the posterior on the
  true parent.
- **calibration**: queries simulated from the HMM itself, as in
  `verification.py` (`ModelCalibration`). For forward/backward we measure
  the reliability error, and 90% credible-set coverage minus the expected
  coverage.
- **sc2ts**: queries from `benchmarking/` on sc2ts sub-ARGs, using the tsls
  benchmark engine settings. We measure the loss in Viterbi path
  log-likelihood, how often the path is unchanged, time, and stored values.
  At 10k samples we also measure the forward/backward posterior
  total-variation distance, at 100 evenly spaced sites.

## Findings (quick run)

These numbers come from `--quick`, which uses:
- 3 replicates of threading, with 50 samples only;
- 3 ARGs per calibration setting;
- 3 sc2ts queries on the 10k sub-ARG (an exact copy, and recombinants with 2
  and 5 extra mutations);
- a reduced grid: t and f in {1e-20, 1e-8, 1e-4, 1e-2}, and n in
  {10, 100, 1000}.

Treat them as indicative. The full grid, with 50–1000 samples and sc2ts
sub-ARGs up to 100k, was not run to completion.

### 1. For Viterbi, compression limits storage more than thresholding does

On the 10k sc2ts sub-ARG, exact Viterbi stores 312–1,248 nodes per site,
but only **4–6 distinct values** per site. With 50 simulated samples, it
stores 9.4 nodes for 4.4 distinct values.

Thresholding changes values, not the structure of which nodes need their
own value. The best any method achieved without loss was about a 2×
smaller traceback:

| sc2ts 10k, Viterbi | traceback vs exact | time vs exact | log-likelihood loss | identical paths |
|---|---|---|---|---|
| `lossless_floor` | 1.00 | 0.96 | 0 | 3/3 |
| `top_n` = 10 | 0.52 | 0.61 | 0 | 3/3 |
| `relative` = 1e-4 | 1.00 | 1.04 | 0 | 3/3 |
| `floor` = 1e-4 | 0.93 | 0.90 | 0.06 | 1/3 |
| `floor` = 1e-2 | 0.44 | 0.36 | 1.5 | 1/3 |
| `relative` = 1e-2 | 0.44 | 0.34 | 5.9 | 1/3 |

**Correction.** An earlier version of this README suggested merging
siblings with equal values. That is not possible: every node in the tree is
a state with its own value, so the engine's list is already the minimum for
its encoding. See `experiments/compression/` for an analysis of where the
stored nodes come from.

### 2. For Viterbi, a floor above `t*` is harmful, and pruning should be used instead

A floor raises weak nodes up to `t`. Once `t` is above the recombination
value, staying on a raised node beats switching, so stale copying targets
survive. Pruning sets weak nodes close to 0 instead, so they lose to
recombination as they should.

| threading (50 samples), Viterbi | traceback vs exact | log-likelihood loss | accuracy drop |
|---|---|---|---|
| `floor` = 1e-4 | 0.92 | 200 | 0.34 |
| `relative` = 1e-4 | 0.92 | 0 | 0 |
| `top_n` = 10 | 0.98 | 0 | 0 |
| `lossless_floor` (2.4e-6) | 0.99 | 0 | 0 |

The lossless floor explains why every threshold up to 1e-6 was free in
earlier runs: with sc2ts settings, `t* = ρ ≈ 2.4e-6`.

### 3. For forward/backward, every method distorts the posteriors, in different directions

- **`floor`** works by adding probability to every node, so posteriors no
  longer sum to 1 (they are off by about 0.99 at 50 samples). It is still
  the least biased of the three.
  - On sc2ts 10k, `floor` = 1e-4 cut the stored forward values by about
    200× and the forward-backward time to under 1%. The posterior
    total-variation distance from exact was 0.02.
  - With 50 samples the same setting gave a total-variation distance of
    0.07, and calibration was as good as exact.
  - `floor` = 1e-2 made the posteriors under-confident: 90% coverage was
    9 points above expected.
- **`relative`** and **`top_n`** make the posteriors over-confident:
  - `relative` = 1e-4 gave a total-variation distance of 0.44 with 50
    samples;
  - `top_n` = 10 gave 0.13 with 50 samples and 0.68 on sc2ts;
  - 90% coverage was 7–28 points below expected.

  With `top_n` = 100 on sc2ts, more values were stored than exact (1.23×),
  because pruned values stop compressing as before.

## Recommendations

- **Viterbi:** use the lossless floor `t*` as the default threshold; it
  costs nothing. If more speed is needed, use `relative` or `top_n` rather
  than a floor above `t*`.
- **Forward/backward:** treat any thresholding as approximate. A floor
  around 1e-4 gives large speedups on sc2ts with small posterior error.
  Check the posterior total-variation distance for the intended use.
- **Next step:** see `experiments/compression/`, which captures the exact
  per-site values and analyses alternative encodings.

## Exact prune thresholds (`exact_thresholds.py`)

`docs/likelihood_model.md` derives `Hmm(prune_threshold=t)`, with
`t = tsls.model.path_threshold(k, ...)`. It finds an optimal path whenever
one has at most k mismatches, and certifies a path as optimal if its odds
are at least t. This replaces sc2ts's floor at `0.99 mu^k`, which is 6–15%
below the exact `e^k`. It has no effect at k = 3, since it is below q.
`exact_thresholds.py` compares the two on the 10 sc2ts scaling queries
used by `paired.py`, against the 100k sub-ARG (r = 0.21 from sc2ts's
q = 2.41e-6):

| k | setting | time (s) | mean traceback | certified | optimal |
|---|---|---|---|---|---|
| | exact | 1.14 | 4761 | | 10/10 |
| 0 | floor 0.99 mu^k | 0.50 | 979 | 1/10 | 5/10 |
| 0 | prune e^k | 0.54 | 608 | 1/10 | 8/10 |
| 1 | floor 0.99 mu^k | 0.83 | 2537 | 2/10 | 6/10 |
| 1 | prune e^k | 0.84 | 1977 | 2/10 | 9/10 |
| 2 | floor 0.99 mu^k | 0.92 | 4981 | 4/10 | 9/10 |
| 2 | prune e^k | 0.93 | 4731 | 4/10 | 10/10 |
| 3 | either | 0.95 | 4761 | 4/10 | 10/10 |

The exact prune stores 20–40% fewer values than the floor at k = 0 and 1,
for the same time, and finds the optimal path more often. Every certified
path was optimal, with both methods. Only the prune guarantees this; the
floor could certify a suboptimal path. The pruned uncertified runs are
not guaranteed to be optimal, and the sc2ts escalation reruns them with a
larger k.
