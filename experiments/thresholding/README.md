# Thresholding experiments

How well can the tsls engine limit the nodes it tracks, and what does each
method cost in accuracy? `evaluate.py` answers this for Viterbi, forward
and backward under the unified model (`docs/likelihood_model.md`). It runs
on the sc2ts sub-ARGs and on tsinfer's chr20p ancestor and sample
matching. Every strategy is compared with the exact run of the same
query.

```
uv sync
uv run python evaluate.py --quick -t 2    # a few minutes
uv run python evaluate.py -t 2            # about 20 minutes on 4 cores
uv run python evaluate.py --analyse-only  # redo results/summary.md
```

It needs `benchmarking/data` (the sc2ts sub-ARGs and queries, and the
chr20p workloads from `tsinfer_data.py`). Results go to
`results/viterbi.csv`, `results/forward_backward.csv` and
`results/summary.md`, which are not tracked by git. The numbers below are
from one full run (2026-10-27).

## Setup

### Workloads

| workload | nodes | sites matched | queries | model |
|---|---|---|---|---|
| sc2ts 10k sub-ARG | 11,430 | 29,565 | 10 | sc2ts k = 3: mu = 0.0125, q = 2.41e-6, r = 0.027 |
| sc2ts 100k sub-ARG | 111,248 | 29,565 | 10 | the same q, so r = 0.21 |
| chr20p ancestors, group 84 | 33,934 | 1,519 (mean) | 6 | tsinfer: r = 0.01, mu = 1e-20 |
| chr20p ancestors, group 1000 | 98,045 | 2,829 | 6 | tsinfer |
| chr20p ancestors, group 1485 | 353,726 | 34,000 | 6 | tsinfer |
| chr20p samples | 432,853 | 580,909 | 8 per mu | r = 0.01 and four values of mu (below) |

**Sample matching** allows a small chance of mismatch instead of
recombination. With r = 0.01 and n = 432,853, the switch odds are
q = 2.33e-8. We set mu so that k mismatches cost the same as one switch
(`e^k = q`):

| mu label | mu | effect |
|---|---|---|
| 1e-20 | 1e-20 | tsinfer's setting: no mismatches |
| k=1 | 2.33e-8 | one mismatch costs the same as a switch |
| k=2 | 1.53e-4 | two mismatches cost a switch |
| k=4 | 1.22e-2 | four mismatches cost a switch |

### Scope

- Viterbi runs over each query's whole window. Forward and backward run on
  a window of up to 5,000 sites in the middle of the query. The reference is
  cut with `keep_intervals`, which keeps node IDs.
- Exact forward-backward over all 581k chr20p sites is not feasible on this
  machine. Sample matching at the k = 4 mu stores about 1,000 forward and
  1,300 backward values per site (up to about 3,000). A first attempt over
  the whole sequence ran the machine out of memory. Each worker is now
  limited to 5 GB.
- The simulated threading and calibration scenarios (`thresholding.py`)
  were not rerun.

### Strategies

| algorithm | strategy | what it does | guarantee |
|---|---|---|---|
| Viterbi | `floor` t | raises values below t to t (`likelihood_threshold`) | none if t > q |
| | `lossless_prune` | drops nodes below q (`tsls.model.lossless_threshold`) | changes no later values |
| | `prune_k` k | drops nodes below `e^k` (`tsls.model.path_threshold`) | optimal if the optimal path has ≤ k mismatches; certifies a path with odds ≥ t |
| | `top_n` n | keeps the n largest values (since removed) | none; turns off sparse sites |
| | `escalation` | `prune_k` at k = 0, 1, 2, 4 until the path is certified, then exact | always optimal |
| Forward/backward | `floor` t | raises values below t to t | none |
| | `relative` f | drops nodes below f × max | none |
| | `relative_q` c | drops nodes below c·q × max, below the mass recombination alone gives a node | none proven; near-exact here |
| | `top_n` n | keeps the n largest values (since removed) | none |

### Measures

- **Viterbi accuracy:**
  - loss = exact path log odds − strategy path log odds, under the model
    (0 = optimal);
  - whether the path is identical to the exact one;
  - whether it is certified;
  - "certified suboptimal": the number of certified paths that weren't
    optimal.
- **Forward-backward accuracy:**
  - the error in the forward log likelihood;
  - at 50 sites: the total variation (TV) distance of the posterior from
    exact, the largest absolute error in one node's posterior, and whether
    the posterior argmax node agrees;
  - "no-mass sites": sites where forward and backward keep no node in
    common, so there is no posterior at all (counted as TV = 1).
- **Nodes tracked:**
  - `Hmm.mean_num_values` / `max_num_values`: the likelihood list after
    each site, i.e. the nodes with their own value;
  - `mean_num_updated`: the node values computed per site, which is fewer
    for sparse Viterbi sites;
  - the values stored per site in the forward and backward matrices.
- **Time:** wall time, and its ratio to exact within the same task, with 2
  tasks running at once. Ratios within ±10% are noise.

## Nodes tracked by the exact algorithms

| workload | mismatches | switches | V time | V values mean / max | V updated | sparse sites | F values mean / max | B stored | FB time (5k sites) |
|---|---|---|---|---|---|---|---|---|---|
| sc2ts 10k | 3.4 | 0.5 | 5 ms | 812 / 2,120 | 1.3 | 100% | 170 / 257 | 171 | 0.06 s |
| sc2ts 100k | 3.6 | 0.4 | 97 ms | 4,590 / 11,000 | 15 | 99% | 256 / 435 | 257 | 0.22 s |
| ancestors g84 | 1 | 0.2 | 1.5 ms | 2.4 / 7 | 2.7 | 0 | 8.4 / 40 | 11 | 0.002 s |
| ancestors g1000 | 1 | 0.7 | 5 ms | 2.9 / 10 | 3.6 | 0 | 13 / 67 | 20 | 0.01 s |
| ancestors g1485 | 2.2 | 9.8 | 0.14 s | 8.2 / 169 | 15 | 0 | 126 / 505 | 295 | 0.22 s |
| samples, mu 1e-20 | 0 | 250 | 2.5 s | 15 / 497 | 23 | 0 | 178 / 677 | 352 | 0.23 s |
| samples, k=1 | 48 | 176 | 2.5 s | 15 / 497 | 23 | 0 | 466 / 1,987 | 708 | 0.45 s |
| samples, k=2 | 94 | 135 | 4.0 s | 57 / 1,250 | 65 | 0 | 692 / 2,606 | 978 | 0.60 s |
| samples, k=4 | 177 | 103 | 4.6 s | 100 / 1,660 | 108 | 0 | 972 / 3,084 | 1,330 | 0.77 s |

- **Viterbi tracks few nodes on the tsinfer data:** 2–15 on average, and at
  most 169–497 without mismatches. With little to prune, no strategy can
  change Viterbi time there by more than noise. Mismatches raise the list
  6× (k = 4).
- **On sc2ts, Viterbi tracks hundreds to thousands of nodes:** but sparse
  sites recompute only 1–15 per site, so the list size barely affects
  time.
- **Forward and backward track 10–100× more nodes than Viterbi on the
  tsinfer data:** 170–970 per site for samples, against 15–100. Mismatches
  multiply them 5×. Backward stores 1.4–2.3× as many values as forward.
  The forward-backward matrices dominate memory: about 2,300 values per
  site over 581k sites is about 16 GB.

## Viterbi results

Mean over queries; time and values are ratios to exact.

| workload | strategy | time | values | optimal | max loss | certified |
|---|---|---|---|---|---|---|
| sc2ts 100k | floor 1e-8 | 0.99 | 1.00 | 10/10 | 0 | 5/10 |
| | floor 1e-4 | 0.80 | 1.00 | 7/10 | 4.5 | 3/10 |
| | floor 1e-2 | 0.62 | 0.45 | 6/10 | 8.7 | 2/10 |
| | lossless_prune | 0.98 | 1.00 | 10/10 | 0 | |
| | prune_k 0 | 0.38 | 0.11 | 6/10 | 26 | 1/10 |
| | prune_k 1 | 0.78 | 0.45 | 7/10 | 13 | 2/10 |
| | prune_k 2 | 0.98 | 0.98 | 10/10 | 0 | 3/10 |
| | top_n 10 | 51 | 0.42 | 7/10 | 26 | |
| | top_n 100 | 103 | 0.75 | 9/10 | 13 | |
| ancestors g1485 | floor 1e-4, 1e-2 | 0.96 | 0.94 | 0/6 | 21,000 | 0 |
| | floor 1e-8, lossless, prune_k, top_n | 0.91–1.07 | 0.85–1.00 | 6/6 | 0 | prune_k 3: 2/6 |
| samples, mu 1e-20 | floor 1e-4, 1e-2 | 0.93–0.99 | 0.96 | 0/8 | 2.6e6 | 0 |
| | all others | 0.94–1.10 | 0.92–1.00 | 8/8 | 0 | 0 |
| samples, k=2 | floor 1e-4 | 0.68 | 0.69 | 0/8 | 1.9e5 | 0 |
| | prune_k 0 | 0.58 | 0.13 | 0/8 | 2.0e5 | 0 |
| | prune_k 1 | 0.73 | 0.50 | 8/8 | 4.5e-13 | 0 |
| | prune_k 2, 3; lossless; floor 1e-8 | 1.01–1.04 | 0.98–1.00 | 8/8 | 0 | 0 |
| | top_n 100 | 1.18 | 0.98 | 2/8 | 62 | |
| samples, k=4 | floor 1e-4 | 0.62 | 0.48 | 0/8 | 9.5e4 | 0 |
| | prune_k 0, 1, 2 | 0.49–0.52 | 0.08–0.17 | 0/8 | 9.9e4 | 0 |
| | prune_k 3 | 0.81 | 0.62 | 8/8 | 0 | 0 |
| | lossless; floor 1e-8 | 0.98 | 0.98 | 8/8 | 0 | |
| | top_n 10, 100 | 0.74–1.15 | 0.36–0.83 | 0/8 | 8.9e4 | |

**Escalation** (time over all its runs, as a ratio to exact):

| workload | time | certified at k |
|---|---|---|
| sc2ts 10k | 3.4 | 0: 1, 1: 1, 2: 1, 4: 1, exact: 6 |
| sc2ts 100k | 3.1 | 0: 1, 1: 1, 2: 1, 4: 2, exact: 5 |
| ancestors g84 / g1000 | 2.2 / 2.4 | all at k = 1 or 2 |
| ancestors g1485 | 4.2 | 4: 4, exact: 2 |
| samples (all mu) | 3.5–5.2 | exact: 32 of 32 |

Findings:

1. **The guarantees hold.**
   - No certified path was suboptimal: 74 of the 490 floor and `prune_k`
     runs were certified, and all 74 were optimal.
   - The lossless prune and floors below q (1e-8) never lost anything.
2. **A floor above q is catastrophic on the tsinfer data.** At 1e-4 and
   1e-2, every sample query and every group 1485 ancestor query got a
   wrong path, with losses of 1e3–1e6 in log odds. In group 1000 it was 4
   of 12 runs. Group 84 was unaffected, since its queries are short with
   no switches. When mu is tiny, a floored node keeps "copying"
   through mismatches that the model rules out, instead of recombining. On
   sc2ts (mu = 0.0125) the damage is smaller (losses up to 9), but 3–4 of
   10 paths are still wrong.
3. **Pruning only pays when many nodes are tracked.**
   - On sc2ts 100k, `prune_k 0` takes 0.38 of the time, and on samples at
     k = 4, 0.49. But both are suboptimal on most queries: the optimal
     paths have more mismatches than the budget.
   - The k at which pruning stays optimal (2 on sc2ts, 1–3 on samples)
     saves 0–27%.
   - On the tsinfer ancestors and on samples without mismatches, nothing
     changes time by more than noise.
4. **The certificate is too strict for sample matching.** Paths have 100–250
   switches, and the certificate budget only covers mismatches. So no
   sample path was certified, even when `prune_k 1` was optimal.
   Escalation always fell through to exact and cost 3.5–5.2× exact. On
   sc2ts and the ancestors, half or more of queries also needed several
   reruns, costing 2.2–4.2×.
5. **`top_n` is slow and inexact.** It turns off sparse sites and sorts
   every site, so on sc2ts it is 50–270× slower than exact. On samples
   with mismatches it is also inexact.

## Forward-backward results

Mean over queries; time and values (stored forward + backward) are ratios
to exact. Errors are the largest over queries, except the mean TV.

| workload | strategy | time | values | max \|Δ logL\| | mean TV | max TV | no-mass sites |
|---|---|---|---|---|---|---|---|
| sc2ts 100k | floor 1e-8 | 1.01 | 1.00 | 0 | 0 | 0 | 0 |
| | floor 1e-4 | 0.034 | 0.014 | 12,000 | 0.001 | 0.016 | 0 |
| | relative 1e-2 | 1.02 | 1.00 | 0.002 | 4e-8 | 2e-5 | 0 |
| | relative_q 1 | 1.00 | 1.00 | 0 | 0 | 0 | 0 |
| | top_n 100 | 2.4 | 1.45 | 13 | 0.022 | 0.094 | 0 |
| sc2ts 10k | floor 1e-4 | 0.016 | 0.007 | 669 | 0.012 | 0.26 | 0 |
| | relative 1e-2 | 1.09 | 1.00 | 0.51 | 0.001 | 0.15 | 0 |
| ancestors g1485 | floor 1e-8 | 1.01 | 0.98 | 0.26 | 4e-4 | 0.023 | 0 |
| | floor 1e-4 | 0.73 | 0.64 | 404 | 0.45 | 1 | 0 |
| | relative 1e-8; relative_q 0.01, 1 | 0.92–1.03 | 0.98 | 1e-9 | 1e-12 | 2e-10 | 0 |
| | relative 1e-4 | 0.52 | 0.21 | 40 | 0.16 | 1 | 0 |
| | top_n 1000 | 0.93 | 1.00 | 0 | 3e-7 | 8e-6 | 0 |
| samples, mu 1e-20 | floor 1e-8 | 0.95 | 0.98 | 0.22 | 0.001 | 0.038 | 0 |
| | relative 1e-8; relative_q | 0.98–1.06 | 0.98 | 4e-10 | 3e-12 | 8e-11 | 0 |
| | relative 1e-4 | 0.50 | 0.27 | 40 | 0.25 | 1 | 0 |
| | top_n 100 | 1.01 | 0.69 | 4.5 | 0.088 | 0.99 | 0 |
| samples, k=2 | floor 1e-8 | 0.73 | 0.72 | 0.066 | 0.001 | 0.038 | 0 |
| | relative 1e-8 | 0.80 | 0.80 | 3e-4 | 1e-5 | 1e-4 | 0 |
| | relative_q 1 | 0.80 | 0.77 | 0.001 | 2e-5 | 3e-4 | 0 |
| | relative_q 0.01 | 1.01 | 0.94 | 4e-5 | 2e-7 | 2e-6 | 0 |
| | relative 1e-2 | 0.18 | 0.04 | 2,080 | 0.48 | 1 | 86 of 400 |
| | top_n 1000 | 1.13 | 0.90 | 6e-4 | 0.022 | 1 | 0 |
| samples, k=4 | floor 1e-8 | 0.75 | 0.79 | 0.056 | 6e-4 | 0.034 | 0 |
| | relative_q 1 | 0.93 | 0.90 | 0.003 | 1e-5 | 2.5e-4 | 0 |
| | relative 1e-4 | 0.20 | 0.09 | 1,030 | 0.42 | 1 | 22 of 400 |
| | top_n 100 | 0.39 | 0.19 | 5.3 | 0.19 | 1 | 0 |

Findings:

1. **Only thresholds at or below the switch odds q are accurate.**
   - `relative_q`, `relative 1e-8` and `floor 1e-8` (all about q or less)
     keep the log likelihood within 0.003 for pruning, and the posteriors
     within a TV of 3e-4.
   - `floor 1e-8` is slightly worse (logL error up to 0.26, max TV 0.04),
     because it adds mass.
   - On sample matching with mismatches, they save 10–28% of values and
     7–27% of time.
2. **Anything coarser fails on the tsinfer data.**
   - `relative` at 1e-4 or 1e-2 drops the recombination background that a
     switching path needs.
   - Forward and backward prune independently, so between switch points
     they can keep disjoint sets of nodes. Up to 86 of 400 posterior sites
     had no mass at all.
   - TV reaches 1, and the log likelihood is off by up to 2,000.
   - Floors at 1e-4 and above add so much mass that the log likelihood is
     off by 10^3–10^4 everywhere.
3. **On sc2ts, the cheap floor is tempting but wrong in a different way.**
   `floor 1e-4` cuts time 30–60× with posterior TV ≤ 0.26. But the log
   likelihood is off by 10^3–10^4, so it can only be used for posteriors.
   Relative pruning prunes almost nothing on sc2ts: with r = 0.03–0.2 per
   site, every node carries recombination mass above 1e-4 × max.
4. **`top_n` is unreliable.** 1000 is nearly exact, but 10 and 100 have TV
   up to 1 and slow sc2ts down 2–7×, because they keep sorting.
5. **Argmax agreement is a poor measure here.** Agreement below 1 with TV of
   1e-5 (`relative_q` on samples) comes from ties between nodes with the
   same posterior.

## Recommendations

- **Viterbi, sc2ts:**
  - Run exact. Sparse sites make it 5–100 ms per query, and pruning saves
    at most about 60% only by giving up optimality.
  - If a k-mismatch budget is wanted, use `prune_threshold =
    path_threshold(k)` with the certificate, never a floor above q.
  - Escalation from k = 0 costs 3× exact on these queries, because most
    need a rerun.
- **Viterbi, tsinfer:**
  - Run exact, or use `lossless_threshold`. Viterbi tracks too few nodes
    for thresholds to matter.
  - Never floor above q (the old `0.99 mu^k` style).
  - The k-mismatch certificate doesn't suit sample matching, where paths
    have hundreds of switches.
- **Forward-backward:**
  - Use `relative_prune_threshold=c * q` with c ≤ 1. It is accurate to about
    1e-3 in the log likelihood here, and saves 10–25% with mismatches.
  - Larger thresholds trade large, unbounded errors for speed.
  - Memory, not time, is the limit for forward-backward on tsinfer data.
    The useful next step is a bound on the mass each prune removes, and a
    forward pass that keeps less.

## Sample matching under a recombination map (2026-10-28)

`tgp_samples_map` reruns sample matching with per-site recombination
probabilities from the HapMap II GRCh38 chr20 map
(`benchmarking/genetic_map.py`; Haldane's map function, as tsinfer's
`recombination_rate`), with mu = 1e-20, 1e-8 and 1e-4, on 4 samples each:

```
uv run python evaluate.py --workloads tgp_samples_map --results results/map -t 2
```

It takes about 5 minutes. On these sites the switch odds q range from
7e-18 to about 1e-9, with median 1.9e-13; tsinfer's constant r = 0.01 gives
q = 2.3e-8 at every site. The grid is the strategies kept after the
constant-rate runs (see below), floor 1e-8 (to check that a floor that was
below q is now harmful), and thresholds at c times the median q
(`prune_qmed` for Viterbi, `relative_qmed` for forward-backward), which
have no guarantee. Escalation is not run.

### Exact runs

| mu | mismatches | switches | V time | V values mean / max | F values mean / max | B stored | FB time (5k sites) |
|---|---|---|---|---|---|---|---|
| 1e-20 | 25 | 232 | 4.1 s | 111 / 988 | 202 / 975 | 407 | 0.21 s |
| 1e-8 | 94 | 159 | 4.7 s | 140 / 1,356 | 446 / 1,993 | 702 | 0.38 s |
| 1e-4 | 142 | 135 | 5.5 s | 173 / 1,720 | 665 / 2,571 | 952 | 0.51 s |

Viterbi tracks 2–7× as many nodes as with the constant rate (15–100), and
takes up to twice as long. Forward and backward track about as many as
with the constant rate at similar numbers of mismatches.

### Viterbi

Mean over queries; time and values are ratios to exact.

| mu | strategy | time | values | optimal | max loss |
|---|---|---|---|---|---|
| 1e-20 | floor 1e-8 | 0.45 | 0.17 | 0/4 | 1e6 |
| | lossless_prune (7e-18) | 1.04 | 0.99 | 4/4 | 0 |
| | prune_k 0 | 0.51 | 0.16 | 0/4 | 1,560 |
| | prune_k 1, 2, 3 | 1.00–1.04 | 0.99–1.00 | 4/4 | 0 |
| | prune_qmed 0.01 / 1 | 1.02 / 1.03 | 0.99 / 0.96 | 3/4 / 0/4 | 0.51 / 17 |
| 1e-8 | floor 1e-8 | 0.96 | 1.04 | 0/4 | 4e5 |
| | lossless_prune | 1.02 | 0.99 | 4/4 | 0 |
| | prune_k 0, 1 | 0.47–0.57 | 0.05–0.25 | 0/4 | 4e5 |
| | prune_k 2, 3 | 0.96–0.99 | 0.99–1.00 | 4/4 | 0 |
| | prune_qmed 0.01 / 1 | 1.01 / 1.06 | 0.99 / 0.97 | 4/4 / 2/4 | 0 / 5.9 |
| 1e-4 | floor 1e-8 | 0.90 | 0.89 | 0/4 | 1.7e5 |
| | lossless_prune | 0.95 | 1.00 | 4/4 | 0 |
| | prune_k 0, 1, 2 | 0.36–0.52 | 0.04–0.27 | 0/4 | 2e5 |
| | prune_k 3 | 1.00 | 0.94 | 0/4 | 11 |
| | prune_qmed 0.01 / 1 | 1.00 / 0.95 | 0.99 / 0.97 | 4/4 / 2/4 | 0 / 4.9 |

No path was certified.

### Forward-backward

| mu | strategy | time | values | max \|Δ logL\| | max TV |
|---|---|---|---|---|---|
| 1e-20 | floor 1e-8 | 0.95 | 0.88 | 29 | 0.89 |
| | relative_q 0.01, 1 | 1.01–1.03 | 0.94 | 2e-12 | 2e-14 |
| | relative_qmed 0.01 / 1 | 1.02 / 0.99 | 0.94 | 2e-12 / 7e-6 | 9e-11 / 3e-5 |
| 1e-8 | floor 1e-8 | 0.60 | 0.53 | 14 | 0.89 |
| | relative_q 0.01 / 1 | 1.14 / 0.91 | 0.99 / 0.89 | 2e-13 / 1e-7 | 2e-13 / 4e-10 |
| | relative_qmed 0.01 / 1 | 0.85 / 0.84 | 0.79 / 0.76 | 1e-7 | 4e-9 / 3e-5 |
| 1e-4 | floor 1e-8 | 0.47 | 0.43 | 12 | 0.60 |
| | relative_q 0.01 / 1 | 1.06 / 1.00 | 1.00 | 4e-13 | 4e-13 |
| | relative_qmed 0.01 / 1 | 0.96 / 0.80 | 0.97 / 0.81 | 3e-9 / 1.3e-3 | 2e-9 / 3e-6 |

Here `relative_q` uses the smallest q in the 5,000-site window, which is
larger than the smallest over the whole query.

### Findings

1. **A floor of 1e-8, safe with the constant rate, is now catastrophic.**
   Every Viterbi path is wrong (losses of 1e5–1e6), and forward-backward
   log likelihoods are off by 5–29 with TV up to 0.89. No fixed number is
   safe under a map, because q moves by orders of magnitude between sites.
2. **Thresholds at the smallest q do nothing.** `lossless_prune` is 7e-18,
   and changes neither values nor time.
3. **Viterbi pruning still doesn't pay.** Every threshold that kept the
   optimal path (`lossless_prune`, `prune_k` at k ≥ 1 or 2, `prune_qmed`
   0.01) changed time by no more than noise, and values by 1–6%. The
   thresholds that save time (`prune_k` below the path's mismatch rate, a
   floor) give wrong paths. Even at the median q (`prune_qmed 1`), half or
   more of the paths are wrong and nothing is saved: the nodes Viterbi
   tracks are mostly within the switch odds of the maximum, so there is
   little to prune at any safe level. At mu = 1e-4, `prune_k 3` isn't
   enough either: the paths have about 140 mismatches.
4. **Forward-backward pruning relative to a typical q looks promising.**
   `relative_qmed 1` saves 16–24% of values and time with mismatches, with
   log-likelihood errors of at most 1.3e-3 and TV of at most 3e-5. It has no
   guarantee: a per-site cutoff (`docs/likelihood_model.md`, later work)
   would make it principled.

## Strategies dropped from further exploration

On the constant-rate and map results together:

- **Floors above q** (including sc2ts's `0.99 mu^k`): wrong paths in every
  scenario, unboundedly so on tsinfer data. Under a map, any fixed floor is
  above q at most sites.
- **Floors in general**: below q they do nothing, above q they are
  harmful, and in forward-backward they add mass. `likelihood_threshold`
  stays only as an underflow guard.
- **`top_n`**: slow (it turns off sparse sites), inexact, no guarantee.
  Removed from the engine; relative pruning is now
  `Hmm(relative_prune_threshold=f)`.
- **`relative` with a fixed fraction**: harmful above q, and can't follow
  a map. Thresholds relative to q replace it.
- **Escalation**: 2–5× exact everywhere, and never certifies sample paths.
  Revisit only if the certificate is extended to switches.
- **`lossless_prune` at the smallest q**: does nothing under a map. Only a
  per-site version is worth keeping.

Kept: exact (the reference), `prune_k` / `path_threshold` (the only
guarantee, for sc2ts-style mismatch budgets), and pruning relative to q,
to be made per-site, mainly for forward-backward.

## Earlier experiments (`thresholding.py`, superseded)

These ran before the unified model, on simulated data and sc2ts sub-ARGs
only. Their Viterbi conclusions agree with the results above, and are made
precise there. The `lossless_floor` below is `lossless_prune`.

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

### Methods

Every method is applied after the values are normalised at each site:
by the maximum for Viterbi, and by the sum for forward/backward. A "pruned"
node is given the value `likelihood_threshold`, which by default is the
smallest positive double. It then merges into its parent, and can only be
reached again by recombination.

- **`floor`** is the existing `likelihood_threshold`: any value below `t` is
  set to `t`.
- **`relative`** prunes nodes with a value below `f` times the maximum. Use
  `relative_prune_threshold=f`.
- **`top_n`** prunes all but the `n` nodes with the largest values. Nodes tied
  with the n-th value are kept. (Since removed from the engine.)
- **`lossless_floor`** is `floor` at `t* = (ρ/n) / (1 − ρ + ρ/n)`. In Viterbi,
  any node below `t*` loses to recombination at the next site, so this floor
  changes no Viterbi likelihoods. With `weight_by_n=False`, as sc2ts uses,
  `t* = ρ`.

`Hmm(count_distinct=True)` also records the mean number of
**distinct** values per site in Viterbi (`mean_num_distinct_values`). This
is compared with the number of stored nodes (`mean_traceback_size`).

### Scenarios

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

### Findings (quick run)

These numbers come from `--quick`, which uses:
- 3 replicates of threading, with 50 samples only;
- 3 ARGs per calibration setting;
- 3 sc2ts queries on the 10k sub-ARG (an exact copy, and recombinants with 2
  and 5 extra mutations);
- a reduced grid: t and f in {1e-20, 1e-8, 1e-4, 1e-2}, and n in
  {10, 100, 1000}.

Treat them as indicative. The full grid, with 50–1000 samples and sc2ts
sub-ARGs up to 100k, was not run to completion.

#### 1. For Viterbi, compression limits storage more than thresholding does

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

#### 2. For Viterbi, a floor above `t*` is harmful, and pruning should be used instead

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

#### 3. For forward/backward, every method distorts the posteriors, in different directions

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

### Recommendations

- **Viterbi:** use the lossless floor `t*` as the default threshold; it
  costs nothing. If more speed is needed, use `relative` or `top_n` rather
  than a floor above `t*`.
- **Forward/backward:** treat any thresholding as approximate. A floor
  around 1e-4 gives large speedups on sc2ts with small posterior error.
  Check the posterior total-variation distance for the intended use.
- **Next step:** see `experiments/compression/`, which captures the exact
  per-site values and analyses alternative encodings.

### Exact prune thresholds (`exact_thresholds.py`)

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
