# Coarsening experiments

Under a recombination map, Viterbi on chr20p sample matching stores about
8× the traceback of tsinfer's constant rate, and takes about 2× as long
(`benchmarking/README.md`, "Recombination maps"). Can we get that back by
coarsening, and what does it cost in accuracy? `coarsening.py` tries two
kinds of coarsening:

- **Input:** coarsen the per-site recombination probabilities r before
  running the HMM.
- **Values:** round each normalised Viterbi value to the nearest
  exp(−k d), so that values within a factor of about e^d become equal and
  can be coalesced. With a prune threshold t there are at most about
  ln(1/t)/d distinct values. This used an experimental
  `Hmm(log_value_step=d)` option, which didn't pay off and has been
  removed from the engine; `coarsening.py` now runs only the input
  variants. To rerun the value rounding, check out commit `69af95b`
  (which added the option) and restore the value-step strategies from
  `b3d970f`.

```
uv sync
uv run python coarsening.py -t 2         # about 4 minutes on 4 cores
uv run python coarsening.py --analyse-only
```

It needs the chr20p samples workload (`benchmarking/tsinfer_data.py`) and
the HapMap II GRCh38 chr20 map (`benchmarking/genetic_map.py`). Results
go to `results/` (not tracked). The numbers below are from one run
(2026-10-28): 6 samples, at mu = 1e-20 (tsinfer's default) and 1e-8, with
2 tasks running at once.

## Setup

| variant | what it is | distinct r |
|---|---|---|
| map | the HapMap II map, Haldane's map function (the reference) | 476,419 |
| constant | tsinfer's r = 0.01 at every site | 1 |
| constant_mean | the map's mean r (8.75e-7) at every site | 1 |
| uniform | a constant rate per bp, with the map's genetic length (r varies with site spacing) | 620 |
| smooth_100kb, smooth_1000kb | the map averaged over 100 kb or 1 Mb windows | 75,298, 13,561 |
| round_2, round_10 | each r rounded to the nearest power of 2 or 10 | 28, 10 |

Every path is scored under the map: the **loss** is the map's optimal log
odds minus the path's, so 0 is optimal. For scale, one switch costs about
29 (ln 1/q at the median q), and a factor of 2 error in one site's rate
changes a switch's cost by 0.7. "Breakpoints kept" is the fraction of the
map-optimal switch sites that the path also switches at.

## Results

Means over 6 samples; ratios are to the exact run on the map.

| variant | mu | time | traceback | loss (mean / max) | breakpoints kept |
|---|---|---|---|---|---|
| map (exact) | 1e-20 | 3.7 s | 92 | 0 | 1 |
| | 1e-8 | 4.4 s | 107 | 0 | 1 |
| constant | 1e-20 | 0.50 | 0.13 | 911 / 1,590 | 0.03 |
| | 1e-8 | 0.44 | 0.13 | 917 / 1,480 | 0.04 |
| constant_mean | 1e-20 | 0.49 | 0.15 | 623 / 979 | 0.04 |
| | 1e-8 | 0.46 | 0.21 | 416 / 684 | 0.04 |
| uniform | 1e-20 | 1.05 | 0.95 | 236 / 316 | 0.43 |
| | 1e-8 | 0.94 | 0.95 | 150 / 207 | 0.48 |
| smooth_100kb | 1e-20 | 0.98 | 0.95 | 230 / 311 | 0.43 |
| | 1e-8 | 0.94 | 0.95 | 148 / 201 | 0.48 |
| smooth_1000kb | 1e-20 | 1.04 | 0.95 | 236 / 320 | 0.43 |
| | 1e-8 | 0.93 | 0.95 | 150 / 210 | 0.48 |
| round_2 | 1e-20 | 0.90 | 0.82 | 21 / 35 | 0.56 |
| | 1e-8 | 0.98 | 1.00 | 14 / 22 | 0.61 |
| round_10 | 1e-20 | 0.77 | 0.59 | 104 / 168 | 0.28 |
| | 1e-8 | 0.85 | 0.82 | 70 / 111 | 0.30 |
| value step 0.1 | 1e-20 | 1.34 | 0.97 | 0.19 / 0.37 | 0.96 |
| | 1e-8 | 1.22 | 0.97 | 0.10 / 0.22 | 0.96 |
| value step 0.5 | 1e-20 | 1.20 | 0.87 | 2.7 / 4.2 | 0.83 |
| | 1e-8 | 1.14 | 0.88 | 2.0 / 3.5 | 0.84 |
| value step 1 | 1e-20 | 1.12 | 0.77 | 13 / 19 | 0.67 |
| | 1e-8 | 1.04 | 0.78 | 7.7 / 11 | 0.70 |
| value step ln 10 | 1e-20 | 1.02 | 0.59 | 55 / 78 | 0.42 |
| | 1e-8 | 0.90 | 0.62 | 33 / 48 | 0.45 |
| round_10 + value step ln 10 | 1e-20 | 0.94 | 0.59 | 104 / 169 | 0.28 |
| | 1e-8 | 0.90 | 0.62 | 71 / 123 | 0.29 |

The numbers of switches and mismatches hardly change, except with the
constant rates (which have more switches, or fewer mismatches).

## Findings

1. **The map's cost comes from r varying between sites, not from its
   scale.** A constant r at the map's mean is as cheap as tsinfer's
   constant 0.01 (time 0.46–0.49, traceback 0.15–0.21), although its switch
   odds are about 10⁵ times smaller. Any variation brings the cost back,
   and it saturates quickly with the number of distinct values of r: 10
   (round_10) gives 0.6–0.8 of the map's traceback, 28 (round_2) 0.8–1.0,
   and the 620 values of a uniform rate per bp, which come only from site
   spacing, 0.95. Values relative to the maximum are ratios of the odds of
   different switch sites, so with variable q they rarely coincide.
2. **Smoothing the map spatially doesn't help.** Averaging over 100 kb or
   1 Mb windows costs as much as the map and loses as much accuracy as a
   uniform rate (loss 150–240, half the breakpoints moved). The
   variation that matters for cost is between adjacent sites, which
   smoothing keeps; the variation that matters for the paths is in the
   fine-scale hotspots, which smoothing removes.
3. **Rounding r to a factor of 2 is within the map's false precision.** It
   costs a loss of 14–21 over 120–180 switches, about 0.1 per switch,
   less than the effect of a factor of 2 error in one site's rate. But it
   saves only 0–18% of the traceback and 2–10% of the time. Rounding to a
   factor of 10 saves 15–23% of time and 18–41% of traceback, at about 0.6
   per switch.
4. **Rounding the values directly is the more accurate trade, but doesn't
   save time.** For the same traceback (0.59), the value step ln 10 loses
   about half as much as round_10 (33–55 against 70–104). Small steps are
   nearly free in accuracy (0.1: loss under 0.4). But the extra log and
   exp per value make every step except ln 10 slower than exact, and
   combining the two roundings adds nothing to round_10 alone.
5. **"Breakpoints kept" overstates the change.** round_2 keeps only 56–61%
   of the switch sites with a loss of 0.1 per switch: many sites near a
   switch are nearly equally likely, and small changes move it among
   them.

Neither kind of coarsening gets back more than about a quarter of the
time the map costs, and the gains come with losses of tens to hundreds in
log odds. The traceback is reduced more than the time, which follows the
number of values the algorithm updates.
