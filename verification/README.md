# Statistical validation of tsls

`verification.py` runs empirical validations of the tsls HMM engine on
ARGs simulated with msprime, and writes plots and CSV files. It is
modelled on msprime's `verification.py`.

These are not unit tests. They don't make hard assertions, and they are
not run in CI. They are meant to give you a feel for how the engine
behaves statistically, and to catch problems that only show up on
realistic ARGs.

```
uv sync
uv run python verification.py --list          # list the tests
uv run python verification.py -t 4            # run everything on 4 processes
uv run python verification.py test_calibration_default
uv run python verification.py -c HeldOutThreading
```

Output goes to `tmp__NOBACKUP__/<class>/<test>/` (`-d` changes the base
directory). Each test takes a few minutes on one core.

## Calibration under the model (`ModelCalibration`)

These tests simulate queries from the HMM itself:

1. Sample a copying path exactly from the engine's prior over paths on an
   msprime ARG. This uses a backward pass with no data, then forward
   sampling, so it allows for nodes entering and leaving the tree.
2. Add mismatches using the engine's emission model.

Because the data come from the model, the forward-backward posteriors
should be calibrated.

- **`reliability.png`**: (site, node) pairs are binned by posterior
  probability. The plot shows the fraction in each bin where the node is
  the true one, with 95% intervals. The points should lie on y = x.
- **`credible_sets.png`**: the smallest set of nodes reaching each
  posterior level should contain the true node at about that rate. Set
  sizes show how uncertain the copying target is. Nodes with identical
  haplotypes over a region can't be told apart.
- **`decoding.png`** (from the `decoding_accuracy` tests): per-site accuracy
  of Viterbi and posterior-max decoding, the mean posterior of the true
  node, and Viterbi against true switch counts, as μ or ρ varies.

With `weight_by_n=False` (rho given as the recombination of an engine that
does not divide by n), the prior over paths is a constant times that of
the model tsls runs, with `r = tsls.model.recombination_probability(rho, n)`
(see `docs/likelihood_model.md`). We sample from it exactly, so the
posteriors should still be calibrated.

## Held-out sample threading (`HeldOutThreading`)

A sample is held out of an msprime simulation, and its haplotype is
matched against the ARG of the other samples. The "truth" at each site is
the held-out sample's parent in the full ARG: the node where its lineage
joins the others, which is the best node to copy. These tests need no
assumptions about the HMM. They show how well the copying path tracks the
true genealogy as the mutation rate, recombination rate, sample size and
HMM ρ vary. The HMM uses sc2ts-style parameters (`k` mismatches cost
about one recombination, with `weight_by_n=False`).

- **`threading.png`** shows four panels:
  - the fraction of sites where the Viterbi node is the parent, or within
    2 edges of it;
  - the posterior mass on the parent;
  - the time error of the Viterbi node;
  - the number of switches, true and Viterbi.
- **`breakpoints.png`** shows the distance from each true switch of the
  parent to the nearest Viterbi switch, and true against Viterbi switch
  counts.

## Numerics and approximations (`Numerics`)

- **`test_likelihood_threshold`** shows what the likelihood threshold costs
  and gains: the loss in Viterbi path likelihood, the distance of the
  posteriors from the exact ones, traceback size, time and threading
  accuracy.
- **`test_posterior_normalisation`** shows Σ_u F·B, which should be 1 at
  every site, and the forward log-likelihood per site, along sequences
  with up to about 10⁴ sites.
- **`test_dense_agreement_random_args`** compares the engine with the
  dense reference (`tsls/tests/dense.py`) on many small random ARGs with
  random parameters. Errors should be around machine precision.
- **`test_weight_by_n`** compares threading with `weight_by_n` on and off,
  with the same per-node switching ratio.
