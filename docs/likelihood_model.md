# The likelihood model

This document defines the HMM that tsls computes, shows that sc2ts's
"k mismatches per recombination" setting is a choice of parameters in this
model, and derives exact Viterbi thresholds from it. The functions in
`tsls.model` implement the formulas, and `tsls/tests/test_model.py` checks
them.

## The model

The states at site j are the nodes in the local tree T_j: the nodes with a
parent or at least one child. Let n be the number of nodes in the tree
sequence, r_j the recombination probability and mu_j the mismatch
probability at site j, and A_j the number of alleles at the site.

**Start.** The path starts at a node chosen uniformly from all n nodes.

**Transition** into site j. With probability r_j the path recombines and
copies a node chosen uniformly from all n nodes. Otherwise it stays on the
same node. The path ends (it is *killed*) if the node it copies is not in
T_j. A node that is not in the tree over the whole interval since the
previous site counts as absent: it can't be stayed on, only reached by
recombination.

    T_j(u -> v) = (1 - r_j) [u = v, v continuous] + r_j / n,   v in T_j

**Emission.** The node's allele is copied with probability
`1 - (A_j - 1) mu_j`, and each other allele with probability mu_j. Missing
data has probability 1.

The rows of T_j sum to at most 1, so this is a sub-stochastic (killed)
Markov chain. It is a proper probability model. The forward algorithm
computes `P(h, path survives)`, and the posteriors are conditional on the
path surviving. This is the model the engine has always used with
`weight_by_n=True`. The only change is that the forward algorithm now starts
from 1/n per node rather than 1, so the log likelihood is the probability
above.

## Viterbi depends only on two odds

At each site every path picks up one transition factor and one emission
factor:

| | factor | relative to the common case |
|---|---|---|
| stay | `s_j = 1 - r_j + r_j / n` | 1 |
| switch to a given node | `r_j / n` | `q_j = (r_j / n) / (1 - r_j + r_j / n)` |
| match | `m_j = 1 - (A_j - 1) mu_j` | 1 |
| mismatch | `mu_j` | `e_j = mu_j / (1 - (A_j - 1) mu_j)` |

(Staying applies when the stay factor beats the switch factor, which is
true whenever `q_j <= 1`.) Every path at site j is multiplied by
`s_j m_j` times its odds, so dividing by the maximum at each site cancels
`s_j m_j`. Define the **cost** of a path as

    C(path) = (number of switches) * (-log q) + (number of mismatches) * (-log e)

(with per-site q_j and e_j if the rates vary). Then the normalised Viterbi
value of node u at site j is exactly

    V_j(u) = exp(-(C_j(u) - min_v C_j(v))),

where C_j(u) is the cost of the best path that is at u at site j
(`test_viterbi_values_are_odds`). Viterbi needs only q_j and e_j. Forward
and backward also need r_j itself, because they sum over paths.

`tsls.model.switch_odds(r, n)` and `tsls.model.recombination_probability(q, n)`
convert between r and q:

    q = (r / n) / (1 - r + r / n),        r = q n / (1 - q + q n).

## sc2ts: specifying a recombination by its cost

sc2ts wants one recombination to cost about k mismatches. That fixes the
switch odds `q ≈ e^k`, which is a statement about odds, not about the
probability of recombining. sc2ts used an engine that doesn't divide by n
(`weight_by_n=False`). Its transition is `(1 - rho) [u = v] + rho`, so
staying has factor 1 and switching has factor rho, and **q = rho**.

That transition is a constant times the transition of the model above with
`r = recombination_probability(rho, n)`. This holds for every pair of
states, including nodes that leave or enter the tree:

    (1 - rho) [u = v] + rho  =  (1 - rho) / (1 - r) * ((1 - r) [u = v] + r / n).

Proof: the two sides have the same ratio of switching to staying,
`rho / (1 - rho) = (r / n) / (1 - r)`. So the old engine with rho and the
model with r (`TestModelEquivalence`) have:

- the same Viterbi values and paths;
- the same normalised forward and backward values, and so the same
  posteriors;
- normalisation factors that differ by `(1 - rho) / (1 - r)` at each site
  (and n at the first site). So only the old *log likelihood* was wrong: it
  was not the probability of anything.

So nothing about the sc2ts model needs a separate code path. Pass
`r = recombination_probability(rho, n)`. What the model makes explicit is what
the sc2ts setting means as a probability model. Fixing the cost of switching
to *each* of n nodes means the total recombination probability grows with
n. With sc2ts's parameters (k = 3, mu = 0.0125, 5 alleles, rho = 2.51e-6):

| n | r per site | (1 - rho) / (1 - r) |
|---|---|---|
| 20,000 | 0.048 | 1.05 |
| 100,000 | 0.20 | 1.25 |
| 3,400,000 | 0.89 | 9.5 |

That is why forward-backward with sc2ts parameters gives diffuse posteriors.
They are the correct posteriors of a model in which recombination is
frequent. No model in this family has both the sc2ts switch cost and a
small r: in any of them `r / n ≈ q (1 - r)`. For posteriors with a
plausible recombination probability, run forward-backward with that r. In
Viterbi, this charges a switch about `log(n / r)` in log odds, which is more
than k mismatches.

Since r depends on n, the r for an sc2ts-style model must be computed from
the tree sequence that is being matched against. As r approaches 1 (q n much
larger than 1), the engine computes `1 - r` with relative error about
`1e-16 / (1 - r)`. This is about 1e-15 at n = 3.4M.

## Exact Viterbi thresholds

**Pruning, not flooring.** `Hmm(prune_threshold=t)` gives every node whose
normalised Viterbi value is below t the value `likelihood_threshold`
(effectively 0). From then on it can only be reached by recombination.
Pruning only lowers values. The old `likelihood_threshold` *floor* raises
values instead. Above q, a raised node beats recombination and keeps stale
copying targets alive (`experiments/thresholding`, finding 2). The floor is
now only there to prevent underflow, and should stay below q.

**Theorem.** Let P* be an optimal path, with cost C*. If `t <= exp(-C*)`,
Viterbi with prune threshold t returns an optimal path.

*Proof.* Costs are non-negative (q, e <= 1). At every site j, the prefix of
P* has cost `C*_j <= C*`, and the best node at j has cost at least 0. So the
node P* occupies has value `V_j >= exp(-C*_j) >= exp(-C*) >= t`, and is never
pruned. (Pruning other nodes can only raise the best cost at a site, which only
raises the normalised value of P*.) Every path's value in the pruned run is at most
its value without pruning, and P* keeps its value. So the path returned has
unpruned value at least that of P*, and is optimal.

**Thresholds for k mismatches.** The cost of k mismatches and s switches is
at most the sum of the k largest mismatch costs and the s largest switch
costs over the sites. So

    t(k, s) = (product of the k smallest e_j) * (product of the s smallest q_j)

finds the optimal path whenever it has at most k mismatches and s switches.
With constant rates this is `e^k q^s`. This is
`tsls.model.path_threshold(k, mu, num_alleles, num_switches=s, ...)`.

**It is the largest such threshold.** If an optimal path has its k
mismatches at the first k sites, at which another node matches, then it is
at exactly `e^k` after site k. Any larger threshold prunes it
(`test_threshold_is_tight`).

**Certificate.** Run with any threshold t. If the returned path's odds are
at least t (`e^(mismatches) q^(switches) >= t`), it is optimal: an optimal
path has odds at least as large, so it was never pruned. Otherwise, rerun
with a smaller t. This makes sc2ts's escalation (`mismatch_threshold` = 1,
then larger) exact. The returned path can be checked without knowing the
optimum (`test_certificate`).

**Rounding.** Values are renormalised at every site, so they carry a relative
error of a few ulps per site, about 1e-11 over 30,000 sites.
`path_threshold` multiplies by `1 - 1e-9` (`THRESHOLD_TOLERANCE`) to absorb
this.

**Free pruning.** A node with value below `q_{j+1}` is overtaken at site
j + 1 by recombination from the best node, whatever its value. So pruning
below `min_j q_j` changes no values at later sites
(`tsls.model.lossless_threshold`). The k-mismatch threshold is useful when
it is above this, that is, when k mismatches cost less than a switch.

### Worked example: sc2ts parameters

With mu = 0.0125 and 5 alleles, `e = 0.0125 / 0.95 = 0.01316`:

| k | t(k, 0) = e^k (1 - 1e-9) | sc2ts's floor 0.99 mu^k |
|---|---|---|
| 0 | 1.0 | |
| 1 | 1.316e-2 | 1.238e-2 |
| 2 | 1.731e-4 | 1.547e-4 |
| 3 | 2.278e-6 | 1.934e-6 |

sc2ts's floor was 6–15% below the exact value, so it kept more nodes than
needed. It was also a floor rather than a prune. At k = 3, `0.99 mu^3` is below
q = 2.51e-6, so it had no effect at all. At k = 1 and 2 it raised weak nodes
above q, so its result was not guaranteed to be optimal even when the
returned path was within budget.

## Pruning in sparse Viterbi sites

The prune is applied as each value is normalised, in both the full and the
sparse site updates. This is the same function of the value at every site
with the same rates, which is what sparse sites rely on: a node unchanged
at the previous site, and that matches at both, is unchanged. A pruned node
gets a new value, and so is recomputed at the next site. So, unlike the
experimental `pruning` option, `prune_threshold` keeps sparse sites.

## Later work

- **Recombination maps.** Per-site r_j already works. The thresholds use the
  extreme q_j and e_j over the sites, which is conservative when they vary
  a lot. A per-site cutoff (for example `max(t, q_{j+1})`) is the natural
  extension.
- **Forward-backward thresholds.** Pruning a node removes its mass, so the
  forward value is a lower bound on the likelihood. The mass removed at each
  site bounds the error, which gives a route to thresholds with guaranteed
  error.
