# The likelihood model

This document defines the HMM that tsls computes, shows that sc2ts's
"k mismatches per recombination" setting is a choice of parameters in this
model, and derives exact Viterbi thresholds from it. The functions
`tsls.switch_odds`, `recombination_probability`, `mismatch_odds`,
`mismatch_probability` and `path_threshold` implement the formulas, and
`tests/test_model.py` checks them.

## The model

The states at site j are the nodes in the local tree T_j: the nodes with a
parent or at least one child. Let n be the number of nodes in the tree
sequence, r_j the recombination probability and mu_j the mismatch
probability at site j, and A_j the number of alleles at the site. If the
alleles are given (`tsls.TreeSequence(ts, alleles=...)`), A_j is the length
of site j's list, whether or not the alleles are all in the tree sequence
there. Otherwise A_j is the number of distinct states at the site in the
tree sequence, and at least 2.

**Start.** The path starts at a node chosen uniformly from all n nodes. The
transition below is applied at the first site too, so each node in T_0
starts with probability `(1 - r_0 + r_0 P_0 / n) / n`, where P_0 is the
number of nodes in T_0. This is the same for every node, so it doesn't
affect Viterbi paths or posteriors, but it makes the likelihood depend on
which end of the sequence the path starts from.

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

`tsls.switch_odds(r, n)` and `tsls.recombination_probability(q, n)`
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
copying targets alive: on the larger tsinfer chr20p benchmarks (ancestor
group 1485 and sample matching), floors of 1e-4 and
1e-2 missed the optimal path for every query. The floor is
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
`tsls.path_threshold(k, mu, num_alleles, num_switches=s, ...)`.

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

**Rounding.** Viterbi works with costs (below), which are sums of the
per-site switch and mismatch costs. A cost of C carries an absolute error of
a few ulps of C per site: on a 580,000-site chr20p sample C is about 1e4, so
the error in a value relative to the best is about 1e-12 per operation.
`path_threshold` multiplies by `1 - 1e-9` (`THRESHOLD_TOLERANCE`) to absorb
this.

**Free pruning.** A node with value below `q_{j+1}` is overtaken at site
j + 1 by recombination from the best node, whatever its value. So pruning
below `min_j q_j` changes no values at later sites
(`tsls.lossless_threshold`). The k-mismatch threshold is useful when
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

## Deferred Viterbi costs

The engine stores Viterbi values as costs, not as renormalised odds. Write
c_s(j) = -log q_j and c_m(j) = -log e_j. The cost of node u at site j is

    C_j(u) = min(C_{j-1}(u), ceiling_j) + [u mismatches at j] c_m(j),
    ceiling_j = C*_{j-1} + c_s(j),

where C* is the smallest cost, the best node's. Then V_j(u) = exp(-(C_j(u) -
C*_j)). Costs are absolute, so nothing is renormalised. The flag that says
the best path into u at j switches is C_{j-1}(u) >= ceiling_j, the same tie
rule as `p_no_recomb <= p_recomb`. A node whose normalised value is below
the prune threshold t costs more than C* - log t, and gets the cost of
likelihood_threshold. The floor caps every cost at C* - log(likelihood_threshold).

**Deferred state.** A node that matches only changes if the ceiling falls
below its cost. So each listed node keeps a state (C, s): the cost it last
changed to, at site s. Its cost now is C lowered by the smallest ceiling
since s, and a stack of the increasing ceilings back from the last site
answers that by binary search. Only changes other than matching are
recorded: a mismatch, a prune, the cap, and copying the state of the
ancestor a node inherits from. This holds for any per-site rates.

**Flags in the traceback.** The recombination flags are not stored. Each
site stores its ceiling, and the last earlier site with a smaller one
(pse_j). Node u's cost after site j - 1, given its state (C, s), is at least
the ceiling iff `C >= ceiling_j and pse_j <= s`. The traceback undoes each
site's changes to the states, moving right to left, and reads the flag of
the path's representative this way.

**Sparse sites.** If the best node matches at a site, the best cost is
unchanged: its own cost is at most the ceiling, and the other costs can only
rise. So only the listed nodes that mismatch need touching. In a tree with
enough sites left (`index_min_sites`, 16 by default), the engine numbers the
part of the tree the sites need in preorder. The listed nodes that
mismatch are then the listed nodes in the preorder ranges of the site's
mutations whose alleles differ from the query's. If the best node
mismatches, the site is processed in full. There is no condition on the
rates, so maps don't stop sites being sparse.

**Dead costs.** Let D_j = ceiling_{j+1}. A node that costs at least D_j at j,
whether it mismatches there or not, costs D_j plus its emission at j + 1,
and has the flag set. So mismatching such a node changes nothing from then
on, and nodes that both cost at least D_j are equivalent when coalescing.
The engine uses this unless values are output (`viterbi_matrix`) or
counted, or `sparse=False`.

Pruning only lowers values. A node's value relative to the best can only
fall when it is touched, so the prune is applied only to touched nodes, and
`prune_threshold` and `relative_prune_threshold` both keep sparse sites (in
Viterbi they are the same prune).
