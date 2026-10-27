"""
The likelihood model, and the thresholds that follow from it. See
docs/likelihood_model.md for the derivations.

At site j the HMM recombines with probability r_j, and then copies a node
chosen uniformly from all n nodes of the tree sequence (the path ends if
that node isn't in the tree). Otherwise it stays on the same node. A node
copies the haplotype with probability 1 - (A_j - 1) mu_j for each of the
A_j alleles at the site, and mismatches with probability mu_j for each other
allele.

Viterbi only depends on two odds at each site, relative to staying on a
node that matches:

- the switch odds q_j = (r_j / n) / (1 - r_j + r_j / n), of switching to a
  given node rather than staying;
- the mismatch odds e_j = mu_j / (1 - (A_j - 1) mu_j), of mismatching rather
  than matching.

The Viterbi value of a node, normalised by the maximum, is the product of
the odds along its best path divided by the largest such product at the
site.
"""

import numpy as np

# The relative error allowed in the Viterbi values, which are renormalised
# at every site, when comparing them with a threshold.
THRESHOLD_TOLERANCE = 1e-9


def switch_odds(recombination, num_nodes):
    """
    Return the switch odds q of the specified recombination probability r:
    q = (r / n) / (1 - r + r / n), with n = num_nodes.
    """
    r = np.asarray(recombination, dtype=float)
    return (r / num_nodes) / (1 - r + r / num_nodes)


def recombination_probability(switch_odds, num_nodes):
    """
    Return the recombination probability r that gives the specified switch
    odds q, the inverse of switch_odds: r = q n / (1 - q + q n), with
    n = num_nodes. The odds must be between 0 and 1.

    As q n grows r approaches 1, and the engine computes 1 - r with a
    relative error of about 1e-16 / (1 - r).

    This is how to specify a model by the cost of a recombination, as sc2ts
    does: an engine that does not divide by n with recombination rho has the
    same Viterbi paths and forward-backward posteriors as this model with
    r = recombination_probability(rho, n).
    """
    q = np.asarray(switch_odds, dtype=float)
    if np.any(q < 0) or np.any(q > 1):
        raise ValueError("Switch odds must be between 0 and 1")
    return q * num_nodes / (1 - q + q * num_nodes)


def mismatch_odds(mismatch, num_alleles):
    """
    Return the mismatch odds e = mu / (1 - (A - 1) mu) of the specified
    mismatch probability mu, where A = num_alleles.
    """
    mu = np.asarray(mismatch, dtype=float)
    return mu / (1 - (np.asarray(num_alleles) - 1) * mu)


def _largest_product(odds, k):
    """
    Return the smallest product of k of the specified odds, the odds of the
    k most costly events. Scalar odds are the same at every site.
    """
    if np.ndim(odds) == 0:
        return float(odds) ** k
    if k > len(odds):
        raise ValueError("More events than sites")
    return float(np.prod(np.sort(odds)[:k]))


def path_threshold(
    num_mismatches,
    mismatch,
    num_alleles,
    *,
    num_switches=0,
    recombination=None,
    num_nodes=None,
    tolerance=THRESHOLD_TOLERANCE,
):
    """
    Return the largest Viterbi prune threshold that is guaranteed to find
    an optimal path, whenever an optimal path has at most the specified
    numbers of mismatches and switches.

    A path's odds are the product of the odds of its mismatches and switches.
    A node whose normalised value falls below t has odds less than t times
    those of the best node at the site, and so does any path through it. The
    threshold is the smallest product of the odds of num_mismatches
    mismatches and num_switches switches at any sites (with constant rates,
    e^k q^s), reduced by tolerance for rounding.

    The returned path is optimal if its own odds are at least the threshold.
    If they are not, the run should be repeated with a lower threshold.

    The mismatch and recombination probabilities, and the number of
    alleles, can be scalars or per-site arrays.
    """
    if num_mismatches < 0 or num_switches < 0:
        raise ValueError("Numbers of mismatches and switches must be >= 0")
    t = _largest_product(mismatch_odds(mismatch, num_alleles), num_mismatches)
    if num_switches > 0:
        if recombination is None or num_nodes is None:
            raise ValueError("Switches need the recombination and num_nodes")
        t *= _largest_product(switch_odds(recombination, num_nodes), num_switches)
    return t * (1 - tolerance)


def lossless_threshold(recombination, num_nodes, *, tolerance=THRESHOLD_TOLERANCE):
    """
    Return the largest Viterbi prune threshold that changes no Viterbi
    values at later sites: below the smallest switch odds q, a node loses to
    recombination at the next site whatever its value. Reduced by tolerance
    for rounding.
    """
    return float(np.min(switch_odds(recombination, num_nodes))) * (1 - tolerance)
