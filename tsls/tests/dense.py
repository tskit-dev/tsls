"""
A dense implementation of the Li and Stephens HMM over all the nodes of a
tree sequence, used as a reference for the tree-based engine.

The states at a site are the nodes in the tree at that site (nodes with a
parent or at least one child). We can only continue copying a node from one
site to the next if it is in the tree over the whole interval between them
(in every tree from the tree of the previous site to the tree of this one);
otherwise reaching it again requires a recombination. Nodes entering the
tree have probability zero, and so can only be reached by recombination.
Recombination redistributes the total over all states at the previous site,
including nodes that are no longer in the tree.
"""

import numpy as np


def node_states(ts, alleles=None):
    """
    Return the (A, present, continuous) tuple of (num_sites, num_nodes)
    arrays, where A[j, u] is the allele index of node u at site j,
    present[j, u] is True if node u is in the tree at site j, and
    continuous[j, u] is True if node u is in every tree between site j - 1
    and site j (continuous[0] = present[0]). If alleles is None, the
    ancestral state is 0 and every derived state is 1.
    """
    A = np.zeros((ts.num_sites, ts.num_nodes), dtype=np.int8)
    present = np.zeros((ts.num_sites, ts.num_nodes), dtype=bool)
    tree_index = np.zeros(ts.num_sites, dtype=int)
    in_tree = np.zeros((ts.num_trees, ts.num_nodes), dtype=bool)

    def encode(state, default):
        return default if alleles is None else alleles.index(state)

    for tree in ts.trees():
        for u in range(ts.num_nodes):
            in_tree[tree.index, u] = tree.parent(u) != -1 or tree.num_children(u) > 0
        for site in tree.sites():
            state = {}
            for mutation in site.mutations:
                state[mutation.node] = encode(mutation.derived_state, 1)
            ancestral = encode(site.ancestral_state, 0)
            for u in range(ts.num_nodes):
                v = u
                while v != -1 and v not in state:
                    v = tree.parent(v)
                A[site.id, u] = ancestral if v == -1 else state[v]
            present[site.id] = in_tree[tree.index]
            tree_index[site.id] = tree.index
    continuous = present.copy()
    for j in range(1, ts.num_sites):
        trees = in_tree[tree_index[j - 1] : tree_index[j] + 1]
        continuous[j] = np.all(trees, axis=0)
    return A, present, continuous


def emission(A_site, h_site, mu, num_alleles):
    match = (A_site == h_site) | (h_site == -1)
    return np.where(match, 1 - (num_alleles - 1) * mu, mu)


def forward(A, present, continuous, h, rho, mu, num_alleles, n, threshold=0):
    """
    Return the (F, c) tuple of normalised forward probabilities and
    normalisation factors. We start from the uniform distribution over the
    n nodes, of which the nodes not in the tree at the first site are lost.
    """
    m, N = A.shape
    F = np.zeros((m, N))
    c = np.zeros(m)
    f = present[0] / n
    for j in range(m):
        total = np.sum(f)
        f = np.where(continuous[j], f, 0)
        f = (f * (1 - rho[j]) + rho[j] / n * total) * emission(
            A[j], h[j], mu[j], num_alleles
        )
        f = np.where(present[j], f, 0)
        c[j] = np.sum(f)
        f = np.where(present[j], np.maximum(f / c[j], threshold), 0)
        F[j] = f
    return F, c


def backward(A, present, continuous, h, rho, mu, num_alleles, n, c, threshold=0):
    """
    Return the backward probabilities, scaled by the forward normalisation
    factors c.
    """
    m, N = A.shape
    B = np.zeros((m, N))
    b = present[m - 1].astype(float)
    total = 0
    for j in range(m - 1, -1, -1):
        if j < m - 1:
            b = np.where(continuous[j + 1], b, 0)
            b = ((1 - rho[j + 1]) * b + rho[j + 1] * total / n) / c[j + 1]
            b = np.where(present[j], np.maximum(b, threshold), 0)
        B[j] = b
        b = np.where(present[j], b * emission(A[j], h[j], mu[j], num_alleles), 0)
        total = np.sum(b)
    return B


def viterbi_matrix(A, present, continuous, h, rho, mu, num_alleles, n):
    """
    Return the (V, c) tuple of Viterbi values normalised by the maximum at
    each site, and the maximum values c, where staying on a node has
    probability (1 - rho + rho / n) and switching to a given node has
    probability rho / n.
    """
    m, N = A.shape
    V = np.zeros((m, N))
    c = np.zeros(m)
    v = present[0].astype(float)
    for j in range(m):
        v = np.where(continuous[j], v, 0)
        v = np.maximum(v * (1 - rho[j] + rho[j] / n), rho[j] / n)
        v = np.where(present[j], v * emission(A[j], h[j], mu[j], num_alleles), 0)
        c[j] = np.max(v)
        v = v / c[j]
        V[j] = v
    return V, c


def viterbi_log_likelihood(A, present, continuous, h, rho, mu, num_alleles, n):
    """
    Return the log likelihood of the most likely path.
    """
    _, c = viterbi_matrix(A, present, continuous, h, rho, mu, num_alleles, n)
    return np.sum(np.log(c))


def path_log_likelihood(A, present, continuous, h, rho, mu, num_alleles, n, path):
    """
    Return the log likelihood of the specified path (the node copied at
    each site) under the same model as viterbi_log_likelihood. Copying the
    same node at consecutive sites counts as a switch if the node is not in
    the tree over the whole interval between them.
    """
    log_likelihood = 0
    for j, u in enumerate(path):
        assert present[j, u]
        if j == 0 or (path[j - 1] == u and continuous[j, u]):
            p_t = 1 - rho[j] + rho[j] / n
        else:
            p_t = rho[j] / n
        e = emission(A[j, u], h[j], mu[j], num_alleles)
        log_likelihood += np.log(p_t * e)
    return log_likelihood
