"""
The Li and Stephens HMM on a tree sequence.
"""

import numpy as np

import _tsls


class TreeSequence(_tsls.TreeSequence):
    """
    A tskit tree sequence in the form used by the HMM, with its alleles
    encoded as integers. It can be shared between HMMs, including HMMs in
    different threads, so that it is only built once.

    By default the alleles are numbered at each site as tskit does: the
    ancestral state is 0 and derived states are numbered from 1 in order of
    their first appearance at the site. Every site has at least two alleles,
    so that 1 can be matched at a site with no mutations (as in tsinfer, for
    an ancestor's focal sites). If ``alleles`` is specified, every state is
    instead its index in this list.
    """

    def __init__(self, ts, *, alleles=None):
        tables = ts.dump_tables()
        ll_tables = _tsls.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        if alleles is not None:
            alleles = list(alleles)
            if len(set(alleles)) != len(alleles):
                raise ValueError("Alleles must be unique")
        self.ts = ts
        self.alleles = alleles
        super().__init__(ll_tables, alleles=alleles)


class CompressedMatrix:
    """
    The output of the Viterbi, forward or backward algorithm. At each site we
    store a list of (node, value) pairs, and the value for a node in the tree
    is the value of its nearest ancestor (including itself) in this list.

    For Viterbi, the values are the normalised likelihoods stored in the
    traceback at each site (before compression), the normalisation factor is
    the maximum likelihood before normalisation, and ``flags`` gives the
    recombination_required flag of each stored node. For forward and backward
    ``flags`` is None.
    """

    def __init__(self, ts, normalisation_factor, offsets, nodes, values, flags=None):
        self.ts = ts
        self.normalisation_factor = normalisation_factor
        self.offsets = offsets
        self.nodes = nodes
        self.values = values
        self.flags = flags

    def dump(self, path):
        """
        Save the matrix to the specified .npz file, which can be loaded with
        load_compressed_matrix given the same tree sequence.
        """
        arrays = {
            "normalisation_factor": self.normalisation_factor,
            "offsets": self.offsets,
            "nodes": self.nodes,
            "values": self.values,
            "sites_position": self.ts.sites_position,
            "num_nodes": np.array(self.ts.num_nodes),
        }
        if self.flags is not None:
            arrays["flags"] = self.flags
        np.savez_compressed(path, **arrays)

    @property
    def num_sites(self):
        return len(self.normalisation_factor)

    @property
    def log_likelihood(self):
        """
        The log likelihood of the haplotype (for the forward matrix).
        """
        return float(np.sum(np.log(self.normalisation_factor)))

    def get_site(self, site):
        """
        Return the (nodes, values) arrays stored for the specified site.
        """
        start = self.offsets[site]
        stop = self.offsets[site + 1]
        return self.nodes[start:stop], self.values[start:stop]

    def decode(self, nodes=None):
        """
        Return the dense (num_sites, len(nodes)) array of values for the
        specified nodes (default: all nodes). Nodes that are not in the tree
        at a site have value 0.
        """
        N = self.ts.num_nodes
        if nodes is None:
            nodes = np.arange(N)
        nodes = np.asarray(nodes, dtype=np.int32)
        D = np.zeros((self.num_sites, len(nodes)))
        for tree in self.ts.trees():
            # The parent array has an extra entry for the virtual root, which
            # is the parent of the roots (index -1) and never has a value.
            parent = tree.parent_array
            in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
            for site in tree.sites():
                site_nodes, site_values = self.get_site(site.id)
                value = np.full(N + 1, np.nan)
                value[site_nodes] = site_values
                # Each node takes the value of its nearest ancestor with a
                # value: at each step, copy values from the ancestor and move
                # up a level, so this takes as many steps as the tree is deep.
                ancestor = parent.copy()
                unset = np.where(in_tree & np.isnan(value[:N]))[0]
                while len(unset) > 0:
                    if np.any(ancestor[unset] == -1):
                        raise ValueError("Node in tree with no stored ancestor")
                    value[unset] = value[ancestor[unset]]
                    ancestor[unset] = parent[ancestor[unset]]
                    unset = unset[np.isnan(value[unset])]
                D[site.id] = np.where(in_tree[nodes], value[nodes], 0)
        return D


def load_compressed_matrix(path, ts):
    """
    Load a CompressedMatrix saved with CompressedMatrix.dump for the specified
    tree sequence.
    """
    with np.load(path) as data:
        if int(data["num_nodes"]) != ts.num_nodes or not np.array_equal(
            data["sites_position"], ts.sites_position
        ):
            raise ValueError("Compressed matrix does not match the tree sequence")
        return CompressedMatrix(
            ts,
            data["normalisation_factor"],
            data["offsets"],
            data["nodes"],
            data["values"],
            data["flags"] if "flags" in data else None,
        )


class Hmm(_tsls.Hmm):
    """
    Runs the Viterbi, forward and backward algorithms for haplotypes against
    the specified tree sequence (a tskit.TreeSequence, or a tsls.TreeSequence
    to share one between HMMs), using the specified per-site recombination
    and mismatch probabilities. ``alleles`` is passed to tsls.TreeSequence,
    and can only be given with a tskit.TreeSequence.

    At each site we recombine with the recombination probability r, and then
    copy a node chosen uniformly from all the nodes of the tree sequence.
    See docs/likelihood_model.md, and tsls.model for the conversions
    between probabilities and the odds that Viterbi depends on. To specify
    a recombination by its cost in mismatches, as sc2ts does, use
    ``tsls.model.recombination_probability``.

    The states of the HMM at a given position are the nodes in the tree at
    that position: nodes with a parent or at least one child. Isolated
    nodes cannot be copied.

    Haplotypes are integer encoded, as described in tsls.TreeSequence.

    In Viterbi, nodes whose values (normalised by the maximum at each site)
    are below ``prune_threshold`` are dropped, and can only be reached again
    by recombination. Any path whose odds relative to the best are at least
    the threshold is kept: ``tsls.model.path_threshold`` gives the threshold
    that finds the optimal path if it has at most k mismatches.

    After normalisation at each site, values below likelihood_threshold are
    set to it; it is there to avoid underflow, and should be well below the
    switch odds. ``relative_prune_threshold`` f also sets the values of
    nodes less than f times the maximum to likelihood_threshold, in Viterbi,
    forward and backward. This is approximate (pruning relative to the
    switch odds, f = c q with c <= 1, was accurate on the sc2ts and tsinfer
    benchmarks); in Viterbi it is the same as a prune threshold of f. If
    ``count_distinct`` is True, the mean number of distinct values per site
    in Viterbi is recorded in ``mean_num_distinct_values``.

    Viterbi works with costs and keeps each node's cost as of the site it
    last changed (see docs/likelihood_model.md, "Deferred Viterbi costs").
    By default, in trees with at least ``index_min_sites`` sites left (16),
    it processes a site sparsely when the best node matches, touching only
    the nodes that mismatch (``num_sparse_sites`` counts these sites in the
    last run); this works with any per-site rates. It also leaves nodes that
    cost too much to matter at the next site unchanged. This finds a path
    with the same likelihood as processing every site in full with exact
    values (``sparse=False``), but may choose a different path among
    equally likely ones. ``index_min_sites`` is for tuning and tests.

    After each run, ``mean_num_values`` and ``max_num_values`` give the
    mean and maximum number of nodes with their own value (the likelihood
    list) after each site, ``mean_num_updated`` the mean number of node
    values computed per site, which is less than the list for sparse sites,
    and ``mean_delta_size`` the mean number of traceback entries per site.
    """

    def __init__(
        self,
        ts,
        recombination,
        mismatch,
        *,
        alleles=None,
        likelihood_threshold=None,
        prune_threshold=None,
        weight_by_n=None,
        extended_checks=False,
        relative_prune_threshold=None,
        count_distinct=False,
        sparse=True,
        index_min_sites=None,
    ):
        if weight_by_n is not None:
            raise ValueError(
                "weight_by_n has been removed: recombination is always divided "
                "by the number of nodes. For the weight_by_n=False model with "
                "recombination rho, use tsls.model.recombination_probability"
                "(rho, ts.num_nodes)"
            )
        if not isinstance(ts, TreeSequence):
            ts = TreeSequence(ts, alleles=alleles)
        elif alleles is not None:
            raise ValueError("Specify alleles when making the tsls.TreeSequence")
        kwargs = {
            "extended_checks": extended_checks,
            "count_distinct": count_distinct,
            "sparse": sparse,
        }
        if relative_prune_threshold is not None:
            kwargs["relative_prune_threshold"] = relative_prune_threshold
        if likelihood_threshold is not None:
            kwargs["likelihood_threshold"] = likelihood_threshold
        if prune_threshold is not None:
            kwargs["prune_threshold"] = prune_threshold
        if index_min_sites is not None:
            kwargs["index_min_sites"] = index_min_sites
        self.tree_sequence = ts
        self.ts = ts.ts
        super().__init__(ts, recombination=recombination, mismatch=mismatch, **kwargs)

    def find_path(self, h, start, end, match_out):
        """
        Run the Viterbi algorithm on the haplotype h over sites [start, end),
        writing the matched haplotype into match_out, and return the
        (left, right, parent) arrays of the copying path, in right-to-left
        order.
        """
        path_len, left, right, parent, matched = super().find_path(h, start, end)
        match_out[:] = matched
        return left[:path_len], right[:path_len], parent[:path_len]

    def viterbi_matrix(self, h, start=0, end=None):
        """
        Run the Viterbi algorithm on the haplotype h over sites [start, end)
        and return the ((left, right, parent), matched, matrix) tuple: the
        copying path (right-to-left, as find_path), the matched haplotype, and
        the CompressedMatrix of the values stored in the traceback.
        """
        if end is None:
            end = self.ts.num_sites
        path_len, left, right, parent, matched, arrays = super().viterbi_matrix(
            h, start, end
        )
        path = (left[:path_len], right[:path_len], parent[:path_len])
        return path, matched, CompressedMatrix(self.ts, *arrays)

    def forward_matrix(self, h):
        """
        Run the forward algorithm on the haplotype h and return the
        CompressedMatrix of normalised forward probabilities.
        """
        return CompressedMatrix(self.ts, *super().forward_matrix(h))

    def backward_matrix(self, h, normalisation_factor):
        """
        Run the backward algorithm on the haplotype h, scaled by the
        normalisation factors of the forward matrix, and return the
        CompressedMatrix of backward probabilities.
        """
        return CompressedMatrix(
            self.ts, *super().backward_matrix(h, normalisation_factor)
        )
