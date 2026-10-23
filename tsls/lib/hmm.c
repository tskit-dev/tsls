/*
 * The Li and Stephens HMM on a tree sequence.
 *
 * The states of the HMM at a given position are all the nodes in the tree
 * at that position, that is, the nodes with a parent or at least one child.
 * Likelihoods are stored on a subset of the nodes: a node with
 * NULL_LIKELIHOOD has the same likelihood as its nearest ancestor with a
 * value. After processing each site we remove the value from any node
 * whose value equals the value it would inherit from its parent. Every root
 * always has a value.
 *
 * The same machinery runs the Viterbi, forward and backward algorithms.
 * At each site, the probability of staying on a node is
 * p * (1 - rho + rho / n) (Viterbi) or p * (1 - rho) + rho / n * S (forward
 * and backward, where S is the total over all states at the previous site),
 * and the probability of switching to a given node is rho / n. If weight_by_n is set n is the
 * number of nodes in the tree sequence, otherwise n = 1. Values are
 * normalised at each site (by the maximum for Viterbi and the sum for
 * forward), and then values below likelihood_threshold are set to it.
 *
 * When a node enters the tree it gets the value 0, and so it can only be
 * reached by recombination. When a node leaves the tree, the probability
 * of continuing from it is only the probability of recombination. So a node
 * can only be copied from one site to the next without recombination if it
 * is in the tree over the whole interval between them.
 */

#include "tsls.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define MODE_VITERBI  0
#define MODE_FORWARD  1
#define MODE_BACKWARD 2

static inline void
remove_edge(tsk_id_t p, tsk_id_t c, tsk_id_t *restrict parent,
    tsk_id_t *restrict left_child, tsk_id_t *restrict right_child,
    tsk_id_t *restrict left_sib, tsk_id_t *restrict right_sib)
{
    const tsk_id_t lsib = left_sib[c];
    const tsk_id_t rsib = right_sib[c];

    if (lsib == NULL_NODE) {
        left_child[p] = rsib;
    } else {
        right_sib[lsib] = rsib;
    }
    if (rsib == NULL_NODE) {
        right_child[p] = lsib;
    } else {
        left_sib[rsib] = lsib;
    }
    parent[c] = NULL_NODE;
    left_sib[c] = NULL_NODE;
    right_sib[c] = NULL_NODE;
}

static inline void
insert_edge(tsk_id_t p, tsk_id_t c, tsk_id_t *restrict parent,
    tsk_id_t *restrict left_child, tsk_id_t *restrict right_child,
    tsk_id_t *restrict left_sib, tsk_id_t *restrict right_sib)
{
    const tsk_id_t u = right_child[p];

    parent[c] = p;
    if (u == NULL_NODE) {
        left_child[p] = c;
        left_sib[c] = NULL_NODE;
        right_sib[c] = NULL_NODE;
    } else {
        right_sib[u] = c;
        left_sib[c] = u;
        right_sib[c] = NULL_NODE;
    }
    right_child[p] = c;
}

/* Returns the allelic state of node u, given the allelic_state array in
 * which the nodes with mutations at the current site are set. */
static inline allele_t
get_allelic_state(const allele_t *restrict allelic_state,
    const tsk_id_t *restrict parent, tsk_id_t u, allele_t ancestral_state)
{
    while (allelic_state[u] == TSK_NULL) {
        if (parent[u] == NULL_NODE) {
            return ancestral_state;
        }
        u = parent[u];
    }
    return allelic_state[u];
}

/* As get_allelic_state, but for a tree in which every root has its
 * allelic state set (see tsls_hmm_set_root_allelic_state), so the walk up
 * the tree doesn't need to check for the root. This is the innermost loop
 * of the HMM, so there is deliberately no assert here (the Python module is
 * compiled with asserts on); the roots are checked in tsls_hmm_check_state. */
static inline allele_t
get_marked_allelic_state(
    const allele_t *restrict allelic_state, const tsk_id_t *restrict parent, tsk_id_t u)
{
    while (allelic_state[u] == TSK_NULL) {
        u = parent[u];
    }
    return allelic_state[u];
}

/* Position the specified null tree position at the tree containing x,
 * seeking from whichever end of the sequence is closer, as
 * tsk_tree_seek_from_null does. The edges of the tree are then the edges
 * in the tp->in range (stepping by tp->direction) that cover x. */
static int WARN_UNUSED
seek_tree_position(tsk_tree_position_t *tp, double x)
{
    int ret = 0;
    const tsk_treeseq_t *ts = tp->tree_sequence;
    const double *restrict breakpoints = ts->breakpoints;
    const tsk_size_t num_trees = ts->num_trees;
    tsk_id_t index;

    assert(tp->index == -1);
    assert(0 <= x && x < ts->tables->sequence_length);
    index = (tsk_id_t) tsk_search_sorted(breakpoints, num_trees + 1, x);
    if (breakpoints[index] > x) {
        index--;
    }
    if (x <= ts->tables->sequence_length / 2) {
        ret = tsk_tree_position_seek_forward(tp, index);
    } else {
        ret = tsk_tree_position_seek_backward(tp, index);
    }
    return ret;
}

/*===================================================================
 * Compressed matrix
 *===================================================================
 */

int
compressed_matrix_alloc(compressed_matrix_t *self, size_t num_sites, size_t block_size)
{
    int ret = 0;

    memset(self, 0, sizeof(*self));
    self->num_sites = num_sites;
    self->normalisation_factor = calloc(num_sites + 1, sizeof(double));
    self->num_values = calloc(num_sites + 1, sizeof(tsk_size_t));
    self->nodes = calloc(num_sites + 1, sizeof(tsk_id_t *));
    self->values = calloc(num_sites + 1, sizeof(double *));
    self->flags = calloc(num_sites + 1, sizeof(int8_t *));
    if (self->normalisation_factor == NULL || self->num_values == NULL
        || self->nodes == NULL || self->values == NULL || self->flags == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    if (tsi_blkalloc_init(&self->allocator, TSK_MAX(block_size, 1024)) != 0) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
out:
    return ret;
}

int
compressed_matrix_free(compressed_matrix_t *self)
{
    tsi_safe_free(self->normalisation_factor);
    tsi_safe_free(self->num_values);
    tsi_safe_free(self->nodes);
    tsi_safe_free(self->values);
    tsi_safe_free(self->flags);
    tsi_blkalloc_free(&self->allocator);
    return 0;
}

size_t
compressed_matrix_get_total_values(const compressed_matrix_t *self)
{
    size_t j;
    size_t total = 0;

    for (j = 0; j < self->num_sites; j++) {
        total += self->num_values[j];
    }
    return total;
}

/* Round up to a multiple of 8 bytes so that the doubles stay aligned. */
static inline size_t
aligned_size(size_t size)
{
    return (size + 7) & ~((size_t) 7);
}

/* Store the values of the specified nodes at this site, and if R is not NULL
 * their recombination_required flags. */
static int WARN_UNUSED
compressed_matrix_store_site(compressed_matrix_t *self, tsk_id_t site,
    double normalisation_factor, int num_values, const tsk_id_t *restrict nodes,
    const double *restrict L, const int8_t *restrict R)
{
    int ret = 0;
    int j;
    tsk_id_t *restrict nodes_out;
    double *restrict values_out;
    int8_t *restrict flags_out = NULL;

    if (R != NULL) {
        flags_out = tsi_blkalloc_get(
            &self->allocator, aligned_size((size_t) num_values * sizeof(int8_t)));
        if (flags_out == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < num_values; j++) {
            flags_out[j] = R[nodes[j]];
        }
    }

    values_out = tsi_blkalloc_get(
        &self->allocator, aligned_size((size_t) num_values * sizeof(double)));
    nodes_out = tsi_blkalloc_get(
        &self->allocator, aligned_size((size_t) num_values * sizeof(tsk_id_t)));
    if (values_out == NULL || nodes_out == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    for (j = 0; j < num_values; j++) {
        nodes_out[j] = nodes[j];
        values_out[j] = L[nodes[j]];
    }
    self->normalisation_factor[site] = normalisation_factor;
    self->num_values[site] = (tsk_size_t) num_values;
    self->nodes[site] = nodes_out;
    self->values[site] = values_out;
    self->flags[site] = flags_out;
out:
    return ret;
}

/*===================================================================
 * HMM
 *===================================================================
 */

static inline bool
tsls_hmm_in_tree(const tsls_hmm_t *self, tsk_id_t u)
{
    return self->parent[u] != NULL_NODE || self->left_child[u] != NULL_NODE;
}

static void
tsls_hmm_check_state(const tsls_hmm_t *self)
{
    int num_likelihoods = 0;
    int j;
    tsk_id_t u;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        assert(self->likelihood[u] >= 0);
    }
    for (u = 0; u < (tsk_id_t) self->num_nodes; u++) {
        if (self->likelihood[u] >= 0) {
            num_likelihoods++;
        }
        if (tsls_hmm_in_tree(self, u)) {
            assert(self->likelihood[u] != NOT_IN_TREE);
            if (self->parent[u] == NULL_NODE) {
                /* Roots always have a likelihood */
                assert(self->likelihood[u] >= 0);
            }
        } else {
            assert(self->likelihood[u] == NOT_IN_TREE);
        }
        assert(self->allelic_state[u] == TSK_NULL);
        assert(self->likelihood_cache[u] == CACHE_UNSET);
    }
    assert(num_likelihoods == self->num_likelihood_nodes);
    (void) num_likelihoods;
}

int
tsls_hmm_print_state(tsls_hmm_t *self, FILE *out)
{
    int j, k;
    tsk_id_t u;

    fprintf(out, "HMM state\n");
    fprintf(out, "site\tposition\trecomb_rate\tmut_rate\tnum_alleles\tancestral\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        fprintf(out, "%d\t%f\t%f\t%f\t%d\t%d\n", j, self->sites_position[j],
            self->recombination_rate[j], self->mismatch_rate[j],
            (int) self->num_alleles[j], (int) self->ancestral_state[j]);
    }
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlchild\trchild\tlsib\trsib\tlikelihood\n");
    for (j = 0; j < (int) self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t%d\t%d\t%f\n", (int) j, self->parent[j],
            self->left_child[j], self->right_child[j], self->left_sib[j],
            self->right_sib[j], self->likelihood[j]);
    }
    fprintf(out, "likelihood nodes\n");
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        fprintf(out, "\t%d -> %f\n", u, self->likelihood[u]);
    }
    fprintf(out, "traceback\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        fprintf(out, "\t%d:%d (%d)\t", (int) j, self->max_likelihood_node[j],
            self->traceback[j].size);
        for (k = 0; k < self->traceback[j].size; k++) {
            fprintf(out, "(%d, %d)", self->traceback[j].node[k],
                self->traceback[j].recombination_required[k]);
        }
        fprintf(out, "\n");
    }
    tsi_blkalloc_print_state(&self->traceback_allocator, out);
    return 0;
}

/* Returns the index of the specified state in the list of alleles, or -1
 * if it is not in the list. */
static int
find_allele(const char *state, tsk_size_t length, const char **alleles,
    const tsk_size_t *allele_length, tsk_size_t num_alleles)
{
    tsk_size_t j;

    for (j = 0; j < num_alleles; j++) {
        if (allele_length[j] == length && memcmp(alleles[j], state, length) == 0) {
            return (int) j;
        }
    }
    return -1;
}

/* Returns the code for the specified state. If the state is not in the list
 * of alleles and add is true, it is appended to the list. */
static int
encode_allele(const char *state, tsk_size_t length, const char **alleles,
    tsk_size_t *allele_length, tsk_size_t *num_alleles, bool add)
{
    int ret = find_allele(state, length, alleles, allele_length, *num_alleles);

    if (ret < 0) {
        if (!add) {
            ret = TSLS_ERR_UNKNOWN_ALLELE;
        } else if (*num_alleles == TSLS_MAX_ALLELES) {
            ret = TSLS_ERR_TOO_MANY_ALLELES;
        } else {
            alleles[*num_alleles] = state;
            allele_length[*num_alleles] = length;
            ret = (int) *num_alleles;
            (*num_alleles)++;
        }
    }
    return ret;
}

/* Encode the ancestral and derived states of the tree sequence as integers.
 * If alleles is NULL, the alleles are numbered at each site as in tskit's
 * genotype decoding: the ancestral state is 0, and the derived states are
 * numbered in order of their first appearance at the site, with at least
 * two alleles at each site. Otherwise, each
 * state is its index in the specified list of alleles. */
static int WARN_UNUSED
tsls_hmm_encode_alleles(
    tsls_hmm_t *self, const char **alleles, tsk_size_t num_global_alleles)
{
    int ret = 0;
    const tsk_treeseq_t *ts = self->tree_sequence;
    const tsk_site_table_t *sites = &ts->tables->sites;
    const bool global = alleles != NULL;
    const char *site_alleles[TSLS_MAX_ALLELES];
    tsk_size_t allele_length[TSLS_MAX_ALLELES];
    const char **list = global ? alleles : site_alleles;
    const tsk_mutation_t *mutation;
    tsk_size_t j, k, n, offset;
    int code;

    if (global) {
        if (num_global_alleles > TSLS_MAX_ALLELES) {
            ret = TSLS_ERR_TOO_MANY_ALLELES;
            goto out;
        }
        for (j = 0; j < num_global_alleles; j++) {
            allele_length[j] = (tsk_size_t) strlen(alleles[j]);
        }
    }
    for (j = 0; j < self->num_sites; j++) {
        n = global ? num_global_alleles : 0;
        offset = sites->ancestral_state_offset[j];
        code = encode_allele(sites->ancestral_state + offset,
            sites->ancestral_state_offset[j + 1] - offset, list, allele_length, &n,
            !global);
        if (code < 0) {
            ret = code;
            goto out;
        }
        self->ancestral_state[j] = (allele_t) code;
        for (k = 0; k < ts->site_mutations_length[j]; k++) {
            mutation = &ts->site_mutations[j][k];
            code = encode_allele(mutation->derived_state,
                mutation->derived_state_length, list, allele_length, &n, !global);
            if (code < 0) {
                ret = code;
                goto out;
            }
            self->derived_state[mutation->id] = (allele_t) code;
        }
        /* A site always has at least two alleles, so that a haplotype can
         * differ from a site with no mutations (such as a tsinfer
         * ancestor's focal sites, which are not yet in the tree sequence). */
        self->num_alleles[j] = TSK_MAX(n, 2);
    }
out:
    return ret;
}

int
tsls_hmm_alloc(tsls_hmm_t *self, const tsk_treeseq_t *tree_sequence,
    const char **alleles, tsk_size_t num_alleles, const double *recombination_rate,
    const double *mismatch_rate, double likelihood_threshold, int flags)
{
    int ret = 0;
    const size_t N = tsk_treeseq_get_num_nodes(tree_sequence);
    const size_t num_mutations = tsk_treeseq_get_num_mutations(tree_sequence);

    memset(self, 0, sizeof(tsls_hmm_t));
    self->flags = flags;
    self->likelihood_threshold = likelihood_threshold;
    self->tree_sequence = tree_sequence;
    self->num_sites = tsk_treeseq_get_num_sites(tree_sequence);
    self->num_nodes = N;
    self->sites_position = tree_sequence->tables->sites.position;
    self->sequence_length = tsk_treeseq_get_sequence_length(tree_sequence);
    self->num_alleles = malloc((self->num_sites + 1) * sizeof(*self->num_alleles));
    self->ancestral_state
        = malloc((self->num_sites + 1) * sizeof(*self->ancestral_state));
    self->derived_state = malloc((num_mutations + 1) * sizeof(*self->derived_state));
    self->recombination_rate
        = malloc((self->num_sites + 1) * sizeof(*self->recombination_rate));
    self->mismatch_rate = malloc((self->num_sites + 1) * sizeof(*self->mismatch_rate));
    self->traceback = calloc(self->num_sites + 1, sizeof(node_state_list_t));
    self->max_likelihood_node = malloc((self->num_sites + 1) * sizeof(tsk_id_t));

    self->parent = malloc((N + 1) * sizeof(*self->parent));
    self->left_child = malloc((N + 1) * sizeof(*self->left_child));
    self->right_child = malloc((N + 1) * sizeof(*self->right_child));
    self->left_sib = malloc((N + 1) * sizeof(*self->left_sib));
    self->right_sib = malloc((N + 1) * sizeof(*self->right_sib));
    self->num_below = malloc((N + 1) * sizeof(*self->num_below));
    self->num_inheriting = malloc((N + 1) * sizeof(*self->num_inheriting));
    self->likelihood = malloc((N + 1) * sizeof(*self->likelihood));
    self->allelic_state = malloc((N + 1) * sizeof(*self->allelic_state));
    self->recombination_required
        = malloc((N + 1) * sizeof(*self->recombination_required));
    self->likelihood_cache = malloc((N + 1) * sizeof(*self->likelihood_cache));
    self->recombination_required_cache
        = malloc((N + 1) * sizeof(*self->recombination_required_cache));
    self->likelihood_nodes = malloc((N + 1) * sizeof(*self->likelihood_nodes));
    self->likelihood_nodes_tmp = malloc((N + 1) * sizeof(*self->likelihood_nodes_tmp));
    self->values_tmp = malloc((N + 1) * sizeof(*self->values_tmp));

    if (self->num_alleles == NULL || self->ancestral_state == NULL
        || self->derived_state == NULL || self->recombination_rate == NULL
        || self->mismatch_rate == NULL || self->traceback == NULL
        || self->max_likelihood_node == NULL || self->parent == NULL
        || self->left_child == NULL || self->right_child == NULL
        || self->left_sib == NULL || self->right_sib == NULL || self->num_below == NULL
        || self->num_inheriting == NULL || self->recombination_required == NULL
        || self->likelihood == NULL || self->likelihood_cache == NULL
        || self->recombination_required_cache == NULL
        || self->likelihood_nodes == NULL || self->likelihood_nodes_tmp == NULL
        || self->values_tmp == NULL || self->allelic_state == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    ret = tsls_hmm_encode_alleles(self, alleles, num_alleles);
    if (ret != 0) {
        goto out;
    }
    /* Alloc in blocks of at least 64MiB, and large enough to hold the
     * traceback for a site in which every node has a likelihood. */
    self->traceback_block_size
        = TSK_MAX(64 * 1024 * 1024, aligned_size((N + 1) * sizeof(tsk_id_t)) * 2);
    /* If the traceback allocator is using more than 2GiB of RAM free it, so
     * that other threads can use the memory */
    self->traceback_realloc_size = 2L * 1024L * 1024L * 1024L;
    if (tsi_blkalloc_init(&self->traceback_allocator, self->traceback_block_size)
        != 0) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    memcpy(self->recombination_rate, recombination_rate,
        self->num_sites * sizeof(*self->recombination_rate));
    memcpy(self->mismatch_rate, mismatch_rate,
        self->num_sites * sizeof(*self->mismatch_rate));
    memset(self->allelic_state, 0xff, N * sizeof(*self->allelic_state));
    memset(self->max_likelihood_node, 0xff, self->num_sites * sizeof(tsk_id_t));
out:
    return ret;
}

int
tsls_hmm_free(tsls_hmm_t *self)
{
    tsi_safe_free(self->num_alleles);
    tsi_safe_free(self->ancestral_state);
    tsi_safe_free(self->derived_state);
    tsi_safe_free(self->recombination_rate);
    tsi_safe_free(self->mismatch_rate);
    tsi_safe_free(self->parent);
    tsi_safe_free(self->left_child);
    tsi_safe_free(self->right_child);
    tsi_safe_free(self->left_sib);
    tsi_safe_free(self->right_sib);
    tsi_safe_free(self->num_below);
    tsi_safe_free(self->num_inheriting);
    tsi_safe_free(self->recombination_required);
    tsi_safe_free(self->likelihood);
    tsi_safe_free(self->likelihood_cache);
    tsi_safe_free(self->recombination_required_cache);
    tsi_safe_free(self->likelihood_nodes);
    tsi_safe_free(self->likelihood_nodes_tmp);
    tsi_safe_free(self->values_tmp);
    tsi_safe_free(self->allelic_state);
    tsi_safe_free(self->max_likelihood_node);
    tsi_safe_free(self->traceback);
    tsi_blkalloc_free(&self->traceback_allocator);
    return 0;
}

int
tsls_hmm_set_pruning(tsls_hmm_t *self, int mode, double parameter)
{
    int ret = 0;

    if (mode == TSLS_PRUNE_NONE) {
        parameter = 0;
    } else if (mode == TSLS_PRUNE_RELATIVE) {
        if (!(parameter >= 0 && parameter <= 1)) {
            ret = TSLS_ERR_BAD_PRUNING;
            goto out;
        }
    } else if (mode == TSLS_PRUNE_TOP_N) {
        if (!(parameter >= 1)) {
            ret = TSLS_ERR_BAD_PRUNING;
            goto out;
        }
    } else {
        ret = TSLS_ERR_BAD_PRUNING;
        goto out;
    }
    self->pruning_mode = mode;
    self->pruning_parameter = parameter;
out:
    return ret;
}

static int
cmp_double(const void *a, const void *b)
{
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return (x > y) - (x < y);
}

/* Copy the current likelihood values into values_tmp, sorted. */
static void
tsls_hmm_sort_values(tsls_hmm_t *self)
{
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        self->values_tmp[j] = self->likelihood[self->likelihood_nodes[j]];
    }
    qsort(self->values_tmp, (size_t) self->num_likelihood_nodes, sizeof(double),
        cmp_double);
}

/* Apply the experimental pruning, giving pruned nodes the value
 * likelihood_threshold. The values must be normalised. */
static void
tsls_hmm_prune(tsls_hmm_t *self)
{
    double *restrict L = self->likelihood;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    double max_L, cutoff;
    int j, n;

    if (self->pruning_mode == TSLS_PRUNE_NONE || num_likelihood_nodes == 0) {
        return;
    }
    if (self->pruning_mode == TSLS_PRUNE_RELATIVE) {
        max_L = 0;
        for (j = 0; j < num_likelihood_nodes; j++) {
            max_L = TSK_MAX(max_L, L[L_nodes[j]]);
        }
        cutoff = self->pruning_parameter * max_L;
    } else {
        n = (int) self->pruning_parameter;
        if (num_likelihood_nodes <= n) {
            return;
        }
        tsls_hmm_sort_values(self);
        /* Keep values >= the nth largest (ties at the cutoff are kept) */
        cutoff = self->values_tmp[num_likelihood_nodes - n];
    }
    for (j = 0; j < num_likelihood_nodes; j++) {
        if (L[L_nodes[j]] < cutoff) {
            L[L_nodes[j]] = self->likelihood_threshold;
        }
    }
}

/* Count the number of distinct likelihood values at this site. */
static void
tsls_hmm_count_distinct(tsls_hmm_t *self)
{
    int j;
    size_t num_distinct;

    if (!(self->flags & TSLS_COUNT_DISTINCT) || self->num_likelihood_nodes == 0) {
        return;
    }
    tsls_hmm_sort_values(self);
    num_distinct = 1;
    for (j = 1; j < self->num_likelihood_nodes; j++) {
        if (self->values_tmp[j] != self->values_tmp[j - 1]) {
            num_distinct++;
        }
    }
    self->total_distinct_values += num_distinct;
}

static inline void
tsls_hmm_add_likelihood(tsls_hmm_t *self, tsk_id_t u, double value)
{
    self->likelihood[u] = value;
    self->likelihood_nodes[self->num_likelihood_nodes] = u;
    self->num_likelihood_nodes++;
}

/* Remove the specified node from the list of nodes with likelihoods. */
static void
tsls_hmm_delete_likelihood(tsls_hmm_t *self, const tsk_id_t node)
{
    int j, k;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;

    k = 0;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        L_nodes[k] = L_nodes[j];
        if (L_nodes[j] != node) {
            k++;
        }
    }
    assert(self->num_likelihood_nodes == k + 1);
    self->num_likelihood_nodes = k;
}

static inline void
tsls_hmm_update_num_below(tsls_hmm_t *self, tsk_id_t u, int64_t delta)
{
    const tsk_id_t *restrict parent = self->parent;
    int64_t *restrict num_below = self->num_below;

    for (; u != NULL_NODE; u = parent[u]) {
        num_below[u] += delta;
    }
}

/* Returns the likelihood inherited by node u, filling in the likelihood
 * cache along the path so that later queries are fast. */
static double
tsls_hmm_get_inherited_likelihood(tsls_hmm_t *self, tsk_id_t u)
{
    const double *restrict L = self->likelihood;
    double *restrict L_cache = self->likelihood_cache;
    const tsk_id_t *restrict parent = self->parent;
    double value;
    tsk_id_t v;

    v = u;
    while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
        v = parent[v];
        assert(v != NULL_NODE);
    }
    value = L_cache[v];
    if (unlikely(value == CACHE_UNSET)) {
        value = L[v];
    }
    assert(value >= 0);
    v = u;
    while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
        L_cache[v] = value;
        v = parent[v];
    }
    return value;
}

static void
tsls_hmm_reset_likelihood_cache(tsls_hmm_t *self, tsk_id_t u)
{
    double *restrict L_cache = self->likelihood_cache;
    const tsk_id_t *restrict parent = self->parent;

    while (u != NULL_NODE && L_cache[u] != CACHE_UNSET) {
        L_cache[u] = CACHE_UNSET;
        u = parent[u];
    }
}

/* Remove the edge from p to c from the tree, and if update_likelihoods is
 * true make sure that the child keeps its likelihood. Nodes that are
 * isolated after removing the edge are added to the isolated list; they
 * only leave the tree if they are still isolated after all the edges at
 * this position have been inserted (see tsls_hmm_update_tree). */
static void
tsls_hmm_remove_edge(tsls_hmm_t *self, tsk_id_t p, tsk_id_t c,
    bool update_likelihoods, tsk_id_t *restrict isolated, int *num_isolated)
{
    double *restrict L = self->likelihood;

    if (self->track_num_below) {
        tsls_hmm_update_num_below(self, p, -self->num_below[c]);
    }
    if (update_likelihoods && L[c] == NULL_LIKELIHOOD) {
        tsls_hmm_add_likelihood(self, c, tsls_hmm_get_inherited_likelihood(self, p));
    }
    remove_edge(p, c, self->parent, self->left_child, self->right_child,
        self->left_sib, self->right_sib);
    if (update_likelihoods) {
        /* Each node can only become isolated once when removing edges, so
         * the isolated list has at most num_nodes entries. */
        if (!tsls_hmm_in_tree(self, c)) {
            isolated[*num_isolated] = c;
            (*num_isolated)++;
        }
        if (!tsls_hmm_in_tree(self, p)) {
            /* p was a root, and so has a likelihood */
            assert(L[p] >= 0);
            isolated[*num_isolated] = p;
            (*num_isolated)++;
        }
    }
}

/* Insert the edge from p to c into the tree. Nodes entering the tree get
 * likelihood 0, and so can only be reached by recombination. */
static void
tsls_hmm_insert_edge(tsls_hmm_t *self, tsk_id_t p, tsk_id_t c, bool update_likelihoods)
{
    double *restrict L = self->likelihood;

    if (update_likelihoods) {
        if (L[p] == NOT_IN_TREE) {
            tsls_hmm_add_likelihood(self, p, 0);
        }
        if (L[c] == NOT_IN_TREE) {
            tsls_hmm_add_likelihood(self, c, 0);
        }
        /* Otherwise the child was a root and keeps its likelihood */
        assert(L[c] >= 0);
    }
    insert_edge(p, c, self->parent, self->left_child, self->right_child,
        self->left_sib, self->right_sib);
    if (self->track_num_below) {
        tsls_hmm_update_num_below(self, p, self->num_below[c]);
    }
}

/* Apply the edge removals and insertions recorded in the tree position,
 * which has just moved to the next tree in its direction of travel. */
static void
tsls_hmm_update_tree(tsls_hmm_t *self, const tsk_tree_position_t *tp)
{
    const tsk_id_t *restrict edge_parent = self->tree_sequence->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tree_sequence->tables->edges.child;
    const int direction = tp->direction;
    tsk_id_t *restrict isolated = self->likelihood_nodes_tmp;
    double *restrict L = self->likelihood;
    tsk_id_t j, e, u;
    int k, num_isolated = 0;

    for (j = tp->out.start; j != tp->out.stop; j += direction) {
        e = tp->out.order[j];
        tsls_hmm_remove_edge(
            self, edge_parent[e], edge_child[e], true, isolated, &num_isolated);
    }
    for (j = tp->out.start; j != tp->out.stop; j += direction) {
        tsls_hmm_reset_likelihood_cache(self, edge_parent[tp->out.order[j]]);
    }
    for (j = tp->in.start; j != tp->in.stop; j += direction) {
        e = tp->in.order[j];
        tsls_hmm_insert_edge(self, edge_parent[e], edge_child[e], true);
    }
    /* Nodes that are still isolated have left the tree. Nodes that were
     * isolated and then inserted again keep their likelihoods. */
    for (k = 0; k < num_isolated; k++) {
        u = isolated[k];
        if (!tsls_hmm_in_tree(self, u)) {
            tsls_hmm_delete_likelihood(self, u);
            L[u] = NOT_IN_TREE;
        }
    }
}

/* Reset the HMM and build the tree containing the specified position,
 * giving each root the specified initial likelihood. The tree position must
 * be null. */
static int WARN_UNUSED
tsls_hmm_init_sweep(
    tsls_hmm_t *self, tsk_tree_position_t *tp, double position, double initial_value)
{
    int ret = 0;
    const size_t N = self->num_nodes;
    const tsk_id_t *restrict edge_parent = self->tree_sequence->tables->edges.parent;
    const tsk_id_t *restrict edge_child = self->tree_sequence->tables->edges.child;
    const double *restrict edge_left = self->tree_sequence->tables->edges.left;
    const double *restrict edge_right = self->tree_sequence->tables->edges.right;
    tsk_id_t u, j, e;
    size_t k;

    memset(self->parent, 0xff, N * sizeof(*self->parent));
    memset(self->left_child, 0xff, N * sizeof(*self->left_child));
    memset(self->right_child, 0xff, N * sizeof(*self->right_child));
    memset(self->left_sib, 0xff, N * sizeof(*self->left_sib));
    memset(self->right_sib, 0xff, N * sizeof(*self->right_sib));
    memset(self->recombination_required, 0xff, N * sizeof(*self->recombination_required));
    for (k = 0; k < N; k++) {
        self->likelihood_cache[k] = CACHE_UNSET;
        self->num_below[k] = 1;
    }
    ret = seek_tree_position(tp, position);
    if (ret != 0) {
        goto out;
    }
    for (j = tp->in.start; j != tp->in.stop; j += tp->direction) {
        e = tp->in.order[j];
        if (edge_left[e] <= position && position < edge_right[e]) {
            tsls_hmm_insert_edge(self, edge_parent[e], edge_child[e], false);
        }
    }
    self->num_likelihood_nodes = 0;
    for (u = 0; u < (tsk_id_t) N; u++) {
        if (self->parent[u] != NULL_NODE) {
            self->likelihood[u] = NULL_LIKELIHOOD;
        } else if (self->left_child[u] != NULL_NODE) {
            tsls_hmm_add_likelihood(self, u, initial_value);
        } else {
            self->likelihood[u] = NOT_IN_TREE;
        }
    }
out:
    return ret;
}

static inline void
tsls_hmm_set_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const tsk_mutation_t *restrict mutations = self->tree_sequence->site_mutations[site];
    const tsk_size_t num_mutations = self->tree_sequence->site_mutations_length[site];
    tsk_size_t j;

    for (j = 0; j < num_mutations; j++) {
        self->allelic_state[mutations[j].node] = self->derived_state[mutations[j].id];
    }
}

static inline void
tsls_hmm_unset_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const tsk_mutation_t *restrict mutations = self->tree_sequence->site_mutations[site];
    const tsk_size_t num_mutations = self->tree_sequence->site_mutations_length[site];
    tsk_size_t j;

    for (j = 0; j < num_mutations; j++) {
        self->allelic_state[mutations[j].node] = TSK_NULL;
    }
}

/* Set the allelic state of every root to the ancestral state, so that
 * get_marked_allelic_state finds a state for every node in the tree.
 * Every root has a likelihood, so the roots are found in the likelihood
 * list. This must be called before tsls_hmm_set_allelic_state, so that
 * mutations above a root take precedence, and the likelihood list must not
 * change before tsls_hmm_unset_root_allelic_state. */
static inline void
tsls_hmm_set_root_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const allele_t ancestral_state = self->ancestral_state[site];
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    allele_t *restrict allelic_state = self->allelic_state;
    tsk_id_t u;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        if (parent[u] == NULL_NODE) {
            allelic_state[u] = ancestral_state;
        }
    }
}

static inline void
tsls_hmm_unset_root_allelic_state(tsls_hmm_t *self)
{
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    allele_t *restrict allelic_state = self->allelic_state;
    tsk_id_t u;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        if (parent[u] == NULL_NODE) {
            allelic_state[u] = TSK_NULL;
        }
    }
}

/* Make sure that every node with a mutation at this site that is in the
 * tree has its own likelihood. */
static void
tsls_hmm_insert_mutation_likelihoods(tsls_hmm_t *self, const tsk_id_t site)
{
    const double *restrict L = self->likelihood;
    const tsk_id_t *restrict parent = self->parent;
    const tsk_mutation_t *restrict mutations = self->tree_sequence->site_mutations[site];
    const tsk_size_t num_mutations = self->tree_sequence->site_mutations_length[site];
    tsk_size_t j;
    tsk_id_t u, node;

    for (j = 0; j < num_mutations; j++) {
        node = mutations[j].node;
        if (L[node] == NULL_LIKELIHOOD) {
            u = node;
            while (L[u] == NULL_LIKELIHOOD) {
                u = parent[u];
                assert(u != NULL_NODE);
            }
            tsls_hmm_add_likelihood(self, node, L[u]);
        }
    }
}

/* As tsls_hmm_get_inherited_likelihood, but also return the
 * recombination_required flag that u inherits, caching it along the path
 * with the likelihood. */
static inline double
tsls_hmm_get_inherited_likelihood_and_flag(
    tsls_hmm_t *self, tsk_id_t u, int8_t *flag)
{
    const double *restrict L = self->likelihood;
    const int8_t *restrict R = self->recombination_required;
    double *restrict L_cache = self->likelihood_cache;
    int8_t *restrict R_cache = self->recombination_required_cache;
    const tsk_id_t *restrict parent = self->parent;
    double value;
    int8_t r;
    tsk_id_t v;

    v = u;
    while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
        v = parent[v];
        assert(v != NULL_NODE);
    }
    if (L_cache[v] == CACHE_UNSET) {
        value = L[v];
        r = R[v];
    } else {
        value = L_cache[v];
        r = R_cache[v];
    }
    assert(value >= 0);
    v = u;
    while (likely(L[v] == NULL_LIKELIHOOD) && likely(L_cache[v] == CACHE_UNSET)) {
        L_cache[v] = value;
        R_cache[v] = r;
        v = parent[v];
    }
    *flag = r;
    return value;
}

/* Remove likelihoods that are equal to the likelihood inherited from the
 * parent. If check_recombination_required is true, also require the
 * recombination_required flag to equal the one inherited from the parent,
 * so that the traceback, in which nodes inherit the flag of their nearest
 * ancestor in the list, is unchanged. */
static void
tsls_hmm_coalesce_likelihoods(
    tsls_hmm_t *self, bool check_recombination_required)
{
    const int8_t *restrict R = self->recombination_required;
    double L_p;
    int8_t R_p;
    tsk_id_t u, p;
    double *restrict L = self->likelihood;
    const tsk_id_t *restrict parent = self->parent;
    tsk_id_t *restrict cached_paths = self->likelihood_nodes_tmp;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    int j, num_cached_paths, num_likelihood_nodes;

    num_cached_paths = 0;
    num_likelihood_nodes = 0;
    for (j = 0; j < old_num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p = parent[u];
        if (p != NULL_NODE) {
            cached_paths[num_cached_paths] = p;
            num_cached_paths++;
            if (check_recombination_required) {
                /* A removed node had the inherited value and flag, so the
                 * cached values below it stay valid. */
                L_p = tsls_hmm_get_inherited_likelihood_and_flag(
                    self, p, &R_p);
                if (L[u] == L_p && R[u] == R_p) {
                    L[u] = NULL_LIKELIHOOD;
                }
            } else {
                L_p = tsls_hmm_get_inherited_likelihood(self, p);
                if (L[u] == L_p) {
                    L[u] = NULL_LIKELIHOOD;
                }
            }
        }
        if (L[u] >= 0) {
            L_nodes[num_likelihood_nodes] = L_nodes[j];
            num_likelihood_nodes++;
        }
    }
    self->num_likelihood_nodes = num_likelihood_nodes;
    for (j = 0; j < num_cached_paths; j++) {
        tsls_hmm_reset_likelihood_cache(self, cached_paths[j]);
    }
}

static inline double
emission_probability(allele_t node_state, allele_t state, double mu, double num_alleles)
{
    if (state == TSLS_MISSING_DATA || node_state == state) {
        return 1 - (num_alleles - 1) * mu;
    }
    return mu;
}

static inline int
check_haplotype_state(const tsls_hmm_t *self, tsk_id_t site, allele_t state)
{
    const tsk_size_t num_alleles = self->num_alleles[site];

    if (state < TSLS_MISSING_DATA || (state >= 0 && (tsk_size_t) state >= num_alleles)) {
        return TSLS_ERR_BAD_HAPLOTYPE_ALLELE;
    }
    return 0;
}

static inline double
tsls_hmm_get_n(const tsls_hmm_t *self)
{
    double n = 1.0;
    if (!(self->flags & TSLS_DISABLE_WEIGHT_BY_N)) {
        n = (double) self->num_nodes;
    }
    return n;
}

/* Compute the number of nodes in the tree that inherit the likelihood of
 * each likelihood node. */
static void
tsls_hmm_compute_num_inheriting(tsls_hmm_t *self)
{
    const double *restrict L = self->likelihood;
    const tsk_id_t *restrict parent = self->parent;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int64_t *restrict num_below = self->num_below;
    int64_t *restrict num_inheriting = self->num_inheriting;
    tsk_id_t u, w;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        num_inheriting[u] = num_below[u];
    }
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        w = parent[u];
        if (w != NULL_NODE) {
            while (L[w] == NULL_LIKELIHOOD) {
                w = parent[w];
            }
            num_inheriting[w] -= num_below[u];
        }
    }
}

/* Returns the sum of the likelihoods over all the nodes in the tree. */
static double
tsls_hmm_total_likelihood(const tsls_hmm_t *self)
{
    const double *restrict L = self->likelihood;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int64_t *restrict num_inheriting = self->num_inheriting;
    double total = 0;
    tsk_id_t u;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        assert(num_inheriting[u] > 0);
        total += (double) num_inheriting[u] * L[u];
    }
    return total;
}

/* Store the recombination_required state in the traceback */
static int WARN_UNUSED
tsls_hmm_store_traceback(tsls_hmm_t *self, const tsk_id_t site_id)
{
    int ret = 0;
    tsk_id_t u;
    int j;
    int8_t *restrict list_R;
    tsk_id_t *restrict list_node;
    node_state_list_t *restrict list;
    node_state_list_t *restrict T = self->traceback;
    const tsk_id_t *restrict nodes = self->likelihood_nodes;
    const int8_t *restrict R = self->recombination_required;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    bool match;

    /* Check to see if the previous site has the same recombination_required. If so,
     * we can reuse the same list. */
    match = false;
    if (site_id > 0) {
        list = &T[site_id - 1];
        if (list->size == num_likelihood_nodes) {
            list_node = list->node;
            list_R = list->recombination_required;
            match = true;
            for (j = 0; j < num_likelihood_nodes; j++) {
                if (list_node[j] != nodes[j] || list_R[j] != R[nodes[j]]) {
                    match = false;
                    break;
                }
            }
        }
    }

    if (match) {
        T[site_id].size = T[site_id - 1].size;
        T[site_id].node = T[site_id - 1].node;
        T[site_id].recombination_required = T[site_id - 1].recombination_required;
    } else {
        list_node = tsi_blkalloc_get(&self->traceback_allocator,
            aligned_size((size_t) num_likelihood_nodes * sizeof(tsk_id_t)));
        list_R = tsi_blkalloc_get(&self->traceback_allocator,
            aligned_size((size_t) num_likelihood_nodes * sizeof(int8_t)));
        if (list_node == NULL || list_R == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        T[site_id].node = list_node;
        T[site_id].recombination_required = list_R;
        T[site_id].size = num_likelihood_nodes;
        for (j = 0; j < num_likelihood_nodes; j++) {
            u = nodes[j];
            list_node[j] = u;
            list_R[j] = R[u];
        }
    }
    self->total_traceback_size += (size_t) num_likelihood_nodes;
out:
    return ret;
}

static int WARN_UNUSED
tsls_hmm_process_site_viterbi(tsls_hmm_t *self, const tsk_id_t site,
    const allele_t state, compressed_matrix_t *output)
{
    int ret = 0;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    double *restrict L = self->likelihood;
    int8_t *restrict recombination_required = self->recombination_required;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->num_alleles[site];
    double max_L, p_no_recomb, p_recomb, p_t, p_e;
    tsk_id_t u, max_L_node;
    int j;

    tsls_hmm_insert_mutation_likelihoods(self, site);
    tsls_hmm_set_root_allelic_state(self, site);
    tsls_hmm_set_allelic_state(self, site);
    max_L = -1;
    max_L_node = NULL_NODE;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p_no_recomb = L[u] * (1 - rho + rho / n);
        p_recomb = rho / n;
        recombination_required[u] = false;
        if (p_no_recomb > p_recomb) {
            p_t = p_no_recomb;
        } else {
            p_t = p_recomb;
            recombination_required[u] = true;
        }
        p_e = emission_probability(
            get_marked_allelic_state(allelic_state, parent, u), state, mu, num_alleles);
        L[u] = p_t * p_e;
        if (L[u] > max_L) {
            max_L = L[u];
            max_L_node = u;
        }
    }
    tsls_hmm_unset_allelic_state(self, site);
    tsls_hmm_unset_root_allelic_state(self);
    (void) num_likelihood_nodes;

    if (max_L <= 0) {
        if (mu <= 0 || mu >= 1) {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA;
        } else if (rho == 0) {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION;
        } else {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE;
        }
        goto out;
    }
    assert(max_L_node != NULL_NODE);
    self->max_likelihood_node[site] = max_L_node;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] = TSK_MAX(L[u] / max_L, self->likelihood_threshold);
    }
    tsls_hmm_prune(self);
    tsls_hmm_count_distinct(self);
    /* Remove redundant nodes before storing the traceback, keeping those
     * whose recombination_required flag differs from the inherited one.
     * Those are left for the next site, where their flags are recomputed. */
    tsls_hmm_coalesce_likelihoods(self, true);
    ret = tsls_hmm_store_traceback(self, site);
    if (ret != 0) {
        goto out;
    }
    if (output != NULL) {
        ret = compressed_matrix_store_site(output, site, max_L,
            self->num_likelihood_nodes, L_nodes, L, recombination_required);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* Recombination redistributes the total probability over the states at the
 * previous site, including nodes that have since left the tree, so we
 * compute the total at the end of each site and use it at the next. At the
 * first site, previous_total is negative and we use the current total. */
static int WARN_UNUSED
tsls_hmm_process_site_forward(tsls_hmm_t *self, const tsk_id_t site,
    const allele_t state, double *previous_total, compressed_matrix_t *output)
{
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    double *restrict L = self->likelihood;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->num_alleles[site];
    double total, normalisation_factor, p_e;
    tsk_id_t u;
    int j;

    tsls_hmm_insert_mutation_likelihoods(self, site);
    tsls_hmm_compute_num_inheriting(self);
    total = *previous_total;
    if (total < 0) {
        total = tsls_hmm_total_likelihood(self);
    }
    tsls_hmm_set_root_allelic_state(self, site);
    tsls_hmm_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p_e = emission_probability(
            get_marked_allelic_state(allelic_state, parent, u), state, mu, num_alleles);
        L[u] = (L[u] * (1 - rho) + rho / n * total) * p_e;
    }
    tsls_hmm_unset_allelic_state(self, site);
    tsls_hmm_unset_root_allelic_state(self);
    normalisation_factor = tsls_hmm_total_likelihood(self);
    if (!(normalisation_factor > 0)) {
        ret = TSLS_ERR_MATCH_IMPOSSIBLE;
        goto out;
    }
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] = TSK_MAX(L[u] / normalisation_factor, self->likelihood_threshold);
    }
    tsls_hmm_prune(self);
    *previous_total = tsls_hmm_total_likelihood(self);
    ret = compressed_matrix_store_site(output, site, normalisation_factor,
        self->num_likelihood_nodes, L_nodes, L, NULL);
    if (ret != 0) {
        goto out;
    }
    tsls_hmm_coalesce_likelihoods(self, false);
out:
    return ret;
}

/* The backward transition from site l to site l - 1 depends on the tree at
 * site l - 1 (nodes may have entered or left the tree), so we apply it
 * lazily when we reach site l - 1, using these saved values from site l.
 */
typedef struct {
    bool pending;
    double total;
    double rho;
    double normalisation_factor;
} backward_transition_t;

static int WARN_UNUSED
tsls_hmm_process_site_backward(tsls_hmm_t *self, const tsk_id_t site,
    const allele_t state, const double *forward_normalisation_factor,
    backward_transition_t *transition, compressed_matrix_t *output)
{
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    double *restrict L = self->likelihood;
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->num_alleles[site];
    double x;
    tsk_id_t u;
    int j;

    if (transition->pending) {
        for (j = 0; j < self->num_likelihood_nodes; j++) {
            u = L_nodes[j];
            x = (1 - transition->rho) * L[u] + transition->rho * transition->total / n;
            L[u] = TSK_MAX(
                x / transition->normalisation_factor, self->likelihood_threshold);
        }
        tsls_hmm_prune(self);
    }
    tsls_hmm_insert_mutation_likelihoods(self, site);
    ret = compressed_matrix_store_site(output, site, forward_normalisation_factor[site],
        self->num_likelihood_nodes, L_nodes, L, NULL);
    if (ret != 0) {
        goto out;
    }
    tsls_hmm_compute_num_inheriting(self);
    tsls_hmm_set_root_allelic_state(self, site);
    tsls_hmm_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] *= emission_probability(
            get_marked_allelic_state(allelic_state, parent, u), state, mu, num_alleles);
    }
    tsls_hmm_unset_allelic_state(self, site);
    tsls_hmm_unset_root_allelic_state(self);
    transition->pending = true;
    transition->total = tsls_hmm_total_likelihood(self);
    transition->rho = self->recombination_rate[site];
    transition->normalisation_factor = forward_normalisation_factor[site];
    tsls_hmm_coalesce_likelihoods(self, false);
out:
    return ret;
}

/* Run the specified algorithm over sites [start, end). */
static int WARN_UNUSED
tsls_hmm_run(tsls_hmm_t *self, int mode, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, const double *forward_normalisation_factor,
    compressed_matrix_t *output)
{
    int ret = 0;
    tsk_tree_position_t tp;
    backward_transition_t transition;
    const double *restrict position = self->sites_position;
    const int direction = mode == MODE_BACKWARD ? TSK_DIR_REVERSE : TSK_DIR_FORWARD;
    tsk_id_t site, first_site;
    double previous_total = -1;
    bool valid;

    tsk_tree_position_init(&tp, self->tree_sequence, 0);
    for (site = start; site < end; site++) {
        ret = check_haplotype_state(self, site, haplotype[site]);
        if (ret != 0) {
            goto out;
        }
    }
    self->track_num_below = mode != MODE_VITERBI;
    first_site = direction > 0 ? start : end - 1;
    ret = tsls_hmm_init_sweep(self, &tp, position[first_site], 1.0);
    if (ret != 0) {
        goto out;
    }
    transition.pending = false;

    for (site = first_site; start <= site && site < end; site += direction) {
        while (!(tp.interval.left <= position[site]
                 && position[site] < tp.interval.right)) {
            valid = direction > 0 ? tsk_tree_position_next(&tp)
                                  : tsk_tree_position_prev(&tp);
            assert(valid);
            (void) valid;
            tsls_hmm_update_tree(self, &tp);
        }
        if (self->flags & TSLS_EXTENDED_CHECKS) {
            tsls_hmm_check_state(self);
        }
        if (mode == MODE_VITERBI) {
            ret = tsls_hmm_process_site_viterbi(self, site, haplotype[site], output);
        } else if (mode == MODE_FORWARD) {
            ret = tsls_hmm_process_site_forward(
                self, site, haplotype[site], &previous_total, output);
        } else {
            ret = tsls_hmm_process_site_backward(self, site, haplotype[site],
                forward_normalisation_factor, &transition, output);
        }
        if (ret != 0) {
            goto out;
        }
    }
out:
    tsk_tree_position_free(&tp);
    return ret;
}

/* Resets the recombination_required array from the traceback at the specified site.
 */
static inline void
tsls_hmm_set_recombination_required(
    tsls_hmm_t *self, tsk_id_t site, int8_t *restrict recombination_required)
{
    int j;
    const int8_t *restrict R = self->traceback[site].recombination_required;
    const tsk_id_t *restrict node = self->traceback[site].node;
    const int size = self->traceback[site].size;

    for (j = 0; j < size; j++) {
        recombination_required[node[j]] = R[j];
    }
}

/* Unsets the likelihood array from the traceback at the specified site.
 */
static inline void
tsls_hmm_unset_recombination_required(
    tsls_hmm_t *self, tsk_id_t site, int8_t *restrict recombination_required)
{
    int j;
    const tsk_id_t *restrict node = self->traceback[site].node;
    const int size = self->traceback[site].size;

    for (j = 0; j < size; j++) {
        recombination_required[node[j]] = -1;
    }
}

/* Run the traceback over sites [start, end), moving right-to-left through
 * the trees. Only the parent array of the tree is needed. */
static int WARN_UNUSED
tsls_hmm_run_traceback(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    allele_t *restrict match, size_t *path_length_out, double *restrict path_left,
    double *restrict path_right, tsk_id_t *restrict path_parent)
{
    int ret = 0;
    tsk_tree_position_t tp;
    tsk_id_t site, j, e, u, max_likelihood_node;
    double x;
    bool valid;
    tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    int8_t *restrict recombination_required = self->recombination_required;
    const double *restrict sites_position = self->sites_position;
    const tsk_edge_table_t *edges = &self->tree_sequence->tables->edges;
    const tsk_id_t *restrict edge_parent = edges->parent;
    const tsk_id_t *restrict edge_child = edges->child;
    const double *restrict edge_left = edges->left;
    const double *restrict edge_right = edges->right;
    const double start_pos = start == 0 ? 0 : sites_position[start];
    const double end_pos
        = end == (tsk_id_t) self->num_sites ? self->sequence_length : sites_position[end];
    size_t path_length = 0;

    tsk_tree_position_init(&tp, self->tree_sequence, 0);

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    path_right[path_length] = end_pos;
    max_likelihood_node = self->max_likelihood_node[end - 1];
    assert(max_likelihood_node != NULL_NODE);
    path_parent[path_length] = max_likelihood_node;

    /* Build the tree at the last site, and then go through the trees in
     * reverse, running the traceback at each site. */
    memset(parent, 0xff, self->num_nodes * sizeof(*parent));
    memset(
        recombination_required, 0xff, self->num_nodes * sizeof(*recombination_required));
    x = sites_position[end - 1];
    ret = seek_tree_position(&tp, x);
    if (ret != 0) {
        goto out;
    }
    for (j = tp.in.start; j != tp.in.stop; j += tp.direction) {
        e = tp.in.order[j];
        if (edge_left[e] <= x && x < edge_right[e]) {
            parent[edge_child[e]] = edge_parent[e];
        }
    }
    for (site = end - 1; site >= start; site--) {
        while (sites_position[site] < tp.interval.left) {
            valid = tsk_tree_position_prev(&tp);
            assert(valid);
            (void) valid;
            for (j = tp.out.start; j != tp.out.stop; j--) {
                parent[edge_child[tp.out.order[j]]] = NULL_NODE;
            }
            for (j = tp.in.start; j != tp.in.stop; j--) {
                e = tp.in.order[j];
                parent[edge_child[e]] = edge_parent[e];
            }
        }
        tsls_hmm_set_allelic_state(self, site);
        u = path_parent[path_length];
        match[site]
            = get_allelic_state(allelic_state, parent, u, self->ancestral_state[site]);
        tsls_hmm_unset_allelic_state(self, site);

        /* Mark the traceback nodes on the tree */
        tsls_hmm_set_recombination_required(self, site, recombination_required);

        /* Traverse up the tree from the current node. The first marked
         * node that we meet tells us whether we need to recombine. Every
         * root has a likelihood, and so is marked. */
        while (recombination_required[u] == -1) {
            u = parent[u];
            assert(u != NULL_NODE);
        }
        if (recombination_required[u] && site > start) {
            max_likelihood_node = self->max_likelihood_node[site - 1];
            assert(max_likelihood_node != NULL_NODE);
            path_left[path_length] = sites_position[site];
            path_length++;
            /* Start the next output edge */
            path_right[path_length] = sites_position[site];
            path_parent[path_length] = max_likelihood_node;
        }
        /* Unset the values in the tree for the next site. */
        tsls_hmm_unset_recombination_required(self, site, recombination_required);
    }
    path_left[path_length] = start_pos;
    path_length++;
    *path_length_out = path_length;
out:
    tsk_tree_position_free(&tp);
    return ret;
}

static int
tsls_hmm_reset_traceback(tsls_hmm_t *self)
{
    int ret = 0;

    if (self->traceback_allocator.total_size > self->traceback_realloc_size) {
        tsi_blkalloc_free(&self->traceback_allocator);
        if (tsi_blkalloc_init(&self->traceback_allocator, self->traceback_block_size)
            != 0) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
    } else {
        tsi_blkalloc_reset(&self->traceback_allocator);
    }
    self->total_traceback_size = 0;
    self->total_distinct_values = 0;
out:
    return ret;
}

/* Run the Viterbi algorithm and traceback. If output is not NULL, the
 * normalised values and recombination_required flags stored in the
 * traceback at each site are also stored in output, with the maximum
 * likelihood (before normalisation) as the normalisation factor. */
int
tsls_hmm_find_path_matrix(tsls_hmm_t *self, tsk_id_t start,
    tsk_id_t end, const allele_t *haplotype, allele_t *matched_haplotype,
    size_t *path_length, double *path_left, double *path_right,
    tsk_id_t *path_parent, compressed_matrix_t *output)
{
    int ret = 0;
    bool valid_window = false;

    if (start < 0 || end > (tsk_id_t) self->num_sites || start >= end) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    if (output != NULL && output->num_sites != self->num_sites) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    valid_window = true;
    ret = tsls_hmm_reset_traceback(self);
    if (ret != 0) {
        goto out;
    }
    ret = tsls_hmm_run(self, MODE_VITERBI, start, end, haplotype, NULL, output);
    if (ret != 0) {
        goto out;
    }
    ret = tsls_hmm_run_traceback(self, start, end, matched_haplotype, path_length,
        path_left, path_right, path_parent);
out:
    /* Reset the per-site state for the next call. Only the sites in the
     * window can have been set, and resetting just these matters for short
     * windows (tsinfer's early ancestors) on long sequences. */
    if (valid_window) {
        memset(self->traceback + start, 0,
            (size_t) (end - start) * sizeof(*self->traceback));
        memset(self->max_likelihood_node + start, 0xff,
            (size_t) (end - start) * sizeof(*self->max_likelihood_node));
    }
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

int
tsls_hmm_find_path(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    double *path_left, double *path_right, tsk_id_t *path_parent)
{
    return tsls_hmm_find_path_matrix(self, start, end, haplotype,
        matched_haplotype, path_length, path_left, path_right, path_parent, NULL);
}

int
tsls_hmm_forward(
    tsls_hmm_t *self, const allele_t *haplotype, compressed_matrix_t *output)
{
    int ret = 0;

    if (output->num_sites != self->num_sites) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    if (self->num_sites > 0) {
        ret = tsls_hmm_run(self, MODE_FORWARD, 0, (tsk_id_t) self->num_sites,
            haplotype, NULL, output);
    }
out:
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

int
tsls_hmm_backward(tsls_hmm_t *self, const allele_t *haplotype,
    const double *forward_normalisation_factor, compressed_matrix_t *output)
{
    int ret = 0;
    size_t j;

    if (output->num_sites != self->num_sites) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    for (j = 0; j < self->num_sites; j++) {
        if (!(forward_normalisation_factor[j] > 0)) {
            ret = TSLS_ERR_BAD_NORMALISATION_FACTOR;
            goto out;
        }
    }
    if (self->num_sites > 0) {
        ret = tsls_hmm_run(self, MODE_BACKWARD, 0, (tsk_id_t) self->num_sites,
            haplotype, forward_normalisation_factor, output);
    }
out:
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

double
tsls_hmm_get_mean_traceback_size(tsls_hmm_t *self)
{
    if (self->num_sites == 0) {
        return 0;
    }
    return (double) self->total_traceback_size / ((double) self->num_sites);
}

double
tsls_hmm_get_mean_num_distinct_values(tsls_hmm_t *self)
{
    if (self->num_sites == 0) {
        return 0;
    }
    return (double) self->total_distinct_values / ((double) self->num_sites);
}

size_t
tsls_hmm_get_total_memory(tsls_hmm_t *self)
{
    return self->traceback_allocator.total_size;
}
