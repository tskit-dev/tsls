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
remove_edge(edge_t edge, tsk_id_t *restrict parent, tsk_id_t *restrict left_child,
    tsk_id_t *restrict right_child, tsk_id_t *restrict left_sib,
    tsk_id_t *restrict right_sib)
{
    const tsk_id_t p = edge.parent;
    const tsk_id_t c = edge.child;
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
insert_edge(edge_t edge, tsk_id_t *restrict parent, tsk_id_t *restrict left_child,
    tsk_id_t *restrict right_child, tsk_id_t *restrict left_sib,
    tsk_id_t *restrict right_sib)
{
    const tsk_id_t p = edge.parent;
    const tsk_id_t c = edge.child;
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
 * allelic state set (see ancestor_matcher_set_root_allelic_state), so the
 * walk up the tree doesn't need to check for the root. This is the
 * innermost loop of the matcher, so there is deliberately no assert here
 * (the Python module is compiled with asserts on); the roots are checked
 * in ancestor_matcher_check_state. */
static inline allele_t
get_marked_allelic_state(
    const allele_t *restrict allelic_state, const tsk_id_t *restrict parent, tsk_id_t u)
{
    while (allelic_state[u] == TSK_NULL) {
        u = parent[u];
    }
    return allelic_state[u];
}

static inline bool
is_integer(double x)
{
    return (double) (coordinate_t) x == x;
}

/*===================================================================
 * Matcher indexes
 *===================================================================
 */

int
matcher_indexes_print_state(const matcher_indexes_t *self, FILE *out)
{
    size_t j;
    mutation_list_node_t *u;

    fprintf(out, "Matcher indexes state\n");
    fprintf(out, "flags = %d\n", (int) self->flags);
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_nodes = %d\n", (int) self->num_nodes);
    fprintf(out, "num_edges = %d\n", (int) self->num_edges);

    fprintf(out, "Sites = \n");
    fprintf(out, "site\tposition\tancestral_state\tnum_alleles\t(node, derived_state)\n");
    for (j = 0; j < self->num_sites; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t", (int) j, (int) self->sites.position[j],
            (int) self->sites.ancestral_state[j], (int) self->sites.num_alleles[j]);
        for (u = self->sites.mutations[j]; u != NULL; u = u->next) {
            fprintf(out, "(%d, %d) ", u->node, u->derived_state);
        }
        fprintf(out, "\n");
    }
    return 0;
}

static void
matcher_indexes_copy_edge_indexes(
    matcher_indexes_t *self, const tsk_table_collection_t *tables)
{
    tsk_size_t j;
    tsk_id_t k;
    edge_t e;
    const tsk_id_t *restrict I = tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict O = tables->indexes.edge_removal_order;
    const double *restrict edges_right = tables->edges.right;
    const double *restrict edges_left = tables->edges.left;
    const tsk_id_t *restrict edges_child = tables->edges.child;
    const tsk_id_t *restrict edges_parent = tables->edges.parent;

    for (j = 0; j < self->num_edges; j++) {
        k = I[j];
        e.left = (coordinate_t) edges_left[k];
        e.right = (coordinate_t) edges_right[k];
        e.parent = edges_parent[k];
        e.child = edges_child[k];
        self->left_index_edges[j] = e;

        k = O[j];
        e.left = (coordinate_t) edges_left[k];
        e.right = (coordinate_t) edges_right[k];
        e.parent = edges_parent[k];
        e.child = edges_child[k];
        self->right_index_edges[j] = e;
    }
}

static int WARN_UNUSED
matcher_indexes_add_mutation(
    matcher_indexes_t *self, tsk_id_t site, tsk_id_t node, allele_t derived_state)
{
    int ret = 0;
    mutation_list_node_t *list_node, *tail;

    list_node = tsi_blkalloc_get(&self->allocator, sizeof(mutation_list_node_t));
    if (list_node == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    list_node->node = node;
    list_node->derived_state = derived_state;
    list_node->next = NULL;
    if (self->sites.mutations[site] == NULL) {
        self->sites.mutations[site] = list_node;
    } else {
        tail = self->sites.mutations[site];
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = list_node;
    }
    self->num_mutations++;
out:
    return ret;
}

/* Copy the site and mutation data into the matcher indexes. The
 * ancestral_state (per site) and derived_state (per mutation) arrays
 * give the integer-encoded alleles; if NULL, the ancestral state is 0
 * and every mutation has derived state 1. Mutations must be in tskit
 * order, so that when two mutations at a site are on the same node the
 * later one (the child) takes precedence. */
static int
matcher_indexes_copy_mutation_data(matcher_indexes_t *self,
    const tsk_table_collection_t *tables, const tsk_size_t *num_alleles,
    const allele_t *ancestral_state, const allele_t *derived_state)
{
    int ret = 0;
    tsk_size_t j;
    tsk_id_t site;
    allele_t state;
    const double *restrict sites_position = tables->sites.position;
    const tsk_id_t *restrict mutations_site = tables->mutations.site;
    const tsk_id_t *restrict mutations_node = tables->mutations.node;
    const tsk_size_t total_mutations = tables->mutations.num_rows;
    coordinate_t *restrict converted_position = self->sites.position;

    for (j = 0; j < self->num_sites; j++) {
        converted_position[j] = (coordinate_t) sites_position[j];
        self->sites.mutations[j] = NULL;
        self->sites.num_alleles[j] = num_alleles != NULL ? num_alleles[j] : 2;
        state = ancestral_state != NULL ? ancestral_state[j] : 0;
        if (state < 0 || (tsk_size_t) state >= self->sites.num_alleles[j]) {
            ret = TSLS_ERR_BAD_ANCESTRAL_STATE;
            goto out;
        }
        self->sites.ancestral_state[j] = state;
    }
    converted_position[j] = (coordinate_t) tables->sequence_length;

    for (j = 0; j < total_mutations; j++) {
        site = mutations_site[j];
        state = derived_state != NULL ? derived_state[j] : 1;
        if (state < 0 || (tsk_size_t) state >= self->sites.num_alleles[site]) {
            ret = TSLS_ERR_BAD_MUTATION_DERIVED_STATE;
            goto out;
        }
        ret = matcher_indexes_add_mutation(self, site, mutations_node[j], state);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

static int
matcher_indexes_validate_tables(const tsk_table_collection_t *tables)
{
    tsk_size_t j;
    const tsk_id_t num_nodes = (tsk_id_t) tables->nodes.num_rows;
    const tsk_id_t num_sites = (tsk_id_t) tables->sites.num_rows;

    if (!is_integer(tables->sequence_length)) {
        return TSLS_ERR_BAD_COORDINATE;
    }
    for (j = 0; j < tables->edges.num_rows; j++) {
        if (tables->edges.parent[j] < 0 || tables->edges.parent[j] >= num_nodes
            || tables->edges.child[j] < 0 || tables->edges.child[j] >= num_nodes) {
            return TSLS_ERR_BAD_EDGE_NODE;
        }
        if (!is_integer(tables->edges.left[j]) || !is_integer(tables->edges.right[j])) {
            return TSLS_ERR_BAD_COORDINATE;
        }
    }
    for (j = 0; j < tables->sites.num_rows; j++) {
        if (!is_integer(tables->sites.position[j])) {
            return TSLS_ERR_BAD_COORDINATE;
        }
    }
    for (j = 0; j < tables->mutations.num_rows; j++) {
        if (tables->mutations.node[j] < 0 || tables->mutations.node[j] >= num_nodes) {
            return TSLS_ERR_BAD_MUTATION_NODE;
        }
        if (tables->mutations.site[j] < 0 || tables->mutations.site[j] >= num_sites) {
            return TSLS_ERR_BAD_MUTATION_SITE;
        }
    }
    return 0;
}

/* The tables must be sorted and indexed, as in a valid tree sequence. */
int
matcher_indexes_alloc(matcher_indexes_t *self, const tsk_table_collection_t *tables,
    const tsk_size_t *num_alleles, const allele_t *ancestral_state,
    const allele_t *derived_state, tsk_flags_t flags)
{
    int ret = 0;

    memset(self, 0, sizeof(*self));
    self->flags = flags;
    self->num_edges = tables->edges.num_rows;
    self->num_nodes = tables->nodes.num_rows;
    self->num_sites = tables->sites.num_rows;
    /* num_mutations is incremented as mutations are added */
    self->num_mutations = 0;

    ret = matcher_indexes_validate_tables(tables);
    if (ret != 0) {
        goto out;
    }

    self->left_index_edges = malloc(self->num_edges * sizeof(*self->left_index_edges));
    self->right_index_edges = malloc(self->num_edges * sizeof(*self->right_index_edges));
    self->sites.mutations = malloc(self->num_sites * sizeof(*self->sites.mutations));
    self->sites.num_alleles = malloc(self->num_sites * sizeof(*self->sites.num_alleles));
    self->sites.ancestral_state
        = malloc(self->num_sites * sizeof(*self->sites.ancestral_state));
    self->sites.position
        = malloc((self->num_sites + 1) * sizeof(*self->sites.position));
    if (self->left_index_edges == NULL || self->right_index_edges == NULL
        || self->sites.mutations == NULL || self->sites.position == NULL
        || self->sites.num_alleles == NULL || self->sites.ancestral_state == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    if (tsi_blkalloc_init(&self->allocator, 65536) != 0) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    matcher_indexes_copy_edge_indexes(self, tables);
    ret = matcher_indexes_copy_mutation_data(
        self, tables, num_alleles, ancestral_state, derived_state);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

int
matcher_indexes_free(matcher_indexes_t *self)
{
    tsi_safe_free(self->left_index_edges);
    tsi_safe_free(self->right_index_edges);
    tsi_safe_free(self->sites.mutations);
    tsi_safe_free(self->sites.position);
    tsi_safe_free(self->sites.num_alleles);
    tsi_safe_free(self->sites.ancestral_state);
    tsi_blkalloc_free(&self->allocator);
    return 0;
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
 * Ancestor matcher
 *===================================================================
 */

static inline bool
ancestor_matcher_in_tree(const ancestor_matcher_t *self, tsk_id_t u)
{
    return self->parent[u] != NULL_NODE || self->left_child[u] != NULL_NODE;
}

static void
ancestor_matcher_check_state(const ancestor_matcher_t *self)
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
        if (ancestor_matcher_in_tree(self, u)) {
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
ancestor_matcher_print_state(ancestor_matcher_t *self, FILE *out)
{
    int j, k;
    tsk_id_t u;

    fprintf(out, "Ancestor matcher state\n");
    fprintf(out, "site\trecomb_rate\tmut_rate\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        fprintf(
            out, "%d\t%f\t%f\n", j, self->recombination_rate[j], self->mismatch_rate[j]);
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

int
ancestor_matcher_alloc(ancestor_matcher_t *self,
    const matcher_indexes_t *matcher_indexes, const double *recombination_rate,
    const double *mismatch_rate, double likelihood_threshold, int flags)
{
    int ret = 0;
    const size_t N = matcher_indexes->num_nodes;

    memset(self, 0, sizeof(ancestor_matcher_t));
    self->flags = flags;
    self->likelihood_threshold = likelihood_threshold;
    self->matcher_indexes = matcher_indexes;
    self->num_sites = matcher_indexes->num_sites;
    self->num_nodes = N;
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

    if (self->recombination_rate == NULL || self->mismatch_rate == NULL
        || self->traceback == NULL || self->max_likelihood_node == NULL
        || self->parent == NULL || self->left_child == NULL || self->right_child == NULL
        || self->left_sib == NULL || self->right_sib == NULL || self->num_below == NULL
        || self->num_inheriting == NULL || self->recombination_required == NULL
        || self->likelihood == NULL || self->likelihood_cache == NULL
        || self->recombination_required_cache == NULL
        || self->likelihood_nodes == NULL || self->likelihood_nodes_tmp == NULL
        || self->values_tmp == NULL || self->allelic_state == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
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
ancestor_matcher_free(ancestor_matcher_t *self)
{
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
ancestor_matcher_set_pruning(ancestor_matcher_t *self, int mode, double parameter)
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
ancestor_matcher_sort_values(ancestor_matcher_t *self)
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
ancestor_matcher_prune(ancestor_matcher_t *self)
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
        ancestor_matcher_sort_values(self);
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
ancestor_matcher_count_distinct(ancestor_matcher_t *self)
{
    int j;
    size_t num_distinct;

    if (!(self->flags & TSLS_COUNT_DISTINCT) || self->num_likelihood_nodes == 0) {
        return;
    }
    ancestor_matcher_sort_values(self);
    num_distinct = 1;
    for (j = 1; j < self->num_likelihood_nodes; j++) {
        if (self->values_tmp[j] != self->values_tmp[j - 1]) {
            num_distinct++;
        }
    }
    self->total_distinct_values += num_distinct;
}

static inline void
ancestor_matcher_add_likelihood(ancestor_matcher_t *self, tsk_id_t u, double value)
{
    self->likelihood[u] = value;
    self->likelihood_nodes[self->num_likelihood_nodes] = u;
    self->num_likelihood_nodes++;
}

/* Remove the specified node from the list of nodes with likelihoods. */
static void
ancestor_matcher_delete_likelihood(ancestor_matcher_t *self, const tsk_id_t node)
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
ancestor_matcher_update_num_below(ancestor_matcher_t *self, tsk_id_t u, int64_t delta)
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
ancestor_matcher_get_inherited_likelihood(ancestor_matcher_t *self, tsk_id_t u)
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
ancestor_matcher_reset_likelihood_cache(ancestor_matcher_t *self, tsk_id_t u)
{
    double *restrict L_cache = self->likelihood_cache;
    const tsk_id_t *restrict parent = self->parent;

    while (u != NULL_NODE && L_cache[u] != CACHE_UNSET) {
        L_cache[u] = CACHE_UNSET;
        u = parent[u];
    }
}

/* Remove the specified edge from the tree, and if update_likelihoods is
 * true make sure that the child keeps its likelihood. Nodes that are
 * isolated after removing the edge are added to the isolated list; they
 * only leave the tree if they are still isolated after all the edges at
 * this position have been inserted (see ancestor_matcher_next_tree). */
static void
ancestor_matcher_remove_edge(ancestor_matcher_t *self, edge_t edge,
    bool update_likelihoods, tsk_id_t *restrict isolated, int *num_isolated)
{
    double *restrict L = self->likelihood;
    const tsk_id_t p = edge.parent;
    const tsk_id_t c = edge.child;

    if (self->track_num_below) {
        ancestor_matcher_update_num_below(self, p, -self->num_below[c]);
    }
    if (update_likelihoods && L[c] == NULL_LIKELIHOOD) {
        ancestor_matcher_add_likelihood(
            self, c, ancestor_matcher_get_inherited_likelihood(self, p));
    }
    remove_edge(edge, self->parent, self->left_child, self->right_child, self->left_sib,
        self->right_sib);
    if (update_likelihoods) {
        /* Each node can only become isolated once when removing edges, so
         * the isolated list has at most num_nodes entries. */
        if (!ancestor_matcher_in_tree(self, c)) {
            isolated[*num_isolated] = c;
            (*num_isolated)++;
        }
        if (!ancestor_matcher_in_tree(self, p)) {
            /* p was a root, and so has a likelihood */
            assert(L[p] >= 0);
            isolated[*num_isolated] = p;
            (*num_isolated)++;
        }
    }
}

/* Insert the specified edge into the tree. Nodes entering the tree get
 * likelihood 0, and so can only be reached by recombination. */
static void
ancestor_matcher_insert_edge(
    ancestor_matcher_t *self, edge_t edge, bool update_likelihoods)
{
    double *restrict L = self->likelihood;

    if (update_likelihoods) {
        if (L[edge.parent] == NOT_IN_TREE) {
            ancestor_matcher_add_likelihood(self, edge.parent, 0);
        }
        if (L[edge.child] == NOT_IN_TREE) {
            ancestor_matcher_add_likelihood(self, edge.child, 0);
        }
        /* Otherwise the child was a root and keeps its likelihood */
        assert(L[edge.child] >= 0);
    }
    insert_edge(edge, self->parent, self->left_child, self->right_child, self->left_sib,
        self->right_sib);
    if (self->track_num_below) {
        ancestor_matcher_update_num_below(self, edge.parent, self->num_below[edge.child]);
    }
}

/* The current position in the sequence of trees, moving left-to-right
 * (direction = 1) or right-to-left (direction = -1). */
typedef struct {
    int direction;
    int_fast32_t in_index;
    int_fast32_t out_index;
    coordinate_t left;
    coordinate_t right;
} tree_position_t;

static void
tree_position_init(
    tree_position_t *self, const matcher_indexes_t *matcher_indexes, int direction)
{
    const int_fast32_t M = (int_fast32_t) matcher_indexes->num_edges;
    const coordinate_t L = matcher_indexes->sites.position[matcher_indexes->num_sites];

    self->direction = direction;
    if (direction > 0) {
        self->in_index = 0;
        self->out_index = 0;
        self->left = 0;
        self->right = 0;
    } else {
        self->in_index = M - 1;
        self->out_index = M - 1;
        self->left = L;
        self->right = L;
    }
}

/* Move to the next tree in the direction of travel. */
static void
ancestor_matcher_next_tree(
    ancestor_matcher_t *self, tree_position_t *tp, bool update_likelihoods)
{
    const matcher_indexes_t *mi = self->matcher_indexes;
    const int_fast32_t M = (int_fast32_t) mi->num_edges;
    const coordinate_t sequence_length = mi->sites.position[mi->num_sites];
    const edge_t *restrict left_index = mi->left_index_edges;
    const edge_t *restrict right_index = mi->right_index_edges;
    tsk_id_t *restrict isolated = self->likelihood_nodes_tmp;
    double *restrict L = self->likelihood;
    int_fast32_t j, removal_start;
    int k, num_isolated = 0;
    coordinate_t pos;
    tsk_id_t u;

    if (tp->direction > 0) {
        pos = tp->right;
        removal_start = tp->out_index;
        while (tp->out_index < M && right_index[tp->out_index].right == pos) {
            ancestor_matcher_remove_edge(self, right_index[tp->out_index],
                update_likelihoods, isolated, &num_isolated);
            tp->out_index++;
        }
        if (update_likelihoods) {
            for (j = removal_start; j < tp->out_index; j++) {
                ancestor_matcher_reset_likelihood_cache(self, right_index[j].parent);
            }
        }
        while (tp->in_index < M && left_index[tp->in_index].left == pos) {
            ancestor_matcher_insert_edge(
                self, left_index[tp->in_index], update_likelihoods);
            tp->in_index++;
        }
        tp->left = pos;
        tp->right = sequence_length;
        if (tp->in_index < M) {
            tp->right = TSK_MIN(tp->right, left_index[tp->in_index].left);
        }
        if (tp->out_index < M) {
            tp->right = TSK_MIN(tp->right, right_index[tp->out_index].right);
        }
    } else {
        pos = tp->left;
        removal_start = tp->out_index;
        while (tp->out_index >= 0 && left_index[tp->out_index].left == pos) {
            ancestor_matcher_remove_edge(self, left_index[tp->out_index],
                update_likelihoods, isolated, &num_isolated);
            tp->out_index--;
        }
        if (update_likelihoods) {
            for (j = removal_start; j > tp->out_index; j--) {
                ancestor_matcher_reset_likelihood_cache(self, left_index[j].parent);
            }
        }
        while (tp->in_index >= 0 && right_index[tp->in_index].right == pos) {
            ancestor_matcher_insert_edge(
                self, right_index[tp->in_index], update_likelihoods);
            tp->in_index--;
        }
        tp->right = pos;
        tp->left = 0;
        if (tp->out_index >= 0) {
            tp->left = TSK_MAX(tp->left, left_index[tp->out_index].left);
        }
        if (tp->in_index >= 0) {
            tp->left = TSK_MAX(tp->left, right_index[tp->in_index].right);
        }
    }
    /* Nodes that are still isolated have left the tree. Nodes that were
     * isolated and then inserted again keep their likelihoods. */
    for (k = 0; k < num_isolated; k++) {
        u = isolated[k];
        if (!ancestor_matcher_in_tree(self, u)) {
            ancestor_matcher_delete_likelihood(self, u);
            L[u] = NOT_IN_TREE;
        }
    }
}

/* Move to the next tree in the direction of travel, updating only the tree
 * topology. This is the same as ancestor_matcher_next_tree with
 * update_likelihoods false when num_below isn't tracked, but avoids its
 * per-edge bookkeeping. It is used to build the first tree of a sweep,
 * which may mean processing most of the edges. */
static void
ancestor_matcher_next_tree_topology(ancestor_matcher_t *self, tree_position_t *tp)
{
    const matcher_indexes_t *mi = self->matcher_indexes;
    const int_fast32_t M = (int_fast32_t) mi->num_edges;
    const coordinate_t sequence_length = mi->sites.position[mi->num_sites];
    const edge_t *restrict left_index = mi->left_index_edges;
    const edge_t *restrict right_index = mi->right_index_edges;
    tsk_id_t *restrict parent = self->parent;
    tsk_id_t *restrict left_child = self->left_child;
    tsk_id_t *restrict right_child = self->right_child;
    tsk_id_t *restrict left_sib = self->left_sib;
    tsk_id_t *restrict right_sib = self->right_sib;
    int_fast32_t in_index = tp->in_index;
    int_fast32_t out_index = tp->out_index;
    coordinate_t pos;

    assert(!self->track_num_below);
    if (tp->direction > 0) {
        pos = tp->right;
        while (out_index < M && right_index[out_index].right == pos) {
            remove_edge(right_index[out_index], parent, left_child, right_child,
                left_sib, right_sib);
            out_index++;
        }
        while (in_index < M && left_index[in_index].left == pos) {
            insert_edge(left_index[in_index], parent, left_child, right_child, left_sib,
                right_sib);
            in_index++;
        }
        tp->left = pos;
        tp->right = sequence_length;
        if (in_index < M) {
            tp->right = TSK_MIN(tp->right, left_index[in_index].left);
        }
        if (out_index < M) {
            tp->right = TSK_MIN(tp->right, right_index[out_index].right);
        }
    } else {
        pos = tp->left;
        while (out_index >= 0 && left_index[out_index].left == pos) {
            remove_edge(left_index[out_index], parent, left_child, right_child,
                left_sib, right_sib);
            out_index--;
        }
        while (in_index >= 0 && right_index[in_index].right == pos) {
            insert_edge(right_index[in_index], parent, left_child, right_child,
                left_sib, right_sib);
            in_index--;
        }
        tp->right = pos;
        tp->left = 0;
        if (out_index >= 0) {
            tp->left = TSK_MAX(tp->left, left_index[out_index].left);
        }
        if (in_index >= 0) {
            tp->left = TSK_MAX(tp->left, right_index[in_index].right);
        }
    }
    tp->in_index = in_index;
    tp->out_index = out_index;
}

/* Reset the matcher and build the tree containing the specified position,
 * giving each root the specified initial likelihood. */
static void
ancestor_matcher_init_sweep(ancestor_matcher_t *self, tree_position_t *tp,
    int direction, coordinate_t position, double initial_value)
{
    const size_t N = self->num_nodes;
    tsk_id_t u;
    size_t j;

    memset(self->parent, 0xff, N * sizeof(*self->parent));
    memset(self->left_child, 0xff, N * sizeof(*self->left_child));
    memset(self->right_child, 0xff, N * sizeof(*self->right_child));
    memset(self->left_sib, 0xff, N * sizeof(*self->left_sib));
    memset(self->right_sib, 0xff, N * sizeof(*self->right_sib));
    memset(self->recombination_required, 0xff, N * sizeof(*self->recombination_required));
    for (j = 0; j < N; j++) {
        self->likelihood_cache[j] = CACHE_UNSET;
    }
    tree_position_init(tp, self->matcher_indexes, direction);
    if (self->track_num_below) {
        for (j = 0; j < N; j++) {
            self->num_below[j] = 1;
        }
        do {
            ancestor_matcher_next_tree(self, tp, false);
        } while (!(tp->left <= position && position < tp->right));
    } else {
        do {
            ancestor_matcher_next_tree_topology(self, tp);
        } while (!(tp->left <= position && position < tp->right));
    }

    self->num_likelihood_nodes = 0;
    for (u = 0; u < (tsk_id_t) N; u++) {
        if (self->parent[u] != NULL_NODE) {
            self->likelihood[u] = NULL_LIKELIHOOD;
        } else if (self->left_child[u] != NULL_NODE) {
            ancestor_matcher_add_likelihood(self, u, initial_value);
        } else {
            self->likelihood[u] = NOT_IN_TREE;
        }
    }
}

static inline void
ancestor_matcher_set_allelic_state(ancestor_matcher_t *self, const tsk_id_t site)
{
    mutation_list_node_t *mutation;

    for (mutation = self->matcher_indexes->sites.mutations[site]; mutation != NULL;
        mutation = mutation->next) {
        self->allelic_state[mutation->node] = mutation->derived_state;
    }
}

static inline void
ancestor_matcher_unset_allelic_state(ancestor_matcher_t *self, const tsk_id_t site)
{
    mutation_list_node_t *mutation;

    for (mutation = self->matcher_indexes->sites.mutations[site]; mutation != NULL;
        mutation = mutation->next) {
        self->allelic_state[mutation->node] = TSK_NULL;
    }
}

/* Set the allelic state of every root to the ancestral state, so that
 * get_marked_allelic_state finds a state for every node in the tree.
 * Every root has a likelihood, so the roots are found in the likelihood
 * list. This must be called before ancestor_matcher_set_allelic_state, so
 * that mutations above a root take precedence, and the likelihood list
 * must not change before ancestor_matcher_unset_root_allelic_state. */
static inline void
ancestor_matcher_set_root_allelic_state(ancestor_matcher_t *self, const tsk_id_t site)
{
    const allele_t ancestral_state = self->matcher_indexes->sites.ancestral_state[site];
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
ancestor_matcher_unset_root_allelic_state(ancestor_matcher_t *self)
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
ancestor_matcher_insert_mutation_likelihoods(
    ancestor_matcher_t *self, const tsk_id_t site)
{
    const double *restrict L = self->likelihood;
    const tsk_id_t *restrict parent = self->parent;
    mutation_list_node_t *mutation;
    tsk_id_t u;

    for (mutation = self->matcher_indexes->sites.mutations[site]; mutation != NULL;
        mutation = mutation->next) {
        if (L[mutation->node] == NULL_LIKELIHOOD) {
            u = mutation->node;
            while (L[u] == NULL_LIKELIHOOD) {
                u = parent[u];
                assert(u != NULL_NODE);
            }
            ancestor_matcher_add_likelihood(self, mutation->node, L[u]);
        }
    }
}

/* As ancestor_matcher_get_inherited_likelihood, but also return the
 * recombination_required flag that u inherits, caching it along the path
 * with the likelihood. */
static inline double
ancestor_matcher_get_inherited_likelihood_and_flag(
    ancestor_matcher_t *self, tsk_id_t u, int8_t *flag)
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
ancestor_matcher_coalesce_likelihoods(
    ancestor_matcher_t *self, bool check_recombination_required)
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
                L_p = ancestor_matcher_get_inherited_likelihood_and_flag(
                    self, p, &R_p);
                if (L[u] == L_p && R[u] == R_p) {
                    L[u] = NULL_LIKELIHOOD;
                }
            } else {
                L_p = ancestor_matcher_get_inherited_likelihood(self, p);
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
        ancestor_matcher_reset_likelihood_cache(self, cached_paths[j]);
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
check_haplotype_state(const ancestor_matcher_t *self, tsk_id_t site, allele_t state)
{
    const tsk_size_t num_alleles = self->matcher_indexes->sites.num_alleles[site];

    if (state < TSLS_MISSING_DATA || (state >= 0 && (tsk_size_t) state >= num_alleles)) {
        return TSLS_ERR_BAD_HAPLOTYPE_ALLELE;
    }
    return 0;
}

static inline double
ancestor_matcher_get_n(const ancestor_matcher_t *self)
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
ancestor_matcher_compute_num_inheriting(ancestor_matcher_t *self)
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
ancestor_matcher_total_likelihood(const ancestor_matcher_t *self)
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
ancestor_matcher_store_traceback(ancestor_matcher_t *self, const tsk_id_t site_id)
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
ancestor_matcher_process_site_viterbi(ancestor_matcher_t *self, const tsk_id_t site,
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
    const double n = ancestor_matcher_get_n(self);
    const double num_alleles = (double) self->matcher_indexes->sites.num_alleles[site];
    double max_L, p_no_recomb, p_recomb, p_t, p_e;
    tsk_id_t u, max_L_node;
    int j;

    ancestor_matcher_insert_mutation_likelihoods(self, site);
    ancestor_matcher_set_root_allelic_state(self, site);
    ancestor_matcher_set_allelic_state(self, site);
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
    ancestor_matcher_unset_allelic_state(self, site);
    ancestor_matcher_unset_root_allelic_state(self);
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
    ancestor_matcher_prune(self);
    ancestor_matcher_count_distinct(self);
    /* Remove redundant nodes before storing the traceback, keeping those
     * whose recombination_required flag differs from the inherited one.
     * Those are left for the next site, where their flags are recomputed. */
    ancestor_matcher_coalesce_likelihoods(self, true);
    ret = ancestor_matcher_store_traceback(self, site);
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
ancestor_matcher_process_site_forward(ancestor_matcher_t *self, const tsk_id_t site,
    const allele_t state, double *previous_total, compressed_matrix_t *output)
{
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    double *restrict L = self->likelihood;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = ancestor_matcher_get_n(self);
    const double num_alleles = (double) self->matcher_indexes->sites.num_alleles[site];
    double total, normalisation_factor, p_e;
    tsk_id_t u;
    int j;

    ancestor_matcher_insert_mutation_likelihoods(self, site);
    ancestor_matcher_compute_num_inheriting(self);
    total = *previous_total;
    if (total < 0) {
        total = ancestor_matcher_total_likelihood(self);
    }
    ancestor_matcher_set_root_allelic_state(self, site);
    ancestor_matcher_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p_e = emission_probability(
            get_marked_allelic_state(allelic_state, parent, u), state, mu, num_alleles);
        L[u] = (L[u] * (1 - rho) + rho / n * total) * p_e;
    }
    ancestor_matcher_unset_allelic_state(self, site);
    ancestor_matcher_unset_root_allelic_state(self);
    normalisation_factor = ancestor_matcher_total_likelihood(self);
    if (!(normalisation_factor > 0)) {
        ret = TSLS_ERR_MATCH_IMPOSSIBLE;
        goto out;
    }
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] = TSK_MAX(L[u] / normalisation_factor, self->likelihood_threshold);
    }
    ancestor_matcher_prune(self);
    *previous_total = ancestor_matcher_total_likelihood(self);
    ret = compressed_matrix_store_site(output, site, normalisation_factor,
        self->num_likelihood_nodes, L_nodes, L, NULL);
    if (ret != 0) {
        goto out;
    }
    ancestor_matcher_coalesce_likelihoods(self, false);
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
ancestor_matcher_process_site_backward(ancestor_matcher_t *self, const tsk_id_t site,
    const allele_t state, const double *forward_normalisation_factor,
    backward_transition_t *transition, compressed_matrix_t *output)
{
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    double *restrict L = self->likelihood;
    const double mu = self->mismatch_rate[site];
    const double n = ancestor_matcher_get_n(self);
    const double num_alleles = (double) self->matcher_indexes->sites.num_alleles[site];
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
        ancestor_matcher_prune(self);
    }
    ancestor_matcher_insert_mutation_likelihoods(self, site);
    ret = compressed_matrix_store_site(output, site, forward_normalisation_factor[site],
        self->num_likelihood_nodes, L_nodes, L, NULL);
    if (ret != 0) {
        goto out;
    }
    ancestor_matcher_compute_num_inheriting(self);
    ancestor_matcher_set_root_allelic_state(self, site);
    ancestor_matcher_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u] *= emission_probability(
            get_marked_allelic_state(allelic_state, parent, u), state, mu, num_alleles);
    }
    ancestor_matcher_unset_allelic_state(self, site);
    ancestor_matcher_unset_root_allelic_state(self);
    transition->pending = true;
    transition->total = ancestor_matcher_total_likelihood(self);
    transition->rho = self->recombination_rate[site];
    transition->normalisation_factor = forward_normalisation_factor[site];
    ancestor_matcher_coalesce_likelihoods(self, false);
out:
    return ret;
}

/* Run the specified algorithm over sites [start, end). */
static int WARN_UNUSED
ancestor_matcher_run(ancestor_matcher_t *self, int mode, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, const double *forward_normalisation_factor,
    compressed_matrix_t *output)
{
    int ret = 0;
    tree_position_t tp;
    backward_transition_t transition;
    const coordinate_t *restrict position = self->matcher_indexes->sites.position;
    const int direction = mode == MODE_BACKWARD ? -1 : 1;
    tsk_id_t site, first_site;
    double previous_total = -1;

    for (site = start; site < end; site++) {
        ret = check_haplotype_state(self, site, haplotype[site]);
        if (ret != 0) {
            goto out;
        }
    }
    self->track_num_below = mode != MODE_VITERBI;
    first_site = direction > 0 ? start : end - 1;
    ancestor_matcher_init_sweep(self, &tp, direction, position[first_site], 1.0);
    transition.pending = false;

    for (site = first_site; start <= site && site < end; site += direction) {
        while (!(tp.left <= position[site] && position[site] < tp.right)) {
            ancestor_matcher_next_tree(self, &tp, true);
        }
        if (self->flags & TSLS_EXTENDED_CHECKS) {
            ancestor_matcher_check_state(self);
        }
        if (mode == MODE_VITERBI) {
            ret = ancestor_matcher_process_site_viterbi(
                self, site, haplotype[site], output);
        } else if (mode == MODE_FORWARD) {
            ret = ancestor_matcher_process_site_forward(
                self, site, haplotype[site], &previous_total, output);
        } else {
            ret = ancestor_matcher_process_site_backward(self, site, haplotype[site],
                forward_normalisation_factor, &transition, output);
        }
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* Resets the recombination_required array from the traceback at the specified site.
 */
static inline void
ancestor_matcher_set_recombination_required(
    ancestor_matcher_t *self, tsk_id_t site, int8_t *restrict recombination_required)
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
ancestor_matcher_unset_recombination_required(
    ancestor_matcher_t *self, tsk_id_t site, int8_t *restrict recombination_required)
{
    int j;
    const tsk_id_t *restrict node = self->traceback[site].node;
    const int size = self->traceback[site].size;

    for (j = 0; j < size; j++) {
        recombination_required[node[j]] = -1;
    }
}

static void
ancestor_matcher_run_traceback(ancestor_matcher_t *self, tsk_id_t start, tsk_id_t end,
    allele_t *restrict match, size_t *path_length_out, tsk_id_t *restrict path_left,
    tsk_id_t *restrict path_right, tsk_id_t *restrict path_parent)
{
    tsk_id_t site;
    edge_t edge;
    tsk_id_t u, max_likelihood_node;
    coordinate_t left, right, pos, start_pos, end_pos;
    tsk_id_t *restrict parent = self->parent;
    const allele_t *restrict allelic_state = self->allelic_state;
    int8_t *restrict recombination_required = self->recombination_required;
    const edge_t *restrict in = self->matcher_indexes->right_index_edges;
    const edge_t *restrict out = self->matcher_indexes->left_index_edges;
    const coordinate_t *restrict sites_position = self->matcher_indexes->sites.position;
    const allele_t *restrict ancestral_state
        = self->matcher_indexes->sites.ancestral_state;
    const coordinate_t sequence_length = sites_position[self->num_sites];
    int_fast32_t in_index = (int_fast32_t) self->matcher_indexes->num_edges - 1;
    int_fast32_t out_index = (int_fast32_t) self->matcher_indexes->num_edges - 1;
    size_t path_length = 0;

    start_pos = start == 0 ? 0 : sites_position[start];
    end_pos = sites_position[end];

    /* Prepare for the traceback and get the memory ready for recording
     * the output edges. */
    path_right[path_length] = end_pos;
    max_likelihood_node = self->max_likelihood_node[end - 1];
    assert(max_likelihood_node != NULL_NODE);
    path_parent[path_length] = max_likelihood_node;

    /* Now go through the trees in reverse and run the traceback */
    memset(parent, 0xff, self->num_nodes * sizeof(*parent));
    memset(
        recombination_required, 0xff, self->num_nodes * sizeof(*recombination_required));
    pos = sequence_length;
    site = (tsk_id_t) self->num_sites - 1;

    while (pos > start_pos) {
        while (out_index >= 0 && out[out_index].left == pos) {
            edge = out[out_index];
            out_index--;
            parent[edge.child] = NULL_NODE;
        }
        while (in_index >= 0 && in[in_index].right == pos) {
            edge = in[in_index];
            in_index--;
            parent[edge.child] = edge.parent;
        }
        right = pos;
        left = 0;
        if (out_index >= 0) {
            left = TSK_MAX(left, out[out_index].left);
        }
        if (in_index >= 0) {
            left = TSK_MAX(left, in[in_index].right);
        }
        pos = left;

        /* The tree is ready; perform the traceback at each site in this tree */
        assert(left < right);
        for (; site >= 0 && left <= sites_position[site] && sites_position[site] < right;
            site--) {
            if (start_pos <= sites_position[site] && sites_position[site] < end_pos) {
                ancestor_matcher_set_allelic_state(self, site);
                u = path_parent[path_length];
                match[site] = get_allelic_state(
                    allelic_state, parent, u, ancestral_state[site]);
                ancestor_matcher_unset_allelic_state(self, site);

                /* Mark the traceback nodes on the tree */
                ancestor_matcher_set_recombination_required(
                    self, site, recombination_required);

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
                ancestor_matcher_unset_recombination_required(
                    self, site, recombination_required);
            }
        }
    }
    path_left[path_length] = start_pos;
    path_length++;
    *path_length_out = path_length;
}

static int
ancestor_matcher_reset_traceback(ancestor_matcher_t *self)
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
ancestor_matcher_find_path_matrix(ancestor_matcher_t *self, tsk_id_t start,
    tsk_id_t end, const allele_t *haplotype, allele_t *matched_haplotype,
    size_t *path_length, tsk_id_t *path_left, tsk_id_t *path_right,
    tsk_id_t *path_parent, compressed_matrix_t *output)
{
    int ret = 0;

    if (start < 0 || end > (tsk_id_t) self->num_sites || start >= end) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    if (output != NULL && output->num_sites != self->num_sites) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    ret = ancestor_matcher_reset_traceback(self);
    if (ret != 0) {
        goto out;
    }
    ret = ancestor_matcher_run(self, MODE_VITERBI, start, end, haplotype, NULL, output);
    if (ret != 0) {
        goto out;
    }
    ancestor_matcher_run_traceback(self, start, end, matched_haplotype, path_length,
        path_left, path_right, path_parent);
out:
    /* Reset some memory for the next call */
    memset(self->traceback, 0, self->num_sites * sizeof(*self->traceback));
    memset(self->max_likelihood_node, 0xff,
        self->num_sites * sizeof(*self->max_likelihood_node));
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

int
ancestor_matcher_find_path(ancestor_matcher_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    tsk_id_t *path_left, tsk_id_t *path_right, tsk_id_t *path_parent)
{
    return ancestor_matcher_find_path_matrix(self, start, end, haplotype,
        matched_haplotype, path_length, path_left, path_right, path_parent, NULL);
}

int
ancestor_matcher_forward(
    ancestor_matcher_t *self, const allele_t *haplotype, compressed_matrix_t *output)
{
    int ret = 0;

    if (output->num_sites != self->num_sites) {
        ret = TSLS_ERR_BAD_SITE_RANGE;
        goto out;
    }
    if (self->num_sites > 0) {
        ret = ancestor_matcher_run(self, MODE_FORWARD, 0, (tsk_id_t) self->num_sites,
            haplotype, NULL, output);
    }
out:
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

int
ancestor_matcher_backward(ancestor_matcher_t *self, const allele_t *haplotype,
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
        ret = ancestor_matcher_run(self, MODE_BACKWARD, 0, (tsk_id_t) self->num_sites,
            haplotype, forward_normalisation_factor, output);
    }
out:
    memset(self->allelic_state, 0xff, self->num_nodes * sizeof(*self->allelic_state));
    return ret;
}

double
ancestor_matcher_get_mean_traceback_size(ancestor_matcher_t *self)
{
    if (self->num_sites == 0) {
        return 0;
    }
    return (double) self->total_traceback_size / ((double) self->num_sites);
}

double
ancestor_matcher_get_mean_num_distinct_values(ancestor_matcher_t *self)
{
    if (self->num_sites == 0) {
        return 0;
    }
    return (double) self->total_distinct_values / ((double) self->num_sites);
}

size_t
ancestor_matcher_get_total_memory(ancestor_matcher_t *self)
{
    return self->traceback_allocator.total_size;
}
