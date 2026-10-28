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
 * The same machinery runs the Viterbi, forward and backward algorithms,
 * for the model described in docs/likelihood_model.md. At each site we
 * recombine with probability rho, and then copy a node chosen uniformly
 * from all n nodes of the tree sequence (the path ends if it isn't in the
 * tree); otherwise we stay on the same node. So the value of a node becomes
 * p * (1 - rho + rho / n) or rho / n (Viterbi), or p * (1 - rho) + rho / n * S
 * (forward and backward, where S is the total over all states at the
 * previous site). The forward algorithm starts from a uniform distribution
 * over the n nodes. Values are normalised at each site (by the maximum for
 * Viterbi and the sum for forward). In Viterbi, values below prune_threshold
 * are then set to likelihood_threshold; in all the algorithms, values below
 * likelihood_threshold are set to it.
 *
 * When a node enters the tree it gets the value 0, and so it can only be
 * reached by recombination. When a node leaves the tree, the probability
 * of continuing from it is only the probability of recombination. So a node
 * can only be copied from one site to the next without recombination if it
 * is in the tree over the whole interval between them.
 */

#if defined(__linux__)
/* For posix_memalign and madvise */
#define _DEFAULT_SOURCE
#endif

#include "tsls.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

/* For the innermost helpers, which GCC doesn't always inline. Inlining
 * tsls_hmm_coalesce_likelihoods also specialises it for its constant
 * check_recombination_required argument. */
#ifdef __GNUC__
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

/* Allocate an array that is accessed at random, such as the per-node arrays
 * walked up the tree. On Linux, large arrays are mapped directly, aligned
 * to 2 MiB, and backed by transparent huge pages where available
 * (MADV_HUGEPAGE), which greatly reduces TLB misses in the walks. They are
 * mapped directly because memory from malloc may already have been touched
 * with small pages, which madvise doesn't change. Free with tsls_large_free,
 * with the same size. */
#define HUGE_PAGE_SIZE ((size_t) 2 * 1024 * 1024)

static inline bool
tsls_large_alloc_uses_mmap(size_t size)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    return size >= HUGE_PAGE_SIZE / 2;
#else
    (void) size;
    return false;
#endif
}

static void *
tsls_large_alloc(size_t size)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    size_t rounded, head, tail;
    char *p, *aligned;

    if (tsls_large_alloc_uses_mmap(size)) {
        rounded = (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
        p = mmap(NULL, rounded + HUGE_PAGE_SIZE, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            return NULL;
        }
        /* Trim the mapping to a range aligned to the huge page size */
        aligned = (char *) (((uintptr_t) p + HUGE_PAGE_SIZE - 1)
                            & ~((uintptr_t) HUGE_PAGE_SIZE - 1));
        head = (size_t) (aligned - p);
        tail = HUGE_PAGE_SIZE - head;
        if (head > 0) {
            (void) munmap(p, head);
        }
        if (tail > 0) {
            (void) munmap(aligned + rounded, tail);
        }
        /* Huge pages are only advice, so failure doesn't matter */
        (void) madvise(aligned, rounded, MADV_HUGEPAGE);
        return aligned;
    }
#endif
    return malloc(size);
}

static void
tsls_large_free(void *p, size_t size)
{
    if (p == NULL) {
        return;
    }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (tsls_large_alloc_uses_mmap(size)) {
        (void) munmap(p, (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1));
        return;
    }
#endif
    free(p);
}

/* Free an array of N + 1 elements allocated with tsls_large_alloc */
#define tsls_large_free_array(array, n)                                              \
    do {                                                                                \
        tsls_large_free(array, ((n) + 1) * sizeof(*(array)));                           \
        array = NULL;                                                                   \
    } while (0)

#define MODE_VITERBI  0
#define MODE_FORWARD  1
#define MODE_BACKWARD 2

static inline void
remove_edge(tsk_id_t p, tsk_id_t c, tsls_node_t *restrict tree,
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
    tree[c].parent = NULL_NODE;
    left_sib[c] = NULL_NODE;
    right_sib[c] = NULL_NODE;
}

static inline void
insert_edge(tsk_id_t p, tsk_id_t c, tsls_node_t *restrict tree,
    tsk_id_t *restrict left_child, tsk_id_t *restrict right_child,
    tsk_id_t *restrict left_sib, tsk_id_t *restrict right_sib)
{
    const tsk_id_t u = right_child[p];

    tree[c].parent = p;
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

/* During the forward and backward sweeps, the allelic state of every node
 * without a parent is ROOT_ALLELE, unless it has a mutation at the current
 * site. insert_edge and remove_edge keep this up to date. */
#define ROOT_ALLELE (-2)

/* Returns the allelic state of node u, given a tree in which the nodes with
 * mutations at the current site have their allelic states set and the roots
 * are marked with ROOT_ALLELE, so the walk up the tree doesn't need to
 * check for the root. Every node on the path from u is also given the state
 * found, so that later walks through the path stop there, and recorded in
 * touched so that clear_touched_allelic_state can unset them. Looking up
 * the states of many nodes then costs the size of the union of their paths
 * up to a mutation or root, rather than the sum of the path lengths. Only
 * nodes without a state are written, so the touched nodes are never roots
 * or mutation nodes, and each is touched at most once. This is the
 * innermost loop of the HMM, so there is deliberately no assert here (the
 * Python module is compiled with asserts on); the marks are checked in
 * tsls_hmm_check_state. */
static ALWAYS_INLINE allele_t
get_memoised_allelic_state(tsls_node_t *restrict tree, tsk_id_t u,
    allele_t ancestral_state, tsk_id_t *restrict touched, int *num_touched)
{
    allele_t state;
    tsk_id_t v = u;
    int n = *num_touched;

    while (tree[v].allelic_state == TSK_NULL) {
        v = tree[v].parent;
    }
    state = tree[v].allelic_state;
    while (u != v) {
        tree[u].allelic_state = state;
        touched[n] = u;
        n++;
        u = tree[u].parent;
    }
    *num_touched = n;
    return likely(state != ROOT_ALLELE) ? state : ancestral_state;
}

static inline void
clear_touched_allelic_state(
    tsls_node_t *restrict tree, const tsk_id_t *restrict touched, int num_touched)
{
    int j;

    for (j = 0; j < num_touched; j++) {
        tree[touched[j]].allelic_state = TSK_NULL;
    }
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

/* Store the likelihoods of the specified nodes at this site, and if
 * with_flags is true their recombination_required flags. */
static int WARN_UNUSED
compressed_matrix_store_site(compressed_matrix_t *self, tsk_id_t site,
    double normalisation_factor, int num_values, const tsk_id_t *restrict nodes,
    const tsls_likelihood_t *restrict L, const tsls_node_t *restrict tree,
    bool with_flags)
{
    int ret = 0;
    int j;
    tsk_id_t *restrict nodes_out;
    double *restrict values_out;
    int8_t *restrict flags_out = NULL;

    if (with_flags) {
        flags_out = tsi_blkalloc_get(
            &self->allocator, aligned_size((size_t) num_values * sizeof(int8_t)));
        if (flags_out == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        for (j = 0; j < num_values; j++) {
            flags_out[j] = tree[nodes[j]].recombination_required;
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
        values_out[j] = L[nodes[j]].value;
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
 * Index
 *===================================================================
 */

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
tsls_index_encode_alleles(
    tsls_index_t *self, const char **alleles, tsk_size_t num_global_alleles)
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
    tsk_size_t mutation_offset = 0;
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
        /* Mutations are sorted by site, so the mutations of each site are
         * contiguous in the mutation table */
        self->site_mutations_offset[j] = mutation_offset;
        mutation_offset += ts->site_mutations_length[j];
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
            assert(mutation->id == (tsk_id_t) (self->site_mutations_offset[j] + k));
            self->mutation_node[mutation->id] = mutation->node;
            self->mutation_derived_state[mutation->id] = (allele_t) code;
        }
        /* A site always has at least two alleles, so that a haplotype can
         * differ from a site with no mutations (such as a tsinfer
         * ancestor's focal sites, which are not yet in the tree sequence). */
        self->num_alleles[j] = TSK_MAX(n, 2);
    }
    self->site_mutations_offset[self->num_sites] = mutation_offset;
out:
    return ret;
}

/* Copy the edges in tskit's insertion and removal orders, so that the
 * tree position's ranges can be read sequentially. */
static void
tsls_index_copy_edges(tsls_index_t *self)
{
    const tsk_table_collection_t *tables = self->tree_sequence->tables;
    const tsk_id_t *restrict insertion_order = tables->indexes.edge_insertion_order;
    const tsk_id_t *restrict removal_order = tables->indexes.edge_removal_order;
    const tsk_id_t *restrict edge_parent = tables->edges.parent;
    const tsk_id_t *restrict edge_child = tables->edges.child;
    const double *restrict edge_left = tables->edges.left;
    const double *restrict edge_right = tables->edges.right;
    tsls_edge_t *restrict insertion_edges = self->insertion_edges;
    tsls_edge_t *restrict removal_edges = self->removal_edges;
    size_t j;
    tsk_id_t e;

    for (j = 0; j < self->num_edges; j++) {
        e = insertion_order[j];
        insertion_edges[j].parent = edge_parent[e];
        insertion_edges[j].child = edge_child[e];
        insertion_edges[j].coordinate = edge_right[e];
        e = removal_order[j];
        removal_edges[j].parent = edge_parent[e];
        removal_edges[j].child = edge_child[e];
        removal_edges[j].coordinate = edge_left[e];
    }
}

/* Return the copied edges for the specified order (one of the ranges of a
 * tree position). */
static inline const tsls_edge_t *
tsls_index_get_edges(const tsls_index_t *self, const tsk_id_t *order)
{
    return order == self->tree_sequence->tables->indexes.edge_insertion_order
               ? self->insertion_edges
               : self->removal_edges;
}

/* Returns true if the specified edge in the in range of a tree position
 * just seeked from null (see seek_tree_position) covers x. Edges in the
 * range all have left <= x in insertion order (a forward seek) and right > x
 * in removal order (a backward seek), so one coordinate is enough. */
static inline bool
seek_edge_covers(const tsk_tree_position_t *tp, const tsls_edge_t *edge, double x)
{
    return tp->direction == TSK_DIR_FORWARD ? edge->coordinate > x
                                            : edge->coordinate <= x;
}

int
tsls_index_init(tsls_index_t *self, const tsk_treeseq_t *tree_sequence,
    const char **alleles, tsk_size_t num_alleles)
{
    int ret = 0;
    const size_t num_mutations = tsk_treeseq_get_num_mutations(tree_sequence);

    memset(self, 0, sizeof(*self));
    self->tree_sequence = tree_sequence;
    self->num_nodes = tsk_treeseq_get_num_nodes(tree_sequence);
    self->num_sites = tsk_treeseq_get_num_sites(tree_sequence);
    self->num_edges = tsk_treeseq_get_num_edges(tree_sequence);
    self->sites_position = tree_sequence->tables->sites.position;
    self->sequence_length = tsk_treeseq_get_sequence_length(tree_sequence);
    self->num_alleles = malloc((self->num_sites + 1) * sizeof(*self->num_alleles));
    self->ancestral_state
        = malloc((self->num_sites + 1) * sizeof(*self->ancestral_state));
    self->site_mutations_offset
        = malloc((self->num_sites + 1) * sizeof(*self->site_mutations_offset));
    self->mutation_node = malloc((num_mutations + 1) * sizeof(*self->mutation_node));
    self->mutation_derived_state
        = malloc((num_mutations + 1) * sizeof(*self->mutation_derived_state));
    self->insertion_edges
        = tsls_large_alloc((self->num_edges + 1) * sizeof(*self->insertion_edges));
    self->removal_edges
        = tsls_large_alloc((self->num_edges + 1) * sizeof(*self->removal_edges));
    if (self->num_alleles == NULL || self->ancestral_state == NULL
        || self->site_mutations_offset == NULL || self->mutation_node == NULL
        || self->mutation_derived_state == NULL || self->insertion_edges == NULL
        || self->removal_edges == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    ret = tsls_index_encode_alleles(self, alleles, num_alleles);
    if (ret != 0) {
        goto out;
    }
    tsls_index_copy_edges(self);
out:
    return ret;
}

int
tsls_index_free(tsls_index_t *self)
{
    tsi_safe_free(self->num_alleles);
    tsi_safe_free(self->ancestral_state);
    tsi_safe_free(self->site_mutations_offset);
    tsi_safe_free(self->mutation_node);
    tsi_safe_free(self->mutation_derived_state);
    tsls_large_free_array(self->insertion_edges, self->num_edges);
    tsls_large_free_array(self->removal_edges, self->num_edges);
    return 0;
}

/*===================================================================
 * HMM
 *===================================================================
 */

static inline bool
tsls_hmm_in_tree(const tsls_hmm_t *self, tsk_id_t u)
{
    return self->tree[u].parent != NULL_NODE || self->left_child[u] != NULL_NODE;
}

static void
tsls_hmm_check_state(const tsls_hmm_t *self)
{
    int num_likelihoods = 0;
    int j;
    tsk_id_t u;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        assert(self->likelihood[u].value >= 0);
    }
    for (u = 0; u < (tsk_id_t) self->num_nodes; u++) {
        if (self->likelihood[u].value >= 0) {
            num_likelihoods++;
        }
        if (tsls_hmm_in_tree(self, u)) {
            assert(self->likelihood[u].value != NOT_IN_TREE);
            if (self->tree[u].parent == NULL_NODE) {
                /* Roots always have a likelihood */
                assert(self->likelihood[u].value >= 0);
            }
        } else {
            assert(self->likelihood[u].value == NOT_IN_TREE);
        }
        assert(self->tree[u].allelic_state
               == (self->tree[u].parent == NULL_NODE ? ROOT_ALLELE : TSK_NULL));
        assert(self->tree[u].cache_stamp != self->cache_stamp);
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
            (int) self->index->num_alleles[j], (int) self->index->ancestral_state[j]);
    }
    fprintf(out, "tree = \n");
    fprintf(out, "id\tparent\tlchild\trchild\tlsib\trsib\tlikelihood\n");
    for (j = 0; j < (int) self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%d\t%d\t%d\t%f\n", (int) j, self->tree[j].parent,
            self->left_child[j], self->right_child[j], self->left_sib[j],
            self->right_sib[j], self->likelihood[j].value);
    }
    fprintf(out, "likelihood nodes\n");
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        fprintf(out, "\t%d -> %f\n", u, self->likelihood[u].value);
    }
    fprintf(out, "traceback\n");
    for (j = 0; j < (int) self->num_sites; j++) {
        fprintf(out, "\t%d:%d (%d)\t", (int) j, self->max_likelihood_node[j],
            self->traceback[j].size);
        for (k = 0; k < self->traceback[j].size; k++) {
            fprintf(out, "(%d, %d)", self->traceback[j].node[k],
                self->traceback[j].previous_flag[k]);
        }
        fprintf(out, "\n");
    }
    tsi_blkalloc_print_state(&self->traceback_allocator, out);
    return 0;
}

int
tsls_hmm_alloc(tsls_hmm_t *self, const tsls_index_t *index,
    const double *recombination_rate, const double *mismatch_rate,
    double likelihood_threshold, int flags)
{
    int ret = 0;
    const size_t N = index->num_nodes;

    memset(self, 0, sizeof(tsls_hmm_t));
    self->flags = flags;
    self->likelihood_threshold = likelihood_threshold;
    self->index = index;
    self->tree_sequence = index->tree_sequence;
    self->num_sites = index->num_sites;
    self->num_nodes = N;
    self->sites_position = index->sites_position;
    self->sequence_length = index->sequence_length;
    self->recombination_rate
        = malloc((self->num_sites + 1) * sizeof(*self->recombination_rate));
    self->mismatch_rate = malloc((self->num_sites + 1) * sizeof(*self->mismatch_rate));
    self->traceback = calloc(self->num_sites + 1, sizeof(*self->traceback));
    self->max_likelihood_node = malloc((self->num_sites + 1) * sizeof(tsk_id_t));

    self->tree = tsls_large_alloc((N + 1) * sizeof(*self->tree));
    self->likelihood = tsls_large_alloc((N + 1) * sizeof(*self->likelihood));
    self->left_child = tsls_large_alloc((N + 1) * sizeof(*self->left_child));
    self->right_child = tsls_large_alloc((N + 1) * sizeof(*self->right_child));
    self->left_sib = tsls_large_alloc((N + 1) * sizeof(*self->left_sib));
    self->right_sib = tsls_large_alloc((N + 1) * sizeof(*self->right_sib));
    self->num_below = tsls_large_alloc((N + 1) * sizeof(*self->num_below));
    self->num_inheriting = tsls_large_alloc((N + 1) * sizeof(*self->num_inheriting));
    self->likelihood_nodes = tsls_large_alloc((N + 1) * sizeof(*self->likelihood_nodes));
    self->likelihood_nodes_tmp = tsls_large_alloc((N + 1) * sizeof(*self->likelihood_nodes_tmp));
    self->values_tmp = tsls_large_alloc((N + 1) * sizeof(*self->values_tmp));
    self->traceback_flags_tmp
        = tsls_large_alloc((N + 1) * sizeof(*self->traceback_flags_tmp));
    self->removed_nodes = tsls_large_alloc((N + 1) * sizeof(*self->removed_nodes));
    self->preorder = tsls_large_alloc((N + 1) * sizeof(*self->preorder));
    self->subtree_end = tsls_large_alloc((N + 1) * sizeof(*self->subtree_end));
    self->changed = tsls_large_alloc((N + 1) * sizeof(*self->changed));
    self->changed_next = tsls_large_alloc((N + 1) * sizeof(*self->changed_next));
    self->site_nodes = tsls_large_alloc((N + 1) * sizeof(*self->site_nodes));
    self->stamp = tsls_large_alloc((N + 1) * sizeof(*self->stamp));

    if (self->recombination_rate == NULL
        || self->mismatch_rate == NULL || self->traceback == NULL
        || self->max_likelihood_node == NULL || self->tree == NULL
        || self->likelihood == NULL
        || self->left_child == NULL || self->right_child == NULL
        || self->left_sib == NULL || self->right_sib == NULL || self->num_below == NULL
        || self->num_inheriting == NULL || self->likelihood_nodes == NULL
        || self->likelihood_nodes_tmp == NULL || self->values_tmp == NULL
        || self->traceback_flags_tmp == NULL || self->removed_nodes == NULL
        || self->preorder == NULL || self->subtree_end == NULL
        || self->changed == NULL || self->changed_next == NULL
        || self->site_nodes == NULL || self->stamp == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    /* Alloc in blocks of at least 4MiB, and large enough to hold the
     * traceback delta for a site at which every node changes. */
    self->traceback_block_size
        = TSK_MAX(4 * 1024 * 1024, aligned_size((N + 1) * sizeof(tsk_id_t)) * 2);
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
    memset(self->max_likelihood_node, 0xff, self->num_sites * sizeof(tsk_id_t));
    memset(self->stamp, 0, (N + 1) * sizeof(*self->stamp));
    self->current_stamp = 1;
out:
    return ret;
}

int
tsls_hmm_free(tsls_hmm_t *self)
{
    tsi_safe_free(self->recombination_rate);
    tsi_safe_free(self->mismatch_rate);
    tsls_large_free_array(self->tree, self->num_nodes);
    tsls_large_free_array(self->likelihood, self->num_nodes);
    tsls_large_free_array(self->left_child, self->num_nodes);
    tsls_large_free_array(self->right_child, self->num_nodes);
    tsls_large_free_array(self->left_sib, self->num_nodes);
    tsls_large_free_array(self->right_sib, self->num_nodes);
    tsls_large_free_array(self->num_below, self->num_nodes);
    tsls_large_free_array(self->num_inheriting, self->num_nodes);
    tsls_large_free_array(self->likelihood_nodes, self->num_nodes);
    tsls_large_free_array(self->likelihood_nodes_tmp, self->num_nodes);
    tsls_large_free_array(self->values_tmp, self->num_nodes);
    tsls_large_free_array(self->traceback_flags_tmp, self->num_nodes);
    tsls_large_free_array(self->removed_nodes, self->num_nodes);
    tsls_large_free_array(self->preorder, self->num_nodes);
    tsls_large_free_array(self->subtree_end, self->num_nodes);
    tsls_large_free_array(self->changed, self->num_nodes);
    tsls_large_free_array(self->changed_next, self->num_nodes);
    tsls_large_free_array(self->site_nodes, self->num_nodes);
    tsls_large_free_array(self->stamp, self->num_nodes);
    tsi_safe_free(self->max_likelihood_node);
    tsi_safe_free(self->traceback);
    tsi_blkalloc_free(&self->traceback_allocator);
    return 0;
}

/* In Viterbi, give the nodes whose normalised values are below the
 * threshold the value likelihood_threshold. A node with a value below t
 * has odds less than t of being on an optimal path rather than the best
 * node at the site, so any path whose odds relative to the best path are at
 * least t is kept (see docs/likelihood_model.md). Unlike relative
 * pruning, this works with sparse sites. */
int
tsls_hmm_set_prune_threshold(tsls_hmm_t *self, double threshold)
{
    int ret = 0;

    if (!(threshold >= 0 && threshold <= 1)) {
        ret = TSLS_ERR_BAD_PRUNING;
        goto out;
    }
    self->prune_threshold = threshold;
out:
    return ret;
}

/* After normalisation at each site (by the maximum for Viterbi and the sum
 * for forward and backward), give nodes with values below threshold times
 * the maximum the value likelihood_threshold. This is approximate, and turns
 * off sparse Viterbi sites; 0 turns it off. */
int
tsls_hmm_set_relative_prune_threshold(tsls_hmm_t *self, double threshold)
{
    int ret = 0;

    if (!(threshold >= 0 && threshold <= 1)) {
        ret = TSLS_ERR_BAD_PRUNING;
        goto out;
    }
    self->relative_prune_threshold = threshold;
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
        self->values_tmp[j] = self->likelihood[self->likelihood_nodes[j]].value;
    }
    qsort(self->values_tmp, (size_t) self->num_likelihood_nodes, sizeof(double),
        cmp_double);
}

/* Apply relative pruning, giving pruned nodes the value
 * likelihood_threshold. The values must be normalised. */
static void
tsls_hmm_prune(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    double max_L, cutoff;
    int j;

    if (self->relative_prune_threshold == 0 || num_likelihood_nodes == 0) {
        return;
    }
    max_L = 0;
    for (j = 0; j < num_likelihood_nodes; j++) {
        max_L = TSK_MAX(max_L, L[L_nodes[j]].value);
    }
    cutoff = self->relative_prune_threshold * max_L;
    for (j = 0; j < num_likelihood_nodes; j++) {
        if (L[L_nodes[j]].value < cutoff) {
            L[L_nodes[j]].value = self->likelihood_threshold;
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
    self->likelihood[u].value = value;
    self->likelihood_nodes[self->num_likelihood_nodes] = u;
    self->num_likelihood_nodes++;
}

/* Returns the index of the first node in the likelihood list (which is
 * sorted by preorder) whose preorder is at least x. */
static inline int
tsls_hmm_lower_bound(const tsls_hmm_t *self, int32_t x)
{
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int32_t *restrict preorder = self->preorder;
    int lo = 0;
    int hi = self->num_likelihood_nodes;
    int mid;

    while (lo < hi) {
        mid = lo + (hi - lo) / 2;
        if (preorder[L_nodes[mid]] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/* As tsls_hmm_add_likelihood, keeping the list sorted by preorder. */
static void
tsls_hmm_add_likelihood_sorted(tsls_hmm_t *self, tsk_id_t u, double value)
{
    tsk_id_t *L_nodes = self->likelihood_nodes;
    const int j = tsls_hmm_lower_bound(self, self->preorder[u]);

    memmove(L_nodes + j + 1, L_nodes + j,
        (size_t) (self->num_likelihood_nodes - j) * sizeof(*L_nodes));
    L_nodes[j] = u;
    self->num_likelihood_nodes++;
    self->likelihood[u].value = value;
}

/* Remove u from the likelihood list, which is sorted by preorder. */
static void
tsls_hmm_remove_likelihood_sorted(tsls_hmm_t *self, tsk_id_t u)
{
    tsk_id_t *L_nodes = self->likelihood_nodes;
    const int j = tsls_hmm_lower_bound(self, self->preorder[u]);

    assert(j < self->num_likelihood_nodes && L_nodes[j] == u);
    memmove(L_nodes + j, L_nodes + j + 1,
        (size_t) (self->num_likelihood_nodes - j - 1) * sizeof(*L_nodes));
    self->num_likelihood_nodes--;
    self->likelihood[u].value = NULL_LIKELIHOOD;
}

/* Record that u has been removed from the likelihood list at the current
 * site, if it is in the traceback's set as of the last site stored, so
 * that tsls_hmm_store_traceback can record the change. Each node is
 * recorded at most once per site. */
static inline void
tsls_hmm_record_removed(tsls_hmm_t *self, tsk_id_t u)
{
    if (self->tree[u].traceback_flag != -1 && self->stamp[u] != self->current_stamp) {
        self->stamp[u] = self->current_stamp;
        self->removed_nodes[self->num_removed_nodes] = u;
        self->num_removed_nodes++;
    }
}

/* Remove the nodes that have left the tree (whose likelihood is
 * NOT_IN_TREE) from the list of nodes with likelihoods. */
static void
tsls_hmm_remove_left_nodes(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    int j, k;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;

    k = 0;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        L_nodes[k] = L_nodes[j];
        if (L[L_nodes[j]].value != NOT_IN_TREE) {
            k++;
        }
    }
    self->num_likelihood_nodes = k;
}

static inline void
tsls_hmm_update_num_below(tsls_hmm_t *self, tsk_id_t u, int64_t delta)
{
    tsls_node_t *restrict tree = self->tree;
    int64_t *restrict num_below = self->num_below;

    for (; u != NULL_NODE; u = tree[u].parent) {
        num_below[u] += delta;
    }
}

/* Returns the likelihood inherited by node u, filling in the likelihood
 * cache along the path so that later queries are fast. The walks up the
 * tree here and in tsls_hmm_get_inherited_likelihood_and_flag are inner
 * loops, so they have no asserts (every root has a likelihood, which
 * tsls_hmm_check_state checks). */
static inline double
tsls_hmm_get_inherited_likelihood(tsls_hmm_t *self, tsk_id_t u)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    const uint32_t stamp = self->cache_stamp;
    double value;
    tsk_id_t v;

    v = u;
    while (likely(L[v].value == NULL_LIKELIHOOD) && likely(tree[v].cache_stamp != stamp)) {
        v = tree[v].parent;
    }
    /* A node with a likelihood may have a stale cache from before it was
     * given one, so its likelihood takes precedence */
    value = L[v].value >= 0 ? L[v].value : L[v].cache;
    v = u;
    while (likely(L[v].value == NULL_LIKELIHOOD) && likely(tree[v].cache_stamp != stamp)) {
        L[v].cache = value;
        tree[v].cache_stamp = stamp;
        v = tree[v].parent;
    }
    return value;
}

/* Clear all the likelihood caches, which are only valid while the tree and
 * the set of nodes with likelihoods are unchanged. */
static inline void
tsls_hmm_clear_likelihood_caches(tsls_hmm_t *self)
{
    size_t j;

    self->cache_stamp++;
    if (unlikely(self->cache_stamp == 0)) {
        for (j = 0; j < self->num_nodes; j++) {
            self->tree[j].cache_stamp = 0;
        }
        self->cache_stamp = 1;
    }
}

/* Remove the edge from p to c from the tree, and if update_likelihoods is
 * true make sure that the child keeps its likelihood. Nodes that are
 * isolated after removing the edge are added to the isolated list; they
 * only leave the tree if they are still isolated after all the edges at
 * this position have been inserted (see tsls_hmm_update_tree). */
static inline void
tsls_hmm_remove_edge(tsls_hmm_t *self, tsk_id_t p, tsk_id_t c,
    bool update_likelihoods, tsk_id_t *restrict isolated, int *num_isolated)
{
    tsls_likelihood_t *restrict L = self->likelihood;

    if (self->track_num_below) {
        tsls_hmm_update_num_below(self, p, -self->num_below[c]);
    }
    if (update_likelihoods && L[c].value == NULL_LIKELIHOOD) {
        tsls_hmm_add_likelihood(self, c, tsls_hmm_get_inherited_likelihood(self, p));
    }
    remove_edge(p, c, self->tree, self->left_child, self->right_child,
        self->left_sib, self->right_sib);
    /* c is now a root, and so has a likelihood, which walks over the
     * cache use in preference to any cached value */
    self->tree[c].allelic_state = ROOT_ALLELE;
    if (update_likelihoods) {
        /* Each node can only become isolated once when removing edges, so
         * the isolated list has at most num_nodes entries. */
        if (!tsls_hmm_in_tree(self, c)) {
            isolated[*num_isolated] = c;
            (*num_isolated)++;
        }
        if (!tsls_hmm_in_tree(self, p)) {
            /* p was a root, and so has a likelihood */
            assert(L[p].value >= 0);
            isolated[*num_isolated] = p;
            (*num_isolated)++;
        }
    }
}

/* Insert the edge from p to c into the tree. Nodes entering the tree get
 * likelihood 0, and so can only be reached by recombination. */
static inline void
tsls_hmm_insert_edge(tsls_hmm_t *self, tsk_id_t p, tsk_id_t c, bool update_likelihoods)
{
    tsls_likelihood_t *restrict L = self->likelihood;

    if (update_likelihoods) {
        if (L[p].value == NOT_IN_TREE) {
            tsls_hmm_add_likelihood(self, p, 0);
        }
        if (L[c].value == NOT_IN_TREE) {
            tsls_hmm_add_likelihood(self, c, 0);
        }
        /* Otherwise the child was a root and keeps its likelihood */
        assert(L[c].value >= 0);
    }
    insert_edge(p, c, self->tree, self->left_child, self->right_child,
        self->left_sib, self->right_sib);
    self->tree[c].allelic_state = TSK_NULL;
    if (self->track_num_below) {
        tsls_hmm_update_num_below(self, p, self->num_below[c]);
    }
}

/* Apply the edge removals and insertions recorded in the tree position,
 * which has just moved to the next tree in its direction of travel. */
static void
tsls_hmm_update_tree(tsls_hmm_t *self, const tsk_tree_position_t *tp)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsls_edge_t *restrict out = tsls_index_get_edges(self->index, tp->out.order);
    const tsls_edge_t *restrict in = tsls_index_get_edges(self->index, tp->in.order);
    const int direction = tp->direction;
    tsk_id_t *restrict isolated = self->likelihood_nodes_tmp;
    tsk_id_t j, u;
    int k, num_left, num_isolated = 0;

    for (j = tp->out.start; j != tp->out.stop; j += direction) {
        tsls_hmm_remove_edge(
            self, out[j].parent, out[j].child, true, isolated, &num_isolated);
    }
    tsls_hmm_clear_likelihood_caches(self);
    for (j = tp->in.start; j != tp->in.stop; j += direction) {
        tsls_hmm_insert_edge(self, in[j].parent, in[j].child, true);
    }
    /* Nodes that are still isolated have left the tree. Nodes that were
     * isolated and then inserted again keep their likelihoods. We mark the
     * nodes that have left and then remove them from the likelihood list in
     * one pass, rather than one pass for each node. */
    num_left = 0;
    for (k = 0; k < num_isolated; k++) {
        u = isolated[k];
        if (!tsls_hmm_in_tree(self, u)) {
            L[u].value = NOT_IN_TREE;
            tsls_hmm_record_removed(self, u);
            num_left++;
        }
    }
    if (num_left > 0) {
        tsls_hmm_remove_left_nodes(self);
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
    const tsls_edge_t *restrict in;
    tsk_id_t u, j;
    size_t k;

    memset(self->left_child, 0xff, N * sizeof(*self->left_child));
    memset(self->right_child, 0xff, N * sizeof(*self->right_child));
    memset(self->left_sib, 0xff, N * sizeof(*self->left_sib));
    memset(self->right_sib, 0xff, N * sizeof(*self->right_sib));
    for (k = 0; k < N; k++) {
        /* Every node starts as a root. The likelihoods are set below. */
        self->tree[k].cache_stamp = 0;
        self->tree[k].parent = NULL_NODE;
        self->tree[k].allelic_state = ROOT_ALLELE;
        self->tree[k].recombination_required = -1;
        self->tree[k].recombination_required_cache = -1;
        self->tree[k].traceback_flag = -1;
        self->num_below[k] = 1;
    }
    self->cache_stamp = 1;
    self->tree_numbered = false;
    self->all_changed = true;
    self->num_changed = 0;
    self->num_removed_nodes = 0;
    ret = seek_tree_position(tp, position);
    if (ret != 0) {
        goto out;
    }
    in = tsls_index_get_edges(self->index, tp->in.order);
    for (j = tp->in.start; j != tp->in.stop; j += tp->direction) {
        if (seek_edge_covers(tp, &in[j], position)) {
            tsls_hmm_insert_edge(self, in[j].parent, in[j].child, false);
        }
    }
    self->num_likelihood_nodes = 0;
    for (u = 0; u < (tsk_id_t) N; u++) {
        if (self->tree[u].parent != NULL_NODE) {
            self->likelihood[u].value = NULL_LIKELIHOOD;
        } else if (self->left_child[u] != NULL_NODE) {
            tsls_hmm_add_likelihood(self, u, initial_value);
        } else {
            self->likelihood[u].value = NOT_IN_TREE;
        }
    }
out:
    return ret;
}

/* Number the nodes of the current tree in preorder, and sort the
 * likelihood list by preorder (see tsls_hmm_t). */
static void
tsls_hmm_number_tree(tsls_hmm_t *self)
{
    const tsk_id_t N = (tsk_id_t) self->num_nodes;
    const tsls_node_t *restrict tree = self->tree;
    const tsk_id_t *restrict left_child = self->left_child;
    const tsk_id_t *restrict right_sib = self->right_sib;
    const tsls_likelihood_t *restrict L = self->likelihood;
    int32_t *restrict preorder = self->preorder;
    int32_t *restrict subtree_end = self->subtree_end;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsk_id_t *restrict order = self->likelihood_nodes_tmp;
    tsk_id_t root, u;
    int32_t k, j;
    int n;
    bool done;

    for (u = 0; u < N; u++) {
        preorder[u] = -1;
    }
    k = 0;
    for (root = 0; root < N; root++) {
        if (tree[root].parent != NULL_NODE || left_child[root] == NULL_NODE) {
            continue;
        }
        u = root;
        done = false;
        while (!done) {
            preorder[u] = k;
            order[k] = u;
            k++;
            if (left_child[u] != NULL_NODE) {
                u = left_child[u];
            } else {
                /* Leave u, and every ancestor whose last child we've left */
                while (true) {
                    subtree_end[u] = k;
                    if (u == root) {
                        done = true;
                        break;
                    }
                    if (right_sib[u] != NULL_NODE) {
                        u = right_sib[u];
                        break;
                    }
                    u = tree[u].parent;
                }
            }
        }
    }
    n = 0;
    for (j = 0; j < k; j++) {
        u = order[j];
        if (L[u].value >= 0) {
            L_nodes[n] = u;
            n++;
        }
    }
    assert(n == self->num_likelihood_nodes);
    self->num_numbered = k;
    self->tree_numbered = true;
}

static inline void
tsls_hmm_set_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const tsls_index_t *index = self->index;
    const tsk_id_t *restrict node = index->mutation_node;
    const allele_t *restrict derived_state = index->mutation_derived_state;
    const tsk_size_t stop = index->site_mutations_offset[site + 1];
    tsk_size_t j;

    for (j = index->site_mutations_offset[site]; j < stop; j++) {
        self->tree[node[j]].allelic_state = derived_state[j];
    }
}

/* Unset the allelic states set for the site's mutations, keeping the
 * roots marked with ROOT_ALLELE (during the sweeps). */
static inline void
tsls_hmm_unset_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const tsls_index_t *index = self->index;
    const tsk_id_t *restrict node = index->mutation_node;
    const tsk_size_t stop = index->site_mutations_offset[site + 1];
    tsk_size_t j;
    tsk_id_t u;

    for (j = index->site_mutations_offset[site]; j < stop; j++) {
        u = node[j];
        self->tree[u].allelic_state = self->tree[u].parent == NULL_NODE ? ROOT_ALLELE : TSK_NULL;
    }
}

/* As tsls_hmm_unset_allelic_state, for the traceback, which doesn't mark
 * the roots. */
static inline void
tsls_hmm_clear_allelic_state(tsls_hmm_t *self, const tsk_id_t site)
{
    const tsls_index_t *index = self->index;
    const tsk_id_t *restrict node = index->mutation_node;
    const tsk_size_t stop = index->site_mutations_offset[site + 1];
    tsk_size_t j;

    for (j = index->site_mutations_offset[site]; j < stop; j++) {
        self->tree[node[j]].allelic_state = TSK_NULL;
    }
}

/* Make sure that every node with a mutation at this site that is in the
 * tree has its own likelihood. */
static void
tsls_hmm_insert_mutation_likelihoods(tsls_hmm_t *self, const tsk_id_t site)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    const tsls_index_t *index = self->index;
    const tsk_size_t stop = index->site_mutations_offset[site + 1];
    tsk_size_t j;
    tsk_id_t u, node;

    for (j = index->site_mutations_offset[site]; j < stop; j++) {
        node = index->mutation_node[j];
        if (L[node].value == NULL_LIKELIHOOD) {
            u = node;
            while (L[u].value == NULL_LIKELIHOOD) {
                u = tree[u].parent;
            }
            if (self->tree_numbered) {
                tsls_hmm_add_likelihood_sorted(self, node, L[u].value);
            } else {
                tsls_hmm_add_likelihood(self, node, L[u].value);
            }
        }
    }
}

/* As tsls_hmm_get_inherited_likelihood, but also return the
 * recombination_required flag that u inherits, caching it along the path
 * with the likelihood. */
static ALWAYS_INLINE double
tsls_hmm_get_inherited_likelihood_and_flag(
    tsls_hmm_t *self, tsk_id_t u, int8_t *flag)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    const uint32_t stamp = self->cache_stamp;
    double value;
    int8_t r;
    tsk_id_t v;

    v = u;
    while (likely(L[v].value == NULL_LIKELIHOOD) && likely(tree[v].cache_stamp != stamp)) {
        v = tree[v].parent;
    }
    if (L[v].value >= 0) {
        value = L[v].value;
        r = tree[v].recombination_required;
    } else {
        value = L[v].cache;
        r = tree[v].recombination_required_cache;
    }
    v = u;
    while (likely(L[v].value == NULL_LIKELIHOOD) && likely(tree[v].cache_stamp != stamp)) {
        L[v].cache = value;
        tree[v].recombination_required_cache = r;
        tree[v].cache_stamp = stamp;
        v = tree[v].parent;
    }
    *flag = r;
    return value;
}

/* Set the listed_ancestor of every node in the likelihood list to its
 * nearest proper ancestor with a likelihood (NULL_NODE for a root). Each
 * listed node is first marked, with the stamp of the likelihood caches and
 * itself as its listed_ancestor, so that the walks up the tree only read
 * the node records. The answers are cached on the nodes without
 * likelihoods on the paths, so the cost is the size of the union of the
 * paths up to the nearest nodes with likelihoods, which are much shorter
 * than the paths up to a mutation or root. Clears the likelihood caches,
 * which share the stamp, before; they must be cleared again once the
 * likelihood list changes. */
static void
tsls_hmm_set_listed_ancestors(tsls_hmm_t *self)
{
    tsls_node_t *restrict tree = self->tree;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsk_id_t *restrict ancestor = self->likelihood_nodes_tmp;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    uint32_t stamp;
    tsk_id_t u, v, a;
    int j;

    tsls_hmm_clear_likelihood_caches(self);
    stamp = self->cache_stamp;
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        tree[u].cache_stamp = stamp;
        tree[u].listed_ancestor = u;
    }
    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        a = NULL_NODE;
        if (tree[u].parent != NULL_NODE) {
            v = tree[u].parent;
            while (tree[v].cache_stamp != stamp) {
                v = tree[v].parent;
            }
            a = tree[v].listed_ancestor;
            v = tree[u].parent;
            while (tree[v].cache_stamp != stamp) {
                tree[v].listed_ancestor = a;
                tree[v].cache_stamp = stamp;
                v = tree[v].parent;
            }
        }
        ancestor[j] = a;
    }
    for (j = 0; j < num_likelihood_nodes; j++) {
        tree[L_nodes[j]].listed_ancestor = ancestor[j];
    }
}

/* Returns the allelic state of the node u with a likelihood, given that
 * every node in the likelihood list has its listed_ancestor set, and that
 * the nodes with mutations at the site have their allelic states set and
 * the roots are marked with ROOT_ALLELE. Because every node with a mutation
 * at the site has a likelihood (tsls_hmm_insert_mutation_likelihoods), a
 * node without a mutation has the state of its nearest ancestor with a
 * likelihood, so the walk goes up the list's own tree rather than the
 * whole tree. States found are memoised on the list nodes passed, as in
 * get_memoised_allelic_state. */
static ALWAYS_INLINE allele_t
get_listed_allelic_state(tsls_node_t *restrict tree, tsk_id_t u,
    allele_t ancestral_state, tsk_id_t *restrict touched, int *num_touched)
{
    allele_t state;
    tsk_id_t v = u;
    int n = *num_touched;

    while (tree[v].allelic_state == TSK_NULL) {
        v = tree[v].listed_ancestor;
    }
    state = tree[v].allelic_state;
    while (u != v) {
        tree[u].allelic_state = state;
        touched[n] = u;
        n++;
        u = tree[u].listed_ancestor;
    }
    *num_touched = n;
    return likely(state != ROOT_ALLELE) ? state : ancestral_state;
}

/* Remove the Viterbi likelihoods that are equal to those of the nearest
 * ancestor with a likelihood, with the same recombination_required flag, so
 * that the traceback is unchanged (see tsls_hmm_coalesce_likelihoods). The
 * listed_ancestor of every node in the list must be up to date. A node
 * whose ancestor is itself removed here had the same value and flag as that
 * ancestor, which had the same as its own ancestor, so the nodes are only
 * removed after all the comparisons. */
static void
tsls_hmm_coalesce_listed_likelihoods(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsls_node_t *restrict tree = self->tree;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsk_id_t *restrict removed = self->likelihood_nodes_tmp;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    int j, num_removed, num_likelihood_nodes;
    tsk_id_t u, a;

    num_removed = 0;
    num_likelihood_nodes = 0;
    for (j = 0; j < old_num_likelihood_nodes; j++) {
        u = L_nodes[j];
        a = tree[u].listed_ancestor;
        if (a != NULL_NODE && L[u].value == L[a].value
            && tree[u].recombination_required == tree[a].recombination_required) {
            removed[num_removed] = u;
            num_removed++;
        } else {
            L_nodes[num_likelihood_nodes] = u;
            num_likelihood_nodes++;
        }
    }
    for (j = 0; j < num_removed; j++) {
        L[removed[j]].value = NULL_LIKELIHOOD;
        tsls_hmm_record_removed(self, removed[j]);
    }
    self->num_likelihood_nodes = num_likelihood_nodes;
    tsls_hmm_clear_likelihood_caches(self);
}

/* Remove likelihoods that are equal to the likelihood inherited from the
 * parent. If check_recombination_required is true, also require the
 * recombination_required flag to equal the one inherited from the parent,
 * so that the traceback, in which nodes inherit the flag of their nearest
 * ancestor in the list, is unchanged. */
static ALWAYS_INLINE void
tsls_hmm_coalesce_likelihoods(
    tsls_hmm_t *self, bool check_recombination_required)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    double L_p;
    int8_t R_p;
    tsk_id_t u, p;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    int j, num_likelihood_nodes;

    num_likelihood_nodes = 0;
    for (j = 0; j < old_num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p = tree[u].parent;
        if (p != NULL_NODE) {
            if (check_recombination_required) {
                /* A removed node had the inherited value and flag, so the
                 * cached values below it stay valid. */
                L_p = tsls_hmm_get_inherited_likelihood_and_flag(
                    self, p, &R_p);
                if (L[u].value == L_p && tree[u].recombination_required == R_p) {
                    L[u].value = NULL_LIKELIHOOD;
                    tsls_hmm_record_removed(self, u);
                }
            } else {
                L_p = tsls_hmm_get_inherited_likelihood(self, p);
                if (L[u].value == L_p) {
                    L[u].value = NULL_LIKELIHOOD;
                }
            }
        }
        if (L[u].value >= 0) {
            L_nodes[num_likelihood_nodes] = L_nodes[j];
            num_likelihood_nodes++;
        }
    }
    self->num_likelihood_nodes = num_likelihood_nodes;
    tsls_hmm_clear_likelihood_caches(self);
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
    const tsk_size_t num_alleles = self->index->num_alleles[site];

    if (state < TSLS_MISSING_DATA || (state >= 0 && (tsk_size_t) state >= num_alleles)) {
        return TSLS_ERR_BAD_HAPLOTYPE_ALLELE;
    }
    return 0;
}

static inline double
tsls_hmm_get_n(const tsls_hmm_t *self)
{
    return (double) self->num_nodes;
}

/* Returns the normalised Viterbi value of a node, with the thresholds
 * applied. */
static inline double
tsls_hmm_normalise_viterbi(const tsls_hmm_t *self, double value, double max_L)
{
    value /= max_L;
    if (value < self->prune_threshold) {
        value = self->likelihood_threshold;
    }
    return TSK_MAX(value, self->likelihood_threshold);
}

/* Compute the number of nodes in the tree that inherit the likelihood of
 * each likelihood node. */
static void
tsls_hmm_compute_num_inheriting(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
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
        w = tree[u].parent;
        if (w != NULL_NODE) {
            while (L[w].value == NULL_LIKELIHOOD) {
                w = tree[w].parent;
            }
            num_inheriting[w] -= num_below[u];
        }
    }
}

/* Returns the sum of the likelihoods over all the nodes in the tree. */
static double
tsls_hmm_total_likelihood(const tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int64_t *restrict num_inheriting = self->num_inheriting;
    double total = 0;
    tsk_id_t u;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        assert(num_inheriting[u] > 0);
        total += (double) num_inheriting[u] * L[u].value;
    }
    return total;
}

/* Store the changes to the traceback at this site (see traceback_delta_t).
 * Every node whose flag in the traceback may have changed since the last
 * site stored must be in one of the two lists, with no node appearing
 * twice: a node's flag is its recombination_required flag if it is in the
 * likelihood list, and -1 if not. */
static int WARN_UNUSED
tsls_hmm_store_traceback(tsls_hmm_t *self, const tsk_id_t site,
    const tsk_id_t *restrict nodes_a, int num_a, const tsk_id_t *restrict nodes_b,
    int num_b)
{
    int ret = 0;
    tsk_id_t u;
    int j, num_changes;
    int8_t flag;
    tsls_node_t *restrict tree = self->tree;
    const tsls_likelihood_t *restrict L = self->likelihood;
    tsk_id_t *restrict changed = self->likelihood_nodes_tmp;
    int8_t *restrict previous_flag = self->traceback_flags_tmp;
    traceback_delta_t *restrict delta = &self->traceback[site];
    const tsk_id_t *restrict nodes;
    int num_nodes, k;

    num_changes = 0;
    for (k = 0; k < 2; k++) {
        nodes = k == 0 ? nodes_a : nodes_b;
        num_nodes = k == 0 ? num_a : num_b;
        for (j = 0; j < num_nodes; j++) {
            u = nodes[j];
            flag = L[u].value >= 0 ? tree[u].recombination_required : -1;
            if (tree[u].traceback_flag != flag) {
                changed[num_changes] = u;
                previous_flag[num_changes] = tree[u].traceback_flag;
                num_changes++;
                tree[u].traceback_flag = flag;
            }
        }
    }
    delta->size = num_changes;
    if (num_changes > 0) {
        delta->node = tsi_blkalloc_get(&self->traceback_allocator,
            aligned_size((size_t) num_changes * sizeof(tsk_id_t)));
        delta->previous_flag = tsi_blkalloc_get(&self->traceback_allocator,
            aligned_size((size_t) num_changes * sizeof(int8_t)));
        if (delta->node == NULL || delta->previous_flag == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        memcpy(delta->node, changed, (size_t) num_changes * sizeof(*changed));
        memcpy(delta->previous_flag, previous_flag,
            (size_t) num_changes * sizeof(*previous_flag));
    }
    self->total_traceback_size += (size_t) self->num_likelihood_nodes;
out:
    return ret;
}

/* Returns the normalisation factor of a Viterbi site at which a node with
 * value 1 matches, computed as tsls_hmm_process_site_viterbi computes it
 * for such a node. */
static inline double
tsls_hmm_matching_normalisation_factor(const tsls_hmm_t *self, tsk_id_t site)
{
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->index->num_alleles[site];

    return (1.0 * (1 - rho + rho / n)) * emission_probability(0, 0, mu, num_alleles);
}

/* Process a Viterbi site in full. If track_changes is true, also record the
 * nodes whose values or flags changed or that mismatched, for the next site
 * (see tsls_hmm_process_site_sparse). */
static int WARN_UNUSED
tsls_hmm_process_site_viterbi(tsls_hmm_t *self, const tsk_id_t site,
    const allele_t state, bool track_changes, compressed_matrix_t *output)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsls_node_t *restrict tree = self->tree;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->index->num_alleles[site];
    const allele_t ancestral_state = self->index->ancestral_state[site];
    tsk_id_t *restrict touched = self->likelihood_nodes_tmp;
    double *restrict old_value = self->values_tmp;
    /* The old flag, plus 4 if the node mismatched */
    int8_t *restrict old_flag = self->traceback_flags_tmp;
    tsk_id_t *restrict changed = self->changed_next;
    int num_touched = 0;
    int num_changed = 0;
    double max_L, p_no_recomb, p_recomb, p_t, p_e;
    tsk_id_t u, max_L_node;
    allele_t allele;
    int j;

    tsls_hmm_insert_mutation_likelihoods(self, site);
    tsls_hmm_set_allelic_state(self, site);
    self->site_num_updated = (size_t) self->num_likelihood_nodes;
    /* Find the list's own tree, for the alleles and coalescing */
    tsls_hmm_set_listed_ancestors(self);
    max_L = -1;
    max_L_node = NULL_NODE;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        if (track_changes) {
            old_value[j] = L[u].value;
            old_flag[j] = tree[u].recombination_required;
        }
        p_no_recomb = L[u].value * (1 - rho + rho / n);
        p_recomb = rho / n;
        tree[u].recombination_required = false;
        if (p_no_recomb > p_recomb) {
            p_t = p_no_recomb;
        } else {
            p_t = p_recomb;
            tree[u].recombination_required = true;
        }
        allele = get_listed_allelic_state(
            tree, u, ancestral_state, touched, &num_touched);
        p_e = emission_probability(allele, state, mu, num_alleles);
        if (track_changes && state != TSLS_MISSING_DATA && allele != state) {
            old_flag[j] = (int8_t) (old_flag[j] + 4);
        }
        L[u].value = p_t * p_e;
        if (L[u].value > max_L) {
            max_L = L[u].value;
            max_L_node = u;
        }
    }
    clear_touched_allelic_state(tree, touched, num_touched);
    tsls_hmm_unset_allelic_state(self, site);

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
        L[u].value = tsls_hmm_normalise_viterbi(self, L[u].value, max_L);
        if (track_changes
            && (old_flag[j] >= 3 || L[u].value != old_value[j]
                || tree[u].recombination_required != old_flag[j])) {
            changed[num_changed] = u;
            num_changed++;
        }
    }
    if (track_changes) {
        /* The nodes that didn't change at this site are unchanged at the
         * next by the same map, which a sparse site can only rely on if
         * this site was normalised as a sparse site is. */
        self->changed_next = self->changed;
        self->changed = changed;
        self->num_changed = num_changed;
        self->all_changed = max_L != tsls_hmm_matching_normalisation_factor(self, site);
    } else {
        self->all_changed = true;
    }
    tsls_hmm_prune(self);
    tsls_hmm_count_distinct(self);
    /* Remove redundant nodes before storing the traceback, keeping those
     * whose recombination_required flag differs from the inherited one.
     * Those are left for the next site, where their flags are recomputed. */
    tsls_hmm_coalesce_listed_likelihoods(self);
    ret = tsls_hmm_store_traceback(self, site, self->removed_nodes,
        self->num_removed_nodes, L_nodes, self->num_likelihood_nodes);
    if (ret != 0) {
        goto out;
    }
    if (output != NULL) {
        ret = compressed_matrix_store_site(
            output, site, max_L, self->num_likelihood_nodes, L_nodes, L, self->tree, true);
        if (ret != 0) {
            goto out;
        }
    }
out:
    return ret;
}

/* The most mutations at a site that a sparse site handles */
#define SPARSE_MAX_MUTATIONS 64

/* The subtree of a node with a mutation at a site, as a range of preorder
 * positions, with the mutation's derived state */
typedef struct {
    int32_t start;
    int32_t end;
    allele_t state;
} mutation_range_t;

/* Returns the allelic state at the node at the specified preorder position:
 * the state of the innermost mutation range containing it (the later
 * mutation, if a node has several), or the ancestral state. */
static inline allele_t
mutation_range_allele(const mutation_range_t *restrict ranges, int num_ranges,
    int32_t x, allele_t ancestral_state)
{
    int32_t start = -1;
    allele_t state = ancestral_state;
    int j;

    for (j = 0; j < num_ranges; j++) {
        if (ranges[j].start <= x && x < ranges[j].end && ranges[j].start >= start) {
            start = ranges[j].start;
            state = ranges[j].state;
        }
    }
    return state;
}

static inline void
tsls_hmm_add_site_node(tsls_hmm_t *self, tsk_id_t u, int *num_nodes)
{
    if (self->stamp[u] != self->current_stamp) {
        self->stamp[u] = self->current_stamp;
        self->site_nodes[*num_nodes] = u;
        (*num_nodes)++;
    }
}

/* Process a Viterbi site, recomputing only the values that can change, if
 * possible. Returns 1 if the site was processed, and 0 if it must be
 * processed in full.
 *
 * Suppose the tree is numbered, the previous site's values were all
 * normalised by the same factor as this site's, and the rates are the same
 * at both sites. The previous site's maximum likelihood node has value 1
 * (or inherits it, if it has since been coalesced). If it matches the
 * haplotype here, it has the maximum likelihood again, and the
 * normalisation factor is the same as at the previous site. Then the values
 * of nodes that match here go through the same map as at the previous site.
 * So a node whose value was unchanged at the previous site, and that matched
 * there and matches here, is unchanged. We recompute the values of the
 * nodes that changed or mismatched at the previous site, and of those that
 * mismatch here. The subtree of each node with a mutation at this site is a
 * range of preorder positions, which gives the alleles of the nodes, and the
 * nodes in mismatching ranges of the likelihood list (which is sorted by
 * preorder). A node with a mutation whose match differs from its parent's
 * gets its own likelihood.
 *
 * Only the nodes whose values were recomputed are coalesced, so nodes below
 * them that now have the same value and flag as their parents may stay in
 * the list until the next site processed in full; this doesn't change the
 * values of any node in the tree, or the traceback. The maximum likelihood
 * node stays the same, where processing the site in full might choose
 * another node with value 1. */
static int WARN_UNUSED
tsls_hmm_process_site_sparse(tsls_hmm_t *self, const tsk_id_t site, const allele_t state)
{
    int ret = 0;
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    const tsls_index_t *index = self->index;
    const int32_t *restrict preorder = self->preorder;
    const int32_t *restrict subtree_end = self->subtree_end;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsk_id_t *restrict nodes = self->site_nodes;
    tsk_id_t *restrict changed = self->changed_next;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) index->num_alleles[site];
    const allele_t ancestral_state = index->ancestral_state[site];
    const tsk_size_t mutations_start = index->site_mutations_offset[site];
    const tsk_size_t mutations_stop = index->site_mutations_offset[site + 1];
    const tsk_id_t max_L_node = self->max_likelihood_node[site - 1];
    const double max_L = tsls_hmm_matching_normalisation_factor(self, site);
    const double p_recomb = rho / n;
    mutation_range_t ranges[SPARSE_MAX_MUTATIONS];
    int32_t bounds[2 * SPARSE_MAX_MUTATIONS + 2];
    double value, old_value, p_no_recomb, p_t, L_p;
    int8_t old_flag, R_p;
    allele_t allele;
    bool mismatch;
    int j, k, num_ranges, num_bounds, num_nodes, num_changed;
    int32_t x;
    tsk_size_t i;
    tsk_id_t u, p;

    if (mutations_stop - mutations_start > SPARSE_MAX_MUTATIONS) {
        goto out;
    }
    num_ranges = 0;
    for (i = mutations_start; i < mutations_stop; i++) {
        u = index->mutation_node[i];
        if (preorder[u] >= 0) {
            ranges[num_ranges].start = preorder[u];
            ranges[num_ranges].end = subtree_end[u];
            ranges[num_ranges].state = index->mutation_derived_state[i];
            num_ranges++;
        }
    }
    if (state != TSLS_MISSING_DATA
        && mutation_range_allele(
               ranges, num_ranges, preorder[max_L_node], ancestral_state)
               != state) {
        goto out;
    }

    num_nodes = 0;
    for (j = 0; j < self->num_changed; j++) {
        u = self->changed[j];
        if (L[u].value >= 0) {
            tsls_hmm_add_site_node(self, u, &num_nodes);
        }
    }
    if (state != TSLS_MISSING_DATA) {
        /* Mutation nodes whose match differs from their parent's */
        for (i = mutations_start; i < mutations_stop; i++) {
            u = index->mutation_node[i];
            if (preorder[u] < 0 || L[u].value >= 0) {
                continue;
            }
            /* u is in the tree and has no likelihood, so it isn't a root */
            p = tree[u].parent;
            mismatch = mutation_range_allele(
                           ranges, num_ranges, preorder[u], ancestral_state)
                       != state;
            if (mismatch
                != (mutation_range_allele(
                        ranges, num_ranges, preorder[p], ancestral_state)
                    != state)) {
                value = tsls_hmm_get_inherited_likelihood_and_flag(self, p, &R_p);
                tsls_hmm_clear_likelihood_caches(self);
                tsls_hmm_add_likelihood_sorted(self, u, value);
                tree[u].recombination_required = R_p;
                tsls_hmm_add_site_node(self, u, &num_nodes);
            }
        }
        /* The nodes in the likelihood list in mismatching ranges. Between
         * consecutive range boundaries the allele is the same. */
        num_bounds = 0;
        bounds[num_bounds++] = 0;
        bounds[num_bounds++] = self->num_numbered;
        for (j = 0; j < num_ranges; j++) {
            bounds[num_bounds++] = ranges[j].start;
            bounds[num_bounds++] = ranges[j].end;
        }
        /* Insertion sort: there are few bounds */
        for (j = 1; j < num_bounds; j++) {
            x = bounds[j];
            for (k = j - 1; k >= 0 && bounds[k] > x; k--) {
                bounds[k + 1] = bounds[k];
            }
            bounds[k + 1] = x;
        }
        for (j = 0; j < num_bounds - 1; j++) {
            if (bounds[j] == bounds[j + 1]
                || mutation_range_allele(
                       ranges, num_ranges, bounds[j], ancestral_state)
                       == state) {
                continue;
            }
            for (k = tsls_hmm_lower_bound(self, bounds[j]);
                 k < self->num_likelihood_nodes && preorder[L_nodes[k]] < bounds[j + 1];
                 k++) {
                tsls_hmm_add_site_node(self, L_nodes[k], &num_nodes);
            }
        }
    }

    /* Recompute the values, as tsls_hmm_process_site_viterbi does */
    self->site_num_updated = (size_t) num_nodes;
    num_changed = 0;
    for (j = 0; j < num_nodes; j++) {
        u = nodes[j];
        old_value = L[u].value;
        old_flag = tree[u].recombination_required;
        p_no_recomb = old_value * (1 - rho + rho / n);
        tree[u].recombination_required = false;
        if (p_no_recomb > p_recomb) {
            p_t = p_no_recomb;
        } else {
            p_t = p_recomb;
            tree[u].recombination_required = true;
        }
        allele = mutation_range_allele(ranges, num_ranges, preorder[u], ancestral_state);
        mismatch = state != TSLS_MISSING_DATA && allele != state;
        value = p_t * emission_probability(allele, state, mu, num_alleles);
        value = tsls_hmm_normalise_viterbi(self, value, max_L);
        L[u].value = value;
        if (mismatch || value != old_value
            || tree[u].recombination_required != old_flag) {
            changed[num_changed] = u;
            num_changed++;
        }
    }
    self->max_likelihood_node[site] = max_L_node;
    self->changed_next = self->changed;
    self->changed = changed;
    self->num_changed = num_changed;
    self->all_changed = false;

    /* Coalesce the recomputed nodes, as tsls_hmm_coalesce_likelihoods does */
    for (j = 0; j < num_nodes; j++) {
        u = nodes[j];
        p = tree[u].parent;
        if (p != NULL_NODE) {
            L_p = tsls_hmm_get_inherited_likelihood_and_flag(self, p, &R_p);
            if (L[u].value == L_p && tree[u].recombination_required == R_p) {
                tsls_hmm_remove_likelihood_sorted(self, u);
            }
        }
    }
    tsls_hmm_clear_likelihood_caches(self);
    ret = tsls_hmm_store_traceback(self, site, nodes, num_nodes, NULL, 0);
    if (ret != 0) {
        goto out;
    }
    self->num_sparse_sites++;
    ret = 1;
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
    tsls_likelihood_t *restrict L = self->likelihood;
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsls_node_t *restrict tree = self->tree;
    const double rho = self->recombination_rate[site];
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->index->num_alleles[site];
    const allele_t ancestral_state = self->index->ancestral_state[site];
    tsk_id_t *restrict touched = self->likelihood_nodes_tmp;
    int num_touched = 0;
    double total, normalisation_factor, p_e;
    tsk_id_t u;
    int j;

    tsls_hmm_insert_mutation_likelihoods(self, site);
    tsls_hmm_compute_num_inheriting(self);
    self->site_num_updated = (size_t) self->num_likelihood_nodes;
    total = *previous_total;
    if (total < 0) {
        total = tsls_hmm_total_likelihood(self);
    }
    tsls_hmm_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p_e = emission_probability(
            get_memoised_allelic_state(
                tree, u, ancestral_state, touched, &num_touched),
            state,
            mu, num_alleles);
        L[u].value = (L[u].value * (1 - rho) + rho / n * total) * p_e;
    }
    clear_touched_allelic_state(tree, touched, num_touched);
    tsls_hmm_unset_allelic_state(self, site);
    normalisation_factor = tsls_hmm_total_likelihood(self);
    if (!(normalisation_factor > 0)) {
        ret = TSLS_ERR_MATCH_IMPOSSIBLE;
        goto out;
    }
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u].value = TSK_MAX(L[u].value / normalisation_factor, self->likelihood_threshold);
    }
    tsls_hmm_prune(self);
    *previous_total = tsls_hmm_total_likelihood(self);
    ret = compressed_matrix_store_site(output, site, normalisation_factor,
        self->num_likelihood_nodes, L_nodes, L, self->tree, false);
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
    tsls_likelihood_t *restrict L = self->likelihood;
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsls_node_t *restrict tree = self->tree;
    const double mu = self->mismatch_rate[site];
    const double n = tsls_hmm_get_n(self);
    const double num_alleles = (double) self->index->num_alleles[site];
    const allele_t ancestral_state = self->index->ancestral_state[site];
    tsk_id_t *restrict touched = self->likelihood_nodes_tmp;
    int num_touched = 0;
    double x;
    tsk_id_t u;
    int j;

    if (transition->pending) {
        for (j = 0; j < self->num_likelihood_nodes; j++) {
            u = L_nodes[j];
            x = (1 - transition->rho) * L[u].value + transition->rho * transition->total / n;
            L[u].value = TSK_MAX(
                x / transition->normalisation_factor, self->likelihood_threshold);
        }
        tsls_hmm_prune(self);
    }
    tsls_hmm_insert_mutation_likelihoods(self, site);
    ret = compressed_matrix_store_site(output, site, forward_normalisation_factor[site],
        self->num_likelihood_nodes, L_nodes, L, self->tree, false);
    if (ret != 0) {
        goto out;
    }
    tsls_hmm_compute_num_inheriting(self);
    self->site_num_updated = (size_t) self->num_likelihood_nodes;
    tsls_hmm_set_allelic_state(self, site);
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        L[u].value *= emission_probability(
            get_memoised_allelic_state(
                tree, u, ancestral_state, touched, &num_touched),
            state,
            mu, num_alleles);
    }
    clear_touched_allelic_state(tree, touched, num_touched);
    tsls_hmm_unset_allelic_state(self, site);
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
    const bool sparse = mode == MODE_VITERBI && output == NULL
                        && !(self->flags & (TSLS_DISABLE_SPARSE | TSLS_COUNT_DISTINCT))
                        && self->relative_prune_threshold == 0;
    tsk_id_t site, first_site;
    double previous_total = -1;
    bool valid, new_tree;
    int handled;

    tsk_tree_position_init(&tp, self->tree_sequence, 0);
    self->stats_num_sites = 0;
    self->stats_total_values = 0;
    self->stats_max_values = 0;
    self->stats_total_updated = 0;
    for (site = start; site < end; site++) {
        ret = check_haplotype_state(self, site, haplotype[site]);
        if (ret != 0) {
            goto out;
        }
    }
    self->track_num_below = mode != MODE_VITERBI;
    first_site = direction > 0 ? start : end - 1;
    /* The forward algorithm starts from the uniform distribution over the
     * nodes; the backward starts from 1, and Viterbi values are relative. */
    ret = tsls_hmm_init_sweep(self, &tp, position[first_site],
        mode == MODE_FORWARD ? 1.0 / tsls_hmm_get_n(self) : 1.0);
    if (ret != 0) {
        goto out;
    }
    transition.pending = false;
    new_tree = true;

    for (site = first_site; start <= site && site < end; site += direction) {
        while (!(tp.interval.left <= position[site]
                 && position[site] < tp.interval.right)) {
            valid = direction > 0 ? tsk_tree_position_next(&tp)
                                  : tsk_tree_position_prev(&tp);
            assert(valid);
            (void) valid;
            tsls_hmm_update_tree(self, &tp);
            new_tree = true;
        }
        if (self->flags & TSLS_EXTENDED_CHECKS) {
            tsls_hmm_check_state(self);
        }
        if (sparse) {
            if (new_tree) {
                /* Nodes may have entered or left the tree, so process this
                 * site in full */
                self->tree_numbered = false;
                self->all_changed = true;
                self->tree_end_site = TSK_MIN(end,
                    (tsk_id_t) tsk_search_sorted(
                        position, self->num_sites, tp.interval.right));
                new_tree = false;
            }
            /* Numbering the tree takes about as long as processing a site
             * in full for every node, so do it once the list is long
             * enough for the remaining sites in the tree. */
            if (!self->tree_numbered
                && (double) (self->tree_end_site - site)
                           * (double) self->num_likelihood_nodes
                       >= (double) self->num_nodes) {
                tsls_hmm_number_tree(self);
            }
        }
        if (mode == MODE_VITERBI) {
            handled = 0;
            if (sparse && self->tree_numbered && !self->all_changed && site > start
                && self->recombination_rate[site] == self->recombination_rate[site - 1]
                && self->mismatch_rate[site] == self->mismatch_rate[site - 1]
                && self->index->num_alleles[site] == self->index->num_alleles[site - 1]) {
                handled = tsls_hmm_process_site_sparse(self, site, haplotype[site]);
            }
            if (handled < 0) {
                ret = handled;
            } else if (handled == 0) {
                /* Changes are only needed if the next site can be sparse */
                ret = tsls_hmm_process_site_viterbi(self, site, haplotype[site],
                    sparse && self->tree_numbered, output);
            }
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
        self->stats_num_sites++;
        self->stats_total_values += (size_t) self->num_likelihood_nodes;
        self->stats_max_values
            = TSK_MAX(self->stats_max_values, (size_t) self->num_likelihood_nodes);
        self->stats_total_updated += self->site_num_updated;
        self->num_removed_nodes = 0;
        if (self->current_stamp == UINT32_MAX) {
            memset(self->stamp, 0, (self->num_nodes + 1) * sizeof(*self->stamp));
            self->current_stamp = 0;
        }
        self->current_stamp++;
    }
out:
    tsk_tree_position_free(&tp);
    return ret;
}

/* Undo the changes to the traceback flags made at the specified site, so
 * that they are the flags stored at the previous site. */
static inline void
tsls_hmm_undo_traceback_delta(
    const tsls_hmm_t *self, tsk_id_t site, tsls_node_t *restrict tree)
{
    int j;
    const traceback_delta_t *delta = &self->traceback[site];
    const tsk_id_t *restrict node = delta->node;
    const int8_t *restrict previous_flag = delta->previous_flag;

    for (j = 0; j < delta->size; j++) {
        tree[node[j]].traceback_flag = previous_flag[j];
    }
}

/* Run the traceback over sites [start, end), moving right-to-left through
 * the trees. Only the parent array of the tree is needed. The traceback
 * flags start as stored at the last site, and each site's changes are
 * undone after it is processed, so that they are all unset (-1) again at
 * the end. */
static int WARN_UNUSED
tsls_hmm_run_traceback(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    allele_t *restrict match, size_t *path_length_out, double *restrict path_left,
    double *restrict path_right, tsk_id_t *restrict path_parent)
{
    int ret = 0;
    tsk_tree_position_t tp;
    tsk_id_t site, j, u, max_likelihood_node;
    double x;
    bool valid;
    tsls_node_t *restrict tree = self->tree;
    const double *restrict sites_position = self->sites_position;
    /* Every root of the traceback tree has the virtual root as its parent,
     * whose allelic state is the site's ancestral state, so the walk to find
     * a node's allele doesn't need to check for the root. */
    const tsk_id_t virtual_root = (tsk_id_t) self->num_nodes;
    const tsls_edge_t *restrict in;
    const tsls_edge_t *restrict out;
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
    for (u = 0; u < virtual_root; u++) {
        tree[u].parent = virtual_root;
        tree[u].allelic_state = TSK_NULL;
    }
    tree[virtual_root].parent = NULL_NODE;
    x = sites_position[end - 1];
    ret = seek_tree_position(&tp, x);
    if (ret != 0) {
        goto out;
    }
    in = tsls_index_get_edges(self->index, tp.in.order);
    for (j = tp.in.start; j != tp.in.stop; j += tp.direction) {
        if (seek_edge_covers(&tp, &in[j], x)) {
            tree[in[j].child].parent = in[j].parent;
        }
    }
    for (site = end - 1; site >= start; site--) {
        while (sites_position[site] < tp.interval.left) {
            valid = tsk_tree_position_prev(&tp);
            assert(valid);
            (void) valid;
            out = tsls_index_get_edges(self->index, tp.out.order);
            in = tsls_index_get_edges(self->index, tp.in.order);
            for (j = tp.out.start; j != tp.out.stop; j--) {
                tree[out[j].child].parent = virtual_root;
            }
            for (j = tp.in.start; j != tp.in.stop; j--) {
                tree[in[j].child].parent = in[j].parent;
            }
        }
        tree[virtual_root].allelic_state = self->index->ancestral_state[site];
        tsls_hmm_set_allelic_state(self, site);
        u = path_parent[path_length];
        while (tree[u].allelic_state == TSK_NULL) {
            u = tree[u].parent;
        }
        match[site] = tree[u].allelic_state;
        tsls_hmm_clear_allelic_state(self, site);
        u = path_parent[path_length];

        /* Traverse up the tree from the current node. The first node in the
         * traceback's set that we meet tells us whether we need to
         * recombine. Every root has a likelihood, and so is in the set. */
        while (tree[u].traceback_flag == -1) {
            u = tree[u].parent;
            assert(u != virtual_root);
        }
        if (tree[u].traceback_flag && site > start) {
            max_likelihood_node = self->max_likelihood_node[site - 1];
            assert(max_likelihood_node != NULL_NODE);
            path_left[path_length] = sites_position[site];
            path_length++;
            /* Start the next output edge */
            path_right[path_length] = sites_position[site];
            path_parent[path_length] = max_likelihood_node;
        }
        tsls_hmm_undo_traceback_delta(self, site, tree);
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
    self->num_sparse_sites = 0;
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

double
tsls_hmm_get_mean_num_values(const tsls_hmm_t *self)
{
    if (self->stats_num_sites == 0) {
        return 0;
    }
    return (double) self->stats_total_values / (double) self->stats_num_sites;
}

double
tsls_hmm_get_mean_num_updated(const tsls_hmm_t *self)
{
    if (self->stats_num_sites == 0) {
        return 0;
    }
    return (double) self->stats_total_updated / (double) self->stats_num_sites;
}

size_t
tsls_hmm_get_total_memory(tsls_hmm_t *self)
{
    return self->traceback_allocator.total_size;
}
