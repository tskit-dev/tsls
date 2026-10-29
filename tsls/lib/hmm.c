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

/* The tree is kept as parent links and numbers of children: the walks only
 * go up the tree, and a node is in the tree if it has a parent or a child. */
static inline void
remove_edge(
    tsk_id_t p, tsk_id_t c, tsls_node_t *restrict tree, int32_t *restrict num_children)
{
    tree[c].parent = NULL_NODE;
    num_children[p]--;
}

static inline void
insert_edge(
    tsk_id_t p, tsk_id_t c, tsls_node_t *restrict tree, int32_t *restrict num_children)
{
    tree[c].parent = p;
    num_children[p]++;
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

/* Store the normalised Viterbi values of the specified nodes at this site,
 * exp(-(cost - best_cost)), and their recombination_required flags. */
static int WARN_UNUSED
compressed_matrix_store_costs(compressed_matrix_t *self, tsk_id_t site,
    double normalisation_factor, int num_values, const tsk_id_t *restrict nodes,
    const tsls_likelihood_t *restrict L, const tsls_node_t *restrict tree,
    double best_cost)
{
    int ret = 0;
    int j;
    tsk_id_t *restrict nodes_out;
    double *restrict values_out;
    int8_t *restrict flags_out;

    flags_out = tsi_blkalloc_get(
        &self->allocator, aligned_size((size_t) num_values * sizeof(int8_t)));
    values_out = tsi_blkalloc_get(
        &self->allocator, aligned_size((size_t) num_values * sizeof(double)));
    nodes_out = tsi_blkalloc_get(
        &self->allocator, aligned_size((size_t) num_values * sizeof(tsk_id_t)));
    if (flags_out == NULL || values_out == NULL || nodes_out == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    for (j = 0; j < num_values; j++) {
        nodes_out[j] = nodes[j];
        values_out[j] = exp(-(L[nodes[j]].current - best_cost));
        flags_out[j] = tree[nodes[j]].recombination_required;
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
    return self->tree[u].parent != NULL_NODE || self->num_children[u] > 0;
}

/* Returns the cost of a node whose cost last changed to cost at site,
 * after the last site processed: cost lowered by the smallest ceiling of
 * the sites after site. The stack holds the strictly increasing ceilings
 * back from the last site, so the smallest is that of the first entry
 * after site. */
static inline double
tsls_hmm_deferred_cost(const tsls_hmm_t *self, double cost, int32_t site)
{
    const int32_t *restrict stack_site = self->ceiling_stack_site;
    int lo = 0;
    int hi = self->ceiling_stack_size;
    int mid;

    while (lo < hi) {
        mid = lo + (hi - lo) / 2;
        if (stack_site[mid] <= site) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < self->ceiling_stack_size
               ? TSK_MIN(cost, self->ceiling_stack_value[lo])
               : cost;
}

/* Returns whether two costs at the current site are equivalent: equal, or
 * both dead (at least dead_cost, the next site's ceiling). Dead regions
 * cost the next site's ceiling plus their emission there, with the flag
 * set, whatever their costs here, and neither can be the best here
 * (dead_cost is only finite if it is above the best). */
static inline bool
tsls_hmm_equivalent_costs(const tsls_hmm_t *self, double x, double y)
{
    return x == y || (x >= self->dead_cost && y >= self->dead_cost);
}

static void
tsls_hmm_check_state(const tsls_hmm_t *self)
{
    int num_likelihoods = 0;
    int j;
    tsk_id_t u, v, a;

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
        /* Nodes with likelihoods are marked as listed for the walks */
        assert((self->tree[u].cache_stamp == TSLS_LISTED_STAMP)
               == (self->likelihood[u].value >= 0));
        assert(self->tree[u].cache_stamp == TSLS_LISTED_STAMP
               || self->tree[u].cache_stamp <= self->cache_stamp);
        if (self->likelihood[u].value < 0 && tsls_hmm_in_tree(self, u)
            && self->tree[u].cache_stamp == self->cache_stamp) {
            /* A cached walk leads to the nearest ancestor with a likelihood */
            v = self->tree[u].parent;
            while (self->likelihood[v].value < 0) {
                v = self->tree[v].parent;
            }
            a = self->tree[u].listed_ancestor;
            while (self->tree[a].cache_stamp != TSLS_LISTED_STAMP) {
                a = self->tree[a].listed_ancestor;
            }
            assert(a == v);
        }
    }
    assert(num_likelihoods == self->num_likelihood_nodes);
    (void) num_likelihoods;
    if (self->mode == MODE_VITERBI) {
        /* A listed node's cost last changed at a site already processed,
         * and its state gives its cost (cached in current) */
        for (j = 0; j < self->num_likelihood_nodes; j++) {
            u = self->likelihood_nodes[j];
            assert(self->likelihood[u].site <= self->last_site);
            assert(self->currents_stale
                   || tsls_hmm_deferred_cost(self, self->likelihood[u].value,
                          self->likelihood[u].site)
                          == self->likelihood[u].current);
        }
        for (j = 1; j < self->ceiling_stack_size; j++) {
            assert(self->ceiling_stack_site[j - 1] < self->ceiling_stack_site[j]);
            assert(self->ceiling_stack_value[j - 1] < self->ceiling_stack_value[j]);
        }
    }
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
    fprintf(out, "id\tparent\tnum_children\tlikelihood\n");
    for (j = 0; j < (int) self->num_nodes; j++) {
        fprintf(out, "%d\t%d\t%d\t%f\n", (int) j, self->tree[j].parent,
            self->num_children[j], self->likelihood[j].value);
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
            fprintf(out, "(%d, %d, %d, %f)",
                (int) (self->traceback[j].entries[k].node & ~TSLS_DELTA_LISTED),
                (self->traceback[j].entries[k].node & TSLS_DELTA_LISTED) != 0,
                self->traceback[j].entries[k].site, self->traceback[j].entries[k].cost);
        }
        fprintf(out, "\n");
    }
    tsi_blkalloc_print_state(&self->traceback_allocator, out);
    return 0;
}

/* Returns -log(p), or infinity if p <= 0. */
static inline double
neg_log(double p)
{
    return p > 0 ? -log(p) : INFINITY;
}

/* Returns -log(1 + x), or infinity if x <= -1. */
static inline double
neg_log1p(double x)
{
    return x > -1 ? -log1p(x) : INFINITY;
}

/* Compute the per-site Viterbi costs from the rates (see tsls_hmm_t) for
 * the sites [start, end): only for the window, as short windows (tsinfer's
 * early ancestors) are matched against long sequences. */
static void
tsls_hmm_compute_costs(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end)
{
    const double n = (double) self->num_nodes;
    double r, mu, num_alleles, stay, match, mismatch, base;
    tsk_id_t j;

    for (j = start; j < end; j++) {
        r = self->recombination_rate[j];
        mu = self->mismatch_rate[j];
        num_alleles = (double) self->index->num_alleles[j];
        /* Staying has probability 1 - r + r / n, switching r / n */
        stay = neg_log1p(r / n - r);
        self->stay_cost[j] = stay;
        self->switch_cost[j] = stay == INFINITY ? 0 : neg_log(r / n) - stay;
        match = neg_log1p(-(num_alleles - 1) * mu);
        mismatch = neg_log(mu);
        base = TSK_MIN(match, mismatch);
        self->emission_base[j] = base;
        if (base == INFINITY) {
            self->match_cost[j] = INFINITY;
            self->mismatch_cost[j] = INFINITY;
        } else {
            self->match_cost[j] = match - base;
            self->mismatch_cost[j] = mismatch - base;
        }
    }
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
    self->index_min_sites = 16;
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
    self->num_children = tsls_large_alloc((N + 1) * sizeof(*self->num_children));
    self->num_below = tsls_large_alloc((N + 1) * sizeof(*self->num_below));
    self->num_inheriting = tsls_large_alloc((N + 1) * sizeof(*self->num_inheriting));
    self->likelihood_nodes = tsls_large_alloc((N + 1) * sizeof(*self->likelihood_nodes));
    self->likelihood_nodes_tmp = tsls_large_alloc((N + 1) * sizeof(*self->likelihood_nodes_tmp));
    self->values_tmp = tsls_large_alloc((N + 1) * sizeof(*self->values_tmp));
    self->traceback_flags_tmp
        = tsls_large_alloc((N + 1) * sizeof(*self->traceback_flags_tmp));
    /* Each node is recorded at most once in each part of a site's delta */
    self->delta_buffer = tsls_large_alloc((2 * N + 2) * sizeof(*self->delta_buffer));
    self->stamp = tsls_large_alloc((N + 1) * sizeof(*self->stamp));
    self->switch_cost = malloc((self->num_sites + 1) * sizeof(double));
    self->match_cost = malloc((self->num_sites + 1) * sizeof(double));
    self->mismatch_cost = malloc((self->num_sites + 1) * sizeof(double));
    self->emission_base = malloc((self->num_sites + 1) * sizeof(double));
    self->stay_cost = malloc((self->num_sites + 1) * sizeof(double));
    self->ceiling = malloc((self->num_sites + 1) * sizeof(double));
    self->ceiling_pse = malloc((self->num_sites + 1) * sizeof(int32_t));
    self->ceiling_stack_site = malloc((self->num_sites + 1) * sizeof(int32_t));
    self->ceiling_stack_value = malloc((self->num_sites + 1) * sizeof(double));
    self->preorder = tsls_large_alloc((N + 1) * sizeof(*self->preorder));
    self->subtree_end = tsls_large_alloc((N + 1) * sizeof(*self->subtree_end));
    self->preorder_node = tsls_large_alloc((N + 1) * sizeof(*self->preorder_node));
    self->numbered = tsls_large_alloc((N + 1) * sizeof(*self->numbered));
    self->first_child = tsls_large_alloc((N + 1) * sizeof(*self->first_child));
    self->list_position = tsls_large_alloc((N + 1) * sizeof(*self->list_position));
    self->touched = tsls_large_alloc((N + 1) * sizeof(*self->touched));
    self->prev_touched = tsls_large_alloc((N + 1) * sizeof(*self->prev_touched));
    self->listed_bits = calloc(N / 64 + 2, sizeof(*self->listed_bits));
    self->range_offset = malloc((self->num_sites + 2) * sizeof(*self->range_offset));

    if (self->recombination_rate == NULL
        || self->mismatch_rate == NULL || self->traceback == NULL
        || self->max_likelihood_node == NULL || self->tree == NULL
        || self->likelihood == NULL
        || self->num_children == NULL || self->num_below == NULL
        || self->num_inheriting == NULL || self->likelihood_nodes == NULL
        || self->likelihood_nodes_tmp == NULL || self->values_tmp == NULL
        || self->traceback_flags_tmp == NULL || self->delta_buffer == NULL
        || self->stamp == NULL || self->switch_cost == NULL
        || self->match_cost == NULL || self->mismatch_cost == NULL
        || self->emission_base == NULL || self->stay_cost == NULL
        || self->ceiling == NULL || self->ceiling_pse == NULL
        || self->ceiling_stack_site == NULL || self->ceiling_stack_value == NULL
        || self->preorder == NULL || self->subtree_end == NULL
        || self->preorder_node == NULL || self->numbered == NULL
        || self->first_child == NULL || self->list_position == NULL
        || self->touched == NULL || self->prev_touched == NULL
        || self->listed_bits == NULL || self->range_offset == NULL) {
        ret = TSLS_ERR_NO_MEMORY;
        goto out;
    }
    /* Alloc in blocks of at least 4MiB, and large enough to hold the
     * traceback delta for a site at which every node changes. */
    self->traceback_block_size
        = TSK_MAX(4 * 1024 * 1024, (2 * N + 2) * sizeof(traceback_entry_t));
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
    memset(self->preorder, 0xff, (N + 1) * sizeof(*self->preorder));
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
    tsls_large_free_array(self->num_children, self->num_nodes);
    tsls_large_free_array(self->num_below, self->num_nodes);
    tsls_large_free_array(self->num_inheriting, self->num_nodes);
    tsls_large_free_array(self->likelihood_nodes, self->num_nodes);
    tsls_large_free_array(self->likelihood_nodes_tmp, self->num_nodes);
    tsls_large_free_array(self->values_tmp, self->num_nodes);
    tsls_large_free_array(self->traceback_flags_tmp, self->num_nodes);
    tsls_large_free_array(self->delta_buffer, 2 * self->num_nodes + 1);
    tsls_large_free_array(self->stamp, self->num_nodes);
    tsi_safe_free(self->switch_cost);
    tsi_safe_free(self->match_cost);
    tsi_safe_free(self->mismatch_cost);
    tsi_safe_free(self->emission_base);
    tsi_safe_free(self->stay_cost);
    tsi_safe_free(self->ceiling);
    tsi_safe_free(self->ceiling_pse);
    tsi_safe_free(self->ceiling_stack_site);
    tsi_safe_free(self->ceiling_stack_value);
    tsls_large_free_array(self->preorder, self->num_nodes);
    tsls_large_free_array(self->subtree_end, self->num_nodes);
    tsls_large_free_array(self->preorder_node, self->num_nodes);
    tsls_large_free_array(self->numbered, self->num_nodes);
    tsls_large_free_array(self->first_child, self->num_nodes);
    tsls_large_free_array(self->list_position, self->num_nodes);
    tsls_large_free_array(self->touched, self->num_nodes);
    tsls_large_free_array(self->prev_touched, self->num_nodes);
    tsi_safe_free(self->listed_bits);
    tsi_safe_free(self->range_offset);
    tsi_safe_free(self->ranges);
    tsi_safe_free(self->range_stack);
    tsi_safe_free(self->segment_start);
    tsi_safe_free(self->segment_allele);
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

/* Number a tree for sparse sites (see tsls_hmm_number_tree) if at least
 * this many of its remaining sites could be sparse; 1 numbers every tree
 * with a site left after the first. For tuning and tests. */
int
tsls_hmm_set_index_min_sites(tsls_hmm_t *self, int min_sites)
{
    int ret = 0;

    if (min_sites < 1) {
        ret = TSLS_ERR_BAD_PARAM_VALUE;
        goto out;
    }
    self->index_min_sites = min_sites;
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
    tsk_id_t u;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = self->likelihood_nodes[j];
        self->values_tmp[j] = self->mode == MODE_VITERBI ? self->likelihood[u].current
                                                         : self->likelihood[u].value;
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

/* Clear all the walk caches (see tsls_node_t). The caches stay valid as
 * long as no node on a cached path gets a likelihood or a new parent, so
 * this is only needed when one does (see tsls_hmm_mark_listed and
 * tsls_hmm_remove_edge). Walks stop at listed nodes whatever their stamp,
 * (they have the largest stamp), so a new stamp is enough. */
static void
tsls_hmm_clear_walk_caches(tsls_hmm_t *self)
{
    tsls_node_t *restrict tree = self->tree;
    size_t k;

    self->cache_stamp++;
    if (unlikely(self->cache_stamp == TSLS_LISTED_STAMP)) {
        for (k = 0; k < self->num_nodes; k++) {
            if (tree[k].cache_stamp != TSLS_LISTED_STAMP) {
                tree[k].cache_stamp = 0;
            }
        }
        self->cache_stamp = 1;
    }
}

/* Mark u, which is being given a likelihood, as listed for the walk
 * caches. If u has a cached walk, walks from below may have passed through
 * it to an ancestor that is no longer the nearest, so the caches are
 * cleared first. The caller must not have walked from u itself since it
 * last changed. */
static inline void
tsls_hmm_mark_listed(tsls_hmm_t *self, tsk_id_t u)
{
    tsls_node_t *restrict tree = self->tree;

    if (unlikely(tree[u].cache_stamp == self->cache_stamp)) {
        tsls_hmm_clear_walk_caches(self);
    }
    tree[u].cache_stamp = TSLS_LISTED_STAMP;
    tree[u].listed_ancestor = u;
}

/* Mark u, which has just lost its likelihood, as a node without one whose
 * nearest ancestor with a likelihood is ancestor (via listed_ancestor):
 * walks that reach it continue there. */
static inline void
tsls_hmm_mark_unlisted(tsls_hmm_t *self, tsk_id_t u, tsk_id_t ancestor)
{
    self->tree[u].listed_ancestor = ancestor;
    self->tree[u].cache_stamp = self->cache_stamp;
}

static inline void
tsls_hmm_add_likelihood(tsls_hmm_t *self, tsk_id_t u, double value)
{
    tsls_hmm_mark_listed(self, u);
    self->likelihood[u].value = value;
    self->likelihood_nodes[self->num_likelihood_nodes] = u;
    self->num_likelihood_nodes++;
}

/* Record u's Viterbi state before it changes, at most once in each part of
 * the site's traceback delta (see traceback_delta_t): whether it is listed,
 * its cost and the site of its last change. */
static inline void
tsls_hmm_record_state(
    tsls_hmm_t *self, tsk_id_t u, bool listed, double cost, int32_t site)
{
    traceback_entry_t *entry;

    if (self->stamp[u] != self->current_stamp) {
        self->stamp[u] = self->current_stamp;
        entry = &self->delta_buffer[self->delta_size];
        self->delta_size++;
        entry->node = (uint32_t) u | (listed ? TSLS_DELTA_LISTED : 0);
        entry->site = site;
        entry->cost = cost;
    }
}

/* Record u's current state before it changes, in Viterbi. */
static inline void
tsls_hmm_record(tsls_hmm_t *self, tsk_id_t u)
{
    const tsls_likelihood_t *L = &self->likelihood[u];

    if (self->mode == MODE_VITERBI) {
        tsls_hmm_record_state(self, u, L->value >= 0, L->value, L->site);
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

/* Returns the nearest ancestor of u (or u itself) with a likelihood. The
 * answer is cached on the nodes without likelihoods on the path (see
 * tsls_node_t), so that later walks through them stop there; the caches
 * stay valid across sites until a node on a cached path gets a likelihood
 * or a new parent. A walk stops at a node with the current stamp, which is
 * either a node with a likelihood (its own listed_ancestor) or a cached
 * node, whose listed_ancestor may be a node that has since lost its
 * likelihood, and so is followed until a node with a likelihood. This is
 * the innermost loop of the HMM, so it has no asserts (every root has a
 * likelihood, which tsls_hmm_check_state checks). */
static ALWAYS_INLINE tsk_id_t
tsls_hmm_find_listed(tsls_hmm_t *self, tsk_id_t u)
{
    tsls_node_t *restrict tree = self->tree;
    const uint32_t stamp = self->cache_stamp;
    tsk_id_t v, a;

    v = u;
    while (tree[v].cache_stamp < stamp) {
        v = tree[v].parent;
    }
    a = tree[v].listed_ancestor;
    while (unlikely(tree[a].cache_stamp != TSLS_LISTED_STAMP)) {
        a = tree[a].listed_ancestor;
    }
    v = u;
    while (tree[v].cache_stamp < stamp) {
        tree[v].listed_ancestor = a;
        tree[v].cache_stamp = stamp;
        v = tree[v].parent;
    }
    return a;
}

/* Returns the likelihood inherited by node u. */
static inline double
tsls_hmm_get_inherited_likelihood(tsls_hmm_t *self, tsk_id_t u)
{
    return self->likelihood[tsls_hmm_find_listed(self, u)].value;
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
    tsk_id_t a;

    if (self->track_num_below) {
        tsls_hmm_update_num_below(self, p, -self->num_below[c]);
    }
    if (update_likelihoods && L[c].value == NULL_LIKELIHOOD) {
        /* c keeps the value (and in Viterbi the state) it inherited */
        a = tsls_hmm_find_listed(self, p);
        tsls_hmm_record(self, c);
        tsls_hmm_add_likelihood(self, c, L[a].value);
        L[c].site = L[a].site;
        L[c].current = L[a].current;
    }
    remove_edge(p, c, self->tree, self->num_children);
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
 * likelihood 0 (in Viterbi, infinite cost, as of the last site), and so
 * can only be reached by recombination. */
static inline void
tsls_hmm_insert_edge(tsls_hmm_t *self, tsk_id_t p, tsk_id_t c, bool update_likelihoods)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const double entering = self->mode == MODE_VITERBI ? INFINITY : 0;

    if (update_likelihoods) {
        if (L[p].value == NOT_IN_TREE) {
            tsls_hmm_record(self, p);
            tsls_hmm_add_likelihood(self, p, entering);
            L[p].site = self->last_site;
            L[p].current = entering;
        }
        if (L[c].value == NOT_IN_TREE) {
            tsls_hmm_record(self, c);
            tsls_hmm_add_likelihood(self, c, entering);
            L[c].site = self->last_site;
            L[c].current = entering;
        }
        /* Otherwise the child was a root and keeps its likelihood */
        assert(L[c].value >= 0);
    }
    insert_edge(p, c, self->tree, self->num_children);
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

    /* The numbering is for one tree (see tsls_hmm_number_tree) */
    self->tree_numbered = false;
    for (j = tp->out.start; j != tp->out.stop; j += direction) {
        tsls_hmm_remove_edge(
            self, out[j].parent, out[j].child, true, isolated, &num_isolated);
    }
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
            /* An isolated node has no descendants, so no walk ends here */
            tsls_hmm_record(self, u);
            L[u].value = NOT_IN_TREE;
            self->tree[u].cache_stamp = 0;
            num_left++;
        }
    }
    if (num_left > 0) {
        tsls_hmm_remove_left_nodes(self);
    }
}

static void tsls_hmm_clear_numbering(tsls_hmm_t *self);

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

    memset(self->num_children, 0, N * sizeof(*self->num_children));
    for (k = 0; k < N; k++) {
        /* Every node starts as a root. The likelihoods are set below. */
        self->tree[k].cache_stamp = 0;
        self->tree[k].parent = NULL_NODE;
        self->tree[k].allelic_state = ROOT_ALLELE;
        self->tree[k].recombination_required = -1;
        self->tree[k].traceback_flag = -1;
        self->num_below[k] = 1;
        /* In Viterbi, every value is as of the site before the first */
        self->likelihood[k].site = self->last_site;
    }
    self->cache_stamp = 1;
    tsls_hmm_clear_numbering(self);
    self->currents_stale = false;
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
        } else if (self->num_children[u] > 0) {
            tsls_hmm_add_likelihood(self, u, initial_value);
            self->likelihood[u].current = initial_value;
        } else {
            self->likelihood[u].value = NOT_IN_TREE;
        }
    }
out:
    return ret;
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
            /* Walk from the parent, so that the node's own cache (if any)
             * is seen when it's marked (see tsls_hmm_mark_listed) */
            u = tsls_hmm_find_listed(self, tree[node].parent);
            tsls_hmm_add_likelihood(self, node, L[u].value);
        }
    }
}

/* As tsls_hmm_insert_mutation_likelihoods for Viterbi: each node inserted
 * takes the cost and site of the node it inherits from, which is recorded
 * as its state before the site (when it is not listed). */
static void
tsls_hmm_insert_mutation_costs(tsls_hmm_t *self, const tsk_id_t site)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsls_node_t *restrict tree = self->tree;
    const tsls_index_t *index = self->index;
    const tsk_size_t stop = index->site_mutations_offset[site + 1];
    tsk_size_t j;
    tsk_id_t a, node;

    for (j = index->site_mutations_offset[site]; j < stop; j++) {
        node = index->mutation_node[j];
        if (L[node].value == NULL_LIKELIHOOD) {
            /* Walk from the parent, so that the node's own cache (if any)
             * is seen when it's marked (see tsls_hmm_mark_listed) */
            a = tsls_hmm_find_listed(self, tree[node].parent);
            tsls_hmm_record_state(self, node, false, L[a].value, L[a].site);
            tsls_hmm_add_likelihood(self, node, L[a].value);
            L[node].site = L[a].site;
            L[node].current = L[a].current;
        }
    }
}

/* Set the ancestor of every node in the likelihood list to its nearest
 * proper ancestor with a likelihood (NULL_NODE for a root). The walks are
 * cached (tsls_hmm_find_listed), so the cost is the size of the union of
 * the paths not already cached, up to the nearest nodes with likelihoods,
 * which are much shorter than the paths up to a mutation or root. */
static void
tsls_hmm_set_listed_ancestors(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsls_node_t *restrict tree = self->tree;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int num_likelihood_nodes = self->num_likelihood_nodes;
    tsk_id_t u, p;
    int j;

    for (j = 0; j < num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p = tree[u].parent;
        L[u].ancestor = p == NULL_NODE ? NULL_NODE : tsls_hmm_find_listed(self, p);
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
get_listed_allelic_state(tsls_node_t *restrict tree,
    const tsls_likelihood_t *restrict L, tsk_id_t u, allele_t ancestral_state,
    tsk_id_t *restrict touched, int *num_touched)
{
    allele_t state;
    tsk_id_t v = u;
    int n = *num_touched;

    while (tree[v].allelic_state == TSK_NULL) {
        v = L[v].ancestor;
    }
    state = tree[v].allelic_state;
    while (u != v) {
        tree[u].allelic_state = state;
        touched[n] = u;
        n++;
        u = L[u].ancestor;
    }
    *num_touched = n;
    return likely(state != ROOT_ALLELE) ? state : ancestral_state;
}

/* Remove the Viterbi likelihoods that are equal to those of the nearest
 * ancestor with a likelihood, with the same recombination_required flag, so
 * that the traceback is unchanged (see tsls_hmm_coalesce_likelihoods). The
 * ancestor of every node in the list must be up to date. A node
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
        a = L[u].ancestor;
        if (a != NULL_NODE && tsls_hmm_equivalent_costs(self, L[u].current, L[a].current)
            && tree[u].recombination_required == tree[a].recombination_required) {
            removed[num_removed] = u;
            num_removed++;
        } else {
            L_nodes[num_likelihood_nodes] = u;
            num_likelihood_nodes++;
        }
    }
    for (j = 0; j < num_removed; j++) {
        u = removed[j];
        tsls_hmm_record(self, u);
        L[u].value = NULL_LIKELIHOOD;
        tsls_hmm_mark_unlisted(self, u, L[u].ancestor);
    }
    self->num_likelihood_nodes = num_likelihood_nodes;
}

/* Remove likelihoods that are equal to the likelihood inherited from the
 * parent (in the forward and backward algorithms). */
static ALWAYS_INLINE void
tsls_hmm_coalesce_likelihoods(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    tsk_id_t u, p, a;
    const int old_num_likelihood_nodes = self->num_likelihood_nodes;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    int j, num_likelihood_nodes;

    num_likelihood_nodes = 0;
    for (j = 0; j < old_num_likelihood_nodes; j++) {
        u = L_nodes[j];
        p = tree[u].parent;
        if (p != NULL_NODE) {
            /* A removed node had the inherited value, so walks through it
             * to its own ancestor give the same answer. */
            a = tsls_hmm_find_listed(self, p);
            if (L[u].value == L[a].value) {
                L[u].value = NULL_LIKELIHOOD;
                tsls_hmm_mark_unlisted(self, u, a);
            }
        }
        if (L[u].value >= 0) {
            L_nodes[num_likelihood_nodes] = L_nodes[j];
            num_likelihood_nodes++;
        }
    }
    self->num_likelihood_nodes = num_likelihood_nodes;
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

/* Store the changes recorded at this site as its traceback delta (see
 * traceback_delta_t), and start recording the next site's. */
static int WARN_UNUSED
tsls_hmm_store_traceback(tsls_hmm_t *self, const tsk_id_t site)
{
    int ret = 0;
    traceback_delta_t *restrict delta = &self->traceback[site];
    const tsls_likelihood_t *restrict L = self->likelihood;
    traceback_entry_t *restrict entries = self->delta_buffer;
    int j, size;

    uint32_t *restrict stamp = self->stamp;
    int num_transition;
    tsk_id_t u;

    /* A node that was unlisted before the transitions and is after the
     * site needs no entries, as the traceback only reads the states of
     * listed nodes: such as the root of a subtree moved by a transition,
     * given its inherited state there and then coalesced, or a mutation
     * node coalesced straight away. Such nodes with transition entries are
     * marked with stamp 0, which no part uses. */
    num_transition = 0;
    for (j = 0; j < self->delta_num_transition; j++) {
        u = (tsk_id_t) (entries[j].node & ~TSLS_DELTA_LISTED);
        if (!(entries[j].node & TSLS_DELTA_LISTED) && L[u].value < 0) {
            stamp[u] = 0;
        } else {
            entries[num_transition] = entries[j];
            num_transition++;
        }
    }
    size = num_transition;
    for (j = self->delta_num_transition; j < self->delta_size; j++) {
        u = (tsk_id_t) (entries[j].node & ~TSLS_DELTA_LISTED);
        if (stamp[u] != 0
            && ((entries[j].node & TSLS_DELTA_LISTED) || L[u].value >= 0)) {
            entries[size] = entries[j];
            size++;
        }
    }
    delta->num_transition = num_transition;
    delta->size = size;
    delta->entries = NULL;
    if (size > 0) {
        delta->entries = tsi_blkalloc_get(&self->traceback_allocator,
            aligned_size((size_t) size * sizeof(traceback_entry_t)));
        if (delta->entries == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        memcpy(delta->entries, self->delta_buffer,
            (size_t) size * sizeof(traceback_entry_t));
    }
    self->total_traceback_size += (size_t) self->num_likelihood_nodes;
    self->total_delta_size += (size_t) size;
    self->delta_size = 0;
    self->delta_num_transition = 0;
out:
    return ret;
}

/* Push a site's ceiling onto the stack of ceilings, which are strictly
 * increasing back from the last site, and return the last earlier site
 * whose ceiling is strictly smaller (-1 if none). */
static inline int32_t
tsls_hmm_push_ceiling(tsls_hmm_t *self, tsk_id_t site, double ceiling)
{
    int k = self->ceiling_stack_size;
    int32_t pse;

    while (k > 0 && self->ceiling_stack_value[k - 1] >= ceiling) {
        k--;
    }
    pse = k > 0 ? self->ceiling_stack_site[k - 1] : -1;
    self->ceiling_stack_site[k] = site;
    self->ceiling_stack_value[k] = ceiling;
    self->ceiling_stack_size = k + 1;
    return pse;
}

/* Process a Viterbi site, in costs (see docs/likelihood_model.md). With
 * C* the best cost after the previous site, each node's cost becomes
 *
 *     min(C, C* + switch_cost) + emission cost,
 *
 * and its recombination_required flag is C >= C* + switch_cost (the
 * ceiling). Nodes whose cost is more than prune_cost above the new best
 * get the cost of likelihood_threshold, floor_cost above it, which also
 * caps every cost. The changes are recorded for the traceback. If
 * skip_dead is true, nodes too costly to matter at the next site don't
 * change (see below), so the values stored are not all exact. */
static int WARN_UNUSED
tsls_hmm_process_site_viterbi(tsls_hmm_t *self, const tsk_id_t site,
    const allele_t state, bool skip_dead, compressed_matrix_t *output)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    int ret = 0;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsls_node_t *restrict tree = self->tree;
    const double ceiling = self->best_cost + self->switch_cost[site];
    const double match_cost = self->match_cost[site];
    const double mismatch_cost = self->mismatch_cost[site];
    const allele_t ancestral_state = self->index->ancestral_state[site];
    tsk_id_t *restrict touched = self->likelihood_nodes_tmp;
    double *restrict new_cost = self->values_tmp;
    /* The recombination_required flag, plus 2 if the node mismatched */
    int8_t *restrict flag = self->traceback_flags_tmp;
    int num_touched = 0;
    double best, cost, prune_limit, floor, dead, matched;
    tsk_id_t u, best_node;
    allele_t allele;
    int j;

    self->ceiling[site] = ceiling;
    self->ceiling_pse[site] = tsls_hmm_push_ceiling(self, site, ceiling);
    if (state != TSLS_MISSING_DATA) {
        tsls_hmm_insert_mutation_costs(self, site);
    }
    tsls_hmm_set_allelic_state(self, site);
    self->site_num_updated = (size_t) self->num_likelihood_nodes;
    /* Find the list's own tree, for the alleles and coalescing */
    tsls_hmm_set_listed_ancestors(self);
    best = INFINITY;
    best_node = NULL_NODE;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        cost = L[u].current;
        flag[j] = cost >= ceiling;
        cost = TSK_MIN(cost, ceiling);
        if (state == TSLS_MISSING_DATA) {
            cost += match_cost;
        } else {
            allele = get_listed_allelic_state(
                tree, L, u, ancestral_state, touched, &num_touched);
            if (allele == state) {
                cost += match_cost;
            } else {
                cost += mismatch_cost;
                flag[j] = (int8_t) (flag[j] + 2);
            }
        }
        new_cost[j] = cost;
        if (cost < best) {
            best = cost;
            best_node = u;
        }
    }
    clear_touched_allelic_state(tree, touched, num_touched);
    tsls_hmm_unset_allelic_state(self, site);

    if (best == INFINITY) {
        if (self->mismatch_rate[site] <= 0 || self->mismatch_rate[site] >= 1) {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA;
        } else if (self->recombination_rate[site] == 0) {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION;
        } else {
            ret = TSLS_ERR_MATCH_IMPOSSIBLE;
        }
        goto out;
    }
    self->max_likelihood_node[site] = best_node;
    prune_limit = best + self->prune_cost;
    floor = best + self->floor_cost;
    /* A node is dead if it costs at least the next site's ceiling, dead,
     * whether or not it mismatches here: it then costs dead + its emission
     * at the next site, with the flag set, whatever its cost here, and
     * can't be the best here. So a dead node that mismatches can be given
     * the cost of matching, which saves changing it at both sites. (Only
     * when the cap, floor, is at least dead, and matching is free.) */
    dead = INFINITY;
    if (skip_dead && site + 1 < self->end_site && match_cost == 0
        && self->switch_cost[site + 1] > 0) {
        dead = best + self->switch_cost[site + 1];
        if (!(dead <= floor)) {
            dead = INFINITY;
        }
    }
    self->dead_cost = dead;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        u = L_nodes[j];
        cost = new_cost[j];
        /* The cost if the node matches, which its state gives already */
        matched = TSK_MIN(L[u].current, ceiling);
        if (flag[j] >= 2) {
            if (matched >= dead && matched > best) {
                cost = matched;
            }
            flag[j] = (int8_t) (flag[j] - 2);
        }
        if (cost > prune_limit) {
            cost = floor;
        }
        cost = TSK_MIN(cost, floor);
        tree[u].recombination_required = flag[j];
        /* The state gives the cost as the cost it last changed to,
         * lowered by any smaller ceiling since (see tsls_hmm_run_traceback),
         * so it only changes if the node doesn't just match: a node can
         * also mismatch back to the same cost. */
        if (cost != matched) {
            tsls_hmm_record(self, u);
            L[u].value = cost;
            L[u].site = site;
        }
        L[u].current = cost;
    }
    tsls_hmm_count_distinct(self);
    /* Remove redundant nodes before storing the traceback, keeping those
     * whose recombination_required flag differs from the inherited one. */
    tsls_hmm_coalesce_listed_likelihoods(self);
    if (output != NULL) {
        ret = compressed_matrix_store_costs(output, site,
            exp(-((best - self->best_cost) + self->stay_cost[site]
                  + self->emission_base[site])),
            self->num_likelihood_nodes, L_nodes, L, tree, best);
        if (ret != 0) {
            goto out;
        }
    }
    self->best_cost = best;
    ret = tsls_hmm_store_traceback(self, site);
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
    tsls_hmm_coalesce_likelihoods(self);
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
    tsls_hmm_coalesce_likelihoods(self);
out:
    return ret;
}

/*===================================================================
 * Indexed trees
 *===================================================================
 */

/* Unset the current numbering (see tsls_hmm_t). */
static void
tsls_hmm_clear_numbering(tsls_hmm_t *self)
{
    int32_t j;

    for (j = 0; j < self->num_numbered; j++) {
        self->preorder[self->numbered[j]] = -1;
    }
    memset(self->listed_bits, 0,
        ((size_t) self->num_numbered / 64 + 1) * sizeof(*self->listed_bits));
    self->num_numbered = 0;
    self->tree_numbered = false;
}

/* Add u and its ancestors to the nodes to be numbered, stopping at the
 * first that is already there (see tsls_hmm_number_tree). */
static inline void
tsls_hmm_add_root_path(tsls_hmm_t *self, tsk_id_t u, int32_t *num_nodes)
{
    const tsls_node_t *restrict tree = self->tree;
    int32_t *restrict preorder = self->preorder;
    tsk_id_t *restrict nodes = self->numbered;
    tsk_id_t *restrict first_child = self->first_child;

    while (u != NULL_NODE && preorder[u] == -1) {
        preorder[u] = -2;
        first_child[u] = NULL_NODE;
        nodes[*num_nodes] = u;
        (*num_nodes)++;
        u = tree[u].parent;
    }
}

static inline void
tsls_hmm_set_listed_bit(tsls_hmm_t *self, tsk_id_t u)
{
    const int32_t x = self->preorder[u];

    self->listed_bits[x / 64] |= (uint64_t) 1 << (x % 64);
}

static inline void
tsls_hmm_clear_listed_bit(tsls_hmm_t *self, tsk_id_t u)
{
    const int32_t x = self->preorder[u];

    self->listed_bits[x / 64] &= ~((uint64_t) 1 << (x % 64));
}

/* Set the listed bits and list positions from the likelihood list. */
static void
tsls_hmm_set_list_index(tsls_hmm_t *self)
{
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    int j;

    memset(self->listed_bits, 0,
        ((size_t) self->num_numbered / 64 + 1) * sizeof(*self->listed_bits));
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        tsls_hmm_set_listed_bit(self, L_nodes[j]);
        self->list_position[L_nodes[j]] = j;
    }
}

static int
cmp_mutation_range(const void *a, const void *b)
{
    const mutation_range_t *x = (const mutation_range_t *) a;
    const mutation_range_t *y = (const mutation_range_t *) b;

    /* Outer ranges first, and later mutations on a node last */
    if (x->start != y->start) {
        return x->start < y->start ? -1 : 1;
    }
    if (x->end != y->end) {
        return x->end > y->end ? -1 : 1;
    }
    return (x->order > y->order) - (x->order < y->order);
}

/* Number the part of the current tree that indexed sites need, from the
 * first site of the tree's sites that remain, site, in preorder (see
 * tsls_hmm_t), and sort each remaining site's mutation ranges.
 *
 * The sites need the positions of the listed nodes, of the nodes with
 * mutations at the remaining sites (the only nodes that join the list while
 * the tree is numbered), of the best node, and of the parents of these
 * nodes. Numbering the union of these nodes' paths to the root in preorder
 * gives the same order and ancestry for them as numbering the whole tree,
 * at a cost in proportion to the union rather than to all the nodes. */
static int WARN_UNUSED
tsls_hmm_number_tree(tsls_hmm_t *self, tsk_id_t site)
{
    int ret = 0;
    const tsls_index_t *index = self->index;
    const tsls_node_t *restrict tree = self->tree;
    int32_t *restrict preorder = self->preorder;
    int32_t *restrict subtree_end = self->subtree_end;
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    tsk_id_t *restrict nodes = self->numbered;
    tsk_id_t *restrict first_child = self->first_child;
    /* The next sibling of each node, overwritten as the node is left */
    tsk_id_t *restrict next_sibling = subtree_end;
    tsk_id_t *restrict order = self->preorder_node;
    const tsk_size_t mutations_start = index->site_mutations_offset[site];
    const tsk_size_t mutations_stop
        = index->site_mutations_offset[self->tree_end_site];
    const size_t max_ranges = mutations_stop - mutations_start;
    const tsk_id_t best_node
        = site > 0 ? self->max_likelihood_node[site - 1] : NULL_NODE;
    mutation_range_t *ranges;
    tsk_id_t root, u, p, s, k_site;
    tsk_size_t i;
    int32_t num_nodes, j, k, num_ranges;
    void *tmp;

    tsls_hmm_clear_numbering(self);
    if (max_ranges > self->max_ranges) {
        tmp = realloc(self->ranges, max_ranges * sizeof(*self->ranges));
        if (tmp == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        self->ranges = tmp;
        tmp = realloc(self->range_stack, (max_ranges + 1) * sizeof(*self->range_stack));
        if (tmp == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        self->range_stack = tmp;
        tmp = realloc(
            self->segment_start, (2 * max_ranges + 2) * sizeof(*self->segment_start));
        if (tmp == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        self->segment_start = tmp;
        tmp = realloc(
            self->segment_allele, (2 * max_ranges + 2) * sizeof(*self->segment_allele));
        if (tmp == NULL) {
            ret = TSLS_ERR_NO_MEMORY;
            goto out;
        }
        self->segment_allele = tmp;
        self->max_ranges = max_ranges;
    }

    num_nodes = 0;
    for (j = 0; j < self->num_likelihood_nodes; j++) {
        tsls_hmm_add_root_path(self, L_nodes[j], &num_nodes);
    }
    for (i = mutations_start; i < mutations_stop; i++) {
        u = index->mutation_node[i];
        if (tsls_hmm_in_tree(self, u)) {
            tsls_hmm_add_root_path(self, u, &num_nodes);
        }
    }
    if (best_node != NULL_NODE && tsls_hmm_in_tree(self, best_node)) {
        tsls_hmm_add_root_path(self, best_node, &num_nodes);
    }
    for (j = 0; j < num_nodes; j++) {
        u = nodes[j];
        p = tree[u].parent;
        if (p != NULL_NODE) {
            next_sibling[u] = first_child[p];
            first_child[p] = u;
        }
    }
    k = 0;
    for (j = 0; j < num_nodes; j++) {
        root = nodes[j];
        if (tree[root].parent != NULL_NODE) {
            continue;
        }
        u = root;
        while (true) {
            preorder[u] = k;
            order[k] = u;
            k++;
            if (first_child[u] != NULL_NODE) {
                u = first_child[u];
                continue;
            }
            /* Leave u and the ancestors whose last child it is */
            while (u != root && next_sibling[u] == NULL_NODE) {
                subtree_end[u] = k;
                u = tree[u].parent;
            }
            if (u == root) {
                subtree_end[u] = k;
                break;
            }
            s = next_sibling[u];
            subtree_end[u] = k;
            u = s;
        }
    }
    assert(k == num_nodes);
    self->num_numbered = k;
    self->tree_numbered = true;
    tsls_hmm_set_list_index(self);

    /* The mutation ranges of each remaining site, sorted */
    ranges = self->ranges;
    num_ranges = 0;
    for (k_site = site; k_site < self->tree_end_site; k_site++) {
        self->range_offset[k_site] = num_ranges;
        for (i = index->site_mutations_offset[k_site];
             i < index->site_mutations_offset[k_site + 1]; i++) {
            u = index->mutation_node[i];
            if (preorder[u] >= 0) {
                ranges[num_ranges].start = preorder[u];
                ranges[num_ranges].end = subtree_end[u];
                ranges[num_ranges].order = (int32_t) i;
                ranges[num_ranges].state = index->mutation_derived_state[i];
                num_ranges++;
            }
        }
        qsort(ranges + self->range_offset[k_site],
            (size_t) (num_ranges - self->range_offset[k_site]), sizeof(*ranges),
            cmp_mutation_range);
    }
    self->range_offset[self->tree_end_site] = num_ranges;
    self->compact_size = self->num_likelihood_nodes;
    self->num_prev_touched = 0;
out:
    return ret;
}

static inline int
tsls_hmm_add_segment(tsls_hmm_t *self, int num, int32_t x, allele_t allele)
{
    if (num > 0 && self->segment_start[num - 1] == x) {
        self->segment_allele[num - 1] = allele;
    } else if (num == 0 || self->segment_allele[num - 1] != allele) {
        self->segment_start[num] = x;
        self->segment_allele[num] = allele;
        num++;
    }
    return num;
}

/* Find the alleles of the numbered positions at a site of the numbered
 * tree, as segments: segment k is [segment_start[k], segment_start[k + 1])
 * with segment_start[num] = num_numbered, all with segment_allele[k].
 * Returns the number of segments. */
static int
tsls_hmm_site_segments(tsls_hmm_t *self, tsk_id_t site)
{
    const mutation_range_t *restrict ranges = self->ranges;
    const int32_t stop = self->range_offset[site + 1];
    const allele_t ancestral_state = self->index->ancestral_state[site];
    int32_t *restrict stack = self->range_stack;
    int32_t j, r;
    int top, num;

    num = tsls_hmm_add_segment(self, 0, 0, ancestral_state);
    top = 0;
    for (j = self->range_offset[site]; j < stop; j++) {
        while (top > 0 && ranges[stack[top - 1]].end <= ranges[j].start) {
            top--;
            r = stack[top];
            num = tsls_hmm_add_segment(self, num, ranges[r].end,
                top > 0 ? ranges[stack[top - 1]].state : ancestral_state);
        }
        stack[top] = j;
        top++;
        num = tsls_hmm_add_segment(self, num, ranges[j].start, ranges[j].state);
    }
    while (top > 0) {
        top--;
        r = stack[top];
        num = tsls_hmm_add_segment(self, num, ranges[r].end,
            top > 0 ? ranges[stack[top - 1]].state : ancestral_state);
    }
    /* A segment starting at the end is empty */
    if (self->segment_start[num - 1] == self->num_numbered && num > 1) {
        num--;
    }
    self->segment_start[num] = self->num_numbered;
    return num;
}

/* Returns the allele at the numbered position x, given the site's
 * segments. */
static inline allele_t
tsls_hmm_segment_allele(const tsls_hmm_t *self, int num_segments, int32_t x)
{
    const int32_t *restrict start = self->segment_start;
    int lo = 0;
    int hi = num_segments;
    int mid;

    /* The last segment starting at or before x */
    while (hi - lo > 1) {
        mid = lo + (hi - lo) / 2;
        if (start[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return self->segment_allele[lo];
}

/* Bring every listed node's cached current cost up to date. */
static void
tsls_hmm_refresh_currents(tsls_hmm_t *self)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    int j;

    for (j = 0; j < self->num_likelihood_nodes; j++) {
        L[L_nodes[j]].current
            = tsls_hmm_deferred_cost(self, L[L_nodes[j]].value, L[L_nodes[j]].site);
    }
    self->currents_stale = false;
}

/* Returns whether u's recombination flag at site is set: whether its cost
 * after the previous site was at least the site's ceiling (see
 * tsls_hmm_run_traceback). For a node touched at the site, the flag was
 * computed then. */
static inline bool
tsls_hmm_flag_at(const tsls_hmm_t *self, tsk_id_t u, tsk_id_t site)
{
    const tsls_likelihood_t *L = &self->likelihood[u];

    if (L->site == site) {
        return self->tree[u].recombination_required;
    }
    return L->value >= self->ceiling[site] && self->ceiling_pse[site] <= L->site;
}

/* Remove u from the list of an indexed tree, given its nearest listed
 * ancestor. */
static void
tsls_hmm_unlist_indexed(tsls_hmm_t *self, tsk_id_t u, tsk_id_t ancestor)
{
    tsk_id_t *restrict L_nodes = self->likelihood_nodes;
    const int j = self->list_position[u];
    const tsk_id_t last = L_nodes[self->num_likelihood_nodes - 1];

    tsls_hmm_record(self, u);
    L_nodes[j] = last;
    self->list_position[last] = j;
    self->num_likelihood_nodes--;
    self->likelihood[u].value = NULL_LIKELIHOOD;
    tsls_hmm_mark_unlisted(self, u, ancestor);
    tsls_hmm_clear_listed_bit(self, u);
}

/* Remove the nodes among the specified ones that have the same cost and
 * flag at the site as their nearest listed ancestor. */
static void
tsls_hmm_coalesce_nodes(
    tsls_hmm_t *self, tsk_id_t site, const tsk_id_t *restrict nodes, int num_nodes)
{
    const tsls_likelihood_t *restrict L = self->likelihood;
    const tsls_node_t *restrict tree = self->tree;
    tsk_id_t u, a;
    int j;

    for (j = 0; j < num_nodes; j++) {
        u = nodes[j];
        if (L[u].value < 0 || tree[u].parent == NULL_NODE) {
            continue;
        }
        a = tsls_hmm_find_listed(self, tree[u].parent);
        if (tsls_hmm_equivalent_costs(self,
                tsls_hmm_deferred_cost(self, L[u].value, L[u].site),
                tsls_hmm_deferred_cost(self, L[a].value, L[a].site))
            && tsls_hmm_flag_at(self, u, site) == tsls_hmm_flag_at(self, a, site)) {
            tsls_hmm_unlist_indexed(self, u, a);
        }
    }
}

/* Remove every listed node with the same cost and flag at the site as its
 * nearest listed ancestor, in a preorder sweep of the listed bits with a
 * stack of the listed ancestors. */
static void
tsls_hmm_compact_indexed(tsls_hmm_t *self, tsk_id_t site)
{
    const tsls_likelihood_t *restrict L = self->likelihood;
    const int32_t *restrict preorder = self->preorder;
    const int32_t *restrict subtree_end = self->subtree_end;
    const tsk_id_t *restrict order = self->preorder_node;
    tsk_id_t *restrict stack = self->touched;
    const int32_t num_words = self->num_numbered / 64 + 1;
    uint64_t bits;
    int32_t w, x;
    int top = 0;
    tsk_id_t u, a;

    for (w = 0; w < num_words; w++) {
        bits = self->listed_bits[w];
        while (bits != 0) {
            x = w * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            u = order[x];
            while (top > 0 && subtree_end[stack[top - 1]] <= x) {
                top--;
            }
            if (top > 0) {
                /* The walk finds the same ancestor, but also caches the path
                 * to it, which a node's link to its ancestor relies on:
                 * listing a node on a cached path clears the caches. */
                a = tsls_hmm_find_listed(self, self->tree[u].parent);
                assert(a == stack[top - 1]);
                if (tsls_hmm_equivalent_costs(self,
                        tsls_hmm_deferred_cost(self, L[u].value, L[u].site),
                        tsls_hmm_deferred_cost(self, L[a].value, L[a].site))
                    && tsls_hmm_flag_at(self, u, site)
                           == tsls_hmm_flag_at(self, a, site)) {
                    tsls_hmm_unlist_indexed(self, u, a);
                    continue;
                }
            }
            stack[top] = u;
            top++;
        }
    }
    (void) preorder;
    self->compact_size = self->num_likelihood_nodes;
}

/* Process a Viterbi site of a numbered tree, touching only the listed nodes
 * that mismatch, if the best node matches; otherwise, or if matching isn't
 * free, the site is processed in full. Returns 1 if the site was processed
 * here and 0 if it must be processed in full.
 *
 * A node's state gives its cost as the cost it last changed to, lowered by
 * any smaller ceiling since, which is its cost if it matches (see
 * tsls_hmm_process_site_viterbi). If the best node matches, the best cost is
 * the same as at the previous site: its cost is at most the ceiling, and
 * the costs of the other nodes can only rise. So only the mismatching nodes
 * change: those in the segments of the site's mutation ranges with alleles
 * other than the query's. A node with a mutation at the site whose match
 * differs from its parent's gets its own state first, as a copy of the
 * state it inherits. The touched nodes are coalesced at this site and the
 * next, and the whole list when it has grown enough. */
static int WARN_UNUSED
tsls_hmm_process_site_indexed(
    tsls_hmm_t *self, tsk_id_t site, allele_t state, bool skip_dead)
{
    int ret = 0;
    tsls_likelihood_t *restrict L = self->likelihood;
    tsls_node_t *restrict tree = self->tree;
    const tsls_index_t *index = self->index;
    const int32_t *restrict preorder = self->preorder;
    const tsk_id_t *restrict order = self->preorder_node;
    const tsk_id_t best_node = site > 0 ? self->max_likelihood_node[site - 1] : NULL_NODE;
    const double best = self->best_cost;
    const double ceiling = best + self->switch_cost[site];
    const double mismatch_cost = self->mismatch_cost[site];
    const double prune_limit = best + self->prune_cost;
    const double floor = best + self->floor_cost;
    const tsk_size_t mutations_stop = index->site_mutations_offset[site + 1];
    tsk_id_t *tmp;
    tsk_id_t u, m, a;
    tsk_size_t i;
    double dead, matched, cost;
    int32_t x0, x1, w, x;
    uint64_t bits, mask;
    int num_segments, k;

    if (best_node == NULL_NODE || preorder[best_node] < 0
        || self->match_cost[site] != 0) {
        goto out;
    }
    if (state != TSLS_MISSING_DATA) {
        num_segments = tsls_hmm_site_segments(self, site);
        if (tsls_hmm_segment_allele(self, num_segments, preorder[best_node]) != state) {
            goto out;
        }
    } else {
        num_segments = 0;
    }
    self->ceiling[site] = ceiling;
    self->ceiling_pse[site] = tsls_hmm_push_ceiling(self, site, ceiling);
    self->max_likelihood_node[site] = best_node;
    self->num_touched = 0;
    dead = INFINITY;
    if (skip_dead && site + 1 < self->end_site && self->switch_cost[site + 1] > 0) {
        dead = best + self->switch_cost[site + 1];
        if (!(dead <= floor)) {
            dead = INFINITY;
        }
    }
    self->dead_cost = dead;
    if (state != TSLS_MISSING_DATA) {
        /* Mutation nodes whose match differs from their parent's */
        for (i = index->site_mutations_offset[site]; i < mutations_stop; i++) {
            m = index->mutation_node[i];
            if (preorder[m] < 0 || L[m].value >= 0) {
                continue;
            }
            /* m is in the tree and has no likelihood, so it isn't a root */
            if ((tsls_hmm_segment_allele(self, num_segments, preorder[m]) == state)
                != (tsls_hmm_segment_allele(
                        self, num_segments, preorder[tree[m].parent])
                    == state)) {
                a = tsls_hmm_find_listed(self, tree[m].parent);
                if (tsls_hmm_deferred_cost(self, L[a].value, L[a].site) >= dead) {
                    /* m's region is dead whether it mismatches or not (see
                     * tsls_hmm_equivalent_costs), so it needs no state */
                    continue;
                }
                tsls_hmm_record_state(self, m, false, L[a].value, L[a].site);
                self->list_position[m] = self->num_likelihood_nodes;
                tsls_hmm_add_likelihood(self, m, L[a].value);
                L[m].site = L[a].site;
                tsls_hmm_set_listed_bit(self, m);
            }
        }
        /* The listed nodes in mismatching segments */
        for (k = 0; k < num_segments; k++) {
            if (self->segment_allele[k] == state) {
                continue;
            }
            x0 = self->segment_start[k];
            x1 = self->segment_start[k + 1];
            for (w = x0 / 64; w <= (x1 - 1) / 64; w++) {
                bits = self->listed_bits[w];
                if (w == x0 / 64) {
                    bits &= ~(uint64_t) 0 << (x0 % 64);
                }
                if (w == (x1 - 1) / 64 && x1 % 64 != 0) {
                    mask = ((uint64_t) 1 << (x1 % 64)) - 1;
                    bits &= mask;
                }
                while (bits != 0) {
                    x = w * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1;
                    u = order[x];
                    matched = tsls_hmm_deferred_cost(self, L[u].value, L[u].site);
                    if (matched >= dead && matched > best) {
                        /* Dead: the matching cost serves (see
                         * tsls_hmm_process_site_viterbi) */
                        continue;
                    }
                    tree[u].recombination_required
                        = L[u].value >= ceiling && self->ceiling_pse[site] <= L[u].site;
                    cost = matched + mismatch_cost;
                    if (cost > prune_limit) {
                        cost = floor;
                    }
                    cost = TSK_MIN(cost, floor);
                    tsls_hmm_record(self, u);
                    L[u].value = cost;
                    L[u].site = site;
                    L[u].current = cost;
                    self->touched[self->num_touched] = u;
                    self->num_touched++;
                }
            }
        }
    }
    self->site_num_updated = (size_t) self->num_touched;
    tsls_hmm_coalesce_nodes(self, site, self->touched, self->num_touched);
    tsls_hmm_coalesce_nodes(self, site, self->prev_touched, self->num_prev_touched);
    tmp = self->prev_touched;
    self->prev_touched = self->touched;
    self->touched = tmp;
    self->num_prev_touched = self->num_touched;
    if (self->num_likelihood_nodes > self->compact_size + self->compact_size / 2 + 16) {
        tsls_hmm_compact_indexed(self, site);
        self->num_prev_touched = 0;
    }
    self->currents_stale = true;
    self->num_sparse_sites++;
    ret = tsls_hmm_store_traceback(self, site);
    if (ret == 0) {
        ret = 1;
    }
out:
    return ret;
}

/* Start a new part of the traceback delta being recorded: each node is
 * recorded at most once per part. */
static inline void
tsls_hmm_next_stamp(tsls_hmm_t *self)
{
    if (self->current_stamp == UINT32_MAX) {
        memset(self->stamp, 0, (self->num_nodes + 1) * sizeof(*self->stamp));
        self->current_stamp = 0;
    }
    self->current_stamp++;
}

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
    const double prune = TSK_MAX(self->prune_threshold, self->relative_prune_threshold);
    /* Values are exact at every site only if they are output or counted,
     * or sparse processing is off */
    const bool skip_dead = output == NULL
                           && !(self->flags & (TSLS_DISABLE_SPARSE | TSLS_COUNT_DISTINCT));
    tsk_id_t site, first_site;
    double previous_total = -1;
    bool valid, new_tree;
    int handled;

    tsk_tree_position_init(&tp, self->tree_sequence, 0);
    self->mode = mode;
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
    /* Viterbi costs start at 0 as of the site before the first, and a node
     * is pruned if its value relative to the best is below prune (see
     * tsls_hmm_process_site_viterbi) */
    self->last_site = first_site - 1;
    self->end_site = end;
    if (mode == MODE_VITERBI) {
        tsls_hmm_compute_costs(self, start, end);
    }
    self->best_cost = 0;
    self->dead_cost = INFINITY;
    self->ceiling_stack_size = 0;
    self->delta_size = 0;
    self->delta_num_transition = 0;
    self->prune_cost = prune > 0 ? -log(prune) : INFINITY;
    self->floor_cost = neg_log(self->likelihood_threshold);
    tsls_hmm_next_stamp(self);
    /* The forward algorithm starts from the uniform distribution over the
     * nodes and the backward from 1. */
    ret = tsls_hmm_init_sweep(self, &tp, position[first_site],
        mode == MODE_FORWARD ? 1.0 / tsls_hmm_get_n(self)
                             : (mode == MODE_VITERBI ? 0.0 : 1.0));
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
        if (mode == MODE_VITERBI) {
            if (skip_dead && new_tree) {
                /* Scan forward rather than binary search: trees often
                 * span only a site or two, so the scans add up to a pass
                 * over the sites. */
                self->tree_end_site = site + 1;
                while (self->tree_end_site < end
                       && position[self->tree_end_site] < tp.interval.right) {
                    self->tree_end_site++;
                }
                if (self->tree_end_site - site >= self->index_min_sites) {
                    ret = tsls_hmm_number_tree(self, site);
                    if (ret != 0) {
                        goto out;
                    }
                }
            }
            new_tree = false;
            /* The changes from here on are the site's own */
            self->delta_num_transition = self->delta_size;
            tsls_hmm_next_stamp(self);
            handled = 0;
            if (self->tree_numbered) {
                handled = tsls_hmm_process_site_indexed(
                    self, site, haplotype[site], skip_dead);
            }
            if (handled < 0) {
                ret = handled;
            } else if (handled == 0) {
                if (self->currents_stale) {
                    tsls_hmm_refresh_currents(self);
                }
                ret = tsls_hmm_process_site_viterbi(
                    self, site, haplotype[site], skip_dead, output);
                if (ret == 0 && self->tree_numbered) {
                    tsls_hmm_set_list_index(self);
                    self->num_prev_touched = 0;
                }
            }
            tsls_hmm_next_stamp(self);
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
        self->last_site = site;
        self->stats_num_sites++;
        self->stats_total_values += (size_t) self->num_likelihood_nodes;
        self->stats_max_values
            = TSK_MAX(self->stats_max_values, (size_t) self->num_likelihood_nodes);
        self->stats_total_updated += self->site_num_updated;
    }
out:
    tsk_tree_position_free(&tp);
    return ret;
}

/* Undo the entries [start, stop) of a site's traceback delta, restoring
 * the nodes' costs and sites, and their membership of the list in the
 * traceback_flag (1 if listed, -1 if not). */
static inline void
tsls_hmm_undo_traceback_entries(tsls_hmm_t *self, const traceback_delta_t *delta,
    int start, int stop, tsls_node_t *restrict tree)
{
    tsls_likelihood_t *restrict L = self->likelihood;
    const traceback_entry_t *restrict entries = delta->entries;
    tsk_id_t u;
    int j;

    for (j = start; j < stop; j++) {
        u = (tsk_id_t) (entries[j].node & ~TSLS_DELTA_LISTED);
        L[u].value = entries[j].cost;
        L[u].site = entries[j].site;
        tree[u].traceback_flag = (entries[j].node & TSLS_DELTA_LISTED) ? 1 : -1;
    }
}

/* Run the traceback over sites [start, end), moving right-to-left through
 * the trees. Only the parent array of the tree is needed. The state starts
 * as the forward pass left it, after the last site, and each site's delta
 * is undone after it is processed, so that the state ends as it was before
 * the first site.
 *
 * The flag at site j of the path's node u, whether the path switched into
 * it at j, is C >= ceiling[j] where C is u's cost after site j - 1 (and the
 * tree transitions before j), as tsls_hmm_process_site_viterbi computes
 * it. u inherits the state of its nearest listed ancestor r after site j:
 * any node merged into r at j had the same flag. r's state before j is
 * restored by undoing the site's own changes, and gives C as the cost it
 * last changed to, at a site s, lowered by any smaller ceiling since: so
 * the flag is C >= ceiling[j] and no ceiling after s and before j is
 * smaller than ceiling[j] (ceiling_pse[j] <= s). */
static int WARN_UNUSED
tsls_hmm_run_traceback(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    allele_t *restrict match, size_t *path_length_out, double *restrict path_left,
    double *restrict path_right, tsk_id_t *restrict path_parent)
{
    int ret = 0;
    tsk_tree_position_t tp;
    tsk_id_t site, j, u, max_likelihood_node;
    double x;
    bool valid, recombine;
    tsls_node_t *restrict tree = self->tree;
    const tsls_likelihood_t *restrict L = self->likelihood;
    const traceback_delta_t *delta;
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
        tree[u].traceback_flag = L[u].value >= 0 ? 1 : -1;
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

        /* Traverse up the tree from the current node. The first node in the
         * list after this site that we meet tells us whether we need to
         * recombine. Every root is listed. */
        u = path_parent[path_length];
        while (tree[u].traceback_flag != 1) {
            u = tree[u].parent;
            assert(u != virtual_root);
        }
        delta = &self->traceback[site];
        tsls_hmm_undo_traceback_entries(
            self, delta, delta->num_transition, delta->size, tree);
        assert(L[u].value >= 0);
        recombine = L[u].value >= self->ceiling[site]
                    && self->ceiling_pse[site] <= L[u].site;
        tsls_hmm_undo_traceback_entries(self, delta, 0, delta->num_transition, tree);
        if (recombine && site > start) {
            max_likelihood_node = self->max_likelihood_node[site - 1];
            assert(max_likelihood_node != NULL_NODE);
            path_left[path_length] = sites_position[site];
            path_length++;
            /* Start the next output edge */
            path_right[path_length] = sites_position[site];
            path_parent[path_length] = max_likelihood_node;
        }
    }
    if (self->flags & TSLS_EXTENDED_CHECKS) {
        /* Undoing every site leaves the state before the first: only the
         * roots of the first tree listed, at cost 0 */
        for (u = 0; u < virtual_root; u++) {
            assert(tree[u].traceback_flag == -1
                   || (L[u].value == 0 && L[u].site == start - 1));
        }
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
    self->total_delta_size = 0;
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
tsls_hmm_get_mean_delta_size(const tsls_hmm_t *self)
{
    if (self->stats_num_sites == 0) {
        return 0;
    }
    return (double) self->total_delta_size / (double) self->stats_num_sites;
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
