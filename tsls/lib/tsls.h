#ifndef __TSLS_H__
#define __TSLS_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* No atomics on Windows for now */
#if defined(_WIN32)
#define TSI_NO_ATOMICS 1
#endif

#if !defined(TSI_NO_ATOMICS)
#include <stdatomic.h>
#define TSI_ATOMIC(T) _Atomic T
#else
#define TSI_ATOMIC(T) T
#endif

#include "tskit.h"
#include "err.h"

#define TSLS_MISSING_DATA (-1)

/* Special values in the likelihood array. A node with NULL_LIKELIHOOD
 * inherits the value of its nearest ancestor with a likelihood, and a node
 * that is NOT_IN_TREE is not in the current tree (it has no parent and no
 * children), and so is not a state of the HMM. */
#define NULL_LIKELIHOOD (-1)
#define NOT_IN_TREE     (-2)

#define NULL_NODE (-1)

/* Matcher flags */
#define TSLS_EXTENDED_CHECKS     (1 << 0)
#define TSLS_COUNT_DISTINCT      (1 << 2)
/* Process every Viterbi site in full (see tsls_hmm_process_site_sparse) */
#define TSLS_DISABLE_SPARSE      (1 << 3)

typedef int8_t allele_t;

/* The maximum number of distinct alleles at a site */
#define TSLS_MAX_ALLELES INT8_MAX

/* Allow the CPython accessor Hmm.total_memory to read the
 * current allocator usage accurately while other threads may be allocating.
 * Only total_size is atomic so it can be observed safely from the getter
 * without data races. */
typedef struct {
    size_t chunk_size; /* number of bytes per chunk */
    size_t top;        /* the offset of the next available byte in the current chunk */
    size_t current_chunk;          /* the index of the chunk currently being used */
    TSI_ATOMIC(size_t) total_size; /* the total number of bytes allocated + overhead. */
    size_t total_allocated;        /* the total number of bytes allocated. */
    size_t num_chunks;             /* the number of memory chunks. */
    char **mem_chunks;             /* the memory chunks */
} tsi_blkalloc_t;

extern void tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out);
extern int tsi_blkalloc_reset(tsi_blkalloc_t *self);
extern int tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size);
extern void *tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size);
extern void tsi_blkalloc_free(tsi_blkalloc_t *self);

/* A node's Viterbi state before a change (see traceback_delta_t): its
 * cost and the site of its last change, and whether it was listed (the
 * TSLS_DELTA_LISTED bit of node). */
typedef struct {
    uint32_t node;
    int32_t site;
    double cost;
} traceback_entry_t;

#define TSLS_DELTA_LISTED (1u << 31)

/* The changes to the Viterbi state at a site, so that the traceback can
 * undo them moving right to left. Viterbi stores each listed node's cost
 * and the site at which it last changed; nodes inherit the state of their
 * nearest listed ancestor. The first num_transition entries hold the state
 * after the previous site of the nodes changed by the tree transitions
 * before this site, and the rest the state after the transitions of the
 * nodes changed by the site's update, each node at most once in each part.
 * See tsls_hmm_run_traceback. */
typedef struct {
    int32_t num_transition;
    int32_t size;
    traceback_entry_t *entries;
} traceback_delta_t;

/* The compressed output of the forward and backward algorithms. At each
 * site we store a list of (node, value) pairs: the value for a node in the
 * tree is the value of its nearest ancestor (including itself) in this list.
 */
typedef struct {
    size_t num_sites;
    double *normalisation_factor;
    tsk_size_t *num_values;
    tsk_id_t **nodes;
    double **values;
    /* For Viterbi, the recombination_required flag of each node */
    int8_t **flags;
    tsi_blkalloc_t allocator;
} compressed_matrix_t;

/* An edge in one of tskit's edge orders, with the one coordinate that
 * finding the first tree needs: right in insertion order (in which the
 * edges before a position all have left <= it), and left in removal
 * order. */
typedef struct {
    tsk_id_t parent;
    tsk_id_t child;
    double coordinate;
} tsls_edge_t;

/* Read-only data derived from a tree sequence for the HMM, which is built
 * once and can be shared between HMMs (including in different threads).
 * The tree sequence is borrowed, and must not be changed or freed while the
 * index is in use. */
typedef struct {
    const tsk_treeseq_t *tree_sequence;
    size_t num_nodes;
    size_t num_sites;
    size_t num_edges;
    const double *sites_position;
    double sequence_length;
    /* The integer encoding of the alleles: the number of alleles and the
     * ancestral state of each site */
    tsk_size_t *num_alleles;
    allele_t *ancestral_state;
    /* The mutations at site j, in tskit order, are entries
     * site_mutations_offset[j] to site_mutations_offset[j + 1] of the
     * mutation_node and mutation_derived_state arrays */
    tsk_size_t *site_mutations_offset;
    tsk_id_t *mutation_node;
    allele_t *mutation_derived_state;
    /* Entry j is edge edge_insertion_order[j] (edge_removal_order[j]) of
     * the edge table, so that the ranges of a tsk_tree_position_t index
     * these arrays directly */
    tsls_edge_t *insertion_edges;
    tsls_edge_t *removal_edges;
} tsls_index_t;

/* The state of a node that the walks up the tree read, kept together (in
 * 16 bytes, so that records don't straddle cache lines) so that each step
 * of a walk reads one record. */
typedef struct {
    tsk_id_t parent;
    /* The allelic state at the current site, if the node has a mutation
     * there (or is a root, during the sweeps) */
    allele_t allelic_state;
    int8_t recombination_required;
    int8_t unused;
    /* The node's flag in the traceback as of the last site stored, or -1
     * if it is not in the traceback's set */
    int8_t traceback_flag;
    /* The walks up the tree to the nearest node with a likelihood are
     * cached (see tsls_hmm_find_listed). A node with a likelihood has
     * cache_stamp TSLS_LISTED_STAMP, the largest value. An unlisted node
     * whose cache_stamp equals the HMM's has a listed_ancestor that leads,
     * by following listed_ancestor through unlisted nodes, to its nearest
     * ancestor with a likelihood. Stale stamps are smaller than the HMM's,
     * so a walk stops at the first node with cache_stamp >= the HMM's, and
     * incrementing the HMM's stamp clears every cache. */
    uint32_t cache_stamp;
    tsk_id_t listed_ancestor;
} tsls_node_t;

/* The cache_stamp of a node with a likelihood (see tsls_node_t) */
#define TSLS_LISTED_STAMP UINT32_MAX

/* A node's likelihood. */
typedef struct {
    /* The node's likelihood, or NULL_LIKELIHOOD or NOT_IN_TREE */
    double value;
    /* For a node with a likelihood, in Viterbi, its nearest proper ancestor
     * with a likelihood (NULL_NODE for a root), set at the start of each
     * site (see tsls_hmm_set_listed_ancestors) */
    tsk_id_t ancestor;
    /* In Viterbi, value is the cost the node last changed to, at site:
     * its cost now is that, lowered by any smaller ceiling since (see
     * tsls_hmm_t), which is cached in current. */
    int32_t site;
    double current;
} tsls_likelihood_t;

/* The Li and Stephens HMM on a tree sequence. The index is borrowed, and
 * must not be changed or freed while the HMM is in use. */
typedef struct {
    int flags;
    const tsls_index_t *index;
    const tsk_treeseq_t *tree_sequence;
    size_t num_nodes;
    size_t num_sites;
    const double *sites_position;
    double sequence_length;
    /* Input LS model rates */
    double likelihood_threshold;
    /* After normalisation at each site, nodes with values below this times
     * the maximum are given the value likelihood_threshold, in all three
     * algorithms (0: off; see tsls_hmm_set_relative_prune_threshold) */
    double relative_prune_threshold;
    /* In Viterbi, nodes with normalised values below this are given the
     * value likelihood_threshold (see tsls_hmm_set_prune_threshold) */
    double prune_threshold;
    double *recombination_rate;
    double *mismatch_rate;
    /* Viterbi works with costs, minus the logs of the odds (see
     * docs/likelihood_model.md): per site, the cost of switching relative
     * to staying, and the costs of the two emissions, less the smaller of
     * them (emission_base), so that both are at least 0. stay_cost is minus
     * the log of the staying probability; with emission_base it gives the
     * normalisation factors of viterbi_matrix. */
    double *switch_cost;
    double *match_cost;
    double *mismatch_cost;
    double *emission_base;
    double *stay_cost;
    /* The algorithm running (MODE_VITERBI etc. in hmm.c) */
    int mode;
    /* The per-node state, including the tree's parent links */
    tsls_node_t *tree;
    tsls_likelihood_t *likelihood;
    /* The walk caches are valid on nodes whose cache_stamp equals this;
     * incrementing it clears them all (see tsls_hmm_clear_walk_caches) */
    uint32_t cache_stamp;
    /* The number of children of each node in the current tree */
    int32_t *num_children;
    /* The number of nodes in the subtree below each node, including itself.
     * Only maintained for the forward and backward algorithms. */
    int64_t *num_below;
    bool track_num_below;
    /* The number of nodes in the tree that inherit each likelihood value. */
    int64_t *num_inheriting;
    int num_likelihood_nodes;
    /* At each site, record a node with the maximum likelihood. */
    tsk_id_t *max_likelihood_node;
    tsk_id_t *likelihood_nodes_tmp;
    tsk_id_t *likelihood_nodes;
    traceback_delta_t *traceback;
    /* Scratch buffer for the recombination flags of the list at a site */
    int8_t *traceback_flags_tmp;
    /* Viterbi state. best_cost is the minimum cost after the last site
     * processed, last_site. Per site, the ceiling is the cost of switching
     * into the site (best_cost + switch_cost), and ceiling_pse the last
     * earlier site with a strictly smaller ceiling (-1 if none); a stack of
     * the strictly increasing ceilings back from the last site gives
     * these (see tsls_hmm_push_ceiling). */
    double best_cost;
    tsk_id_t last_site;
    /* The end of the window of sites being processed */
    tsk_id_t end_site;
    /* The cost above the best at which a node is pruned, and the cost of
     * likelihood_threshold above the best (infinite if they are 0) */
    double prune_cost;
    double floor_cost;
    double *ceiling;
    int32_t *ceiling_pse;
    int32_t *ceiling_stack_site;
    double *ceiling_stack_value;
    int ceiling_stack_size;
    /* The changes at the current site, recorded for the traceback delta
     * (see traceback_delta_t), with a per-node stamp so that each node is
     * recorded at most once in each part. */
    traceback_entry_t *delta_buffer;
    int delta_size;
    int delta_num_transition;
    uint32_t *stamp;
    uint32_t current_stamp;
    /* A tree is numbered for sparse sites if at least this many of its
     * remaining sites could be sparse (unused for now) */
    int index_min_sites;
    size_t num_sparse_sites;
    /* Statistics of the last run: over the sites processed, the total and
     * maximum length of the likelihood list after each site, and the total
     * number of nodes whose values were computed (site_num_updated, set by
     * each site's processing) */
    size_t stats_num_sites;
    size_t stats_total_values;
    size_t stats_max_values;
    size_t stats_total_updated;
    size_t site_num_updated;
    tsi_blkalloc_t traceback_allocator;
    size_t total_traceback_size;
    /* The total number of entries in the traceback deltas */
    size_t total_delta_size;
    /* The total number of distinct likelihood values over sites, if
     * TSLS_COUNT_DISTINCT is set. */
    size_t total_distinct_values;
    /* Scratch buffer for values */
    double *values_tmp;
    size_t traceback_block_size;
    size_t traceback_realloc_size;
} tsls_hmm_t;

int compressed_matrix_alloc(
    compressed_matrix_t *self, size_t num_sites, size_t block_size);
int compressed_matrix_free(compressed_matrix_t *self);
size_t compressed_matrix_get_total_values(const compressed_matrix_t *self);

int tsls_index_init(tsls_index_t *self, const tsk_treeseq_t *tree_sequence,
    const char **alleles, tsk_size_t num_alleles);
int tsls_index_free(tsls_index_t *self);

int tsls_hmm_alloc(tsls_hmm_t *self, const tsls_index_t *index,
    const double *recombination_rate, const double *mismatch_rate,
    double likelihood_threshold, int flags);
int tsls_hmm_set_relative_prune_threshold(tsls_hmm_t *self, double threshold);
int tsls_hmm_set_prune_threshold(tsls_hmm_t *self, double threshold);
int tsls_hmm_set_index_min_sites(tsls_hmm_t *self, int min_sites);
int tsls_hmm_free(tsls_hmm_t *self);
double tsls_hmm_get_mean_num_values(const tsls_hmm_t *self);
double tsls_hmm_get_mean_num_updated(const tsls_hmm_t *self);
int tsls_hmm_find_path(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    double *path_left, double *path_right, tsk_id_t *path_parent);
int tsls_hmm_find_path_matrix(tsls_hmm_t *self, tsk_id_t start, tsk_id_t end,
    const allele_t *haplotype, allele_t *matched_haplotype, size_t *path_length,
    double *path_left, double *path_right, tsk_id_t *path_parent,
    compressed_matrix_t *output);
int tsls_hmm_forward(
    tsls_hmm_t *self, const allele_t *haplotype, compressed_matrix_t *output);
int tsls_hmm_backward(tsls_hmm_t *self, const allele_t *haplotype,
    const double *forward_normalisation_factor, compressed_matrix_t *output);
int tsls_hmm_print_state(tsls_hmm_t *self, FILE *out);
double tsls_hmm_get_mean_traceback_size(tsls_hmm_t *self);
double tsls_hmm_get_mean_delta_size(const tsls_hmm_t *self);
double tsls_hmm_get_mean_num_distinct_values(tsls_hmm_t *self);
size_t tsls_hmm_get_total_memory(tsls_hmm_t *self);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)

#endif /* __TSLS_H__ */
