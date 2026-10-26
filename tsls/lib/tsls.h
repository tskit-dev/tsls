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

#define NULL_NODE   (-1)
#define CACHE_UNSET (-1)

/* Matcher flags */
#define TSLS_EXTENDED_CHECKS     (1 << 0)
#define TSLS_DISABLE_WEIGHT_BY_N (1 << 1)
#define TSLS_COUNT_DISTINCT      (1 << 2)

/* Experimental pruning modes. After normalisation at each site, pruned
 * nodes are given the value likelihood_threshold. */
#define TSLS_PRUNE_NONE     0
/* Prune nodes with likelihood < parameter * max likelihood */
#define TSLS_PRUNE_RELATIVE 1
/* Prune all but the parameter nodes with the largest likelihoods */
#define TSLS_PRUNE_TOP_N    2

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

/* The changes to the Viterbi traceback at a site: the traceback at a site
 * is a set of (node, recombination_required) pairs, and a node in the tree
 * takes the flag of its nearest ancestor in the set. We store, for each
 * node whose flag changed (or that entered or left the set) at this site,
 * its flag at the previous site (-1 if it was not in the set), so that the
 * traceback can undo the changes moving right to left. */
typedef struct {
    int32_t size;
    tsk_id_t *node;
    int8_t *previous_flag;
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
 * 8 bytes) so that each step of a walk reads one record. */
typedef struct {
    tsk_id_t parent;
    /* The allelic state at the current site, if the node has a mutation
     * there (or is a root, during the sweeps) */
    allele_t allelic_state;
    int8_t recombination_required;
    /* The recombination_required flag inherited by the node, valid where
     * the likelihood cache is set. Only filled when coalescing Viterbi. */
    int8_t recombination_required_cache;
    /* The node's flag in the traceback as of the last site stored, or -1
     * if it is not in the traceback's set */
    int8_t traceback_flag;
} tsls_node_t;

/* A node's likelihood, with the cache used when coalescing. */
typedef struct {
    /* The node's likelihood, or NULL_LIKELIHOOD or NOT_IN_TREE */
    double value;
    /* The likelihood the node inherits, cached during coalescing */
    double cache;
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
    int pruning_mode;
    double pruning_parameter;
    double *recombination_rate;
    double *mismatch_rate;
    /* The per-node state, with the parent links of the quintuply linked
     * tree, which are completed by the arrays below */
    tsls_node_t *tree;
    tsls_likelihood_t *likelihood;
    tsk_id_t *left_child;
    tsk_id_t *right_child;
    tsk_id_t *left_sib;
    tsk_id_t *right_sib;
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
    /* The nodes in the traceback's set as of the last site stored */
    tsk_id_t *traceback_nodes;
    int num_traceback_nodes;
    /* Scratch buffer for the previous flags of a site's traceback delta */
    int8_t *traceback_flags_tmp;
    tsi_blkalloc_t traceback_allocator;
    size_t total_traceback_size;
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
int tsls_hmm_set_pruning(tsls_hmm_t *self, int mode, double parameter);
int tsls_hmm_free(tsls_hmm_t *self);
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
double tsls_hmm_get_mean_num_distinct_values(tsls_hmm_t *self);
size_t tsls_hmm_get_total_memory(tsls_hmm_t *self);

#define tsi_safe_free(pointer)                                                          \
    do {                                                                                \
        if (pointer != NULL) {                                                          \
            free(pointer);                                                              \
        }                                                                               \
    } while (0)

#endif /* __TSLS_H__ */
