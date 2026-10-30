/* Lean C tests. The algorithms are tested in detail against a dense
 * implementation in the Python tests. */

#include "tsls.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <CUnit/Basic.h>

static FILE *_devnull;

/* A forest over [0, 10): roots 3 (above 0 and 1) and 4 (above 2) on [0, 5),
 * joined by node 6 on [5, 10). Node 5 is isolated. If extra_site is true,
 * add a fifth site with no mutations. */
static void
build_forest_ts(tsk_treeseq_t *ts, bool extra_site)
{
    int ret;
    int j;
    double time[] = { 0, 0, 0, 1, 2, 4, 3 };
    tsk_table_collection_t tables;

    ret = tsk_table_collection_init(&tables, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tables.sequence_length = 10;
    for (j = 0; j < 7; j++) {
        tsk_node_table_add_row(&tables.nodes, time[j] == 0 ? TSK_NODE_IS_SAMPLE : 0,
            time[j], TSK_NULL, TSK_NULL, NULL, 0);
    }
    tsk_edge_table_add_row(&tables.edges, 0, 10, 3, 0, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 10, 3, 1, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 0, 10, 4, 2, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 5, 10, 6, 3, NULL, 0);
    tsk_edge_table_add_row(&tables.edges, 5, 10, 6, 4, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 1, "A", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 3, "C", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 6, "G", 1, NULL, 0);
    tsk_site_table_add_row(&tables.sites, 8, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 3, TSK_NULL, TSK_UNKNOWN_TIME, "C", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 0, 2, TSK_NULL, TSK_UNKNOWN_TIME, "G", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 1, 4, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 2, 0, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 3, 6, TSK_NULL, TSK_UNKNOWN_TIME, "T", 1, NULL, 0);
    tsk_mutation_table_add_row(
        &tables.mutations, 3, 1, TSK_NULL, TSK_UNKNOWN_TIME, "A", 1, NULL, 0);
    if (extra_site) {
        tsk_site_table_add_row(&tables.sites, 9, "T", 1, NULL, 0);
    }
    ret = tsk_table_collection_sort(&tables, NULL, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsk_treeseq_init(
        ts, &tables, TSK_TS_INIT_BUILD_INDEXES | TSK_TS_INIT_COMPUTE_MUTATION_PARENTS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    tsk_table_collection_free(&tables);
}

static const char *acgt[] = { "A", "C", "G", "T" };
static double rho[] = { 0.01, 0.01, 0.01, 0.01, 0.01 };
static double mu[] = { 0.01, 0.01, 0.01, 0.01, 0.01 };

static allele_t
acgt_code(char state)
{
    return state == 'A' ? 0 : state == 'C' ? 1 : state == 'G' ? 2 : 3;
}

#define MAX_TEST_SITES 5

/* The same list of alleles at every site, as tsls_index_init takes it */
typedef struct {
    const char *alleles[MAX_TEST_SITES * (TSLS_MAX_ALLELES + 1)];
    tsk_size_t num_alleles[MAX_TEST_SITES];
} site_alleles_t;

static void
set_site_alleles(
    site_alleles_t *self, const char **alleles, tsk_size_t n, tsk_size_t num_sites)
{
    tsk_size_t j, k;

    CU_ASSERT_FATAL(num_sites <= MAX_TEST_SITES);
    CU_ASSERT_FATAL(n <= TSLS_MAX_ALLELES + 1);
    for (j = 0; j < num_sites; j++) {
        self->num_alleles[j] = n;
        for (k = 0; k < n; k++) {
            self->alleles[j * n + k] = alleles[k % 4];
        }
    }
}

static int
index_init_acgt(tsls_index_t *index, const tsk_treeseq_t *ts)
{
    static site_alleles_t site_alleles;

    set_site_alleles(&site_alleles, acgt, 4, tsk_treeseq_get_num_sites(ts));
    return tsls_index_init(index, ts, site_alleles.alleles, site_alleles.num_alleles);
}

static void
test_allele_encoding(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsls_index_t index;
    tsk_size_t j;
    tsk_mutation_t *mutation;
    site_alleles_t site_alleles;
    const char *too_few[] = { "A", "C" };
    const char *duplicate[] = { "A", "C", "G", "T", "A" };
    /* Sites are A -> C, G; C -> A; G -> A; A -> T, A; T. Every site has at
     * least two alleles. */
    tsk_size_t site_num_alleles[] = { 3, 2, 2, 2, 2 };
    /* A different list at each site, including alleles that aren't in the
     * tree sequence, and a site with one allele. */
    const char *per_site[] = { "G", "C", "A", "C", "A", "T", "G", "A", "A", "T", "T" };
    tsk_size_t per_site_num_alleles[] = { 3, 3, 2, 2, 1 };
    allele_t per_site_ancestral[] = { 2, 0, 0, 0, 0 };

    build_forest_ts(&ts, true);

    ret = index_init_acgt(&index, &ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < tsk_treeseq_get_num_sites(&ts); j++) {
        CU_ASSERT_EQUAL(index.num_alleles[j], 4);
        CU_ASSERT_EQUAL(index.ancestral_state[j],
            acgt_code(ts.tables->sites
                    .ancestral_state[ts.tables->sites.ancestral_state_offset[j]]));
    }
    for (j = 0; j < tsk_treeseq_get_num_mutations(&ts); j++) {
        mutation = &ts.site_mutations_mem[j];
        CU_ASSERT_EQUAL(index.mutation_node[mutation->id], mutation->node);
        CU_ASSERT_EQUAL(index.mutation_derived_state[mutation->id],
            acgt_code(mutation->derived_state[0]));
    }
    for (j = 0; j < tsk_treeseq_get_num_sites(&ts); j++) {
        CU_ASSERT_EQUAL(
            index.site_mutations_offset[j + 1] - index.site_mutations_offset[j],
            ts.site_mutations_length[j]);
    }
    tsls_index_free(&index);

    /* Numbered from the tree sequence, as in tskit */
    ret = tsls_index_init(&index, &ts, NULL, NULL);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < tsk_treeseq_get_num_sites(&ts); j++) {
        CU_ASSERT_EQUAL(index.num_alleles[j], site_num_alleles[j]);
        CU_ASSERT_EQUAL(index.ancestral_state[j], 0);
    }
    /* At the last site, the mutation to A is a back mutation */
    for (j = 0; j < ts.site_mutations_length[3]; j++) {
        mutation = &ts.site_mutations[3][j];
        CU_ASSERT_EQUAL(index.mutation_derived_state[mutation->id],
            mutation->derived_state[0] == 'A' ? 0 : 1);
    }
    tsls_index_free(&index);

    /* A different list of alleles at each site */
    ret = tsls_index_init(&index, &ts, per_site, per_site_num_alleles);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    for (j = 0; j < tsk_treeseq_get_num_sites(&ts); j++) {
        CU_ASSERT_EQUAL(index.num_alleles[j], per_site_num_alleles[j]);
        CU_ASSERT_EQUAL(index.ancestral_state[j], per_site_ancestral[j]);
    }
    /* Site 0's mutations are to C (1) and G (0) */
    for (j = 0; j < ts.site_mutations_length[0]; j++) {
        mutation = &ts.site_mutations[0][j];
        CU_ASSERT_EQUAL(index.mutation_derived_state[mutation->id],
            mutation->derived_state[0] == 'C' ? 1 : 0);
    }
    tsls_index_free(&index);

    set_site_alleles(&site_alleles, too_few, 2, tsk_treeseq_get_num_sites(&ts));
    ret = tsls_index_init(&index, &ts, site_alleles.alleles, site_alleles.num_alleles);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_UNKNOWN_ALLELE);
    tsls_index_free(&index);

    set_site_alleles(&site_alleles, duplicate, 5, tsk_treeseq_get_num_sites(&ts));
    ret = tsls_index_init(&index, &ts, site_alleles.alleles, site_alleles.num_alleles);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_DUPLICATE_ALLELE);
    tsls_index_free(&index);

    set_site_alleles(
        &site_alleles, acgt, TSLS_MAX_ALLELES + 1, tsk_treeseq_get_num_sites(&ts));
    ret = tsls_index_init(&index, &ts, site_alleles.alleles, site_alleles.num_alleles);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_TOO_MANY_ALLELES);
    tsls_index_free(&index);

    tsk_treeseq_free(&ts);
}

static void
test_forest_matching(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsls_index_t index;
    tsls_hmm_t hmm;
    compressed_matrix_t fm, bm;
    /* The haplotype of node 1: C (under 3), C (ancestral), G, A (back mutation) */
    allele_t h[] = { 1, 1, 2, 0 };
    allele_t match[4];
    double left[4], right[4];
    tsk_id_t parent[4];
    size_t path_length, j, k, l;
    double total;

    build_forest_ts(&ts, false);
    ret = index_init_acgt(&index, &ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_alloc(&hmm, &index, rho, mu, DBL_MIN, TSLS_EXTENDED_CHECKS);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsls_hmm_find_path(&hmm, 0, 4, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(parent[0], 1);
    CU_ASSERT_EQUAL(left[0], 0);
    CU_ASSERT_EQUAL(right[0], 10);
    for (j = 0; j < 4; j++) {
        CU_ASSERT_EQUAL(match[j], h[j]);
    }
    CU_ASSERT(tsls_hmm_get_mean_traceback_size(&hmm) > 0);
    CU_ASSERT(tsls_hmm_get_total_memory(&hmm) > 0);
    tsls_hmm_print_state(&hmm, _devnull);

    /* A window within the right tree, which is found by seeking backwards */
    ret = tsls_hmm_find_path(&hmm, 2, 4, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(parent[0], 1);
    CU_ASSERT_EQUAL(left[0], 6);
    CU_ASSERT_EQUAL(right[0], 10);

    ret = tsls_hmm_find_path(&hmm, 2, 1, h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_SITE_RANGE);

    /* Viterbi with the values recorded */
    ret = compressed_matrix_alloc(&fm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_find_path_matrix(
        &hmm, 0, 4, h, match, &path_length, left, right, parent, &fm);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT_EQUAL(path_length, 1);
    CU_ASSERT_EQUAL(parent[0], 1);
    for (j = 0; j < 4; j++) {
        CU_ASSERT(fm.num_values[j] > 0);
        CU_ASSERT(fm.flags[j] != NULL);
        CU_ASSERT(fm.normalisation_factor[j] > 0);
    }
    compressed_matrix_free(&fm);

    ret = compressed_matrix_alloc(&fm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = compressed_matrix_alloc(&bm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_forward(&hmm, h, &fm);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_backward(&hmm, h, fm.normalisation_factor, &bm);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    CU_ASSERT(compressed_matrix_get_total_values(&fm) > 0);

    /* The Python tests check the values against a dense implementation;
     * here we just check that the outputs are populated. */
    for (j = 0; j < 4; j++) {
        CU_ASSERT(fm.normalisation_factor[j] > 0);
        total = 0;
        for (k = 0; k < fm.num_values[j]; k++) {
            for (l = 0; l < bm.num_values[j]; l++) {
                if (fm.nodes[j][k] == bm.nodes[j][l]) {
                    total += fm.values[j][k] * bm.values[j][l];
                }
            }
        }
        CU_ASSERT(total > 0);
    }
    compressed_matrix_free(&fm);
    compressed_matrix_free(&bm);

    tsls_hmm_free(&hmm);
    tsls_index_free(&index);
    tsk_treeseq_free(&ts);
}

static void
test_bad_inputs(void)
{
    int ret;
    tsk_treeseq_t ts;
    tsls_index_t index;
    tsls_hmm_t hmm;
    compressed_matrix_t cm;
    double zero_mu[] = { 0, 0, 0, 0 };
    double norm[] = { 1, 1, 0, 1 };
    allele_t h[] = { 1, 1, 2, 0 };
    allele_t bad_h[] = { 1, 4, 2, 0 };
    allele_t impossible_h[] = { 3, 3, 3, 3 };
    allele_t match[4];
    double left[4], right[4];
    tsk_id_t parent[4];
    size_t path_length;

    build_forest_ts(&ts, false);
    ret = index_init_acgt(&index, &ts);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_alloc(&hmm, &index, rho, zero_mu, DBL_MIN, 0);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    ret = tsls_hmm_find_path(
        &hmm, 0, 4, bad_h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_HAPLOTYPE_ALLELE);
    ret = tsls_hmm_find_path(
        &hmm, 0, 4, impossible_h, match, &path_length, left, right, parent);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA);

    ret = compressed_matrix_alloc(&cm, 4, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_forward(&hmm, impossible_h, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_MATCH_IMPOSSIBLE);
    ret = tsls_hmm_backward(&hmm, h, norm, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_NORMALISATION_FACTOR);
    compressed_matrix_free(&cm);

    ret = compressed_matrix_alloc(&cm, 3, 1024);
    CU_ASSERT_EQUAL_FATAL(ret, 0);
    ret = tsls_hmm_forward(&hmm, h, &cm);
    CU_ASSERT_EQUAL(ret, TSLS_ERR_BAD_SITE_RANGE);
    compressed_matrix_free(&cm);

    tsls_hmm_free(&hmm);
    tsls_index_free(&index);
    tsk_treeseq_free(&ts);
}

static void
test_strerror(void)
{
    int j;
    const char *msg;

    for (j = 0; j < 100; j++) {
        msg = tsls_strerror(-j);
        CU_ASSERT_FATAL(msg != NULL);
        CU_ASSERT(strlen(msg) > 0);
    }
}

static int
tsls_suite_init(void)
{
    _devnull = fopen("/dev/null", "w");
    if (_devnull == NULL) {
        return CUE_SINIT_FAILED;
    }
    return CUE_SUCCESS;
}

static int
tsls_suite_cleanup(void)
{
    if (_devnull != NULL) {
        fclose(_devnull);
    }
    return CUE_SUCCESS;
}

static void
handle_cunit_error(void)
{
    fprintf(stderr, "CUnit error occured: %d: %s\n", CU_get_error(), CU_get_error_msg());
    exit(EXIT_FAILURE);
}

int
main(int argc, char **argv)
{
    int ret;
    CU_pTest test;
    CU_pSuite suite;
    CU_TestInfo tests[] = {
        { "test_allele_encoding", test_allele_encoding },
        { "test_forest_matching", test_forest_matching },
        { "test_bad_inputs", test_bad_inputs },
        { "test_strerror", test_strerror },
        { NULL, NULL },
    };

    if (CUE_SUCCESS != CU_initialize_registry()) {
        handle_cunit_error();
    }
    suite = CU_add_suite("tsls", tsls_suite_init, tsls_suite_cleanup);
    if (NULL == suite) {
        handle_cunit_error();
    }
    for (ret = 0; tests[ret].pName != NULL; ret++) {
        if (NULL == CU_add_test(suite, tests[ret].pName, tests[ret].pTestFunc)) {
            handle_cunit_error();
        }
    }
    CU_basic_set_mode(CU_BRM_VERBOSE);
    if (argc == 1) {
        CU_basic_run_tests();
    } else {
        test = CU_get_test_by_name(argv[1], suite);
        if (test == NULL) {
            printf("Test '%s' not found\n", argv[1]);
        } else {
            CU_basic_run_test(suite, test);
        }
    }
    ret = CU_get_number_of_tests_failed() != 0;
    CU_cleanup_registry();
    return ret;
}
