
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "lib/tsls.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define MODULE_DOC "Low-level tsls HMM interface."

static PyObject *TslsLibraryError;
static PyObject *TslsMatchImpossible;

#include "tskit_lwt_interface.h"

typedef struct {
    PyObject_HEAD
    tsk_treeseq_t *tree_sequence;
    tsls_index_t *index;
} TreeSequence;

typedef struct {
    PyObject_HEAD
    tsls_hmm_t *hmm;
    TreeSequence *tree_sequence;
} Hmm;

static void
handle_library_error(int err)
{
    if (err == TSLS_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else if (err == TSLS_ERR_MATCH_IMPOSSIBLE
               || err == TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA
               || err == TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION) {
        PyErr_Format(TslsMatchImpossible, "%s", tsls_strerror(err));
    } else {
        PyErr_Format(TslsLibraryError, "%s", tsls_strerror(err));
    }
}

/*===================================================================
 * TreeSequence
 *===================================================================
 */

/* A minimal tree sequence type, built from a LightweightTableCollection,
 * with the index the HMM needs (for the specified alleles), which can be
 * shared between HMMs (including in different threads). */

static int
TreeSequence_check_state(TreeSequence *self)
{
    int ret = 0;
    if (self->tree_sequence == NULL || self->index == NULL) {
        PyErr_SetString(PyExc_SystemError, "TreeSequence not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequence_dealloc(TreeSequence *self)
{
    if (self->index != NULL) {
        tsls_index_free(self->index);
        PyMem_Free(self->index);
        self->index = NULL;
    }
    if (self->tree_sequence != NULL) {
        tsk_treeseq_free(self->tree_sequence);
        PyMem_Free(self->tree_sequence);
        self->tree_sequence = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
TreeSequence_init(TreeSequence *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    LightweightTableCollection *tables = NULL;
    PyObject *alleles = Py_None;
    PyObject *alleles_seq = NULL;
    const char **alleles_data = NULL;
    tsk_size_t num_alleles = 0;
    Py_ssize_t j;
    static char *kwlist[] = { "tables", "alleles", NULL };

    self->tree_sequence = NULL;
    self->index = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", kwlist,
            &LightweightTableCollectionType, &tables, &alleles)) {
        goto out;
    }
    if (alleles != Py_None) {
        alleles_seq = PySequence_Fast(alleles, "alleles must be a sequence");
        if (alleles_seq == NULL) {
            goto out;
        }
        num_alleles = (tsk_size_t) PySequence_Fast_GET_SIZE(alleles_seq);
        alleles_data = PyMem_Calloc(num_alleles + 1, sizeof(*alleles_data));
        if (alleles_data == NULL) {
            PyErr_NoMemory();
            goto out;
        }
        for (j = 0; j < (Py_ssize_t) num_alleles; j++) {
            /* The UTF-8 buffer lives as long as the string */
            alleles_data[j]
                = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(alleles_seq, j));
            if (alleles_data[j] == NULL) {
                goto out;
            }
        }
    }
    if (LightweightTableCollection_check_state(tables) != 0) {
        goto out;
    }
    self->tree_sequence = PyMem_Calloc(1, sizeof(*self->tree_sequence));
    if (self->tree_sequence == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_treeseq_init(
        self->tree_sequence, tables->tables, TSK_TS_INIT_BUILD_INDEXES);
    if (err != 0) {
        handle_tskit_error(err);
        goto out;
    }
    self->index = PyMem_Calloc(1, sizeof(*self->index));
    if (self->index == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsls_index_init(self->index, self->tree_sequence, alleles_data, num_alleles);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    PyMem_Free(alleles_data);
    Py_XDECREF(alleles_seq);
    return ret;
}

static PyObject *
TreeSequence_get_num_nodes(TreeSequence *self, void *closure)
{
    if (TreeSequence_check_state(self) != 0) {
        return NULL;
    }
    return Py_BuildValue(
        "n", (Py_ssize_t) tsk_treeseq_get_num_nodes(self->tree_sequence));
}

static PyObject *
TreeSequence_get_num_sites(TreeSequence *self, void *closure)
{
    if (TreeSequence_check_state(self) != 0) {
        return NULL;
    }
    return Py_BuildValue(
        "n", (Py_ssize_t) tsk_treeseq_get_num_sites(self->tree_sequence));
}

static PyGetSetDef TreeSequence_getsetters[] = {
    { "num_nodes", (getter) TreeSequence_get_num_nodes, NULL, "The number of nodes" },
    { "num_sites", (getter) TreeSequence_get_num_sites, NULL, "The number of sites" },
    { NULL } /* Sentinel */
};

static PyTypeObject TreeSequenceType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tsls.TreeSequence",
    .tp_basicsize = sizeof(TreeSequence),
    .tp_dealloc = (destructor) TreeSequence_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "TreeSequence objects",
    .tp_getset = TreeSequence_getsetters,
    .tp_init = (initproc) TreeSequence_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Hmm
 *===================================================================
 */

static int
Hmm_check_state(Hmm *self)
{
    int ret = 0;
    if (self->hmm == NULL) {
        PyErr_SetString(PyExc_SystemError, "Hmm not initialised");
        ret = -1;
    }
    return ret;
}

static void
Hmm_dealloc(Hmm *self)
{
    if (self->hmm != NULL) {
        tsls_hmm_free(self->hmm);
        PyMem_Free(self->hmm);
        self->hmm = NULL;
    }
    Py_XDECREF(self->tree_sequence);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
Hmm_init(Hmm *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    int extended_checks = 0;
    int pruning_mode = TSLS_PRUNE_NONE;
    double pruning_parameter = 0;
    int count_distinct = 0;
    int sparse = 1;
    static char *kwlist[] = { "tree_sequence", "recombination", "mismatch",
        "extended_checks", "likelihood_threshold", "prune_threshold", "pruning_mode",
        "pruning_parameter", "count_distinct", "sparse", NULL };
    TreeSequence *tree_sequence = NULL;
    PyObject *recombination = NULL;
    PyObject *mismatch = NULL;
    PyArrayObject *recombination_array = NULL;
    PyArrayObject *mismatch_array = NULL;
    npy_intp *shape;
    double likelihood_threshold = DBL_MIN;
    double prune_threshold = 0;
    int flags = 0;
    size_t num_sites;

    self->hmm = NULL;
    self->tree_sequence = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|iddidii", kwlist,
            &TreeSequenceType, &tree_sequence, &recombination, &mismatch,
            &extended_checks, &likelihood_threshold, &prune_threshold, &pruning_mode,
            &pruning_parameter, &count_distinct, &sparse)) {
        goto out;
    }
    self->tree_sequence = tree_sequence;
    Py_INCREF(self->tree_sequence);
    if (TreeSequence_check_state(self->tree_sequence) != 0) {
        goto out;
    }
    num_sites = tsk_treeseq_get_num_sites(tree_sequence->tree_sequence);

    recombination_array = (PyArrayObject *) PyArray_FromAny(recombination,
        PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (recombination_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_array);
    if (shape[0] != (npy_intp) num_sites) {
        PyErr_SetString(
            PyExc_ValueError, "Size of recombination array must be num_sites");
        goto out;
    }
    mismatch_array = (PyArrayObject *) PyArray_FromAny(
        mismatch, PyArray_DescrFromType(NPY_FLOAT64), 1, 1, NPY_ARRAY_IN_ARRAY, NULL);
    if (mismatch_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mismatch_array);
    if (shape[0] != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "Size of mismatch array must be num_sites");
        goto out;
    }

    self->hmm = PyMem_Calloc(1, sizeof(tsls_hmm_t));
    if (self->hmm == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    if (extended_checks) {
        flags |= TSLS_EXTENDED_CHECKS;
    }
    if (count_distinct) {
        flags |= TSLS_COUNT_DISTINCT;
    }
    if (!sparse) {
        flags |= TSLS_DISABLE_SPARSE;
    }
    err = tsls_hmm_alloc(self->hmm, self->tree_sequence->index,
        PyArray_DATA(recombination_array), PyArray_DATA(mismatch_array),
        likelihood_threshold, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    err = tsls_hmm_set_pruning(self->hmm, pruning_mode, pruning_parameter);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    err = tsls_hmm_set_prune_threshold(self->hmm, prune_threshold);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(recombination_array);
    Py_XDECREF(mismatch_array);
    return ret;
}

static PyObject *
Hmm_find_path(Hmm *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", "start", "end", NULL };
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape;
    size_t num_edges;
    int start, end;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *match = NULL;
    npy_intp dims[1];

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Oii", kwlist, &haplotype, &start, &end)) {
        goto out;
    }
    haplotype_array
        = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(haplotype_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != (npy_intp) self->hmm->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }

    dims[0] = self->hmm->num_sites;
    left = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    right = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    match = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT8);
    if (left == NULL || right == NULL || parent == NULL || match == NULL) {
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tsls_hmm_find_path(self->hmm, (tsk_id_t) start,
        (tsk_id_t) end, (allele_t *) PyArray_DATA(haplotype_array),
        (allele_t *) PyArray_DATA(match), &num_edges, PyArray_DATA(left),
        PyArray_DATA(right), PyArray_DATA(parent));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue(
        "(kNNNN)", (unsigned long) num_edges, left, right, parent, match);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
    match = NULL;
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(match);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    return ret;
}

/* Return the compressed matrix as a tuple of flat arrays
 * (normalisation_factor, offsets, nodes, values[, flags]), where the values
 * at site j are values[offsets[j]: offsets[j + 1]] and similarly for nodes
 * and flags. */
static PyObject *
compressed_matrix_to_tuple(const compressed_matrix_t *cm, bool with_flags)
{
    PyObject *ret = NULL;
    PyArrayObject *norm = NULL;
    PyArrayObject *offsets = NULL;
    PyArrayObject *nodes = NULL;
    PyArrayObject *values = NULL;
    PyArrayObject *flags = NULL;
    const size_t num_sites = cm->num_sites;
    npy_intp dims[1];
    uint64_t *offsets_data;
    tsk_id_t *nodes_data;
    double *values_data;
    int8_t *flags_data = NULL;
    size_t j, k, offset;

    dims[0] = (npy_intp) num_sites;
    norm = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    dims[0] = (npy_intp) num_sites + 1;
    offsets = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT64);
    dims[0] = (npy_intp) compressed_matrix_get_total_values(cm);
    nodes = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    values = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    flags = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT8);
    if (norm == NULL || offsets == NULL || nodes == NULL || values == NULL
        || flags == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(norm), cm->normalisation_factor, num_sites * sizeof(double));
    offsets_data = PyArray_DATA(offsets);
    nodes_data = PyArray_DATA(nodes);
    values_data = PyArray_DATA(values);
    flags_data = PyArray_DATA(flags);
    offset = 0;
    for (j = 0; j < num_sites; j++) {
        offsets_data[j] = offset;
        for (k = 0; k < cm->num_values[j]; k++) {
            nodes_data[offset] = cm->nodes[j][k];
            values_data[offset] = cm->values[j][k];
            flags_data[offset] = cm->flags[j] == NULL ? -1 : cm->flags[j][k];
            offset++;
        }
    }
    offsets_data[num_sites] = offset;
    if (with_flags) {
        ret = Py_BuildValue("(NNNNN)", norm, offsets, nodes, values, flags);
        flags = NULL;
    } else {
        ret = Py_BuildValue("(NNNN)", norm, offsets, nodes, values);
    }
    if (ret == NULL) {
        goto out;
    }
    norm = NULL;
    offsets = NULL;
    nodes = NULL;
    values = NULL;
out:
    Py_XDECREF(norm);
    Py_XDECREF(offsets);
    Py_XDECREF(nodes);
    Py_XDECREF(values);
    Py_XDECREF(flags);
    return ret;
}

/* Run the forward (if forward_norm is NULL) or backward algorithm, and
 * return the compressed matrix as a tuple of flat arrays
 * (normalisation_factor, offsets, nodes, values), where the values at site
 * j are nodes[offsets[j]: offsets[j + 1]] and similarly for values. */
static PyObject *
Hmm_run_compressed_matrix(
    Hmm *self, PyObject *haplotype, PyObject *forward_norm)
{
    int err;
    PyObject *ret = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyArrayObject *forward_norm_array = NULL;
    const size_t num_sites = self->hmm->num_sites;
    compressed_matrix_t cm;
    bool cm_allocated = false;

    haplotype_array
        = (PyArrayObject *) PyArray_FROMANY(haplotype, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_DIM(haplotype_array, 0) != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }
    if (forward_norm != NULL) {
        forward_norm_array = (PyArrayObject *) PyArray_FROMANY(
            forward_norm, NPY_FLOAT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        if (forward_norm_array == NULL) {
            goto out;
        }
        if (PyArray_DIM(forward_norm_array, 0) != (npy_intp) num_sites) {
            PyErr_SetString(
                PyExc_ValueError, "normalisation_factor must have length num_sites");
            goto out;
        }
    }
    err = compressed_matrix_alloc(&cm, num_sites,
        2 * (self->hmm->num_nodes + 1) * sizeof(double));
    cm_allocated = true;
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    if (forward_norm_array == NULL) {
        err = tsls_hmm_forward(self->hmm,
            (allele_t *) PyArray_DATA(haplotype_array), &cm);
    } else {
        err = tsls_hmm_backward(self->hmm,
            (allele_t *) PyArray_DATA(haplotype_array),
            (double *) PyArray_DATA(forward_norm_array), &cm);
    }
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }

    ret = compressed_matrix_to_tuple(&cm, false);
out:
    if (cm_allocated) {
        compressed_matrix_free(&cm);
    }
    Py_XDECREF(haplotype_array);
    Py_XDECREF(forward_norm_array);
    return ret;
}

/* Run the Viterbi algorithm, returning the find_path outputs followed by
 * the compressed matrix of normalised Viterbi values with flags. */
static PyObject *
Hmm_viterbi_matrix(Hmm *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *matrix = NULL;
    static char *kwlist[] = { "haplotype", "start", "end", NULL };
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *match = NULL;
    compressed_matrix_t cm;
    bool cm_allocated = false;
    size_t num_edges;
    int start, end;
    npy_intp dims[1];
    size_t num_sites;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "Oii", kwlist, &haplotype, &start, &end)) {
        goto out;
    }
    num_sites = self->hmm->num_sites;
    haplotype_array
        = (PyArrayObject *) PyArray_FROMANY(haplotype, NPY_INT8, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_DIM(haplotype_array, 0) != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }
    dims[0] = (npy_intp) num_sites;
    left = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    right = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    match = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT8);
    if (left == NULL || right == NULL || parent == NULL || match == NULL) {
        goto out;
    }
    err = compressed_matrix_alloc(
        &cm, num_sites, 2 * (self->hmm->num_nodes + 1) * sizeof(double));
    cm_allocated = true;
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tsls_hmm_find_path_matrix(self->hmm, (tsk_id_t) start,
        (tsk_id_t) end, (allele_t *) PyArray_DATA(haplotype_array),
        (allele_t *) PyArray_DATA(match), &num_edges, PyArray_DATA(left),
        PyArray_DATA(right), PyArray_DATA(parent), &cm);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    matrix = compressed_matrix_to_tuple(&cm, true);
    if (matrix == NULL) {
        goto out;
    }
    ret = Py_BuildValue(
        "(kNNNNN)", (unsigned long) num_edges, left, right, parent, match, matrix);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
    match = NULL;
    matrix = NULL;
out:
    if (cm_allocated) {
        compressed_matrix_free(&cm);
    }
    Py_XDECREF(haplotype_array);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    Py_XDECREF(match);
    Py_XDECREF(matrix);
    return ret;
}

static PyObject *
Hmm_forward_matrix(Hmm *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", NULL };
    PyObject *haplotype = NULL;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &haplotype)) {
        goto out;
    }
    ret = Hmm_run_compressed_matrix(self, haplotype, NULL);
out:
    return ret;
}

static PyObject *
Hmm_backward_matrix(Hmm *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = { "haplotype", "normalisation_factor", NULL };
    PyObject *haplotype = NULL;
    PyObject *forward_norm = NULL;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO", kwlist, &haplotype, &forward_norm)) {
        goto out;
    }
    ret = Hmm_run_compressed_matrix(self, haplotype, forward_norm);
out:
    return ret;
}

static PyObject *
Hmm_get_mean_traceback_size(Hmm *self, void *closure)
{
    PyObject *ret = NULL;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue(
        "d", tsls_hmm_get_mean_traceback_size(self->hmm));
out:
    return ret;
}

static PyObject *
Hmm_get_num_sparse_sites(Hmm *self, void *closure)
{
    PyObject *ret = NULL;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("n", (Py_ssize_t) self->hmm->num_sparse_sites);
out:
    return ret;
}

static PyObject *
Hmm_get_mean_num_distinct_values(Hmm *self, void *closure)
{
    PyObject *ret = NULL;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue(
        "d", tsls_hmm_get_mean_num_distinct_values(self->hmm));
out:
    return ret;
}

static PyObject *
Hmm_get_total_memory(Hmm *self, void *closure)
{
    PyObject *ret = NULL;
    unsigned long val;

    if (Hmm_check_state(self) != 0) {
        goto out;
    }

#if defined(TSI_NO_ATOMICS)
    /* Without atomics, return an obviously wrong value */
    val = (unsigned long) PY_SSIZE_T_MAX;
#else
    val = tsls_hmm_get_total_memory(self->hmm);
#endif
    ret = Py_BuildValue("k", val);

out:
    return ret;
}

static PyGetSetDef Hmm_getsetters[] = {
    { "mean_traceback_size", (getter) Hmm_get_mean_traceback_size, NULL,
        "The mean size of the traceback per site." },
    { "num_sparse_sites", (getter) Hmm_get_num_sparse_sites, NULL,
        "The number of sites processed sparsely in the last Viterbi run." },
    { "mean_num_distinct_values",
        (getter) Hmm_get_mean_num_distinct_values, NULL,
        "The mean number of distinct likelihood values per site in the last "
        "Viterbi run (if count_distinct is set)." },
    { "total_memory", (getter) Hmm_get_total_memory, NULL,
        "The total amount of memory used by this HMM." },
    { NULL } /* Sentinel */
};

static PyMethodDef Hmm_methods[] = {
    { "find_path", (PyCFunction) Hmm_find_path, METH_VARARGS | METH_KEYWORDS,
        "Returns a best match path for the specified haplotype through the ancestors." },
    { "viterbi_matrix", (PyCFunction) Hmm_viterbi_matrix,
        METH_VARARGS | METH_KEYWORDS,
        "Runs the Viterbi algorithm and also returns the compressed matrix of "
        "normalised values and recombination flags." },
    { "forward_matrix", (PyCFunction) Hmm_forward_matrix,
        METH_VARARGS | METH_KEYWORDS,
        "Runs the forward algorithm and returns the compressed matrix arrays." },
    { "backward_matrix", (PyCFunction) Hmm_backward_matrix,
        METH_VARARGS | METH_KEYWORDS,
        "Runs the backward algorithm and returns the compressed matrix arrays." },
    { NULL } /* Sentinel */
};

static PyTypeObject HmmType = {
    // clang-format off
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_tsls.Hmm",
    .tp_basicsize = sizeof(Hmm),
    .tp_dealloc = (destructor) Hmm_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Hmm objects",
    .tp_methods = Hmm_methods,
    .tp_getset = Hmm_getsetters,
    .tp_init = (initproc) Hmm_init,
    .tp_new = PyType_GenericNew,
    // clang-format on
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef tsls_methods[] = {
    { NULL } /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef tslsmodule
    = { PyModuleDef_HEAD_INIT, "_tsls", /* name of module */
          MODULE_DOC,                      /* module documentation, may be NULL */
          -1, tsls_methods, NULL, NULL, NULL, NULL };

#define INITERROR return NULL

PyObject *
PyInit__tsls(void)

#else
#define INITERROR return

void
init_tsls(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&tslsmodule);
#else
    PyObject *module = Py_InitModule3("_tsls", tsls_methods, MODULE_DOC);
#endif
    if (module == NULL) {
        INITERROR;
    }
    /* Initialise numpy */
    import_array();

    register_lwt_class(module);

    /* TreeSequence type */
    TreeSequenceType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TreeSequenceType) < 0) {
        INITERROR;
    }
    Py_INCREF(&TreeSequenceType);
    PyModule_AddObject(module, "TreeSequence", (PyObject *) &TreeSequenceType);

    /* Hmm type */
    HmmType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&HmmType) < 0) {
        INITERROR;
    }
    Py_INCREF(&HmmType);
    PyModule_AddObject(module, "Hmm", (PyObject *) &HmmType);

    TslsLibraryError = PyErr_NewException("_tsls.LibraryError", NULL, NULL);
    Py_INCREF(TslsLibraryError);
    PyModule_AddObject(module, "LibraryError", TslsLibraryError);

    TslsMatchImpossible = PyErr_NewException("_tsls.MatchImpossible", NULL, NULL);
    Py_INCREF(TslsMatchImpossible);
    PyModule_AddObject(module, "MatchImpossible", TslsMatchImpossible);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
