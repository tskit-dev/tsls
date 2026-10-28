#include "err.h"

const char *
tsls_strerror(int err)
{
    const char *ret = "Unknown error";

    switch (err) {
        case 0:
            ret = "Normal exit condition. This is not an error!";
            break;
        case TSLS_ERR_GENERIC:
            ret = "Generic tsls error - please file a bug report.";
            break;
        case TSLS_ERR_NO_MEMORY:
            ret = "Out of memory";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE:
            ret = "Unexpected failure to find matching haplotype; please open "
                  "an issue on GitHub";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA:
            ret = "Cannot find match: the specified mismatch probability is "
                  "0 or 1 and no matches are possible with these parameters";
            break;
        case TSLS_ERR_MATCH_IMPOSSIBLE_ZERO_RECOMB_PRECISION:
            ret = "Cannot find match: the specified recombination probability is "
                  "zero and no matches could be found";
            break;
        case TSLS_ERR_BAD_HAPLOTYPE_ALLELE:
            ret = "Haplotype allele must be missing (-1) or >= 0 and < num_alleles";
            break;
        case TSLS_ERR_UNKNOWN_ALLELE:
            ret = "An ancestral or derived state is not in the list of alleles";
            break;
        case TSLS_ERR_TOO_MANY_ALLELES:
            ret = "Too many distinct alleles at a site";
            break;
        case TSLS_ERR_BAD_SITE_RANGE:
            ret = "Bad site range: must have 0 <= start < end <= num_sites";
            break;
        case TSLS_ERR_BAD_NORMALISATION_FACTOR:
            ret = "Forward normalisation factors must be > 0";
            break;
        case TSLS_ERR_BAD_PRUNING:
            ret = "Bad pruning: prune threshold and relative factor must be in [0, 1], "
                  "and top_n >= 1";
            break;
    }
    return ret;
}
