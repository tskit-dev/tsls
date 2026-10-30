"""
The Li and Stephens HMM on a tree sequence.
"""

from _tsls import LibraryError, MatchImpossible

from ._version import __version__
from .matching import (
    CompressedMatrix,
    Hmm,
    TreeSequence,
    load_compressed_matrix,
)
from .model import (
    THRESHOLD_TOLERANCE,
    lossless_threshold,
    mismatch_odds,
    mismatch_probability,
    path_threshold,
    read_genetic_map,
    recombination_probabilities,
    recombination_probability,
    switch_odds,
)

__all__ = [
    "THRESHOLD_TOLERANCE",
    "CompressedMatrix",
    "Hmm",
    "LibraryError",
    "MatchImpossible",
    "TreeSequence",
    "__version__",
    "load_compressed_matrix",
    "lossless_threshold",
    "mismatch_odds",
    "mismatch_probability",
    "path_threshold",
    "read_genetic_map",
    "recombination_probabilities",
    "recombination_probability",
    "switch_odds",
]
