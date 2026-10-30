"""
The Li and Stephens HMM on a tree sequence.
"""

from _tsls import LibraryError, MatchImpossible

from ._matching import (
    CompressedMatrix,
    Hmm,
    TreeSequence,
    load_compressed_matrix,
)
from ._model import (
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
from ._version import __version__

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

# Present everything as part of the tsls namespace, rather than the private
# modules it's defined in.
for _name in __all__:
    _obj = globals()[_name]
    if hasattr(_obj, "__module__") and not isinstance(_obj, (str, float)):
        _obj.__module__ = "tsls"
del _name, _obj
