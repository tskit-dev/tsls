"""
Tests for the public API: everything is exported from the tsls namespace.
"""

import types

import tsls

# The names tsinfer and sc2ts use, and the rest of the API.
API = [
    "CompressedMatrix",
    "Hmm",
    "LibraryError",
    "MatchImpossible",
    "THRESHOLD_TOLERANCE",
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


def test_all():
    assert sorted(tsls.__all__) == sorted(API)


def test_public_names():
    # Apart from the modules, every public name is in __all__.
    public = {
        name
        for name in dir(tsls)
        if not name.startswith("_")
        and not isinstance(getattr(tsls, name), types.ModuleType)
    }
    assert public == {name for name in API if not name.startswith("_")}


def test_version():
    assert isinstance(tsls.__version__, str)
    assert len(tsls.__version__) > 0
