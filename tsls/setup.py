import os
import platform

import numpy
from setuptools import Extension, setup

IS_WINDOWS = platform.system() == "Windows"

libdir = "lib"
tskroot = os.path.join("..", "git-submodules", "tskit", "c")
tskdir = os.path.join(tskroot, "tskit")
kasdir = os.path.join(tskroot, "subprojects", "kastore")
includes = ["lwt_interface", libdir, tskroot, tskdir, kasdir]

tsls_source_files = [
    "hmm.c",
    "err.c",
    "tsi_blkalloc.c",
]
# The HMM works on a tsk_treeseq_t, so we need all of tskit.
tsk_source_files = [
    "core.c",
    "tables.c",
    "trees.c",
    "genotypes.c",
    "stats.c",
    "convert.c",
    "haplotype_matching.c",
]
kas_source_files = ["kastore.c"]

sources = (
    ["_tslsmodule.c"]
    + [os.path.join(libdir, f) for f in tsls_source_files]
    + [os.path.join(tskdir, f) for f in tsk_source_files]
    + [os.path.join(kasdir, f) for f in kas_source_files]
)

libraries = ["Advapi32"] if IS_WINDOWS else []

extra_compile_args = [] if IS_WINDOWS else ["-std=c11"]

_tsls_module = Extension(
    "_tsls",
    sources=sources,
    extra_compile_args=extra_compile_args,
    libraries=libraries,
    undef_macros=["NDEBUG"],
    include_dirs=includes + [numpy.get_include()],
)

setup(
    ext_modules=[_tsls_module],
)
