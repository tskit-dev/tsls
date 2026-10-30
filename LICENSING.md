# Licensing

tsls is distributed under the MIT License (see `LICENSE`).

The HMM engine in tsls is derived from tsinfer's Li and Stephens HMM
matching engine. tsinfer is distributed under the GNU General Public
License, version 3 or later (GPL-3.0-or-later). The tsinfer code was first
copied into this repository as `tsinfer-hmm/`, from
[tsinfer](https://github.com/tskit-dev/tsinfer) at commit
`53d7363bc311ada143ae7624d6f44a608018a000`, and relicensed to MIT. The
tsls engine was then developed from that copy, which has since been
removed (it is in the git history). This document records how the
relicensing was done. The paths below are those of the copy, which match
the paths in tsinfer.

## Relicensing statement

I, Jerome Kelleher, am the original author of the tsinfer Li and Stephens
HMM matching engine and wrote the vast majority of the code copied here.
I relicense this code from GPL-3.0-or-later to the MIT License for
inclusion in tsls.

The relicensing is done in a single commit with the title "Relicense
tsinfer HMM code from GPL-3.0-or-later to MIT". That commit changes only
licence headers and metadata, with no code changes. The commit before it
adds the files verbatim, still under GPL-3.0-or-later.

The copyright lines in each file ("Copyright (C) ... University of
Oxford") are kept unchanged. Only the licence terms are changed.

## Files

These files were copied from tsinfer. The paths are relative to this
directory and match the paths in tsinfer.

| File | Original licence |
|------|------------------|
| `lib/ancestor_matcher.c` | GPL-3.0-or-later |
| `lib/tsinfer.h` | GPL-3.0-or-later |
| `lib/err.c` | GPL-3.0-or-later |
| `lib/err.h` | GPL-3.0-or-later (no header; covered by the project licence) |
| `lib/tsi_blkalloc.c` | Adapted from tskit's `tsk_blkalloc` (MIT) |
| `lib/meson.build` | GPL-3.0-or-later (no header) |
| `lib/tests/tests.c` | GPL-3.0-or-later |
| `lib/test_data/*.trees` | Test data |
| `_tsinfermodule.c` | GPL-3.0-or-later (no header) |
| `setup.py`, `pyproject.toml`, `MANIFEST.in` | GPL-3.0-or-later (no header) |
| `tsinfer/matching.py` | GPL-3.0-or-later |
| `tsinfer/arg_ops.py` | GPL-3.0-or-later |
| `tests/test_lshmm.py` | GPL-3.0-or-later (no header) |
| `tests/test_arg_ops.py` | GPL-3.0-or-later (no header) |
| `tests/test_python_c.py` | GPL-3.0-or-later |
| `tests/test_matcher_fixtures.py` | GPL-3.0-or-later |

These tsinfer files were deliberately not copied:

- `lib/ancestor_builder.c`: the ancestor builder, not part of the matching
  engine.
- `lib/avl.c` and `lib/avl.h`: a third-party AVL tree library under
  LGPL-2.1-or-later (Michael H. Buselli and Wessel Dankers). Only the
  ancestor builder uses it.

## Authorship audit

Before relicensing, the tsinfer git history of every copied file was
audited. The audit used `git log --follow` and `git blame -w -C -C` at the
source commit. Surviving lines are counted by author:

| File | Jerome Kelleher | Ben Jeffery | Yan Wong | Duncan Mbuli-Robertson |
|------|----:|----:|----:|----:|
| `lib/ancestor_matcher.c` | 1203 | 11 | 7 | 0 |
| `lib/tsinfer.h` | 203 | 39 | 7 | 2 |
| `lib/err.c` | 124 | 5 | 0 | 0 |
| `lib/err.h` | 48 | 1 | 1 | 0 |
| `lib/tsi_blkalloc.c` | 0 | 112 | 0 | 0 |
| `lib/meson.build` | 30 | 2 | 0 | 0 |
| `lib/tests/tests.c` | 1788 | 1 | 7 | 26 |
| `_tsinfermodule.c` | 807 | 12 | 12 | 73 |
| `tsinfer/matching.py` | 433 | 0 | 0 | 0 |
| `tsinfer/arg_ops.py` | 163 | 0 | 0 | 0 |
| `tests/test_lshmm.py` | 1028 | 0 | 0 | 0 |
| `tests/test_arg_ops.py` | 246 | 0 | 0 | 0 |
| `tests/test_matcher_fixtures.py` | 350 | 0 | 0 | 0 |
| `tests/test_python_c.py` | 321 | 21 | 1 | 7 |

The other contributions are as follows:

- **Ben Jeffery** (October 2025):
  - the thread-safe block allocator (`tsi_blkalloc.c`, which is adapted
    from MIT-licensed tskit code, and the matching declarations in
    `tsinfer.h`);
  - replacing the precision parameter with a likelihood threshold;
  - the flag to disable weighting by number of samples;
  - support for non-zero ancestral states;
  - the Python bindings and tests for these changes.
- **Yan Wong** (2019–2020): mechanical renames ("mutation" to "mismatch",
  "age" to "time", rate parameter names) and one error code.
- **Duncan Mbuli-Robertson** (2025–2026): almost all of these lines are in
  the ancestor builder and its tests. The ancestor builder code is removed
  from tsls when the package is trimmed down to the matching engine. The
  rest are a few lines in shared declarations.

Earlier versions of the matching engine had other contributors, in
`lib/tree_sequence_builder.c`, `lib/object_heap.c` and `tests/algorithm.py`.
Those files were deleted from tsinfer in commits 3930cd0 and eddcdbd and
are not included here.

## Contributor consent

The contributors named above have agreed to the relicensing of their
contributions to the MIT License. Their agreement is recorded below and
in the `Acked-by:` trailers of the relicensing commit.

- Ben Jeffery: agreed
- Yan Wong: agreed
- Duncan Mbuli-Robertson: agreed
