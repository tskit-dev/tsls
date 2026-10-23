# tsls
Li and Stephens HMM on a tree sequence

tsls will replace the two separate Li and Stephens HMM implementations in
[tsinfer](https://github.com/tskit-dev/tsinfer) and
[tskit](https://github.com/tskit-dev/tskit) with a single package.

## Current status

As a first step, the two existing implementations have been copied into
this repository. Each builds and runs its test suite on its own:

- [`tsinfer-hmm/`](tsinfer-hmm): the tsinfer ancestor matching engine,
  copied from tsinfer at commit `53d7363b`. Python package `tsinfer_hmm`,
  extension module `_tsinfer_hmm`.
- [`tskit-hmm/`](tskit-hmm): the tskit `LsHmm` implementation, copied from
  tskit at commit `0e38fc3d`. Python package `tskit_hmm`, extension module
  `_tskit_hmm`.

Both build against the tskit C library in the `git-submodules/tskit`
submodule, so clone with `--recurse-submodules` or run
`git submodule update --init`.

[`tsls/`](tsls) is the merged engine (Python package `tsls`, extension
module `_tsls`). It is based on the tsinfer engine:

- It copies from every node in the tree at a given position (any node
  with a parent or a child); isolated nodes cannot be copied. Copying a
  node from one site to the next without a recombination requires it to
  be in the tree over the whole interval between them.
- It supports arbitrary topologies (multiple roots, no special root node).
- It works directly on a tskit tree sequence (`tsls.TreeSequence`, which
  can be shared between HMMs), and moves between trees with tskit's
  `tsk_tree_position_t`, seeking straight to the first tree of a window.
- It provides the Viterbi algorithm (`Hmm.find_path`, with the same
  interface as tsinfer-hmm's `AncestorMatcher.find_path`) and the forward
  and backward algorithms (`forward_matrix`, `backward_matrix`), which
  return compressed matrices.
- It uses tsinfer's compression (a node's value is dropped when it
  equals its parent's), a likelihood threshold and `weight_by_n`.
- It supports multiple mutations per site and any number of alleles,
  read from the tree sequence and numbered at each site as tskit does
  (or by their index in a given list of alleles), and any (including
  non-integer) coordinates.

Paths match tsinfer-hmm except for ties. To match tsinfer's convention
that its ultimate root node 0 is never copied, remove node 0's edges
before building the `tsls.TreeSequence`; sc2ts ARGs already have an
isolated node 0. The `benchmarking/` directory compares the engines on
the sc2ts ARG, and `verification/` runs statistical validations of tsls on
simulated ARGs (plots, not pass/fail tests).

The git history records exactly what was copied from where. Each
directory was first added verbatim, then changed in separate commits to
make it build on its own.

## Licence

tsls is distributed under the MIT License. tsinfer is licensed under
GPL-3.0-or-later; the tsinfer code copied here was relicensed to MIT by
its author. See [tsinfer-hmm/LICENSING.md](tsinfer-hmm/LICENSING.md) for
details. The merged engine in `tsls/` is derived from tsinfer-hmm and
is covered by the same relicensing. The tskit code is MIT licensed.

## Development

In `tsls/`, `tsinfer-hmm/` or `tskit-hmm/`:

```
# Python tests
uv sync --group test
uv run pytest -n 4

# C tests (needs CUnit)
uvx meson setup build lib
ninja -C build
uvx meson test -C build
```
