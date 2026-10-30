# tsls
The Li and Stephens HMM on a tree sequence.

tsls is the Li and Stephens hidden Markov model engine for
[tsinfer](https://github.com/tskit-dev/tsinfer) and
[sc2ts](https://github.com/tskit-dev/sc2ts). It replaces the separate
implementations that were in tsinfer and [tskit](https://github.com/tskit-dev/tskit).

## The model

- The states of the HMM at a position are the nodes in the tree there:
  every node with a parent or a child. Isolated nodes cannot be copied.
  Copying a node from one site to the next without a recombination
  requires it to be in the tree over the whole interval between them.
- It supports arbitrary topologies (multiple roots, no special root node),
  multiple mutations per site, any number of alleles and any (including
  non-integer) coordinates.
- There is one likelihood model, with recombination always divided by the
  number of nodes, and exact Viterbi prune thresholds derived from it
  ([docs/likelihood_model.md](docs/likelihood_model.md); `tsls.switch_odds`,
  `tsls.path_threshold` etc.).
  sc2ts's model, which specifies a recombination by its cost in
  mismatches, is the same model with
  `r = tsls.recombination_probability(rho, num_nodes)`.

## Usage

`tsls.TreeSequence` holds a tskit tree sequence in the form the HMM uses,
and can be shared between HMMs, including in different threads.
`tsls.Hmm` runs the Viterbi algorithm (`find_path`, `viterbi_matrix`) and
the forward and backward algorithms (`forward_matrix`, `backward_matrix`),
which return compressed matrices. See the docstrings for details.

tsinfer and sc2ts trees have a vestigial root, node 0, whose only child is
node 1. tsls can copy node 0 wherever node 1 can be copied, with the same
likelihood, so callers map copies of node 0 to node 1. This is exact while
node 0's only child is node 1 and neither has mutations.

## Installation

tsls isn't on PyPI yet. It builds against the tskit C library in the
`git-submodules/tskit` submodule, so install it from git, for example:

```
uv add "tsls @ git+https://github.com/tskit-dev/tsls"
```

or clone it with `--recurse-submodules` (or run `git submodule update --init`).

## Development

```
# Python tests
uv sync --group test
uv run pytest -n 4

# C tests (needs CUnit)
uvx meson setup build lib
ninja -C build
uvx meson test -C build

# Lint
uv run --only-group lint prek -c prek.toml run --all-files
```

`verification/` runs statistical validations of tsls on simulated ARGs
(plots, not pass/fail tests); see its README.

## Licence

tsls is distributed under the MIT License. It is derived from tsinfer's
HMM engine, which its authors relicensed from GPL-3.0-or-later to MIT for
tsls (see [LICENSING.md](LICENSING.md)). The tskit code is MIT licensed.
