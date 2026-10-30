# Changelog

## [0.1.0] - 202X-XX-XX

Initial release. tsls replaces the Li and Stephens HMM engines in tsinfer
and tskit.

- Viterbi (`Hmm.find_path`, `Hmm.viterbi_matrix`), forward and backward
  (`Hmm.forward_matrix`, `Hmm.backward_matrix`) algorithms on a tskit tree
  sequence, in which every node in the tree is a state.
- One likelihood model, with functions for its odds and exact Viterbi
  prune thresholds (`switch_odds`, `recombination_probability`,
  `mismatch_odds`, `path_threshold` etc.; see `docs/likelihood_model.md`).
- Everything is in the `tsls` namespace.
- `tsls.TreeSequence` takes one list of alleles for every site, or a list
  for each site (including a 2D array padded with empty strings), and gives
  the number of alleles at each site (`num_alleles`).
- `tsls.mismatch_probability`, the inverse of `mismatch_odds`.
- Bad inputs raise `ValueError`, and `MatchImpossible` is raised when no
  path is possible.
- The matched haplotype from `find_path` and `viterbi_matrix` is -1
  (missing) outside the sites matched.
