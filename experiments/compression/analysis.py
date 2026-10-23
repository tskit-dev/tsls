"""
Measures of likelihood compression for a tsls CompressedMatrix (from
Hmm.viterbi_matrix or forward_matrix) on its tree sequence.

The engine stores, at each site, a list of (node, value) pairs, and every node
in the tree takes the value of its nearest ancestor in the list. Because
every node in the tree is a state with its own value, the smallest list that
represents the values exactly is the set of "change points": nodes whose
value differs from their parent's value, plus the roots.

Measures that only need the stored lists are computed at every site. Measures
that need the value of every node (decoding) are computed at a sample of
sites.
"""

import numpy as np
import pandas as pd


def site_slices(cm):
    return [slice(cm.offsets[j], cm.offsets[j + 1]) for j in range(cm.num_sites)]


class SiteMap:
    """
    The global map that the engine applies to the value of every node whose
    allele matches the query (or where the query is missing), from one site
    to the next. Values that the map does not predict need an explicit
    update in an engine that applies the map implicitly (an "affine"
    encoding).

    - Viterbi: f(v) = max(v * stay, rho / n) * e / c, where stay is
      1 - rho + rho / n, e is the matching emission probability, and c is
      the site's normalisation factor (the maximum). This is exact.
    - Forward: f(v) = a * v + b, where a = (1 - rho) * e / c. The shift b is
      rho / n * T * e / c, where T is the total at the previous site, which
      is not stored, so we fit b as the median over the carried nodes.

    Here n is the number of nodes if weight_by_n, else 1.
    """

    def __init__(self, algorithm, *, rho, mu, n, num_alleles):
        self.algorithm = algorithm
        self.rho = rho
        self.n = n
        self.emission = 1 - (num_alleles - 1) * mu

    def predict(self, previous, normalisation_factor, current):
        """
        Return the predicted values at a site for nodes with the specified
        previous values. For forward, the shift is fitted to the specified
        current values.
        """
        rho, n, e, c = self.rho, self.n, self.emission, normalisation_factor
        if self.algorithm == "viterbi":
            return np.maximum(previous * (1 - rho + rho / n), rho / n) * e / c
        a = (1 - rho) * e / c
        b = np.median(current - a * previous) if len(previous) > 0 else 0
        return a * previous + b


def list_measures(cm, site_map=None):
    """
    Return a DataFrame of per-site measures computed from the stored lists:

    - stored: the number of stored nodes;
    - distinct: the number of distinct stored values;
    - mutation_nodes: stored nodes with a mutation at the site;
    - carried: stored nodes that were also stored at the previous site;
    - added, removed: the number of (node, value) pairs added and removed
      relative to the previous site (the size of a delta encoding);
    - jaccard: the overlap of the stored node sets with the previous site;
    - recombination_flags: stored nodes with recombination_required set
      (Viterbi only);
    - new_list: whether the engine stores a new traceback list at this site
      (Viterbi only). The engine reuses the previous site's list only if the
      nodes and flags are identical, so traceback memory is the sum of stored
      over sites with new_list.

    The encoding measures compare each site with the previous one:

    - entered, left: nodes stored at this site and not the previous one, and
      the reverse (the churn in the list's membership);
    - changed: nodes stored at both sites whose value changed (changed_tol:
      by a relative difference of more than 1e-9);
    - flag_changed: nodes stored at both sites whose recombination_required
      flag changed (Viterbi only);
    - affine_residual: nodes stored at both sites whose value is not
      predicted by the site's global map (see SiteMap), to a relative
      tolerance of 1e-9, if site_map is specified;
    - affine_fit: the fraction of nodes stored at both sites whose value is
      predicted by the map.
    """
    ts = cm.ts
    mutation_nodes = [set() for _ in range(ts.num_sites)]
    for site, node in zip(ts.mutations_site, ts.mutations_node):
        mutation_nodes[site].add(int(node))
    rows = []
    previous_pairs = set()
    previous_nodes = set()
    previous_slice = None
    previous_nonempty = None
    for j, s in enumerate(site_slices(cm)):
        row = carried_measures(cm, previous_nonempty, s, j, site_map)
        nodes = cm.nodes[s].tolist()
        values = cm.values[s].tolist()
        pairs = set(zip(nodes, values))
        node_set = set(nodes)
        union = len(node_set | previous_nodes)
        row |= {
            "site": j,
            "stored": len(nodes),
            "distinct": len(set(values)),
            "mutation_nodes": len(node_set & mutation_nodes[j]),
            "carried": len(node_set & previous_nodes),
            "added": len(pairs - previous_pairs),
            "removed": len(previous_pairs - pairs),
            "jaccard": len(node_set & previous_nodes) / union if union > 0 else 1.0,
        }
        if cm.flags is not None:
            row["recombination_flags"] = int(np.sum(cm.flags[s] == 1))
            same = (
                previous_slice is not None
                and np.array_equal(cm.nodes[s], cm.nodes[previous_slice])
                and np.array_equal(cm.flags[s], cm.flags[previous_slice])
            )
            row["new_list"] = len(nodes) > 0 and not same
        rows.append(row)
        previous_pairs = pairs
        previous_nodes = node_set
        previous_slice = s
        if len(nodes) > 0:
            previous_nonempty = s
    return pd.DataFrame(rows)


def carried_measures(cm, previous, current, site, site_map):
    """
    Return the dict of encoding measures comparing the stored lists in the
    specified previous and current slices (see list_measures). Sites with
    empty lists (outside the matched window) are skipped over.
    """
    nodes = cm.nodes[current]
    if previous is None or len(nodes) == 0:
        return {"entered": len(nodes), "left": 0, "changed": 0, "changed_tol": 0}
    previous_nodes = cm.nodes[previous]
    common, i, k = np.intersect1d(previous_nodes, nodes, return_indices=True)
    before = cm.values[previous][i]
    after = cm.values[current][k]
    row = {
        "entered": len(nodes) - len(common),
        "left": len(previous_nodes) - len(common),
        "changed": int(np.sum(before != after)),
        "changed_tol": int(np.sum(~np.isclose(before, after, rtol=1e-9, atol=0))),
    }
    if cm.flags is not None:
        row["flag_changed"] = int(np.sum(cm.flags[previous][i] != cm.flags[current][k]))
    if site_map is not None:
        predicted = site_map.predict(before, cm.normalisation_factor[site], after)
        on_map = np.isclose(predicted, after, rtol=1e-9, atol=0)
        row["affine_residual"] = int(np.sum(~on_map))
        row["affine_fit"] = float(np.mean(on_map)) if len(common) > 0 else np.nan
    return row


def iter_trees_at_sites(ts, sites):
    """
    Yield (tree, site) for each of the specified sorted site IDs.
    """
    position = ts.sites_position[sites]
    for tree in ts.trees():
        start, stop = np.searchsorted(position, tree.interval)
        for k in range(start, stop):
            yield tree, int(sites[k])


def decode_site(cm, tree, site):
    """
    Return the (value, in_tree) tuple of arrays over all nodes at the site.
    """
    N = cm.ts.num_nodes
    parent = tree.parent_array
    in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
    s = slice(cm.offsets[site], cm.offsets[site + 1])
    value = np.full(N + 1, np.nan)
    value[cm.nodes[s]] = cm.values[s]
    ancestor = parent.copy()
    unset = np.where(in_tree & np.isnan(value[:N]))[0]
    while len(unset) > 0:
        value[unset] = value[ancestor[unset]]
        ancestor[unset] = parent[ancestor[unset]]
        unset = unset[np.isnan(value[unset])]
    return np.where(in_tree, value[:N], np.nan), in_tree


def change_points(value, in_tree, parent):
    """
    Return the boolean array of change points: nodes in the tree that are
    roots or whose value differs from their parent's.
    """
    N = len(value)
    p = parent[:N]
    parent_value = np.where(p >= 0, value[np.maximum(p, 0)], np.nan)
    return in_tree & ((p == -1) | (value != parent_value))


def quantise(value, digits):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(value > 0, np.round(np.log10(value) * digits) / digits, -np.inf)


def samples_only_parsimony(tree, labels, is_sample):
    """
    Return the number of change points needed to give each sample its label
    with internal nodes free: the cost of representing only the samples, as
    in tskit. This is a Fitch-style pass with Hartigan's most-frequent-state
    sets for multifurcating trees; it only keeps the optimal state sets, so
    it is an approximation to the minimum, but good enough for comparison.
    """
    num_changes = 0
    sets = {}
    for root in tree.roots:
        if tree.num_children(root) == 0 and not is_sample[root]:
            continue
        for u in tree.nodes(root, order="postorder"):
            children = [c for c in tree.children(u) if c in sets]
            counts = {}
            for c in children:
                for state in sets[c]:
                    counts[state] = counts.get(state, 0) + 1
            if is_sample[u]:
                # A sample's own label is fixed. Children that disagree cost a
                # change each.
                state = labels[u]
                num_changes += sum(1 for c in children if state not in sets[c])
                sets[u] = {state}
            elif len(children) == 0:
                continue
            else:
                best = max(counts.values())
                sets[u] = {s for s, k in counts.items() if k == best}
                num_changes += len(children) - best
        if root in sets:
            num_changes += 1
    return num_changes


def decoded_measures(cm, num_sites=200, parsimony_sites=20, seed=1, site_map=None):
    """
    Return a DataFrame of per-site measures computed by decoding the values
    at a sample of sites:

    - change_points: the minimum number of stored nodes;
    - redundant: stored nodes with the same value as their parent;
    - redundant_mutation: redundant stored nodes with a mutation at the site;
    - in_tree: the number of nodes in the tree;
    - distinct_all: the number of distinct values over all nodes;
    - quantised_<d>: change points after rounding log10 values to 1/d;
    - mean_clade_size, median_clade_size: subtree sizes of the change points;
    - samples_only: change points if only samples had to be represented (at
      a smaller sample of sites, and only if the tree sequence has samples).

    The engine stores its lists before removing values equal to the parent's,
    and gives nodes their own values when edges change, so the stored lists
    can change more between sites than the values do. To measure the
    encodings on the minimal lists, we also decode the previous non-empty
    site and compare the change points (if site_map is specified):

    - cp_entered, cp_left: nodes that became or stopped being change points;
    - cp_changed: change points at both sites whose value changed;
    - cp_affine_residual: change points at both sites whose value is not
      predicted by the site's global map.
    """
    ts = cm.ts
    rng = np.random.default_rng(seed)
    nonempty = np.where(np.diff(cm.offsets) > 0)[0]
    sites = np.sort(
        rng.choice(nonempty, size=min(num_sites, len(nonempty)), replace=False)
    )
    parsimony = set(
        rng.choice(sites, size=min(parsimony_sites, len(sites)), replace=False)
    )
    sampled = set(sites.tolist())
    if site_map is not None:
        index = np.searchsorted(nonempty, sites)
        previous = nonempty[index[index > 0] - 1]
        sites = np.union1d(sites, previous)
    is_sample = np.zeros(ts.num_nodes, dtype=bool)
    is_sample[ts.samples()] = True
    mutation_nodes = {}
    for site, node in zip(ts.mutations_site, ts.mutations_node):
        mutation_nodes.setdefault(int(site), set()).add(int(node))
    rows = []
    last = None
    for tree, site in iter_trees_at_sites(ts, sites):
        value, in_tree = decode_site(cm, tree, site)
        parent = tree.parent_array
        cp = change_points(value, in_tree, parent)
        cp_nodes = np.where(cp)[0]
        current = (site, cp_nodes, value[cp_nodes])
        previous_site = last
        last = current
        if site not in sampled:
            continue
        s = slice(cm.offsets[site], cm.offsets[site + 1])
        stored = cm.nodes[s]
        redundant = stored[~cp[stored]]
        clade_sizes = np.array([])
        if ts.num_samples > 0:
            clade_sizes = np.array([tree.num_samples(u) for u in np.where(cp)[0]])
        row = {
            "site": site,
            "stored": len(stored),
            "change_points": int(np.sum(cp)),
            "redundant": len(redundant),
            "redundant_mutation": len(
                set(redundant.tolist()) & mutation_nodes.get(site, set())
            ),
            "in_tree": int(np.sum(in_tree)),
            "distinct_all": len(np.unique(value[in_tree])),
            "mean_clade_size": float(np.mean(clade_sizes))
            if len(clade_sizes)
            else np.nan,
            "median_clade_size": float(np.median(clade_sizes))
            if len(clade_sizes)
            else np.nan,
        }
        for digits in [1, 2, 4]:
            q = np.where(in_tree, quantise(value, digits), np.nan)
            row[f"quantised_{digits}"] = int(np.sum(change_points(q, in_tree, parent)))
        if site in parsimony and ts.num_samples > 0:
            _, labels = np.unique(np.nan_to_num(value, nan=-1), return_inverse=True)
            row["samples_only"] = samples_only_parsimony(tree, labels, is_sample)
        if site_map is not None and previous_site is not None:
            row |= change_point_churn(cm, previous_site, current, site_map)
        rows.append(row)
    return pd.DataFrame(rows)


def change_point_churn(cm, previous, current, site_map):
    """
    Return the dict of encoding measures comparing the change points
    (site, nodes, values) at the previous and current sites.
    """
    _, previous_nodes, previous_values = previous
    site, nodes, values = current
    common, i, k = np.intersect1d(previous_nodes, nodes, return_indices=True)
    before = previous_values[i]
    after = values[k]
    predicted = site_map.predict(before, cm.normalisation_factor[site], after)
    return {
        "cp_entered": len(nodes) - len(common),
        "cp_left": len(previous_nodes) - len(common),
        "cp_changed": int(np.sum(before != after)),
        "cp_affine_residual": int(
            np.sum(~np.isclose(predicted, after, rtol=1e-9, atol=0))
        ),
    }


def value_histogram(cm, top=10):
    """
    Return a DataFrame of the most common stored values and the fraction of
    stored entries with each value.
    """
    values, counts = np.unique(cm.values, return_counts=True)
    order = np.argsort(-counts)[:top]
    return pd.DataFrame(
        {"value": values[order], "fraction": counts[order] / np.sum(counts)}
    )
