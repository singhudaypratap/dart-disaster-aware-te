"""Path-set selection rules (which k paths each demand pair precomputes).

Every rule returns {pair: [k paths as arc-index lists]} and starts from the SAME first path, the
shortest path by hops, so normal-condition routing starts from the same primary route for every
rule; rules differ only in the backup paths they add.

Rules
  ksp_hops      k shortest simple paths by hop count (the phase-1 baseline)
  edge_disjoint shortest path, remove its links, repeat (classical link-disjoint protection);
                topped up with ksp_hops paths if fewer than k disjoint paths exist
  geo_diverse   greedily add the candidate with the fewest link sample points within
                `geo_sep_km` of already chosen paths (classical geographic diversity, no data)
  dart_opt      greedily add the candidate that maximises the weighted number of TRAINING failure
                scenarios in which the pair is servable and at least one chosen path survives
                (weighted maximum coverage; greedy is a (1-1/e)-approximation for this
                submodular objective). Learns shared risk from real hazard history.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .data import Topology
from .failures import link_sample_points
from .te import haversine_km


def candidate_paths(topo: Topology, n_candidates: int, max_stretch_km: float):
    """Up to n_candidates loop-free paths per demand pair, by great-circle length, keeping only
    paths at most max_stretch_km x the shortest one. Paths are arc-index lists; the
    shortest-by-hops path is always included as candidate 0."""
    arcs, _, _ = topo.arcs()
    arc_id = {a: i for i, a in enumerate(arcs)}
    Gkm, Ghop = nx.DiGraph(), nx.DiGraph()
    for (u, v) in arcs:
        Gkm.add_edge(u, v, w=float(haversine_km(topo.lonlat[u], topo.lonlat[v])))
        Ghop.add_edge(u, v, w=1.0)
    to_arcs = lambda nodes: [arc_id[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]
    km = lambda p: sum(Gkm.edges[arcs[a]]["w"] for a in p)
    out = {}
    for s, t in zip(*np.nonzero(topo.demand_mbps)):
        s, t = int(s), int(t)
        first = to_arcs(nx.shortest_path(Ghop, s, t, weight="w"))
        cands, L0 = [first], None
        for nodes in nx.shortest_simple_paths(Gkm, s, t, weight="w"):
            p = to_arcs(nodes)
            L0 = km(p) if L0 is None else L0
            if km(p) > max_stretch_km * L0:
                break
            if p != first:
                cands.append(p)
            if len(cands) == n_candidates:
                break
        out[(s, t)] = cands
    return out


def select_ksp_hops(topo, k):
    from .te import k_shortest_paths
    return k_shortest_paths(topo, k, "hops")


def select_edge_disjoint(topo, k, ksp_fill):
    arcs, _, link_of_arc = topo.arcs()
    out = {}
    for (s, t), fill in ksp_fill.items():
        G = nx.Graph()
        G.add_edges_from(topo.links)
        chosen = []
        while len(chosen) < k and nx.has_path(G, s, t):
            nodes = nx.shortest_path(G, s, t)
            ids = {a: i for i, a in enumerate(arcs)}
            p = [ids[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]
            chosen.append(p)
            G.remove_edges_from(topo.links[link_of_arc[a]] for a in p)
        for p in fill:                       # top up with shortest paths, skipping duplicates
            if len(chosen) == k:
                break
            if p not in chosen:
                chosen.append(p)
        out[(s, t)] = chosen
    return out


def select_geo_diverse(topo, k, cands, geo_sep_km, link_step_km):
    _, _, link_of_arc = topo.arcs()
    pts = link_sample_points(topo, link_step_km)
    out = {}
    for pair, C in cands.items():
        chosen = [C[0]]
        while len(chosen) < min(k, len(C)):
            used = np.concatenate([pts[link_of_arc[a]] for p in chosen for a in p])
            best, best_key = None, None
            for ci, p in enumerate(C):
                if p in chosen:
                    continue
                P = np.concatenate([pts[link_of_arc[a]] for a in p])
                # number of this path's points within geo_sep_km of any chosen point
                near = (haversine_km(P[:, None, :], used[None, :, :]) <= geo_sep_km).any(axis=1).sum()
                key = (near, len(p), ci)
                if best_key is None or key < best_key:
                    best, best_key = p, key
            chosen.append(best)
        out[pair] = chosen
    return out


def select_dart_opt(topo, k, cands, train_dead_arc: np.ndarray, train_dead_node: np.ndarray,
                    train_weight: np.ndarray):
    """train_dead_arc: (n_scen, n_arcs) bool; train_dead_node: (n_scen, n_nodes) bool;
    train_weight: (n_scen,) nonnegative scenario weights (sum normalised per family by caller)."""
    assert train_dead_arc.shape[0] == train_dead_node.shape[0] == train_weight.shape[0]
    assert np.all(train_weight >= 0) and train_weight.sum() > 0
    out = {}
    for (s, t), C in cands.items():
        # alive[j, c] = candidate c survives training scenario j
        alive = np.stack([~train_dead_arc[:, p].any(axis=1) for p in C], axis=1)
        # the pair only matters in scenarios where both endpoints survive (dead endpoints are
        # unservable for every rule, so they carry no information about path choice)
        rel = ~(train_dead_node[:, s] | train_dead_node[:, t])
        w = train_weight * rel
        chosen = [0]
        covered = alive[:, 0].copy()
        while len(chosen) < min(k, len(C)):
            gain = ((alive & ~covered[:, None]) * w[:, None]).sum(axis=0)
            gain[chosen] = -1.0
            # tie-break: shorter (fewer arcs), then candidate order (by km)
            order = sorted(range(len(C)), key=lambda c: (-gain[c], len(C[c]), c))
            c = order[0]
            chosen.append(c)
            covered |= alive[:, c]
        out[(s, t)] = [C[c] for c in chosen]
    return out


def _shortest_disjoint_from(topo, s, t, avoid_links):
    """Shortest-hop s-t path avoiding the given undirected links, as arc ids; None if none."""
    arcs, _, _ = topo.arcs()
    ids = {a: i for i, a in enumerate(arcs)}
    G = nx.Graph()
    G.add_edges_from(l for l in topo.links if l not in avoid_links)
    if s not in G or t not in G or not nx.has_path(G, s, t):
        return None
    nodes = nx.shortest_path(G, s, t)
    return [ids[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]


def select_dart_v2(topo, k, cands, train_dead_arc, train_dead_node, train_weight):
    """DART v2 (development version; see reports/ for its history).
      1. primary path = shortest by hops (as every rule);
      2. second path = shortest link-disjoint path (structural protection against ANY single
         failure on the primary, not only the training footprints);
      3. remaining k-2 paths: greedy weighted coverage of training failure scenarios not yet
         survived, ties broken by fewest links shared with already chosen paths, then hops.
    Candidate pool = cands[pair] plus the link-disjoint path of step 2."""
    _, _, link_of_arc = topo.arcs()
    assert train_dead_arc.shape[0] == train_dead_node.shape[0] == train_weight.shape[0]
    out = {}
    for (s, t), C in cands.items():
        C = list(C)
        chosen = [0]
        d = _shortest_disjoint_from(topo, s, t, {topo.links[link_of_arc[a]] for a in C[0]})
        if d is not None:
            if d not in C:
                C.append(d)
            chosen.append(C.index(d))
        alive = np.stack([~train_dead_arc[:, p].any(axis=1) for p in C], axis=1)
        rel = ~(train_dead_node[:, s] | train_dead_node[:, t])
        w = train_weight * rel
        covered = alive[:, chosen].any(axis=1)
        while len(chosen) < min(k, len(C)):
            used_links = {link_of_arc[a] for c in chosen for a in C[c]}
            gain = ((alive & ~covered[:, None]) * w[:, None]).sum(axis=0)
            key = lambda c: (-gain[c], len({link_of_arc[a] for a in C[c]} & used_links), len(C[c]), c)
            c = min((c for c in range(len(C)) if c not in chosen), key=key)
            chosen.append(c)
            covered |= alive[:, c]
        out[(s, t)] = [C[c] for c in chosen]
    return out


def select_dart_v3(topo, k, cands, train_dead_arc, train_dead_node, train_weight):
    """DART v3 (development version):
      1. backbone = greedy mutually link-disjoint shortest-hop paths, at most k-1 of them
         (same construction as the edge_disjoint rule);
      2. remaining slots: greedy weighted coverage of TRAINING failure scenarios in which every
         backbone path fails, ties broken by fewest links shared with chosen paths, then hops.
    The learned risk therefore only decides the slot(s) beyond structural protection."""
    arcs, _, link_of_arc = topo.arcs()
    ids = {a: i for i, a in enumerate(arcs)}
    out = {}
    for (s, t), C in cands.items():
        C = list(C)
        G = nx.Graph(); G.add_edges_from(topo.links)
        backbone = []
        while len(backbone) < k - 1 and nx.has_path(G, s, t):
            nodes = nx.shortest_path(G, s, t)
            p = [ids[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)]
            backbone.append(p)
            G.remove_edges_from(topo.links[link_of_arc[a]] for a in p)
        for p in backbone:
            if p not in C:
                C.append(p)
        chosen = [C.index(p) for p in backbone]
        alive = np.stack([~train_dead_arc[:, p].any(axis=1) for p in C], axis=1)
        rel = ~(train_dead_node[:, s] | train_dead_node[:, t])
        w = train_weight * rel
        covered = alive[:, chosen].any(axis=1)
        while len(chosen) < min(k, len(C)):
            used_links = {link_of_arc[a] for c in chosen for a in C[c]}
            gain = ((alive & ~covered[:, None]) * w[:, None]).sum(axis=0)
            key = lambda c: (-gain[c], len({link_of_arc[a] for a in C[c]} & used_links), len(C[c]), c)
            c = min((c for c in range(len(C)) if c not in chosen), key=key)
            chosen.append(c)
            covered |= alive[:, c]
        out[(s, t)] = [C[c] for c in chosen]
    return out


def select_dart_v4(topo, k, cands, train_dead_arc, train_dead_node, train_weight, ed_paths):
    """DART v4 (development version): start from the edge_disjoint rule's k paths (`ed_paths`)
    and replace its LAST path (or the k-th slot, after the disjoint backbone) only when a
    candidate covers strictly more weighted TRAINING failure scenarios, i.e. when the real
    hazard history shows a measurable benefit; otherwise the classical choice is kept.
    Ties among improving candidates: fewest links shared with the kept paths, then hops."""
    _, _, link_of_arc = topo.arcs()
    out = {}
    for (s, t), C in cands.items():
        base = list(ed_paths[(s, t)])
        if len(base) < 2:
            out[(s, t)] = base
            continue
        keep, last = base[:-1], base[-1]
        pool = [p for p in list(C) + [last] if p not in keep]
        pool = list({tuple(p): p for p in pool}.values())        # dedupe, keep order
        rel = ~(train_dead_node[:, s] | train_dead_node[:, t])
        w = train_weight * rel
        survive = lambda p: ~train_dead_arc[:, p].any(axis=1)
        covered = np.any([survive(p) for p in keep], axis=0)
        gain = lambda p: float(((survive(p) & ~covered) * w).sum())
        g_last = gain(last)
        used = {link_of_arc[a] for p in keep for a in p}
        better = [p for p in pool if gain(p) > g_last + 1e-12]
        if better:
            best = min(better, key=lambda p: (-gain(p), len({link_of_arc[a] for a in p} & used), len(p)))
            out[(s, t)] = keep + [best]
        else:
            out[(s, t)] = base
    return out
