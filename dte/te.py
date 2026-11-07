"""Traffic-engineering solvers and recovery methods.

All loads/demands in Mbit/s. MLU = max over alive arcs of load/capacity (dimensionless).

Metrics returned by every method (see `Outcome`):
  mlu                -- max link utilisation over the traffic the method actually carries
  carried_mbps       -- traffic carried
  lost_unservable    -- demand that NO routing can carry: endpoint node failed or endpoints
                        disconnected in the surviving graph (identical for all methods)
  lost_routing       -- demand that was physically routable but the method dropped
                        (e.g. all its precomputed paths failed)
Methods that drop more traffic can show a lower MLU, so mlu must always be read together with
lost_routing; the aggregation script reports both.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .data import Topology


@dataclass(frozen=True)
class Outcome:
    mlu: float
    carried_mbps: float
    lost_unservable_mbps: float
    lost_routing_mbps: float
    arc_load_mbps: np.ndarray


# ---------------------------------------------------------------- paths
def k_shortest_paths(topo: Topology, k: int, metric: str) -> dict[tuple[int, int], list[list[int]]]:
    """Up to k loop-free paths per demand pair, as lists of ARC indices.
    metric: "hops" or "km" (great-circle length of links)."""
    arcs, _, _ = topo.arcs()
    arc_id = {a: i for i, a in enumerate(arcs)}
    G = nx.DiGraph()
    for (u, v) in arcs:
        if metric == "hops":
            w = 1.0
        elif metric == "km":
            w = float(haversine_km(topo.lonlat[u], topo.lonlat[v]))
        else:
            raise ValueError(f"unknown path metric {metric!r}")
        G.add_edge(u, v, w=w)
    out = {}
    for s, t in zip(*np.nonzero(topo.demand_mbps)):
        ps = []
        for nodes in nx.shortest_simple_paths(G, int(s), int(t), weight="w"):
            ps.append([arc_id[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)])
            if len(ps) == k:
                break
        out[(int(s), int(t))] = ps
    return out


def haversine_km(p, q) -> np.ndarray:
    """Great-circle distance; p, q = (lon, lat) degrees, broadcastable. Mean Earth radius 6371.0088 km (IUGG)."""
    p, q = np.radians(np.asarray(p, float)), np.radians(np.asarray(q, float))
    dlon, dlat = q[..., 0] - p[..., 0], q[..., 1] - p[..., 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(p[..., 1]) * np.cos(q[..., 1]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- helpers
def _servable_pairs(topo: Topology, D: np.ndarray, dead_arc: np.ndarray, dead_node: np.ndarray):
    """Split demand pairs into servable (endpoints alive and connected) and unservable."""
    arcs, _, _ = topo.arcs()
    G = nx.DiGraph()
    G.add_nodes_from(i for i in range(topo.n) if not dead_node[i])
    G.add_edges_from(a for a, d in zip(arcs, dead_arc) if not d)
    # Arcs come in pairs (2*li, 2*li+1) per link and fail together, so reachability is symmetric
    # and s reaches t exactly when both lie in the same strongly connected component.
    comp = {v: ci for ci, nodes in enumerate(nx.strongly_connected_components(G)) for v in nodes}
    assert all(dead_arc[2 * li] == dead_arc[2 * li + 1] for li in range(len(dead_arc) // 2)), \
        "asymmetric arc failure: SCC shortcut invalid"
    servable, lost = [], 0.0
    for s, t in zip(*np.nonzero(D)):
        s, t = int(s), int(t)
        if dead_node[s] or dead_node[t] or comp[s] != comp[t]:
            lost += D[s, t]
        else:
            servable.append((s, t))
    return servable, lost


def _outcome(load, cap, dead_arc, carried, lost_unserv, lost_routing) -> Outcome:
    if np.any(load[dead_arc] > 1e-9):
        raise AssertionError("traffic placed on a failed arc")
    alive = ~dead_arc
    mlu = float(np.max(load[alive] / cap[alive])) if alive.any() else float("nan")
    return Outcome(mlu, float(carried), float(lost_unserv), float(lost_routing), load)


def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, what: str, method: str):
    """method: "highs" (HiGHS default, simplex: deterministic vertex solution) or "highs-ipm".
    Only the edge LP uses IPM: there only the optimal VALUE is used, which is unique. The path LP
    must stay on simplex because its (possibly non-unique) optimal split ratios become the
    pre-failure plan that the no_recovery / local_rescale baselines start from."""
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=method)
    if res.status != 0:
        raise RuntimeError(f"{what}: LP failed: status={res.status} {res.message}")
    return res


# ---------------------------------------------------------------- exact optimum (any routing)
def optimal_mlu_edge_lp(topo: Topology, D: np.ndarray, dead_arc: np.ndarray, dead_node: np.ndarray) -> Outcome:
    """Exact min-MLU over ALL fractional routings of the servable demand (destination-aggregated
    multicommodity flow LP). Lower bound for every method in this package."""
    arcs, cap, _ = topo.arcs()
    servable, lost_unserv = _servable_pairs(topo, D, dead_arc, dead_node)
    alive_arcs = [a for a, d in enumerate(dead_arc) if not d]
    dests = sorted({t for _, t in servable})
    if not dests:                                      # nothing routable: no load, MLU 0 by definition
        return _outcome(np.zeros(len(arcs)), cap, dead_arc, 0.0, lost_unserv, 0.0)
    nA, nT = len(alive_arcs), len(dests)
    nv = nA * nT + 1                                   # flows f[t, a] (block per destination), then U
    U = nv - 1
    # node-arc incidence of the alive graph: +1 at tail, -1 at head
    tails = np.array([arcs[a][0] for a in alive_arcs]); heads = np.array([arcs[a][1] for a in alive_arcs])
    B = sp.csr_matrix((np.r_[np.ones(nA), -np.ones(nA)], (np.r_[tails, heads], np.r_[np.arange(nA), np.arange(nA)])),
                      shape=(topo.n, nA))
    Dserv = np.zeros_like(D)
    for s, t in servable:
        Dserv[s, t] = D[s, t]
    blocks, beq = [], []
    for t in dests:                                    # conservation at every live node v != t
        keep = np.array([v for v in range(topo.n) if v != t and not dead_node[v]])
        blocks.append(B[keep])
        beq.append(Dserv[keep, t])
    A_eq = sp.hstack([sp.block_diag(blocks, format="csr"), sp.csr_matrix((sum(b.shape[0] for b in blocks), 1))],
                     format="csr")
    # sum_t f[t,a] - U*cap_a <= 0
    A_ub = sp.hstack([sp.hstack([sp.identity(nA, format="csr")] * nT), sp.csr_matrix(-cap[alive_arcs][:, None])],
                     format="csr")
    c = np.zeros(nv); c[U] = 1.0
    res = _solve_lp(c, A_ub, np.zeros(nA), A_eq, np.concatenate(beq), "edge LP", "highs-ipm")
    load = np.zeros(len(arcs))
    f = res.x[:-1].reshape(nT, nA).sum(axis=0)
    load[alive_arcs] = f
    carried = sum(D[s, t] for s, t in servable)
    out = _outcome(load, cap, dead_arc, carried, lost_unserv, 0.0)
    assert abs(out.mlu - res.x[U]) <= 1e-6 * max(1.0, res.x[U]), (out.mlu, res.x[U])
    return out


# ---------------------------------------------------------------- path-based TE
def path_lp(topo: Topology, D: np.ndarray, paths, dead_arc: np.ndarray, dead_node: np.ndarray):
    """Min-MLU split ratios over the SURVIVING precomputed paths (what a TE controller re-solves
    after a failure). Returns (Outcome, splits dict pair -> array over that pair's path list)."""
    arcs, cap, _ = topo.arcs()
    servable, lost_unserv = _servable_pairs(topo, D, dead_arc, dead_node)
    var, lost_routing, routed = [], 0.0, []
    for (s, t) in servable:
        alive_p = [pi for pi, p in enumerate(paths[(s, t)]) if not dead_arc[p].any()]
        if not alive_p:
            lost_routing += D[s, t]                    # routable in the graph, but not on any k-path
            continue
        routed.append((s, t))
        var.extend(((s, t), pi) for pi in alive_p)
    nv = len(var) + 1
    U = nv - 1
    rows, cols, vals = [], [], []
    for vi, ((s, t), pi) in enumerate(var):
        for a in paths[(s, t)][pi]:
            rows.append(a); cols.append(vi); vals.append(D[s, t])
    for a in range(len(arcs)):
        rows.append(a); cols.append(U); vals.append(-cap[a])
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(len(arcs), nv))
    pair_row = {p: i for i, p in enumerate(routed)}
    A_eq = sp.csr_matrix(([1.0] * len(var), ([pair_row[p] for p, _ in var], list(range(len(var))))),
                         shape=(len(routed), nv))
    c = np.zeros(nv); c[U] = 1.0
    res = _solve_lp(c, A_ub, np.zeros(len(arcs)), A_eq, np.ones(len(routed)), "path LP", "highs")
    splits = {p: np.zeros(len(paths[p])) for p in routed}
    for vi, (p, pi) in enumerate(var):
        splits[p][pi] = res.x[vi]
    out = evaluate_splits(topo, D, paths, splits, dead_arc, dead_node)
    assert abs(out.mlu - res.x[U]) <= 1e-6 * max(1.0, res.x[U]), (out.mlu, res.x[U])
    return out, splits


def evaluate_splits(topo, D, paths, splits, dead_arc, dead_node, renormalise=False) -> Outcome:
    """Route D with given per-pair split ratios. Traffic assigned to a failed path is LOST unless
    renormalise=True, in which case each pair's surviving paths are rescaled to sum to 1."""
    arcs, cap, _ = topo.arcs()
    servable, lost_unserv = _servable_pairs(topo, D, dead_arc, dead_node)
    load = np.zeros(len(arcs))
    carried = lost_routing = 0.0
    for (s, t) in servable:
        x = np.asarray(splits.get((s, t), np.zeros(len(paths[(s, t)]))), float).copy()
        assert x.shape == (len(paths[(s, t)]),) and np.all(x >= -1e-9)
        alive = np.array([not dead_arc[p].any() for p in paths[(s, t)]])
        if renormalise:
            # Local proportional rescaling over surviving precomputed paths. If every path the pair
            # was actually using failed (all surviving paths had weight 0), the traffic is split
            # equally over the surviving backup paths instead of being dropped; it is lost only if
            # no precomputed path survives.
            if not alive.any():
                lost_routing += D[s, t]
                continue
            tot = x[alive].sum()
            x = np.where(alive, x / tot, 0.0) if tot > 1e-12 else alive / alive.sum()
        else:
            x = np.where(alive, x, 0.0)
        lost_routing += D[s, t] * (1.0 - x.sum())
        carried += D[s, t] * x.sum()
        for pi, p in enumerate(paths[(s, t)]):
            if x[pi] > 0:
                load[p] += D[s, t] * x[pi]
    return _outcome(load, cap, dead_arc, carried, lost_unserv, max(lost_routing, 0.0))


# ---------------------------------------------------------------- OSPF/ECMP re-convergence
def ospf_ecmp(topo: Topology, D: np.ndarray, dead_arc: np.ndarray, dead_node: np.ndarray,
              weight_rule: str) -> Outcome:
    """Shortest-path routing recomputed on the surviving graph with hop-by-hop equal splitting
    over all equal-cost next hops (ECMP). weight_rule: "invcap" (w = max_cap / cap, the Cisco
    reference-bandwidth convention with reference = largest link) or "hops"."""
    arcs, cap, _ = topo.arcs()
    if weight_rule == "invcap":
        w = cap.max() / cap
    elif weight_rule == "hops":
        w = np.ones(len(arcs))
    else:
        raise ValueError(f"unknown weight_rule {weight_rule!r}")
    servable, lost_unserv = _servable_pairs(topo, D, dead_arc, dead_node)
    G = nx.DiGraph()
    for a, (u, v) in enumerate(arcs):
        if not dead_arc[a]:
            G.add_edge(u, v, w=w[a], a=a)
    load = np.zeros(len(arcs))
    for t in sorted({t for _, t in servable}):
        dist = nx.single_source_dijkstra_path_length(G.reverse(copy=False), t, weight="w")
        inflow = np.zeros(topo.n)
        for s, tt in servable:
            if tt == t:
                inflow[s] += D[s, t]
        for v in sorted(dist, key=lambda v: -dist[v]):  # farthest first: DAG order towards t
            if v == t or inflow[v] == 0:
                continue
            nh = [(x, d["a"]) for x, d in G[v].items()
                  if x in dist and abs(dist[v] - (d["w"] + dist[x])) <= 1e-9 * max(1.0, dist[v])]
            assert nh, f"no next hop from {v} to {t}"
            share = inflow[v] / len(nh)
            for x, a in nh:
                load[a] += share
                inflow[x] += share
            inflow[v] = 0.0
    carried = sum(D[s, t] for s, t in servable)
    return _outcome(load, cap, dead_arc, carried, lost_unserv, 0.0)
