"""Validation against hand-computable answers (not just 'it runs')."""
import numpy as np
import pytest

from dte import failures as F
from dte import te
from dte.data import Topology


def _topo(links, caps, D, lonlat=None):
    n = D.shape[0]
    if lonlat is None:
        lonlat = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    return Topology("toy", tuple(f"n{i}" for i in range(n)), np.asarray(lonlat, float), tuple(links),
                    np.asarray(caps, float), np.asarray(D, float), "test")


def _alive(T):
    return np.zeros(len(T.arcs()[0]), bool), np.zeros(T.n, bool)


def diamond():
    # 0 -> {1, 2} -> 3, all capacities 10, demand 0->3 = 10. Optimal: split 5/5, MLU 0.5.
    D = np.zeros((4, 4)); D[0, 3] = 10
    return _topo([(0, 1), (0, 2), (1, 3), (2, 3)], [10, 10, 10, 10], D)


def test_edge_lp_diamond():
    T = diamond()
    o = te.optimal_mlu_edge_lp(T, T.demand_mbps, *_alive(T))
    assert o.mlu == pytest.approx(0.5)
    assert o.carried_mbps == pytest.approx(10)


def test_path_lp_matches_edge_lp_when_paths_complete():
    T = diamond()
    P = te.k_shortest_paths(T, 10, "hops")
    o, splits = te.path_lp(T, T.demand_mbps, P, *_alive(T))
    assert o.mlu == pytest.approx(0.5)
    assert splits[(0, 3)] == pytest.approx([0.5, 0.5])


def test_ecmp_splits_evenly_on_diamond():
    T = diamond()
    o = te.ospf_ecmp(T, T.demand_mbps, *_alive(T), "hops")
    assert o.mlu == pytest.approx(0.5)


def test_failure_forces_single_path_and_losses():
    T = diamond()
    sc = F.Scenario("x", "test", np.zeros(4, bool), np.array([True, False, False, False]))
    da, dn = sc.dead_arc(T), sc.dead_node
    assert te.optimal_mlu_edge_lp(T, T.demand_mbps, da, dn).mlu == pytest.approx(1.0)
    P = te.k_shortest_paths(T, 10, "hops")
    _, base = te.path_lp(T, T.demand_mbps, P, *_alive(T))
    nr = te.evaluate_splits(T, T.demand_mbps, P, base, da, dn, renormalise=False)
    assert nr.lost_routing_mbps == pytest.approx(5) and nr.mlu == pytest.approx(0.5)
    rs = te.evaluate_splits(T, T.demand_mbps, P, base, da, dn, renormalise=True)
    assert rs.lost_routing_mbps == pytest.approx(0) and rs.mlu == pytest.approx(1.0)


def test_disconnection_is_unservable_for_everyone():
    T = diamond()
    sc = F.Scenario("x", "test", np.array([False, False, False, True]), np.zeros(4, bool))
    o = te.optimal_mlu_edge_lp(T, T.demand_mbps, sc.dead_arc(T), sc.dead_node)
    assert o.lost_unservable_mbps == pytest.approx(10) and o.carried_mbps == 0


def test_mlu_linear_in_demand():
    T = diamond()
    a = te.optimal_mlu_edge_lp(T, T.demand_mbps, *_alive(T)).mlu
    b = te.optimal_mlu_edge_lp(T, 3.7 * T.demand_mbps, *_alive(T)).mlu
    assert b == pytest.approx(3.7 * a)


def test_haversine_known_distance():
    # one degree of longitude on the equator = 2*pi*R/360
    assert te.haversine_km((0, 0), (1, 0)) == pytest.approx(2 * np.pi * 6371.0088 / 360, rel=1e-9)


def test_disk_footprint_hits_only_nearby_link():
    D = np.zeros((3, 3)); D[0, 2] = 1
    T = _topo([(0, 1), (1, 2)], [1, 1], D, lonlat=[(0, 0), (1, 0), (10, 0)])
    sc = F.disk_scenario(T, (5.5, 0.0), 100.0, 10.0, "d")
    assert not sc.dead_node.any()
    assert list(sc.dead_link) == [False, True]


def test_hurricane_quadrant_logic():
    # synthetic storm sitting at (0,0) with 64-kt radius 100 nm in NE only
    import datetime as dt
    tp = lambda h: F.TrackPoint(dt.datetime(2020, 1, 1, h), 0.0, 0.0, 100.0,
                                {34: (0, 0, 0, 0), 50: (0, 0, 0, 0), 64: (100, 0, 0, 0)})
    storm = {"name": "TEST", "track": [tp(0), tp(6)]}
    D = np.zeros((2, 2)); D[0, 1] = 1
    T = _topo([(0, 1)], [1], D, lonlat=[(0.5, 0.5), (-0.5, -0.5)])   # NE node inside, SW node outside
    sc = F.hurricane_scenario(T, "X", storm, 64, 1.0, 10.0, missing_radii="error", max_gap_hours=None)
    assert list(sc.dead_node) == [True, False]


def test_rescale_falls_back_to_equal_split_on_unused_backup_paths():
    # base plan uses only path 0 (0-1-3); it fails; backup 0-2-3 had weight 0 -> equal split, no loss
    T = diamond()
    P = te.k_shortest_paths(T, 10, "hops")
    base = {(0, 3): np.array([1.0, 0.0])}
    sc = F.Scenario("x", "test", np.zeros(4, bool), np.array([True, False, False, False]))
    o = te.evaluate_splits(T, T.demand_mbps, P, base, sc.dead_arc(T), sc.dead_node, renormalise=True)
    assert o.lost_routing_mbps == pytest.approx(0) and o.mlu == pytest.approx(1.0)


def test_dart_opt_picks_path_that_survives_training_failures():
    # 0->3 with three candidate routes via 1, 2, 4. Training scenarios always kill link (0,2)
    # together with the primary route via 1; DART must pick the route via 4 as backup, while
    # a pure shortest-length tie-break would pick via 2.
    from dte import pathsel
    D = np.zeros((5, 5)); D[0, 3] = 1
    T = _topo([(0, 1), (1, 3), (0, 2), (2, 3), (0, 4), (4, 3)], [1] * 6, D,
              lonlat=[(0, 0), (1, 1), (1, 0), (2, 0), (1, -1)])
    arcs, _, link_of_arc = T.arcs()
    cands = {(0, 3): [[arcs.index((0, 1)), arcs.index((1, 3))],
                      [arcs.index((0, 2)), arcs.index((2, 3))],
                      [arcs.index((0, 4)), arcs.index((4, 3))]]}
    dead_link = np.array([True, False, True, False, False, False])     # (0,1) and (0,2) fail together
    dead_arc = dead_link[link_of_arc][None, :]
    sel = pathsel.select_dart_opt(T, 2, cands, dead_arc, np.zeros((1, 5), bool), np.ones(1))
    assert sel[(0, 3)][1] == cands[(0, 3)][2]


def test_hurricane_track_crossing_dateline_interpolates_short_way():
    import datetime as dt
    mk = lambda h, lon: F.TrackPoint(dt.datetime(2020, 1, 1, h), lon, 0.0, 100.0,
                                     {34: (100, 100, 100, 100), 50: (0, 0, 0, 0), 64: (0, 0, 0, 0)})
    storm = {"name": "T", "track": [mk(0, 179.0), mk(6, -179.0)]}     # crosses 180 between rows
    D = np.zeros((2, 2)); D[0, 1] = 1
    T = _topo([(0, 1)], [1], D, lonlat=[(180.0 - 1e-9, 0.0), (0.0, 0.0)])  # node 0 on the dateline
    sc = F.hurricane_scenario(T, "X", storm, 34, 1.0, 10.0, missing_radii="error", max_gap_hours=None)
    assert sc.dead_node[0] and not sc.dead_node[1]


def test_drop_rows_does_not_bridge_long_gaps():
    import datetime as dt
    mk = lambda h, lon, r: F.TrackPoint(dt.datetime(2020, 1, 1) + dt.timedelta(hours=h), lon, 0.0, 100.0,
                                        {34: (r, r, r, r), 50: (0, 0, 0, 0), 64: (0, 0, 0, 0)})
    # usable rows at t=0 (lon 0) and t=48 h (lon 10); a 48 h gap must not paint a footprint at lon 5
    storm = {"name": "T", "track": [mk(0, 0.0, 50), mk(24, 5.0, -999), mk(48, 10.0, 50)]}
    D = np.zeros((2, 2)); D[0, 1] = 1
    T = _topo([(0, 1)], [1], D, lonlat=[(5.0, 0.0), (20.0, 0.0)])
    sc = F.hurricane_scenario(T, "X", storm, 34, 1.0, 10.0, missing_radii="drop_rows", max_gap_hours=12.0)
    assert not sc.dead_node[0]
    with pytest.raises(ValueError):
        F.hurricane_scenario(T, "X", storm, 34, 1.0, 10.0, missing_radii="error", max_gap_hours=None)
