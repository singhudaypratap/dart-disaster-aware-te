"""Run every (topology x demand level x scenario) job in parallel and write one row per method.

    python -m dte.run --config config/experiment.yaml --workers 32

Output: <output_dir>/<run.name>_<UTC timestamp>/
    results.jsonl     one JSON row per (topology, target_mlu, scenario, method)
    scenarios.jsonl   every scenario built, including real events that missed the topology
    manifest.json     config, input checksums, package versions, git commit, host, timing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import yaml
from joblib import Parallel, delayed

from . import failures as F
from . import te
from .data import Topology, load_sndlib_xml


# ---------------------------------------------------------------- setup
def load_topologies(cfg) -> list[Topology]:
    out = []
    for t in cfg["topologies"]:
        topo = load_sndlib_xml(os.path.join(cfg["data_dir"], t["file"]), t["capacity_policy"],
                               t["uniform_capacity_mbps"], t["assume_unit_if_missing"])
        if topo.name != t["name"]:
            raise ValueError(f"config name {t['name']!r} != file name {topo.name!r}")
        out.append(topo)
    return out


def build_scenarios(cfg, topo: Topology, rng: np.random.Generator) -> list[F.Scenario]:
    sc_cfg, step = cfg["scenarios"], cfg["geometry"]["link_step_km"]
    out = [F.no_failure(topo)] if sc_cfg["include_no_failure"] else []
    for ev in sc_cfg["shakemap"]["events"]:
        grid = F.parse_shakemap_grid(open(os.path.join(cfg["data_dir"], f"events/shakemap_{ev}.xml")).read())
        for thr in sc_cfg["shakemap"]["mmi_thresholds"]:
            out.append(F.shakemap_scenario(topo, grid, thr, step))
    hc = sc_cfg["hurricane"]
    storms = F.parse_hurdat2(open(os.path.join(cfg["data_dir"], hc["hurdat2_file"])).read())
    for sid in hc["storms"]:
        for kt in hc["wind_kt"]:
            out.append(F.hurricane_scenario(topo, sid, storms[sid], kt, hc["interp_hours"], step))
    rd = sc_cfg["random_disk"]
    for r in rd["radii_km"]:
        out += F.random_disk_scenarios(topo, r, rd["n_per_radius"], rd["margin_deg"], step, rng,
                                       rd["require_failure"], rd["max_tries"])
    lc = sc_cfg["link_cuts"]
    for k in lc["k"]:
        cuts = F.all_link_cuts(topo, k)
        if len(cuts) > lc["max_scenarios_per_k"]:
            raise RuntimeError(f"{topo.name}: {len(cuts)} {k}-link cuts > max_scenarios_per_k; "
                               "raise the limit or switch to a search-based adversary")
        out += cuts
    ids = [s.scenario_id for s in out]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"
    return out


def scale_demand(topo: Topology, target_mlu: float) -> tuple[np.ndarray, float]:
    """Scale demand so that the optimal no-failure MLU equals target_mlu (exact: MLU* is linear in D)."""
    arcs, _, _ = topo.arcs()
    base = te.optimal_mlu_edge_lp(topo, topo.demand_mbps, np.zeros(len(arcs), bool), np.zeros(topo.n, bool))
    alpha = target_mlu / base.mlu
    D = topo.demand_mbps * alpha
    check = te.optimal_mlu_edge_lp(topo, D, np.zeros(len(arcs), bool), np.zeros(topo.n, bool))
    assert abs(check.mlu - target_mlu) <= 1e-6, (check.mlu, target_mlu)
    return D, alpha


# ---------------------------------------------------------------- one job
def run_job(topo, D, target, alpha, paths, base_splits, sc: F.Scenario, methods, ospf_rule):
    dead_arc, dead_node = sc.dead_arc(topo), sc.dead_node
    total = float(D.sum())
    t0 = time.perf_counter()
    opt = te.optimal_mlu_edge_lp(topo, D, dead_arc, dead_node)
    t_opt = time.perf_counter() - t0
    rows = []
    for m in methods:
        t0 = time.perf_counter()
        if m == "optimal":
            o, dt_ = opt, t_opt
        else:
            if m == "no_recovery":
                o = te.evaluate_splits(topo, D, paths, base_splits, dead_arc, dead_node, renormalise=False)
            elif m == "local_rescale":
                o = te.evaluate_splits(topo, D, paths, base_splits, dead_arc, dead_node, renormalise=True)
            elif m == "ospf_ecmp":
                o = te.ospf_ecmp(topo, D, dead_arc, dead_node, ospf_rule)
            elif m == "te_resolve_kpaths":
                o, _ = te.path_lp(topo, D, paths, dead_arc, dead_node)
            else:
                raise ValueError(f"unknown method {m!r}")
            dt_ = time.perf_counter() - t0
        # invariants: demand accounting closes; no method beats the exact optimum on carried traffic
        acct = o.carried_mbps + o.lost_unservable_mbps + o.lost_routing_mbps
        assert abs(acct - total) <= 1e-6 * total, (m, acct, total)
        assert abs(o.lost_unservable_mbps - opt.lost_unservable_mbps) <= 1e-6 * total
        if o.lost_routing_mbps <= 1e-9 * total:
            assert o.mlu >= opt.mlu * (1 - 1e-6), (m, o.mlu, opt.mlu)
        rows.append({
            "topology": topo.name, "target_mlu": target, "demand_scale": alpha,
            "scenario_id": sc.scenario_id, "kind": sc.kind,
            "n_dead_nodes": int(dead_node.sum()), "n_dead_links": int(sc.dead_link.sum()),
            "method": m, "mlu": o.mlu, "optimal_mlu": opt.mlu,
            "mlu_ratio_to_optimal": o.mlu / opt.mlu if opt.mlu > 0 else float("nan"),
            "lost_unservable_frac": o.lost_unservable_mbps / total,
            "lost_routing_frac": o.lost_routing_mbps / total,
            "carried_frac": o.carried_mbps / total,
            "runtime_s": dt_,
        })
    return rows


# ---------------------------------------------------------------- provenance
def _provenance(cfg, cfg_text):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    files = {}
    for dp, _, fs in os.walk(cfg["data_dir"]):
        for fn in sorted(fs):
            p = os.path.join(dp, fn)
            files[os.path.relpath(p, cfg["data_dir"])] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return {
        "config": cfg, "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "git_commit": commit, "git_dirty": dirty, "input_sha256": files,
        "python": sys.version, "platform": platform.platform(), "host": socket.gethostname(),
        "packages": {p: version(p) for p in ["numpy", "scipy", "networkx", "joblib", "PyYAML"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, required=True, help="parallel processes (-1 = all cores)")
    args = ap.parse_args()
    cfg_text = open(args.config).read()
    cfg = yaml.safe_load(cfg_text)
    out_dir = os.path.join(cfg["output_dir"], f"{cfg['run']['name']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    os.makedirs(out_dir, exist_ok=False)
    manifest = _provenance(cfg, cfg_text)
    manifest["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        f.write(cfg_text)

    seeds = np.random.SeedSequence(cfg["run"]["seed"]).spawn(len(cfg["topologies"]))
    jobs, sc_rows = [], []
    for topo, ss in zip(load_topologies(cfg), seeds):
        scen = build_scenarios(cfg, topo, np.random.default_rng(ss))
        for s in scen:
            sc_rows.append({"topology": topo.name, "scenario_id": s.scenario_id, "kind": s.kind,
                            "n_dead_nodes": int(s.dead_node.sum()), "n_dead_links": int(s.dead_link.sum()),
                            "dead_nodes": [topo.nodes[i] for i in np.nonzero(s.dead_node)[0]],
                            "dead_links": [list(map(topo.nodes.__getitem__, topo.links[i]))
                                           for i in np.nonzero(s.dead_link)[0]],
                            "meta": s.meta})
        if cfg["scenarios"]["drop_scenarios_without_failure"]:
            scen = [s for s in scen if s.kind == "none" or s.dead_node.any() or s.dead_link.any()]
        paths = te.k_shortest_paths(topo, cfg["paths"]["k"], cfg["paths"]["metric"])
        for target in cfg["demand"]["target_optimal_mlu"]:
            D, alpha = scale_demand(topo, target)
            arcs, _, _ = topo.arcs()
            _, base_splits = te.path_lp(topo, D, paths, np.zeros(len(arcs), bool), np.zeros(topo.n, bool))
            jobs += [(topo, D, target, alpha, paths, base_splits, s) for s in scen]
    with open(os.path.join(out_dir, "scenarios.jsonl"), "w") as f:
        for r in sc_rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(jobs)} jobs -> {out_dir}", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=args.workers, verbose=5)(
        delayed(run_job)(*j, cfg["methods"], cfg["ospf"]["weight_rule"]) for j in jobs)
    with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
        for rows in results:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["wall_s"] = time.time() - t0
    manifest["n_jobs"] = len(jobs)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f"done in {manifest['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
