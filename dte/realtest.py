"""Real-event test of the frozen DART v4.

    python -m dte.realtest build    --config config/test_real_events.yaml --out results/test_real_events
    python -m dte.realtest evaluate --config config/test_real_events.yaml --out results/test_real_events --workers 4

build    : rule-based event selection (dte/events.py), footprints, train/test split by year.
           Writes scenarios.pkl, events.csv and counts.txt. No routing method is evaluated.
evaluate : requires the `criterion` block in the config (written and committed after `build`,
           before `evaluate`). Selects DART v4 paths from TRAINING events + training disks only,
           then evaluates every rule on the TEST real events.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from . import events as E
from . import failures as F
from . import pathsel
from .pilot import eval_chunk, family_weights, stack
from .run import _provenance, load_topologies, scale_demand

CLASSICAL = ["ksp_hops_k4", "edge_disjoint_k4", "geo_diverse_k4"]


def build(cfg, out):
    os.makedirs(out, exist_ok=False)
    step, cy, eq = cfg["link_step_km"], cfg["cyclones"], cfg["earthquakes"]
    hur = F.parse_hurdat2(open(os.path.join(cfg["data_dir"], cy["hurdat2_file"])).read())
    hur = {k: v for k, v in hur.items() if int(k[-4:]) >= cy["hurdat2_first_season"]}
    for k, v in hur.items():
        v["season"] = int(k[-4:])
    ib = E.parse_ibtracs(os.path.join(cfg["data_dir"], cy["ibtracs_file"]), cy["ibtracs_first_season"])
    storms = {**hur, **ib}
    assert len(storms) == len(hur) + len(ib), "storm id collision"
    manifest, scen, ev_rows, skipped = {}, {}, [], []
    for topo in load_topologies(cfg):
        lst = []
        for kt in cy["wind_kt"]:
            sc, sk = E.cyclone_scenarios(topo, storms, kt, cy["interp_hours"], step,
                                          cy["missing_radii"], cy["max_gap_hours"])
            skipped += [(topo.name, kt, *x) for x in sk]
            for s in sc:
                s.meta.update(year=storms[s.meta["storm_id"]]["season"], family="hurricane", threshold=f"{kt}kt")
            lst += sc
        cands, n_query = E.usgs_candidates(topo, eq["start"], eq["end"], eq["min_magnitude"],
                                           eq["bbox_margin_deg"], eq["prefilter_km"], step)
        for c in cands:
            grid = E.shakemap_grid(cfg["data_dir"], c["id"], manifest)
            for thr in eq["mmi_thresholds"]:
                s = F.shakemap_scenario(topo, grid, thr, step)
                if s.dead_node.any() or s.dead_link.any():
                    s.meta.update(year=c["year"], family="earthquake", threshold=f"mmi{thr}",
                                  magnitude=c["mag"], place=c["place"])
                    lst.append(s)
        for s in lst:
            ev_rows.append({"topology": topo.name, "scenario_id": s.scenario_id, "family": s.meta["family"],
                            "threshold": s.meta["threshold"], "year": s.meta["year"],
                            "split": "test" if s.meta["year"] >= cfg["split_year"] else "train",
                            "n_dead_nodes": int(s.dead_node.sum()), "n_dead_links": int(s.dead_link.sum())})
        scen[topo.name] = lst
        print(f"{topo.name}: {len(lst)} event scenarios ({n_query} USGS events queried, {len(cands)} within prefilter)", flush=True)
    pickle.dump(scen, open(os.path.join(out, "scenarios.pkl"), "wb"))
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(os.path.join(out, "events.csv"), index=False)
    json.dump(manifest, open(os.path.join(out, "shakemap_manifest.json"), "w"), indent=1)
    json.dump(skipped, open(os.path.join(out, "skipped_storms.json"), "w"))
    counts = ev.pivot_table(index=["topology"], columns=["split", "family", "threshold"],
                            values="scenario_id", aggfunc="count", fill_value=0)
    txt = counts.to_string() + f"\n\nstorm-threshold combinations skipped for missing radii: {len(skipped)}"
    open(os.path.join(out, "counts.txt"), "w").write(txt)
    print(txt)


def evaluate(cfg, out, workers):
    crit = cfg["criterion"]                      # crash if not pre-registered
    scen = pickle.load(open(os.path.join(out, "scenarios.pkl"), "rb"))
    step, dk, k = cfg["link_step_km"], cfg["disks"], cfg["k"]
    rows, seeds = [], np.random.SeedSequence(cfg["run"]["seed"]).spawn(len(cfg["topologies"]))
    for topo, ss in zip(load_topologies(cfg), seeds):
        rng_train = np.random.default_rng(ss)
        D, _ = scale_demand(topo, cfg["target_optimal_mlu"])
        evs = scen[topo.name]
        train = [s for s in evs if s.meta["year"] < cfg["split_year"]]
        test = [s for s in evs if s.meta["year"] >= cfg["split_year"]]
        disks = [x for r in dk["radii_km"] for x in F.random_disk_scenarios(
            topo, r, dk["n_per_radius_train"], dk["margin_deg"], step, rng_train, True, dk["max_tries"])]
        nat = train + disks
        fam = lambda x: x.meta.get("family", "disk")
        w = family_weights(nat, fam, cfg["family_weights"])
        cands = pathsel.candidate_paths(topo, cfg["candidates"]["n_candidates"], cfg["candidates"]["max_stretch_km"])
        ksp4 = pathsel.select_ksp_hops(topo, k)
        ed = pathsel.select_edge_disjoint(topo, k, ksp4)
        rules = {"ksp_hops_k4": ksp4, "edge_disjoint_k4": ed,
                 "geo_diverse_k4": pathsel.select_geo_diverse(topo, k, cands, cfg["geo_sep_km"], step),
                 "dart_v4_k4": pathsel.select_dart_v4(topo, k, cands, *stack(topo, nat), w, ed),
                 f"ksp_hops_k{cfg['k_reference']}": pathsel.select_ksp_hops(topo, cfg["k_reference"])}
        print(f"{topo.name}: DART trained on {len(train)} real + {len(disks)} disk scenarios; testing {len(test)}", flush=True)
        if test:
            chunks = [test[i:i + 20] for i in range(0, len(test), 20)]
            for rr in Parallel(n_jobs=workers)(delayed(eval_chunk)(topo, D, rules, c) for c in chunks):
                rows += rr
    df = pd.DataFrame(rows)
    meta = {(t, s.scenario_id): s.meta for t, L in scen.items() for s in L}
    df["family"] = [meta[(t, s)]["family"] for t, s in zip(df.topology, df.scenario_id)]
    df["threshold"] = [meta[(t, s)]["threshold"] for t, s in zip(df.topology, df.scenario_id)]
    df.to_json(os.path.join(out, "eval_rows.jsonl"), orient="records", lines=True)

    prim = df[df.threshold.isin(crit["primary_thresholds"])]
    piv = prim.pivot_table(index=["topology", "scenario_id"], columns="rule", values="lost_routing_frac")
    T = cfg["judged_rule"]
    cvar = lambda x: np.sort(np.asarray(x))[-max(1, int(np.ceil(0.05 * len(x)))):].mean()
    lines = [f"PRIMARY (thresholds {crit['primary_thresholds']}), pooled over topologies: n = {len(piv)} test scenarios",
             piv.agg(["mean", lambda x: x.quantile(0.95), cvar, "max"]).set_axis(["mean", "p95", "cvar95", "worst"]).round(4).to_string()]
    res = []
    for r in CLASSICAL:
        d = piv[T] - piv[r]
        p = wilcoxon(piv[T], piv[r]).pvalue if (d != 0).any() else 1.0
        res.append([r, float(d.mean()), int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()), p])
    m, run_ = len(res), 0.0
    for rank, i in enumerate(np.argsort([x[-1] for x in res])):
        run_ = max(run_, min(1.0, (m - rank) * res[i][-1])); res[i].append(run_)
    lines.append(f"\npaired Wilcoxon {T} minus rule (pooled): rule, mean diff, better, worse, tie, p_raw, p_holm")
    lines += [f"  {r:18s} {md:+.5f} {b:5d} {wo:5d} {ti:5d} {p:.3g} {ph:.3g}" for r, md, b, wo, ti, p, ph in res]
    passed = all(md < 0 and ph < crit["alpha"] for _, md, _, _, _, _, ph in res)
    lines.append(f"\nPRE-REGISTERED CRITERION MET: {'PASS' if passed else 'FAIL'}")
    # secondary: per topology and per threshold (descriptive)
    for key in ["topology", "threshold"]:
        g = df.groupby([key, "rule"])["lost_routing_frac"].mean().unstack().round(4)
        lines.append(f"\nSECONDARY mean routable loss by {key} (all thresholds):\n" + g.to_string())
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "real_events_report.txt"), "w").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "evaluate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.stage == "build":
        build(cfg, a.out)
    else:
        if a.workers is None:
            raise SystemExit("--workers is required for evaluate")
        evaluate(cfg, a.out, a.workers)


if __name__ == "__main__":
    main()
