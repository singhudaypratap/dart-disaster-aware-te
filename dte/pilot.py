"""Pilot of DART-opt path selection vs classical path-selection rules (see config/pilot.yaml).

    python -m dte.pilot --config config/pilot.yaml --workers 4 --out results/pilot

Every rule is evaluated with the same online recovery (LP re-solve of split ratios over the
rule's surviving paths, te.path_lp), so differences come only from WHICH paths were chosen.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from . import failures as F
from . import pathsel
from . import te
from .run import _provenance, load_topologies, scale_demand

CLASSICAL = ["ksp_hops_k4", "edge_disjoint_k4", "geo_diverse_k4"]
TARGET = None  # set from config: the rule judged by the criterion


def hurricane_scen(topo, storms, ids, kts, ih, step):
    return [F.hurricane_scenario(topo, sid, storms[sid], kt, ih, step) for sid in ids for kt in kts]


def stack(topo, scen):
    return (np.stack([s.dead_arc(topo) for s in scen]), np.stack([s.dead_node for s in scen]))


def family_weights(scen, fam_of, fw):
    fams = np.array([fam_of(s) for s in scen])
    w = np.zeros(len(scen))
    for f in set(fams):
        m = fams == f
        w[m] = fw[f] / m.sum()
    return w


def eval_chunk(topo, D, rule_paths, scen):
    rows = []
    for s in scen:
        da, dn = s.dead_arc(topo), s.dead_node
        opt = te.optimal_mlu_edge_lp(topo, D, da, dn)
        tot = D.sum()
        for rule, P in rule_paths.items():
            o, _ = te.path_lp(topo, D, P, da, dn)
            acct = o.carried_mbps + o.lost_unservable_mbps + o.lost_routing_mbps
            assert abs(acct - tot) <= 1e-6 * tot
            rows.append({"topology": topo.name, "scenario_id": s.scenario_id, "kind": s.kind,
                         "rule": rule, "lost_routing_frac": o.lost_routing_mbps / tot,
                         "lost_unservable_frac": o.lost_unservable_mbps / tot,
                         "mlu_ratio": o.mlu / opt.mlu if opt.mlu > 0 else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global TARGET
    cfg_text = open(a.config).read()
    cfg = yaml.safe_load(cfg_text)
    TARGET = cfg["judged_rule"]
    os.makedirs(a.out, exist_ok=False)
    open(os.path.join(a.out, "config.yaml"), "w").write(cfg_text)
    man = _provenance(cfg, cfg_text)
    step, hc = cfg["link_step_km"], cfg["hurricanes"]
    storms = F.parse_hurdat2(open(os.path.join(cfg["data_dir"], hc["hurdat2_file"])).read())
    assert not set(hc["train"]) & set(hc["test"]), "train/test storm overlap"
    grids = {}
    if cfg.get("earthquakes"):
        eqc = cfg["earthquakes"]
        assert not set(eqc["train"]) & set(eqc["test"]), "train/test earthquake overlap"
        for e in eqc["train"] + eqc["test"]:
            grids[e] = F.parse_shakemap_grid(open(os.path.join(cfg["data_dir"], f"events/shakemap_{e}.xml")).read())

    ss = np.random.SeedSequence(cfg["run"]["seed"]).spawn(len(cfg["topologies"]))
    all_rows, sel_info = [], []
    for topo, s in zip(load_topologies(cfg), ss):
        rng_train, rng_test = (np.random.default_rng(x) for x in s.spawn(2))
        D, alpha = scale_demand(topo, cfg["target_optimal_mlu"])
        dk = cfg["disks"]
        nonempty = lambda L: [x for x in L if x.dead_node.any() or x.dead_link.any()]
        tr_h = nonempty(hurricane_scen(topo, storms, hc["train"], hc["wind_kt"], hc["interp_hours"], step))
        tr_d = [x for r in dk["radii_km"] for x in F.random_disk_scenarios(
            topo, r, dk["n_per_radius_train"], dk["margin_deg"], step, rng_train, True, dk["max_tries"])]
        tr_c = [x for k in cfg["cuts"]["train_k"] for x in F.all_link_cuts(topo, k)]
        eq = cfg.get("earthquakes")   # optional block; absent in the pilot/dev configs
        quake = lambda ids: nonempty([F.shakemap_scenario(topo, grids[e], thr, step)
                                      for e in ids for thr in eq["mmi_thresholds"]]) if eq else []
        tr_h = tr_h + quake(eq["train"] if eq else [])
        te_h = nonempty(hurricane_scen(topo, storms, hc["test"], hc["wind_kt"], hc["interp_hours"], step))
        te_d = [x for r in dk["radii_km"] for x in F.random_disk_scenarios(
            topo, r, dk["n_per_radius_test"], dk["margin_deg"], step, rng_test, True, dk["max_tries"])]
        te_d = [F.Scenario("test_" + x.scenario_id, x.kind, x.dead_node, x.dead_link, x.meta) for x in te_d]
        te_c = [x for k in cfg["cuts"]["test_k"][topo.name] for x in F.all_link_cuts(topo, k)]
        te_h = te_h + quake(eq["test"] if eq else [])

        fam = lambda x: {"hurricane": "hurricane", "shakemap": "earthquake", "disk": "disk",
                         "adversarial_links": "cut"}[x.kind]
        cands = pathsel.candidate_paths(topo, cfg["candidates"]["n_candidates"], cfg["candidates"]["max_stretch_km"])
        k = cfg["k"]
        ksp4 = pathsel.select_ksp_hops(topo, k)
        nat = tr_h + tr_d
        adv = tr_h + tr_d + tr_c
        rules = {
            "ksp_hops_k4": ksp4,
            "edge_disjoint_k4": pathsel.select_edge_disjoint(topo, k, ksp4),
            "geo_diverse_k4": pathsel.select_geo_diverse(topo, k, cands, cfg["geo_sep_km"], step),
            "dart_nat_k4": pathsel.select_dart_opt(topo, k, cands, *stack(topo, nat),
                                                   family_weights(nat, fam, cfg["family_weights"])),
            "dart_adv_k4": pathsel.select_dart_opt(topo, k, cands, *stack(topo, adv),
                                                   family_weights(adv, fam, cfg["family_weights"])),
            f"ksp_hops_k{cfg['k_reference']}": pathsel.select_ksp_hops(topo, cfg["k_reference"]),
        }
        if "dart_v2_k4" in cfg["extra_rules"]:
            rules["dart_v2_k4"] = pathsel.select_dart_v2(topo, k, cands, *stack(topo, nat),
                                                         family_weights(nat, fam, cfg["family_weights"]))
        for v, fn in (("dart_v3", pathsel.select_dart_v3),):
            if f"{v}_k4" in cfg["extra_rules"]:
                rules[f"{v}_k4"] = fn(topo, k, cands, *stack(topo, nat), family_weights(nat, fam, cfg["family_weights"]))
            if f"{v}_adv_k4" in cfg["extra_rules"]:
                rules[f"{v}_adv_k4"] = fn(topo, k, cands, *stack(topo, adv), family_weights(adv, fam, cfg["family_weights"]))
        ed = rules["edge_disjoint_k4"]
        if "dart_v4_k4" in cfg["extra_rules"]:
            rules["dart_v4_k4"] = pathsel.select_dart_v4(topo, k, cands, *stack(topo, nat),
                                                         family_weights(nat, fam, cfg["family_weights"]), ed)
        if "dart_v4_adv_k4" in cfg["extra_rules"]:
            rules["dart_v4_adv_k4"] = pathsel.select_dart_v4(topo, k, cands, *stack(topo, adv),
                                                             family_weights(adv, fam, cfg["family_weights"]), ed)
        if "dart_v2_adv_k4" in cfg["extra_rules"]:
            rules["dart_v2_adv_k4"] = pathsel.select_dart_v2(topo, k, cands, *stack(topo, adv),
                                                             family_weights(adv, fam, cfg["family_weights"]))
        if cfg.get("evaluate_rules"):
            rules = {r: rules[r] for r in cfg["evaluate_rules"]}
        for r, P in rules.items():
            assert set(P) == set(ksp4), r
            sel_info.append({"topology": topo.name, "rule": r,
                             "mean_paths_per_pair": float(np.mean([len(v) for v in P.values()])),
                             "mean_hops": float(np.mean([len(p) for v in P.values() for p in v])),
                             "no_failure_mlu": te.path_lp(topo, D, P, np.zeros(len(topo.arcs()[0]), bool),
                                                          np.zeros(topo.n, bool))[0].mlu})
        print(f"{topo.name}: train {len(tr_h)} hurricane / {len(tr_d)} disk / {len(tr_c)} cut; "
              f"test {len(te_h)} hurricane / {len(te_d)} disk / {len(te_c)} cut", flush=True)
        test = te_h + te_d + te_c
        chunks = [test[i:i + 40] for i in range(0, len(test), 40)]
        res = Parallel(n_jobs=a.workers)(delayed(eval_chunk)(topo, D, rules, c) for c in chunks)
        all_rows += [r for rr in res for r in rr]

    df = pd.DataFrame(all_rows)
    df.to_json(os.path.join(a.out, "pilot_rows.jsonl"), orient="records", lines=True)
    sel = pd.DataFrame(sel_info)
    sel.to_csv(os.path.join(a.out, "path_sets.csv"), index=False)
    df["family"] = df["kind"].map({"hurricane": "natural", "shakemap": "natural", "disk": "natural",
                                   "adversarial_links": "cut"})

    # ---- pre-registered evaluation
    lines, verdict, pvals = [], {}, []
    for topo, g in df.groupby("topology"):
        nat = g[g.family == "natural"]
        cvar = lambda x: np.sort(x.values)[-max(1, int(np.ceil(0.05 * len(x)))):].mean()
        st = nat.groupby("rule")["lost_routing_frac"].agg(mean="mean", p95=lambda x: x.quantile(0.95),
                                                          cvar95=cvar, worst="max")
        st["mlu_ratio_mean"] = nat.groupby("rule")["mlu_ratio"].mean()
        cut = g[g.family == "cut"].groupby("rule")["lost_routing_frac"].agg(cut_p95=lambda x: x.quantile(0.95), cut_worst="max")
        st = st.join(cut).join(sel[sel.topology == topo].set_index("rule")[["no_failure_mlu", "mean_hops"]])
        lines.append(f"\n=== {topo}: TEST natural scenarios n={nat.scenario_id.nunique()} (+ cuts n={g[g.family=='cut'].scenario_id.nunique()})\n"
                     + st.round(4).to_string())
        best_p95, best_worst = st.loc[CLASSICAL, "p95"].min(), st.loc[CLASSICAL, "worst"].min()
        nf_ok = st.loc[TARGET, "no_failure_mlu"] <= 1.05 * st.loc["ksp_hops_k4", "no_failure_mlu"]
        if cfg.get("criterion", "pilot") == "pilot":
            verdict[topo] = bool(st.loc[TARGET, "p95"] < best_p95 and st.loc[TARGET, "worst"] < best_worst and nf_ok)
        else:  # "final_v4": see config/test_dart_v4.yaml (sig. part applied after Holm below)
            verdict[topo] = bool(st.loc[TARGET, "cvar95"] < st.loc[CLASSICAL, "cvar95"].min()
                                 and st.loc[TARGET, "mean"] < st.loc[CLASSICAL, "mean"].min() and nf_ok)
        piv = nat.pivot_table(index="scenario_id", columns="rule", values="lost_routing_frac")
        for r in CLASSICAL:
            d = piv[TARGET] - piv[r]
            p = wilcoxon(piv[TARGET], piv[r], zero_method="wilcox").pvalue if (d != 0).any() else 1.0
            pvals.append((topo, r, float(np.median(d)), float(d.mean()), int((d < 0).sum()), int((d > 0).sum()), p))
    # Holm correction
    order = np.argsort([p[-1] for p in pvals]); m = len(pvals); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i][-1])); adj[i] = run
    lines.append("\n=== paired Wilcoxon, " + TARGET + " minus rule, per-scenario lost_routing_frac (natural test scenarios)")
    lines.append(f"{'topology':10s} {'vs rule':18s} {'median d':>9s} {'mean d':>9s} {'target better':>11s} {'target worse':>10s} {'p_holm':>8s}")
    for (t, r, med, mean, nb, nw, _), pa in zip(pvals, adj):
        lines.append(f"{t:10s} {r:18s} {med:9.4f} {mean:9.4f} {nb:11d} {nw:10d} {pa:8.2g}")
    if cfg.get("criterion", "pilot") == "final_v4":
        for (t, r, med, mean, nb, nw, _), pa in zip(pvals, adj):
            if not (pa < 0.05 and mean < 0):
                verdict[t] = False
    lines.append(f"\nPRE-REGISTERED CRITERION MET: {verdict}  -> overall {'PASS' if all(verdict.values()) else 'FAIL'}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, "pilot_report.txt"), "w").write(txt)
    man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
