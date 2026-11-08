"""Generate supplementary tables (run from disaster-te/ after paper/analysis.py):

    PYTHONPATH=. .venv/bin/python paper/supp_tables.py

Development tables use the raw values exactly as the development runs judged them (Supplementary S1
is a record of what was seen at the time); TOL-corrected values are given alongside.
"""
import json

import numpy as np
import pandas as pd

TOL = 1e-9
N = json.load(open("paper/numbers.json"))
NAME = {"ksp_hops_k4": "Shortest paths", "edge_disjoint_k4": "Edge-disjoint", "geo_diverse_k4": "Geo-diverse",
        "dart_nat_k4": "DART v1", "dart_v2_k4": "DART v2", "dart_v3_k4": "DART v3", "dart_v4_k4": "DART v4",
        "ksp_hops_k8": "Shortest paths, $k=8$", "dart_adv_k4": "DART v1 (+cuts)", "dart_v2_adv_k4": "DART v2 (+cuts)",
        "dart_v3_adv_k4": "DART v3 (+cuts)", "dart_v4_adv_k4": "DART v4 (+cuts)"}
TOPO = {"attmpls": "AttMpls", "btnorthamerica": "BtNorthAmerica", "geant2012": "G\\'EANT", "iij": "IIJ",
        "cernet": "CERNET", "garr201201": "GARR", "grnet": "GRNET", "abilene": "Abilene", "bellsouth": "BellSouth"}


def cvar(x):
    x = np.sort(np.asarray(x)); return x[-max(1, int(np.ceil(0.05 * len(x)))):].mean()


def w(name, s):
    open(f"paper/tables/{name}.tex", "w").write(s); print("wrote", name)


# S1: development history (natural test scenarios of the development networks)
rows = []
for stage, judged, label in [("pilot", "dart_nat_k4", "Pilot (v1)"), ("dev_v2", "dart_v2_k4", "Dev. v2"),
                             ("dev_v3", "dart_v3_k4", "Dev. v3"), ("dev_v4", "dart_v4_k4", "Dev. v4 (frozen)")]:
    d = pd.read_csv(f"paper/data/{stage}_natural_rows.csv.gz")
    ps = pd.read_csv(f"paper/data/{stage}_path_sets.csv").set_index(["topology", "rule"])
    for top in ["abilene", "bellsouth"]:
        g = d[d.topology == top]
        for r in [judged, "edge_disjoint_k4"]:
            x = g[g.rule == r].lost_routing_frac.values
            xt = np.where(x < TOL, 0.0, x)
            rows.append(f"{label} & {TOPO[top]} & {NAME[r]} & {100*x.mean():.2f} & {100*np.quantile(x,0.95):.2f} & "
                        f"{100*x.max():.2f} & {100*cvar(xt):.2f} & {ps.loc[(top, r), 'no_failure_mlu']:.3f} \\\\")
        rows.append("\\addlinespace")
w("tab_s1_dev", "\\begin{tabular}{lllrrrrr}\n\\toprule\nStage & Network & Rule & Mean (\\%) & P95 (\\%) & Worst (\\%) & CVaR$_{0.95}$ (\\%) & No-failure MLU \\\\\n\\midrule\n"
  + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# S3: pre-registered (raw) vs tolerance-corrected real-event comparison
rows = []
for raw, tol in zip(N["real_paired_preregistered_raw"], N["real_paired"]):
    f = lambda p: f"{p:.1e}"
    rows.append(f"{NAME[raw['rule']]} & {raw['better']}/{raw['worse']}/{raw['tie']} & {f(raw['p_holm'])} & "
                f"{tol['better']}/{tol['worse']}/{tol['tie']} & {f(tol['p_holm'])} & {tol['mean_diff_pp']:+.3f} \\\\")
w("tab_s3_noise", "\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Raw values (pre-registered)} & \\multicolumn{2}{c}{Values $<10^{-9}$ set to 0} & \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nDART vs & B/W/T & $p_{\\mathrm{Holm}}$ & B/W/T & $p_{\\mathrm{Holm}}$ & Mean diff. (pp) \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# S4a: link cuts (k = 1, 2) on Test-1 networks
rows = []
for t, v in N["cuts"].items():
    rows.append(f"{TOPO[t]} & " + " & ".join(f"{v[r]['p95']:.2f} / {v[r]['worst']:.2f}" for r in ["ksp_hops_k4", "geo_diverse_k4", "edge_disjoint_k4", "dart_v4_k4"]) + " \\\\")
w("tab_s4_cuts", "\\begin{tabular}{lrrrr}\n\\toprule\nNetwork & Shortest paths & Geo-diverse & Edge-disjoint & DART \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# S4b: clustered and stricter-threshold analyses
rows = []
for lab, key, n in [("Event-clustered (34 kt / MMI VII)", "clustered", N["real_events_distinct"]),
                    ("Stricter thresholds (64 kt / MMI VIII)", "strict_paired", N["strict_n"])]:
    for x in N[key]:
        rows.append(f"{lab if x['rule']=='ksp_hops_k4' else ''} & {n if x['rule']=='ksp_hops_k4' else ''} & {NAME[x['rule']]} & "
                    f"{x['mean_diff_pp']:+.3f} [{x['ci_lo_pp']:+.3f}, {x['ci_hi_pp']:+.3f}] & {x['better']}/{x['worse']}/{x['tie']} & {x['p_holm']:.1e} & {x['r_rb']:.2f} \\\\")
    rows.append("\\addlinespace")
w("tab_s4_secondary", "\\begin{tabular}{lrlrrrr}\n\\toprule\nAnalysis & $n$ & DART vs & Mean diff. (pp) [95\\% CI] & B/W/T & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$ \\\\\n\\midrule\n"
  + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")

# S4c: per-topology means (real events)
pt = N["real_per_topology_mean_pct"]
rows = [f"{TOPO[t]} & " + " & ".join(f"{pt[t][r]:.3f}" for r in ["ksp_hops_k4", "geo_diverse_k4", "edge_disjoint_k4", "ksp_hops_k8", "dart_v4_k4"]) + " \\\\"
        for t in ["attmpls", "btnorthamerica", "iij", "cernet", "geant2012", "grnet", "garr201201"]]
w("tab_s4_pertopo", "\\begin{tabular}{lrrrrr}\n\\toprule\nNetwork & Shortest paths & Geo-diverse & Edge-disjoint & Shortest, $k=8$ & DART \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# S5: real test earthquakes
ev = pd.read_csv("paper/data/real_events_list.csv")
q = ev[(ev.family == "earthquake") & (ev.split == "test")]
rows = [f"{TOPO[r.topology]} & \\texttt{{{r.scenario_id.replace('_', '\\_')}}} & {r.year} & {r.n_dead_nodes} & {r.n_dead_links} \\\\" for r in q.itertuples()]
w("tab_s5_eq", "\\begin{tabular}{llrrr}\n\\toprule\nNetwork & Scenario (USGS id, threshold) & Year & Failed nodes & Failed links \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
