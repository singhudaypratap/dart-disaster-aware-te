"""Compute every number and table used in the manuscript (run from disaster-te/):

    PYTHONPATH=. .venv/bin/python paper/analysis.py

Inputs (committed): paper/data/real_events_eval_rows.jsonl, paper/data/real_events_list.csv,
paper/data/final_test_rows.csv.gz, config/test_real_events.yaml.
Outputs: paper/numbers.json (every scalar quoted in the text) and paper/tables/*.tex.

NUMERICAL TOLERANCE: losses below TOL (1e-9 of demand) are LP rounding noise (observed values
~1e-16; smallest genuine loss 3.1e-4) and are set to 0. The pre-registered analyses compared raw
values; both versions are reported (see Supplementary S3).
"""
import json
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon

TOL = 1e-9
CLASSICAL = ["ksp_hops_k4", "edge_disjoint_k4", "geo_diverse_k4"]
T = "dart_v4_k4"
NAME = {"ksp_hops_k4": "Shortest paths", "edge_disjoint_k4": "Edge-disjoint", "geo_diverse_k4": "Geo-diverse",
        "dart_v4_k4": "DART", "ksp_hops_k8": "Shortest paths, $k=8$"}
TOPO = {"attmpls": "AttMpls", "btnorthamerica": "BtNorthAmerica", "geant2012": "G\\'EANT", "iij": "IIJ",
        "cernet": "CERNET", "garr201201": "GARR", "grnet": "GRNET"}
os.makedirs("paper/tables", exist_ok=True)
N = {}


def cvar(x, a=0.95):
    x = np.sort(np.asarray(x)); return x[-max(1, int(np.ceil((1 - a) * len(x)))):].mean()


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d)); return float((r[d < 0].sum() - r[d > 0].sum()) / r.sum())   # >0 favours DART


def boot_ci(d, seed=1, n=10000):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1); return np.quantile(m, [0.025, 0.975])


def paired(piv, rules, tol_label):
    rows, ps = [], []
    for r in rules:
        d = (piv[T] - piv[r]).values
        p = wilcoxon(piv[T], piv[r]).pvalue if (d != 0).any() else 1.0
        lo, hi = boot_ci(d)
        rows.append(dict(rule=r, mean_diff_pp=100 * d.mean(), ci_lo_pp=100 * lo, ci_hi_pp=100 * hi,
                         better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()),
                         p=p, r_rb=rank_biserial(d)))
        ps.append(p)
    for row, a in zip(rows, holm(np.array(ps))):
        row["p_holm"] = a
    return rows


def fmt_p(p):
    return "$<10^{-15}$" if p < 1e-15 else (f"${p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}$" if p < 1e-3 else f"{p:.3f}")


# ------------------------------------------------------------------ data overview
cfg = yaml.safe_load(open("config/test_real_events.yaml"))
ev = pd.read_csv("paper/data/real_events_list.csv")
real = pd.read_json("paper/data/real_events_eval_rows.jsonl", lines=True, precise_float=True)
raw_real = real.copy()
real.loc[real.lost_routing_frac < TOL, "lost_routing_frac"] = 0.0
N["noise_rows"] = int(((raw_real.lost_routing_frac > 0) & (raw_real.lost_routing_frac < TOL)).sum())
N["min_genuine_loss"] = float(raw_real.lost_routing_frac[raw_real.lost_routing_frac >= TOL].min())

cnt = ev.pivot_table(index="topology", columns=["split"], values="scenario_id", aggfunc="count", fill_value=0)
prim_ev = ev[ev.threshold.isin(["34kt", "mmi7"])]
cnt2 = prim_ev.pivot_table(index="topology", columns=["split", "family"], values="scenario_id", aggfunc="count", fill_value=0)

# ------------------------------------------------------------------ real-event test (primary)
prim = real[real.threshold.isin(["34kt", "mmi7"])]
piv = prim.pivot_table(index=["topology", "scenario_id"], columns="rule", values="lost_routing_frac")
N["real_n"] = len(piv)
N["real_events_distinct"] = int(len({s.rsplit("_", 1)[0] for _, s in piv.index}))
summ = pd.DataFrame({r: dict(mean=100 * piv[r].mean(), p95=100 * piv[r].quantile(0.95), cvar=100 * cvar(piv[r]),
                             worst=100 * piv[r].max(), zero=(piv[r] == 0).mean()) for r in CLASSICAL + [T, "ksp_hops_k8"]}).T
N["real_summary"] = summ.round(4).to_dict()
N["real_paired"] = paired(piv, CLASSICAL, "tol")
rp = raw_real[raw_real.threshold.isin(["34kt", "mmi7"])].pivot_table(index=["topology", "scenario_id"], columns="rule", values="lost_routing_frac")
N["real_paired_preregistered_raw"] = paired(rp, CLASSICAL, "raw")

# event-clustered
e = piv.copy(); e["event"] = [s.rsplit("_", 1)[0] for _, s in e.index]
em = e.groupby("event").mean(numeric_only=True)
N["clustered"] = paired(em, CLASSICAL, "tol")
# stricter thresholds
sec = real[real.threshold.isin(["64kt", "mmi8"])].pivot_table(index=["topology", "scenario_id"], columns="rule", values="lost_routing_frac")
N["strict_n"] = len(sec)
N["strict_paired"] = paired(sec, CLASSICAL, "tol")
N["strict_summary"] = {r: 100 * sec[r].mean() for r in CLASSICAL + [T]}
# earthquakes
eq = real[real.family == "earthquake"]
N["eq_n_scenarios"] = int(eq.scenario_id.nunique()) if "family" in eq else 0
N["eq_max_routable_loss"] = float(eq.lost_routing_frac.max()) if len(eq) else 0.0
# unservable share
N["real_unservable_mean_pct"] = float(100 * prim[prim.rule == T].lost_unservable_frac.mean())
N["real_routable_mean_pct_dart"] = float(100 * prim[prim.rule == T].lost_routing_frac.mean())
# per topology
pt = 100 * prim.groupby(["topology", "rule"]).lost_routing_frac.mean().unstack()
N["real_per_topology_mean_pct"] = pt.round(4).to_dict(orient="index")

# ------------------------------------------------------------------ held-out disk test
dk = pd.read_csv("paper/data/final_test_rows.csv.gz")
dk.loc[dk.lost_routing_frac < TOL, "lost_routing_frac"] = 0.0
disk = dk[dk.kind == "disk"]
N["disk"] = {}
for top, g in disk.groupby("topology"):
    p = g.pivot_table(index="scenario_id", columns="rule", values="lost_routing_frac")
    N["disk"][top] = {"n": len(p), "summary": {r: dict(mean=100 * p[r].mean(), p95=100 * p[r].quantile(0.95),
                      cvar=100 * cvar(p[r]), worst=100 * p[r].max()) for r in CLASSICAL + [T, "ksp_hops_k8"]},
                      "paired": paired(p, CLASSICAL, "tol")}
nf = dk[dk.kind == "adversarial_links"]
N["cuts"] = {top: {r: dict(p95=100 * g[g.rule == r].lost_routing_frac.quantile(0.95), worst=100 * g[g.rule == r].lost_routing_frac.max())
                   for r in CLASSICAL + [T]} for top, g in nf.groupby("topology")}

json.dump(N, open("paper/numbers.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ LaTeX tables
def w(name, s):
    open(f"paper/tables/{name}.tex", "w").write(s); print("wrote table", name)

# Table: data set (topologies and event counts)
import sys
sys.path.insert(0, ".")
from dte.run import load_topologies
tops = {t.name: t for t in load_topologies(cfg)}
lines = []
for t in ["attmpls", "btnorthamerica", "geant2012", "iij", "cernet", "garr201201", "grnet"]:
    tr = prim_ev[(prim_ev.topology == t) & (prim_ev.split == "train")]
    te = prim_ev[(prim_ev.topology == t) & (prim_ev.split == "test")]
    lines.append(f"{TOPO[t]} & {tops[t].n} & {len(tops[t].links)} & {(tr.family=='hurricane').sum()} & {(tr.family=='earthquake').sum()} & "
                 f"{(te.family=='hurricane').sum()} & {(te.family=='earthquake').sum()} \\\\")
w("tab_data", "\\begin{tabular}{lrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{Training (before 2018)} & \\multicolumn{2}{c}{Test (2018--2025)} \\\\\n"
  "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nNetwork & Nodes & Links & Cyclones & Earthquakes & Cyclones & Earthquakes \\\\\n\\midrule\n"
  + "\n".join(lines) + f"\n\\midrule\nTotal & & & {int((prim_ev[prim_ev.split=='train'].family=='hurricane').sum())} & {int((prim_ev[prim_ev.split=='train'].family=='earthquake').sum())} & "
  f"{int((prim_ev[prim_ev.split=='test'].family=='hurricane').sum())} & {int((prim_ev[prim_ev.split=='test'].family=='earthquake').sum())} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# Table: real-event results
pr = {x["rule"]: x for x in N["real_paired"]}
rows = []
for r in ["ksp_hops_k4", "geo_diverse_k4", "edge_disjoint_k4", "ksp_hops_k8", T]:
    s = N["real_summary"]
    base = f"{NAME[r]} & {s['mean'][r]:.2f} & {s['p95'][r]:.2f} & {s['cvar'][r]:.2f} & {s['worst'][r]:.2f}"
    if r in pr:
        x = pr[r]
        base += f" & {x['mean_diff_pp']:+.2f} [{x['ci_lo_pp']:+.2f}, {x['ci_hi_pp']:+.2f}] & {x['better']}/{x['worse']}/{x['tie']} & {fmt_p(x['p_holm'])} & {x['r_rb']:.2f}"
    else:
        base += " & -- & -- & -- & --"
    rows.append(base + " \\\\")
w("tab_real", "\\begin{tabular}{lrrrrlrrr}\n\\toprule\n & \\multicolumn{4}{c}{Routable loss (\\% of demand)} & \\multicolumn{4}{c}{DART minus rule (paired)} \\\\\n"
  "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nRule ($k=4$ unless stated) & Mean & P95 & CVaR$_{0.95}$ & Worst & Mean diff. [95\\% CI] & B/W/T & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$ \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# Table: held-out disks
rows = []
for t in ["attmpls", "btnorthamerica", "geant2012"]:
    D = N["disk"][t]; pr = {x["rule"]: x for x in D["paired"]}
    for r in ["ksp_hops_k4", "geo_diverse_k4", "edge_disjoint_k4", T]:
        s = D["summary"][r]
        tail = f" & {pr[r]['mean_diff_pp']:+.2f} & {fmt_p(pr[r]['p_holm'])} & {pr[r]['r_rb']:.2f}" if r in pr else " & -- & -- & --"
        lead = f"\\multirow{{4}}{{*}}{{{TOPO[t]} ($n={D['n']}$)}}" if r == "ksp_hops_k4" else ""
        rows.append(f"{lead} & {NAME[r]} & {s['mean']:.2f} & {s['cvar']:.2f} & {s['worst']:.2f}{tail} \\\\")
    rows.append("\\midrule" if t != "geant2012" else "")
w("tab_disk", "\\begin{tabular}{llrrrrrr}\n\\toprule\nNetwork & Rule & Mean (\\%) & CVaR$_{0.95}$ (\\%) & Worst (\\%) & Mean diff. (pp) & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$ \\\\\n\\midrule\n"
  + "\n".join(x for x in rows if x) + "\n\\bottomrule\n\\end{tabular}\n")
print(json.dumps({k: N[k] for k in ["noise_rows", "min_genuine_loss", "real_n", "real_events_distinct", "strict_n", "eq_n_scenarios", "eq_max_routable_loss", "real_unservable_mean_pct", "real_routable_mean_pct_dart"]}, indent=1))
for x in N["real_paired"]: print("real", x)
for x in N["clustered"]: print("clustered", x)
for x in N["strict_paired"]: print("strict", x)
