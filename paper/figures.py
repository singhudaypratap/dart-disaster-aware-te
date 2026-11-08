"""Generate every manuscript figure from committed result files (run from disaster-te/):

    .venv/bin/python paper/figures.py

Style rules (CLAUDE.md): presentation style, black/grey only, series separated by line style /
marker / hatch, sans-serif 12 pt at final printed size (Liberation Sans = metric-compatible Arial),
vector PDF. Figures are drawn at their final width (16 cm) so 12 pt here is 12 pt in print.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from dte import events as E
from dte import failures as F
from dte.data import Topology
from dte.run import load_topologies
import yaml

CM = 1 / 2.54
W = 16 * CM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"], "font.size": 12,
    "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.titlesize": 12, "axes.edgecolor": "black", "axes.linewidth": 0.8, "lines.color": "black",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "hatch.linewidth": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
})
OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)
RULES = ["ksp_hops_k4", "geo_diverse_k4", "edge_disjoint_k4", "dart_v4_k4"]
LABEL = {"ksp_hops_k4": "Shortest paths", "geo_diverse_k4": "Geo-diverse", "edge_disjoint_k4": "Edge-disjoint",
         "dart_v4_k4": "DART", "ksp_hops_k8": "Shortest paths (k = 8)"}
STYLE = {"ksp_hops_k4": dict(ls=":", marker="v"), "geo_diverse_k4": dict(ls="-.", marker="s"),
         "edge_disjoint_k4": dict(ls="--", marker="o"), "dart_v4_k4": dict(ls="-", marker="D")}
HATCH = {"ksp_hops_k4": "....", "geo_diverse_k4": "xxx", "edge_disjoint_k4": "///", "dart_v4_k4": ""}
FILL = {"ksp_hops_k4": "white", "geo_diverse_k4": "white", "edge_disjoint_k4": "white", "dart_v4_k4": "0.25"}
TOPO = {"attmpls": "AttMpls", "btnorthamerica": "BtNorthAmerica", "geant2012": "GÉANT", "iij": "IIJ",
        "cernet": "CERNET", "garr201201": "GARR", "grnet": "GRNET"}


TOL = 1e-9   # LP rounding noise threshold (see paper/analysis.py)


def load_real():
    df = pd.read_json("paper/data/real_events_eval_rows.jsonl", lines=True, precise_float=True)
    df.loc[df.lost_routing_frac < TOL, "lost_routing_frac"] = 0.0
    return df[df.threshold.isin(["34kt", "mmi7"])]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1: method pipeline
def fig_pipeline():
    """Two columns: offline planning (left), online recovery (right). The single link between
    them is the tunnel set, drawn as one horizontal arrow."""
    fig, ax = plt.subplots(figsize=(W, 11.5 * CM))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11.6); ax.axis("off")
    bw, bh = 6.4, 2.2                       # box width / height (axes units = cm)
    xl, xr = 0.1, 9.5                       # left / right column x
    ys = [8.2, 5.5, 2.8, 0.1]               # bottoms of rows 1-4

    def box(x, y, text, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=fill, ec="black", lw=1.0))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=12, linespacing=1.2)

    def down(x, y_from, y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0))

    ax.text(xl + bw / 2, 11.25, "Offline (planning)", ha="center", fontsize=12, fontweight="bold")
    ax.text(xr + bw / 2, 11.25, "Online (after a disaster)", ha="center", fontsize=12, fontweight="bold")

    box(xl, ys[0], "Training scenarios:\nreal events before 2018\nand random disks")
    box(xl, ys[1], "Backbone: $k-1$ link-\ndisjoint shortest paths\nper demand pair")
    box(xl, ys[2], "Last tunnel: replaced\nonly if it survives more\ntraining scenarios", fill="0.88")
    for r in range(2):
        down(xl + bw / 2, ys[r], ys[r + 1] + bh)

    box(xr, ys[0], "Disaster strikes")
    box(xr, ys[1], "Failed links and nodes\nare observed")
    box(xr, ys[2], "Drop the failed tunnels\nof each demand pair")
    box(xr, ys[3], "Re-solve split ratios\n(min-MLU linear program)\nover surviving tunnels")
    for r in range(3):
        down(xr + bw / 2, ys[r], ys[r + 1] + bh)

    yc = ys[2] + bh / 2                      # horizontal link: tunnel set -> online drop step
    ax.annotate("", xy=(xr, yc), xytext=(xl + bw, yc),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0))
    ax.text((xl + bw + xr) / 2, yc + 0.18, "$k$ tunnels\nper pair", ha="center", va="bottom", fontsize=12)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ Fig. 2: real footprints on real networks
def _coast(ax, lon0, lon1, lat0, lat1):
    gj = json.load(open("paper/data/ne_50m_coastline.geojson"))
    for f in gj["features"]:
        g = f["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lines:
            a = np.asarray(ln)
            if ((a[:, 0] > lon0 - 5) & (a[:, 0] < lon1 + 5) & (a[:, 1] > lat0 - 5) & (a[:, 1] < lat1 + 5)).any():
                ax.plot(a[:, 0], a[:, 1], color="0.6", lw=0.6, zorder=1)


def _footprint_mask(storm_id, storm, kt, lon, lat):
    """Evaluate the same footprint rule as the experiments on a lon/lat grid (pseudo-topology)."""
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    T = Topology("grid", tuple(map(str, range(len(pts)))), pts, ((0, 1),), np.ones(1),
                 np.zeros((len(pts), len(pts))), "grid")
    sc = F.hurricane_scenario(T, storm_id, storm, kt, 1.0, 10.0, missing_radii="drop_rows", max_gap_hours=12.0)
    return sc.dead_node.reshape(LAT.shape)


def fig_map():
    cfg = yaml.safe_load(open("config/test_real_events.yaml"))
    topos = {t.name: t for t in load_topologies(cfg)}
    ev = pd.read_csv("paper/data/real_events_list.csv")
    hur = F.parse_hurdat2(open("data/raw/events/hurdat2-1851-2025-092326.txt").read())
    for k, v in hur.items():
        v["season"] = int(k[-4:])
    ib = E.parse_ibtracs("data/raw/events/ibtracs.WP.list.v04r01.csv", 2001)
    storms = {**hur, **ib}
    scen = pickle.load(open("results/test_real_events/scenarios.pkl", "rb"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 9.5 * CM))
    for ax, tname, box_ in zip(axes, ["attmpls", "iij"], [(-100, -66, 24, 46), (127, 146, 29, 45)]):
        t = topos[tname]
        cand = ev[(ev.topology == tname) & (ev.split == "test") & (ev.threshold == "34kt")]
        row = cand.sort_values(["n_dead_links", "scenario_id"], ascending=[False, True]).iloc[0]
        sc = next(s for s in scen[tname] if s.scenario_id == row.scenario_id)
        sid = sc.meta["storm_id"]
        lon = np.linspace(box_[0], box_[1], 180); lat = np.linspace(box_[2], box_[3], 140)
        m = _footprint_mask(sid, storms[sid], 34, lon, lat)
        ax.contourf(lon, lat, m.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"], zorder=0)
        ax.contour(lon, lat, m.astype(float), levels=[0.5], colors="black", linewidths=0.8, zorder=2)
        _coast(ax, *box_)
        for li, (i, j) in enumerate(t.links):
            (x0, y0), (x1, y1) = t.lonlat[i], t.lonlat[j]
            if x1 - x0 > 180: x1 -= 360          # draw trans-Pacific links the short way round
            elif x0 - x1 > 180: x1 += 360
            dead = sc.dead_link[li] or sc.dead_node[i] or sc.dead_node[j]
            ax.plot([x0, x1], [y0, y1], color="black", lw=2.2 if dead else 0.8, ls="--" if dead else "-", zorder=3)
        ax.scatter(t.lonlat[:, 0], t.lonlat[:, 1], s=14, facecolor="white", edgecolor="black", lw=0.8, zorder=4)
        dn = sc.dead_node
        ax.scatter(t.lonlat[dn, 0], t.lonlat[dn, 1], s=30, facecolor="black", edgecolor="black", zorder=5)
        name = storms[sid]["name"].title()
        ax.set_title(f"{'(a)' if tname == 'attmpls' else '(b)'} {TOPO[tname]}, {name} ({storms[sid]['season']})", fontsize=12)
        ax.set_xlim(box_[0], box_[1]); ax.set_ylim(box_[2], box_[3])
        ax.set_xlabel("Longitude (°)"); ax.set_aspect(1 / np.cos(np.radians(np.mean(box_[2:]))))
        print(f"map {tname}: {row.scenario_id} {name} {storms[sid]['season']} failed links {row.n_dead_links} nodes {row.n_dead_nodes}")
    axes[0].set_ylabel("Latitude (°)")
    save(fig, "fig2_footprints")


# ------------------------------------------------------------------ Fig. 3: real-event loss distribution
def fig_real_exceedance():
    df = load_real()
    fig, ax = plt.subplots(figsize=(W, 8 * CM))
    for r in RULES:
        x = np.sort(100 * df[df.rule == r].lost_routing_frac.values)
        n = len(x)
        xs = np.unique(x[x > 0])
        exc = np.array([(x >= v).mean() for v in xs])
        ax.step(np.r_[xs, xs[-1] * 1.3], np.r_[exc, 0.5 / n], where="post", color="black", ls=STYLE[r]["ls"], lw=1.4,
                label=f"{LABEL[r]} ({100 * (x == 0).mean():.0f}% of events: no loss)")
        ax.plot(xs[::max(1, len(xs) // 6)], exc[::max(1, len(xs) // 6)], ls="none", color="black",
                marker=STYLE[r]["marker"], ms=5, mfc="white" if r != "dart_v4_k4" else "black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Routable traffic lost (% of total demand)")
    ax.set_ylabel("Fraction of events with\nloss $\\geq$ x")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, which="major", color="0.85", lw=0.5)
    save(fig, "fig3_real_exceedance")


# ------------------------------------------------------------------ Fig. 4: per-topology means (real events)
def _bars(ax, table, ylabel):
    tops = list(table.index)
    x = np.arange(len(tops)); w = 0.2
    for k, r in enumerate(RULES):
        ax.bar(x + (k - 1.5) * w, table[r].values, width=w, facecolor=FILL[r], edgecolor="black",
               hatch=HATCH[r], lw=0.8, label=LABEL[r])
    ax.set_xticks(x); ax.set_xticklabels([TOPO[t] for t in tops], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2, loc="upper right")


def fig_real_by_topology():
    df = load_real()
    tab = 100 * df.groupby(["topology", "rule"]).lost_routing_frac.mean().unstack()
    n = df[df.rule == "dart_v4_k4"].groupby("topology").size()
    # GRNET (n = 2) and GARR (n = 1) have zero routable loss for every rule; reported in the text
    order = ["attmpls", "btnorthamerica", "iij", "cernet", "geant2012"]
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(W, 8.5 * CM))
    _bars(ax, tab, "Mean routable loss (%)")
    short = {"btnorthamerica": "BtNorth-\nAmerica"}
    ax.set_xticklabels([f"{short.get(t, TOPO[t])}\n(n = {n[t]})" for t in order], rotation=0, ha="center")
    ax.set_ylim(0, tab.values.max() * 1.45)
    save(fig, "fig4_real_by_topology")


# ------------------------------------------------------------------ Fig. 5: held-out disks, CVaR95
def fig_disks():
    d = pd.read_csv("paper/data/final_test_rows.csv.gz")
    d.loc[d.lost_routing_frac < TOL, "lost_routing_frac"] = 0.0
    d = d[d.kind == "disk"]
    cvar = lambda x: np.sort(np.asarray(x))[-max(1, int(np.ceil(0.05 * len(x)))):].mean()
    tab = 100 * d.groupby(["topology", "rule"]).lost_routing_frac.agg(cvar).unstack()
    tab = tab.loc[["attmpls", "btnorthamerica", "geant2012"]]
    fig, ax = plt.subplots(figsize=(W, 8 * CM))
    _bars(ax, tab, "CVaR$_{0.95}$ of routable\nloss (%)")
    ax.set_xticklabels([TOPO[t] for t in tab.index], rotation=0, ha="center")
    ax.set_ylim(0, tab.values.max() * 1.45)
    save(fig, "fig5_disk_cvar")


if __name__ == "__main__":
    fig_pipeline(); fig_map(); fig_real_exceedance(); fig_real_by_topology(); fig_disks()
