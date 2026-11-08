"""Summarise a run directory into tables.

    python -m dte.aggregate results/<run_dir> --bootstrap 2000 --seed 1

Writes summary.csv (one row per topology x load x scenario family x method) and prints it.
Scenario families: real earthquakes (by MMI threshold), real hurricanes (by wind class),
random disks (by radius), adversarial k-link cuts (worst case = max over all cuts).

Descriptive statistics only. Confirmatory tests between methods are pre-specified separately
(see README, "Statistical analysis plan") and run once the proposed method exists.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def family(row) -> str:
    if row["kind"] == "shakemap":
        return f"earthquake_mmi{row['scenario_id'].rsplit('mmi', 1)[1]}"
    if row["kind"] == "hurricane":
        return f"hurricane_{row['scenario_id'].rsplit('_', 1)[1]}"
    if row["kind"] == "disk":
        return f"disk_r{row['scenario_id'].split('_')[1][1:]}km"
    if row["kind"] == "adversarial_links":
        return f"cut_k{len(row['scenario_id'].split('_')[1].split('-'))}"
    return row["kind"]


def boot_ci_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarise(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for (topo, load, fam, m), g in df.groupby(["topology", "target_mlu", "family", "method"], sort=True):
        r = g["mlu_ratio_to_optimal"].to_numpy()
        lr = g["lost_routing_frac"].to_numpy()
        lo, hi = boot_ci_mean(r, n_boot, rng)
        out.append({"topology": topo, "target_mlu": load, "family": fam, "method": m, "n_scenarios": len(g),
                    "mlu_ratio_mean": r.mean(), "mlu_ratio_ci95_lo": lo, "mlu_ratio_ci95_hi": hi,
                    "mlu_ratio_median": np.median(r), "mlu_ratio_p95": np.quantile(r, 0.95),
                    "mlu_ratio_worst": r.max(),
                    "lost_routing_mean": lr.mean(), "lost_routing_worst": lr.max(),
                    "lost_unservable_mean": g["lost_unservable_frac"].mean(),
                    "frac_scen_mlu_over_1": float((g["mlu"] > 1.0).mean()),
                    "runtime_ms_median": 1000 * g["runtime_s"].median()})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bootstrap", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    df = pd.read_json(os.path.join(a.run_dir, "results.jsonl"), lines=True, precise_float=True)
    df["family"] = df.apply(family, axis=1)
    s = summarise(df, a.bootstrap, a.seed)
    s.to_csv(os.path.join(a.run_dir, "summary.csv"), index=False)
    with open(os.path.join(a.run_dir, "summary_args.json"), "w") as f:
        json.dump(vars(a), f)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
