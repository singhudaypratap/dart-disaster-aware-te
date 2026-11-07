# disaster-te: traffic-engineering recovery under real disasters and adversarial link cuts

Research code for the follow-up to EAAI-26-25931. It evaluates how well traffic-engineering (TE)
recovery methods reroute traffic on **real backbone topologies** after **real historical disasters**
(USGS ShakeMap earthquake intensity grids, NOAA HURDAT2 hurricane wind radii) and after
**worst-case link cuts** chosen by an adversary, against the exact LP optimum.

Status: **phase 1 (classical baselines and the evaluation harness) is complete and validated.**
Phase 2 adds the learned methods (proposed model, PathGNN reimplementation, rule distillation).

## Run on the server

```bash
git clone -b claude/awesome-gauss-9xxnge https://github.com/singhudaypratap/nipa-book-chapter.git
cd nipa-book-chapter/disaster-te
bash scripts/run_server.sh config/experiment.yaml 32      # 32 = parallel worker processes
```

The script creates a virtualenv with pinned versions, runs the validation tests, downloads and
checksums every input, runs all jobs in parallel and writes `results/<run>/summary.csv`.
Phase 1 on Abilene: 4,182 jobs x 5 methods, about 20 s on 4 cores.

### Background runs with nohup, several configs in parallel

```bash
bash scripts/setup_server.sh config/paths_k4.yaml config/paths_k8.yaml        # once, and after each git pull
bash scripts/launch_nohup.sh 32 config/paths_k4.yaml config/paths_k8.yaml     # 32 workers split across configs
tail -f logs/*.log                                                            # watch; safe to log out
```

Each config needs a distinct `run.name`; each writes its own `results/<run.name>_<time>/` and
`logs/<config>_<time>.log`, which ends with `FINISHED <run dir>` on success. PIDs are in
`logs/pids_<time>.txt` for stopping runs.

## What is real and what is assumed

| Input | Source | Real / assumed |
|---|---|---|
| Abilene topology, capacities, coordinates | SNDlib `abilene.xml` (pinned mirror + sha256) | real |
| Demand matrix | Yin Zhang Abilene TM as packaged by SNDlib (scaled peak, 2004) | real pattern; absolute level rescaled (see below) |
| Earthquakes | USGS ShakeMap `grid.xml`, events listed in config | real recorded/estimated shaking |
| Hurricanes | NOAA HURDAT2 best track + wind radii, storms listed in config | real |
| Damage rule | node/link fails if inside footprint (MMI >= threshold; inside wind radius) | **assumption**, swept (MMI 7/8; 34/64 kt) |
| Link geometry | great-circle arc between endpoints | **assumption** (real fibre follows roads/rail) |
| Demand level | rescaled so optimal no-failure MLU = 0.4 / 0.6 / 0.8 | **assumption**, swept |
| Adversary | all k-link cuts enumerated, k = 1..3 | exact worst case for each method |

## Methods (phase 1)

* `optimal` -- exact min-MLU over all fractional routings (edge LP). Lower bound for everything.
* `te_resolve_kpaths` -- controller re-solves split ratios over the surviving k precomputed paths.
* `local_rescale` -- pre-failure splits renormalised over surviving paths (equal split over
  surviving backups if all used paths failed); no controller computation.
* `ospf_ecmp` -- OSPF re-convergence, inverse-capacity weights, hop-by-hop ECMP.
* `no_recovery` -- traffic on failed paths is dropped.

## Metrics: always read them together

* `mlu_ratio_to_optimal` -- method MLU / optimal MLU on the same scenario.
* `lost_routing_frac` -- demand that was routable but the method dropped.
* `lost_unservable_frac` -- demand no routing can carry (dead endpoint or disconnected); equal for all methods.

A method that drops traffic carries less and can show MLU *below* optimal (e.g. `no_recovery`).
MLU must never be reported without `lost_routing_frac`.

## Statistical analysis plan (fixed before phase-2 results exist)

* Unit of analysis: scenario (paired across methods: every method sees the same scenarios).
* Primary comparison: proposed method vs `te_resolve_kpaths` and vs PathGNN, per topology x load x
  scenario family, on `mlu_ratio_to_optimal` and `lost_routing_frac`.
* Test: two-sided Wilcoxon signed-rank (paired, non-normal ratios); Holm correction across all
  primary comparisons; effect size = median paired difference with 95% bootstrap CI.
* Adversarial families are reported as worst case (max over cuts), not tested.
* Real-event families have few scenarios (n = 1..10); they are reported per event, not tested.

## Layout

```
config/experiment.yaml   every experimental parameter (no defaults in code)
dte/data.py              SNDlib loader (fails loudly on missing capacities/units)
dte/te.py                edge LP, path LP, ECMP, recovery evaluation
dte/failures.py          ShakeMap / HURDAT2 / disk / k-cut scenarios
dte/fetch.py             downloads + checksums -> data/raw/MANIFEST.json
dte/run.py               parallel runner + provenance manifest
dte/aggregate.py         summary tables with bootstrap CIs
tests/                   validation against hand-computed answers
```
