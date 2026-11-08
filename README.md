# DART: disaster-aware tunnel selection for traffic-engineering recovery

Code, data pipeline, pre-registration records and manuscript for

> U. P. Singh and B. Kumari, *Choosing backup paths from hazard history: disaster-aware path
> selection for traffic-engineering recovery, tested on real cyclones and earthquakes*
> (manuscript in preparation for *Computer Networks*).

**DART** (Disaster-Aware, Risk-learned Tunnel selection) chooses the `k` precomputed tunnels per
demand pair that a traffic-engineering controller can use after a failure. It keeps `k-1`
link-disjoint tunnels and replaces the last one only when real past hazard footprints show that the
replacement would have survived more disasters (`dte/pathsel.py: select_dart_v4`). After a disaster,
split ratios are re-solved over the surviving tunnels with a min-MLU linear program (`dte/te.py`).

## Main results (held-out tests, criteria committed before each test)

| Test | Data | DART mean routable loss | Best classical rule (mean) |
|---|---|---|---|
| 1: unseen networks | 800 random disks x 3 networks (AttMpls, BtNorthAmerica, GEANT) | 0.32-0.48 % | 0.43-0.92 % |
| 2: real events | 211 cyclone/earthquake footprints, 2018-2025, 7 networks | 0.16 % | 0.30 % (edge-disjoint) |

In Test 2 DART and the edge-disjoint rule tied in 162 events; DART lost less in 47 and more in 2
(Holm-corrected Wilcoxon p = 8.6e-8). The absolute gains are small: endpoints isolated or destroyed
by the disaster account for a mean loss of 12.4 % that no routing can recover, and real earthquakes
broke too few backbone links to separate any rule. The first version of the method **failed** its
pilot; the full development history is in `reports/` and Supplementary Section S1.

## Real data used

| Source | What | Where in code |
|---|---|---|
| Internet Topology Zoo via TopoHub; SNDlib | real backbone topologies with coordinates | `dte/data.py` |
| USGS ComCat / ShakeMap `grid.xml` | earthquake intensity footprints | `dte/events.py`, `dte/failures.py` |
| NOAA HURDAT2 | Atlantic hurricane tracks and wind radii | `dte/failures.py` |
| NCEI IBTrACS v04r01 (WP) | western Pacific typhoon tracks and wind radii | `dte/events.py` |

Assumptions (uniform capacities and gravity demand on Topology Zoo networks, great-circle links,
"inside footprint = failed") are listed in the paper's limitations and swept where possible.

## Reproduce

Python 3.12+. One-time setup (virtualenv, validation tests, data download with checksums):

```bash
bash scripts/setup_server.sh config/stage2_real_disasters.yaml config/test_real_events.yaml
```

Test 1 (unseen networks):

```bash
.venv/bin/python -m dte.pilot --config config/test_dart_v4.yaml --workers 32 --out results/t1
```

Test 2 (real events), event selection then evaluation:

```bash
.venv/bin/python -m dte.realtest build --config config/test_real_events.yaml --out results/t2
```

```bash
.venv/bin/python -m dte.realtest evaluate --config config/test_real_events.yaml --out results/t2 --workers 32
```

Paper numbers, tables and figures (from the committed data in `paper/data/`):

```bash
PYTHONPATH=. .venv/bin/python paper/analysis.py && PYTHONPATH=. .venv/bin/python paper/supp_tables.py && PYTHONPATH=. .venv/bin/python paper/figures.py
```

Long runs in the background, several configs in parallel:

```bash
bash scripts/launch_nohup.sh 32 config/paths_k4.yaml config/paths_k8.yaml
```

## Layout

```
dte/            implementation (data loading, LP recovery, footprints, DART, runners)
config/         every experimental parameter; pre-registered test configs
tests/          validation against hand-computed answers (pytest)
reports/        verbatim pilot, development and test reports (including the failed pilot)
paper/          manuscript (main.tex/pdf), supplementary, figures, tables, analysis scripts, data
CLAUDE.md       manuscript-writing rules used for this paper
```

## Pre-registration trail

The development history, criteria and outcomes were first committed in
`singhudaypratap/nipa-book-chapter` (branch `claude/awesome-gauss-9xxnge`, directory `disaster-te`);
commit identifiers in Supplementary Table S2 refer to that repository.

## License

Apache License 2.0 (see `LICENSE`).
