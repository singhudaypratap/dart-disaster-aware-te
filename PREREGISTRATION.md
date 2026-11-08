# Pre-registration trail

The full development history is in the branch [`preregistration-history`](https://github.com/singhudaypratap/dart-disaster-aware-te/commits/preregistration-history)
of this repository. It was mirrored from the development repository
[`singhudaypratap/nipa-book-chapter`](https://github.com/singhudaypratap/nipa-book-chapter)
(branch `claude/awesome-gauss-9xxnge`, directory `disaster-te`) with `git subtree split`, which keeps
each commit's message, author and timestamp but gives it a new identifier. The original pushes, with
GitHub's own record of when they happened, remain in the development repository.

Supplementary Table S2 of the manuscript cites the **original** identifiers. The table below maps each
one to its identifier in the `preregistration-history` branch.

| Date (commit time) | Original id | `preregistration-history` id | Role | Commit message |
|---|---|---|---|---|
| 2026-11-07 18:46 | `ebdfc43` | `4f5cfeb` |  | Add disaster-te phase 1: real-disaster TE evaluation harness |
| 2026-11-07 18:52 | `7763132` | `016cfa8` |  | Add nohup parallel launcher, one-time setup script and path-count sweep configs |
| 2026-11-07 19:10 | `c8bd6d1` | `749f9ce` |  | Stage 2a: real Topology Zoo networks, 15 earthquakes and 21 hurricanes, faster solvers |
| 2026-11-08 03:17 | `88d30a8` | `7d73083` | Pilot criterion | Add DART-opt path selection pilot with pre-registered success criterion |
| 2026-11-08 03:32 | `ac90413` | `112a54d` | Pilot outcome: FAIL | Record DART-opt pilot outcome: pre-registered criterion FAILED on both topologies |
| 2026-11-08 03:38 | `29f2244` | `c8db858` |  | Add DART v2 (disjoint backbone + learned risk coverage) and development config |
| 2026-11-08 03:59 | `0acf5de` | `8e07af7` |  | Add DART v3 (link-disjoint backbone of k-1 paths + learned-risk final slot), dev config |
| 2026-11-08 04:21 | `9da2e8f` | `08c1e36` |  | Add DART v4: edge-disjoint paths, last path replaced only when hazard data shows benefit |
| 2026-11-08 04:45 | `0cfd320` | `2c5a72d` | Test 1 criterion; method frozen | Freeze DART v4 and pre-register the final one-shot test on untouched topologies |
| 2026-11-08 05:34 | `bcc202f` | `69b319d` | Test 1 outcome: PASS | Record final one-shot test of frozen DART v4: pre-registered criterion PASSED on all 3 untouched topologies |
| 2026-11-08 07:46 | `16bcd66` | `988e8ed` |  | Add systematic real-event selection (IBTrACS WP, USGS catalogue) and real-event test runner |
| 2026-11-08 08:02 | `269fa4f` | `064b088` | Test 2 event-selection fix (before evaluation) | Explicit missing-radii policy for cyclone footprints (error / drop_rows with 12 h gap limit) |
| 2026-11-08 08:11 | `6594535` | `ad6f7f7` | Test 2 criterion | Pre-register the pooled real-event test of frozen DART v4 (before evaluation) |
| 2026-11-08 09:08 | `d8f7e5e` | `4c74fac` | Test 2 outcome: PASS | Record pooled real-event test of frozen DART v4: pre-registered criterion PASSED |
| 2026-11-08 10:40 | `d172304` | `a032c24` |  | Manuscript materials: verified bibliography, figure and analysis scripts, committed result data |
| 2026-11-08 10:47 | `a56f50d` | `cfceebd` |  | Draft manuscript and supplementary material (Computer Networks, elsarticle) |
| 2026-11-08 12:59 | `cea1791` | `d3c3421` |  | Manuscript: AEGIS positioned from its abstract, funding/AI/data statements filled, Holm confirmed |
| 2026-11-08 13:03 | `365d3c2` | `fda3d45` |  | Manuscript: describe PathGNN exactly as its abstract states (24.4% load balancing under dynamic traffic; up to 35.3% performance under multiple link failures) |
| 2026-11-08 13:05 | `3827eb3` | `ebf54e8` |  | Manuscript: add PathGNN's reported lower end-to-end delay under dynamic traffic |
| 2026-11-08 13:26 | `6b21f4c` | `de268c3` |  | Supplementary: note that the pre-registration history is mirrored in dart-disaster-aware-te |
