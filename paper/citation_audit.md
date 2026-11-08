# Citation-to-claim audit (2026-11-08)

Every sentence of `main.tex` that contains `\cite{}` (88 citation pairs, Table 1 included) was checked
against the cited work's own abstract or primary record, fetched during the audit from OpenAlex,
Semantic Scholar, Crossref or the official USENIX paper page. Foundational method/data citations were
checked only for being the correct source of the method or data set named.

Verdicts: **S** supported, **P** partly supported (sentence rewritten to match the source),
**N** not supported (citation removed or replaced), **U** unverifiable (no abstract available anywhere).

| Ref key | Where | Claim (short) | Evidence (source) | Verdict | Action |
|---|---|---|---|---|---|
| ffc14, teavar19 | Intro ¶1 | TE "re-optimises" over a small set of precomputed tunnels | FFC is proactive (spreads traffic for up to k faults); TeaVaR abstract does not mention tunnels (OpenAlex) | P | Re-cited to prete25 ("traffic allocation among updated tunnels") and aegis26 ("traditional TE relies on pre-computed paths"); sentence reworded |
| neumayer11, agarwal13 | Intro ¶2 | correlated failures studied >10 years | geographic line/disk cuts; probabilistic geographic failures in WDM (OpenAlex) | S | kept; wording made specific (vulnerability analysis) |
| habib13 | Intro ¶2, Rel. work | survey of disaster protection | no abstract in OpenAlex, Semantic Scholar or Crossref | U | removed |
| giraosilva20 | Intro ¶2, contrib. 2, Rel. work, setup, Table 1 | SRLG-disjoint geodiverse protection; path-length/transponder cost vs link-disjointness | abstract states exactly this (OpenAlex) | S | kept; contribution-2 contrast reworded |
| jimenez23 (new) | Intro ¶2, Rel. work, Limitations, Table 1 | seismic reliability with PSHA and fragility curves | "Probabilistic Seismic Hazard Analysis ... nodes fail according to fragility curves" (OpenAlex) | S | added |
| dote23 | Intro ¶2, Rel. work, Table 1 | learns/optimises TE from historical demands | "directly optimizing traffic flow ... using only historical data about traffic demands" (USENIX) | P (intro) / S | intro wording "learn split ratios" → "optimise TE directly from historical traffic demands" |
| teal23 | Intro ¶2, Rel. work, Table 1 | GNN + multi-agent RL to accelerate TE | "flow-centric GNN ... multi-agent RL ... ADMM" (Semantic Scholar) | S | kept |
| harp24 | Intro, Rel. work, Table 1, Limitations | handles topology changes incl. failures | "robust to topology changes ... WAN evolution, failures" (OpenAlex) | S | kept |
| geminet26 | Intro, Rel. work, Table 1 | handles changing topologies | "lightweight ML-based TE framework that can handle changing topologies" (USENIX) | S | kept |
| pathgnn25 | Intro, Rel. work, Table 1, Limitations | path-link bipartite GNN, distributed TE, 24.4 % / 35.3 % | abstract (OpenAlex; Scopus export) | S | kept |
| figret24 | Rel. work | robust to traffic bursts | "management of sudden traffic bursts ... burst-aware loss" (OpenAlex) | S | kept |
| shakemap99 | Contrib. 1, Sec. 4.1 | "ShakeMap gives MMI on a grid" | paper gives PGA/PGV–MMI regressions, not the ShakeMap product (OpenAlex) | N | replaced by trinet99 |
| trinet99 (new) | Contrib. 1, Sec. 4.1 | ShakeMap = interpolated maps of ground motion and intensity | "spatial interpolation of the measured ground motions ... maps of ... shaking intensity" (OpenAlex) | S | added; grid format attributed to USGS distribution, not to the paper |
| hurdat13 | Contrib. 1, Sec. 4.1, Setup | HURDAT2 best tracks / format | title "... Presentation of a New Database Format"; abstract on best-track uncertainty (OpenAlex) | S | kept (data-set source) |
| ibtracs10 | Contrib. 1, Sec. 4.1, Setup | IBTrACS best-track archive | dataset description (OpenAlex) | S | kept (data-set source; version v04r01 named in text) |
| alenazi23 (new) | Contrib. 1, Rel. work, Table 1 | real U.S. topologies evaluated against Hurricanes Ivan and Katrina | "four major real-world U.S.-based network providers ... against two major hurricanes ... real-time weather data" (OpenAlex) | S | added; novelty statement now acknowledges this prior use of real hurricanes |
| teavar19 | Contrib. 2, Rel. work, Table 1, Limitations | "optimises CVaR ... given tunnels and per-link probabilities" | abstract: empirical probabilistic failure model, maximise bandwidth under availability target; CVaR/tunnels not stated | P | rewritten to the abstract's wording; Table 1 cell "empirical failure model" |
| prete25 | Intro, Contrib. 2, Rel. work, Table 1 | predicts fibre cuts, moves traffic | "fiber degradation facilitates failure predictions and traffic tunnels to be proactively updated, followed by traffic allocation" (OpenAlex) | P | "moves traffic" → "updates tunnels proactively and then reallocates traffic" |
| aegis26 | Intro, Contrib. 2, Rel. work, Table 1 | CVaR of throughput loss via LP, no precomputed paths | abstract (OpenAlex) | S | kept |
| ffc14 | Rel. work, Table 1, Limitations | "any k link or switch failures cause no congestion" | "freedom from congestion ... under arbitrary combinations of up to k faults" (faults = link failures, switch configuration delays) | P | reworded to the abstract |
| metaopt24 | Rel. work | finds worst inputs of TE heuristics | "find performance gaps and their corresponding adversarial inputs"; production TE heuristic (USENIX) | P | reworded to "performance gap ... and the adversarial inputs that cause it" |
| dts25 | Rel. work | MARL + teacher–student distillation, QoS routing with link failures | abstract + Scopus export (52 % improvement with link failures) | S | kept |
| rarl26 | Rel. work | adversarial RL for SAGIN routing | abstract (OpenAlex) | S | kept |
| metis20 | Rel. work | DNN policies → decision trees | "decision tree and hypergraph ... converts DNN policies to interpretable rule-based controllers" | S | kept |
| clearnet25 | Rel. work | XAI for TE models | "XAI-based approach ... data-driven traffic engineering solutions" | S | kept |
| vasileva26 | Rel. work | incident triage from active measurements | abstract (OpenAlex) | S | kept |
| neumayer10 | Rel. work | random line disasters | "disasters that take the form of a 'random' line" | S | kept |
| neumayer11 | Rel. work, Sec. 4.1 (disks) | line segment and disk models | "model the disaster event as a line segment or a disk" | S | kept |
| rak20 | Rel. work | special issue on resilience | editorial of the *Networks* issue on random and disaster-induced failures | P | sentence narrowed to "a special issue collected optimisation problems ..." |
| andresthio22 | Rel. work, Discussion | adds links for line-segment disaster resilience | "augmenting networks in the plane for disaster-resilience, where a disaster is modeled by a straight-line segment" | P | "so that no single disaster disconnects them" removed (stronger than the abstract) |
| ferdousi20 | Rel. work | progressive network + DC recovery | abstract | S | kept |
| fedrdr24 | Rel. work | federated reinforcement distillation routing after disasters | abstract | S | kept |
| sreeveda26 | Rel. work | GT vs static routing on Abilene, single-link and correlated failures | abstract | S | kept |
| du24 | Rel. work, Discussion | 5G regional TI post-earthquake functionality; "one event can sever many links" | abstract: framework, significant performance degradation after major earthquakes; links not stated | P (Discussion) | Discussion sentence reworded to what the abstract reports |
| chiew24, bouchmal25 | Rel. work | QAOA multi-objective routing | abstracts | S | kept |
| davies25 | Rel. work | QUBO RWA with shared-risk avoidance | abstract | S | kept |
| harb26 | Rel. work | QAOA, dual-link failures, 5-vertex graph | abstract | S | kept |
| cvar00, suurballe74, yen71, nemhauser78, wilcoxon45, holm79 | Method/Setup | method sources | correct foundational sources for CVaR, disjoint paths, k shortest loopless paths, submodular greedy analysis, signed-rank test, Holm procedure | S | kept |
| topozoo11, topohub23, sndlib10 | Setup | topology/data sources | Topology Zoo dataset; TopoHub repository of reference and real-world topologies; SNDlib | S | kept |
| highs18, scipy20, networkx08 | Setup | software | correct software papers | S | kept |

## Summary

- Pairs checked: 88 (sentence × reference, Table 1 included).
- Not supported → removed/replaced: shakemap99 (wrong paper for the ShakeMap product; replaced by trinet99).
- Unverifiable → removed: habib13 (no abstract in any source).
- Partly supported → rewritten: ffc14/teavar19 in Intro ¶1 (re-cited), teavar19, prete25, ffc14, metaopt24, rak20, andresthio22, du24 (Discussion), dote23 (Intro), contribution-2 contrast.
- Added (published, peer-reviewed, abstract read): trinet99 (Earthquake Spectra 1999, doi:10.1193/1.1586057; correct ShakeMap source), alenazi23 (Mathematics 2023, doi:10.3390/math11204250; closest prior use of real hurricanes on real topologies), jimenez23 (IEEE Access 2023, doi:10.1109/ACCESS.2023.3255794; seismic reliability with fragility curves).
- Novelty wording updated: the paper no longer implies that using real hurricanes on real topologies is new; it states that real footprints are used to *choose TE tunnels* and tested on later events.
- References: 46 → 47. No result number or claim about our own results was changed.


## Full-text check (authors, 2026-11-08)
- alenazi23: full text read (Mathematics 11(20):4250). Confirmed: four U.S. providers (AT&T, Internet2, Level3, Sprint) with physical fibre routes from a KMI Corporation dataset; Hurricanes Ivan (2004) and Katrina (2005) from HURDAT; nodes inside a buffer radius around the track fail; metrics node/link failure, largest component size, flow robustness; no traffic engineering. Related-work sentence, Table 1 row and limitation (i) updated accordingly.
