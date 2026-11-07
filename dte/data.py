"""Loading real network topologies and demand matrices.

Units: link capacity and demand in Mbit/s (SNDlib <unit>MBITPERSEC</unit>);
node coordinates as (longitude, latitude) in decimal degrees (SNDlib coordinatesType="geographical").
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

SNDLIB_NS = {"s": "http://sndlib.zib.de/network"}


@dataclass(frozen=True)
class Topology:
    name: str
    nodes: tuple[str, ...]                      # node ids, index = position in tuple
    lonlat: np.ndarray                          # shape (n_nodes, 2), degrees
    links: tuple[tuple[int, int], ...]          # undirected links (i, j), i != j
    link_capacity_mbps: np.ndarray              # shape (n_links,), capacity of EACH direction
    demand_mbps: np.ndarray                     # shape (n_nodes, n_nodes), directed s->t, zero diagonal
    source: str                                 # provenance string from the file

    @property
    def n(self) -> int:
        return len(self.nodes)

    def arcs(self) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        """Directed arcs. Each undirected link (i,j) becomes arcs i->j and j->i, each with the
        link's full capacity (full-duplex convention, as in SNDlib and the TE literature).
        Returns (arcs, capacity per arc, link index per arc)."""
        arcs, cap, link_of_arc = [], [], []
        for li, (i, j) in enumerate(self.links):
            for a in ((i, j), (j, i)):
                arcs.append(a)
                cap.append(self.link_capacity_mbps[li])
                link_of_arc.append(li)
        return arcs, np.asarray(cap, dtype=float), np.asarray(link_of_arc, dtype=int)


def load_sndlib_xml(path: str, capacity_policy: str, uniform_capacity_mbps: float | None,
                    assume_unit_if_missing: str | None) -> Topology:
    """Parse an SNDlib native-XML network file.

    capacity_policy:
      "preinstalled" -- use <preInstalledModule><capacity>; crash if any link lacks it.
      "uniform"      -- every link gets `uniform_capacity_mbps` (an explicit modelling
                        assumption for planning instances such as GEANT that ship no installed
                        capacity; must be stated in the paper).
    assume_unit_if_missing: some SNDlib files (e.g. nobel-us) have no <meta>; then the demand unit
      must be given explicitly. Results are reported as fractions/ratios, so only consistency matters.
    """
    root = ET.parse(path).getroot()
    meta = root.find("s:meta", SNDLIB_NS)
    if meta is None:
        if assume_unit_if_missing is None:
            raise ValueError(f"{path}: no <meta>/<unit>; set assume_unit_if_missing explicitly")
        unit, origin = assume_unit_if_missing, "no <meta> in file; unit assumed by config"
    else:
        unit = meta.find("s:unit", SNDLIB_NS).text.strip()
        origin = (meta.find("s:origin", SNDLIB_NS).text or "").strip()
    if unit != "MBITPERSEC":
        raise ValueError(f"{path}: unit {unit!r}, expected MBITPERSEC")
    if (capacity_policy == "uniform") != (uniform_capacity_mbps is not None):
        raise ValueError("uniform_capacity_mbps must be set if and only if capacity_policy == 'uniform'")

    nodes_el = root.find("s:networkStructure/s:nodes", SNDLIB_NS)
    if nodes_el.get("coordinatesType") != "geographical":
        raise ValueError(f"{path}: coordinatesType={nodes_el.get('coordinatesType')!r}, need geographical")
    node_ids, lonlat = [], []
    for nd in nodes_el.findall("s:node", SNDLIB_NS):
        node_ids.append(nd.get("id"))
        lonlat.append((float(nd.find("s:coordinates/s:x", SNDLIB_NS).text),
                       float(nd.find("s:coordinates/s:y", SNDLIB_NS).text)))
    idx = {v: k for k, v in enumerate(node_ids)}
    lonlat = np.asarray(lonlat, dtype=float)
    assert np.all(np.abs(lonlat[:, 0]) <= 180) and np.all(np.abs(lonlat[:, 1]) <= 90), "bad coordinates"

    links, caps, seen = [], [], set()
    for lk in root.findall("s:networkStructure/s:links/s:link", SNDLIB_NS):
        i, j = idx[lk.find("s:source", SNDLIB_NS).text], idx[lk.find("s:target", SNDLIB_NS).text]
        if i == j:
            raise ValueError(f"{path}: self-loop on link {lk.get('id')}")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"{path}: parallel link {lk.get('id')}; multigraphs not supported")
        seen.add(key)
        if capacity_policy == "preinstalled":
            pre = lk.find("s:preInstalledModule/s:capacity", SNDLIB_NS)
            if pre is None:
                raise ValueError(f"{path}: link {lk.get('id')} has no preInstalledModule capacity; "
                                 "choose an explicit capacity_policy for this topology")
            caps.append(float(pre.text))
        elif capacity_policy == "uniform":
            caps.append(float(uniform_capacity_mbps))
        else:
            raise ValueError(f"unknown capacity_policy {capacity_policy!r}")
        links.append(key)
    caps = np.asarray(caps, dtype=float)
    assert np.all(caps > 0)

    D = np.zeros((len(node_ids), len(node_ids)))
    for dm in root.findall("s:demands/s:demand", SNDLIB_NS):
        s, t = idx[dm.find("s:source", SNDLIB_NS).text], idx[dm.find("s:target", SNDLIB_NS).text]
        if s == t:
            raise ValueError(f"{path}: demand {dm.get('id')} has source == target")
        D[s, t] += float(dm.find("s:demandValue", SNDLIB_NS).text)
    assert np.all(D >= 0) and D.sum() > 0

    name = path.rsplit("/", 1)[-1].removesuffix(".xml")
    return Topology(name, tuple(node_ids), lonlat, tuple(links), caps, D, origin)


def load_topohub_json(path: str, name: str, uniform_capacity_mbps: float, demand_model: str) -> Topology:
    """Topology Zoo network as packaged by TopoHub (node-link JSON with node "pos" = [lon, lat]).

    Topology Zoo provides real topology and geography but NO capacities and NO traffic, so two
    explicit modelling assumptions are required (both must be stated in the paper):
      * every link gets `uniform_capacity_mbps`;
      * demand_model "gravity_degree": D[s,t] = M_s * M_t / sum(M) for s != t with node mass
        M_i = degree(i) (= total attached capacity under uniform capacities). The absolute level
        is irrelevant because demand is later rescaled to a target optimal MLU.
    """
    import json
    d = json.load(open(path))
    if d.get("directed") or d.get("multigraph"):
        raise ValueError(f"{path}: directed/multigraph JSON not supported")
    ids = [n["id"] for n in d["nodes"]]
    idx = {v: k for k, v in enumerate(ids)}
    lonlat = np.asarray([n["pos"] for n in d["nodes"]], dtype=float)
    assert lonlat.shape == (len(ids), 2)
    assert np.all(np.abs(lonlat[:, 0]) <= 180) and np.all(np.abs(lonlat[:, 1]) <= 90), "bad coordinates"
    links, seen = [], set()
    for e in d["edges"]:
        i, j = idx[e["source"]], idx[e["target"]]
        if i == j:
            raise ValueError(f"{path}: self-loop")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"{path}: duplicate link {key}")
        seen.add(key)
        links.append(key)
    caps = np.full(len(links), float(uniform_capacity_mbps))
    deg = np.zeros(len(ids))
    for i, j in links:
        deg[i] += 1; deg[j] += 1
    if demand_model == "gravity_degree":
        D = np.outer(deg, deg) / deg.sum()
        np.fill_diagonal(D, 0.0)
    else:
        raise ValueError(f"unknown demand_model {demand_model!r}")
    names = tuple(n.get("name", str(n["id"])) for n in d["nodes"])
    return Topology(name, names, lonlat, tuple(links), caps, D,
                    f"TopoHub/Topology Zoo {d['graph'].get('name')}; uniform capacity; {demand_model} demand")
