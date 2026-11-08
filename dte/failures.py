"""Failure scenarios from real disaster footprints and from adversarial / random models.

Geometry conventions:
  * points are (lon, lat) in decimal degrees;
  * a link is modelled as the great-circle arc between its endpoints, sampled every
    `link_step_km` (real fibre follows roads/rail; this is a stated modelling assumption);
  * a NODE fails if its location is inside the footprint;
  * a LINK fails if any of its sample points (endpoints included) is inside the footprint.

Footprints:
  * ShakeMap  -- USGS ShakeMap grid.xml (real recorded/estimated shaking). Inside = MMI >= threshold.
                 Points outside the ShakeMap grid extent are treated as NOT damaged (ShakeMap
                 grids are sized to cover the region of significant shaking).
  * Hurricane -- NOAA HURDAT2 best track (6-hourly) with wind radii by quadrant (NE, SE, SW, NW),
                 linearly interpolated to `interp_hours`. Inside = within the radius of the
                 requested wind class (34, 50 or 64 kt) for the quadrant containing the point.
                 Wind-radii exist only from 2004 onward; older storms are rejected.
  * Disk      -- circle of given radius (the "geographic/disk failure" model of the
                 disaster-resilience literature), used for random and worst-case placements.
"""
from __future__ import annotations

import datetime as dt
import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .data import Topology
from .te import haversine_km

NM_TO_KM = 1.852  # nautical mile, exact by definition


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    kind: str                              # "none" | "shakemap" | "hurricane" | "disk" | "adversarial_links"
    dead_node: np.ndarray                  # bool (n_nodes,)
    dead_link: np.ndarray                  # bool (n_links,)  (undirected)
    meta: dict = field(default_factory=dict)

    def dead_arc(self, topo: Topology) -> np.ndarray:
        arcs, _, link_of_arc = topo.arcs()
        dead = self.dead_link[link_of_arc].copy()
        for k, (u, v) in enumerate(arcs):
            if self.dead_node[u] or self.dead_node[v]:
                dead[k] = True
        return dead


def no_failure(topo: Topology) -> Scenario:
    return Scenario("none", "none", np.zeros(topo.n, bool), np.zeros(len(topo.links), bool))


# ---------------------------------------------------------------- geometry
def link_sample_points(topo: Topology, step_km: float) -> list[np.ndarray]:
    """Points along each link's great-circle arc (spherical linear interpolation), endpoints included."""
    pts = []
    for (i, j) in topo.links:
        a, b = topo.lonlat[i], topo.lonlat[j]
        L = float(haversine_km(a, b))
        n = max(2, int(np.ceil(L / step_km)) + 1)
        va, vb = _unit(a), _unit(b)
        omega = np.arccos(np.clip(va @ vb, -1.0, 1.0))
        ts = np.linspace(0.0, 1.0, n)
        if omega < 1e-12:
            P = np.repeat(va[None], n, axis=0)
        else:
            P = (np.sin((1 - ts) * omega)[:, None] * va + np.sin(ts * omega)[:, None] * vb) / np.sin(omega)
        pts.append(_lonlat(P))
    return pts


def _unit(lonlat):
    lon, lat = np.radians(lonlat[0]), np.radians(lonlat[1])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _lonlat(P):
    return np.column_stack([np.degrees(np.arctan2(P[:, 1], P[:, 0])),
                            np.degrees(np.arcsin(np.clip(P[:, 2], -1, 1)))])


def _bearing_deg(p, q):
    """Initial bearing from p to q (0 = north, clockwise), degrees in [0, 360). Broadcasts over q."""
    p, q = np.radians(np.asarray(p, float)), np.radians(np.asarray(q, float))
    dlon = q[..., 0] - p[..., 0]
    y = np.sin(dlon) * np.cos(q[..., 1])
    x = np.cos(p[..., 1]) * np.sin(q[..., 1]) - np.sin(p[..., 1]) * np.cos(q[..., 1]) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _scenario_from_inside(topo, inside_fn, link_step_km, sid, kind, meta) -> Scenario:
    dead_node = inside_fn(topo.lonlat)
    pts = link_sample_points(topo, link_step_km)
    hit = inside_fn(np.concatenate(pts))                       # one vectorised call for all links
    owner = np.repeat(np.arange(len(pts)), [len(p) for p in pts])
    dead_link = np.zeros(len(pts), bool)
    dead_link[np.unique(owner[hit])] = True
    meta = dict(meta, n_dead_nodes=int(dead_node.sum()), n_dead_links=int(dead_link.sum()))
    return Scenario(sid, kind, dead_node, dead_link, meta)


# ---------------------------------------------------------------- disk
def disk_scenario(topo, center_lonlat, radius_km, link_step_km, sid) -> Scenario:
    c = np.asarray(center_lonlat, float)
    return _scenario_from_inside(topo, lambda P: haversine_km(c, P) <= radius_km, link_step_km, sid,
                                 "disk", {"center_lon": float(c[0]), "center_lat": float(c[1]),
                                          "radius_km": float(radius_km)})


def random_disk_scenarios(topo, radius_km, n, margin_deg, link_step_km, rng: np.random.Generator,
                          require_failure: bool, max_tries: int) -> list[Scenario]:
    """Disk centres uniform in the (lon, lat) bounding box of the nodes, expanded by margin_deg."""
    lo, hi = topo.lonlat.min(0) - margin_deg, topo.lonlat.max(0) + margin_deg
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"only {len(out)}/{n} non-empty disks of r={radius_km} km in {max_tries} tries")
        c = rng.uniform(lo, hi)
        sc = disk_scenario(topo, c, radius_km, link_step_km, f"disk_r{radius_km:g}_{len(out):04d}")
        if require_failure and not (sc.dead_node.any() or sc.dead_link.any()):
            continue
        out.append(sc)
    return out


# ---------------------------------------------------------------- ShakeMap
@dataclass(frozen=True)
class ShakeGrid:
    event_id: str
    lon_min: float
    lat_max: float
    dlon: float
    dlat: float
    mmi: np.ndarray        # (nlat, nlon), row 0 = lat_max, column 0 = lon_min


def parse_shakemap_grid(xml_text: str) -> ShakeGrid:
    ns = {"g": "http://earthquake.usgs.gov/eqcenter/shakemap"}
    root = ET.fromstring(xml_text)
    spec = root.find("g:grid_specification", ns)
    nlon, nlat = int(spec.get("nlon")), int(spec.get("nlat"))
    lon_min, lon_max = float(spec.get("lon_min")), float(spec.get("lon_max"))
    lat_min, lat_max = float(spec.get("lat_min")), float(spec.get("lat_max"))
    fields = {f.get("name"): int(f.get("index")) - 1 for f in root.findall("g:grid_field", ns)}
    data = np.array(root.find("g:grid_data", ns).text.split(), dtype=float).reshape(nlat * nlon, len(fields))
    lon, lat, mmi = data[:, fields["LON"]], data[:, fields["LAT"]], data[:, fields["MMI"]]
    dlon, dlat = (lon_max - lon_min) / (nlon - 1), (lat_max - lat_min) / (nlat - 1)
    # verify the documented layout (rows by descending latitude, longitude ascending) instead of assuming it
    exp_lon = np.tile(lon_min + dlon * np.arange(nlon), nlat)
    exp_lat = np.repeat(lat_max - dlat * np.arange(nlat), nlon)
    if not (np.allclose(lon, exp_lon, atol=dlon * 0.51) and np.allclose(lat, exp_lat, atol=dlat * 0.51)):
        raise ValueError("ShakeMap grid is not in the expected regular row-major layout")
    assert np.all((mmi >= 1.0) & (mmi <= 12.0)), "MMI outside the 1..12 scale"
    return ShakeGrid(root.get("event_id"), lon_min, lat_max, dlon, dlat, mmi.reshape(nlat, nlon))


def shakemap_scenario(topo, grid: ShakeGrid, mmi_threshold, link_step_km) -> Scenario:
    nlat, nlon = grid.mmi.shape

    def inside(P):
        P = np.atleast_2d(P)
        ci = np.rint((P[:, 0] - grid.lon_min) / grid.dlon).astype(int)
        ri = np.rint((grid.lat_max - P[:, 1]) / grid.dlat).astype(int)
        ok = (ci >= 0) & (ci < nlon) & (ri >= 0) & (ri < nlat)
        res = np.zeros(len(P), bool)
        res[ok] = grid.mmi[ri[ok], ci[ok]] >= mmi_threshold
        return res

    return _scenario_from_inside(topo, inside, link_step_km, f"eq_{grid.event_id}_mmi{mmi_threshold:g}",
                                 "shakemap", {"event_id": grid.event_id, "mmi_threshold": float(mmi_threshold),
                                              "max_mmi_in_grid": float(grid.mmi.max())})


# ---------------------------------------------------------------- HURDAT2
@dataclass(frozen=True)
class TrackPoint:
    time: dt.datetime
    lon: float
    lat: float
    vmax_kt: float
    radii_nm: dict          # {34: (NE, SE, SW, NW), 50: (...), 64: (...)}; -999 = missing in HURDAT2


def parse_hurdat2(text: str) -> dict[str, dict]:
    """Parse the NOAA HURDAT2 Atlantic file into {storm_id: {"name": str, "track": [TrackPoint]}}.
    Format: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf"""
    storms, lines = {}, [ln for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        hdr = [c.strip() for c in lines[i].split(",")]
        sid, name, nrows = hdr[0], hdr[1], int(hdr[2])
        track = []
        for ln in lines[i + 1:i + 1 + nrows]:
            c = [x.strip() for x in ln.split(",")]
            t = dt.datetime.strptime(c[0] + c[1], "%Y%m%d%H%M")
            lat = float(c[4][:-1]) * (1 if c[4][-1] == "N" else -1)
            lon = float(c[5][:-1]) * (-1 if c[5][-1] == "W" else 1)
            r = [float(x) for x in c[8:20]]
            track.append(TrackPoint(t, lon, lat, float(c[6]),
                                    {34: tuple(r[0:4]), 50: tuple(r[4:8]), 64: tuple(r[8:12])}))
        storms[sid] = {"name": name, "track": track}
        i += 1 + nrows
    return storms


def hurricane_scenario(topo, storm_id, storm, wind_kt, interp_hours, link_step_km) -> Scenario:
    if wind_kt not in (34, 50, 64):
        raise ValueError("wind_kt must be 34, 50 or 64 (HURDAT2 wind-radii classes)")
    # Off-synoptic rows (landfall / peak-intensity entries, not at 00/06/12/18 UTC) carry no wind
    # radii (-999) by NHC convention; they are dropped and covered by interpolation between the
    # 6-hourly synoptic rows. A missing radius on a synoptic row is an error.
    synoptic = lambda p: p.time.minute == 0 and p.time.hour % 6 == 0
    tr = [p for p in storm["track"] if synoptic(p) or min(p.radii_nm[wind_kt]) >= 0]
    if any(min(p.radii_nm[wind_kt]) < 0 for p in tr):
        raise ValueError(f"{storm_id}: missing {wind_kt}-kt wind radii (-999) on a synoptic row; "
                         "HURDAT2 radii exist only from 2004")
    # linear interpolation in time of position and radii
    T = np.array([(p.time - tr[0].time).total_seconds() / 3600 for p in tr])
    tq = np.arange(0.0, T[-1] + 1e-9, interp_hours)
    # unwrap longitude so tracks crossing the 180th meridian interpolate along the short way
    lon_unw = np.degrees(np.unwrap(np.radians([p.lon for p in tr])))
    lon = (np.interp(tq, T, lon_unw) + 180.0) % 360.0 - 180.0
    lat = np.interp(tq, T, [p.lat for p in tr])
    R = np.column_stack([np.interp(tq, T, [p.radii_nm[wind_kt][q] for p in tr]) for q in range(4)]) * NM_TO_KM

    def inside(P):
        P = np.atleast_2d(P)
        res = np.zeros(len(P), bool)
        for k in range(len(tq)):
            if R[k].max() <= 0:
                continue
            c = np.array([lon[k], lat[k]])
            d = haversine_km(c, P)
            quad = (_bearing_deg(c, P) // 90).astype(int)          # 0 NE, 1 SE, 2 SW, 3 NW
            res |= d <= R[k][quad]
        return res

    return _scenario_from_inside(topo, inside, link_step_km, f"hu_{storm_id}_{wind_kt}kt", "hurricane",
                                 {"storm_id": storm_id, "name": storm["name"], "wind_kt": wind_kt,
                                  "peak_vmax_kt": max(p.vmax_kt for p in tr)})


# ---------------------------------------------------------------- adversarial link cuts
def all_link_cuts(topo: Topology, k: int) -> list[Scenario]:
    """Every set of exactly k failed links (exhaustive; caller checks the count is affordable)."""
    out = []
    for combo in itertools.combinations(range(len(topo.links)), k):
        dl = np.zeros(len(topo.links), bool)
        dl[list(combo)] = True
        out.append(Scenario("cut_" + "-".join(map(str, combo)), "adversarial_links",
                            np.zeros(topo.n, bool), dl, {"links": list(combo), "k": k}))
    return out
