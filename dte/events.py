"""Systematic real-event selection: IBTrACS (global tropical cyclones) and USGS earthquake catalogues.

Selection is rule-based (no hand-picking):
  * cyclones: every storm in the chosen IBTrACS basin file / HURDAT2 file with usable wind radii;
  * earthquakes: every USGS ComCat event with M >= min_magnitude and a ShakeMap whose epicentre lies
    in the topology bounding box expanded by bbox_margin_deg and within prefilter_km of a link
    sample point (pure download-size prefilter; MMI >= 7 farther than a few hundred km from the
    epicentre is rare for the magnitudes involved -- a stated assumption).
An event becomes a scenario only if its footprint fails at least one node or link.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
import urllib.parse

import numpy as np

from . import failures as F
from .fetch import _get, _save, _sha256
from .te import haversine_km

RADII = {34: ("USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"),
         50: ("USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW"),
         64: ("USA_R64_NE", "USA_R64_SE", "USA_R64_SW", "USA_R64_NW")}


def parse_ibtracs(path: str, first_season: int):
    """IBTrACS v04 CSV -> {sid: {"name", "season", "track": [TrackPoint], "excluded": reason|None}}.

    Uses the US agency (JTWC/NHC) wind radii, as HURDAT2 does for the Atlantic. Only TRACK_TYPE
    "main" rows. Radii blank on a row whose USA_WIND is below the wind class are set to 0 (no
    winds of that class can exist); radii blank while USA_WIND >= class are genuinely missing and
    are encoded -999, which hurricane_scenario rejects on synoptic rows (storm then excluded
    for that wind class by the caller)."""
    storms = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        next(rd)                                     # units row
        for r in rd:
            if r["TRACK_TYPE"] != "main" or int(r["SEASON"]) < first_season:
                continue
            w = r["USA_WIND"].strip()
            vmax = float(w) if w else float("nan")
            radii = {}
            for kt, cols in RADII.items():
                vals = []
                for c in cols:
                    x = r[c].strip()
                    if x:
                        vals.append(float(x))
                    elif not np.isnan(vmax) and vmax < kt:
                        vals.append(0.0)
                    else:
                        vals.append(-999.0)
                radii[kt] = tuple(vals)
            t = dt.datetime.strptime(r["ISO_TIME"], "%Y-%m-%d %H:%M:%S")
            s = storms.setdefault(r["SID"], {"name": r["NAME"], "season": int(r["SEASON"]), "track": []})
            s["track"].append(F.TrackPoint(t, float(r["LON"]), float(r["LAT"]), vmax, radii))
    return storms


def cyclone_scenarios(topo, storms, wind_kt, interp_hours, step):
    """One scenario per storm whose footprint fails something. Storms with missing radii on a
    synoptic row (for this wind class) are skipped and counted, never silently fixed."""
    out, skipped = [], []
    for sid, s in storms.items():
        if len(s["track"]) < 2:
            skipped.append((sid, "fewer than 2 rows")); continue
        try:
            sc = F.hurricane_scenario(topo, sid, s, wind_kt, interp_hours, step)
        except ValueError as e:
            if "missing" in str(e):
                skipped.append((sid, "missing radii")); continue
            raise
        if sc.dead_node.any() or sc.dead_link.any():
            sc.meta["season"] = s.get("season", sc.meta.get("season"))
            out.append(sc)
    return out, skipped


def usgs_candidates(topo, start, end, min_mag, bbox_margin_deg, prefilter_km, step):
    lo, hi = topo.lonlat.min(0) - bbox_margin_deg, topo.lonlat.max(0) + bbox_margin_deg
    q = {"format": "geojson", "starttime": start, "endtime": end, "minmagnitude": min_mag,
         "minlongitude": max(-180, lo[0]), "maxlongitude": min(180, hi[0]),
         "minlatitude": max(-90, lo[1]), "maxlatitude": min(90, hi[1]), "producttype": "shakemap",
         "orderby": "time-asc"}
    feats = json.loads(_get("https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(q), 120))["features"]
    pts = np.concatenate([topo.lonlat] + F.link_sample_points(topo, step))
    keep = []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        if haversine_km((lon, lat), pts).min() <= prefilter_km:
            keep.append({"id": f["id"], "mag": f["properties"]["mag"], "time_ms": f["properties"]["time"],
                         "year": dt.datetime.fromtimestamp(f["properties"]["time"] / 1000, dt.timezone.utc).year,
                         "place": f["properties"]["place"]})
    return keep, len(feats)


def shakemap_grid(data_dir, event_id, manifest):
    """Download (once) the preferred ShakeMap grid.xml for an event; record url + sha256."""
    rel = f"events/shakemap_{event_id}.xml"
    p = os.path.join(data_dir, rel)
    if not os.path.exists(p):
        det = json.loads(_get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&eventid={event_id}", 60))
        sm = det["properties"]["products"]["shakemap"][0]
        url = sm["contents"]["download/grid.xml"]["url"]
        _save(p, _get(url, 300))
        manifest[rel] = {"url": url, "shakemap_source": sm["source"], "version": sm["properties"].get("version")}
        time.sleep(0.5)
    data = open(p, "rb").read()
    manifest.setdefault(rel, {})["sha256"] = _sha256(data)
    return F.parse_shakemap_grid(data.decode())
