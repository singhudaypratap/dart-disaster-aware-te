"""Download every real input listed in the config, verify checksums, write data/raw/MANIFEST.json.

    python -m dte.fetch --config config/experiment.yaml

Files that are already present are re-verified, not re-downloaded. Topology files carry a pinned
sha256 in the config and the run stops on a mismatch. USGS ShakeMaps and HURDAT2 have no pinned
hash in the config (USGS may reissue ShakeMap versions); their sha256 is recorded in the manifest
so every result can be traced to the exact bytes used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request

import yaml

UA = {"User-Agent": "disaster-te-research/0.1 (academic network-resilience study)"}


def _get(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _save(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    os.chmod(path, 0o444)  # raw data is read-only


def fetch(cfg: dict) -> dict:
    root = cfg["data_dir"]
    manifest = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}

    for t in cfg["topologies"]:
        p = os.path.join(root, t["file"])
        data = open(p, "rb").read() if os.path.exists(p) else _get(t["url"], 120)
        h = _sha256(data)
        if h != t["sha256"]:
            raise RuntimeError(f"{t['name']}: sha256 {h} != pinned {t['sha256']} ({t['url']})")
        if not os.path.exists(p):
            _save(p, data)
        manifest["files"][t["file"]] = {"url": t["url"], "sha256": h}

    if "scenarios" not in cfg:          # e.g. test_real_events.yaml: events fetched by dte/events.py
        with open(os.path.join(root, "MANIFEST_topologies_" + cfg["run"]["name"] + ".json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return manifest
    hc = cfg["scenarios"]["hurricane"]
    p = os.path.join(root, hc["hurdat2_file"])
    data = open(p, "rb").read() if os.path.exists(p) else _get(hc["hurdat2_url"], 120)
    if not os.path.exists(p):
        _save(p, data)
    manifest["files"][hc["hurdat2_file"]] = {"url": hc["hurdat2_url"], "sha256": _sha256(data)}

    for ev in cfg["scenarios"]["shakemap"]["events"]:
        rel = f"events/shakemap_{ev}.xml"
        p = os.path.join(root, rel)
        detail_url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&eventid={ev}"
        detail = json.loads(_get(detail_url, 60))
        sm = detail["properties"]["products"]["shakemap"][0]   # USGS "preferred" product is first
        grid_url = sm["contents"]["download/grid.xml"]["url"]
        if os.path.exists(p):
            data = open(p, "rb").read()
        else:
            data = _get(grid_url, 300)
            _save(p, data)
        manifest["files"][rel] = {"url": grid_url, "sha256": _sha256(data),
                                  "event_title": detail["properties"]["title"],
                                  "magnitude": detail["properties"]["mag"],
                                  "shakemap_source": sm["source"],
                                  "shakemap_version": sm["properties"].get("version")}
        time.sleep(1.0)  # be polite to the USGS API

    with open(os.path.join(root, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    m = fetch(yaml.safe_load(open(ap.parse_args().config)))
    print(f"{len(m['files'])} files verified; manifest written")
