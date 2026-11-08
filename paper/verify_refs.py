"""Verify every reference DOI against Crossref and print the registered metadata.
Usage: python verify_refs.py refs_to_check.json   (list of {"key","doi"} or {"key","url"})
A reference is used in the manuscript only if this script confirms it (DOI resolves in Crossref,
type is a journal article / proceedings paper / book / book chapter, NOT posted-content)."""
import json, sys, time, urllib.error, urllib.parse, urllib.request
UA = {"User-Agent": "disaster-te-reference-check/0.1 (academic manuscript preparation)"}
def get(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    for a in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code != 429 or a == 5:
                raise
            time.sleep(10 * (a + 1))
out = []
for r in json.load(open(sys.argv[1])):
    if "doi" not in r:
        out.append({**r, "status": "NO-DOI (verify URL manually)"}); continue
    m = get(r["doi"])
    if m is None:
        out.append({**r, "status": "NOT FOUND"}); continue
    au = m.get("author", [])
    out.append({**r, "status": "OK" if m["type"] != "posted-content" else "PREPRINT-REJECT", "type": m["type"],
                "title": (m.get("title") or [""])[0], "container": (m.get("container-title") or [""])[0],
                "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "first_author": (au[0].get("family", "") if au else ""), "n_authors": len(au),
                "volume": m.get("volume", ""), "issue": m.get("issue", ""), "page": m.get("page", ""),
                "publisher": m.get("publisher", "")})
    time.sleep(1.0)
json.dump(out, open(sys.argv[1].replace(".json", "_verified.json"), "w"), indent=1)
for o in out:
    print(f"{o['status']:16s} {o['key']:14s} {o.get('year','')} {o.get('first_author','')[:14]:14s} | {o.get('title','')[:85]} | {o.get('container','')[:40]}")
