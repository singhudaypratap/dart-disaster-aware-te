"""Build refs.bib: BibTeX for every DOI is fetched from Crossref (content negotiation) so that
author lists, titles and pages come from the registry, not from memory. Keys are renamed to ours."""
import re, sys, time, urllib.error, urllib.parse, urllib.request
UA = {"User-Agent": "disaster-te-reference-check/0.1 (academic manuscript preparation)"}
DOIS = {
 "teal23": "10.1145/3603269.3604857", "harp24": "10.1145/3651890.3672237", "figret24": "10.1145/3651890.3672258",
 "pathgnn25": "10.1109/JSAC.2025.3528815", "teavar19": "10.1145/3341302.3342069", "ffc14": "10.1145/2619239.2626314",
 "prete25": "10.1145/3718958.3750508", "clearnet25": "10.1109/TNSM.2025.3567654", "metis20": "10.1145/3387514.3405859",
 "aegis26": "10.1109/ton.2026.3684425", "pathgnn_dummy": None,
 "topozoo11": "10.1109/JSAC.2011.111002", "sndlib10": "10.1002/net.20371", "topohub23": "10.1016/j.softx.2023.101540",
 "hurdat13": "10.1175/MWR-D-12-00254.1", "ibtracs10": "10.1175/2009BAMS2755.1", "shakemap99": "10.1193/1.1586058",
 "neumayer10": "10.1109/infcom.2010.5461984", "neumayer11": "10.1109/tnet.2011.2128879", "agarwal13": "10.1109/tnet.2012.2232111",
 "habib13": "10.1016/j.comcom.2013.01.004", "rak20": "10.1002/net.21940", "giraosilva20": "10.1002/net.21931",
 "andresthio22": "10.1002/net.22138", "ferdousi20": "10.1109/tnsm.2020.2983822", "du24": "10.1016/j.iintel.2024.100084",
 "wilcoxon45": "10.2307/3001968", "cvar00": "10.21314/JOR.2000.038", "nemhauser78": "10.1007/BF01588971",
 "suurballe74": "10.1002/net.3230040204", "fortz00": "10.1109/INFCOM.2000.832225", "yen71": "10.1287/mnsc.17.11.712",
 "highs18": "10.1007/s12532-017-0130-5", "scipy20": "10.1038/s41592-019-0686-2", "networkx08": "10.25080/TCWV9851",
 "davies25": "10.1364/JOCN.553657", "harb26": "10.1109/TQE.2026.3702858", "sreeveda26": "10.18280/ijsse.160720",
}
def fetch(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi) + "/transform/application/x-bibtex"
    for a in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == 5:
                raise
            time.sleep(15 * (a + 1))
out = []
for key, doi in DOIS.items():
    if doi is None:
        continue
    b = fetch(doi).strip()
    b = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + key + ",", b, count=1)
    b = re.sub(r"\s*month\s*=\s*\w+,?", "", b)     # Crossref month macros are not needed
    out.append(b)
    print("ok", key, file=sys.stderr)
    time.sleep(1.5)
open("refs_crossref.bib", "w").write("\n\n".join(out) + "\n")
