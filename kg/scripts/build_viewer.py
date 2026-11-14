#!/usr/bin/env python3
"""Embed graph/graph.json into the self-contained HTML explorer -> viewer/index.html.

Usage (run build_graph.py first; rerun after every build):
    KG="/Users/byounghanlee/Documents/kubectl/clear-ec challenge/clear-ec-research/kg"
    /usr/bin/python3 "$KG/scripts/build_viewer.py"

Ported from /Users/byounghanlee/EHR-KG/scripts/build_viewer.py (2026-11-09). Positions come from graph_layout.py
(needs numpy). Long body-only paper fields and frontmatter link lists (already present as edges) are dropped.
The page loads nothing from the network: no scripts, stylesheets or fonts.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graph_layout  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEEP = {
    # all types
    "id", "type", "title", "name", "summary", "tldr", "_path", "degree", "stub", "status", "date", "origin", "source",
    # experiment
    "lane", "direction", "metrics", "primary_metric", "folds", "baseline", "value", "delta", "delta_num",
    "ambiguous", "ambiguous_note",
    # idea
    "aims_at", "novelty_check", "owner_lane", "cost", "cost_hours", "cost_unit", "potential_gain",
    "potential_metric", "potential_basis",
    # decision, constraint, method, dataset
    "deciders", "scope", "code", "access", "location", "aliases",
    # paper
    "authors", "year", "venue", "url", "group", "relevance_clearec", "relevance_circe", "relevance_metrics",
    "relevance_note", "key_numbers", "suggests_try", "superseded", "tier", "evidence", "results", "code_release",
    "weights_release", "review_depth",
    # layout
    "x", "y", "size", "region", "short", "imp",
}
# Long or bibliographic paper fields and link lists that never need to reach the page.
DROP = {"citation", "search_keywords", "tags", "data_scale", "sources_read", "has_limitation", "doi", "arxiv",
        "pmcid", "reviewed_on", "root", "body", "text"}
EDGE_KEEP = ("source", "target", "type", "origin", "relation", "closeness", "note", "metric", "result", "locator",
             "value", "stated_by", "confidence", "bb")
EXTRA_MAX = 400  # unknown scalar fields up to this length are kept, so new note fields show up without a code change


def is_link_value(v):
    if isinstance(v, str):
        return v.strip().startswith("[[")
    if isinstance(v, list):
        return any(is_link_value(x) or (isinstance(x, dict) and "target" in x) for x in v)
    return False


def slim(n, edge_types):
    out = {}
    for k, v in n.items():
        if v is None or v == "" or v == []:
            continue
        if k in KEEP:
            out[k] = v
        elif k in DROP or k in edge_types or k.startswith("_") or is_link_value(v):
            continue
        elif isinstance(v, (str, int, float, bool)) and len(str(v)) <= EXTRA_MAX:
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v) and len(json.dumps(v)) <= EXTRA_MAX:
            out[k] = v
    return out


def main():
    g = json.load(open(ROOT / "graph" / "graph.json", encoding="utf-8"))
    nodes_in, edges_in = g["nodes"], g["edges"]
    regions = graph_layout.compute(nodes_in, edges_in)
    edge_types = {e["type"] for e in edges_in} | set(graph_layout.EDGE_WEIGHT)
    ids = {n["id"] for n in nodes_in}
    nodes = [slim(n, edge_types) for n in nodes_in]
    edges = [{k: e[k] for k in EDGE_KEEP if e.get(k) not in (None, "")}
             for e in edges_in if e["source"] in ids and e["target"] in ids]
    data = json.dumps({"generated": g.get("generated"), "regions": regions, "nodes": nodes, "edges": edges},
                      ensure_ascii=False, separators=(",", ":"), default=str)
    # "<" only occurs inside JSON strings; \u003c keeps "</script>" and "<!--" from ending the script block.
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = (ROOT / "scripts" / "viewer_template.html").read_text(encoding="utf-8")
    marker = "/*GRAPH_DATA*/null"
    if html.count(marker) != 1:
        sys.exit(f"template must contain {marker} exactly once")
    out = ROOT / "viewer" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html.replace(marker, data), encoding="utf-8")
    by_type = collections.Counter(n["type"] for n in nodes)
    by_origin = collections.Counter(e.get("origin", "?") for e in edges)
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024:.0f} KB, {size / 1e6:.2f} MB)")
    print(f"nodes={len(nodes)} {dict(sorted(by_type.items()))} stubs={sum(1 for n in nodes if n.get('stub'))}")
    print(f"edges={len(edges)} {dict(sorted(by_origin.items()))} backbone={sum(1 for e in edges if e.get('bb'))} "
          f"regions={len(regions)} generated={g.get('generated')}")


if __name__ == "__main__":
    main()
