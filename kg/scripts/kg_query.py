#!/usr/bin/env python3
"""Query the CLEAR-EC KG from the shell (run build_graph.py first if notes changed).

Usage (PY=/usr/bin/python3, KG=<kg root>):
  $PY "$KG/scripts/kg_query.py" check <words...>           idea check: every node (incl. note bodies) matching ALL words,
                                                        with status, metric, folds, baseline, value, delta, source
  $PY "$KG/scripts/kg_query.py" find <text>                nodes whose id/name/title/summary/direction contain text
  $PY "$KG/scripts/kg_query.py" node <id> [--all]          fields + typed edges in/out (--all adds body mentions)
  $PY "$KG/scripts/kg_query.py" edges <type> [--to ID] [--from ID]
  $PY "$KG/scripts/kg_query.py" experiments [--lane L] [--status S] [--metric M]
  $PY "$KG/scripts/kg_query.py" closed [--lane L] [--metric M]   closed directions with their best try
  $PY "$KG/scripts/kg_query.py" kept [--lane L]            kept steps, running work and references per lane
  $PY "$KG/scripts/kg_query.py" ideas [--status S]
  $PY "$KG/scripts/kg_query.py" findings
  $PY "$KG/scripts/kg_query.py" constraints               active constraints (PI rules first) and what they limit
  $PY "$KG/scripts/kg_query.py" papers
"""
import argparse
import collections
import json
import re

import kglib
from kglib import OUT, ROOT, label, parse_note, short

WEAK = {"mentions"}


def load():
    g = json.load(open(OUT / "graph.json", encoding="utf-8"))
    return {n["id"]: n for n in g["nodes"]}, g["edges"]


def attrs(e):
    keep = [f"{k}={short(e[k], 120)}" for k in ("relation", "closeness", "stated_by", "result", "metric", "value", "locator", "note")
            if e.get(k) not in (None, "")]
    return f"  [{'; '.join(keep)}]" if keep else ""


def exp_line(n):
    amb = " AMBIGUOUS" if n.get("ambiguous") else ""
    metric = n.get("primary_metric") or ",".join(map(str, n.get("metrics") or []))
    return (f"{n.get('status', ''):<11} {n.get('lane', ''):<10} {n['id']}{amb}\n"
            f"            {metric} folds {n.get('folds')} | value {short(n.get('value'), 60)} | delta {short(n.get('delta'), 60)}\n"
            f"            baseline: {short(n.get('baseline'), 150)}\n"
            f"            source: {short(n.get('source'), 150)}")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "find", "node", "edges", "experiments", "closed", "kept", "ideas",
                                    "findings", "constraints", "papers"])
    ap.add_argument("arg", nargs="*")
    ap.add_argument("--to")
    ap.add_argument("--from", dest="src")
    ap.add_argument("--lane")
    ap.add_argument("--status")
    ap.add_argument("--metric")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    nodes, edges = load()
    out_ = collections.defaultdict(list)
    in_ = collections.defaultdict(list)
    for e in edges:
        out_[e["source"]].append(e)
        in_[e["target"]].append(e)
    text = " ".join(a.arg)

    if a.cmd == "find":
        q = text.lower()
        for n in nodes.values():
            hay = " ".join(str(n.get(k, "")) for k in ("id", "title", "name", "tldr", "summary", "direction",
                                                       "search_keywords", "aliases")).lower()
            if q in hay:
                print(f"{n['type']:<11} {n.get('status', ''):<12} {n['id']:<48} {short(label(n), 80)}")

    elif a.cmd == "check":
        words = [w.lower() for w in a.arg]
        if not words:
            raise SystemExit("usage: kg_query.py check <words...>")
        hits = collections.defaultdict(list)
        for n in nodes.values():
            if n.get("stub"):
                continue
            hay = " ".join(str(n.get(k, "")) for k in ("id", "title", "name", "tldr", "summary", "direction",
                                                       "search_keywords", "aliases", "baseline", "value"))
            if n.get("_path"):
                try:
                    hay += " " + parse_note(ROOT / n["_path"])[1]
                except OSError:
                    pass
            hay = hay.lower()
            if all(re.search(re.escape(w), hay) for w in words):
                hits[n["type"]].append(n)
        if not hits:
            print("No node matches all of:", words, "\nTry fewer or broader words, then grep wiki/ directly.")
        for t in ("experiment", "idea", "finding", "paper", "method", "constraint", "decision", "dataset", "synthesis"):
            if t not in hits:
                continue
            print(f"\n== {t} ({len(hits[t])})")
            for n in sorted(hits[t], key=lambda x: (str(x.get("status")), x["id"])):
                if t == "experiment":
                    print(exp_line(n))
                elif t == "idea":
                    tb = ", ".join(f"{e['target']}({e.get('closeness', '?')})" for e in out_[n["id"]] if e["type"] == "tested_by")
                    print(f"{n.get('status', ''):<17} {n['id']:<44} {short(n.get('name'), 70)}\n            tested_by: {tb or '-'}")
                elif t == "paper":
                    links = sorted({e["target"] for e in out_[n["id"]] if e["type"] != "mentions"
                                    and nodes.get(e["target"], {}).get("type") in ("idea", "finding", "method", "decision")}
                                   | {e["source"] for e in in_[n["id"]] if e["type"] == "literature"})
                    print(f"{str(n.get('year', '')):<5} {n['id']:<40} {short(n.get('title'), 80)}\n            links: {', '.join(links) or '-'}")
                else:
                    print(f"{n.get('status', ''):<11} {n['id']:<44} {short(label(n), 90)}")
        print("\nNext: read the notes behind the hits (wiki/<folder>/<id>.md), then a quick web search; "
              "record the link (idea `tested_by`, paper `evidence_for`, or a new idea node) and log it.")

    elif a.cmd == "node":
        n = nodes.get(text)
        if not n:
            raise SystemExit(f"unknown id: {text}")
        print(f"{n['id']} ({n['type']}{', stub' if n.get('stub') else ''}) — {label(n)}")
        skip = {"id", "type", "name", "title", "degree"}
        for k, v in n.items():
            if k in skip or isinstance(v, (list, dict)) and k in kglib.load_schema()["edges"]:
                continue
            if v not in (None, "", []):
                print(f"  {k}: {short(v, 300)}")
        for direction, mine, other in (("out", "source", "target"), ("in", "target", "source")):
            groups = collections.defaultdict(list)
            for e in edges:
                if e[mine] == text and (a.all or e["type"] not in WEAK):
                    groups[e["type"]].append(e)
            for t in sorted(groups):
                print(f"\n  {direction} {t} ({len(groups[t])})")
                for e in groups[t]:
                    print(f"    {e[other]:<46} {short(label(nodes.get(e[other])), 60)}{attrs(e)}")

    elif a.cmd == "edges":
        for e in edges:
            if e["type"] == text and (not a.to or e["target"] == a.to) and (not a.src or e["source"] == a.src):
                print(f"{e['source']} -> {e['target']}{attrs(e)}")

    elif a.cmd in ("experiments", "kept"):
        want = {"kept", "running", "reference"} if a.cmd == "kept" else None
        for n in sorted((n for n in nodes.values() if n["type"] == "experiment"),
                        key=lambda n: (str(n.get("lane")), str(n.get("status")), n["id"])):
            if a.lane and n.get("lane") != a.lane:
                continue
            if a.status and n.get("status") != a.status:
                continue
            if want and n.get("status") not in want:
                continue
            if a.metric and a.metric.upper() not in [str(x).upper() for x in (n.get("metrics") or [])]:
                continue
            print(exp_line(n))

    elif a.cmd == "closed":
        by_dir = collections.defaultdict(list)
        for n in nodes.values():
            if n["type"] == "experiment" and n.get("status") == "closed":
                if a.lane and n.get("lane") != a.lane:
                    continue
                if a.metric and a.metric.upper() not in [str(x).upper() for x in (n.get("metrics") or [])]:
                    continue
                by_dir[str(n.get("direction"))].append(n)
        for d in sorted(by_dir):
            ms = by_dir[d]
            withd = [m for m in ms if num(m.get("delta_num")) is not None]
            clear = [m for m in withd if not m.get("ambiguous")]
            best = min(clear or withd, key=lambda m: num(m["delta_num"])) if withd else ms[0]
            print(f"\n## {d}  ({len(ms)} closed tries; best shown)")
            print(exp_line(best))
            others = [m["id"] for m in ms if m is not best]
            if others:
                print(f"            other tries: {', '.join(sorted(others))}")

    elif a.cmd == "ideas":
        for n in sorted((n for n in nodes.values() if n["type"] == "idea"), key=lambda n: (str(n.get("status")), n["id"])):
            if a.status and n.get("status") != a.status:
                continue
            tb = ", ".join(f"{e['target']}({e.get('closeness', '?')})" for e in out_[n["id"]] if e["type"] == "tested_by")
            print(f"{n.get('status', ''):<17} {n['id']:<44} {short(n.get('name'), 70)}")
            print(f"{'':<17} aims at {', '.join(map(str, n.get('aims_at') or [])) or '-'}; tested_by: {tb or '-'}")

    elif a.cmd == "findings":
        for n in sorted((n for n in nodes.values() if n["type"] == "finding"), key=lambda n: (n.get("status") != "robust", n["id"])):
            sup = [e["source"] for e in in_[n["id"]] if e["type"] == "supports_finding"]
            print(f"{n.get('status', ''):<10} {n['id']:<46} {short(n.get('name'), 80)}")
            print(f"{'':<10} supported by: {', '.join(sup) or '-'} | source: {short(n.get('source'), 80)}")

    elif a.cmd == "constraints":
        for n in sorted((n for n in nodes.values() if n["type"] == "constraint"),
                        key=lambda n: (n.get("status") != "active", n.get("origin") != "user", n["id"])):
            tg = [e["target"] for e in out_[n["id"]] if e["type"] == "constrains"]
            who = "PI" if n.get("origin") == "user" else n.get("origin")
            print(f"{n.get('status', ''):<7} {who:<8} {n['id']:<44} {short(n.get('summary'), 100)}")
            print(f"{'':<16} limits: {', '.join(tg) or '-'}")

    elif a.cmd == "papers":
        for n in sorted((n for n in nodes.values() if n["type"] == "paper" and not n.get("stub")),
                        key=lambda n: -int(n.get("year") or 0)):
            rel = n.get("relevance") or n.get("relevance_clearec") or n.get("relevance_clear_ec") or n.get("relevance_circe") or ""
            print(f"{str(n.get('year', '')):<5} {str(rel):<7} {n['id']:<40} {short(n.get('title'), 80)}")


if __name__ == "__main__":
    main()
