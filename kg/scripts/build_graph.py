#!/usr/bin/env python3
"""Compile the CLEAR-EC KG wiki into a graph, views and a lint report.

Usage:  /usr/bin/python3 scripts/build_graph.py        (from any directory; paths resolve from this file)
Reads:  wiki/**/*.md (YAML frontmatter + [[wikilinks]]), schema/edges.yaml
Writes: graph/graph.json, graph/lint.md, graph/references.bib, wiki/index.md, wiki/_views/*.md

Paper notes (wiki/papers/, written from the EHR-KG paper template) are accepted as they are: a paper link to an
id that has no note becomes a stub node and a lint warning, not an error. Every other broken link is an error.
Exit code 1 on blocking errors (broken links from non-paper notes, YAML errors, duplicate or missing ids).
"""
import collections
import datetime
import json
import sys

import kglib
from kglib import LINK, OUT, ROOT, WIKI, guess_type, link_ids, parse_note

ORDER = ["paper", "method", "dataset", "experiment", "finding", "idea", "decision", "constraint", "synthesis"]
LANE_ORDER = ["cd", "cv", "hex", "backbone", "deploy", "rootcause-1109", "gpu0-1106", "gpu1-1106", "gpu2-1106", "gpu3-1106", "pre-lanes", "coordinator"]


def clean(s, k=None):
    s = " ".join(str(s if s is not None else "").replace("|", "/").split())
    return s if not k or len(s) <= k else s[:k - 1] + "…"


def lane_key(lane):
    return LANE_ORDER.index(lane) if lane in LANE_ORDER else len(LANE_ORDER)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    schema = kglib.load_schema()
    edge_defs, node_types, values = schema["edges"], schema["node_types"], schema.get("values", {})
    nodes, bodies, errors, warnings, infos = {}, {}, [], [], []

    for path in kglib.note_paths():
        rel = path.relative_to(ROOT).as_posix()
        meta, body = parse_note(path)
        if meta is None:
            (warnings if "/papers/" in rel else errors).append(("no-frontmatter", rel, ""))
            continue
        if "_yaml_error" in meta:
            errors.append(("yaml-error", rel, meta["_yaml_error"][:160]))
            continue
        nid = str(meta.get("id") or "").strip()
        if not nid:
            errors.append(("missing-id", rel, ""))
            continue
        if nid != path.stem:
            errors.append(("id-filename-mismatch", rel, nid))
        if nid in nodes:
            errors.append(("duplicate-id", rel, nid))
            continue
        ntype = meta.get("type")
        if ntype not in node_types:
            errors.append(("unknown-type", rel, str(ntype)))
        else:
            for f in node_types[ntype].get("required", []):
                if meta.get(f) in (None, "", []):
                    (warnings if ntype == "paper" else errors).append(("missing-field", rel, f))
            for f in node_types[ntype].get("recommended", []):
                if meta.get(f) in (None, "", []):
                    infos.append(("missing-recommended-field", rel, f))
        for key in ("status", "lane"):
            allowed = values.get(f"{ntype}.{key}")
            if allowed and meta.get(key) not in (None, "") and meta.get(key) not in allowed:
                errors.append(("bad-value", rel, f"{key}={meta.get(key)!r}, allowed {allowed}"))
        if ntype in ("idea", "constraint") and meta.get("origin") not in values.get("origin", []):
            errors.append(("bad-value", rel, f"origin={meta.get('origin')!r}"))
        meta["_path"] = rel
        nodes[nid] = meta
        bodies[nid] = body

    # ---- edges ----
    edges = []
    for nid, meta in nodes.items():
        authored = set()
        for key in edge_defs:
            for tgt, attrs in link_ids(meta.get(key), strict=meta.get("type") == "paper"):
                e = {"source": nid, "target": tgt, "type": key, "origin": "frontmatter"}
                e.update({k: v for k, v in attrs.items() if v not in (None, "")})
                edges.append(e)
                authored.add(tgt)
        for tgt in sorted(set(t.strip() for t in LINK.findall(bodies[nid]))):
            if tgt not in authored and tgt != nid:
                edges.append({"source": nid, "target": tgt, "type": "mentions", "origin": "body"})

    # ---- resolve links; paper links to missing ids become stubs ----
    stubs = {}
    kept_edges = []
    for e in edges:
        src = nodes[e["source"]]
        if e["target"] not in nodes:
            if src.get("type") == "paper":
                if e["origin"] == "frontmatter":
                    warnings.append(("paper-link-unresolved", src["_path"], f"{e['type']} -> {e['target']}"))
                    st = stubs.setdefault(e["target"], {"id": e["target"], "type": guess_type(e["target"]),
                                                        "name": e["target"], "stub": True, "_path": None})
                    kept_edges.append(e)
                else:
                    infos.append(("paper-body-link-unresolved", src["_path"], e["target"]))
                continue
            if e["origin"] == "frontmatter":
                errors.append(("broken-link", src["_path"], f"{e['type']} -> {e['target']}"))
            else:
                errors.append(("broken-body-link", src["_path"], f"[[{e['target']}]]"))
            continue
        spec = edge_defs.get(e["type"])
        if e["origin"] == "frontmatter" and spec:
            tgt_t = nodes[e["target"]].get("type")
            if spec.get("to") and tgt_t not in spec["to"]:
                errors.append(("edge-type-mismatch", src["_path"], f"{e['type']} -> {e['target']} ({tgt_t})"))
            if spec.get("from") and src.get("type") not in spec["from"]:
                errors.append(("edge-source-mismatch", src["_path"], f"{src.get('type')} cannot have {e['type']}"))
        kept_edges.append(e)
    edges = kept_edges
    for sid, st in stubs.items():
        nodes[sid] = st
        bodies[sid] = ""

    out_ = collections.defaultdict(lambda: collections.defaultdict(list))
    in_ = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in edges:
        out_[e["source"]][e["type"]].append(e)
        in_[e["target"]][e["type"]].append(e)
    degree = collections.Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    # ---- lint: content rules ----
    for nid, m in nodes.items():
        if m.get("stub"):
            continue
        t, rel = m.get("type"), m["_path"]
        typed = {k for k in out_[nid] if k != "mentions"} | {k for k in in_[nid] if k != "mentions"}
        if t in ("method", "dataset", "idea", "decision") and not typed:
            warnings.append(("orphan-no-typed-edges", rel, nid))
        if t in ("finding", "constraint") and not typed:
            (infos if m.get("source") else warnings).append(("no-typed-edges (anchored to its source document)", rel, nid))
        if t == "experiment":
            if m.get("status") not in ("reference",) and not out_[nid].get("uses_method"):
                warnings.append(("experiment-without-method", rel, nid))
            if m.get("status") in ("kept", "dropped", "closed", "superseded") and num(m.get("delta_num")) is None \
                    and not m.get("ambiguous"):
                infos.append(("experiment-without-delta_num", rel, nid))
            if m.get("status") == "superseded" and not out_[nid].get("superseded_by"):
                warnings.append(("superseded-without-target", rel, nid))
            if m.get("ambiguous"):
                infos.append(("ambiguous-number", rel, clean(m.get("ambiguous_note") or "see note", 140)))
        if t == "idea":
            if not m.get("motivated_by"):
                warnings.append(("idea-without-motivation", rel, nid))
            if m.get("status") in ("partially-tested", "closed", "dropped") and not out_[nid].get("tested_by"):
                warnings.append(("tested-idea-without-tested_by", rel, nid))
            if not m.get("novelty_check"):
                infos.append(("idea-without-literature-check", rel, nid))
        if t == "finding" and not in_[nid].get("supports_finding") and not m.get("source"):
            warnings.append(("finding-without-evidence", rel, nid))
        if t == "constraint" and m.get("status", "active") == "active" and not m.get("constrains") and m.get("scope") != "global":
            warnings.append(("constraint-without-targets", rel, nid))
        if t == "paper" and not (set(out_[nid]) - {"mentions"}) and not (set(in_[nid]) - {"mentions"}):
            infos.append(("paper-without-typed-edges", rel, nid))
        if t == "idea" and m.get("status") in (schema.get("ranking") or {}).get("open_statuses", []):
            missing = [f for f in ("owner_lane", "cost", "potential_metric") if m.get(f) in (None, "", [])]
            if missing:
                infos.append(("open-idea-without-ranking-fields", rel, ", ".join(missing)))

    OUT.mkdir(exist_ok=True)
    keep = lambda m: {k: v for k, v in m.items() if not k.startswith("_") or k == "_path"}
    generated = datetime.datetime.now().isoformat(timespec="seconds")
    graph = {"generated": generated, "root": str(ROOT),
             "nodes": [dict(keep(m), degree=degree[n]) for n, m in nodes.items()],
             "edges": edges}
    json.dump(graph, open(OUT / "graph.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    write_bib(nodes)

    by_type = collections.Counter(m.get("type") for m in nodes.values() if not m.get("stub"))
    ek = collections.Counter(x[0] for x in errors)
    wk = collections.Counter(x[0] for x in warnings)
    ik = collections.Counter(x[0] for x in infos)
    with open(OUT / "lint.md", "w", encoding="utf-8") as f:
        f.write(f"# Lint report\n\nGenerated {generated}\n\n")
        f.write("Nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items(), key=str))
                + f"; stubs={len(stubs)}\n\nEdges: {len(edges)}\n\n")
        for title, rows, counts in (("Errors (blocking)", errors, ek), ("Warnings", warnings, wk),
                                    ("Info (tracked, not errors)", infos, ik)):
            f.write(f"## {title}\n\n" + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none") + "\n\n")
            for kind, rel, detail in sorted(rows, key=lambda r: (r[0], str(r[1]), str(r[2]))):
                f.write(f"- **{kind}** `{rel}` {detail}\n")
            f.write("\n")
        orphans = sorted((m.get("type"), nid) for nid, m in nodes.items() if not m.get("stub")
                         and not (set(out_[nid]) - {"mentions"}) and not (set(in_[nid]) - {"mentions"}))
        f.write(f"## Orphans (no typed edge in or out)\n\n{len(orphans)} nodes. Body mentions do not count.\n\n")
        for typ, nid in orphans:
            f.write(f"- `{nid}` ({typ})\n")
        f.write("\n")

    write_index(nodes)
    write_views(nodes, edges, out_, in_, schema)
    print(f"nodes={sum(by_type.values())} {dict(sorted(by_type.items(), key=str))} stubs={len(stubs)} "
          f"edges={len(edges)} errors={dict(ek)} warnings={sum(wk.values())} info={sum(ik.values())}")
    return 1 if any(k in ("broken-link", "broken-body-link", "yaml-error", "duplicate-id", "missing-id",
                          "id-filename-mismatch") for k in ek) else 0


def write_bib(nodes):
    def c(v):
        return str(v).replace("{", "").replace("}", "")
    with open(OUT / "references.bib", "w", encoding="utf-8") as f:
        for n in sorted((m for m in nodes.values() if m.get("type") == "paper" and not m.get("stub")),
                        key=lambda x: x["id"]):
            authors = [a for a in (n.get("authors") or []) if a and a != "et al."]
            fields = {"title": n.get("title"), "author": " and ".join(map(str, authors)), "year": n.get("year"),
                      "journal": n.get("venue"), "doi": n.get("doi"), "eprint": n.get("arxiv"), "url": n.get("url")}
            body = ",\n".join(f"  {k} = {{{c(v)}}}" for k, v in fields.items() if v)
            f.write(f"@misc{{{n['id']},\n{body}\n}}\n\n")


def write_index(nodes):
    groups = collections.defaultdict(list)
    for m in nodes.values():
        if not m.get("stub"):
            groups[m.get("type", "unknown")].append(m)
    L = ["---", "id: index", "type: index", "---", "", "# CLEAR-EC KG index", "",
         "Generated by `scripts/build_graph.py`; do not edit. Start with `purpose.md` and `CLAUDE.md`.", "",
         "Views: [[tried-and-closed]] · [[kept-stack]] · [[experiments-by-lane]] · [[open-ideas]] · [[open-ideas-ranked]] · [[findings-list]]"
         " · [[constraint-list]] · [[decision-log]] · [[papers-table]]", "",
         "Counts: " + ", ".join(f"{t} {len(groups[t])}" for t in ORDER + sorted(set(groups) - set(ORDER)) if groups.get(t)), ""]
    for t in ORDER + sorted(set(groups) - set(ORDER)):
        if t not in groups:
            continue
        L += [f"## {t} ({len(groups[t])})", ""]
        items = groups[t]
        if t == "paper":
            items = sorted(items, key=lambda x: (-(int(x.get("year") or 0)), x["id"]))
        elif t == "experiment":
            items = sorted(items, key=lambda x: (lane_key(x.get("lane")), str(x.get("direction")), x["id"]))
        else:
            items = sorted(items, key=lambda x: x["id"])
        cur_lane = None
        for m in items:
            if t == "experiment" and m.get("lane") != cur_lane:
                cur_lane = m.get("lane")
                L += ["", f"### lane: {cur_lane}", ""]
            lab = m.get("title") or m.get("name") or m["id"]
            extra = f" ({m.get('year')})" if m.get("year") else ""
            if t == "experiment":
                extra = f" — {m.get('status')} · {', '.join(map(str, m.get('metrics') or []))} · folds {m.get('folds')} · {clean(m.get('delta'), 60)}"
            elif m.get("status"):
                extra += f" — {m.get('status')}"
            summ = clean(m.get("tldr") or m.get("summary"), 130)
            L.append(f"- [[{m['id']}]] {clean(lab, 110)}{extra}" + (f" — {summ}" if summ and t != "experiment" else ""))
        L.append("")
    (WIKI / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_views(nodes, edges, out_, in_, schema=None):
    vdir = WIKI / "_views"
    vdir.mkdir(exist_ok=True)

    def head(vid, title, intro):
        return ["---", f"id: {vid}", "type: index", "---", "", f"# {title}", "",
                "Generated by `scripts/build_graph.py`; edit the notes, not this page.", "", intro, ""]

    of = lambda t: [m for m in nodes.values() if m.get("type") == t and not m.get("stub")]
    links = lambda ids: ", ".join(f"[[{i}]]" for i in ids) or "-"
    tgts = lambda nid, k: [e["target"] for e in out_[nid].get(k, [])]
    srcs = lambda nid, k: [e["source"] for e in in_[nid].get(k, [])]
    exps = of("experiment")
    amb = lambda m: " (ambiguous)" if m.get("ambiguous") else ""

    # --- tried and closed ---
    L = head("tried-and-closed", "Tried and closed",
             "Check this page before spending GPU time. A direction is listed when its experiments are `closed` "
             "(closed as tested). \"Best\" is the try with the lowest `delta_num` against its own baseline; a "
             "negative delta is better. Compare a new idea with the baseline and folds, not with the bare number. "
             "Closed means closed as tested: the same test on the same inputs is not worth rerunning, a clearly "
             "different version is fine ([[con-closed-means-closed-as-tested]]).")
    by_dir = collections.defaultdict(list)
    for m in exps:
        by_dir[str(m.get("direction") or "(no direction)")].append(m)
    closed_dirs = []
    for d, ms in by_dir.items():
        cl = [m for m in ms if m.get("status") == "closed"]
        if cl:
            closed_dirs.append((d, ms, cl))
    for metric_group in ("CD", "CV", "HEX", "all metrics / other"):
        rows = []
        for d, ms, cl in closed_dirs:
            mets = sorted({str(x) for m in cl for x in (m.get("metrics") or [])})
            prim = {str(m.get("primary_metric") or (m.get("metrics") or ["?"])[0]) for m in cl}
            grp = next(iter(prim)) if len(prim) == 1 and next(iter(prim)) in ("CD", "CV", "HEX") else "all metrics / other"
            if grp != metric_group:
                continue
            withd = [m for m in cl if num(m.get("delta_num")) is not None]
            clear = [m for m in withd if not m.get("ambiguous")]
            best = min(clear or withd, key=lambda m: num(m["delta_num"])) if withd else cl[0]
            still_open = [m["id"] for m in ms if m.get("status") in ("kept", "running")]
            rows.append((lane_key(best.get("lane")), d, best, cl, still_open))
        if not rows:
            continue
        L += [f"## {metric_group}", "",
              "| Direction | Lane | Best try | Metric | Folds | Best value | Delta | Baseline | Tries closed | Source |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, d, b, cl, still_open in sorted(rows, key=lambda r: (r[0], r[1])):
            note = f" (kept/running in same direction: {links(still_open)})" if still_open else ""
            L.append(f"| {clean(d, 70)}{note} | {b.get('lane')} | [[{b['id']}]]{amb(b)} | "
                     f"{clean(b.get('primary_metric') or ', '.join(map(str, b.get('metrics') or [])))} | {clean(b.get('folds'))} | "
                     f"{clean(b.get('value'), 60)} | {clean(b.get('delta'), 60)} | {clean(b.get('baseline'), 90)} | "
                     f"{len(cl)}: {links(sorted(m['id'] for m in cl))} | {clean(b.get('source'), 70)} |")
        L.append("")
    closed_ideas = [m for m in of("idea") if m.get("status") in ("closed", "dropped")]
    if closed_ideas:
        L += ["## Closed ideas", "", "| Idea | Status | Tested by (closeness) |", "|---|---|---|"]
        for m in sorted(closed_ideas, key=lambda x: x["id"]):
            tb = [f"[[{e['target']}]] ({e.get('closeness', '?')})" for e in out_[m["id"]].get("tested_by", [])]
            L.append(f"| [[{m['id']}]] {clean(m.get('name'), 80)} | {m.get('status')} | {', '.join(tb) or '-'} |")
        L.append("")
    dropped = [m for m in exps if m.get("status") in ("dropped", "superseded")]
    if dropped:
        L += ["## Single versions dropped or superseded (direction not closed)", "",
              "These failed or were replaced once; the direction may still be open (three clearly different "
              "versions before dropping a direction, [[con-three-versions-before-drop]]).", "",
              "| Experiment | Lane | Status | Metric | Folds | Value | Delta | Baseline |", "|---|---|---|---|---|---|---|---|"]
        for m in sorted(dropped, key=lambda x: (lane_key(x.get("lane")), x["id"])):
            L.append(f"| [[{m['id']}]]{amb(m)} | {m.get('lane')} | {m.get('status')} | "
                     f"{clean(m.get('primary_metric') or ', '.join(map(str, m.get('metrics') or [])))} | {clean(m.get('folds'))} | "
                     f"{clean(m.get('value'), 50)} | {clean(m.get('delta'), 50)} | {clean(m.get('baseline'), 80)} |")
        L.append("")
    (vdir / "tried-and-closed.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- kept stack per lane ---
    L = head("kept-stack", "Kept steps and running work per lane",
             "Kept gains stay kept ([[con-kept-gains-stay-kept]]). Each row's delta is against the baseline named "
             "in that row; deltas against different baselines do not add up.")
    for lane in LANE_ORDER:
        ms = [m for m in exps if m.get("lane") == lane and m.get("status") in ("kept", "running", "reference")]
        if not ms:
            continue
        L += [f"## {lane}", "", "| Experiment | Status | Date | Metric | Folds | Value | Delta | Baseline |",
              "|---|---|---|---|---|---|---|---|"]
        order = {"reference": 0, "kept": 1, "running": 2}
        for m in sorted(ms, key=lambda x: (order.get(x.get("status"), 9), str(x.get("date")), x["id"])):
            L.append(f"| [[{m['id']}]]{amb(m)} | {m.get('status')} | {m.get('date')} | "
                     f"{clean(m.get('primary_metric') or ', '.join(map(str, m.get('metrics') or [])))} | {clean(m.get('folds'))} | "
                     f"{clean(m.get('value'), 50)} | {clean(m.get('delta'), 60)} | {clean(m.get('baseline'), 90)} |")
        L.append("")
    (vdir / "kept-stack.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- all experiments by lane ---
    L = head("experiments-by-lane", "All experiments by lane", "One row per experiment node.")
    for lane in LANE_ORDER + sorted({str(m.get("lane")) for m in exps} - set(LANE_ORDER)):
        ms = [m for m in exps if str(m.get("lane")) == lane]
        if not ms:
            continue
        L += [f"## {lane} ({len(ms)})", "", "| Experiment | Status | Direction | Metric | Folds | Value | Delta | Baseline | Source |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(ms, key=lambda x: (str(x.get("direction")), x["id"])):
            L.append(f"| [[{m['id']}]]{amb(m)} | {m.get('status')} | {clean(m.get('direction'), 50)} | "
                     f"{clean(m.get('primary_metric') or ', '.join(map(str, m.get('metrics') or [])))} | {clean(m.get('folds'))} | "
                     f"{clean(m.get('value'), 50)} | {clean(m.get('delta'), 50)} | {clean(m.get('baseline'), 70)} | "
                     f"{clean(m.get('source'), 60)} |")
        L.append("")
    (vdir / "experiments-by-lane.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- ideas ---
    L = head("open-ideas", "Ideas by status",
             "Before running an idea: read its `tested_by` experiments, the constraints, and the literature links.")
    order = ["never-tried", "open", "partially-tested", "running", "closed", "dropped"]
    for st in order:
        ms = [m for m in of("idea") if m.get("status") == st]
        if not ms:
            continue
        L += [f"## {st} ({len(ms)})", "", "| Idea | Aims at | Tested by (closeness) | Motivated by | Literature | Constraints |",
              "|---|---|---|---|---|---|"]
        for m in sorted(ms, key=lambda x: x["id"]):
            tb = [f"[[{e['target']}]] ({e.get('closeness', '?')})" for e in out_[m["id"]].get("tested_by", [])]
            lit = sorted(set(srcs(m["id"], "evidence_for") + srcs(m["id"], "evidence_against") + tgts(m["id"], "literature")
                             + [t for t in tgts(m["id"], "builds_on") + tgts(m["id"], "motivated_by")
                                if nodes.get(t, {}).get("type") == "paper"]))
            L.append(f"| [[{m['id']}]] {clean(m.get('name'), 70)} | {clean(', '.join(map(str, m.get('aims_at') or [])))} | "
                     f"{', '.join(tb) or '-'} | {links(tgts(m['id'], 'motivated_by'))} | {links(lit)} | "
                     f"{links(srcs(m['id'], 'constrains'))} |")
        L.append("")
    (vdir / "open-ideas.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- findings ---
    L = head("findings-list", "Findings", "Facts that held up, with the experiments behind them.")
    L += ["| Finding | Status | Metric | Supported by | Contradicted by | Contradicts / qualifies | Source |", "|---|---|---|---|---|---|---|"]
    for m in sorted(of("finding"), key=lambda x: (x.get("status") != "robust", x["id"])):
        against = srcs(m['id'], 'contradicts_finding') + srcs(m['id'], 'contradicts')
        mine = [f"contradicts [[{t}]]" for t in tgts(m['id'], 'contradicts')] + [f"qualifies [[{t}]]" for t in tgts(m['id'], 'qualifies')]
        L.append(f"| [[{m['id']}]] {clean(m.get('name'), 90)} | {m.get('status')} | "
                 f"{clean(', '.join(map(str, m.get('metrics') or [])))} | {links(srcs(m['id'], 'supports_finding'))} | "
                 f"{links(against)} | {', '.join(mine) or '-'} | {clean(m.get('source'), 80)} |")
    (vdir / "findings-list.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- constraints ---
    L = head("constraint-list", "Constraints (PI rules first)", "Run `kg_query.py constraints` before proposing work.")
    L += ["| Constraint | Status | Origin | Since | Limits | Source |", "|---|---|---|---|---|---|"]
    for m in sorted(of("constraint"), key=lambda x: (x.get("origin") != "user", x["id"])):
        L.append(f"| [[{m['id']}]] {clean(m.get('name'), 90)} | {m.get('status')} | {m.get('origin')} | "
                 f"{m.get('date', '')} | {links(tgts(m['id'], 'constrains'))} | {clean(m.get('source'), 70)} |")
    (vdir / "constraint-list.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- decisions ---
    L = head("decision-log", "Decision log", "Newest first. Superseded decisions stay on record.")
    L += ["| Date | Decision | Status | Deciders | Supersedes |", "|---|---|---|---|---|"]
    for m in sorted(of("decision"), key=lambda x: (str(x.get("date")), x["id"]), reverse=True):
        L.append(f"| {m.get('date')} | [[{m['id']}]] {clean(m.get('name'), 90)} | {m.get('status')} | "
                 f"{clean(', '.join(map(str, m.get('deciders') or [])))} | {links(tgts(m['id'], 'supersedes'))} |")
    (vdir / "decision-log.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- papers ---
    L = head("papers-table", "Papers", "Paper notes are written by the paper agent (EHR-KG paper template). "
             "Links to ideas, findings and methods come from their typed edges.")
    L += ["| Paper | Year | Relevance | Linked project nodes (both directions) | TL;DR |", "|---|---|---|---|---|"]
    for m in sorted(of("paper"), key=lambda x: (-(int(x.get("year") or 0)), x["id"])):
        proj = sorted({e["target"] for k, es in out_[m["id"]].items() if k != "mentions" for e in es
                       if nodes.get(e["target"], {}).get("type") in ("idea", "finding", "method", "decision", "experiment", "dataset")
                       and not nodes.get(e["target"], {}).get("stub")})
        proj = sorted(set(proj) | {e["source"] for e in in_[m["id"]].get("literature", [])})
        rel = m.get("relevance") or m.get("relevance_clearec") or m.get("relevance_clear_ec") or m.get("relevance_circe") or ""
        L.append(f"| [[{m['id']}]] {clean(m.get('title'), 80)} | {m.get('year', '')} | {rel} | {links(proj)} | "
                 f"{clean(m.get('tldr'), 120)} |")
    stub_rows = [m for m in nodes.values() if m.get("stub")]
    if stub_rows:
        L += ["", "## Ids that paper notes link to but no note exists", "",
              "Create the node (method-, ds-, idea-, find-) if it matters for the project, or leave it.", ""]
        for m in sorted(stub_rows, key=lambda x: x["id"]):
            L.append(f"- `{m['id']}` ({m['type']}) — from {', '.join(sorted({e['source'] for es in in_[m['id']].values() for e in es}))}")
    (vdir / "papers-table.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- open ideas ranked by the share of the gap to first place they could close ---
    rk = (schema or {}).get("ranking") or {}
    gaps = rk.get("gap_to_first") or {}
    total = sum(gaps.values()) or 1.0
    open_st = rk.get("open_statuses") or ["open", "never-tried", "partially-tested", "running"]
    order_close = {"same": 0, "close": 1, "related": 2}

    def closest(nid):
        tb = sorted(out_[nid].get("tested_by", []), key=lambda e: order_close.get(e.get("closeness"), 3))
        if not tb:
            return "-"
        e = tb[0]
        x = nodes.get(e["target"], {})
        return (f"[[{e['target']}]] ({e.get('closeness', '?')}; {x.get('status', '?')}; "
                f"{clean(x.get('primary_metric') or '', 5)} {clean(x.get('delta'), 45)})")

    def share(m):
        g = num(m.get("potential_gain"))
        met = str(m.get("potential_metric") or "")
        if g is None:
            return None, None
        if met == "all":
            return 3 * g / total, None
        return g / total, (g / gaps[met] if gaps.get(met) else None)

    def cost_key(m):
        h = num(m.get("cost_hours"))
        return (1 if str(m.get("cost_unit")) == "GPU-h" else 0, h if h is not None else 1e9)

    ideas = [m for m in of("idea") if m.get("status") in open_st]
    ranked = sorted(ideas, key=lambda m: (share(m)[0] is None, -(share(m)[0] or 0), cost_key(m), m["id"]))
    L = head("open-ideas-ranked", "Open ideas ranked by the share of the gap to first place they could close",
             "Gaps are projected-test gaps to first place if submission 1's test-minus-OOF repeats ("
             + clean(rk.get("source"), 200) + "): " + ", ".join(f"{k} {v}" for k, v in gaps.items())
             + f"; sum {total:.2f} in metric units. Each metric is a third of the score, so lowering one metric by g lowers "
             "the mean score by g/3. **Potential gains are estimates written on each idea (`potential_gain`, "
             "`potential_basis`), not measured numbers**; ideas without an estimate come last. Share of total gap = "
             "gain / sum of gaps (x3 for ideas that lower every metric). Ties are broken by cost (CPU before GPU, then hours). "
             "Statuses included: " + ", ".join(open_st) + ". Closest tried = the idea's nearest `tested_by` (same > close > related).")
    L += ["| Rank | Idea | Status | Metric | Potential (estimate) | Share of metric gap | Share of total gap | Cost | Owner lane | Closest tried (closeness; status; delta) | Basis of the estimate |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, m in enumerate(ranked, 1):
        s_tot, s_met = share(m)
        g = num(m.get("potential_gain"))
        L.append(f"| {i} | [[{m['id']}]] {clean(m.get('name'), 70)} | {m.get('status')} | {clean(m.get('potential_metric'))} | "
                 f"{'-' if g is None else f'{g:.3f}'} | {'-' if s_met is None else f'{100 * s_met:.0f} %'} | "
                 f"{'-' if s_tot is None else f'{100 * s_tot:.1f} %'} | {clean(m.get('cost'), 70)} | {clean(m.get('owner_lane'))} | "
                 f"{closest(m['id'])} | {clean(m.get('potential_basis'), 160)} |")
    lit_closed = [m for m in of("idea") if m.get("status") in ("closed", "dropped") and m.get("origin") == "literature"]
    if lit_closed:
        L += ["", "## Literature ideas already closed in the KG", "",
              "| Idea | Status | Tested by (closeness) | Why closed |", "|---|---|---|---|"]
        for m in sorted(lit_closed, key=lambda x: x["id"]):
            tb = [f"[[{e['target']}]] ({e.get('closeness', '?')})" for e in out_[m["id"]].get("tested_by", [])]
            L.append(f"| [[{m['id']}]] {clean(m.get('name'), 70)} | {m.get('status')} | {', '.join(tb) or '-'} | "
                     f"{clean(m.get('potential_basis') or m.get('summary'), 160)} |")
    (vdir / "open-ideas-ranked.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
