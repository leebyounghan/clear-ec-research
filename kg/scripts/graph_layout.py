#!/usr/bin/env python3
"""Deterministic map layout for the CLEAR-EC KG explorer (used by build_viewer.py).

Ported from /Users/byounghanlee/EHR-KG/scripts/graph_layout.py (2026-11-09). A force layout over the whole graph
collapses into a hairball because hub nodes (the training set, recipe A) pull everything to the centre, so the
layout works in steps:
1. Regions: papers go to a literature region by their `group`; experiments go to a lane region by `lane`;
   findings, ideas, decisions, constraints and syntheses go to "Project knowledge"; methods, datasets and anything
   else follow the regions of their neighbours, and nodes spread across many regions go to a shared region.
2. Packing: regions become discs sized by node count, placed so strongly linked regions sit next to each other.
3. Inside each disc: a force layout with hub-normalised edge weights and collision removal.
4. Backbone: for each node only its strongest link inside the region (plus its strongest lineage link).
Also computes short labels and label priority. Positions are stable for the same input.
"""
import collections
import math
import re

import numpy as np

SEED = 20261109
SPACING = 36.0  # target distance between neighbouring nodes, in map units

# id, label, kind, matcher. Paper regions match `group`; lane regions match experiment lanes (see _lane_region).
# A paper group that is not listed is not an error: those papers are placed by their neighbours.
REGIONS = [
    ("t1", "Endothelium segmentation", "papers", {"t1-endothelium-segmentation"}),
    ("t2", "Device morphometry", "papers", {"t2-device-morphometry"}),
    ("t3", "Challenge", "papers", {"t3-challenge"}),
    ("t4", "Cell foundation models", "papers", {"t4-cell-foundation"}),
    ("t5", "Weak supervision and counting", "papers", {"t5-weak-supervision"}),
    ("t6", "Noisy and MAPE regression", "papers", {"t6-noisy-regression"}),
    ("t7", "Domain shift", "papers", {"t7-domain-shift"}),
    ("t8", "Synthetic data", "papers", {"t8-synthetic"}),
    ("lane-cd", "CD lane", "lane", set()),
    ("lane-cv", "CV lane", "lane", set()),
    ("lane-hex", "HEX lane", "lane", set()),
    ("lane-bb", "Backbone lane", "lane", set()),
    ("lane-rc", "Root cause and references", "lane", set()),
    ("lane-old", "Old lanes and history", "lane", set()),
    ("project", "Project knowledge", "project", set()),
    ("shared", "Shared methods and data", "shared", set()),
]
PROJECT_TYPES = {"finding", "idea", "decision", "constraint", "synthesis"}
COMPAT_TYPES = {"model", "concept", "limitation", "gap", "code", "benchmark"}  # paper-template targets; usually stubs

EDGE_WEIGHT = {
    # experiments
    "stacked_on": 4, "superseded_by": 4, "baseline_ref": 2, "supports_finding": 2, "contradicts_finding": 2,
    "uses_method": 1.5, "uses_dataset": 0.5,
    # ideas, decisions, constraints
    "tested_by": 2.5, "motivated_by": 1.5, "builds_on": 1.5, "depends_on": 2, "decides": 2.5, "alternative": 1.5,
    "supersedes": 3, "constrains": 2, "literature": 1.5, "contradicts": 2,
    "evidence_for": 1.5, "evidence_against": 1.5, "qualifies": 2,
    # entities
    "variant_of": 3, "part_of": 3,
    # paper template
    "proposes": 5, "releases_code": 5, "implemented_in": 5, "same_lineage": 4, "extends": 4,
    "supports": 3, "disputes": 3, "critiques": 3, "compares_with": 2.5, "uses_model": 2, "evaluates_on": 1.5,
    "studies": 1.5, "discusses": 0.6, "has_limitation": 0.5, "raises": 1, "addresses": 1.5, "cites": 1,
    "related_to": 0.4,
}
DEFAULT_WEIGHT = 1.0  # a recorded (frontmatter) edge type not listed above
SOFT_WEIGHT = {"mentions": 0.2}  # placement inside a region only; never regions, never backbone
LINEAGE = {"stacked_on", "superseded_by", "baseline_ref", "supersedes", "variant_of", "extends", "same_lineage",
           "supports", "disputes", "qualifies", "critiques", "contradicts", "contradicts_finding"}
TYPE_BONUS = {"decision": 3, "synthesis": 2.5, "finding": 1.8, "idea": 1.5, "constraint": 1.2, "dataset": 0.6,
              "method": -0.3, "experiment": 0.0}
EXP_STATUS_BONUS = {"kept": 1.5, "reference": 2.0, "running": 1.0, "measurement": 0.3}
LANE_PREFIX = re.compile(r"^(.{1,24}?):\s+(.{10,})$")


def _lane_region(n):
    """Lane region for an experiment, by `lane` (then by id prefix); None when unknown."""
    lane = str(n.get("lane") or "").lower()
    nid = n["id"]
    if nid.startswith("exp-ref-"):
        return "lane-rc"
    for prefix, rid in (("cd", "lane-cd"), ("cv", "lane-cv"), ("hex", "lane-hex"), ("backbone", "lane-bb"),
                        ("bb", "lane-bb"), ("rootcause", "lane-rc"), ("coordinator", "lane-rc"),
                        ("gpu", "lane-old"), ("pre-lanes", "lane-old"), ("hist", "lane-old")):
        if lane.startswith(prefix):
            return rid
    for prefix, rid in (("exp-cd-", "lane-cd"), ("exp-cv-", "lane-cv"), ("exp-hex-", "lane-hex"),
                        ("exp-bb-", "lane-bb"), ("exp-rc-", "lane-rc"), ("exp-gpu", "lane-old"),
                        ("exp-hist-", "lane-old")):
        if nid.startswith(prefix):
            return rid
    return None


def _clean_name(name):
    name = str(name or "").split(" | ")[0]
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip() or name.strip()


def _clip(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut if len(cut) >= limit * 0.6 else text[:limit]).rstrip(" ,;:-/") + "…"


def short_label(n):
    t = n["type"]
    if t == "paper":
        authors = n.get("authors") or []
        last = ""
        if authors and not str(authors[0]).lower().startswith("et al"):
            last = re.sub(r"[^\w-]", "", str(authors[0]).split()[-1]) if str(authors[0]).split() else ""
        m = re.match(r"([a-z-]+)(\d{4})([a-z0-9]*)$", n["id"])
        if not last or len(last) < 2:
            last = (m.group(1).capitalize() if m else n["id"])
        base = f"{last} {n.get('year') or (m.group(2) if m else '')}".strip()
        word = m.group(3) if m else ""
        return f"{base} · {word}" if word and len(word) <= 14 else base
    name = str(n.get("name") or n.get("title") or n["id"])
    if t == "experiment":  # the region already names the lane: "CD: ConvNeXt B member" -> "ConvNeXt B member"
        m = LANE_PREFIX.match(name)
        if m:
            name = m.group(2)
            name = name[0].upper() + name[1:]
    return _clip(_clean_name(name), 34)


def _assign_regions(nodes, typed):
    by_group = {g: rid for rid, _, kind, groups in REGIONS if kind == "papers" for g in groups}
    region = {}
    for n in nodes:
        t = n["type"]
        if t in PROJECT_TYPES:
            region[n["id"]] = "project"
        elif t == "paper" and n.get("group") in by_group:
            region[n["id"]] = by_group[n["group"]]
        elif t == "experiment":
            lane = _lane_region(n)
            if lane:
                region[n["id"]] = lane
    fixed = set(region)
    size = collections.Counter(region.values())
    ids = sorted(n["id"] for n in nodes)
    for _ in range(4):  # entities linked only to other entities pick up a region on later passes
        for v in ids:
            if v in fixed:
                continue
            votes = collections.Counter()
            for u, w in typed[v]:
                r = region.get(u)
                if r and r not in ("project", "shared"):
                    votes[r] += w / math.sqrt(max(size[r], 1))
            if not votes:
                continue
            top, val = votes.most_common(1)[0]
            region[v] = "shared" if len(typed[v]) >= 8 and val / sum(votes.values()) < 0.5 else top
    for v in ids:  # nothing but project links, or no links at all
        if v not in region:
            region[v] = "project" if any(region.get(u) == "project" for u, _ in typed[v]) else "shared"
    return region


def _pack_regions(radius, meta, rng):
    """Discs that do not overlap; linked regions adjacent; wider than tall to suit a landscape stage."""
    ids = sorted(radius)
    n = len(ids)
    if n == 1:
        return {ids[0]: np.zeros(2)}
    r = np.array([radius[i] for i in ids])
    w = np.zeros((n, n))
    for (a, b), v in meta.items():
        i, j = ids.index(a), ids.index(b)
        w[i, j] = w[j, i] = v
    w = w / (w.max() or 1)
    margin = 18.0
    scale = r.sum() / 2
    best, best_cost = None, math.inf
    for _ in range(24):
        pos = rng.normal(size=(n, 2)) * scale * np.array([1.4, 0.8])
        for it in range(900):
            d = pos[:, None, :] - pos[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
            np.fill_diagonal(dist, 1e9)
            gap = dist - (r[:, None] + r[None, :] + margin)
            force = -(w[..., None] * d / dist[..., None]).sum(1) * 6.0
            force -= pos * np.array([0.012, 0.035])  # gravity is stronger vertically, so the map spreads sideways
            overlap = np.minimum(gap, 0)
            force -= (overlap[..., None] * d / dist[..., None]).sum(1) * 0.6
            pos += force * (1.0 - it / 900) * 0.8
        for _ in range(300):  # remove any overlap left
            d = pos[:, None, :] - pos[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
            np.fill_diagonal(dist, 1e9)
            overlap = np.minimum(dist - (r[:, None] + r[None, :] + margin), 0)
            pos -= (overlap[..., None] * d / dist[..., None]).sum(1) * 0.5
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1))
        span = (pos + r[:, None]).max(0) - (pos - r[:, None]).min(0)
        cost = (w * dist).sum() / 2 + span[0] * span[1] / scale + abs(span[0] / span[1] - 1.55) * scale
        if cost < best_cost:
            best, best_cost = pos.copy(), cost
    best -= ((best + r[:, None]).max(0) + (best - r[:, None]).min(0)) / 2
    return {i: best[k] for k, i in enumerate(ids)}


def _soft_spheres(pos, spacing):
    """Linear repulsion between nodes closer than 2 * spacing; settles into even spacing without crowding the rim."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
    np.fill_diagonal(dist, 1e9)
    push = np.maximum(2 * spacing - dist, 0) / (2 * spacing)
    return (push[..., None] * d / dist[..., None]).sum(1)


def _layout_disc(ids, sizes, imp, outer, links, pull, R, rng):
    """Force layout inside a disc of radius R, centred on the origin.

    Outer nodes (stubs, compatibility types) are placed second, in the gaps, so hiding them moves nothing.
    """
    n = len(ids)
    if n == 1:
        return np.zeros((1, 2))
    a, b, w = links if links else (np.array([], int), np.array([], int), np.array([]))
    limit = R - 26
    primary = np.where(outer == 0)[0]
    secondary = np.where(outer > 0)[0]
    if len(primary) == 0:
        primary, secondary = secondary, primary
    golden = math.pi * (3 - math.sqrt(5))
    pos = np.zeros((n, 2))
    for k, i in enumerate(primary[np.argsort(-imp[primary], kind="stable")]):  # important nodes start at the centre
        rad = limit * math.sqrt((k + 0.5) / len(primary))
        pos[i] = rad * math.cos(k * golden), rad * math.sin(k * golden)
    spacing = math.sqrt(math.pi * limit * limit / len(primary)) * 0.92

    def step(active, spacing, t, spring_rest):
        force = np.zeros_like(pos)
        force[active] = _soft_spheres(pos[active], spacing) * spacing * 0.5
        if len(a):
            vec = pos[b] - pos[a]
            length = np.sqrt((vec ** 2).sum(-1)) + 1e-6
            spring = (w * (length - spring_rest) / length)[:, None] * vec * 0.05
            np.add.at(force, a, spring)
            np.add.at(force, b, -spring)
        force += pull * 2.5  # lean toward regions this node is linked to
        moved = np.zeros_like(pos)
        moved[active] = np.clip(force[active], -8 * t - 0.5, 8 * t + 0.5) * 0.5
        pos[:] += moved
        rad = np.sqrt((pos ** 2).sum(-1)) + 1e-6
        pos[:] *= np.minimum(limit / rad, 1.0)[:, None]

    iters = 480
    for it in range(iters):
        step(primary, spacing, 1 - it / iters, spacing * 0.9)

    if len(secondary):
        index = collections.defaultdict(list)
        for x, y, weight in zip(a, b, w):
            index[int(x)].append((int(y), weight))
            index[int(y)].append((int(x), weight))
        placed = set(primary.tolist())
        for i in secondary[np.argsort(-imp[secondary], kind="stable")]:  # next to what they connect to
            near = [(j, wt) for j, wt in index[int(i)] if j in placed]
            if near:
                total = sum(wt for _, wt in near)
                pos[i] = sum(pos[j] * wt for j, wt in near) / total
            else:
                angle = rng.uniform(0, 2 * math.pi)
                pos[i] = limit * 0.92 * math.cos(angle), limit * 0.92 * math.sin(angle)
            pos[i] += rng.normal(scale=6.0, size=2)
            placed.add(int(i))
        everyone = np.arange(n)
        fixed = pos[primary].copy()
        dense = math.sqrt(math.pi * limit * limit / n) * 0.8
        for it in range(240):
            step(everyone, dense, 1 - it / 240, dense)
            pos[primary] = fixed

    min_d = (sizes[:, None] + sizes[None, :]) / 2 + 9
    mobility = np.where(outer > 0, 1.0, 0.25)[:, None]
    for _ in range(140):  # collision removal
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
        np.fill_diagonal(dist, 1e9)
        overlap = np.minimum(dist - min_d, 0)
        pos -= (overlap[..., None] * d / dist[..., None]).sum(1) * 0.3 * mobility
        rad = np.sqrt((pos ** 2).sum(-1)) + 1e-6
        pos *= np.minimum(limit / rad, 1.0)[:, None]
    return pos


def compute(nodes, edges):
    """Add x, y, size, region, short, imp to nodes; set bb=1 on backbone edges; return the region list."""
    rng = np.random.default_rng(SEED)
    by_id = {n["id"]: n for n in nodes}
    typed = collections.defaultdict(list)
    soft = []
    degree = collections.Counter()
    strength = collections.Counter()
    usable = []
    for i, e in enumerate(edges):
        s, t, kind = e["source"], e["target"], e["type"]
        if s not in by_id or t not in by_id or s == t:
            continue
        if kind in SOFT_WEIGHT or e.get("origin") == "body":
            soft.append((s, t, SOFT_WEIGHT.get(kind, 0.2)))
            continue
        w = EDGE_WEIGHT.get(kind, DEFAULT_WEIGHT)
        typed[s].append((t, w))
        typed[t].append((s, w))
        strength[s] += w
        strength[t] += w
        usable.append((i, s, t, kind, w))
        degree[s] += 1
        degree[t] += 1
    norm = lambda u, v, w: w / math.sqrt(max(strength[u], 1) * max(strength[v], 1))

    region = _assign_regions(nodes, typed)
    is_outer = lambda n: bool(n.get("stub")) or n["type"] in COMPAT_TYPES

    for n in nodes:
        td = degree[n["id"]]
        t = n["type"]
        n["region"] = region[n["id"]]
        n["short"] = short_label(n)
        n["size"] = round(7 + min(15, 2.0 * math.sqrt(td)) + (2 if t in PROJECT_TYPES else 0)
                          - (2 if is_outer(n) else 0), 1)
        bonus = TYPE_BONUS.get(t, 0)
        if t == "paper":
            rel = n.get("relevance_clearec") or n.get("relevance_circe")
            bonus += {"high": 1.5, "medium": 0.5}.get(rel, 0) + (0.5 if n.get("tier") == "core" else 0)
        if t == "experiment":
            bonus += EXP_STATUS_BONUS.get(n.get("status"), 0)
        if t == "constraint" and n.get("scope") == "global":
            bonus += 1
        if is_outer(n):
            bonus -= 2
        n["imp"] = round(math.log1p(td) + bonus, 3)

    members = collections.defaultdict(list)
    for n in sorted(nodes, key=lambda n: n["id"]):
        members[n["region"]].append(n["id"])
    weight_of = lambda v: 0.4 if is_outer(by_id[v]) else 1.0  # discs are sized for real notes
    radius = {r: max(70.0, SPACING * math.sqrt(sum(weight_of(v) for v in ids)) * 0.8) + 18 for r, ids in members.items()}

    meta = collections.Counter()
    for _, s, t, _, w in usable:
        rs, rt = region[s], region[t]
        if rs != rt:
            meta[tuple(sorted((rs, rt)))] += norm(s, t, w)
    centers = _pack_regions(radius, meta, rng)

    for rid in sorted(members):
        ids = members[rid]
        index = {v: k for k, v in enumerate(ids)}
        weights = collections.Counter()
        pull = np.zeros((len(ids), 2))
        for _, s, t, _, w in usable:
            nw = norm(s, t, w)
            if s in index and t in index:
                weights[(index[s], index[t])] += nw
            elif s in index or t in index:  # lean toward the neighbour's region, so links across discs stay short
                inside, other = (s, t) if s in index else (t, s)
                vec = centers[region[other]] - centers[rid]
                pull[index[inside]] += nw * vec / (np.linalg.norm(vec) + 1e-6)
        for s, t, w in soft:
            if s in index and t in index:
                weights[(index[s], index[t])] += norm(s, t, w)
        links = None
        if weights:
            pairs = sorted(weights.items())
            a = np.array([p[0][0] for p in pairs])
            b = np.array([p[0][1] for p in pairs])
            w = np.array([p[1] for p in pairs])
            links = (a, b, np.minimum(w / (np.percentile(w, 90) or 1), 1.5))
        sizes = np.array([by_id[v]["size"] for v in ids])
        imp = np.array([by_id[v]["imp"] for v in ids])
        outer = np.array([1.0 if is_outer(by_id[v]) else 0.0 for v in ids])
        pos = _layout_disc(ids, sizes, imp, outer, links, pull, radius[rid], rng)
        for v, p in zip(ids, pos):
            by_id[v]["x"], by_id[v]["y"] = round(float(p[0] + centers[rid][0]), 1), round(float(p[1] + centers[rid][1]), 1)

    best = {}
    best_lineage = {}
    for i, s, t, kind, w in usable:
        if region[s] != region[t]:
            continue
        nw = norm(s, t, w)
        for u in (s, t):
            if nw > best.get(u, (0, None))[0]:
                best[u] = (nw, i)
            if kind in LINEAGE and nw > best_lineage.get(u, (0, None))[0]:
                best_lineage[u] = (nw, i)
    for table in (best, best_lineage):
        for _, i in table.values():
            edges[i]["bb"] = 1

    return [{"id": rid, "label": label, "kind": kind, "x": round(float(centers[rid][0]), 1),
             "y": round(float(centers[rid][1]), 1), "r": round(radius[rid], 1), "n": len(members[rid])}
            for rid, label, kind, _ in REGIONS if rid in members]


if __name__ == "__main__":
    import json
    import pathlib

    g = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "graph" / "graph.json", encoding="utf-8"))
    regions = compute(g["nodes"], g["edges"])
    counts = collections.Counter((n["region"], n["type"]) for n in g["nodes"])
    for r in regions:
        kinds = ", ".join(f"{t}:{c}" for (rr, t), c in sorted(counts.items(), key=lambda kv: -kv[1]) if rr == r["id"])
        print(f"{r['label']:<30} n={r['n']:<4} r={r['r']:<6} ({r['x']}, {r['y']})  {kinds}")
    print("backbone edges:", sum(1 for e in g["edges"] if e.get("bb")))
    print("shared:", [n["id"] for n in g["nodes"] if n["region"] == "shared"][:30])
    print("sample labels:", [n["short"] for n in g["nodes"]][:12])
