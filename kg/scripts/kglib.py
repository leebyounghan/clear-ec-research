"""Shared helpers for the CLEAR-EC KG scripts (parse notes, resolve links, load the schema).

Needs PyYAML. If the interpreter running a script lacks it, the script re-executes itself with
/usr/bin/python3 (macOS system Python, which has PyYAML in the user site on this machine).
"""
import os
import pathlib
import re
import sys

try:
    import yaml  # noqa: F401
except ImportError:  # re-exec once with an interpreter that has PyYAML
    for _cand in ("/usr/bin/python3",):
        if os.path.exists(_cand) and os.path.realpath(_cand) != os.path.realpath(sys.executable) \
                and not os.environ.get("KG_REEXEC"):
            os.environ["KG_REEXEC"] = "1"
            os.execv(_cand, [_cand] + sys.argv)
    raise SystemExit("PyYAML not found. Run the KG scripts with /usr/bin/python3.")

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
OUT = ROOT / "graph"
LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
SKIP_DIRS = {"_templates", "_views"}
SKIP_FILES = {"index.md", "log.md", "overview.md"}

PREFIX_TYPE = {"method-": "method", "ds-": "dataset", "exp-": "experiment", "find-": "finding",
               "idea-": "idea", "dec-": "decision", "con-": "constraint", "syn-": "synthesis",
               "model-": "model", "concept-": "concept", "lim-": "limitation", "gap-": "gap",
               "code-": "code", "bench-": "benchmark"}


def load_schema():
    return yaml.safe_load(open(ROOT / "schema" / "edges.yaml", encoding="utf-8"))


def parse_note(path):
    """Return (meta, body). meta is None without frontmatter; {'_yaml_error': ...} on bad YAML."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    m = FM.match(text)
    if not m:
        return None, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {"_yaml_error": "frontmatter is not a mapping"}
    except yaml.YAMLError as exc:
        meta = {"_yaml_error": str(exc)}
    return meta, text[m.end():]


def note_paths():
    for path in sorted(WIKI.rglob("*.md")):
        if path.name in SKIP_FILES or SKIP_DIRS & set(path.parts):
            continue
        yield path


IDLIKE = re.compile(r"^((method|ds|exp|find|idea|dec|con|syn|model|concept|lim|gap|code|bench)-[a-z0-9_.-]+|[a-z]+\d{4}[a-z0-9]+)$")


def link_ids(value, strict=False):
    """Targets from a frontmatter value: '[[id]]', 'id', lists of either, or {target: ..., attrs}.
    Returns [(target_id, attrs_dict)]. strict=True (used for paper notes) ignores free-text values such as
    'Dense U-Net' or 'kernel-quantile-regression': only [[wikilinks]], prefixed ids (method-, ds-, idea-, ...) and
    citekeys (lastnameYEARword) count as links."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for it in items:
        if isinstance(it, dict):
            attrs = {k: v for k, v in it.items() if k != "target"}
            out += [(t, attrs) for t, _ in link_ids(it.get("target"), strict)]
            continue
        s = str(it)
        found = LINK.findall(s)
        if found:
            out += [(f.strip(), {}) for f in found]
        elif s.strip() and (not strict or IDLIKE.match(s.strip())):
            out.append((s.strip(), {}))
    return out


def guess_type(nid):
    for k, v in PREFIX_TYPE.items():
        if nid.startswith(k):
            return v
    return "paper"


def short(text, k=110):
    text = " ".join(str(text if text is not None else "").split())
    return text if len(text) <= k else text[:k - 1] + "…"


def label(n):
    if not n:
        return "?"
    return n.get("title") or n.get("name") or n.get("id")
