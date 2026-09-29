"""Collapse per-tag rows to one row per model + size (e.g. gemma4:12b), with per-host fit."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DROP_TAG = re.compile(r"mlx|nvfp4|mxfp8")
# Leading size token of a tag: 12b, e4b, 0.6b, 270m ("26b-a4b-it-q8_0" -> "26b").
SIZE_TOKEN = re.compile(r"^(e?\d+(?:\.\d+)?[bm])(?=$|-)", re.I)
# Fine-tunes that get their own row instead of folding into the base size ("27b-coding").
VARIANTS = ("coding",)
# Columns the user may edit; their edits survive re-runs.
EDITABLE = ("run_on",)

BY_MODEL_COLUMNS = [
    "name", "representative_tag", "run_on", "fits_min_gb", "fit_note", "fit_warn", "pulled_on", "family",
    "parameter_size", "quantization", "context_length", "disk_size_gb", "size_source", "tools", "vision", "pull_count",
    "first_published_approx", "all_tags", "url",
]


def _label_of(tag):
    """'35b-a3b-coding-bf16' -> '35b-coding'; 'latest' -> ''."""
    m = SIZE_TOKEN.match(tag)
    if not m:
        return ""
    variant = next((v for v in VARIANTS if re.search(rf"(^|-){v}($|-)", tag)), "")
    return m.group(1).lower() + (f"-{variant}" if variant else "")


def _assign_labels(rows):
    """Label for each row. Unsized tags (latest, q8_0, bf16) take the label of a sized tag with the
    same digest, else the model's only label; otherwise they stay unlabelled."""
    labels = {}
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["name"].split(":")[0]].append(r)
    for model_rows in by_model.values():
        digest_label = {r["digest"][:12]: _label_of(r["tag"]) for r in model_rows if _label_of(r["tag"])}
        known = set(digest_label.values())
        for r in model_rows:
            label = _label_of(r["tag"]) or digest_label.get(r["digest"][:12], "")
            if not label and len(known) == 1:
                label = next(iter(known))
            labels[id(r)] = label
    return labels


def _representative(label, members):
    """Ollama's default for the group: the bare label tag (gemma4:12b), else :latest, else a Q4_K_M tag."""
    by_tag = {r["tag"]: r for r in members}
    for tag in (label, "latest"):
        if tag and tag in by_tag:
            return by_tag[tag]
    q4 = sorted((r for r in members if r["quantization"] == "Q4_K_M"), key=lambda r: len(r["tag"]))
    return (q4 or sorted(members, key=lambda r: len(r["tag"])))[0]


DEFAULT_MARGIN_GB = 2
NEAR_LIMIT = 0.95  # warn when a model uses 95% or more of a host's usable limit


def _usable(spec):
    return spec["max_gb"] - spec.get("margin_gb", DEFAULT_MARGIN_GB)


def fit(disk_size_gb, hosts):
    """Return (fits_min_gb, fit_note, fit_warn, default run_on) against config/hosts.json.
    A model fits a host when disk_size_gb <= usable (max_gb - margin_gb); fits_min_gb is the smallest usable."""
    if disk_size_gb in ("", None):
        return "", "no disk size", "", ""
    size = float(disk_size_gb)
    fits = [name for name, spec in hosts.items() if size <= _usable(spec)]
    warns = []
    for name, spec in hosts.items():
        usable, margin = _usable(spec), spec.get("margin_gb", DEFAULT_MARGIN_GB)
        if usable < size <= usable + margin:
            warns.append(f"{size:g} GB misses {name} {usable:g} by {size - usable:.2g}")
        elif name in fits and size >= NEAR_LIMIT * usable:
            warns.append(f"{size:g} GB vs {name} {usable:g}, within 5%")
    fit_warn = "; ".join(warns)
    if fits:
        return min(_usable(hosts[n]) for n in fits), "", fit_warn, ";".join(fits)
    name, spec = max(hosts.items(), key=lambda kv: _usable(kv[1]))
    return "", f"{size:g} GB > {name} {_usable(spec):g}", fit_warn, ""


def collapse(rows, hosts):
    rows = [r for r in rows if not DROP_TAG.search(r["tag"])]
    labels = _assign_labels(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["name"].split(":")[0], labels[id(r)])].append(r)

    out = []
    for (model, label), members in groups.items():
        rep = _representative(label, members)
        pulled_hosts = {h for r in members for h in r.get("host", "").split(";") if h}
        fits_min_gb, fit_note, fit_warn, run_on = fit(rep.get("disk_size_gb"), hosts)
        out.append({
            **{k: rep.get(k, "") for k in BY_MODEL_COLUMNS},
            "name": f"{model}:{label}" if label else model,
            "representative_tag": rep["tag"],
            "run_on": run_on,
            "fits_min_gb": fits_min_gb,
            "fit_note": fit_note,
            "fit_warn": fit_warn,
            "pulled_on": ";".join(sorted(pulled_hosts)),
            # * marks tags pulled on at least one host.
            "all_tags": " ".join(sorted(r["tag"] + ("*" if r.get("pulled") else "") for r in members)),
            # Model-level fields: take them from any member that has them.
            **{k: next((r[k] for r in members if r.get(k) not in ("", None)), "")
               for k in ("pull_count", "first_published_approx")},
        })
    return sorted(out, key=lambda r: r["name"])


def write_by_model(rows, path, defaults_path):
    """Write the collapsed CSV. A value in an EDITABLE column that differs from the default we last
    wrote is a user edit and is carried over; unedited cells take the new default."""
    path, defaults_path = Path(path), Path(defaults_path)
    current, last_defaults = {}, {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            current = {r["name"]: r for r in csv.DictReader(f)}
    if defaults_path.exists():
        last_defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    new_defaults = {}
    for r in rows:
        new_defaults[r["name"]] = {c: str(r[c]) for c in EDITABLE}
        old, old_default = current.get(r["name"]), last_defaults.get(r["name"])
        for c in EDITABLE:
            if old is None or not old.get(c):
                continue
            # No record of what we wrote last time: trust the file as-is.
            if old_default is None or old[c] != old_default.get(c):
                r[c] = old[c]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BY_MODEL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    defaults_path.write_text(json.dumps(new_defaults, indent=1), encoding="utf-8")
