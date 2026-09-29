"""Stage 2: recent local thinking models from the ollama.com catalog.

ollama.com shows tag ages only as go-humanize text ("5 months ago"). Each text maps to an
age range in days (months are 30 days, years 360), checked against the absolute "Updated"
tooltips on the search page. A model's first_published range comes from its oldest tag.
"""
import html as htmllib
import re
from datetime import datetime, timedelta, timezone

from .local import is_cloud

BASE = "https://www.ollama.com"
SEARCH = BASE + "/search?o=newest&c=thinking&page={page}"
# ollama.com returns namespaced (community) models only when q is non-empty. A single space
# matches every model and returns community models only, so it is not a text filter.
SEARCH_COMMUNITY = BASE + "/search?q=%20&o=newest&c=thinking&page={page}"
CUTOFF_DAYS = 183
MAX_PAGES = 500  # safety cap; the crawl normally stops at the cutoff
_TOOLTIP = r'title="([A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M UTC)"'

_UNIT_DAYS = {"second": 0, "minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 360}


def age_bounds(text):
    """'5 months ago' -> (150, 180): age in days, lower inclusive, upper exclusive. None if unparsable."""
    t = text.strip().lower()
    if t in ("now", "just now"):
        return (0, 1)
    m = re.fullmatch(r"(\d+|an?) (second|minute|hour|day|week|month|year)s? ago", t)
    if not m:
        return None
    n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
    unit = _UNIT_DAYS[m.group(2)]
    if unit == 0:
        return (0, 1)
    if m.group(2) == "day":
        return (n, n + 1)
    if m.group(2) == "week":
        return (7 * n, min(7 * n + 7, 30))  # "4 weeks ago" rolls over to "1 month" at 30 days
    return (unit * n, unit * (n + 1))


def _text(s):
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_search(page_html):
    """Return (cards, has_more). Each card: {path, badges, pull_count, updated_title}."""
    cards = []
    for path, body in re.findall(r'<a href="/([^"]+)" class="group w-full">(.*?)</a>\s*</li>', page_html, re.S):
        badges = re.findall(r'<span\s+class="inline-flex[^"]*">([^<]+)</span>', body)
        pulls = re.search(r"<span >([^<]+)</span>\s*<span[^>]*>&nbsp;Pulls", body)
        updated = re.search(_TOOLTIP, body)
        cards.append({"path": path, "badges": [b.strip() for b in badges], "pull_count": pulls.group(1) if pulls else "",
                      "updated_title": updated.group(1) if updated else ""})
    return cards, 'hx-get="/search?page=' in page_html


def parse_model_page(page_html):
    badges = re.findall(r'<span[^>]*class="inline-flex[^"]*"[^>]*>([^<]+)</span>', page_html)
    pulls = re.search(r"<span >([^<]+)</span>\s*<span[^>]*>&nbsp;Downloads", page_html)
    updated = re.search(_TOOLTIP, page_html)
    return {"badges": [b.strip() for b in badges], "pull_count": pulls.group(1) if pulls else "",
            "updated_title": updated.group(1) if updated else ""}


def parse_tags_page(page_html):
    """One dict per tag row, from the row's summary line:
    'digest • 18GB (or "High Usage" for cloud) • 256K context window • Text, Image input • 3 days ago'."""
    tags = []
    for block in page_html.split('<div class="group px-4 py-3">')[1:]:
        name = re.search(r'<a href="/([^"]+:[^"]+)" class="group-hover:underline">', block)
        summary = re.search(r'<span class="font-mono">(.*?)<div class="flex sm:hidden">', block, re.S)
        if not name or not summary:
            continue
        parts = [p.strip() for p in re.split(r"\s[^\w\s,]\s", _text(summary.group(1))) if p.strip()]
        row = {"name": name.group(1), "digest": parts[0], "size": "", "usage": "", "context": "", "input": "", "updated_text": ""}
        for part in parts[1:]:
            if re.fullmatch(r"[\d.]+\s*[KMGT]?B", part):
                row["size"] = part
            elif part.endswith("Usage"):
                row["usage"] = part
            elif part.endswith("context window"):
                row["context"] = part.removesuffix("context window").strip()
            elif part.endswith(" input"):
                row["input"] = part.removesuffix(" input")
            elif part.endswith("ago") or part in ("now", "just now"):
                row["updated_text"] = part
        tags.append(row)
    return tags


def parse_tag_page(page_html):
    """Arch/parameters/quantization of the main 'model' blob (ignores the projector)."""
    m = re.search(r'/blobs/[0-9a-f]+">\s*model\s*</a>(.*?)</div>\s*</div>\s*</div>', page_html, re.S)
    out = {}
    if m:
        for key in ("arch", "parameters", "quantization"):
            v = re.search(rf">{key}</span><span[^>]*>([^<]+)</span>", m.group(1))
            out[key] = v.group(1).strip() if v else ""
    return out


def _context_tokens(text):
    """'256K' -> 262144 (Ollama's K is 1024: local qwen3.8 reports 262144 for '256K')."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KM]?)", text.strip())
    if not m:
        return ""
    return int(float(m.group(1)) * {"": 1, "K": 1024, "M": 1024 ** 2}[m.group(2)])


def _size_gb(text):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(GB|MB|TB|KB)", text.strip())
    if not m:
        return ""
    return round(float(m.group(1)) * {"KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3}[m.group(2)], 3)


def first_published(tags, fetched_at, now, updated_title=""):
    """Publication range from the oldest tag's relative age, capped by the model's absolute
    'Updated' tooltip (a model cannot be first published after its last update).
    Returns {first_published_approx, date_source_text, age_lo, age_hi} or None."""
    dated = [(age_bounds(t["updated_text"]), t) for t in tags]
    dated = [(b, t) for b, t in dated if b]
    if not dated:
        return None
    (lo, hi), oldest = max(dated, key=lambda x: x[0])
    earliest = fetched_at - timedelta(days=hi)
    latest = fetched_at - timedelta(days=lo)
    source = f"oldest tag {oldest['name']}: '{oldest['updated_text']}' (tags page fetched {fetched_at:%Y-%m-%d %H:%M} UTC)"
    if updated_title:
        updated = parse_tooltip(updated_title)
        source += f"; model Updated tooltip '{updated_title}'"
        latest = min(latest, updated)
    return {
        "first_published_approx": f"{earliest.date()}/{latest.date()}",
        "date_source_text": source,
        "age_lo": (now - latest).total_seconds() / 86400,
        "age_hi": (now - earliest).total_seconds() / 86400,
    }


def parse_tooltip(text):
    return datetime.strptime(text, "%b %d, %Y %I:%M %p UTC").replace(tzinfo=timezone.utc)


def recency(pub, cutoff_days=CUTOFF_DAYS):
    """True = published within cutoff, False = older, None = unknown or straddles the cutoff."""
    if pub is None:
        return None
    if pub["age_hi"] <= cutoff_days:
        return True
    if pub["age_lo"] > cutoff_days:
        return False
    return None


def model_path(name):
    """'qwen3.8:latest' -> 'library/qwen3.8'; 'user/m:tag' -> 'user/m'."""
    base = name.split(":", 1)[0]
    return base if "/" in base else f"library/{base}"


def fetch_model(fetcher, path, now):
    """Fetch model + tags pages. Returns dict with tags, model info and publication info."""
    model_html, _ = fetcher.get(f"{BASE}/{path}")
    tags_html, fetched_at = fetcher.get(f"{BASE}/{path}/tags")
    tags = parse_tags_page(tags_html)
    model = parse_model_page(model_html)
    return {"path": path, "model": model, "tags": tags,
            "pub": first_published(tags, fetched_at, now, model["updated_title"])}


def _crawl_search(fetcher, url, now, stats, key, problems, skip_old_by_tooltip=False):
    """Walk one search listing page by page; stop when a whole page is older than the cutoff."""
    models = []
    stats[f"{key}_models_skipped_old"] = 0
    for page in range(1, MAX_PAGES + 1):
        cards, has_more = parse_search(fetcher.get(url.format(page=page), htmx=True)[0])
        page_recency = []
        for card in cards:
            if skip_old_by_tooltip and card["updated_title"]:
                age = (now - parse_tooltip(card["updated_title"])).total_seconds() / 86400
                if age > CUTOFF_DAYS:
                    # First publish can't be later than the last update: definitely too old, skip 2 fetches.
                    stats[f"{key}_models_skipped_old"] += 1
                    page_recency.append(False)
                    continue
            m = fetch_model(fetcher, card["path"], now)
            m["card"] = card
            models.append(m)
            page_recency.append(recency(m["pub"]))
        stats[f"{key}_search_pages"] = page
        if not has_more or (page_recency and all(r is False for r in page_recency)):
            break
    stats[f"{key}_models_listed"] = len(models) + stats[f"{key}_models_skipped_old"]

    # The newest sort follows creation order. A model counted as recent but listed after one
    # that is definitely older likely had all its tags re-pushed, so its oldest tag understates its age.
    older_seen = None
    for m in models:
        r = recency(m["pub"])
        m["date_flag"] = ""
        if r is False and older_seen is None:
            older_seen = m["path"]
        elif r is True and older_seen:
            m["date_flag"] = f"listed after {older_seen} (older than cutoff) in newest sort; oldest tag may be a re-push"
            problems.append(f"{m['path']}: kept, but {m['date_flag']}")
    return models


def crawl(fetcher, include_community=False, now=None):
    """Return (rows, stats, problems). Tag counts exclude community models skipped by their tooltip."""
    now = now or datetime.now(timezone.utc)
    stats = {}
    problems = []
    models = _crawl_search(fetcher, SEARCH, now, stats, "library", problems)
    if include_community:
        models += _crawl_search(fetcher, SEARCH_COMMUNITY, now, stats, "community", problems, skip_old_by_tooltip=True)

    rows = [(m, t) for m in models for t in m["tags"]]
    stats["tags_all"] = len(rows)
    for m in models:
        if not m["tags"]:
            problems.append(f"{m['path']}: no tags parsed from tags page")

    rows = [(m, t) for m, t in rows if not is_cloud(t["name"].split("/")[-1])]
    stats["tags_after_cloud"] = len(rows)
    stats["models_after_cloud"] = len({m["path"] for m, _ in rows})

    for m in models:
        r = recency(m["pub"])
        if m["pub"] is None:
            problems.append(f"{m['path']}: no parsable tag date -> excluded")
        elif r is None:
            problems.append(f"{m['path']}: ambiguous vs {CUTOFF_DAYS}-day cutoff, {m['pub']['date_source_text']}, "
                            f"range {m['pub']['first_published_approx']} -> excluded")
    rows = [(m, t) for m, t in rows if recency(m["pub"]) is True]
    stats["tags_after_date"] = len(rows)
    stats["models_after_date"] = len({m["path"] for m, _ in rows})

    if not include_community:
        rows = [(m, t) for m, t in rows if m["path"].startswith("library/")]
    stats["tags_after_community"] = len(rows)
    stats["models_after_community"] = len({m["path"] for m, _ in rows})

    # Tags sharing a digest are aliases of the same blob: fetch one detail page per digest.
    details = {}
    for m, t in rows:
        key = (m["path"], t["digest"])
        if key not in details:
            details[key] = parse_tag_page(fetcher.get(f"{BASE}/{t['name']}")[0])
    blank = sorted(t["name"] for m, t in rows if not details[(m["path"], t["digest"])])
    if blank:
        problems.append(f"{len(blank)} tags have no arch/parameters/quantization on their tag page "
                        f"(MLX/mxfp8/nvfp4 safetensors layouts), left blank; e.g. {', '.join(blank[:3])}")
    return [_to_row(m, t, details[(m["path"], t["digest"])]) for m, t in rows], stats, problems


def _to_row(m, t, detail):
    badges = set(m["model"]["badges"] or m["card"]["badges"])
    full = t["name"].removeprefix("library/")
    row = {
        "name": full,
        "tag": full.split(":", 1)[1],
        "source": "catalog",
        "is_cloud": False,
        "is_community": not m["path"].startswith("library/"),
        "family": detail.get("arch", ""),
        "parameter_size": detail.get("parameters", ""),
        "quantization": detail.get("quantization", ""),
        "context_length": _context_tokens(t["context"]),
        "disk_size_gb": _size_gb(t["size"]),
        "size_source": "catalog",
        "thinking": "thinking" in badges,
        "tools": "tools" in badges,
        # Per-tag input column is more specific than the model badge (e.g. text-only variants).
        "vision": ("Image" in t["input"]) if t["input"] else "vision" in badges,
        "embedding": "embedding" in badges,
        "pull_count": m["model"]["pull_count"] or m["card"]["pull_count"],
        "first_published_approx": m["pub"]["first_published_approx"],
        "date_source_text": m["pub"]["date_source_text"],
        "recent_6mo": True,
        "date_flag": m["date_flag"],
        "digest": t["digest"],
        "url": f"{BASE}/{t['name']}",
    }
    return row
