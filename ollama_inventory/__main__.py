"""CLI: python -m ollama_inventory [--host NAME] [--ollama-url URL] [--refresh] [--local-only] [--include-community]"""
import argparse
import csv
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from . import catalog, collapse, local
from .fetch import Fetcher

OUT_DIR = Path("output")
CACHE_DIR = OUT_DIR / "cache"
CSV_PATH = OUT_DIR / "ollama_models.csv"
BY_MODEL_PATH = OUT_DIR / "ollama_models_by_model.csv"
HOSTS_PATH = Path("config") / "hosts.json"

COLUMNS = [
    "name", "tag", "source", "host", "pulled", "family", "parameter_size", "quantization", "context_length",
    "disk_size_gb", "thinking", "tools", "vision", "pull_count", "first_published_approx", "date_source_text",
    "date_flag", "recent_6mo", "modified_at", "digest", "url",
]
# Catalog fields copied onto a pulled local row; local API values win for everything else.
CATALOG_ONLY = ["pull_count", "first_published_approx", "date_source_text", "date_flag", "url"]


def cached(name, fetch, refresh):
    """Load raw data from output/cache/<name>.json unless --refresh or missing."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def default_host(hosts):
    """The hosts.json key this machine's hostname starts with (e.g. LAPTOP-AB12CD -> laptop)."""
    me = socket.gethostname().lower()
    matches = [h for h in hosts if me.startswith(h.lower())]
    return matches[0] if len(matches) == 1 else None


def load_local_rows(hosts, host, url, refresh):
    """Inventory `host` via `url`; other hosts come from their cache untouched, so runs never clobber each other."""
    rows, problems = [], []
    for h in hosts:
        path = CACHE_DIR / f"local_{h}.json"
        if h == host:
            raw = cached(f"local_{h}", lambda: local.fetch_raw(url), refresh)
        elif path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            continue
        problems += [f"{h}: /api/show failed for {i['tag'].get('name')}: {i['show_error']}" for i in raw if i.get("show_error")]
        rows += [local.to_row(i, h) for i in raw]
    return rows, problems


def write_csv(rows, columns, path=CSV_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def add_local_recency(local_rows, fetcher, now, problems):
    """recent_6mo for pulled models, using the same oldest-tag rule as the catalog."""
    for row in local_rows:
        if row["is_cloud"]:
            row["recent_6mo"] = ""
            continue
        path = catalog.model_path(row["name"])
        try:
            info = catalog.fetch_model(fetcher, path, now)
        except Exception as e:  # noqa: BLE001 - report and leave blank
            problems.append(f"local {row['name']}: could not fetch {path}: {e}")
            row["recent_6mo"] = ""
            continue
        r = catalog.recency(info["pub"])
        row["recent_6mo"] = "" if r is None else r
        row["pull_count"] = info["model"]["pull_count"]
        row["url"] = f"{catalog.BASE}/{path}"
        if info["pub"]:
            row["first_published_approx"] = info["pub"]["first_published_approx"]
            row["date_source_text"] = info["pub"]["date_source_text"]
        if r is None:
            problems.append(f"local {row['name']}: recency unknown/ambiguous "
                            f"({info['pub']['date_source_text'] if info['pub'] else 'no date'})")
        catalog_tag = next((t for t in info["tags"] if t["name"].removeprefix("library/") == row["name"]), None)
        if catalog_tag and row.get("digest") and not catalog_tag["digest"].startswith(row["digest"][:12]):
            problems.append(f"local {row['name']}: local digest {row['digest'][:12]} != catalog {catalog_tag['digest']} "
                            f"(your pull is out of date, or the tag was re-pushed)")


def merge(local_rows, catalog_rows):
    """One row per host + name:tag. Pulled catalog tags collapse into the local row(s), keeping catalog-only fields."""
    by_name = {r["name"]: r for r in catalog_rows}
    for row in local_rows:
        row["pulled"] = True
        match = by_name.get(row["name"])
        if match:
            for k in CATALOG_ONLY:
                row[k] = match[k]
    for row in local_rows:
        by_name.pop(row["name"], None)
    for row in by_name.values():
        row["pulled"] = False
    return local_rows + list(by_name.values())


def main(argv=None):
    p = argparse.ArgumentParser(prog="ollama_inventory")
    p.add_argument("--refresh", action="store_true", help="re-pull data instead of using output/cache")
    p.add_argument("--local-only", action="store_true", help="skip catalog and benchmark stages")
    p.add_argument("--include-community", action="store_true", help="include namespaced (user/model) catalog models")
    p.add_argument("--ollama-url", default=local.OLLAMA_URL, help="Ollama API to inventory (default %(default)s)")
    p.add_argument("--host", help="name from config/hosts.json for that Ollama (default: matched from this hostname)")
    args = p.parse_args(argv)

    hosts = json.loads(HOSTS_PATH.read_text(encoding="utf-8"))
    host = args.host or default_host(hosts)
    if host not in hosts:
        p.error(f"--host must be one of {sorted(hosts)} (could not infer it from hostname {socket.gethostname()!r})")
    how = "from --host" if args.host else f"from hostname {socket.gethostname()}"
    print(f"host: {host} ({how}), ollama: {args.ollama_url}")
    local_rows, problems = load_local_rows(hosts, host, args.ollama_url, args.refresh)
    catalog_rows = []

    if not args.local_only:
        now = datetime.now(timezone.utc)
        fetcher = Fetcher(CACHE_DIR, refresh=args.refresh)
        catalog_rows, stats, cat_problems = catalog.crawl(fetcher, args.include_community, now)
        problems += cat_problems
        add_local_recency(local_rows, fetcher, now, problems)
        print("filter counts:", json.dumps(stats, indent=1))

    rows = sorted(merge(local_rows, catalog_rows), key=lambda r: (r["source"], r["name"], r.get("host", "")))
    write_csv(rows, COLUMNS)
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    by_model = collapse.collapse(rows, hosts)
    collapse.write_by_model(by_model, BY_MODEL_PATH, CACHE_DIR / "by_model_defaults.json")
    print(f"wrote {BY_MODEL_PATH} ({len(by_model)} rows)")
    for msg in problems:
        print("WARN", msg)


if __name__ == "__main__":
    main()
