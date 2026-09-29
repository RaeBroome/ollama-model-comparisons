"""CLI: python -m ollama_inventory [--host NAME] [--ollama-url URL] [--refresh] [--local-only] [--include-community]"""
import argparse
import csv
import json
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import catalog, collapse, local
from .fetch import Fetcher

OUT_DIR = Path("output")
CACHE_DIR = OUT_DIR / "cache"
CSV_PATH = OUT_DIR / "ollama_models.csv"
BY_MODEL_PATH = OUT_DIR / "ollama_models_by_model.csv"
HOSTS_PATH = Path("config") / "hosts.json"
# The last catalog crawl, so --local-only can rebuild the CSVs without crawling.
CATALOG_PATH = CACHE_DIR / "catalog_rows.json"

COLUMNS = [
    "name", "tag", "host", "pulled", "family", "parameter_size", "quantization", "context_length",
    "disk_size_gb", "size_source", "tools", "vision", "pull_count", "first_published_approx", "modified_at", "digest",
    "url",
]

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


def apply_local_lookup(catalog_rows, local_rows, problems):
    """Local inventories only annotate catalog rows (host, pulled, modified_at, exact size); they never add rows."""
    by_name = defaultdict(list)
    for r in local_rows:
        by_name[r["name"]].append(r)
    for row in catalog_rows:
        pulls = sorted(by_name.get(row["name"], []), key=lambda r: r["host"])
        row["pulled"] = bool(pulls)
        row["host"] = ";".join(r["host"] for r in pulls)
        row["modified_at"] = ";".join(r["modified_at"] for r in pulls)
        sizes = {r["disk_size_gb"] for r in pulls if r["disk_size_gb"] != ""}
        if sizes:
            # Exact bytes from the Ollama API beat the catalog's whole-GB rounding.
            row["disk_size_gb"], row["size_source"] = pulls[0]["disk_size_gb"], "local"
            if len(sizes) > 1:
                problems.append(f"{row['name']}: local sizes differ between hosts ({', '.join(f'{r['host']} {r['disk_size_gb']}' for r in pulls)}); using {pulls[0]['host']}")
        for r in pulls:
            if r["digest"] and not r["digest"].startswith(row["digest"]):
                problems.append(f"{r['host']} {r['name']}: local digest {r['digest'][:12]} != catalog {row['digest']} "
                                f"(your pull is out of date, or the tag was re-pushed)")
    catalog_names = {r["name"] for r in catalog_rows}
    missing = sorted({f"{r['name']} ({r['host']})" for r in local_rows if r["name"] not in catalog_names})
    if missing:
        problems.insert(0, f"{len(missing)} pulled models not in catalog, excluded: {', '.join(missing)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ollama_inventory")
    p.add_argument("--refresh", action="store_true", help="re-pull data instead of using output/cache")
    p.add_argument("--local-only", action="store_true",
                   help="re-fetch the local inventory and rebuild both CSVs from the last catalog crawl, without crawling")
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
    local_rows, problems = load_local_rows(hosts, host, args.ollama_url, args.refresh or args.local_only)

    if args.local_only:
        if not CATALOG_PATH.exists():
            p.error("--local-only needs a previous catalog crawl; run once without it first")
        saved = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        rows, cat_problems = saved["rows"], saved["problems"]
        print(f"catalog: reusing crawl from {saved['crawled_at']} (--local-only)")
    else:
        now = datetime.now(timezone.utc)
        fetcher = Fetcher(CACHE_DIR, refresh=args.refresh)
        rows, stats, cat_problems = catalog.crawl(fetcher, args.include_community, now)
        print("filter counts:", json.dumps(stats, indent=1))
        CATALOG_PATH.write_text(json.dumps({"crawled_at": now.isoformat(timespec="seconds"), "rows": rows,
                                            "problems": cat_problems}, indent=1), encoding="utf-8")
    problems += cat_problems
    apply_local_lookup(rows, local_rows, problems)
    rows.sort(key=lambda r: r["name"])
    write_csv(rows, COLUMNS)
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    by_model = collapse.collapse(rows, hosts)
    collapse.write_by_model(by_model, BY_MODEL_PATH, CACHE_DIR / "by_model_defaults.json")
    print(f"wrote {BY_MODEL_PATH} ({len(by_model)} rows)")
    for msg in problems:
        print("WARN", msg)


if __name__ == "__main__":
    main()
