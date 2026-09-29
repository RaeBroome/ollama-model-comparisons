"""Polite, cached HTTP GETs for ollama.com pages (max 1 request/second)."""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ollama-inventory/0.1 (+personal model comparison)"}
MIN_INTERVAL = 1.0


class Fetcher:
    def __init__(self, cache_dir, refresh=False):
        self.dir = Path(cache_dir) / "pages"
        self.refresh = refresh
        self._last = 0.0
        self._seen = set()  # with --refresh, re-fetch each URL once per run

    def _path(self, key):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:80]
        return self.dir / f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}.json"

    def get(self, url, htmx=False):
        """Return (html, fetched_at: aware UTC datetime). Relative dates must use fetched_at, not now."""
        key = f"{url}|htmx" if htmx else url
        path = self._path(key)
        if path.exists() and (not self.refresh or key in self._seen):
            d = json.loads(path.read_text(encoding="utf-8"))
            return d["html"], datetime.fromisoformat(d["fetched_at"])
        wait = MIN_INTERVAL - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        headers = {**HEADERS, **({"HX-Request": "true"} if htmx else {})}
        r = requests.get(url, headers=headers, timeout=30)
        self._last = time.monotonic()
        r.raise_for_status()
        fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "fetched_at": fetched_at.isoformat(), "html": r.text}), encoding="utf-8")
        self._seen.add(key)
        return r.text, fetched_at
