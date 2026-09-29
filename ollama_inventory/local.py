"""Stage 1: inventory models from the local Ollama HTTP API."""
from datetime import datetime, timezone

import requests

OLLAMA_URL = "http://localhost:11434"
CAPABILITIES = ("thinking", "tools", "vision", "embedding", "completion")


def is_cloud(name, remote_host=None):
    """Cloud models are named '*-cloud' (optionally with a :tag) or carry a remote_host."""
    if remote_host:
        return True
    base, _, tag = (name or "").partition(":")
    return base.endswith("-cloud") or tag.endswith("-cloud") or tag == "cloud"


def fetch_raw(base_url=OLLAMA_URL, timeout=30):
    """Return [{'tag': <entry from /api/tags>, 'show': <response from /api/show or None>}]."""
    tags = requests.get(f"{base_url}/api/tags", timeout=timeout)
    tags.raise_for_status()
    raw = []
    for entry in tags.json().get("models", []):
        show, error = None, ""
        try:
            r = requests.post(f"{base_url}/api/show", json={"model": entry["name"]}, timeout=timeout)
            r.raise_for_status()
            show = r.json()
        except requests.RequestException as e:
            error = str(e)
        raw.append({"tag": entry, "show": show, "show_error": error})
    return raw


def _context_length(show, details):
    info = (show or {}).get("model_info") or {}
    arch = info.get("general.architecture")
    if arch and f"{arch}.context_length" in info:
        return info[f"{arch}.context_length"]
    return details.get("context_length", "")


def _utc(ts):
    """Ollama reports local time with an offset; store UTC so the CSV doesn't reveal the host's timezone."""
    try:
        return datetime.fromisoformat(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return ts or ""


def to_row(item, host=""):
    tag, show = item["tag"], item["show"] or {}
    details = {**(tag.get("details") or {}), **(show.get("details") or {})}
    caps = show.get("capabilities", tag.get("capabilities"))
    size = tag.get("size")
    row = {
        "name": tag.get("name", ""),
        "tag": tag.get("name", "").partition(":")[2],
        "source": "local",
        "host": host,
        "is_cloud": is_cloud(tag.get("name"), tag.get("remote_host") or show.get("remote_host")),
        "family": details.get("family", ""),
        "parameter_size": details.get("parameter_size", ""),
        "quantization": details.get("quantization_level", ""),
        "context_length": _context_length(show, details),
        # Decimal GB. Cloud models report a stub size, so leave it blank for them.
        "disk_size_gb": round(size / 1e9, 2) if isinstance(size, int) and size > 0 else "",
        "modified_at": _utc(tag.get("modified_at")),
        "digest": tag.get("digest", ""),
    }
    if row["is_cloud"]:
        row["disk_size_gb"] = ""
    for cap in CAPABILITIES:
        row[cap] = "" if caps is None else cap in caps
    return row
