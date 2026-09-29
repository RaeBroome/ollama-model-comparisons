import csv

from ollama_inventory import __main__ as cli
from ollama_inventory.catalog import age_bounds
from ollama_inventory.collapse import BY_MODEL_COLUMNS, collapse, fit, write_by_model
from ollama_inventory.local import is_cloud, to_row

HOSTS = {"shadow": {"max_gb": 20, "margin_gb": 1}, "framework": {"max_gb": 98, "margin_gb": 2}}


def test_cloud_flag_by_name():
    assert is_cloud("gpt-oss:120b-cloud")
    assert is_cloud("gemma4:cloud")
    assert is_cloud("kimi-cloud")
    assert not is_cloud("qwen3.8:latest")
    assert not is_cloud("cloudy-model:7b")  # "cloud" inside a name is not a cloud tag


def test_cloud_flag_by_remote_host():
    assert is_cloud("qwen3.8:latest", remote_host="https://ollama.com:443")


def test_cloud_row_has_blank_disk_size():
    row = to_row({"tag": {"name": "x:cloud", "size": 384}, "show": None})
    assert row["is_cloud"] is True and row["disk_size_gb"] == ""


def test_per_tag_csv_column_order(tmp_path):
    path = tmp_path / "t.csv"
    cli.write_csv([{"name": "a:1b", "source": "local"}], cli.COLUMNS, path)
    with open(path, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == cli.COLUMNS
    assert cli.COLUMNS[:3] == ["name", "tag", "host"]


def test_by_model_csv_column_order(tmp_path):
    rows = collapse([_row("m:8b", "8b", "local", 5.0)], HOSTS)
    write_by_model(rows, tmp_path / "b.csv", tmp_path / "d.json")
    with open(tmp_path / "b.csv", newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == BY_MODEL_COLUMNS


def test_age_bounds_match_go_humanize_buckets():
    assert age_bounds("3 days ago") == (3, 4)
    assert age_bounds("4 weeks ago") == (28, 30)
    assert age_bounds("5 months ago") == (150, 180)
    assert age_bounds("1 year ago") == (360, 720)
    assert age_bounds("Updated recently") is None


def test_fit_uses_usable_limit_and_warns():
    assert fit(18.6, HOSTS)[:3] == (19, "", "18.6 GB vs shadow 19, within 5%")  # 97.9% of usable
    assert fit(18.0, HOSTS)[2] == ""                                          # 94.7%: no warning
    assert fit(20.0, HOSTS)[2] == "20 GB misses shadow 19 by 1"
    assert fit(120, HOSTS)[:2] == ("", "120 GB > framework 96")
    assert fit("", HOSTS)[1] == "no disk size"


def test_collapse_splits_coding_and_drops_mlx():
    rows = [_row("q:27b", "27b"), _row("q:27b-coding", "27b-coding"), _row("q:27b-mlx", "27b-mlx")]
    names = sorted(r["name"] for r in collapse(rows, HOSTS))
    assert names == ["q:27b", "q:27b-coding"]


def test_hand_edits_survive_rerun(tmp_path):
    out, defaults = tmp_path / "b.csv", tmp_path / "d.json"
    write_by_model(collapse([_row("m:8b", "8b")], HOSTS), out, defaults)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["run_on"] = "framework"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BY_MODEL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    write_by_model(collapse([_row("m:8b", "8b")], HOSTS), out, defaults)
    with open(out, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["run_on"] == "framework"


def _row(name, tag, source="catalog", size=5.0):
    return {"name": name, "tag": tag, "source": source, "host": "shadow" if source == "local" else "",
            "digest": name, "quantization": "Q4_K_M", "disk_size_gb": size, "context_length": 4096}
