# ollama-model-comparisons

`ollama_inventory` builds a CSV inventory of Ollama models so you can decide what to pull and where to run it. It combines:

- the models already pulled on each of your machines, read from the local Ollama HTTP API;
- recent **thinking** models from the ollama.com catalog that you haven't pulled yet;
- a per-host "does it fit" check against the memory limits in `config/hosts.json`.

Dependencies are `requests` and the Python standard library (Python 3.11+).

## Setup

```sh
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

Run everything from the repo root. Outputs are written to `./output/`.

## Regenerate the CSVs

```sh
python -m ollama_inventory               # normal run
python -m ollama_inventory --local-only  # after pulling or removing models (seconds)
python -m ollama_inventory --refresh     # force fresh data (about 5 minutes)
```

The tool saves what it downloads in `output/cache/` (your Ollama model list and the ollama.com pages) and reuses it on later runs:

- **First run, or a fresh clone:** there's no cache yet, so a normal run downloads everything (about 5 minutes).
- **Later runs:** a normal run reuses the cache and finishes in seconds. It won't notice models you've pulled or removed since, or new catalog releases.
- **`--local-only`:** re-reads this machine's Ollama model list and rebuilds both CSVs from the last catalog crawl, without contacting ollama.com. Use it after pulling or removing models. It needs one earlier normal run.
- **`--refresh`:** ignores the cache and downloads everything again (about 5 minutes). Use it to check for new catalog releases.

Both CSVs are rewritten on every run; your `run_on` edits are kept.
For your other machine, add `--host framework --ollama-url http://<framework-address>:11434`.

## Stages

| Stage | What it does | Status |
|---|---|---|
| 1. Local inventory | Calls `/api/tags`, then `/api/show` for each model, on one host. It adds no rows; it marks which catalog tags are pulled on which host, when they were pulled (UTC), and their exact size. Pulled models that aren't in the catalog are listed in a warning and left out. | done |
| 2. Catalog | Crawls `ollama.com/search?o=newest&c=thinking` and each model's page and tags page. Drops cloud tags and models first published more than 183 days ago. **This is the only source of rows**, so every row is a local thinking model published in the last 6 months. | done |
| 3. Benchmarks | Benchmark columns filled only from fetched sources, plus your RCAEval results. | not built yet |
| 4. Output | Final column order and a summary. | not built yet |

ollama.com shows tag ages only as relative text ("5 months ago"). Each model's publish date is therefore a **range** (`first_published_approx`), derived from its oldest tag and capped by the model's absolute "Updated" timestamp. Models whose range straddles the cutoff are excluded and reported. Models that are probably older than their tags suggest are kept, but each run prints a warning naming them. The sign is that the catalog's newest-first listing puts them after a model already known to be older than the cutoff, which suggests their tags were re-pushed.

## Flags

| Flag | Effect |
|---|---|
| *(none)* | Inventory this machine's Ollama (from cache if present), crawl the catalog (from cache if present), and write both CSVs. |
| `--host NAME` | Which `config/hosts.json` entry this run inventories. Defaults to the entry whose name starts the machine's hostname; if none matches, you must pass it. The chosen host is printed at the start of every run. |
| `--ollama-url URL` | Ollama API to inventory (default `http://localhost:11434`). Combine with `--host` to inventory another machine, e.g. `--host framework --ollama-url http://framework.local:11434`. |
| `--refresh` | Re-fetch everything instead of using `output/cache/`. Without it, cached API responses and pages are reused. |
| `--local-only` | Re-fetch this host's local inventory and rebuild both CSVs from the last catalog crawl (`output/cache/catalog_rows.json`), skipping the crawl. Fails if there's no earlier crawl. |
| `--include-community` | Also crawl namespaced `user/model` catalog entries (off by default). ollama.com only lists these when the search box is non-empty, so this crawls `q=%20` (a single space). That's roughly 60 pages for 6 months, so it's slow at 1 request per second. |

Each host's Stage 1 inventory is cached separately (`output/cache/local_<host>.json`). A run against one host never overwrites another host's inventory, and the pulled markers always combine every host inventoried so far.

The catalog crawl makes at most 1 request per second and caches every page. A full first run takes about 5 minutes.

## Outputs

**`output/ollama_models.csv`: one row per catalog tag** (e.g. `gemma4:12b-it-q8_0`). MLX/nvfp4/mxfp8 tags are included here, but their parameter size and quantization are blank because their pages don't list them. For tags you've pulled:

- `pulled` is `True`.
- `host` lists the hosts that have the tag, joined with `;` (e.g. `shadow;framework`). It's blank if the tag isn't pulled anywhere.
- `modified_at` lists when each of those hosts pulled it (UTC), joined with `;` in the same order as `host`.
- `disk_size_gb` is the exact size reported by your Ollama, and `size_source` is `local`. For tags you haven't pulled, the size is the catalog's figure, rounded to whole GB, and `size_source` is `catalog`.

**`output/ollama_models_by_model.csv`: one row per model + size** (e.g. `gemma4:12b`, `qwen3.6:27b-coding`). MLX/nvfp4/mxfp8 tags are dropped. Each row shows one representative tag: Ollama's default for that size, i.e. the bare size tag, else `latest`. The other tags are listed in `all_tags`, with `*` marking pulled ones. `pulled_on` lists the hosts that have any tag in the group. `disk_size_gb` and `size_source` come from the representative tag, so the size is exact only when the representative tag itself is pulled. Fit columns (computed from that size):

- `fits_min_gb`: the smallest usable limit (`max_gb - margin_gb`) among hosts the model fits. Blank if it fits none, with the reason in `fit_note`.
- `fit_warn`: set when the model misses a host by less than that host's margin, or when it fits a host but is within 5% of that host's usable limit.

### Columns you edit by hand

In `ollama_models_by_model.csv`, **`run_on`** (semicolon-separated host names) is yours to edit; clear it to exclude a model. Re-runs keep your edits: the tool remembers the defaults it last wrote (`output/cache/by_model_defaults.json`) and only overwrites cells you haven't changed. That's why the CSVs are committed and `output/cache/` is not.

## Host limits

`config/hosts.json`:

```json
{"shadow": {"max_gb": 20, "margin_gb": 1}, "framework": {"max_gb": 98, "margin_gb": 2}}
```

`max_gb` is the host's practical capacity for model weights, and `margin_gb` (default 2) is the headroom subtracted from it. A model fits when `disk_size_gb <= max_gb - margin_gb`. The margin covers two things:

- `disk_size_gb` is weights only. The KV cache grows with context, so a model near the limit may still fail at long context.
- The catalog rounds sizes to whole GB.

`context_length` is in tokens. Ollama displays context in units of 1024, so its numbers read lower than ours (202752 tokens shows as "198K").

## Known limitations

- The catalog is scraped from ollama.com's HTML. A site redesign will break the parsers; the tool warns rather than guessing.
- `first_published_approx` is a range, not a date. It relies on tag ages, and re-pushed tags make a model look newer than it is.
- Catalog sizes, pull counts and context lengths are the values as displayed (rounded).

## Tests

```sh
python -m pytest ollama_inventory/tests
```
