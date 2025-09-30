# General Info
First member of the homedoc scripts that query local computer or network feature, parses its content and produces a small (local) LLM enhanced report; homedoc-tailscale-status queries status of tailscale and writes a timestamped report.

I am experimenting to find potential use cases for local LLMs as semantic engines that convert the often too technical program outputs into more understandable reports/explanation/documentation for us half or two third literates in the world of technical jargon. Maybe this or similar scripts can be useful to somebody.

# Disclaimer
This and other scripts (as well as accompanying texts/files/documentation) are written by (mostly) GPT-5, so be aware of potential security issues or plain nonsense; never run code that you haven't inspected. I tried to minimize the potential damage by sticking to the very simple approach of single file scripts with as little dependencies as possible.

If you want to commit, feel free to fork, mess around and put "ai slop" on my "ai slop", or maybe deslop it enirely, but there is no garantuee that I will incorporate changes.

# homedoc-tailscale-status

Single-file tool that:
- collects `tailscale status --json` (no sudo),
- normalizes it into a compact snapshot,
- optionally calls a local LLM (Ollama) for a short markdown summary,
- writes timestamped artifacts to an output folder (**local time** by default).

**License:** GPL-3.0-or-later — see [LICENSE](LICENSE).  
**Version:** `0.1.0` (script constant `__version__`)

---

## Quick start (no install)

```bash
python3 homedoc_tailscale_status.py --help

python3 homedoc_tailscale_status.py   --ollama http://localhost:11434   --model "gemma3:12b"   --llm-mode markdown   --timeout 900   --num-predict 256   --stream   --stream-chunk-log 0   --debug
```

Artifacts are written to a per-run folder under `outputs/`, e.g.:

```
outputs/tailnet-report_2025-09-30_14-07-12/
  tailnet-report_2025-09-30_14-07-12_status.raw.json
  tailnet-report_2025-09-30_14-07-12_snapshot.json
  tailnet-report_2025-09-30_14-07-12_insights.json
  tailnet-report_2025-09-30_14-07-12_report.md
  tailnet-report_2025-09-30_14-07-12_llm.raw.txt
  tailnet-report_2025-09-30_14-07-12.log
```

---

## Install as a CLI (pipx or pip)

> This repo includes a minimal `pyproject.toml` so you can install and run the tool like a proper command.

### Using pipx (recommended)

Isolates the tool from your system Python and keeps a clean, single-purpose environment.

```bash
# once per machine
python3 -m pip install --user pipx
pipx ensurepath

# from the repo root
pipx install .

# run from anywhere
homedoc-tailscale-status --help
homedoc-tailscale-status --ollama http://localhost:11434 --model "gemma3:12b"
```

**Upgrade to the latest commit** (reinstall from the repo directory):

```bash
# from repo root after pulling updates
pipx reinstall .
```

**Uninstall**:

```bash
pipx uninstall homedoc-tailscale-status
```

### Using pip (editable dev install)

Good for local development with quick iteration.

```bash
# from repo root
python3 -m pip install --upgrade pip
python3 -m pip install -e .

# now the CLI is available
homedoc-tailscale-status --help
```

To remove the editable install:

```bash
python3 -m pip uninstall homedoc-tailscale-status
```

---

## CLI options (highlights)

- `--out outputs` root output dir  
- `--prefix tailnet-report` filename prefix  
- `--flat` write files directly into `--out` (no run subdir)  
- `--no-llm` skip model, write basic report only  
- `--ollama http://host:11434` ollama base URL  
- `--model "gemma3:12b"` model tag (resolver tries `<family>:latest` if exact tag missing)  
- `--api auto|generate|chat` endpoint selection (auto tries `/api/generate` then `/api/chat`)  
- `--llm-mode markdown|full`  
  - `markdown` (default): model returns only markdown; findings computed locally.  
  - `full`: model must return strict JSON `{markdown, findings{offline_devices[], tag_summary{}}}`.  
- `--timeout 900` HTTP timeout seconds  
- `--num-predict 256` cap output tokens (CPU friendly)  
- `--num-ctx 2048` request context window  
- `--tz local|utc` timestamps in local time (default) or UTC  
- `--stream` enable streaming (default)  
- `--stream-chunk-log 2000` streaming progress interval chars (0 disables)  
- `--debug` log payloads + first 600 chars of model output  

---

## Requirements / notes

- `tailscale` must be on PATH (the tool runs `tailscale status --json`).  
- Ollama is optional (`--no-llm` skips it). With Ollama:  
  - Make sure the model is pulled, e.g. `ollama pull gemma3:12b`.  
  - If an exact tag isn’t found, the script tries `<family>:latest`.
- Outputs are timestamped with **local time** by default; switch to UTC with `--tz utc`.
- All logs and raw model output are saved in each run folder for debugging.

---

## Versioning

- Script defines `__version__`.
- Tag releases in Git: `git tag v0.1.0 && git push --tags`.

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).  
SPDX-License-Identifier: GPL-3.0-or-later



