# General Info
First member of the homedoc scripts that query local computer or network feature, parses its content and produces a small (local) LLM enhanced report; homedoc-tailscale-status queries status of tailscale and writes a timestamped report.

I am experimenting to find potential use cases for local LLMs as semantic engines that convert the often too technical program outputs into more understandable reports/explanation/documentation for us half or two third literates in the world of technical jargon. Maybe this or similar scripts can be useful to somebody.

# Disclaimer
This and other scripts (as well as accompanying texts/files/documentation) are written by (mostly) GPT-5, so be aware of potential security issues or plain nonsense; never run code that you haven't inspected. I tried to minimize the potential damage by sticking to the very simple approach of single file scripts with as little dependencies as possible.

If you want to commit, feel free to fork, mess around and put "ai slop" on my "ai slop", or maybe deslop it enirely, but there is no garantuee that I will incorporate changes.

# HomeDoc — Tailscale Status Snapshot & Report

![version](https://img.shields.io/badge/version-0.1.2-blue.svg)
![license](https://img.shields.io/badge/license-GPLv3-blue.svg)

Single-file, stdlib-only utility that:
1. collects `tailscale status --json`,
2. normalizes it into a compact snapshot,
3. (optionally) uses a local LLM (Ollama-compatible HTTP) to produce a concise Markdown report,
4. writes artifacts per-run (`status.json`, `snapshot.json`, `insights.json`, `report.md`, `llm_raw.txt`, `homedoc.log`).

## Run or Install

```bash
# Run directly from the repo (no install)
python homedoc_tailscale_status.py \
  --out ./homedoc_out \
  --tz local \
  --model gemma3:12b \
  --server http://127.0.0.1:11434 \
  --stream

# Install locally
pipx install .
# or
pip install .
```

## Usage

```bash
homedoc-tailscale-status \
  --out ./homedoc_out \
  --tz local \
  --model gemma3:12b \
  --server http://127.0.0.1:11434 \
  --stream
```

Key flags:
- `--no-llm` — skip LLM; still generates snapshot and local Markdown table + findings.
- `--json-only` — write JSON artifacts only, skip Markdown entirely.
- `--flat` — write directly into `--out` without the timestamped subdirectory.
- `--input-json <file>` — offline mode: use a saved `tailscale status --json` output.
- `--llm-mode auto|generate|chat` and `--[no-]stream` — control HTTP path & streaming.

See [USAGE.md](USAGE.md) for a full rundown of every CLI option, defaults, and examples.

## Outputs
- `status.json` — raw `tailscale status` JSON
- `snapshot.json` — normalized compact snapshot
- `insights.json` — reserved for future structured findings
- `report.md` — Markdown report; includes local table+findings, plus an LLM section if enabled
- `llm_raw.txt` — raw LLM HTTP response (for debugging)
- `homedoc.log` — timestamped log (also printed to stdout)

## Security notes
- Designed for local LLMs over `http://localhost`. If pointing to remote endpoints, prefer `https://` and be mindful of credentials.

## Changelog

### 0.1.2 — 2025-10-01
- Hardened `--input-json` parsing so malformed snapshots are reported cleanly instead of crashing the run.
- Expanded docs: refreshed the README quickstart and added a dedicated `USAGE.md` reference guide.

### 0.1.1 — 2025-09-30
- Logger uses bounded buffer (deque) + line-buffered file writes to avoid unbounded memory growth.
- Consolidated error handling: helpers raise; `main()` maps to consistent exit codes.
- Safer JSON extraction via `json.JSONDecoder.raw_decode`.
- Streaming hardened with timeout watchdog and graceful non-stream fallback.
- Markdown table escapes `|`; “Findings” rendered even without LLM.
- New `--json-only`; clarified `--flat`; early log creation.

### 0.1.0 — 2025-09-30
- Initial release.
