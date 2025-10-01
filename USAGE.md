# Usage Guide

This document captures complete command-line usage for `homedoc_tailscale_status.py`, including how to run it straight from a checkout and what every flag does.

## Run without installing

From the repository root you can execute the script directly. All artifacts (JSON, Markdown, logs) will be written into `./homedoc_out/<timestamp>` by default:

```bash
python homedoc_tailscale_status.py \
  --out ./homedoc_out \
  --tz local \
  --model gemma3:12b \
  --server http://127.0.0.1:11434 \
  --stream
```

Prefer `--help` if you want to inspect the CLI interactively:

```bash
python homedoc_tailscale_status.py --help
```

## After installation

Once installed via `pip`/`pipx`, the entry point becomes `homedoc-tailscale-status`:

```bash
homedoc-tailscale-status \
  --out ~/tailscale_runs \
  --tz utc \
  --model llama3.1:8b \
  --server http://localhost:11434 \
  --no-llm
```

## Command-line options

| Flag | Description | Default / Notes |
| --- | --- | --- |
| `--out PATH` | Directory where run artifacts are stored. | `./homedoc_out` |
| `--flat` | Skip per-run timestamped subdirectory; write directly into `--out`. | Disabled |
| `--tz {local,utc}` | Timezone for timestamped folder names and log stamps. | `local` (overridable with `HOMEDOC_TZ`) |
| `--debug` | Emit verbose logging to stdout and log file. | Disabled |
| `--log-file PATH` | Write logs to an explicit path instead of `OUT/<run>/homedoc.log`. | `None` (auto) |
| `--timeout SECONDS` | Timeout for `tailscale` subprocess and LLM HTTP calls. | `60` (via `HOMEDOC_HTTP_TIMEOUT`) |
| `--input-json FILE` | Use a saved `tailscale status --json` output instead of calling `tailscale`; malformed files now produce a clear error instead of a crash. | Requires readable file |
| `--no-llm` | Skip LLM call; still writes snapshot and Markdown table/findings. | Disabled |
| `--json-only` | Only emit JSON artifacts (`status.json`, `snapshot.json`, `insights.json`). | Disabled |
| `--llm-mode {auto,generate,chat}` | Select Ollama HTTP path preference. | `auto` (via `HOMEDOC_LLM_MODE`) |
| `--server URL` | Base URL for the Ollama-compatible LLM server. | `http://127.0.0.1:11434` (via `HOMEDOC_LLM_SERVER`) |
| `--model TAG` | LLM model identifier/tag. | `gemma3:12b` (via `HOMEDOC_LLM_MODEL`) |
| `--stream` / `--no-stream` | Force-enable or disable server streaming responses. | Streaming enabled unless `--no-stream` or `HOMEDOC_STREAM=0` |

## Environment variables

The CLI defaults are derived from a few optional environment variables:

- `HOMEDOC_LLM_MODEL` — default for `--model`.
- `HOMEDOC_LLM_SERVER` — default for `--server`.
- `HOMEDOC_TZ` — default timezone (`local` or `utc`).
- `HOMEDOC_HTTP_TIMEOUT` — default timeout in seconds.
- `HOMEDOC_STREAM` — set to `0`, `false`, or `False` to disable streaming by default.
- `HOMEDOC_LLM_MODE` — default mode for `--llm-mode`.
- `HOMEDOC_MAX_LOG_LINES` — maximum in-memory log lines kept by the logger buffer.

## Prerequisites

- The `tailscale` CLI must be available on your `PATH`, unless you provide `--input-json` with a captured `tailscale status --json` output.
- To use the LLM functionality, run an Ollama-compatible HTTP server and ensure `--server` points to it.
