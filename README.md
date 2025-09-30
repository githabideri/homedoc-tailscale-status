# General Info
First member of the homedoc scripts that query local computer or network feature, parses its content and produces a small (local) LLM enhanced report; homedoc-tailscale-status queries status of tailscale and writes a time stamped report.

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

## Quick start

```bash
python3 homedoc_tailscale_status.py --help

python3 homedoc_tailscale_status.py \
  --ollama http://localhost:11434 \
  --model "gemma3:12b" \
  --llm-mode markdown \
  --timeout 900 \
  --num-predict 256 \
  --stream \
  --stream-chunk-log 0 \
  --debug
```

---

## Outputs

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

## CLI options (highlights)

- `--out outputs` root output dir  
- `--prefix tailnet-report` filename prefix  
- `--flat` write files directly into `--out` (no run subdir)  
- `--input-json PATH` use an existing Tailscale JSON file (skip calling `tailscale`)  
- `--no-llm` skip model, write basic report only  
- `--ollama http://host:11434` Ollama base URL  
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
- Logs and raw LLM output are saved per run; set `--no-log-file` to avoid the `.log`.

---

## Versioning

- Script defines `__version__`.
- Tag releases in Git:  
  ```bash
  git tag v0.1.0
  git push --tags
  ```

---

## Notes

- Requires `tailscale` on PATH.
- Ollama integration is optional (`--no-llm`). For reliability, prefer `--llm-mode markdown`; `--llm-mode full` is stricter and more brittle.
- Streaming is enabled by default; tune verbosity with `--stream-chunk-log`.

---

## Contributing

- Keep the tool single-file and stdlib-only.
- Prefer small, reviewable PRs.
- Match the project's logging style and local-time default.
- For features that add dependencies, discuss first and gate behind flags
- (human remark from githabideri): I actually don't know what kind of contributions I would even expect, so as always with writing, you can do whatever you want, as long as it is good :)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the release checklist.

---

```
SPDX-License-Identifier: GPL-3.0-or-later
```
