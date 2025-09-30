#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
homedoc-tailscale-status: collect `tailscale status --json`, normalize,
optionally summarize with a local LLM (Ollama), and write timestamped
artifacts (local time by default) for quick tailnet documentation.

Usage (examples):
  python homedoc_tailscale_status.py --ollama http://localhost:11434 --model gemma3:12b
  python homedoc_tailscale_status.py --no-llm  # basic report only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request

__version__ = "0.1.0"

# ===== Defaults =====
DEFAULT_OUTDIR = os.environ.get("HOMEDOC_OUTDIR", "outputs")
DEFAULT_PREFIX = os.environ.get("HOMEDOC_PREFIX", "tailnet-report")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("HOMEDOC_MODEL", "gemma3:12b")
DEFAULT_TIMEOUT_S = int(os.environ.get("HOMEDOC_TIMEOUT", "900"))
DEFAULT_MAX_RETRIES = int(os.environ.get("HOMEDOC_MAX_RETRIES", "1"))
DEFAULT_API_MODE = "auto"  # auto|generate|chat
DEFAULT_STREAM = True
DEFAULT_NUM_PREDICT = int(os.environ.get("HOMEDOC_NUM_PREDICT", "256"))
DEFAULT_NUM_CTX = int(os.environ.get("HOMEDOC_NUM_CTX", "2048"))
DEFAULT_LLM_MODE = "markdown"  # markdown|full
DEFAULT_TZ = "local"  # local|utc
DEFAULT_STREAM_CHUNK_LOG = int(os.environ.get("HOMEDOC_STREAM_CHUNK_LOG", "2000"))
DEFAULT_WRITE_LOG = True

# ===== Time =====

def _now_local():
    return _dt.datetime.now().astimezone()


def _now_utc():
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)


def _fmt_stamp(dt: _dt.datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


# ===== Logger =====
class Logger:
    def __init__(self, debug: bool = False, tz: str = DEFAULT_TZ):
        self.debug_enabled = debug
        self.lines: List[str] = []
        self.tz = tz

    def _now(self) -> _dt.datetime:
        return _now_local() if self.tz == "local" else _now_utc()

    def _ts(self) -> str:
        return self._now().strftime("%Y-%m-%d %H:%M:%S")

    def _emit(self, level: str, msg: str) -> None:
        line = f"[{self._ts()}] {level}: {msg}"
        self.lines.append(line)
        print(line)

    def info(self, msg: str) -> None: self._emit("INFO", msg)
    def warn(self, msg: str) -> None: self._emit("WARN", msg)
    def error(self, msg: str) -> None: self._emit("ERROR", msg)
    def debug(self, msg: str) -> None:
        if self.debug_enabled: self._emit("DEBUG", msg)
    def write_to(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ===== CLI =====

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tailscale tailnet reporter (single-file, stdlib-only)")
    p.add_argument("--out", default=DEFAULT_OUTDIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX)
    p.add_argument("--flat", action="store_true")
    p.add_argument("--input-json", default=None, help="Use existing tailscale status JSON file")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--ollama", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--api", choices=["auto","generate","chat"], default=DEFAULT_API_MODE)
    p.add_argument("--stream", dest="stream", action="store_true", default=DEFAULT_STREAM)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    p.add_argument("--llm-mode", choices=["markdown","full"], default=DEFAULT_LLM_MODE)
    p.add_argument("--tz", choices=["local","utc"], default=DEFAULT_TZ)
    p.add_argument("--stream-chunk-log", type=int, default=DEFAULT_STREAM_CHUNK_LOG)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--no-log-file", action="store_true")
    return p.parse_args()


# ===== Ollama =====

def fetch_installed_models(ollama_url: str, timeout: int, log: Logger, debug: bool) -> List[str]:
    try:
        url = f"{ollama_url.rstrip('/')}/api/tags"
        with request.urlopen(request.Request(url), timeout=min(timeout, 10)) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        if debug: log.debug("Installed models: " + (", ".join(names) if names else "(none)"))
        return [n for n in names if isinstance(n, str)]
    except Exception as e:
        if debug: log.debug(f"Failed to fetch installed models: {e}")
        return []


def resolve_model_tag(requested: str, installed: List[str], log: Logger) -> str:
    if requested in installed: return requested
    fam = requested.split(":", 1)[0]
    if f"{fam}:latest" in installed:
        log.info(f"Resolved model tag: {requested} -> {fam}:latest")
        return f"{fam}:latest"
    for n in installed:
        if n.startswith(fam + ":"):
            log.info(f"Resolved model tag: {requested} -> {n}")
            return n
    return requested


def stream_lines(url: str, payload: Dict[str, Any], timeout: int, log: Logger, debug: bool, progress_every: int) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if debug: log.debug("Payload:\n" + json.dumps(payload, indent=2))
    buf: List[str] = []
    chars = last = 0
    with request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if debug: log.debug(f"Non-JSON line: {line[:200]}")
                    continue
                s = obj.get("response")
                if s is None and isinstance(obj.get("message"), dict):
                    s = obj["message"].get("content")
                if s:
                    buf.append(s); chars += len(s)
                    if progress_every > 0 and (chars - last) >= progress_every:
                        last = chars; log.info(f"…streamed ~{chars} chars")
                if obj.get("done"):
                    metrics = obj.get("metrics") or {}
                    if metrics.get("total_duration") is not None:
                        log.info(f"Ollama done. tokens={metrics.get('eval_count')} total_duration={metrics.get('total_duration')}ns")
                    return "".join(buf)
    return "".join(buf)


def _call_once(url: str, payload: Dict[str, Any], timeout: int) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    if isinstance(obj.get("response"), str):
        return obj["response"]
    if isinstance(obj.get("message"), dict) and isinstance(obj["message"].get("content"), str):
        return obj["message"]["content"]
    raise RuntimeError("Unexpected Ollama response body")


def call_ollama(ollama_url: str, model: str, api_mode: str, llm_mode: str, prompt: Dict[str, str],
                timeout: int, stream: bool, num_predict: int, num_ctx: int, log: Logger, debug: bool,
                progress_every: int) -> Tuple[str, str]:
    base = ollama_url.rstrip('/')
    eff_model = resolve_model_tag(model, fetch_installed_models(ollama_url, timeout, log, debug), log)
    opts = {"temperature": 0.1, "num_predict": num_predict, "num_ctx": num_ctx}

    def do_generate():
        payload = {"model": eff_model, "options": opts, "stream": stream}
        if llm_mode == "full": payload["format"] = "json"
        payload["prompt"] = prompt["system"] + "\n\n" + prompt["user"]
        url = f"{base}/api/generate"
        return (stream_lines(url, payload, timeout, log, debug, progress_every) if stream else _call_once(url, payload, timeout)), "/api/generate"

    def do_chat():
        payload = {"model": eff_model, "options": opts, "stream": stream, "messages": [
            {"role":"system","content":prompt["system"]},
            {"role":"user","content":prompt["user"]},
        ]}
        if llm_mode == "full": payload["format"] = "json"
        url = f"{base}/api/chat"
        return (stream_lines(url, payload, timeout, log, debug, progress_every) if stream else _call_once(url, payload, timeout)), "/api/chat"

    if api_mode in ("auto", "generate"):
        try: return do_generate()
        except urlerror.HTTPError as e:
            if api_mode == "auto" and e.code == 404: log.warn("/api/generate 404 -> falling back to /api/chat")
            else: raise
    return do_chat()


# ===== Prompt & findings =====

def compact_for_llm(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    small = []
    for d in snapshot.get("devices", []):
        small.append({
            "role": d.get("role"),
            "name": d.get("name"),
            "dns": d.get("dns") or d.get("hostname"),
            "addr": (d.get("addrs") or [None])[0],
            "tags": d.get("tags") or [],
            "online": bool(d.get("online")),
        })
    return {"run_id": snapshot.get("run_id"), "device_count": len(small), "devices": small}


def compute_findings(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    offline, tag_summary = [], {}
    for d in snapshot.get("devices", []):
        if not d.get("online"):
            label = d.get("name") or d.get("dns") or (d.get("addrs") or [None])[0] or "unknown"
            offline.append(str(label))
        for t in d.get("tags", []) or []:
            tag_summary[t] = tag_summary.get(t, 0) + 1
    return {"offline_devices": offline, "tag_summary": tag_summary}


def build_prompts(snapshot: Dict[str, Any], llm_mode: str, prev_err: Optional[str]) -> Dict[str, str]:
    small = compact_for_llm(snapshot)
    if llm_mode == "full":
        schema = (
            'Return ONLY a single JSON object with this shape:\n'
            '{\n  "markdown": string,\n  "findings": {\n    "offline_devices": string[],\n    "tag_summary": { [tag: string]: number }\n  }\n}\n'
            'No extra keys. No trailing commas. No code fences. No commentary.'
        )
        system = (
            "You are a meticulous infra doc writer. Never invent facts.\n"
            "Use ONLY the provided device summary JSON. Be concise.\n" + schema
        )
        hint = f"\nPrevious validation error: {prev_err}\n" if prev_err else ""
        user = hint + "DEVICE SUMMARY JSON FOLLOWS:\n" + json.dumps(small, ensure_ascii=False, indent=2)
        return {"system": system, "user": user}
    system = (
        "Write terse, actionable infra notes (<=10 bullets). Do NOT echo JSON. Focus on: \n"
        "- offline devices (names),\n- tag counts/oddities,\n- 1–2 hygiene suggestions."
    )
    user = "DEVICE SUMMARY JSON FOLLOWS:\n" + json.dumps(small, ensure_ascii=False, indent=2)
    return {"system": system, "user": user}


# ===== JSON validation (full mode) =====

def try_extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find('{')
    if start == -1: return None
    depth = 0; in_str = False; esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth += 1
            elif ch == '}': depth -= 1; \
                (0 == depth) and (lambda: None)()
            if depth == 0 and i >= start:
                try: return json.loads(text[start:i+1])
                except json.JSONDecodeError: return None
    return None


def validate_full_json(obj_or_str: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(obj_or_str) if isinstance(obj_or_str, str) else obj_or_str
    except json.JSONDecodeError:
        data = try_extract_json_object(obj_or_str if isinstance(obj_or_str, str) else "")
        if data is None: return None, "invalid JSON and no recoverable object found"
    if not isinstance(data, dict): return None, "top-level is not an object"
    if "markdown" not in data or "findings" not in data: return None, "missing required keys: markdown and findings"
    if not isinstance(data.get("markdown"), str): return None, "markdown must be a string"
    findings = data.get("findings")
    if not isinstance(findings, dict): return None, "findings must be an object"
    if "offline_devices" not in findings or "tag_summary" not in findings:
        return None, "findings missing keys: offline_devices, tag_summary"
    if not isinstance(findings["offline_devices"], list) or not all(isinstance(x, str) for x in findings["offline_devices"]):
        return None, "offline_devices must be an array of strings"
    if not isinstance(findings["tag_summary"], dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in findings["tag_summary"].items()):
        return None, "tag_summary must be an object of string->number"
    return data, None


# ===== Rendering =====

def render_basic_markdown(snapshot: Dict[str, Any]) -> str:
    lines = [f"# Tailscale Tailnet Report — {snapshot['run_id']}", "",
             f"_Devices discovered_: **{snapshot.get('device_count', 0)}**", "",
             "## Devices", "",
             "| Role | Name | Hostname | DNS | Addrs | Tags | Online | OS | Last seen |",
             "|---|---|---|---|---|---|---:|---|---|"]
    for d in snapshot.get("devices", []):
        addrs = ", ".join(d.get("addrs") or []); tags = ", ".join(d.get("tags") or [])
        online = "yes" if d.get("online") else "no"
        lines.append(f"| {d.get('role','')} | {d.get('name','')} | {d.get('hostname') or '—'} | {d.get('dns') or '—'} | {addrs or '—'} | {tags or '—'} | {online} | {d.get('os','') or '—'} | {d.get('last_seen') or '—'} |")
    lines.append("")
    return "\n".join(lines)


def render_from_llm(snapshot: Dict[str, Any], insights: Dict[str, Any]) -> str:
    base = render_basic_markdown(snapshot)
    lines = [base, "## Findings", ""]
    md = (insights.get("markdown") or "").strip()
    if md: lines += [md, ""]
    lines.append("### Structured findings (summary)")
    offline = insights.get("findings", {}).get("offline_devices", [])
    tag_summary = insights.get("findings", {}).get("tag_summary", {})
    lines.append(f"- Offline devices: {', '.join(offline) if offline else 'none'}")
    if tag_summary:
        lines.append("- Tag summary:")
        for k, v in sorted(tag_summary.items()):
            lines.append(f"  - {k}: {v}")
    lines += ["", "_See `*_insights.json` for machine-parseable details._", ""]
    return "\n".join(lines)


# ===== Tailscale collect/normalize =====

def _iter_peers(ts: Dict[str, Any]) -> List[Dict[str, Any]]:
    peers = ts.get("Peer") or ts.get("Peers") or []
    if isinstance(peers, dict): return list(peers.values())
    if isinstance(peers, list): return peers
    return []


def normalize_tailscale(ts: Dict[str, Any], log: Logger, tz: str) -> Dict[str, Any]:
    log.info("Normalizing snapshot")
    now = _now_local() if tz == "local" else _now_utc()
    run_id = _fmt_stamp(now)
    devices: List[Dict[str, Any]] = []

    def extract(node: Dict[str, Any], role: str) -> Dict[str, Any]:
        name = node.get("HostName") or node.get("DNSName") or node.get("Name") or ""
        dns = node.get("DNSName"); host = node.get("HostName")
        ips = node.get("TailscaleIPs") or node.get("Addresses") or []
        tags = node.get("Tags") or []
        online = bool(node.get("Online", True))
        osname = node.get("OS") or node.get("PeerOS") or ""
        last_seen = node.get("LastSeen") or node.get("LastSeenTime") or ""
        user = node.get("User") or node.get("UserID") or ""
        return {"role": role, "name": name, "hostname": host, "dns": dns, "addrs": ips,
                "tags": tags, "online": online, "os": osname, "user": user, "last_seen": last_seen}

    if isinstance(ts.get("Self"), dict): devices.append(extract(ts["Self"], "self"))
    for peer in _iter_peers(ts): devices.append(extract(peer, "peer"))

    snap = {"run_id": run_id, "timezone": tz, "source": "tailscale status --json",
            "device_count": len(devices), "devices": devices}
    log.info(f"Snapshot contains {snap['device_count']} device(s)")
    return snap


def run_tailscale_status_json(log: Logger) -> Dict[str, Any]:
    log.info("Collecting: `tailscale status --json`")
    try:
        proc = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, check=True)
        if log.debug_enabled:
            log.debug(f"tailscale stdout length: {len(proc.stdout)} bytes")
            if proc.stderr: log.debug(f"tailscale stderr: {proc.stderr.strip()}")
    except FileNotFoundError:
        log.error("`tailscale` not found on PATH. Install Tailscale or pass --input-json."); sys.exit(1)
    except subprocess.CalledProcessError as e:
        log.error(f"tailscale failed: {e.stderr.strip() or e.stdout.strip() or e}"); sys.exit(1)
    try:
        data = json.loads(proc.stdout); log.info("Collected tailscale status JSON."); return data
    except json.JSONDecodeError:
        log.error("tailscale returned invalid JSON."); sys.exit(1)


# ===== Output paths =====

def compute_paths(out_root: Path, prefix: str, flat: bool, tz: str) -> Dict[str, Path]:
    now = _now_local() if tz == "local" else _now_utc(); stamp = _fmt_stamp(now)
    if flat:
        base = out_root; base.mkdir(parents=True, exist_ok=True); stem = f"{prefix}_{stamp}"
        return {"base": base, "stem": stem,
                "status": base / f"{stem}_status.raw.json",
                "snapshot": base / f"{stem}_snapshot.json",
                "insights": base / f"{stem}_insights.json",
                "report": base / f"{stem}_report.md",
                "llm_raw": base / f"{stem}_llm.raw.txt",
                "log": base / f"{stem}.log"}
    run_dir = out_root / f"{prefix}_{stamp}"; run_dir.mkdir(parents=True, exist_ok=True); stem = f"{prefix}_{stamp}"
    return {"base": run_dir, "stem": stem,
            "status": run_dir / f"{stem}_status.raw.json",
            "snapshot": run_dir / f"{stem}_snapshot.json",
            "insights": run_dir / f"{stem}_insights.json",
            "report": run_dir / f"{stem}_report.md",
            "llm_raw": run_dir / f"{stem}_llm.raw.txt",
            "log": run_dir / f"{stem}.log"}


# ===== Preflight =====

def preflight(args: argparse.Namespace, log: Logger) -> None:
    if args.input_json: return
    if shutil.which("tailscale") is None:
        log.error("Preflight: `tailscale` not found on PATH. Install it or use --input-json.")
        sys.exit(1)


# ===== Main =====

def main() -> int:
    args = parse_args()
    log = Logger(debug=args.debug, tz=args.tz)

    out_root = Path(os.path.expanduser(args.out))
    paths = compute_paths(out_root, args.prefix, args.flat, args.tz)

    log_path: Optional[Path] = None
    if not args.no_log_file and DEFAULT_WRITE_LOG:
        log_path = Path(os.path.expanduser(args.log_file)) if args.log_file else paths["log"]

    log.info("Starting homedoc run")
    log.info(f"Config: out_root={out_root} flat={args.flat} prefix={args.prefix} tz={args.tz}")
    log.info(
        "LLM: enabled=%s model=%s server=%s timeout=%ss retries=%s api=%s stream=%s num_predict=%s num_ctx=%s llm_mode=%s stream_chunk_log=%s"
        % (not args.no_llm, args.model, args.ollama, args.timeout, args.max_retries, args.api, args.stream, args.num_predict, args.num_ctx, args.llm_mode, args.stream_chunk_log)
    )
    log.info(f"Debug mode: {args.debug}")

    preflight(args, log)

    # 1) tailscale JSON
    try:
        if args.input_json:
            ts_status = json.loads(Path(args.input_json).read_text())
            log.info(f"Loaded tailscale JSON from file: {args.input_json}")
        else:
            ts_status = run_tailscale_status_json(log)
        paths["status"].write_text(json.dumps(ts_status, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Wrote raw tailscale status: {paths['status']}")
    except SystemExit as e:
        if log_path:
            try: log.write_to(log_path); print(f"Log written: {log_path}")
            except Exception: pass
        return int(e.code) if isinstance(e.code, int) else 1

    # 2) normalize
    snapshot = normalize_tailscale(ts_status, log, args.tz)
    paths["snapshot"].write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Wrote snapshot: {paths['snapshot']}")

    # 3) LLM or basic report
    if args.no_llm:
        md = render_basic_markdown(snapshot)
        paths["report"].write_text(md, encoding="utf-8")
        log.info("LLM disabled (--no-llm). Basic report written.")
        log.info(f"Report: {paths['report']}")
        if log_path: log.write_to(log_path); print(f"Log written: {log_path}")
        print("Wrote:\n  %s\n  %s\n  %s" % (paths["status"], paths["snapshot"], paths["report"]))
        return 0

    prev_err = None
    insights: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    endpoint_used: Optional[str] = None

    for attempt in range(args.max_retries + 1):
        log.info(f"LLM attempt {attempt+1}/{args.max_retries+1}")
        try:
            prompts = build_prompts(snapshot, args.llm_mode, prev_err)
            raw_text, endpoint_used = call_ollama(
                args.ollama, args.model, args.api, args.llm_mode,
                prompts, args.timeout, args.stream, args.num_predict, args.num_ctx, log, args.debug,
                args.stream_chunk_log
            )
        except urlerror.HTTPError as e:
            log.error(f"HTTPError from Ollama: {e.code} {e.reason}"); prev_err = f"HTTP {e.code} {e.reason}"; break
        except Exception as e:
            prev_err = f"Ollama call failed: {e}"; log.error(prev_err); break

        try:
            paths["llm_raw"].write_text(raw_text or "", encoding="utf-8"); log.info(f"Saved raw LLM output: {paths['llm_raw']}")
        except Exception as e:
            log.warn(f"Could not write raw LLM output: {e}")
        if args.debug and raw_text:
            head = raw_text[:600].replace("\n", " "); log.debug(f"LLM raw (first 600 chars): {head}{'…' if len(raw_text)>600 else ''}")

        if args.llm_mode == "markdown":
            insights = {"markdown": (raw_text or "").strip(), "findings": compute_findings(snapshot)}
            break

        parsed, err = validate_full_json(raw_text)
        if err is None: insights = parsed; break
        prev_err = err
        snip = (raw_text or "").strip().replace("\n", " ")[:200]
        log.warn(f"Validation failed: {err}; snippet: {snip}{'…' if raw_text and len(raw_text)>200 else ''}")

    if insights is None:
        log.warn("Falling back to basic report due to LLM failure/timeout. Note: server-side generation may continue.")
        md = render_basic_markdown(snapshot)
        paths["report"].write_text(md, encoding="utf-8")
        log.info(f"Report: {paths['report']}")
        if log_path: log.write_to(log_path); print(f"Log written: {log_path}")
        print("Wrote:\n  %s\n  %s\n  %s" % (paths["status"], paths["snapshot"], paths["report"]))
        return 1

    paths["insights"].write_text(json.dumps(insights, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Wrote insights: {paths['insights']}")

    md = render_from_llm(snapshot, insights)
    paths["report"].write_text(md, encoding="utf-8")
    log.info(f"Report: {paths['report']}")

    if log_path:
        log.write_to(log_path); print(f"Log written: {log_path}")

    print("Wrote:")
    for k in ("status", "snapshot", "insights", "report", "llm_raw"):
        print(f"  {paths[k]}")
    if endpoint_used:
        print(f"LLM: model={args.model} server={args.ollama} api={args.api} endpoint_used={endpoint_used} stream={args.stream} num_predict={args.num_predict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
