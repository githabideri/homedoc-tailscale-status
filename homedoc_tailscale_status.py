#!/usr/bin/env python3
"""
HomeDoc — Tailscale Status Snapshot & Report

Single-file, stdlib-only utility that:
  1) collects `tailscale status --json`,
  2) normalizes it to a compact snapshot,
  3) (optionally) queries a local LLM via HTTP to produce a Markdown report with findings,
  4) writes artifacts into a per-run folder with a local-time timestamp (unless --flat).

v0.1.2 (2025-10-01)
- Hardened `--input-json` parsing so malformed snapshots are reported cleanly instead of crashing the run.
- Expanded docs: refreshed the README quickstart and added a dedicated `USAGE.md` reference guide.

License: GPLv3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

__version__ = "0.1.2"

DEFAULT_MODEL = os.environ.get("HOMEDOC_LLM_MODEL", "gemma3:12b")
DEFAULT_SERVER = os.environ.get("HOMEDOC_LLM_SERVER", "http://127.0.0.1:11434")
DEFAULT_TZ = os.environ.get("HOMEDOC_TZ", "local")  # "local" or "utc"
DEFAULT_TIMEOUT = int(os.environ.get("HOMEDOC_HTTP_TIMEOUT", "60"))
DEFAULT_STREAM = os.environ.get("HOMEDOC_STREAM", "1") not in {"0", "false", "False"}
DEFAULT_LLM_MODE = os.environ.get("HOMEDOC_LLM_MODE", "auto")  # auto|generate|chat
MAX_LOG_LINES = int(os.environ.get("HOMEDOC_MAX_LOG_LINES", "5000"))

# ------------------------------- Logging -----------------------------------
class Logger:
    def __init__(self, *, debug: bool = False, tz: str = DEFAULT_TZ, max_lines: int = MAX_LOG_LINES, log_path: Optional[str] = None):
        self.debug_enabled = debug
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.tz = tz
        self._log_file_fp: Optional[io.TextIOWrapper] = None
        if log_path:
            # line-buffered writes (buffering=1) to minimize lost tail on abrupt exit
            os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
            self._log_file_fp = open(log_path, "w", buffering=1, encoding="utf-8", newline="\n")

    def _stamp(self) -> str:
        now = _dt.datetime.now(_dt.timezone.utc).astimezone() if self.tz == "local" else _dt.datetime.utcnow()
        # Use ISO-like human time (no TZ abbreviation to keep portable)
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level: str, msg: str) -> None:
        line = f"[{self._stamp()}] {level}: {msg}"
        self.lines.append(line)
        print(line)
        if self._log_file_fp is not None:
            try:
                self._log_file_fp.write(line + "\n")
            except Exception:
                # Never let logging crash the program
                pass

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)

    def debug(self, msg: str) -> None:
        if self.debug_enabled:
            self._write("DEBUG", msg)

    def close(self) -> None:
        if self._log_file_fp is not None:
            try:
                self._log_file_fp.flush()
                os.fsync(self._log_file_fp.fileno())
            except Exception:
                pass
            try:
                self._log_file_fp.close()
            except Exception:
                pass

# ------------------------------ Utilities ----------------------------------
@dataclass
class Paths:
    run_dir: str
    log_path: Optional[str]
    status_json_path: str
    snapshot_json_path: str
    insights_json_path: str
    report_md_path: str
    raw_llm_path: str


def compute_paths(base_dir: str, *, flat: bool, tz: str, log_file: Optional[str]) -> Paths:
    # Timestamp based on local or UTC time
    now = _dt.datetime.now() if tz == "local" else _dt.datetime.utcnow()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir if flat else os.path.join(base_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "homedoc.log") if log_file is None else log_file

    return Paths(
        run_dir=run_dir,
        log_path=log_path,
        status_json_path=os.path.join(run_dir, "status.json"),
        snapshot_json_path=os.path.join(run_dir, "snapshot.json"),
        insights_json_path=os.path.join(run_dir, "insights.json"),
        report_md_path=os.path.join(run_dir, "report.md"),
        raw_llm_path=os.path.join(run_dir, "llm_raw.txt"),
    )


# ------------------------------ Preflight ----------------------------------
def preflight(args: argparse.Namespace) -> None:
    if args.input_json:
        return  # offline mode: no dependency on tailscale binary
    if shutil.which("tailscale") is None:
        raise FileNotFoundError("`tailscale` not found on PATH; install Tailscale or provide --input-json")


# -------------------------- Tailscale collection ----------------------------
def run_tailscale_status_json(timeout: int, *, log: Logger) -> Dict[str, Any]:
    cmd = ["tailscale", "status", "--json"]
    log.debug(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"tailscale status timed out after {timeout}s") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(f"tailscale status failed: {stderr}") from e

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ValueError("tailscale returned non-JSON output") from e
    return data


def _load_status_from_input(path: str, *, log: Logger) -> Dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error(f"Input JSON not found: {path}")
        log.close()
        sys.exit(1)
    except PermissionError as e:
        log.error(f"Permission denied reading input JSON {path}: {e}")
        log.close()
        sys.exit(1)
    except OSError as e:
        log.error(f"Failed to read input JSON {path}: {e}")
        log.close()
        sys.exit(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Input JSON is not valid JSON ({path}): {e}")
        log.close()
        sys.exit(1)


# ---------------------------- Normalization ---------------------------------
# NOTE: Tailscale status schema evolves; we normalize a subset for reporting.
def _first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v:
            return v
    return None


def normalize_snapshot(status: Dict[str, Any]) -> Dict[str, Any]:
    me = status.get("Self", {})
    peers = status.get("Peer", {}) or status.get("Peers", {}) or {}
    if isinstance(peers, list):
        peer_list = peers
    else:
        peer_list = list(peers.values())

    def _labels(node: Dict[str, Any]) -> List[str]:
        tags = node.get("Tags") or []
        if isinstance(tags, dict):
            tags = list(tags.values())
        elif not isinstance(tags, list):
            tags = [str(tags)]
        return [str(t) for t in tags]

    def _ip_list(node: Dict[str, Any]) -> List[str]:
        ips = node.get("TailscaleIPs") or node.get("Addresses") or []
        return [str(x) for x in ips]

    def _hostname(node: Dict[str, Any]) -> Optional[str]:
        return _first_nonempty(node.get("HostName"), node.get("DNSName"), node.get("Name"))

    def _os(node: Dict[str, Any]) -> Optional[str]:
        osv = node.get("OS") or node.get("OSVersion")
        return str(osv) if osv is not None else None

    devices: List[Dict[str, Any]] = []

    def _mk(node: Dict[str, Any], role: str) -> Dict[str, Any]:
        return {
            "role": role,
            "id": node.get("ID") or node.get("NodeID"),
            "name": node.get("Name") or node.get("HostName") or node.get("DNSName"),
            "hostname": _hostname(node),
            "ips": _ip_list(node),
            "user": node.get("User"),
            "online": bool(node.get("Online", True)),
            "os": _os(node),
            "tags": _labels(node),
            "last_seen": node.get("LastSeen"),
            "created": node.get("Created"),
            "expired": bool(node.get("Expired", False)),
            "key_expired": bool(node.get("KeyExpired", False)),
        }

    if me:
        devices.append(_mk(me, "self"))
    for p in peer_list:
        devices.append(_mk(p, "peer"))

    return {
        "schema": 1,
        "generated_at": int(time.time()),
        "device_count": len(devices),
        "online_count": sum(1 for d in devices if d.get("online")),
        "offline_count": sum(1 for d in devices if not d.get("online")),
        "expired_count": sum(1 for d in devices if d.get("expired") or d.get("key_expired")),
        "devices": devices,
    }


# ------------------------------ Markdown -----------------------------------
def _md_escape(x: Optional[str]) -> str:
    s = "" if x is None else str(x)
    # escape pipes to avoid breaking tables
    return s.replace("|", "\\|")


def render_markdown(snapshot: Dict[str, Any], findings: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Tailscale Status Report\n")
    lines.append(f"Generated: <time>{_dt.datetime.now().isoformat(timespec='seconds')}</time>\n")
    lines.append("")

    lines.append("## Summary\n")
    lines.append(f"* Devices: {snapshot['device_count']}  ")
    lines.append(f"* Online: {snapshot['online_count']}  ")
    lines.append(f"* Offline: {snapshot['offline_count']}  ")
    lines.append(f"* Key/Node expired: {snapshot['expired_count']}\n")

    if findings:
        lines.append("## Findings\n")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("## Devices\n")
    lines.append("| role | name | hostname | online | ips | os | tags | last_seen | expired |")
    lines.append("|---|---|---|:--:|---|---|---|---|:--:|")
    for d in snapshot.get("devices", []):
        row = "| {role} | {name} | {hostname} | {online} | {ips} | {os} | {tags} | {last_seen} | {expired} |".format(
            role=_md_escape(d.get("role")),
            name=_md_escape(d.get("name")),
            hostname=_md_escape(d.get("hostname")),
            online="✅" if d.get("online") else "❌",
            ips=_md_escape(", ".join(d.get("ips", []))),
            os=_md_escape(d.get("os")),
            tags=_md_escape(", ".join(d.get("tags", []))),
            last_seen=_md_escape(d.get("last_seen")),
            expired="⚠️" if (d.get("expired") or d.get("key_expired")) else "—",
        )
        lines.append(row)

    return "\n".join(lines) + "\n"


def compute_findings(snapshot: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    if snapshot.get("offline_count", 0) > 0:
        findings.append(f"{snapshot['offline_count']} device(s) offline")
    if snapshot.get("expired_count", 0) > 0:
        findings.append(f"{snapshot['expired_count']} device(s) with expired/invalid keys")
    # Example heuristic: too many peers without tags
    untagged = sum(1 for d in snapshot.get("devices", []) if not d.get("tags"))
    if untagged:
        findings.append(f"{untagged} device(s) without tags — consider labeling for ops hygiene")
    return findings


# ------------------------------- LLM client ---------------------------------
class HttpClient:
    def __init__(self, base_url: str, timeout: int, log: Logger):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log = log

    def _open(self, url: str, data: Optional[bytes]) -> io.BufferedReader:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=self.timeout)  # nosec - local URLs expected

    def post_json(self, path: str, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        self.log.debug(f"HTTP POST {url} payload_bytes={len(data)}")
        try:
            with self._open(url, data) as resp:
                out = resp.read().decode("utf-8", errors="replace")
                return out
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"HTTP {e.code} for {path}: {text}") from e
        except (urllib.error.URLError, socket.timeout) as e:
            raise RuntimeError(f"HTTP error for {path}: {e}") from e

    def stream_lines(self, path: str, payload: Dict[str, Any], *, watchdog_seconds: int = 20) -> Iterator[str]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        self.log.debug(f"HTTP STREAM {url} payload_bytes={len(data)}")
        last_progress = time.time()
        try:
            with self._open(url, data) as resp:
                while True:
                    chunk = resp.readline()
                    if not chunk:
                        break
                    s = chunk.decode("utf-8", errors="replace").strip()
                    if s:
                        last_progress = time.time()
                        yield s
                    # minimal-progress watchdog
                    if time.time() - last_progress > watchdog_seconds:
                        raise RuntimeError(f"No streaming progress for >{watchdog_seconds}s")
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"HTTP {e.code} for {path}: {text}") from e
        except (urllib.error.URLError, socket.timeout) as e:
            raise RuntimeError(f"HTTP error for {path}: {e}") from e


# JSON helpers

def try_extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Recover the first top-level JSON object in text using raw_decode.
    Returns None if no valid object is found or the object isn't a dict."""
    if not text:
        return None
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def validate_full_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # Minimal schema: expect keys 'summary' (str) and optional 'bullets' (list[str])
    if not isinstance(obj, dict):
        return False, "not an object"
    if "summary" not in obj or not isinstance(obj["summary"], str):
        return False, "missing 'summary' string"
    if "bullets" in obj and not (isinstance(obj["bullets"], list) and all(isinstance(x, str) for x in obj["bullets"])):
        return False, "'bullets' must be a list of strings"
    return True, "ok"


def llm_generate_markdown(client: HttpClient, snapshot: Dict[str, Any], *, model: str, mode: str, stream: bool, timeout: int, log: Logger) -> Tuple[str, str]:
    """Return (markdown_text, raw_text). Tries /api/generate first, then /api/chat.
    If streaming fails, falls back to non-streaming."""
    prompt = (
        "You are a systems SRE. Given the following compact JSON snapshot of a Tailscale network, "
        "write a short Markdown report with a 'Findings' section (bulleted), and a concise table. "
        "Focus on offline devices and expired keys. Keep it factual and compact." 
    )
    input_json = json.dumps(snapshot, separators=(",", ":"))

    payload_generate = {
        "model": model,
        "prompt": prompt + "\nSNAPSHOT=" + input_json,
        "stream": stream,
        "options": {"temperature": 0},
    }

    payload_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a terse SRE who writes concise Markdown."},
            {"role": "user", "content": prompt + "\nSNAPSHOT=" + input_json},
        ],
        "stream": stream,
        "options": {"temperature": 0},
    }

    def _collect_stream(path: str, payload: Dict[str, Any]) -> str:
        buf = []
        for line in client.stream_lines(path, payload):
            # Expect provider-specific line JSON with a 'done' flag or partial text
            obj = try_extract_json_object(line)
            if obj:
                if obj.get("done"):
                    break
                # Common fields used by local LLM servers
                txt = obj.get("response") or obj.get("message") or obj.get("content") or ""
                if txt:
                    buf.append(txt)
            else:
                # If plain text lines are streamed, accept them
                buf.append(line)
        return "".join(buf)

    def _post(path: str, payload: Dict[str, Any]) -> str:
        return client.post_json(path, payload)

    tried_paths: List[str] = []

    def _attempt(path: str, payload: Dict[str, Any]) -> str:
        tried_paths.append(path)
        if stream:
            try:
                return _collect_stream(path, payload)
            except Exception as e:
                log.error(f"streaming failed on {path}: {e}; retrying without stream")
                # fall through to non-streaming
        # non-streaming
        return _post(path, {**payload, "stream": False})

    raw = ""
    if mode in ("auto", "generate"):
        try:
            raw = _attempt("/api/generate", payload_generate)
            endpoint_used = "/api/generate"
        except Exception as e:
            log.error(f"/api/generate failed: {e}; trying /api/chat")
            raw = ""
        else:
            if raw:
                md = raw.strip()
                return md, raw

    # Fallback or forced chat mode
    try:
        raw = _attempt("/api/chat", payload_chat)
        endpoint_used = "/api/chat"
    except Exception as e:
        raise RuntimeError(f"Both generate/chat failed: {e}") from e

    md = raw.strip()
    return md, raw


# ------------------------------ Main program --------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Tailscale status, normalize snapshot, optional LLM report.")
    p.add_argument("--out", default="./homedoc_out", help="Output directory (default: ./homedoc_out)")
    p.add_argument("--flat", action="store_true", help="Write artifacts directly to --out without a timestamp subdir")
    p.add_argument("--tz", default=DEFAULT_TZ, choices=["local", "utc"], help="Timestamp timezone for run folder name")
    p.add_argument("--debug", action="store_true", help="Verbose debug logging")
    p.add_argument("--log-file", default=None, help="Write logs to this file (default: OUT/run/homedoc.log)")

    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Subprocess/HTTP timeout seconds")

    # Offline input for testing
    p.add_argument("--input-json", default=None, help="Read Tailscale status from this JSON file instead of running tailscale")

    # LLM controls
    p.add_argument("--no-llm", action="store_true", help="Skip LLM step; only collect & render local report")
    p.add_argument("--json-only", action="store_true", help="Write JSON artifacts, skip Markdown entirely")
    p.add_argument("--llm-mode", default=DEFAULT_LLM_MODE, choices=["auto", "generate", "chat"], help="API path preference")
    p.add_argument("--server", default=DEFAULT_SERVER, help="LLM HTTP base URL (default env HOMEDOC_LLM_SERVER or local)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM model tag (default env HOMEDOC_LLM_MODEL)")
    p.add_argument("--stream", dest="stream", action="store_true", default=DEFAULT_STREAM, help="Enable streaming (default)")
    p.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    paths = compute_paths(args.out, flat=args.flat, tz=args.tz, log_file=args.log_file)
    log = Logger(debug=args.debug, tz=args.tz, max_lines=MAX_LOG_LINES, log_path=paths.log_path)
    log.info(f"HomeDoc v{__version__}")
    log.info(f"Output dir: {paths.run_dir}")

    # Preflight
    try:
        preflight(args)
    except Exception as e:
        log.error(str(e))
        log.close()
        return 2

    # Collect status
    try:
        if args.input_json:
            status = _load_status_from_input(args.input_json, log=log)
            log.info(f"Loaded input JSON: {args.input_json}")
        else:
            status = run_tailscale_status_json(args.timeout, log=log)
            log.info("Collected tailscale status")
        with open(paths.status_json_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Failed to obtain status JSON: {e}")
        log.close()
        return 3

    # Normalize
    try:
        snapshot = normalize_snapshot(status)
        with open(paths.snapshot_json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        log.info("Wrote snapshot.json")
    except Exception as e:
        log.error(f"Normalization failed: {e}")
        log.close()
        return 3

    # Local findings
    local_findings = compute_findings(snapshot)

    # Markdown (always available unless --json-only)
    if not args.json_only:
        try:
            md = render_markdown(snapshot, local_findings)
            with open(paths.report_md_path, "w", encoding="utf-8") as f:
                f.write(md)
            log.info("Wrote report.md (local)")
        except Exception as e:
            log.error(f"Failed to write base Markdown: {e}")

    if args.no_llm:
        log.info("Skipping LLM (per --no-llm)")
        log.close()
        return 0

    # LLM step
    client = HttpClient(args.server, args.timeout, log)
    log.info(f"LLM model requested: {args.model}; mode={args.llm_mode}; stream={args.stream}")

    try:
        md_llm, raw = llm_generate_markdown(client, snapshot, model=args.model, mode=args.llm_mode, stream=args.stream, timeout=args.timeout, log=log)
        # Save raw and final MD (unless json-only, in which case we still save raw text for debugging)
        with open(paths.raw_llm_path, "w", encoding="utf-8") as f:
            f.write(raw)
        log.info("Wrote llm_raw.txt")
        if not args.json_only:
            # If LLM produced something, overwrite/append to report
            with open(paths.report_md_path, "a", encoding="utf-8") as f:
                f.write("\n\n---\n\n## LLM Analysis\n\n")
                f.write(md_llm)
            log.info("Appended LLM section to report.md")
    except Exception as e:
        log.error(f"LLM step failed: {e}")
        log.close()
        return 4

    log.info("Done")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
