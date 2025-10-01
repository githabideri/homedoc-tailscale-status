# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-10-01
### Fixed
- Improved error handling for `--input-json` snapshots to gracefully report malformed data instead of crashing.

### Documentation
- Added a complete CLI usage guide (including bare `python` invocation) in the README and the new `USAGE.md` reference.

## [0.1.1] - 2025-09-30
### Added
- `--json-only` flag to skip Markdown and write only JSON artifacts.
- Early directory + log file creation; line-buffered writes to reduce tail loss.

### Changed
- Logger now uses a bounded buffer (`collections.deque`) to prevent unbounded memory growth.
- Helpers raise exceptions; `main()` maps to consistent exit codes.
- Safer JSON recovery via `json.JSONDecoder.raw_decode`.
- Streaming hardened with a minimal-progress watchdog and graceful non-stream fallback.
- Markdown rendering now escapes `|` in table cells.
- “Findings” section rendered even without LLM.

### Fixed
- Clearer timeout/streaming error messages and fallback behavior.

## [0.1.0] - 2025-09-30
### Added
- Initial public release.
- Collect `tailscale status --json`, normalize to a compact snapshot, optional LLM Markdown report.
- Writes per-run artifacts: `status.json`, `snapshot.json`, `insights.json`, `report.md`, `llm_raw.txt`, `homedoc.log`.
