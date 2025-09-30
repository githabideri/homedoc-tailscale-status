# Contributing

- Keep the tool single-file and stdlib-only.
- Prefer small, reviewable PRs.
- Match the project's logging style and local-time default.
- For features that add dependencies, discuss first (issue) and gate behind flags.
- (again human remark from githabideri): I actually don't know what kind of contributions I would even expect, so as always with writing, you can do whatever you want, as long as it is good :)

## Release process

1. Update `__version__` in `homedoc_tailscale_status.py`.
2. Update `CHANGELOG.md`.
3. Commit and tag:
   ```bash
   git commit -am "release: v0.1.0"
   git tag v0.1.0
   git push && git push --tags
   ```
