# Contributing

1. Open an issue describing the temporal contract or failure mode.
2. Add a minimal failing test before changing implementation code.
3. Keep the package platform-neutral; platform-specific code belongs in `examples/` or an adapter module.
4. Do not add a backtester, optimizer, experiment tracker, or performance-ranking layer to the core package.
5. Run:

```bash
pytest
ruff check .
```

Pull requests should explain whether a new check is necessary, what false positives it can produce, and what it still cannot prove.
