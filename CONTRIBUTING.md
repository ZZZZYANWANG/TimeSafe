# Contributing

TimeSafe treats temporal-integrity claims as executable contracts.

1. Open an issue that states the failure mode and the temporal contract it violates.
2. Add a minimal failing or adversarial regression test before changing implementation code.
3. Preserve evidence semantics: sampled evidence is not an exhaustive `PASS`, and zero evidence is `INCONCLUSIVE`.
4. Keep the core platform-neutral. Platform adapters and external research packages belong in `examples/` or `experiments/`.
5. Do not add a backtester, optimizer, multiple-testing library, or execution simulator to the core package.
6. Never vendor private signals, model weights, credentials, or research-only data without explicit redistribution rights.

Run before opening a pull request:

```bash
pip install -e ".[dev]"
ruff check .
python -m compileall -q src tests examples experiments
bash -n experiments/actr20_public/run_all.sh
pytest --cov=timesafe_quant --cov-fail-under=85
```

A pull request must explain what the new check can prove, what false positives or false negatives remain, and which scope claims changed.
