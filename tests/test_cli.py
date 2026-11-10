from timesafe_quant.cli import main


def test_demo_cli_writes_markdown(tmp_path):
    output = tmp_path / "demo.md"
    code = main(["demo", "--output", str(output)])
    assert code == 0
    assert output.exists()
    assert "TimeSafe demonstration report" in output.read_text(encoding="utf-8")


def test_availability_cli_detects_materialized_future_value(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.syspath_prepend(str((__import__("pathlib").Path(__file__).parents[1])))
    dates = pd.bdate_range("2024-01-02", periods=15)
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "asset": "A",
                "future_target": float(i),
                "target_available": dates[min(i + 2, len(dates) - 1)],
            }
        )
    csv_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    output = tmp_path / "availability.json"
    code = main(
        [
            "availability",
            "--input",
            str(csv_path),
            "--callable",
            "examples.availability_pipeline:read_materialized_target",
            "--time-col",
            "date",
            "--asset-col",
            "asset",
            "--rule",
            "future_target:target_available:after_close",
            "--checkpoint-policy",
            "all",
            "--min-history",
            "5",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["prefix"]["status"] == "FAIL"
