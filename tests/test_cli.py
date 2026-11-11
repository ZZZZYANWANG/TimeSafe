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


def test_prefix_cli_does_not_silently_sort_unsorted_input(tmp_path, monkeypatch, capsys):
    import pandas as pd

    monkeypatch.syspath_prepend(str((__import__("pathlib").Path(__file__).parents[1])))
    dates = pd.bdate_range("2024-01-02", periods=35)
    rows = [
        {"date": date, "asset": "A", "close": float(i + 1), "return": 0.01}
        for i, date in enumerate(dates)
    ]
    rows[10], rows[11] = rows[11], rows[10]
    csv_path = tmp_path / "unsorted.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    code = main(
        [
            "prefix",
            "--input",
            str(csv_path),
            "--callable",
            "examples.custom_pipeline:my_factor",
            "--time-col",
            "date",
            "--asset-col",
            "asset",
            "--checkpoint-policy",
            "all",
            "--min-history",
            "10",
        ]
    )
    assert code == 3
    assert "sorted in non-decreasing order" in capsys.readouterr().err
