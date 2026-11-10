from timesafe_quant.cli import main


def test_demo_cli_writes_markdown(tmp_path):
    output = tmp_path / "demo.md"
    code = main(["demo", "--output", str(output)])
    assert code == 0
    assert output.exists()
    assert "TimeSafe demonstration report" in output.read_text(encoding="utf-8")
