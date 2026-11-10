import json

from timesafe_quant.demo import run_demo_report
from timesafe_quant.report import render_markdown, write_report


def test_report_renders_markdown_and_json(tmp_path):
    report = run_demo_report()
    markdown = render_markdown(report)
    assert "Evidence matrix" in markdown
    assert "Prefix invariance" in markdown
    assert "coverage=" in markdown

    md_path = write_report(report, tmp_path / "report.md")
    json_path = write_report(report, tmp_path / "report.json")
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "TimeSafe demonstration report"
    assert payload["prefix"]["status"] == "FAIL"
