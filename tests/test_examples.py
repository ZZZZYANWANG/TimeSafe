import runpy
from pathlib import Path


def test_leakage_gallery_runs(capsys):
    example = Path(__file__).parents[1] / "examples" / "leakage_gallery.py"
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "global z-score (leak)" in output
    assert "causal EMA" in output
    assert "first failing stage" in output
