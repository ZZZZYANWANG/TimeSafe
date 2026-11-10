import timesafe_quant


def test_version_and_public_api():
    assert timesafe_quant.__version__ == "0.1.0"
    assert callable(timesafe_quant.audit_prefix_invariance)
    assert callable(timesafe_quant.build_label_intervals)
