import timesafe_quant as tq


def test_public_api_exports_v03_features():
    assert tq.__version__ == "0.3.1"
    assert callable(tq.audit_prefix_invariance)
    assert callable(tq.audit_refit_prefix_invariance)
    assert callable(tq.purged_train_test_masks)
