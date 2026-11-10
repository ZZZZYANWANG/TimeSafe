import timesafe_quant as tq


def test_public_api_exports_v02_features():
    assert tq.__version__ == "0.2.0"
    assert callable(tq.audit_prefix_invariance)
    assert callable(tq.audit_refit_prefix_invariance)
    assert callable(tq.purged_train_test_masks)
