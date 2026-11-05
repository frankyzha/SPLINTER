"""scikit-learn estimator checks (`check_estimator`) on the public estimators.

LicketySPLINTER and RESPLINTER (binary only) pass every check. Every
failure of SPLINTER and ShapeSPLINTER must be in the allow-list below.
"""
import warnings

from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import check_estimator

from splinter import SPLINTER, ShapeSPLINTER, LicketySPLINTER, RESPLINTER

# The solver counts each row as round(w * n / sum(w)) whole copies, and leaf
# probabilities use those copies, so integer weights are not equivalent to
# repeated rows (the tree is the same on the check's data; the probabilities
# differ by the rounding).
SPLINTER_EXPECTED_FAILURES = {"check_sample_weight_equivalence_on_dense_data"}


def _failures(estimator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(estimator, on_fail=None)
    assert results
    return [r for r in results if r["status"] == "failed"]


def test_splinter_check_estimator_failures_are_expected():
    failed = _failures(SPLINTER(lgbm_n_estimators=4, max_depth=2))
    names = {r["check_name"] for r in failed}
    assert names <= SPLINTER_EXPECTED_FAILURES, sorted(names - SPLINTER_EXPECTED_FAILURES)


def test_shape_splinter_check_estimator_failures_are_expected():
    failed = _failures(ShapeSPLINTER(lgbm_n_estimators=4, max_depth=2))
    names = {r["check_name"] for r in failed}
    assert names <= SPLINTER_EXPECTED_FAILURES, sorted(names - SPLINTER_EXPECTED_FAILURES)


def test_lickety_passes_check_estimator():
    failed = _failures(LicketySPLINTER(lgbm_n_estimators=4, max_depth=2))
    assert not failed, [(r["check_name"], str(r["exception"])[:200]) for r in failed]


def test_resplinter_passes_check_estimator():
    assert get_tags(RESPLINTER()).classifier_tags.multi_class is False
    assert RESPLINTER()._more_tags() == {"binary_only": True}
    failed = _failures(RESPLINTER(lgbm_n_estimators=4, max_depth=2))
    assert not failed, [(r["check_name"], str(r["exception"])[:200]) for r in failed]
