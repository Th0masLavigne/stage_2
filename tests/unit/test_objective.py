"""Unit tests for surroptim.inverse.objective.compute_mse.

Dolfinx-free by construction: forward_model here is a plain Python
callable returning a numpy array, not a real FE solve. This is the point
-- compute_mse's correctness should never depend on FEniCSx being
installed, only on numpy arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from surroptim.inverse import compute_mse

PARAM_NAMES = ["h", "power"]


def fake_forward_model_factory(mapping):
    """Build a forward_model that returns a fixed array per (h, power)."""

    def forward_model(theta):
        return np.asarray(mapping[(theta["h"], theta["power"])], dtype=float)

    return forward_model


@pytest.mark.xfail(reason="Issue #3: compute_mse not yet implemented.", strict=False)
def test_mse_is_zero_when_simulated_matches_measured():
    measured = np.array([1.0, 2.0, 3.0])
    forward_model = fake_forward_model_factory({(2.0, 1.0): measured})

    mse = compute_mse(
        np.array([2.0, 1.0]),
        param_names=PARAM_NAMES,
        forward_model=forward_model,
        measured=measured,
    )
    assert mse == pytest.approx(0.0, abs=1e-12)


@pytest.mark.xfail(reason="Issue #3: compute_mse not yet implemented.", strict=False)
def test_mse_matches_manual_computation():
    measured = np.array([1.0, 2.0, 3.0])
    simulated = np.array([1.5, 2.5, 2.0])
    forward_model = fake_forward_model_factory({(5.0, 0.5): simulated})

    expected = np.mean((simulated - measured) ** 2)
    mse = compute_mse(
        np.array([5.0, 0.5]),
        param_names=PARAM_NAMES,
        forward_model=forward_model,
        measured=measured,
    )
    assert mse == pytest.approx(expected, rel=1e-12)


@pytest.mark.xfail(reason="Issue #3: compute_mse not yet implemented.", strict=False)
def test_param_order_is_respected_not_assumed():
    """Catches the classic bug: silently zipping params in the wrong order.

    forward_model here is deliberately sensitive to which value is "h"
    and which is "power" -- if compute_mse ever zips param_names/params
    in a different order than documented, this test must fail.
    """
    measured = np.array([0.0])
    forward_model = fake_forward_model_factory(
        {
            (3.0, 9.0): np.array([100.0]),  # h=3, power=9  -> far from measured
            (9.0, 3.0): np.array([0.0]),  # h=9, power=3  -> matches measured
        }
    )

    mse_correct_order = compute_mse(
        np.array([9.0, 3.0]),  # h=9, power=3
        param_names=PARAM_NAMES,
        forward_model=forward_model,
        measured=measured,
    )
    assert mse_correct_order == pytest.approx(0.0, abs=1e-12)
