"""Objective function(s) for SciPy-based parameter identification.

Deliberately dolfinx-free: this module only combines plain numpy arrays
and a user-supplied ``forward_model`` callable. That callable is where
the actual FE solve (and the SensorLocator / dolfinx.fem.Constant usage)
lives -- see notebooks/main_inverse_problem.ipynb. Keeping compute_mse
itself free of any FEniCSx import is what makes it fast to unit-test
(see tests/unit/test_objective.py: no mesh, no MPI, no dolfinx).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def compute_mse(
    params: np.ndarray,
    *,
    param_names: Sequence[str],
    forward_model: Callable[[Mapping[str, float]], np.ndarray],
    measured: np.ndarray,
) -> float:
    """Mean squared error between simulated and measured sensor histories.

    This is the function `scipy.optimize.minimize` calls directly: it
    takes the flat parameter vector SciPy manipulates, turns it back
    into a named dict (`forward_model` should not have to guess
    positional ordering), runs the forward model, and reduces the
    comparison to the single scalar SciPy minimises.

    Args:
        params: Trial parameter vector, in the same order as
            ``param_names`` -- this is exactly the array
            ``scipy.optimize.minimize`` passes in at every iteration.
        param_names: Names to zip with ``params``, matching the keys
            ``forward_model`` expects (e.g. ``["h", "power"]``). Keeping
            this explicit (rather than relying on positional order
            inside ``forward_model``) is what lets you reorder or add
            parameters later without silently mismatching values.
        forward_model: Callable mapping a parameter dict to simulated
            sensor temperatures, with the same shape as ``measured``.
            This is where the FE solve, the ``dolfinx.fem.Constant``
            updates, and the cached ``SensorLocator.eval`` call live --
            none of that belongs in this function.
        measured: Reference sensor data (real thermocouples, or
            synthetic data with noise -- see the "test with synthetic
            data first" note in issue #3).

    Returns:
        Scalar mean squared error, ``mean((simulated - measured) ** 2)``.

    Raises:
        NotImplementedError: scaffold only -- implement to satisfy
            ``tests/unit/test_objective.py``.
    """
    raise NotImplementedError(
        "Issue #3: theta = dict(zip(param_names, params)); "
        "simulated = forward_model(theta); "
        "return float(np.mean((simulated - measured) ** 2)). "
        "See tests/unit/test_objective.py for the exact expected values."
    )
