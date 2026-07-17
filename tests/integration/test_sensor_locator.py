"""Integration tests for surroptim.inverse.sensors.SensorLocator.

Requires the full FEniCSx stack (tiny mesh + fem.Function), hence
tests/integration/ rather than tests/unit/. Two things are checked:
correctness (does it evaluate the right values at the right points) and
the actual point of issue #3's warning -- that the bounding-box tree is
built exactly ONCE, not once per eval() call.
"""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem, mesh
import dolfinx.geometry as geometry

from surroptim.inverse import SensorLocator


@pytest.fixture
def tiny_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V


@pytest.mark.xfail(reason="Issue #3: SensorLocator not yet implemented.", strict=False)
def test_eval_matches_known_function(tiny_space):
    """A linear field T(x,y) = x + y should be recovered exactly at each sensor."""
    domain, V = tiny_space

    uh = fem.Function(V)
    uh.interpolate(lambda x: x[0] + x[1])

    points = np.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.1]])
    locator = SensorLocator(domain, points)

    values = locator.eval(uh)

    assert locator.ok.all()
    expected = points[:, 0] + points[:, 1]
    assert np.allclose(values, expected, atol=1e-10)


@pytest.mark.xfail(reason="Issue #3: SensorLocator not yet implemented.", strict=False)
def test_point_outside_mesh_is_flagged_not_of():
    """A sensor outside [0,1]x[0,1] must be flagged via `ok`, not crash."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("Lagrange", 1))
    uh = fem.Function(V)
    uh.interpolate(lambda x: x[0] + x[1])

    points = np.array([[0.5, 0.5], [5.0, 5.0]])  # second point is outside
    locator = SensorLocator(domain, points)
    values = locator.eval(uh)

    assert locator.ok[0] and not locator.ok[1]
    assert np.isnan(values[1])


@pytest.mark.xfail(reason="Issue #3: SensorLocator not yet implemented.", strict=False)
def test_bb_tree_is_built_exactly_once(monkeypatch, tiny_space):
    """The whole point of issue #3's warning: cache, don't recompute.

    Wraps dolfinx.geometry.bb_tree with a call counter. Construction
    should call it once; three subsequent eval() calls should call it
    zero additional times. If this test fails with count > 1, the
    caching wasn't actually implemented -- correctness alone
    (test_eval_matches_known_function passing) is not enough proof.
    """
    domain, V = tiny_space
    uh = fem.Function(V)
    uh.interpolate(lambda x: x[0] + x[1])

    calls = {"n": 0}
    original_bb_tree = geometry.bb_tree

    def counting_bb_tree(*args, **kwargs):
        calls["n"] += 1
        return original_bb_tree(*args, **kwargs)

    monkeypatch.setattr(geometry, "bb_tree", counting_bb_tree)

    points = np.array([[0.2, 0.3], [0.5, 0.5]])
    locator = SensorLocator(domain, points)
    locator.eval(uh)
    locator.eval(uh)
    locator.eval(uh)

    assert calls["n"] == 1, (
        f"dolfinx.geometry.bb_tree was called {calls['n']} times; "
        "expected exactly 1 (once in __init__, never in eval())."
    )
