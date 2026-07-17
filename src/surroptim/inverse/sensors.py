"""Cached point evaluation for virtual thermocouples.

The legacy `AxisymHeatProblem.solve()` rebuilds the bounding-box tree
(`dolfinx.geometry.bb_tree`) and re-locates the sensor cells every single
time it is called with `sensors_xy` -- including at every SciPy
optimisation iteration, where the mesh/geometry never actually changes
between calls. This is exactly the "[WARNING!]" in issue #3: the fix is
architectural (cache once per mesh), not a numerical one.

Import note: this module imports `from dolfinx import geometry` (the
module), not `from dolfinx.geometry import bb_tree` (the bound function).
Keep it that way -- tests monkeypatch `dolfinx.geometry.bb_tree` to prove
it is only called once, and that only works if lookups go through the
module object at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dolfinx import mesh as dolfinx_mesh
    from dolfinx.fem import Function


class SensorLocator:
    """Locate and evaluate fixed (r, z) points on a FIXED mesh, once.

    Build one instance per mesh/geometry (e.g. right after
    `AxisymHeatProblem._build_mesh_if_needed` runs), then reuse
    ``.eval(uh)`` across every forward solve / SciPy iteration. The
    bounding-box tree and the cell each sensor lives in are computed
    exactly once, in ``__init__`` -- never inside ``eval``.

    Args:
        domain: The dolfinx mesh the sensors live on.
        points_rz: ``(N, 2)`` array of ``(r, z)`` sensor coordinates.

    Attributes:
        ok: Boolean mask of shape ``(N,)``, True where a sensor was
            found inside the mesh (mirrors the legacy ``ok`` return
            value of ``solve()``).

    Raises:
        NotImplementedError: scaffold only -- implement to satisfy
            ``tests/integration/test_sensor_locator.py``.
    """

    def __init__(self, domain: "dolfinx_mesh.Mesh", points_rz: np.ndarray):
        raise NotImplementedError(
            "Issue #3: build the bb_tree ONCE here -- "
            "dolfinx.geometry.bb_tree(domain, domain.topology.dim), then "
            "compute_collisions_points + compute_colliding_cells for "
            "points_rz, and store the resulting per-sensor cell indices "
            "(and the `ok` mask) as instance attributes. Do not repeat any "
            "of this work in eval()."
        )

    def eval(self, uh: "Function") -> np.ndarray:
        """Evaluate ``uh`` at the cached sensor points.

        Must reuse the cell indices computed in ``__init__`` -- must NOT
        call ``dolfinx.geometry.bb_tree`` (or recompute collisions)
        again here.

        Args:
            uh: Current temperature field.

        Returns:
            ``(N,)`` array; ``np.nan`` for sensors where ``self.ok`` is
            False (outside the mesh).

        Raises:
            NotImplementedError: scaffold only.
        """
        raise NotImplementedError(
            "Issue #3: uh.eval(points[self._idx_ok], self._cells[self._idx_ok]), "
            "same pattern as the legacy inline code in solve(), but reading "
            "from attributes set in __init__ instead of recomputing them."
        )
