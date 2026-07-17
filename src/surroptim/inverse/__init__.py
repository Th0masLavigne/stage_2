"""Inverse-problem building blocks: virtual sensors + SciPy objective.

Scope for issue #3 (parameter identification): extract simulated
temperatures at fixed (r,z) points efficiently (see the explicit warning
in the issue about the bounding-box tree being rebuilt every call), and
turn "simulated vs measured" into a scalar SciPy can minimise.

This subpackage does NOT touch `AxisymHeatProblem.solve()` itself
(that's issue #8) and does NOT decide which parameters become
`dolfinx.fem.Constant` (that refactor happens where the forward model is
assembled, e.g. in the guidance/main notebooks) -- it only provides the
two genuinely reusable, testable pieces: point evaluation and the
objective function.
"""

from __future__ import annotations

from .objective import compute_mse
from .sensors import SensorLocator

__all__ = ["SensorLocator", "compute_mse"]
