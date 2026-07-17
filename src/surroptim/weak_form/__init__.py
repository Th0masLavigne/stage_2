"""Weak-form building blocks for the axisymmetric thermal problem.

This subpackage isolates the *physics* of the weak formulation (boundary
loss terms, residual, jacobian) from the *orchestration* code that still
lives in ``AxisymHeatProblem`` (mesh, time loop, I/O, sensors -- see
issue #8 for that separate refactor).

Scope for issue #2 (thermal radiation): only what is needed to move from
the linear bilinear form ``a(T, v) = L(v)`` to the non-linear residual
``F(T; v) = 0`` lives here. See ``physique_formulations.md`` (section 4)
for the full derivation this module implements.
"""

from __future__ import annotations

from .boundary_terms import radiative_flux, robin_loss
from .residual import build_jacobian, build_residual

__all__ = [
    "radiative_flux",
    "robin_loss",
    "build_residual",
    "build_jacobian",
]
