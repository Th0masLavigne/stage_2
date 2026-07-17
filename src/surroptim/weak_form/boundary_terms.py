"""Boundary loss terms for the thermal problem (Robin + radiative).

Both functions are written to be polymorphic over plain Python floats
and UFL expressions (``ufl.Argument``, ``dolfinx.fem.Function``, ...), so
that the exact same implementation can be:

- unit-tested with plain floats (fast, no mesh, no MPI, no dolfinx
  import needed for these particular calls), and
- used directly inside a UFL form (``ufl.ds`` integral) when composing
  the residual in :mod:`surroptim.weak_form.residual`.

Do NOT special-case Gamma_D vs Gamma_N inside these functions: which
boundary they get integrated over is decided at the call site (in the
residual / the notebook), not baked in here. See issue #2's acceptance
criteria ("Physical Clarification") -- the status of Gamma_D (z=0) is
still an open *physical* question, not a numerical one, and keeping it
out of these functions is what keeps that question a one-line decision
later instead of a rewrite.
"""

from __future__ import annotations

from typing import Union

import ufl

Scalar = Union[float, ufl.core.expr.Expr]


def robin_loss(T: Scalar, h: float) -> Scalar:
    """Linear convective-type loss, q = h * T.

    This is the term already implemented in the current linear code
    (``advection_coeff * du * v * r * ufl.ds``). Extracting it here lets
    it be reused unchanged when composing the new residual, and gives
    you a "known good" linear reference to validate the new boundary
    selection against before introducing any non-linearity (see the
    Implementation Notes on issue #2: test the new boundary with a
    dummy linear loss term first).

    Args:
        T: Relative temperature (0 == ambient reference), float or UFL
            expression.
        h: Linear loss coefficient (``advection_coeff`` in the legacy
            code -- the name is misleading, it is a Robin/convective
            coefficient, not an advection term).

    Returns:
        Same type as ``T``: ``h * T``.
    """
    return h * T


def radiative_flux(
    T: Scalar,
    epsilon: float,
    sigma: float,
    t_amb_k: float = 293.15,
) -> Scalar:
    """Radiative loss q_rad = epsilon * sigma * ((T + t_amb_k)**4 - t_amb_k**4).

    ``T`` is the relative temperature used everywhere else in the code
    (0 == 20 degC == ``t_amb_k`` Kelvin) -- this is why the Dirichlet
    condition at Gamma_D is already 0, and why this function MUST
    return exactly 0 at T=0 (see the mandatory unit test
    ``test_radiative_flux_zero_at_ambient``: it is not just a sanity
    check, it is an acceptance criterion for issue #2 -- it is also what
    guarantees the first non-linear time step reproduces the linear
    model exactly).

    Args:
        T: Relative temperature, float or UFL expression.
        epsilon: Surface emissivity (dimensionless, in [0, 1]).
        sigma: Stefan-Boltzmann constant (~5.67e-8 W/m^2/K^4). Passed
            explicitly rather than hard-coded so it can be swapped for
            a non-dimensionalised value if Newton conditioning becomes
            an issue (see issue #2's "Matrix Conditioning" note: sigma
            introduces scales very different from thermal conduction).
        t_amb_k: Ambient reference temperature in Kelvin (293.15 K =
            20 degC), matching the offset already used at Gamma_D.

    Returns:
        Same type as ``T``: the radiative flux.

    Raises:
        NotImplementedError: scaffold only -- implement to satisfy
            ``tests/unit/test_boundary_terms.py``.
    """
    raise NotImplementedError(
        "Issue #2: implement radiative_flux. See the docstring above "
        "and tests/unit/test_boundary_terms.py for the exact expected "
        "behaviour (in particular: it must return exactly 0.0 at T=0)."
    )
