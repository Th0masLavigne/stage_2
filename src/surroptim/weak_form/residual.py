"""Non-linear residual F(T; v) and its jacobian for the thermal problem.

This module replaces the linear pair ``a_ufl`` / ``L_ufl`` (bilinear form
+ linear form, assembled once outside the time loop) with a single
residual form ``F``, evaluated and linearised at every Newton iteration.
See ``physique_formulations.md`` (section 4) for the full derivation;
this module only wires it into UFL/dolfinx.

Structural change to be aware of: ``T`` here is a ``dolfinx.fem.Function``
(the actual unknown being iterated on by the non-linear solver), NOT a
``ufl.TrialFunction`` as in the legacy linear code. There is no separate
trial function for the residual itself -- only the jacobian introduces
one internally, and UFL derives it automatically (see
:func:`build_jacobian`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ufl

from .boundary_terms import radiative_flux, robin_loss

if TYPE_CHECKING:
    # Import only for type hints. Deliberately NOT imported at runtime:
    # this keeps `surroptim.weak_form` importable (and therefore
    # tests/unit/test_boundary_terms.py collectible) without the FEniCSx
    # stack installed. build_residual only ever combines UFL expressions
    # from the fem.Function objects it is given -- it never needs to call
    # dolfinx.fem itself (fem.form()/assemble happen later, in the
    # notebook, outside this function).
    from dolfinx import fem


def build_residual(
    T: fem.Function,
    T_n: fem.Function,
    v: ufl.Argument,
    r: fem.Function,
    dt: float,
    *,
    thermal_capacity: float,
    diffusion_coeff: float,
    advection_coeff: float,
    source: fem.Function,
    epsilon: float,
    sigma: float,
    t_amb_k: float = 293.15,
    ds: Optional[ufl.Measure] = None,
) -> ufl.Form:
    r"""Build the non-linear residual F(T; v) = 0 for one implicit-Euler step.

    F = inertia(T, T_n) + diffusion(T) + boundary_losses(T) - source, i.e::

        F = rho*cp/dt * (T - T_n) * v * r * dx
          + k * grad(T) . grad(v) * r * dx
          + [h*T + eps*sigma*((T+t_amb_k)**4 - t_amb_k**4)] * v * r * ds
          - f * v * r * dx

    Use :func:`surroptim.weak_form.boundary_terms.robin_loss` and
    :func:`surroptim.weak_form.boundary_terms.radiative_flux` for the two
    bracketed terms above -- do not re-derive them inline here.

    Args:
        T: Unknown temperature at t^{n+1} (the Newton iterate -- a
            ``fem.Function``, not a ``ufl.TrialFunction``: this is the
            key structural change from the linear code).
        T_n: Known temperature at t^n.
        v: Test function on the same space as T.
        r: r-coordinate weight function (axisymmetric measure), same as
            the legacy ``r_weight``.
        dt: Time step.
        thermal_capacity: rho * cp.
        diffusion_coeff: k (isotropic).
        advection_coeff: h, the existing Robin/convective coefficient
            (kept under its legacy name for continuity with parameter
            dicts already in use elsewhere in the codebase).
        source: Volumetric source term (interpolated Gaussian), same
            object as the legacy ``self.f``.
        epsilon: Surface emissivity for the radiative term.
        sigma: Stefan-Boltzmann constant.
        t_amb_k: Ambient reference temperature in Kelvin.
        ds: Boundary measure to integrate the loss terms over. Defaults
            to ``ufl.ds`` (the whole exterior boundary) if not given.
            NOTE: whether Gamma_D (z=0) should be included is the open
            physical question from issue #2's acceptance criteria --
            pass a restricted/tagged measure (e.g. ``ds(GAMMA_N)``, see
            ``notebooks/guidance/linear_stand_in.ipynb``) once that is
            resolved, rather than deciding it inside this function.

    Returns:
        The UFL residual form F.

    Raises:
        NotImplementedError: scaffold only.
    """
    raise NotImplementedError(
        "Issue #2: compose F from grad_cyl-equivalent diffusion, the "
        "mass/inertia term, robin_loss + radiative_flux on `ds`, minus "
        "the source term. See physique_formulations.md section 4 for "
        "the exact form, and tests/integration/test_residual_radiation.py "
        "for the expected behaviour at T=0 with no source."
    )


def build_jacobian(F: ufl.Form, T: fem.Function) -> ufl.Form:
    """Return the Gateaux derivative (tangent form) of F with respect to T.

    This is a thin, deliberately trivial wrapper: the point of issue #2
    is to NOT hand-derive the 4*epsilon*sigma*(T+t_amb_k)**3 term. UFL's
    symbolic differentiation does it exactly::

        J = ufl.derivative(F, T)

    Keeping this as a named function (rather than inlining the one-liner
    in the notebook) is only so the notebook reads as
    ``J = build_jacobian(F, T)`` right next to ``F = build_residual(...)``
    -- the intent should be visible, not hidden in a bare UFL call.

    Args:
        F: The residual form returned by :func:`build_residual`.
        T: The unknown with respect to which to differentiate (the same
            ``fem.Function`` passed into :func:`build_residual`).

    Returns:
        The jacobian (tangent) form J.
    """
    return ufl.derivative(F, T)
