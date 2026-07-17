"""Unit tests for surroptim.weak_form.boundary_terms.

These tests use plain Python floats only: no mesh, no MPI, no dolfinx
Function objects. They are the fast, first line of defense for issue #2
and should run in milliseconds.

test_robin_loss_is_linear is NOT marked xfail: robin_loss is already
implemented (it is a direct extraction of existing, working code), so
this test must pass right now. The radiative_flux tests ARE marked
xfail: they are the executable spec for issue #2 -- remove the marker
once radiative_flux has a real body, at which point they must go green.
"""

from __future__ import annotations

import pytest

from surroptim.weak_form import radiative_flux, robin_loss

EPSILON = 0.8
SIGMA = 5.670374419e-8
T_AMB_K = 293.15


def test_robin_loss_is_linear():
    """robin_loss must be exactly h * T (no offset, no saturation)."""
    h = 2.5
    assert robin_loss(0.0, h) == 0.0
    assert robin_loss(10.0, h) == pytest.approx(25.0)
    assert robin_loss(-4.0, h) == pytest.approx(-10.0)


@pytest.mark.xfail(reason="Issue #2: radiative_flux not yet implemented.", strict=False)
def test_radiative_flux_zero_at_ambient():
    """Mandatory acceptance criterion: q_rad(T=0) must be EXACTLY zero.

    T=0 means the surface is already at t_amb_k (20 degC): there is no
    temperature difference to radiate away, so this must hold to
    floating-point precision, not just "close to zero". This is what
    guarantees the first non-linear time step reproduces the linear
    model exactly (see issue #2's acceptance criteria).
    """
    assert radiative_flux(0.0, EPSILON, SIGMA, T_AMB_K) == 0.0


@pytest.mark.xfail(reason="Issue #2: radiative_flux not yet implemented.", strict=False)
def test_radiative_flux_positive_when_hotter_than_ambient():
    """A surface hotter than ambient must lose heat (q_rad > 0)."""
    assert radiative_flux(50.0, EPSILON, SIGMA, T_AMB_K) > 0.0


@pytest.mark.xfail(reason="Issue #2: radiative_flux not yet implemented.", strict=False)
def test_radiative_flux_negative_when_colder_than_ambient():
    """A surface colder than ambient must gain heat (q_rad < 0)."""
    assert radiative_flux(-50.0, EPSILON, SIGMA, T_AMB_K) < 0.0


@pytest.mark.xfail(reason="Issue #2: radiative_flux not yet implemented.", strict=False)
def test_radiative_flux_matches_stefan_boltzmann_formula():
    """Cross-check against the explicit formula for a non-trivial T."""
    T = 30.0
    expected = EPSILON * SIGMA * ((T + T_AMB_K) ** 4 - T_AMB_K**4)
    assert radiative_flux(T, EPSILON, SIGMA, T_AMB_K) == pytest.approx(expected, rel=1e-12)
