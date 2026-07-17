"""Integration test: assemble build_residual/build_jacobian on a tiny mesh.

Uses a 2x2 unit-square mesh (fast) purely to validate that build_residual
produces a form dolfinx can assemble -- not to check physical accuracy
(that needs the real cylinder geometry and is exercised manually via
notebooks/guidance/linear_stand_in.ipynb and notebooks/main_nonlinear.ipynb).

Requires the full FEniCSx stack (unlike tests/unit/test_boundary_terms.py),
hence living under tests/integration/ rather than tests/unit/.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh

import dolfinx.fem.petsc  # noqa: F401  -- see notebooks/main_nonlinear.ipynb note:
# dolfinx.fem.petsc must be imported explicitly, `from dolfinx import fem`
# alone does not pull in the PETSc-dependent submodule.

from surroptim.weak_form import build_jacobian, build_residual

EPSILON = 0.8
SIGMA = 5.670374419e-8
T_AMB_K = 293.15


@pytest.fixture
def tiny_space():
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1))
    return domain, V


@pytest.mark.xfail(reason="Issue #2: build_residual not yet implemented.", strict=False)
def test_residual_is_zero_at_ambient_with_no_source(tiny_space):
    """At T=T_n=0 with no source, F must assemble to (near) zero.

    This is the FEM-level equivalent of the acceptance criterion "the
    first time step must yield the exact same result as the linear
    model": with T=0 everywhere and no heating yet, nothing should be
    driving the system, radiative term included.
    """
    domain, V = tiny_space

    T = fem.Function(V)  # T = 0 (default)
    T_n = fem.Function(V)  # T_n = 0 (default)
    v = ufl.TestFunction(V)

    r = fem.Function(V)
    r.interpolate(lambda x: np.maximum(np.abs(x[0]), 1e-14))

    source = fem.Function(V)  # zero source

    F = build_residual(
        T,
        T_n,
        v,
        r,
        dt=1.0e-3,
        thermal_capacity=1.0,
        diffusion_coeff=1.0,
        advection_coeff=1.0,
        source=source,
        epsilon=EPSILON,
        sigma=SIGMA,
        t_amb_k=T_AMB_K,
    )
    b = dolfinx.fem.petsc.assemble_vector(fem.form(F))
    assert np.allclose(b.array, 0.0, atol=1e-12)


@pytest.mark.xfail(reason="Issue #2: build_residual not yet implemented.", strict=False)
def test_jacobian_is_assemblable(tiny_space):
    """build_jacobian(F, T) must produce a form dolfinx can assemble.

    Does not check numerical values here (that is what comparing against
    the dummy linear loss term in the guidance notebook is for) -- only
    that ufl.derivative + assemble_matrix succeed end to end.
    """
    domain, V = tiny_space

    T = fem.Function(V)
    T_n = fem.Function(V)
    v = ufl.TestFunction(V)

    r = fem.Function(V)
    r.interpolate(lambda x: np.maximum(np.abs(x[0]), 1e-14))
    source = fem.Function(V)

    F = build_residual(
        T,
        T_n,
        v,
        r,
        dt=1.0e-3,
        thermal_capacity=1.0,
        diffusion_coeff=1.0,
        advection_coeff=1.0,
        source=source,
        epsilon=EPSILON,
        sigma=SIGMA,
        t_amb_k=T_AMB_K,
    )
    J = build_jacobian(F, T)
    A = dolfinx.fem.petsc.assemble_matrix(fem.form(J))
    A.assemble()
