"""Tests d'installation / d'environnement.

Ces tests ne testent PAS la physique du probleme : ils verifient que
l'environnement numerique (image Docker ou conda) fournit bien la pile
FEniCSx attendue, dans les versions figees pour ce projet.

Ils doivent etre les premiers a tourner (localement, en CI, ou au build de
l'image Docker) : si l'un d'eux echoue, il ne sert a rien de debugger la
physique ou le solveur avant d'avoir un socle d'environnement stable.

Executer uniquement ce fichier :
    pytest -m install -q
ou directement :
    pytest tests/test_install.py -q
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.install

# Versions figees pour ce projet (image ghcr.io/fenics/dolfinx/dolfinx:v0.11.0).
# Si l'image de base est mise a jour volontairement, mettre a jour ce dict
# en connaissance de cause plutot que de supprimer le test.
EXPECTED_VERSIONS = {
    "dolfinx": "0.11.0.post0",
    "ufl": "2026.1.0",
    "basix": "0.11.0",
    "ffcx": "0.11.0",
}


def test_fenicsx_imports():
    """La pile FEniCSx doit s'importer sans erreur."""
    import basix  # noqa: F401
    import dolfinx  # noqa: F401
    import ffcx  # noqa: F401
    import ufl  # noqa: F401


@pytest.mark.parametrize("module_name,expected", list(EXPECTED_VERSIONS.items()))
def test_fenicsx_versions(module_name, expected):
    """Les versions installees doivent correspondre a celles figees pour le projet."""
    module = pytest.importorskip(module_name)
    actual = module.__version__
    assert actual == expected, (
        f"{module_name} version installee = {actual!r}, attendue = {expected!r}. "
        "Mettre a jour EXPECTED_VERSIONS si le changement est volontaire."
    )


def test_mpi_available():
    """mpi4py doit etre utilisable, meme en execution mono-processus."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size >= 1


def test_petsc_ksp_available():
    """PETSc doit etre accessible pour construire un solveur lineaire direct (LU)."""
    from mpi4py import MPI
    from petsc4py import PETSc

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)


def test_minimal_scalar_function_space():
    """Smoke test : maillage + espace P1 scalaire, interpolation et dofs > 0.

    Reprend le test minimal historiquement present dans la cellule
    d'installation Colab, pour verifier que l'assemblage FE de base
    fonctionne (pas seulement que les imports passent).
    """
    from mpi4py import MPI
    from dolfinx import fem, mesh

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    Q = fem.functionspace(domain, ("Lagrange", 1))

    q = fem.Function(Q)
    q.interpolate(lambda x: x[0] + x[1])

    n_dofs = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    assert n_dofs > 0


def test_minimal_vector_function_space():
    """Smoke test : espace vectoriel P1 (utilise pour le champ de deplacement)."""
    from mpi4py import MPI
    from dolfinx import fem, mesh

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    u = fem.Function(V)
    u.interpolate(lambda x: (x[0], x[1]))

    assert V.dofmap.index_map_bs == domain.geometry.dim


def test_ufl_symbolic_derivative_available():
    """La differentiation automatique UFL doit fonctionner.

    C'est le mecanisme dont on aura besoin pour la jacobienne du terme
    radiatif non lineaire (ufl.derivative), sans avoir a la deriver a la main.
    """
    from mpi4py import MPI
    import ufl
    from dolfinx import fem, mesh

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1))

    T = fem.Function(V)
    v = ufl.TestFunction(V)

    F = (T**4) * v * ufl.dx
    J = ufl.derivative(F, T)

    assert J is not None
