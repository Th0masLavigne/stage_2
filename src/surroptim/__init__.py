"""surroptim : solveur EF thermique axisymetrique (dolfinx) + couplage thermo-mecanique.

Ce package est la "source de verite" du code numerique. Le notebook
(notebooks/main.ipynb) l'importe et s'en sert comme d'un point d'entree
("main") : il ne doit contenir que de l'orchestration (choix des parametres,
appels de haut niveau, visualisation interactive), plus aucune logique metier.

Etat actuel : SQUELETTE. Les modules ci-dessous restent a extraire de
surroptim_bfgs.py :
    - geometry.py        : grad_cyl, eps_cyl, sigma_iso, eps_therm, ...
    - sources.py          : TimeGatedGaussian
    - problem.py          : AxisymHeatProblem (thermique + mecanique)
    - postprocessing.py   : ensure_pyvista_colab, plot_pv

C'est une etape volontairement laissee ouverte : la migration doit se faire
AVEC l'etudiant (il choisit les coupures de modules, les signatures, les
docstrings), pas etre livree toute faite a sa place.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _version

    __version__ = _version("surroptim")
except Exception:  # pragma: no cover - fallback en environnement non installe
    __version__ = "0.0.0+unknown"
