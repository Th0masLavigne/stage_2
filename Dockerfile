# syntax=docker/dockerfile:1

# Image officielle FEniCSx, EPINGLEE sur le tag exact (pas "stable" ni "nightly",
# pour ne pas subir une mise a jour silencieuse de dolfinx/ufl/basix/ffcx).
# Versions embarquees dans v0.11.0 (verifiees sur conda-forge / GHCR le 2026-07-17) :
#   dolfinx 0.11.0.post0  |  ufl 2026.1.0  |  basix 0.11.0  |  ffcx 0.11.0
# Image multi-arch (linux/amd64, linux/arm64) : fonctionne nativement sur Mac Intel
# et Apple Silicon, Docker choisit la bonne plateforme automatiquement.
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.11.0

LABEL org.opencontainers.image.title="surroptim" \
      org.opencontainers.image.description="Solveur EF thermique axisymetrique (dolfinx) - accompagnement these SIMS"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependances systeme pour le rendu offscreen PyVista (Xvfb/Mesa) : necessaires
# pour plot_pv() dans un conteneur sans affichage graphique.
# NB : libgl1-mesa-glx est un paquet de transition obsolete depuis Ubuntu 23.10+
# (absent des depots de l'image de base, qui est sur une Ubuntu recente) ;
# le remplacement officiel est libgl1 + libglx-mesa0.
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- Etape 1 : dependances Python -------------------------------------------------
# On ne copie que pyproject.toml (+README) d'abord pour profiter du cache Docker :
# tant que les dependances ne changent pas, ce layer n'est pas reconstruit.
COPY pyproject.toml README.md ./

# Squelette minimal du package pour que `pip install -e .` reussisse avant meme
# d'avoir copie le vrai code source (evite d'invalider le cache a chaque edit).
RUN mkdir -p src/surroptim && touch src/surroptim/__init__.py

# dev + notebook installes par defaut dans l'image (boucle de developpement rapide).
# `docs` reste opt-in (`pip install -e .[docs]`) pour ne pas alourdir l'image de
# base avec la toolchain Sphinx tant qu'elle n'est pas utilisee.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -e ".[dev,notebook]"

# --- Etape 2 : code source et tests -----------------------------------------------
COPY src/ src/
COPY tests/ tests/

# Tests d'installation : verifient que dolfinx/ufl/basix/ffcx sont bien dans les
# versions attendues et qu'un maillage/espace fonctionnel minimal se construit.
# Si l'image de base derive un jour, le BUILD echoue ici plutot que de laisser
# l'etudiant deboguer une erreur physique qui n'en est pas une.
RUN python3 -m pytest tests/test_install.py -q

# Jupyter (VS Code peut se connecter au serveur, ou attacher directement au
# conteneur via l'extension Dev Containers).
EXPOSE 8888

CMD ["bash"]
