# Physique et formulations — problème thermique axisymétrique

Notations : domaine méridien $\omega = (0,R)\times(0,e) \subset \mathbb{R}^2_{(r,z)}$ (le domaine 3D $\Omega$ est obtenu par révolution de $\omega$ autour de l'axe $r=0$). Hypothèse d'axisymétrie : $\partial(\cdot)/\partial\theta = 0$. $T(r,z,t)$ est la température **relative** à l'ambiante $20\,°\mathrm{C}$ (donc $T=0 \Leftrightarrow 20\,°\mathrm{C}$). Correspondance code : $\rho c_p \leftrightarrow$ `thermal_capacity`, $k \leftrightarrow$ `diffusion_coeff`, $h \leftrightarrow$ `advection_coeff`.

---

## 1. Forme forte — modèle actuel (linéaire)

### 1.1 Équation de bilan (dans $\omega$)

En coordonnées cylindriques, sans dépendance en $\theta$, le laplacien scalaire s'écrit :

$$
\nabla\cdot(k\nabla T) = \frac{1}{r}\frac{\partial}{\partial r}\!\left(k\,r\,\frac{\partial T}{\partial r}\right) + \frac{\partial}{\partial z}\!\left(k\,\frac{\partial T}{\partial z}\right)
$$

d'où l'équation de la chaleur transitoire :

$$
\rho c_p \,\frac{\partial T}{\partial t} - \frac{1}{r}\frac{\partial}{\partial r}\!\left(k\,r\,\frac{\partial T}{\partial r}\right) - \frac{\partial}{\partial z}\!\left(k\,\frac{\partial T}{\partial z}\right) = f(r,z,t), \qquad (r,z)\in\omega,\ t\in(0,t_f]
$$

### 1.2 Terme source (actuel)

$$
f(r,z,t) = P\,\exp\!\left[-\frac{1}{2}\left(\left(\frac{r-r_0}{w_r}\right)^2+\left(\frac{z-z_0}{w_z}\right)^2\right)\right]\cdot \mathbb{1}_{\{t\le t_{\text{heat}}\}}
$$

avec par défaut $r_0=0$, $z_0=e$, $w_z \ll w_r$ (gaussienne resserrée près de la face $z=e$).

### 1.3 Conditions aux limites (actuelles)

$$
\begin{cases}
T = 0 & \text{sur } \Gamma_D = \{z=0\} \quad \text{(Dirichlet)}\\[4pt]
-k\dfrac{\partial T}{\partial n} = h\,T & \text{sur } \Gamma_N = \{z=e\}\cup\{r=R\} \quad \text{(Robin / perte convective)}
\end{cases}
$$

### 1.4 Condition initiale

$$
T(r,z,0) = 0 \quad \text{dans } \omega
$$

> **Remarque — poids axisymétrique et axe $r=0$.** Toutes les intégrales du code sont pondérées par $r$ (mesure $2\pi r\,dr\,dz$, le $2\pi$ étant une constante omise). Sur l'axe $r=0$, ce poids s'annule identiquement : c'est pour cela qu'aucune condition explicite n'est imposée en $r=0$ dans le code (`ufl.ds` intègre pourtant géométriquement ce bord du rectangle, mais sa contribution est nulle par construction). C'est la condition de symétrie naturelle $\partial T/\partial r = 0$ obtenue "gratuitement" par la formulation faible axisymétrique — un bon point à vérifier avec l'étudiant plutôt qu'à considérer comme acquis.

> **Remarque — pourquoi `grad_cyl` diffère de `eps_cyl`.** Le gradient scalaire axisymétrique n'a pas de terme de courbure ($\nabla T = (\partial_r T,\partial_z T)$ suffit), contrairement au tenseur des déformations mécaniques qui porte un terme $u_r/r$ (`eps_cyl`). C'est cohérent : seule une grandeur vectorielle "tourne" avec $\theta$, pas un scalaire. Bon test de compréhension à poser à l'étudiant en 5 min.

> **Remarque — source volumique vs flux surfacique.** Le mail décrit un "flux d'entrée gaussien appliqué sur une face". Le code, lui, modélise ce flux comme une **source volumique** concentrée près de $z=e$ (gaussienne étroite en $z$), pas comme un vrai flux de Neumann $-k\,\partial T/\partial n = -q(r,t)$ imposé sur $\{z=e\}$. C'est une approximation numérique répandue (plus simple à coder, pas de marquage de sous-frontière), mais `wz`/`ratio_width_z` pilotent alors une profondeur de pénétration *numérique* sans réalité physique directe — point important si on identifie ensuite ces paramètres par recalage (ils ne seront pas directement interprétables physiquement).

---

## 2. Formulation faible — modèle actuel (linéaire, tel qu'implémenté)

Discrétisation temporelle : Euler implicite, $\partial T/\partial t \approx (T^{n+1}-T^n)/\Delta t$.

Trouver $T^{n+1}\in V_h$ (avec $T^{n+1}=0$ sur $\Gamma_D$) tel que $\forall v\in V_h^0$ :

$$
\underbrace{\int_\omega \rho c_p\,\frac{T^{n+1}}{\Delta t}\,v\,r\,dr\,dz + \int_\omega k\,\nabla T^{n+1}\cdot\nabla v\,r\,dr\,dz + \int_{\Gamma_N} h\,T^{n+1}\,v\,r\,ds}_{a(T^{n+1},v)} = \underbrace{\int_\omega \rho c_p\,\frac{T^{n}}{\Delta t}\,v\,r\,dr\,dz + \int_\omega f^{n+1}\,v\,r\,dr\,dz}_{L(v)}
$$

Ceci correspond exactement à `a_ufl` / `L_ufl` dans le code. $a(\cdot,\cdot)$ est **bilinéaire** (donc indépendante de $T^{n+1}$) car $\rho c_p$, $k$, $h$, $\Delta t$ sont constants : la matrice $A$ est assemblée **une seule fois** hors de la boucle temporelle, seul le second membre $b$ change à chaque pas (résolution directe LU via KSP).

> **Remarque — pourquoi ça casse avec le rayonnement.** Toute la performance de ce schéma repose sur $A$ fixe. Dès qu'un terme dépend de $T^{n+1}$ de façon non affine (ici $T^4$), $a(\cdot,\cdot)$ n'est plus bilinéaire : on ne peut plus séparer "matrice fixe" et "second membre variable". C'est le vrai point de rupture architecturale à faire identifier à l'étudiant — pas un détail de syntaxe UFL.

---

## 3. Extension physique — perte radiative (demande de l'étudiant)

Nouveau terme de flux sortant, ajouté sur la frontière :

$$
q_{\text{rad}}(T) = \varepsilon\,\sigma\Big[(T+293.15)^4 - 293.15^4\Big] \quad [\mathrm{W/m^2}]
$$

Nouvelle condition sur $\Gamma_N$ :

$$
-k\,\frac{\partial T}{\partial n} = h\,T + \varepsilon\,\sigma\Big[(T+293.15)^4-293.15^4\Big] \qquad \text{sur } \Gamma_N
$$

L'équation de bilan dans $\omega$ (section 1.1) est **inchangée** : le rayonnement est un phénomène de bord, pas de volume.

> **Remarque — et sur $\Gamma_D$ ?** Le mail demande le rayonnement "sur l'ensemble de la frontière". Or $\Gamma_D=\{z=0\}$ est actuellement en Dirichlet strict ($T=0$ imposé) : y ajouter un terme de flux radiatif n'aurait numériquement aucun effet, la valeur y étant de toute façon forcée. Il faut trancher avec l'étudiant : soit $z=0$ représente physiquement une face maintenue à température fixe (contact avec un bâti massif, un support froid...) et le Dirichlet reste justifié malgré la formulation "toute frontière" du mail, soit c'est en réalité une frontière libre et il faut la faire basculer en Robin+rayonnement comme les autres. C'est une clarification physique à faire, pas une décision numérique — bon exemple de "vérifier la modification avant de coder".

> **Remarque — cohérence de l'offset 293.15.** Le fait que $T=0$ corresponde exactement à $20\,°\mathrm{C}=293.15\,\mathrm{K}$ n'est pas un hasard : c'est la même convention que la CL Dirichlet actuelle (déjà à $0$). Bon test unitaire immédiat à suggérer à l'étudiant : à $T\equiv 0$ (état initial), $q_{\text{rad}}(0)=0$ exactement — donc le résidu du tout premier pas de temps (avant que la source n'ait chauffé quoi que ce soit) doit être identique à la version sans rayonnement. Si ce n'est pas le cas dans le code, il y a un bug.

---

## 4. Formulation faible — modèle futur (non linéaire, Newton)

Le problème n'a plus de découpage $a(\cdot,\cdot)=L(\cdot)$ : on écrit un **résidu** $F$ et on cherche son zéro.

Trouver $T^{n+1}\in V_h$ (avec $T^{n+1}=0$ sur $\Gamma_D$) tel que $\forall v\in V_h^0$ :

$$
F(T^{n+1};v) = \int_\omega \rho c_p\,\frac{T^{n+1}-T^{n}}{\Delta t}\,v\,r\,dr\,dz + \int_\omega k\,\nabla T^{n+1}\cdot\nabla v\,r\,dr\,dz \\[4pt]
+ \int_{\Gamma_N}\Big[h\,T^{n+1} + \varepsilon\sigma\big((T^{n+1}+293.15)^4-293.15^4\big)\Big]\,v\,r\,ds - \int_\omega f^{n+1}\,v\,r\,dr\,dz = 0
$$

### 4.1 Linéarisation de Newton

À partir de $T^{n+1,0}=T^n$, itérer jusqu'à convergence :

$$
J\big(T^{n+1,k};\,\delta T^k,\,v\big) = -F\big(T^{n+1,k};v\big) \quad \forall v\in V_h^0, \qquad T^{n+1,k+1} = T^{n+1,k}+\delta T^k
$$

avec la forme tangente (Gateaux-différentielle de $F$) :

$$
J(T;\delta T,v) = \int_\omega \rho c_p\,\frac{\delta T}{\Delta t}\,v\,r\,dr\,dz + \int_\omega k\,\nabla \delta T\cdot\nabla v\,r\,dr\,dz + \int_{\Gamma_N}\Big[h + 4\varepsilon\sigma(T+293.15)^3\Big]\delta T\,v\,r\,ds
$$

> **Remarque — ne pas dériver ça à la main dans le code.** L'expression analytique ci-dessus sert à *vérifier* ce qui se passe, mais dans dolfinx/UFL on obtient $J$ automatiquement par différentiation symbolique : `J = ufl.derivative(F, T, dT)`. Le solveur non linéaire (`dolfinx.fem.petsc.NonlinearProblem` + `NewtonSolver`) gère ensuite les itérations. Le vrai travail d'accompagnement est de faire *comprendre* d'où vient le terme $4\varepsilon\sigma(T+293.15)^3$ (pour que l'étudiant sache lire/valider ce que UFL calcule), pas de le lui faire coder à la main.

### 4.2 Alternative de test progressif (avant Newton complet)

Deux paliers possibles, du plus simple au plus fidèle :

1. **Semi-implicite / explicite sur le terme radiatif** : évaluer $q_{\text{rad}}$ avec $T^n$ (connu) et le passer au second membre. Le système reste **linéaire** ($A$ toujours fixe), donc réutilise l'architecture actuelle quasi telle quelle. Coût : stabilité conditionnelle — un $\Delta t$ trop grand peut faire diverger le schéma à cause de la raideur du terme en $T^4$.
2. **Implicite complet (Newton, section 4.1)** : inconditionnellement stable, mais demande le passage résidu + jacobienne.

> **Remarque — stratégie pédagogique de test.** Avant même d'introduire $T^4$, faire remplacer temporairement le rayonnement par un coefficient de perte linéaire artificiel supplémentaire (donc rester dans le cas §2) permet de valider indépendamment : (a) que la frontière $\Gamma_N$ ciblée est correcte, (b) que le test unitaire "T=0 → contribution nulle" passe, avant d'ajouter la complexité de la non-linéarité elle-même. C'est le principe : isoler la nouveauté "nouvelle frontière physique" de la nouveauté "solveur non linéaire" plutôt que de les débugger ensemble.

> **Remarque — mise à l'échelle / conditionnement.** $\varepsilon\sigma T^4$ et $k\nabla T\cdot\nabla v$ vivent à des échelles très différentes selon les unités choisies ($\sigma=5.67\times10^{-8}$). Si les résidus du Newton stagnent ou divergent, la première piste n'est pas forcément un bug de jacobienne mais un problème de conditionnement/échelle — à vérifier avant de suspecter la physique.

---

## Table de correspondance rapide

| Symbole physique | Variable code | Section |
|---|---|---|
| $\rho c_p$ | `thermal_capacity` | §1, §2 |
| $k$ | `diffusion_coeff` | §1, §2 |
| $h$ | `advection_coeff` (mal nommé — pas une advection) | §1.3 |
| $f(r,z,t)$ | `TimeGatedGaussian` / `self.f` | §1.2 |
| $\Gamma_D$ | `boundary_bottom`, `self.bcs` | §1.3 |
| $\Gamma_N$ | `ufl.ds` (frontière entière, moins $\Gamma_D$ écrasée) | §1.3, §3 |
| $a(\cdot,\cdot)$, $L(\cdot)$ | `a_ufl`, `L_ufl` | §2 |
| $F(T;v)$ (futur) | à créer, remplace `a_ufl - L_ufl` | §4 |
| $J$ (futur) | `ufl.derivative(F, T, dT)` | §4.1 |
