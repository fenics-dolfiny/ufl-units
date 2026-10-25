# # Dimensional analysis of Neo-Hooke hyperelasticity
#
# The compressible Neo-Hooke strain energy density
# $$
# \tilde W = \frac{\mu}{2} (I_1 - 3 - 2 \ln J) + \frac{\lambda}{2} (J - 1)^2,
# \qquad \mathbf F = \mathbf I + \nabla \mathbf u,
# $$
# depends on the shear modulus $\mu$ and first Lamé parameter $\lambda$, and with the
# external work of a traction $\tau_\text{ref} \mathbf t$ on the scales $l_\text{ref}$,
# $u_\text{ref}$ and $\tau_\text{ref}$.

# %%
import logging

from mpi4py import MPI

import dolfinx
import ufl

import numpy as np
import sympy.physics.units as syu
import ufl_units
from ufl_units.backends.dolfinx import Quantity

logging.basicConfig(level=logging.INFO, format="%(message)s")

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
V = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
u = dolfinx.fem.Function(V, name="u")
t = dolfinx.fem.Constant(mesh, np.array([0.0, 0.0, -1.0]))  # dimensionless traction direction

right = dolfinx.mesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[0], 1.0))
facet_tags = dolfinx.mesh.meshtags(mesh, 2, right, np.full_like(right, 1))
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# %% [markdown]
# ## Reference quantities
#
# Poisson's ratio and Young's modulus only define $\mu$ and $\lambda$, so
# they are not part of the analysis.

# %%
nu = 0.4
E = 2.0  # GPa
GPa = syu.giga * syu.pascal

μ = Quantity(mesh, E / (2 * (1 + nu)), GPa, "μ")
λ = Quantity(mesh, E * nu / ((1 + nu) * (1 - 2 * nu)), GPa, "λ")
l_ref = Quantity(mesh, 1.0, syu.millimeter, "l_ref")
τ_ref = Quantity(mesh, 100, syu.kilo * syu.pascal, "τ_ref")
u_ref = Quantity(mesh, 0.001, syu.millimeter, "u_ref")
quantities = [μ, λ, l_ref, τ_ref, u_ref]

mapping = {mesh.ufl_domain(): l_ref, u: u_ref * u}

# %% [markdown]
# ## Buckingham Pi analysis (Figure 14)
#
# The three groups are $\lambda / \mu = 4$, $\tau_\text{ref} / \mu = 0.00014$ and
# $u_\text{ref} / l_\text{ref} = 0.001$.

# %%
ufl_units.buckingham_pi_analysis(quantities)

# %% [markdown]
# ## Factorization of the standard form
#
# The deformation gradient adds the identity to $\Pi_3 \nabla \mathbf u$, two terms of very
# different magnitude: $\mathbf F = \mathbf I + \Pi_3 \nabla \mathbf u$. Factorization exposes
# this as the dimensionless number $u_\text{ref} / l_\text{ref}$, the source of the loss of
# about $k$ digits for $\Pi_3 = 10^{-k}$.

# %%
F = ufl.Identity(3) + ufl.grad(u)
C = F.T * F
J = ufl.det(F)
W_standard = (μ / 2 * (ufl.tr(C) - 3 - 2 * ufl.ln(J)) + λ / 2 * (J - 1) ** 2) * ufl.dx

standard = ufl_units.factorize(W_standard, quantities, mapping=mapping)
dimless = sorted(
    {str(ufl_units.expand(d, [q.symbol for q in quantities])) for d in standard.dimless_numbers}
)
logging.info(f"Dimensionless numbers inside the standard Neo-Hooke energy: {dimless}")
assert "u_ref/l_ref" in dimless

# %% [markdown]
# ## Expansion in the Green-Lagrange strain (Figure 15)
#
# With $\mathbf E_1 = \operatorname{sym} \nabla \mathbf u$ and
# $\mathbf E_2 = \frac{1}{2} \nabla \mathbf u^\mathsf{T} \nabla \mathbf u$, the energy expanded
# to third order consists of four homogeneous terms. Normalizing by the volumetric second-order term
# gives the coefficients $\mu / \lambda = 0.25$, $\Pi_3 \mu / \lambda = 0.00025$,
# $\Pi_3 = 0.001$ and $l_\text{ref} \tau_\text{ref} / (u_\text{ref} \lambda) = 0.035$.

# %%
H = ufl.grad(u)
E1 = ufl.sym(H)
E2 = 0.5 * H.T * H
tr = ufl.tr

terms = {
    "shear_2": μ * tr(E1 * E1) * ufl.dx,
    "vol_2": λ / 2 * tr(E1) ** 2 * ufl.dx,
    "shear_3": μ * (2 * ufl.inner(E1, E2) - 4 / 3 * tr(E1 * E1 * E1)) * ufl.dx,
    "vol_3": λ * (tr(E1) * tr(E2) + 1 / 2 * tr(E1) ** 3 - tr(E1) * tr(E1 * E1)) * ufl.dx,
    "force": -τ_ref * ufl.inner(t, u) * ds(1),
}

factorized = ufl_units.factorize(terms, quantities, mapping=mapping)
ufl_units.normalize(factorized, "vol_2", quantities)

expected = {"shear_2": 0.25, "vol_2": 1.0, "shear_3": 0.00025, "vol_3": 0.001, "force": 0.035}
for key, value in expected.items():
    ratio = factorized[key].factor - factorized["vol_2"].factor
    coefficient = float(ufl_units.expand(ratio, [q.factor for q in quantities]))
    assert np.isclose(coefficient, value, rtol=1e-3), (key, coefficient)
