# # Dimensional analysis of the Navier-Stokes equations
#
# The demo does not solve the nonlinear lid-driven cavity problem. It normalizes the weak form
# of the transient incompressible Navier-Stokes equations by the convective scale, assembles
# the Jacobian of the normalized residual at the zero state and reports its condition number
# as a function of the Euler number.
#
# On the unit square $\Omega = (0, 1)^2$ find velocity $\mathbf{v}$ and pressure $p$ with
# \begin{align}
#    \rho \left( \partial_t \mathbf{v} + (\mathbf{v} \cdot \nabla)\mathbf{v} \right) -
#    \nabla \cdot \left( 2 \rho \nu D(\mathbf{v}) - p \mathbf{I} \right)
#    &= \rho g_\text{ref} \mathbf{b}, \\
#    \nabla \cdot \mathbf{v} &= 0,
# \end{align}
# where $D(\mathbf{v}) = \text{sym}(\nabla \mathbf{v})$ and $\mathbf{b} = (0, -1)$.

# %%
import argparse
import logging

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import default_scalar_type as scalar

import numpy as np
import sympy.physics.units as syu
import ufl_units
from ufl_units.backends.dolfinx import Quantity

# Show the analysis tables of ufl_units, but not the INFO messages of the form compiler
logging.basicConfig(format="%(message)s")
logging.getLogger("ufl_units").setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Navier-Stokes dimensional analysis")
parser.add_argument("--eu-min", type=float, default=0.1, help="Smallest Euler number")
parser.add_argument("--eu-max", type=float, default=2.0, help="Largest Euler number")
parser.add_argument("--num-eu", type=int, default=50, help="Number of Euler numbers")
parser.add_argument(
    "--output", default="navier_stokes_condition_number.txt", help="File for the results"
)
args = parser.parse_args()

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError("Dense condition numbers need a single MPI rank.")

mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
Vf = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 2, shape=(2,)))
Pf = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1))

v = dolfinx.fem.Function(Vf, name="v")
v0 = dolfinx.fem.Function(Vf, name="v0")  # velocity at the previous time level
p = dolfinx.fem.Function(Pf, name="p")
W = ufl.MixedFunctionSpace(Vf, Pf)
δv, δp = ufl.TestFunctions(W)
dv, dp = ufl.TrialFunctions(W)

b = dolfinx.fem.Constant(mesh, scalar((0.0, -1.0)))  # dimensionless body-force direction
num_steps_per_t_ref = dolfinx.fem.Constant(mesh, scalar(1))  # dt = t_ref / num_steps_per_t_ref

# %% [markdown]
# ## Reference quantities
#
# | Symbol | Value | Meaning |
# | --- | --- | --- |
# | $v_\text{ref}$ | $1\,\mathrm{m}/\mathrm{s}$ | reference velocity |
# | $l_\text{ref}$ | $1\,\mathrm{m}$ | reference length |
# | $\rho$ | $5000\,\mathrm{kg}/\mathrm{m}^3$ | density |
# | $\nu$ | $1000\,\mathrm{mm}^2/\mathrm{s}$ | kinematic viscosity |
# | $g_\text{ref}$ | $10\,\mathrm{m}/\mathrm{s}^2$ | gravity scale |
# | $p_\text{ref}$ | $5000\,\mathrm{Pa}$, then swept | reference pressure |
# | $t_\text{ref}$ | $\frac{1}{60}\,\mathrm{min} = 1\,\mathrm{s}$ | reference time |

# %%
# `collect_quantities` returns the quantities in creation order, which fixes the column
# order of the dimension matrix and thus the Pi groups found. This order yields the
# classical Reynolds, Froude, Euler and Strouhal numbers.
v_ref = Quantity(mesh, 1, syu.meter / syu.second, "v_ref")
l_ref = Quantity(mesh, 1, syu.meter, "l_ref")
rho = Quantity(mesh, 5000, syu.kilogram / syu.m**3, "rho")
nu = Quantity(mesh, 1000, syu.millimeter**2 / syu.second, "nu")
g_ref = Quantity(mesh, 10, syu.meter / syu.second**2, "g_ref")
p_ref = Quantity(mesh, 5000.0, syu.pascal, "p_ref")
t_ref = Quantity(mesh, 1 / 60, syu.minute, "t_ref")

# %% [markdown]
# ## Weak form
#
# The residual is written term by term so that each term is factorized on its own. The
# `pressure_bc` term is identically zero: it only makes dolfinx allocate the pressure-pressure
# block, whose diagonal entry is set by the pressure pinning condition. The factor
# $1 / (p_\text{ref} t_\text{ref})$ is a dimensional placeholder inside this zero term.
#
# The mapping scales coordinates by $l_\text{ref}$, velocities by $v_\text{ref}$ and pressures
# by $p_\text{ref}$.


# %%
def D(u):
    """Strain rate tensor."""
    return ufl.sym(ufl.grad(u))


terms = {
    "unsteady": ufl.inner(δv, rho * (v - v0) / (t_ref / num_steps_per_t_ref)) * ufl.dx,
    "convection": ufl.inner(δv, rho * ufl.dot(ufl.grad(v), v)) * ufl.dx,
    "viscous": ufl.inner(D(δv), 2 * rho * nu * D(v)) * ufl.dx,
    "pressure": -ufl.inner(ufl.div(δv), p) * ufl.dx,
    "force": -ufl.inner(δv, rho * g_ref * b) * ufl.dx,
    "incompressibility": -δp * ufl.div(v) * ufl.dx,
    "pressure_bc": -dolfinx.fem.Constant(mesh, scalar(0.0))
    / p_ref
    / t_ref
    * ufl.inner(δp, p)
    * ufl.dx,
}

mapping = {
    mesh.ufl_domain(): l_ref,
    v: v_ref * v,
    v0: v_ref * v0,
    p: p_ref * p,
    δv: v_ref * δv,
    δp: p_ref * δp,
}

# %% [markdown]
# ## Dimensional analysis
#
# The quantities are collected from the form, the Buckingham Pi analysis reports four Pi
# groups, and the whole form must have the dimension of power per length. Normalizing by the
# convection term yields the Reynolds, Euler, Froude and Strouhal numbers
# \begin{align}
#    \mathrm{Re} &= \frac{v_\text{ref} l_\text{ref}}{\nu}, &
#    \mathrm{Eu} &= \frac{p_\text{ref}}{\rho v_\text{ref}^2}, &
#    \mathrm{Fr}^2 &= \frac{v_\text{ref}^2}{g_\text{ref} l_\text{ref}}, &
#    \mathrm{St} &= \frac{l_\text{ref}}{v_\text{ref} t_\text{ref}}.
# \end{align}

# %%
form = sum(terms.values(), ufl.form.Zero())

quantities = ufl_units.collect_quantities(form, mapping=mapping)
assert len(quantities) == 7
ufl_units.buckingham_pi_analysis(quantities)

dimsys = syu.si.SI.get_dimension_system()
assert dimsys.equivalent_dims(ufl_units.get_dimension(D(v), quantities, mapping), 1 / syu.time)
form_dim = ufl_units.get_dimension(form, quantities, mapping)
assert dimsys.equivalent_dims(form_dim, syu.power / syu.length)

terms_fact = ufl_units.factorize(terms, quantities, mapping=mapping)
terms_norm = ufl_units.normalize(terms_fact, "convection", quantities)
form_nondimensional = sum(terms_norm.values(), ufl.form.Zero())

# %% [markdown]
# ## Lid-driven cavity boundary conditions
#
# No-slip on the left, right and bottom walls, unit tangential velocity on the lid, and the
# pressure pinned at one point to remove the constant nullspace. The point $(0, 0.9)$ is the
# one pinned for the manuscript; the condition number for small Euler numbers depends on it.

# %%
facets = dolfinx.mesh.locate_entities_boundary(
    mesh, 1, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)
)
bc0 = dolfinx.fem.dirichletbc(
    np.zeros(2, dtype=scalar), dolfinx.fem.locate_dofs_topological(Vf, 1, facets), Vf
)

lid_velocity = dolfinx.fem.Function(Vf)
lid_velocity.interpolate(lambda x: np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))
facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 1.0))
bc1 = dolfinx.fem.dirichletbc(lid_velocity, dolfinx.fem.locate_dofs_topological(Vf, 1, facets))

dof0 = dolfinx.fem.locate_dofs_geometrical(
    Pf, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.9)
)
bc2 = dolfinx.fem.dirichletbc(scalar(0.0), dof0, Pf)

bcs = [bc0, bc1, bc2]

# %% [markdown]
# ## Matrix conditioning under pressure scaling
#
# The Jacobian of the normalized residual is assembled at the zero state
# $(\mathbf{v}, p) = (\mathbf{0}, 0)$ into one monolithic PETSc matrix and densified for
# `numpy.linalg.cond`, the spectral condition number $\kappa_2$. The Euler number is swept by
# changing $p_\text{ref}$; the compiled form reads the new value of the constant.

# %%
jacobian = ufl.derivative(form_nondimensional, v, dv) + ufl.derivative(form_nondimensional, p, dp)
J = dolfinx.fem.form(ufl.extract_blocks(jacobian))


def condition_number() -> float:
    A = dolfinx.fem.petsc.assemble_matrix(J, bcs=bcs)
    A.assemble()
    cond = np.linalg.cond(A.convert("dense").getDenseArray())
    A.destroy()
    return float(cond)


euler_values = np.linspace(args.eu_min, args.eu_max, args.num_eu)
condition_numbers = np.empty_like(euler_values)

for i, eu in enumerate(euler_values):
    p_ref.scale = float(eu * rho.factor * v_ref.factor**2)
    condition_numbers[i] = condition_number()
    print(f"Eu = {eu:.4f}, kappa = {condition_numbers[i]:.6g}")

np.savetxt(
    args.output,
    np.column_stack((euler_values, condition_numbers)),
    header="Euler number, Condition number",
    fmt="%g",
)
