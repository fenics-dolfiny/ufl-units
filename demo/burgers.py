# # Burgers equation, with units
#
# The Burgers equation is a non-linear equation for the advection and diffusion of momentum.
# Here we choose to write the Burgers equation in two dimensions to demonstrate the use of
# vector function spaces:
# $$
#    \frac{\partial u}{\partial t} + (u\cdot\nabla) u - \nu\nabla^2 u = 0
#
#    (n\cdot \nabla) u = 0 \ \textrm{on}\ \Gamma
# $$
# where $\Gamma$ is the domain boundary and $\nu$ is a constant scalar viscosity. The
# solution $u$ is sought in some suitable vector-valued function space $V$. We take the inner
# product with an arbitrary test function $v\in V$ and integrate the viscosity term by parts:
# $$
#    \int_\Omega\frac{\partial u}{\partial t}\cdot v +
#    ((u\cdot\nabla) u)\cdot v + \nu\nabla u\cdot\nabla v \ \mathrm d x = 0.
# $$
# The boundary condition has been used to discard the surface integral. Next, we need to
# discretise in time. For simplicity and stability we elect to use a backward Euler
# discretisation:
# $$
#    \int_\Omega\frac{u^{n+1}-u^n}{dt}\cdot v +
#    ((u^{n+1}\cdot\nabla) u^{n+1})\cdot v + \nu\nabla u^{n+1}\cdot\nabla v \ \mathrm d x = 0.
# $$
#
# It is the Firedrake demo of the same name
# (https://www.firedrakeproject.org/demos/burgers.py.html), with physical units attached by
# `ufl_units`, step by step as in the guide `doc/adding-units-firedrake.md`. The original is
# dimensional, but its units are implicit. Here the domain is a square of side
# $1\,\mathrm{m}$, the velocity is in $\mathrm{m}/\mathrm{s}$ and time is in seconds.
#
# We can now proceed to set up the problem. We choose a resolution and set up a square mesh.
# The tables of the dimensional analysis are printed through the logger of `ufl_units`:

# %%
from logging import INFO, basicConfig, getLogger

import sympy.physics.units as syu
import ufl_units
from firedrake import *
from ufl_units.backends.firedrake import Quantity

basicConfig(format="%(message)s")
getLogger("ufl_units").setLevel(INFO)

n = 30
mesh = UnitSquareMesh(n, n)

# %% [markdown]
# We choose degree 2 continuous Lagrange polynomials. We also need a piecewise linear space
# for output purposes:

# %%
V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

# %% [markdown]
# We also need solution functions for the current and the next timestep. Note that, since
# this is a nonlinear problem, we don't define trial functions:

# %%
u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")

v = TestFunction(V)

# %% [markdown]
# For this problem we need an initial condition:

# %%
x = SpatialCoordinate(mesh)
ic = project(as_vector([sin(pi * x[0]), 0]), V)

# %% [markdown]
# We start with current value of u set to the initial condition, but we also use the initial
# condition as our starting guess for the next value of u:

# %%
u_.assign(ic)
u.assign(ic)

# %% [markdown]
# $\nu$ is set to a (fairly arbitrary) small constant value. It is a quantity: a Firedrake
# constant which also carries a unit and a symbol. Its value is the one of the original,
# $10^{-4}$, read in $\mathrm{m}^2/\mathrm{s}$:

# %%
nu = Quantity(0.0001, syu.meter**2 / syu.second, "nu")

# %% [markdown]
# The timestep is set to produce an advective Courant number of around 1. Since we are
# employing backward Euler, this is stricter than is required for stability, but ensures good
# temporal resolution of the system's evolution:

# %%
timestep = Quantity(1.0 / n, syu.second, "dt")

# %% [markdown]
# The mesh and the velocity fields hold numbers, so their units are quantities as well: the
# mesh is in units of $l_\text{ref} = 1\,\mathrm{m}$, and the velocity is in units of
# $u_\text{ref} = 1\,\mathrm{m}/\mathrm{s}$, the amplitude of the initial condition.

# %%
l_ref = Quantity(1.0, syu.meter, "l_ref")
u_ref = Quantity(1.0, syu.meter / syu.second, "u_ref")

# %% [markdown]
# Here we finally get to define the residual of the equation. In the advection term we need
# to contract the test function $v$ with $(u\cdot\nabla)u$, which is the derivative of the
# velocity in the direction $u$. This directional derivative can be written as
# `dot(u,nabla_grad(u))` since `nabla_grad(u)[i,j]`$=\partial_i u_j$. Note once again that
# for a nonlinear problem, there are no trial functions in the formulation. These will be
# created automatically when the residual is differentiated by the nonlinear solver. The
# residual is kept as a dictionary of its terms, so that each can be analysed on its own:

# %%
terms = {
    "time": inner((u - u_) / timestep, v) * dx,
    "advection": inner(dot(u, nabla_grad(u)), v) * dx,
    "viscous": nu * inner(grad(u), grad(v)) * dx,
}

# %% [markdown]
# ## Dimensional analysis
#
# The mapping gives the units of the mesh, of the velocities, and of the test function,
# which tests the momentum equation and so is scaled as a velocity. The analysis finds two
# dimensionless groups, the inverse Reynolds number
# $\nu / (l_\text{ref} u_\text{ref}) = 10^{-4}$ and the timestep
# $\Delta t \, u_\text{ref} / l_\text{ref}$, checks that all terms have the units of an
# acceleration times a velocity times an area, and normalizes them by the advection term.

# %%
mapping = {mesh: l_ref, u: u_ref * u, u_: u_ref * u_, v: u_ref * v}
quantities = [l_ref, u_ref, nu, timestep]
ufl_units.buckingham_pi_analysis(quantities)

dimension = ufl_units.get_dimension(sum(terms.values(), Form([])), quantities, mapping=mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(dimension, syu.length**4 / syu.time**3)

terms = ufl_units.factorize(terms, quantities, mapping=mapping)
terms = ufl_units.normalize(terms, "advection", quantities)

# %% [markdown]
# The normalized terms add up to the residual that is solved. It is the residual of the
# original divided by the scale of the advection term, so with all reference scales equal to
# one in SI units, the solution is the one of the original.

# %%
F = sum(terms.values(), Form([]))

# %% [markdown]
# We now create an object for output visualisation:

# %%
outfile = VTKFile("output/burgers.pvd")

# %% [markdown]
# Output only supports visualisation of linear fields (either P1, or P1DG). In this example
# we project to a linear space by hand. Another option is to let the `VTKFile` object manage
# the decimation. It supports both interpolation to linears (the default) or projection (by
# passing `project_output=True` when creating the `VTKFile`). Outputting data is carried out
# using the `write` method of `VTKFile` objects:

# %%
outfile.write(project(u, V_out, name="Velocity"))

# %% [markdown]
# Finally, we loop over the timesteps solving the equation each time and outputting each
# result. Firedrake's default solver parameters are used, which amount to applying a full LU
# decomposition as a preconditioner. The time `t` is a Python number, so it is advanced by the
# value of the timestep in seconds, `timestep.factor`.

# %%
t = 0.0
end = 0.5
while t <= end:
    solve(F == 0, u)
    u_.assign(u)
    t += timestep.factor
    outfile.write(project(u, V_out, name="Velocity"))
