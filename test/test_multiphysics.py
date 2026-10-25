"""Coupled multiphysics problems normalized with ufl-units and solved with dolfinx."""

import basix.ufl
import ufl

import numpy as np
import pytest
import sympy.physics.units as syu
from ufl_units import collect_quantities, expand, factorize, normalize

# dolfinx is not part of the `test` dependency group; it only exists in the backend images.
dolfinx = pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx.fem.petsc")
MPI = pytest.importorskip("mpi4py").MPI
Quantity = pytest.importorskip("ufl_units.backends.dolfinx").Quantity


def test_natural_convection():
    """Differentially heated cavity: Navier-Stokes coupled to heat transport (Boussinesq).

    Each residual block is normalized by its own reference term. The Rayleigh and Prandtl
    numbers emerge as coefficients, and the average Nusselt number matches the benchmark of
    de Vahl Davis (1983), Nu = 2.243 at Ra = 1e4 and Pr = 0.71.
    """
    Ra, Pr = 1e4, 0.71

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 2, shape=(2,)))
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 1))
    S = dolfinx.fem.functionspace(mesh, basix.ufl.element("P", mesh.basix_cell(), 2))
    W = ufl.MixedFunctionSpace(V, Q, S)

    v, p, T = dolfinx.fem.Function(V), dolfinx.fem.Function(Q), dolfinx.fem.Function(S)
    δv, δp, δT = ufl.TestFunctions(W)
    e_y = dolfinx.fem.Constant(mesh, np.array([0.0, 1.0]))

    # Air in a 5 cm cavity, the temperature difference chosen to give the Rayleigh number
    L, nu_, g_, beta_ = 0.05, 1.5e-5, 9.81, 1 / 300
    alpha_ = nu_ / Pr
    rho = Quantity(mesh, 1.2, syu.kilogram / syu.meter**3, "rho")
    nu = Quantity(mesh, nu_, syu.meter**2 / syu.second, "nu")
    alpha = Quantity(mesh, alpha_, syu.meter**2 / syu.second, "alpha")
    g = Quantity(mesh, g_, syu.meter / syu.second**2, "g")
    beta = Quantity(mesh, beta_, 1 / syu.kelvin, "beta")
    l_ref = Quantity(mesh, L, syu.meter, "l_ref")
    dT = Quantity(mesh, Ra * nu_ * alpha_ / (g_ * beta_ * L**3), syu.kelvin, "dT")
    v_ref = Quantity(mesh, alpha_ / L, syu.meter / syu.second, "v_ref")
    p_ref = Quantity(mesh, 1.2 * nu_ * alpha_ / L**2, syu.pascal, "p_ref")

    blocks = {
        "momentum": {
            "convection": ufl.inner(rho * ufl.dot(ufl.grad(v), v), δv) * ufl.dx,
            "viscous": ufl.inner(rho * nu * ufl.grad(v), ufl.grad(δv)) * ufl.dx,
            "pressure": -p * ufl.div(δv) * ufl.dx,
            "buoyancy": -rho * beta * g * T * ufl.inner(e_y, δv) * ufl.dx,
        },
        "mass": {"incompressibility": ufl.div(v) * δp * ufl.dx},
        "energy": {
            "convection": ufl.dot(v, ufl.grad(T)) * δT * ufl.dx,
            "conduction": ufl.inner(alpha * ufl.grad(T), ufl.grad(δT)) * ufl.dx,
        },
    }
    references = {"momentum": "viscous", "mass": "incompressibility", "energy": "conduction"}
    # T is the temperature above that of the cold wall, in units of the wall difference
    mapping = {mesh.ufl_domain(): l_ref, v: v_ref * v, p: p_ref * p, T: dT * T}

    quantities = collect_quantities(
        sum((sum(b.values(), ufl.form.Zero()) for b in blocks.values()), ufl.form.Zero()), mapping
    )
    factorized = {key: factorize(b, quantities, mapping=mapping) for key, b in blocks.items()}

    momentum = factorized["momentum"]
    for term, expected in [("convection", 1 / Pr), ("buoyancy", Ra)]:
        ratio = momentum[term].factor - momentum["viscous"].factor
        assert float(expand(ratio, [q.factor for q in quantities])) == pytest.approx(expected)

    F = sum(
        (sum(normalize(factorized[k], r, quantities).values()) for k, r in references.items()),
        ufl.form.Zero(),
    )
    residual = ufl.extract_blocks(F)

    # The zero pressure-pressure block gives the pinned pressure its diagonal entry
    trial = ufl.TrialFunctions(W)
    J = [[ufl.derivative(Fi, w, dw) for w, dw in zip([v, p, T], trial)] for Fi in residual]
    J[1][1] = dolfinx.fem.Constant(mesh, 0.0) * trial[1] * δp * ufl.dx

    # No-slip walls, hot wall at x = 0, cold wall at x = 1, adiabatic top and bottom,
    # and the pressure pinned in one corner
    walls = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.full(x.shape[1], True))

    def dofs(space, marker):
        return dolfinx.fem.locate_dofs_geometrical(space, marker)

    bcs = [
        dolfinx.fem.dirichletbc(np.zeros(2), dolfinx.fem.locate_dofs_topological(V, 1, walls), V),
        dolfinx.fem.dirichletbc(1.0, dofs(S, lambda x: np.isclose(x[0], 0.0)), S),
        dolfinx.fem.dirichletbc(0.0, dofs(S, lambda x: np.isclose(x[0], 1.0)), S),
        dolfinx.fem.dirichletbc(0.0, dofs(Q, lambda x: np.isclose(x[0] + x[1], 0.0)), Q),
    ]

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual,
        [v, p, T],
        J=J,
        bcs=bcs,
        kind="mpi",
        petsc_options_prefix="natural_convection_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    problem.solve()
    assert problem.solver.getConvergedReason() > 0

    # Average horizontal heat flux through the cavity, in units of pure conduction
    nusselt = dolfinx.fem.form((v[0] * T - T.dx(0)) * ufl.dx)
    Nu = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(nusselt), op=MPI.SUM)
    assert Nu == pytest.approx(2.243, rel=5e-3)
