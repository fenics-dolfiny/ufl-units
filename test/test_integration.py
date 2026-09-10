"""End-to-end use of the package, exercising all modules together."""

import ufl

import sympy.physics.units as syu
import ufl_units
from ufl_units import Quantity, factorize, get_dimension, normalize


def test_poisson(mesh, V) -> None:
    """
    +--------------------------------------------+
    | Poisson dimensional demo                   |
    +--------------------------------------------+

    PDE
    ---

        -∇ · (κ ∇u) = f

    Units
    -----

        [x]     = l_ref = L
        [u]     = u_ref = Θ
        [κ]     = κ_ref = W / (m K)
        [f]     = f_ref = W / m³

    Dimensional relation
    --------------------

        [f] = [κ] [u] / [x]²

    or equivalently

        f_ref = κ_ref u_ref / l_ref²

    Dimensionless group
    -------------------

        Π = κ_ref u_ref / (f_ref l_ref²)

    or its inverse, depending on normalization.
    """
    T = ufl.Coefficient(V)
    f = ufl.Coefficient(V)
    v = ufl.TestFunction(V)

    kappa = Quantity(mesh, 1.0, syu.W / (syu.K * syu.m), "kappa")
    l_ref = Quantity(mesh, 1.0, syu.m, "l_ref")
    T_ref = Quantity(mesh, 1.0, syu.K, "T_ref")
    f_ref = Quantity(mesh, 1.0, syu.W / syu.m**3, "f_ref")

    terms = {
        "source": f * v * ufl.dx,
        "diss": ufl.inner(kappa * ufl.grad(T), ufl.grad(v)) * ufl.dx,
    }
    mapping = {
        mesh: l_ref,
        T: T_ref * T,
        f: f_ref * f,
        v: T_ref * v,
    }

    quantities = ufl_units.collect_quantities(sum(terms.values(), ufl.form.Zero()), mapping=mapping)
    assert set(quantities) == {T_ref, f_ref, l_ref, kappa}

    _, _, pi_groups = ufl_units.buckingham_pi_analysis(quantities)
    assert len(pi_groups) == 1

    # Dimensional consistency using mapping
    diffusion_dim = get_dimension(terms["diss"], quantities, mapping=mapping)
    rhs_dim = get_dimension(terms["source"], quantities, mapping=mapping)
    assert syu.si.SI.get_dimension_system().equivalent_dims(diffusion_dim, rhs_dim)

    factorized = factorize(terms, quantities, mode="factorize", mapping=mapping)
    normalize(factorized, "source", quantities)
