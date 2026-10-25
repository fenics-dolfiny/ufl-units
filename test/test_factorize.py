import fractions

import ufl
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives

import numpy as np
import pytest
import sympy as sy
import sympy.physics.units as syu
from ufl_units import (
    FactorizedExpr,
    InconsistentDimensions,
    Quantity,
    collect_quantities,
    expand,
    factorize,
    get_dimension,
    normalize,
)


def test_expand(mesh):
    """An exponent vector expands to the product of the bases raised to those powers."""
    a, b = sy.Symbol("a"), sy.Symbol("b")
    assert expand([2, -1], [a, b]) == a**2 / b
    assert expand([0.5, 0], [a, b]) == sy.sqrt(a)
    assert expand([], []) == 1


def test_get_dimension(mesh):
    """A velocity built from a length over a time has dimension length/time."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    time = Quantity(mesh, 1.0, syu.second, "T")

    dimsys = syu.si.SI.get_dimension_system()
    velocity_dim = get_dimension(length / time, [length, time])
    assert dimsys.equivalent_dims(velocity_dim, syu.length / syu.time)


def test_consistent_units(mesh):
    """Test that factorization works correctly for expressions with consistent units."""
    L_symbol = sy.Symbol("L")
    T_symbol = sy.Symbol("T")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T_symbol)
    expr = length * time
    result = factorize(expr, [length, time])
    # The factorized expression should be 1 (dimensionless constants)
    # The factor should represent the unit powers
    assert result.factor[0] == 1  # length factor
    assert result.factor[1] == 1  # time factor


def test_inconsistent_units(mesh):
    """Test that factorization raises an error when trying to add incompatible units."""
    L_symbol = sy.Symbol("L")
    T_symbol = sy.Symbol("T")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T_symbol)
    expr = length + time
    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(expr, [length, time])


def test_sin_nontrivial_arg(mesh):
    """No scale can be pulled out of sin(k*ell), so its argument stays in the expression."""
    k = Quantity(mesh, 1.0, unit=1 / syu.meter, symbol=sy.Symbol("k"))
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    factorized = factorize(ufl.sin(k * ell), [k, ell])
    assert factorized.factor == pytest.approx([0.0, 0.0])
    assert set(collect_quantities(factorized.expr)) == {k, ell}


def test_sin_trivial_arg(mesh):
    """sin(ell/ell) has a zero exponent vector and is factorizable."""
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    factorized = factorize(ufl.sin(ell / ell), [ell])
    assert factorized.factor is not None
    assert factorized.factor[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "function",
    [
        ufl.exp,
        ufl.ln,
        ufl.sin,
        ufl.cos,
        ufl.tan,
        ufl.sinh,
        ufl.cosh,
        ufl.tanh,
        ufl.asin,
        ufl.acos,
        ufl.atan,
        ufl.erf,
        lambda x: ufl.bessel_J(1, x),
    ],
)
def test_nontrivial_dimensionless_arg(mesh, V, function):
    """An Arrhenius-type argument E/(R*T) is kept whole, and reported as a dimensionless group."""
    u = ufl.Coefficient(V)
    energy = Quantity(mesh, 5e4, syu.joule / syu.mol, "E")
    gas = Quantity(mesh, 8.314, syu.joule / (syu.mol * syu.kelvin), "R")
    temp = Quantity(mesh, 300.0, syu.kelvin, "T")

    factorized = factorize(function(energy / (gas * temp * u)), [energy, gas, temp])
    assert factorized.factor == pytest.approx([0.0, 0.0, 0.0])
    assert set(collect_quantities(factorized.expr)) == {energy, gas, temp}
    assert_same_span(factorized.dimless_numbers, [[1.0, -1.0, -1.0]])

    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(function(energy / (gas * u)), [energy, gas, temp])


def test_atan2(mesh):
    """atan2 of two lengths is a dimensionless angle, their ratio stays in the expression."""
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")
    y, x = ufl.Constant(mesh), ufl.Constant(mesh)

    factorized = factorize(ufl.atan2(l_one * y, l_two * x), [l_one, l_two])
    assert factorized.factor == pytest.approx([0.0, 0.0])
    assert_same_span(factorized.dimless_numbers, [[1.0, -1.0]])


def test_atan2_inconsistent(mesh):
    """atan2 of a length and a time is meaningless."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    time = Quantity(mesh, 1.0, syu.second, "T")

    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(ufl.atan2(length * ufl.Constant(mesh), time * ufl.Constant(mesh)), [length, time])


def assert_same_span(dimless_numbers, expected):
    """Compare dimensionless numbers by the subspace they span."""

    def rank(vectors):
        return np.linalg.matrix_rank(np.asarray(vectors, dtype=float)) if len(vectors) else 0

    assert rank(dimless_numbers) == rank(expected) == rank([*dimless_numbers, *expected]), (
        f"{dimless_numbers} and {expected} span different subspaces"
    )


def test_dimless_numbers_of_nonhomogeneous_node(mesh):
    """The argument of sin must be dimensionless, so its factor is a dimensionless group."""
    k = Quantity(mesh, 1.0, unit=1 / syu.meter, symbol=sy.Symbol("k"))
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    checked = factorize(ufl.sin(k * ell), [k, ell])
    assert_same_span(checked.dimless_numbers, [[1.0, 1.0]])


def test_dimless_numbers_of_sum(mesh, V):
    """Two lengths added together are one group, whichever order the operands come in."""
    u = ufl.Coefficient(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    checked = factorize(l_one * u + l_two * u, [l_one, l_two])
    assert_same_span(checked.dimless_numbers, [[1.0, -1.0]])


def test_dimless_numbers_of_powers_and_roots(mesh, V):
    """A group, its inverse, its square and its root all lie on one line."""
    u = ufl.Coefficient(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    expr = (
        (l_one * u + l_two * u)
        * (l_two * l_two * u + l_one * l_one * u)
        * (ufl.sqrt(l_one * u) + ufl.sqrt(l_two * u))
    )

    checked = factorize(expr, [l_one, l_two])
    assert_same_span(checked.dimless_numbers, [[1.0, -1.0]])


def test_dimless_numbers_of_form_integrals(mesh, V):
    """Integrals that share a dimension but not a factor differ by a dimensionless group."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    form = l_one * u * v * ufl.dx(1) + l_two * u * v * ufl.dx(2)

    checked = factorize(form, [l_one, l_two])
    assert_same_span(checked.dimless_numbers, [[1.0, -1.0]])


def test_dimless_numbers_of_form_sum(mesh, V):
    """Components of a FormSum are compared with their weights, and so are their groups."""
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    form_sum = ufl.FormSum((ufl.Cofunction(V.dual()), l_one), (ufl.Cofunction(V.dual()), l_two))

    checked = factorize(form_sum, [l_one, l_two])
    assert_same_span(checked.dimless_numbers, [[1.0, -1.0]])


def test_no_dimless_numbers_of_homogeneous_form(mesh, V):
    """Terms that all carry the same factor differ by no dimensionless number."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    form = length * u * v * ufl.dx + length * ufl.sin(u) * v * ufl.dx
    assert_same_span(factorize(form, [length]).dimless_numbers, [])


def test_no_quantities(mesh, V):
    """With nothing to factorize out, every term trivially agrees and no group is reported."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)

    checked = factorize(u * v * ufl.dx, [])
    assert checked.factor == pytest.approx([])
    assert_same_span(checked.dimless_numbers, [])


@pytest.mark.parametrize(
    ("build", "equivalent"),
    [
        # Reassociation and reordering of a sum
        (
            lambda q, x: (q[0] * x[0] + q[1] * x[1]) + q[2] * x[2],
            lambda q, x: q[0] * x[0] + (q[1] * x[1] + q[2] * x[2]),
        ),
        (
            lambda q, x: (q[0] * x[0] + q[1] * x[1]) + q[2] * x[2],
            lambda q, x: (q[2] * x[2] + q[0] * x[0]) + q[1] * x[1],
        ),
        # Distributivity
        (
            lambda q, x: q[0] * (q[1] * x[0] + q[2] * x[1]),
            lambda q, x: q[0] * q[1] * x[0] + q[0] * q[2] * x[1],
        ),
        # Shared and duplicated subexpressions
        (
            lambda q, x: (q[0] * x[0] + q[1] * x[1]) ** 2,
            lambda q, x: (q[0] * x[0] + q[1] * x[1]) * (q[0] * x[0] + q[1] * x[1]),
        ),
        # Terms UFL simplifies away
        (
            lambda q, x: q[0] * x[0] + 0 * q[1] * x[1],
            lambda q, x: q[0] * x[0] * 1,
        ),
    ],
)
def test_equivalent_expressions(mesh, build, equivalent):
    """Equivalent expressions agree in dimension and in the span of their dimensionless numbers.

    The root factor of a sum follows the operand order, so it is fixed only up to one of
    those numbers. Either way the factorization is exact.
    """
    q = [Quantity(mesh, s, syu.meter, f"L{i}") for i, s in enumerate((2.0, 3.0, 5.0))]
    x = [ufl.Constant(mesh) for _ in range(3)]
    values = {k: ufl.as_ufl(v) for k, v in zip([*q, *x], [2.0, 3.0, 5.0, 0.7, -1.3, 0.4])}
    dimsys = syu.si.SI.get_dimension_system()

    a, b = factorize(build(q, x), q), factorize(equivalent(q, x), q)

    dim_a, dim_b = (expand(f.factor, [qi.dimension for qi in q]).simplify() for f in (a, b))
    assert dimsys.equivalent_dims(dim_a, dim_b)
    assert_same_span(a.dimless_numbers, b.dimless_numbers)
    assert_same_span([*a.dimless_numbers, a.factor - b.factor], a.dimless_numbers)

    for expr, f in ((build(q, x), a), (equivalent(q, x), b)):
        exact = float(ufl.replace(expr, values))
        assert float(ufl.replace(expand(f.factor, q) * f.expr, values)) == pytest.approx(exact)


def test_equivalent_derivatives(mesh, V):
    """A derivative factorizes alike whether or not it is expanded first.

    Expanding reorders the terms, so the root factors agree up to a dimensionless number.
    """
    u, du = ufl.Coefficient(V), ufl.TestFunction(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    derivative = ufl.derivative((l_one * u * u + l_two * u) * ufl.dx, u, du)
    expanded = apply_derivatives(apply_algebra_lowering(derivative))

    a, b = factorize(derivative, [l_one, l_two]), factorize(expanded, [l_one, l_two])
    assert_same_span(a.dimless_numbers, b.dimless_numbers)
    assert_same_span([*a.dimless_numbers, a.factor - b.factor], a.dimless_numbers)


def test_cancellation_is_reported(mesh):
    """UFL does not cancel (a + b) - b, so the number b/a is reported, unlike for a alone."""
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")
    a, b = l_one * ufl.Constant(mesh), l_two * ufl.Constant(mesh)

    assert_same_span(factorize(a, [l_one, l_two]).dimless_numbers, [])
    assert_same_span(factorize((a + b) - b, [l_one, l_two]).dimless_numbers, [[1.0, -1.0]])


def test_normalize_expr(mesh):
    """Expr normalization should use the same factor-ratio sign as Form normalization."""
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    factorized = {
        "reference": factorize(ell, [ell]),
        "higher": factorize(ell**2, [ell]),
    }
    normalized = normalize(factorized, "reference", [ell])

    normalized_higher = factorize(normalized["higher"], [ell])
    assert normalized_higher.factor is not None
    assert normalized_higher.factor[0] == pytest.approx(1.0)


def test_normalize_invalid_input(mesh):
    """Normalization needs factorized expressions and a reference key that exists."""
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    with pytest.raises(TypeError):
        normalize({"reference": ell}, "reference", [ell])

    with pytest.raises(KeyError):
        normalize({"reference": factorize(ell, [ell])}, "missing", [ell])


def test_sqrt(mesh):
    """A square root halves the exponents of the factor."""
    area = Quantity(mesh, 1.0, syu.meter**2, "A")

    factorized = factorize(ufl.sqrt(area), [area])
    assert factorized.factor[0] == pytest.approx(0.5)

    dim = get_dimension(ufl.sqrt(area), [area])
    assert syu.si.SI.get_dimension_system().equivalent_dims(dim, syu.length)


def test_rational_power(mesh):
    """A cube root of a volume is a length, though 1/3 is held as the nearest float."""
    volume = Quantity(mesh, 1.0, syu.meter**3, "V")
    length = Quantity(mesh, 1.0, syu.meter, "L")

    factorized = factorize(volume ** (1 / 3), [volume])
    assert factorized.factor[0] == fractions.Fraction(1 / 3)

    checked = factorize(volume ** (1 / 3) + length, [volume, length])
    assert list(checked.factor) == [0, 1]
    assert [list(g) for g in checked.dimless_numbers] == [[fractions.Fraction(1 / 3), -1]]

    with pytest.raises(InconsistentDimensions):
        factorize(volume ** (1 / 3) + length**2, [volume, length])


def test_plain_constant(mesh):
    """A UFL constant that is not among the quantities is left in place, unscaled."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    c = ufl.Constant(mesh)

    factorized = factorize(c * length, [length])
    assert factorized.factor[0] == pytest.approx(1.0)
    # The quantity is pulled out, the ordinary constant stays in the expression
    assert c in ufl.algorithms.extract_type(factorized.expr, ufl.Constant)


def test_variable(mesh, V):
    """A variable is linear, so it carries the factor of its expression."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    # Not `ufl.variable`, which ty resolves to the submodule of that name.
    factorized = factorize(ufl.operators.variable(length * u), [length])
    assert factorized.factor[0] == pytest.approx(1.0)


def test_nonscalar_quantity(mesh):
    """Factorizing a non-scalar quantity is not implemented."""
    tensor = Quantity(mesh, 1.0, syu.meter, "S", shape=(2,))

    with pytest.raises(NotImplementedError, match="non-scalar quantities"):
        factorize(ufl.sqrt(tensor[0] * tensor[0]), [tensor])


def test_invalid_type(mesh):
    """Only expressions, forms and dictionaries of those can be factorized."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    with pytest.raises(TypeError, match="Unsupported type for factorization"):
        factorize("not an expression", [length])  # ty: ignore[no-matching-overload]


def test_form_inconsistent_integral_dimensions(mesh, V):
    """Integrals of one form must agree dimensionally."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    form = u * v * ufl.dx(1) + length * u * v * ufl.dx(2)
    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions across integrals"):
        factorize(form, [length])


def test_form_inconsistent_integral_factors(mesh, V):
    """Integrals that share a dimension but not a factor are rescaled to the first one."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    form = l_one * u * v * ufl.dx(1) + l_two * u * v * ufl.dx(2)

    factorized = factorize(form, [l_one, l_two])
    assert factorized.factor == pytest.approx([1.0, 0.0])
    assert set(collect_quantities(factorized.expr)) == {l_one, l_two}


def test_interpolate_is_linear(mesh, V):
    """Interpolation keeps the dimension of what is interpolated, and factorizes through."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dimsys = syu.si.SI.get_dimension_system()
    assert dimsys.equivalent_dims(
        get_dimension(ufl.Interpolate(length * u, V), [length]), syu.length
    )

    factorized = factorize(ufl.Interpolate(length * u, V), [length])
    assert factorized.factor == pytest.approx([1.0])
    assert collect_quantities(factorized.expr) == []


def test_interpolate_of_dimensional_expression(mesh, V):
    """The dual argument of an Interpolate is an argument slot, and constrains nothing."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    time = Quantity(mesh, 1.0, syu.second, "T")

    dimsys = syu.si.SI.get_dimension_system()
    dim = get_dimension(ufl.Interpolate(length / time * u, V), [length, time])
    assert dimsys.equivalent_dims(dim, syu.length / syu.time)


@pytest.mark.parametrize(
    "build",
    [
        lambda V: ufl.Cofunction(V.dual()),
        lambda V: ufl.Matrix(V, V),
        lambda V: ufl.ZeroBaseForm(()),
    ],
)
def test_base_form_terminal_is_dimensionless(mesh, V, build):
    """A dual object holds no operands, so nothing can be pulled out of it."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    factorized = factorize(build(V), [length])
    assert factorized.factor == pytest.approx([0.0])


def test_form_sum_weight_is_factorized(mesh, V):
    """A quantity scaling a dual object sits in a weight, and is pulled out of it."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    cofunction = ufl.Cofunction(V.dual())

    factorized = factorize(length * cofunction, [length])

    assert factorized.factor == pytest.approx([1.0])
    assert collect_quantities(factorized.expr) == []


def test_form_sum_inconsistent_components(mesh, V):
    """The components of a sum of base forms must agree dimensionally, as integrals do."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    time = Quantity(mesh, 1.0, syu.second, "T")
    cofunction = ufl.Cofunction(V.dual())

    form_sum = ufl.FormSum((cofunction, length), (ufl.Cofunction(V.dual()), time))
    with pytest.raises(
        InconsistentDimensions, match="Inconsistent dimensions across components of FormSum"
    ):
        factorize(form_sum, [length, time])


def test_normalize_none_factor(mesh):
    """An expression whose factor could not be determined cannot be normalized."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    unfactorized = {"reference": FactorizedExpr(length * length, None)}

    with pytest.raises(ValueError, match="has a None factor"):
        normalize(unfactorized, "reference", [length])


@pytest.mark.parametrize(
    "build",
    [
        lambda q, u: abs(q * u),
        lambda q, u: ufl.as_vector([q * u, q * u])[0],
        lambda q, u: ufl.max_value(q * u, q),
        lambda q, u: ufl.min_value(q * u, q),
        lambda q, u: ufl.real(q * u),
        lambda q, u: ufl.imag(q * u),
        lambda q, u: ufl.cell_avg(q * u),
        lambda q, u: ufl.facet_avg(q * u),
        lambda q, u: (q * u)("+"),
        lambda q, u: (q * u)("-"),
        lambda q, u: ufl.conditional(ufl.gt(q * u, q), q * u, q),
    ],
)
def test_scalar_nodes(mesh, V, build):
    """These nodes pass the dimension of their operands through unchanged."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")

    dimsys = syu.si.SI.get_dimension_system()
    assert dimsys.equivalent_dims(get_dimension(build(u_ref, u), [u_ref]), syu.temperature)


@pytest.mark.parametrize(
    "build",
    [
        lambda q, u: ufl.grad(q * u),
        lambda q, u: ufl.nabla_grad(q * u),
        lambda q, u: ufl.div(q * u),
        lambda q, u: ufl.nabla_div(q * u),
        lambda q, u: ufl.curl(q * u),
        lambda q, u: ufl.skew(ufl.grad(q * u)),
        lambda q, u: ufl.perp(q * u),
    ],
)
def test_vector_nodes(mesh, W, build):
    """Derivatives and shape-preserving tensor algebra keep the dimension of the operand."""
    u = ufl.Coefficient(W)
    u_ref = Quantity(mesh, 1.0, syu.meter / syu.second, "u_ref")

    dimsys = syu.si.SI.get_dimension_system()
    dim = get_dimension(build(u_ref, u), [u_ref])
    assert dimsys.equivalent_dims(dim, syu.length / syu.time)


@pytest.mark.parametrize(
    ("build", "space"),
    [
        (lambda q, u: ufl.classes.ReferenceGrad(q * ufl.classes.ReferenceValue(u)), "V"),
        (lambda q, u: ufl.classes.ReferenceDiv(q * ufl.classes.ReferenceValue(u)), "W"),
        (lambda q, u: ufl.classes.ReferenceCurl(q * ufl.classes.ReferenceValue(u)), "W"),
    ],
)
def test_reference_nodes(mesh, V, W, build, space):
    """A derivative against the reference cell keeps the dimension of its operand."""
    u = ufl.Coefficient({"V": V, "W": W}[space])
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")

    dimsys = syu.si.SI.get_dimension_system()
    assert dimsys.equivalent_dims(get_dimension(build(u_ref, u), [u_ref]), syu.temperature)


def test_tensor_algebra(mesh, W):
    """Outer products and the determinant family combine dimensions multiplicatively."""
    u = ufl.Coefficient(W)
    u_ref = Quantity(mesh, 1.0, syu.meter / syu.second, "u_ref")
    velocity = syu.length / syu.time
    grad_u = ufl.grad(u_ref * u)

    dimsys = syu.si.SI.get_dimension_system()
    for expr, expected in [
        (ufl.outer(u_ref * u, u_ref * u), velocity**2),
        (ufl.det(grad_u), velocity**2),
        (ufl.cofac(grad_u), velocity),
        (ufl.inv(grad_u), 1 / velocity),
    ]:
        assert dimsys.equivalent_dims(get_dimension(expr, [u_ref]), expected)


def test_factorize_tensor_algebra(mesh, W):
    """The scale comes out raised to the power the operation applies to it."""
    u = ufl.Coefficient(W)
    u_ref = Quantity(mesh, 1.0, syu.meter, "u_ref")
    grad_u = ufl.grad(u_ref * u)

    assert factorize(ufl.det(grad_u), [u_ref]).factor[0] == 2
    assert factorize(ufl.cofac(grad_u), [u_ref]).factor[0] == 1
    assert factorize(ufl.inv(grad_u), [u_ref]).factor[0] == -1
    assert factorize((u_ref * u)("+"), [u_ref]).factor[0] == 1


def test_zero(mesh, V):
    """Zero carries no dimension of its own, so it may stand beside a dimensional term."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")

    dimsys = syu.si.SI.get_dimension_system()
    expr = ufl.conditional(ufl.gt(u, 1), u_ref * u, ufl.zero())
    assert dimsys.equivalent_dims(get_dimension(expr, [u_ref]), syu.temperature)


def test_condition_dimensionless(mesh, V):
    """A comparison is a boolean, so it contributes no dimension to the conditional."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dimsys = syu.si.SI.get_dimension_system()
    expr = ufl.conditional(ufl.gt(length * u, length), u_ref, 2 * u_ref)
    assert dimsys.equivalent_dims(get_dimension(expr, [u_ref, length]), syu.temperature)


def test_condition_operands(mesh, V):
    """Only quantities of one dimension can be compared."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")
    length = Quantity(mesh, 1.0, syu.meter, "L")

    expr = ufl.conditional(ufl.gt(u_ref * u, length), u_ref, 2 * u_ref)
    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(expr, [u_ref, length])


def test_conditional_branches(mesh, V):
    """The two branches of a conditional must share a dimension."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")
    length = Quantity(mesh, 1.0, syu.meter, "L")

    expr = ufl.conditional(ufl.gt(u, 1), u_ref * u, length)
    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(expr, [u_ref, length])


def test_list_tensor(mesh, V):
    """The components assembled into a tensor must share a dimension."""
    u = ufl.Coefficient(V)
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")
    length = Quantity(mesh, 1.0, syu.meter, "L")

    with pytest.raises(InconsistentDimensions, match="Inconsistent dimensions"):
        factorize(ufl.as_vector([u_ref * u, length * u]), [u_ref, length])
