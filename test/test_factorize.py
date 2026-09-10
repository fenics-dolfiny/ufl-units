import ufl

import pytest
import sympy as sy
import sympy.physics.units as syu
from ufl_units import (
    FactorizedExpr,
    Quantity,
    QuantityFactorizer,
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
    with pytest.raises(RuntimeError, match="Inconsistent dimensions"):
        factorize(expr, [length, time], mode="check")


def test_invalid_mode(mesh):
    """Only "factorize" and "check" are accepted modes."""
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol="ell")
    with pytest.raises(RuntimeError, match="not a valid factorisation mode"):
        factorize(ell, [ell], mode="nonsense")


def test_sin_nontrivial_arg(mesh):
    """No scale can be pulled out of sin(k*ell), even though its argument is dimensionless."""
    k = Quantity(mesh, 1.0, unit=1 / syu.meter, symbol=sy.Symbol("k"))
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    with pytest.raises(RuntimeError):
        factorize(ufl.sin(k * ell), [k, ell], mode="factorize")


def test_sin_nontrivial_arg_check(mesh):
    """The same expression should still be admissible in dimensional check mode."""
    k = Quantity(mesh, 1.0, unit=1 / syu.meter, symbol=sy.Symbol("k"))
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    checked = factorize(ufl.sin(k * ell), [k, ell], mode="check")
    assert checked.factor is not None


def test_sin_trivial_arg(mesh):
    """sin(ell/ell) has a zero exponent vector and is factorizable."""
    ell = Quantity(mesh, 1.0, unit=syu.meter, symbol=sy.Symbol("ell"))

    factorized = factorize(ufl.sin(ell / ell), [ell], mode="factorize")
    assert factorized.factor is not None
    assert factorized.factor[0] == pytest.approx(0.0)


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

    factorized = factorize(ufl.variable(length * u), [length])
    assert factorized.factor[0] == pytest.approx(1.0)


def test_nonscalar_quantity(mesh):
    """Factorizing a non-scalar quantity is not implemented."""
    tensor = Quantity(mesh, 1.0, syu.meter, "S", shape=(2,))

    with pytest.raises(NotImplementedError, match="non-scalar quantities"):
        factorize(ufl.sqrt(tensor[0] * tensor[0]), [tensor])


def test_factorizer_invalid_mode(mesh):
    """The factorizer rejects an unknown mode on construction."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    with pytest.raises(ValueError, match="Invalid mode"):
        QuantityFactorizer([length], mode="nonsense")


def test_invalid_type(mesh):
    """Only expressions, forms and dictionaries of those can be factorized."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    with pytest.raises(TypeError, match="Unsupported type for factorization"):
        factorize("not an expression", [length])


def test_form_inconsistent_integral_dimensions(mesh, V):
    """Integrals of one form must agree dimensionally."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    form = u * v * ufl.dx(1) + length * u * v * ufl.dx(2)
    with pytest.raises(RuntimeError, match="Inconsistent dimensions across integrals"):
        factorize(form, [length], mode="check")


def test_form_inconsistent_integral_factors(mesh, V):
    """Integrals may share a dimension and still carry different factors."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    l_one = Quantity(mesh, 1.0, syu.meter, "L1")
    l_two = Quantity(mesh, 2.0, syu.meter, "L2")

    form = l_one * u * v * ufl.dx(1) + l_two * u * v * ufl.dx(2)

    # Dimensionally consistent, so check mode accepts it
    factorize(form, [l_one, l_two], mode="check")

    with pytest.raises(RuntimeError, match="Inconsistent factors across integrals"):
        factorize(form, [l_one, l_two], mode="factorize")


def test_normalize_none_factor(mesh):
    """An expression whose factor could not be determined cannot be normalized."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    unfactorized = {"reference": FactorizedExpr(length * length, None)}

    with pytest.raises(ValueError, match="has a None factor"):
        normalize(unfactorized, "reference", [length])
