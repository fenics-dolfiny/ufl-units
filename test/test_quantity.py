import pytest
import sympy as sy
import sympy.physics.units as syu
from ufl_units import Quantity, get_factor, to_base_units


def test_str_and_symbol(mesh):
    """Test that Quantity objects correctly store and display their symbol."""
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    assert str(length) == "L"
    assert length.symbol == L_symbol


def test_factor(mesh):
    """The factor is the scale converted to the base units of the unit system."""
    length = Quantity(mesh, 2.0, unit=syu.kilometer, symbol="L")
    assert length.factor == pytest.approx(2000.0)
    assert length.scale == 2.0


def test_dimensionless(mesh):
    """A quantity without unit is dimensionless and has a unit factor."""
    alpha = Quantity(mesh, 10.0, unit=None, symbol="alpha")
    assert alpha.factor == pytest.approx(10.0)
    assert syu.si.SI.get_dimension_system().equivalent_dims(alpha.dimension, sy.S(1))


@pytest.mark.parametrize(
    ("scale", "unit", "symbol", "error"),
    [
        ([1.0, 2.0], syu.meter, "L", TypeError),
        (1.0, "meter", "L", TypeError),
        (1.0, syu.meter, 1.0, TypeError),
    ],
)
def test_invalid_args(mesh, scale, unit, symbol, error):
    """A non-scalar scale, a non-sympy unit and a non-symbol are all rejected."""
    with pytest.raises(error):
        Quantity(mesh, scale, unit=unit, symbol=symbol)


def test_scale_setter(mesh):
    """Test that Quantity scale can be updated correctly."""
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.kilometer, symbol=L_symbol)
    original_scale = length.scale
    # Set a new scale
    length.scale = 2.0
    assert length.scale == 2.0
    # The value should be updated accordingly
    assert length.scale != original_scale
    assert length.factor == pytest.approx(2000.0)


def test_unit_and_expr(mesh):
    """The unit is kept as given, while expr is the scale applied to it."""
    length = Quantity(mesh, 2.0, unit=syu.kilometer, symbol="L")
    assert length.unit == syu.kilometer
    assert length.expr == 2.0 * syu.kilometer


def test_repr(mesh):
    """The repr names the class and reports all four defining arguments."""
    length = Quantity(mesh, 2.0, unit=syu.meter, symbol="L")
    text = repr(length)
    assert text.startswith("Quantity(")
    assert "scale=2.0" in text
    assert "unit=meter" in text
    assert "symbol=L" in text
    assert "unit_system=SI" in text


def test_to_base_units():
    """Conversion rewrites a compound unit in the base units of the system."""
    converted = to_base_units(2.0 * syu.kilometer / syu.hour)
    coefficient = converted / (syu.meter / syu.second)
    assert float(coefficient) == pytest.approx(2000.0 / 3600.0)


def test_get_factor_unconvertible():
    """A unit outside the system's base units cannot be reduced to a number."""
    time_only = syu.UnitSystem(
        base_units=[syu.second], dimension_system=syu.si.SI.get_dimension_system()
    )
    with pytest.raises(ValueError, match="Cannot convert"):
        get_factor(1.0, syu.meter, time_only)


def test_custom_unit_system(mesh, structural):
    """A quantity reduces to the base units of the system it is given, not to SI."""
    unit_system, GPa = structural

    mu = Quantity(mesh, 100, GPa, "mu", unit_system)

    # 100 GPa is 100 where GPa is the base pressure unit, rather than 1e11 Pa
    assert mu.scale == 100
    assert mu.factor == pytest.approx(100.0)
    assert mu.unit_system is unit_system
    assert mu.dimension == syu.pressure


def test_custom_unit_system_mixed_units(mesh, structural):
    """Units of the system that are not base units are still reduced correctly."""
    unit_system, _ = structural

    # A stress given in Pa, in a system whose base pressure unit is GPa
    sigma = Quantity(mesh, 2e9, syu.pascal, "sigma", unit_system)
    assert sigma.factor == pytest.approx(2.0)

    length = Quantity(mesh, 5.0, syu.millimeter, "l", unit_system)
    assert length.factor == pytest.approx(0.005)


def test_custom_unit_system_scale_setter(mesh, structural):
    """The scale setter re-reduces in the quantity's own unit system."""
    unit_system, GPa = structural
    mu = Quantity(mesh, 100, GPa, "mu", unit_system)

    mu.scale = 250
    assert mu.factor == pytest.approx(250.0)
