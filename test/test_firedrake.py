"""Assembly of a quantity-scaled form with Firedrake as the value backend."""

import ufl

import pytest
import sympy.physics.units as syu
from ufl_units import collect_quantities, expand, factorize, get_dimension, transform

# firedrake is not part of the `test` dependency group; it only exists in the backend images.
firedrake = pytest.importorskip("firedrake")
Quantity = pytest.importorskip("ufl_units.backends.firedrake").Quantity


@pytest.fixture
def unit_square():
    return firedrake.UnitSquareMesh(4, 4)


@pytest.fixture
def flux(unit_square):
    """A conductive flux functional, its unit mapping and the quantities it is built from."""
    V = firedrake.FunctionSpace(unit_square, "Lagrange", 1)
    x = ufl.SpatialCoordinate(unit_square)
    temperature = firedrake.Function(V).interpolate(1.0 + x[0] ** 2)

    kappa = Quantity(1.5, syu.watt / (syu.kelvin * syu.meter), "kappa")
    l_ref = Quantity(0.1, syu.meter, "l_ref")
    T_ref = Quantity(300.0, syu.kelvin, "T_ref")

    form = -kappa * ufl.grad(temperature)[0] * ufl.dx
    # A Firedrake mesh is itself the `ufl.Mesh`, so it keys the mapping directly.
    mapping = {unit_square: l_ref, temperature: T_ref * temperature}
    return form, mapping, [kappa, l_ref, T_ref]


def assemble(form):
    """Assemble a scalar form. Firedrake reduces across all ranks itself."""
    return firedrake.assemble(form)


def test_assemble_quantity(unit_square):
    """Integrating a quantity over the unit square yields its factor in base units."""
    length = Quantity(2.0, syu.kilometer, "L")
    assert length.factor == pytest.approx(2000.0)

    assert assemble(length * ufl.dx(unit_square)) == pytest.approx(2000.0)


def test_assemble_after_scale_change(unit_square):
    """Setting ``scale`` reaches the assembled value through ``_update_value``."""
    length = Quantity(2.0, syu.kilometer, "L")
    form = length * ufl.dx(unit_square)

    length.scale = 3.0
    assert assemble(form) == pytest.approx(3000.0)


def test_factorize_quantity(unit_square):
    """Factorizing pulls the quantity out, leaving the area of the unit square behind."""
    length = Quantity(2.0, syu.kilometer, "L")

    dimensionless, factor = factorize(length * ufl.dx(unit_square), [length])

    assert collect_quantities(dimensionless) == []
    assert float(expand(factor, [length.factor])) == pytest.approx(2000.0)
    assert assemble(dimensionless) == pytest.approx(1.0)


def test_factorize_form(flux):
    """The scale of a flux functional comes out, and rescaling recovers the original."""
    form, mapping, (kappa, l_ref, T_ref) = flux

    quantities = collect_quantities(form, mapping=mapping)
    dimensionless, factor = factorize(form, quantities, mapping=mapping)

    dimsys = syu.si.SI.get_dimension_system()
    assert dimsys.equivalent_dims(get_dimension(form, quantities, mapping=mapping), syu.power)
    assert quantities == [kappa, l_ref, T_ref]
    assert factor == pytest.approx([1.0, 1.0, 1.0])
    assert collect_quantities(dimensionless) == []

    scale = float(expand(factor, [q.factor for q in quantities]))
    assert scale == pytest.approx(kappa.factor * l_ref.factor * T_ref.factor)
    assert scale * assemble(dimensionless) == pytest.approx(assemble(transform(form, mapping)))


def test_factorize_after_scale_change(flux):
    """A scale change moves into the factor and leaves the dimensionless form alone."""
    form, mapping, (kappa, _, _) = flux

    quantities = collect_quantities(form, mapping=mapping)
    dimensionless, factor = factorize(form, quantities, mapping=mapping)
    remainder = assemble(dimensionless)

    kappa.scale = 3.0
    scale = float(expand(factor, [q.factor for q in quantities]))

    assert assemble(dimensionless) == pytest.approx(remainder)
    assert scale * remainder == pytest.approx(assemble(transform(form, mapping)))
