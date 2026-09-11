"""Assembly of a quantity-scaled form with Firedrake as the value backend."""

import ufl

import pytest
import sympy.physics.units as syu

# firedrake is not part of the `test` dependency group; it only exists in the backend images.
firedrake = pytest.importorskip("firedrake")
Quantity = pytest.importorskip("ufl_units.backends.firedrake").Quantity


@pytest.fixture
def unit_square():
    return firedrake.UnitSquareMesh(4, 4)


def test_assemble_quantity(unit_square):
    """Integrating a quantity over the unit square yields its factor in base units."""
    length = Quantity(2.0, syu.kilometer, "L")
    assert length.factor == pytest.approx(2000.0)

    value = firedrake.assemble(length * ufl.dx(domain=unit_square))
    assert value == pytest.approx(2000.0)


def test_assemble_after_scale_change(unit_square):
    """Setting ``scale`` reaches the assembled value through ``_update_value``."""
    length = Quantity(2.0, syu.kilometer, "L")
    form = length * ufl.dx(domain=unit_square)

    length.scale = 3.0
    value = firedrake.assemble(form)
    assert value == pytest.approx(3000.0)
