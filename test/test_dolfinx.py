"""Assembly of a quantity-scaled form with dolfinx as the value backend."""

import ufl

import pytest
import sympy.physics.units as syu
from ufl_units import collect_quantities, expand, factorize, get_dimension, transform

# dolfinx is not part of the `test` dependency group; it only exists in the backend images.
dolfinx = pytest.importorskip("dolfinx")
MPI = pytest.importorskip("mpi4py").MPI
Quantity = pytest.importorskip("ufl_units.backends.dolfinx").Quantity


@pytest.fixture
def unit_square():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)


@pytest.fixture
def flux(unit_square):
    """A conductive flux functional, its unit mapping and the quantities it is built from."""
    V = dolfinx.fem.functionspace(unit_square, ("Lagrange", 1))
    temperature = dolfinx.fem.Function(V)
    temperature.interpolate(lambda x: 1.0 + x[0] ** 2)

    kappa = Quantity(unit_square, 1.5, syu.watt / (syu.kelvin * syu.meter), "kappa")
    l_ref = Quantity(unit_square, 0.1, syu.meter, "l_ref")
    T_ref = Quantity(unit_square, 300.0, syu.kelvin, "T_ref")

    form = -kappa * ufl.grad(temperature)[0] * ufl.dx
    mapping = {unit_square.ufl_domain(): l_ref, temperature: T_ref * temperature}
    return form, mapping, [kappa, l_ref, T_ref]


def assemble(form, mesh):
    """Assemble a scalar form across all ranks."""
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)


def test_assemble_quantity(unit_square):
    """Integrating a quantity over the unit square yields its factor in base units."""
    length = Quantity(unit_square, 2.0, syu.kilometer, "L")
    assert length.factor == pytest.approx(2000.0)

    form = dolfinx.fem.form(length * ufl.dx(unit_square))
    value = unit_square.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
    assert value == pytest.approx(2000.0)


def test_assemble_after_scale_change(unit_square):
    """Setting ``scale`` reaches the assembled value through ``_update_value``."""
    length = Quantity(unit_square, 2.0, syu.kilometer, "L")
    form = dolfinx.fem.form(length * ufl.dx(unit_square))

    length.scale = 3.0
    value = unit_square.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)
    assert value == pytest.approx(3000.0)


def test_factorize_quantity(unit_square):
    """Factorizing pulls the quantity out, leaving the area of the unit square behind."""
    length = Quantity(unit_square, 2.0, syu.kilometer, "L")

    dimensionless, factor = factorize(length * ufl.dx(unit_square), [length])

    assert collect_quantities(dimensionless) == []
    assert float(expand(factor, [length.factor])) == pytest.approx(2000.0)
    assert assemble(dimensionless, unit_square) == pytest.approx(1.0)


def test_factorize_form(unit_square, flux):
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
    assert scale * assemble(dimensionless, unit_square) == pytest.approx(
        assemble(transform(form, mapping), unit_square)
    )


def test_factorize_after_scale_change(unit_square, flux):
    """A scale change moves into the factor and leaves the dimensionless form alone."""
    form, mapping, (kappa, _, _) = flux

    quantities = collect_quantities(form, mapping=mapping)
    dimensionless, factor = factorize(form, quantities, mapping=mapping)
    remainder = assemble(dimensionless, unit_square)

    kappa.scale = 3.0
    scale = float(expand(factor, [q.factor for q in quantities]))

    assert assemble(dimensionless, unit_square) == pytest.approx(remainder)
    assert scale * remainder == pytest.approx(assemble(transform(form, mapping), unit_square))
