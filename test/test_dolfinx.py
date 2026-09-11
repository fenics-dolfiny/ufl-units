"""Assembly of a quantity-scaled form with dolfinx as the value backend."""

import ufl

import pytest
import sympy.physics.units as syu

# dolfinx is not part of the `test` dependency group; it only exists in the backend images.
dolfinx = pytest.importorskip("dolfinx")
MPI = pytest.importorskip("mpi4py").MPI
Quantity = pytest.importorskip("ufl_units.backends.dolfinx").Quantity


@pytest.fixture
def unit_square():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)


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
