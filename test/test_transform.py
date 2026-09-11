import ufl

import pytest
import sympy as sy
import sympy.physics.units as syu
from ufl_units import Quantity, collect_quantities, factorize, get_dimension, transform


def test_collect(mesh, V):
    """Only the quantities actually reachable from the expression are collected."""
    u = ufl.Coefficient(V)
    v = ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    u_ref = Quantity(mesh, 1.0, syu.kelvin, "u_ref")
    unused = Quantity(mesh, 1.0, syu.second, "T")

    assert set(collect_quantities(u_ref * u)) == {u_ref}
    assert set(collect_quantities(ufl.inner(u_ref * u, v) * ufl.dx)) == {u_ref}
    assert unused not in collect_quantities(u_ref * u)

    # The mesh scale enters through the mapping, not through the expression
    quantities = collect_quantities(ufl.grad(u), mapping={mesh: length, u: u_ref * u})
    assert set(quantities) == {length, u_ref}


def test_collect_is_ordered_by_construction(mesh, V):
    """The order is the order the quantities were built in, not that of a set."""
    u = ufl.Coefficient(V)
    first = Quantity(mesh, 1.0, syu.meter, "first")
    second = Quantity(mesh, 1.0, syu.second, "second")
    third = Quantity(mesh, 1.0, syu.kelvin, "third")

    # Assembled in an order unrelated to construction
    expr = third * u + second * u + first * u

    assert collect_quantities(expr) == [first, second, third]
    assert collect_quantities(expr * second) == [first, second, third]


def test_collect_invalid_type(mesh):
    """Only expressions and forms can be traversed for quantities."""
    with pytest.raises(TypeError, match="collecting quantities"):
        collect_quantities("not an expression")


def test_gradient(mesh, V):
    """Test that gradient expressions can be transformed with dimensional scaling."""
    u = ufl.Coefficient(V)
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    mapping = {u: length * u, mesh: length}
    grad_expr = ufl.grad(u)
    transformed = transform(grad_expr, mapping)
    # This should transform the gradient appropriately
    assert transformed is not None


def test_one_mesh_required(mesh, V):
    """The mapping must carry exactly one mesh, to scale the coordinates with."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol="L")
    with pytest.raises(ValueError, match="exactly one Mesh"):
        transform(ufl.grad(u), {u: length * u})


def test_quantity_scale_required(mesh, V):
    """The mesh scale has to be a Quantity, a plain number carries no unit."""
    u = ufl.Coefficient(V)
    with pytest.raises(TypeError, match="Mesh scale must be a Quantity"):
        transform(ufl.grad(u), {mesh: 2.0})


def test_form(mesh, V):
    """Test dimensional transformation and factorization of UFL bilinear forms."""
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    L_symbol = sy.Symbol("L")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L_symbol)
    u_ref = Quantity(mesh, 1.0, unit=syu.kelvin, symbol=L_symbol)

    # Create a simple form
    form = length**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    mapping = {u: u_ref * u, v: u_ref * v, mesh: length}

    # Transform the form
    transformed_form = transform(form, mapping)
    assert transformed_form is not None
    assert isinstance(transformed_form, ufl.Form)

    # Test factorization of the transformed form
    # The transformed form should be factorizable with the length quantity
    factorized = factorize(transformed_form, [length, u_ref])
    assert factorized is not None
    assert factorized.factor[0] == 2  # length factor should be 2 (L^2)
    assert factorized.factor[1] == 2


def test_spatial_coordinate(mesh):
    """Coordinates of a unit-less mesh carry the reference length."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    x = ufl.SpatialCoordinate(mesh)

    dim = get_dimension(x[0], [length], mapping={mesh: length})
    assert syu.si.SI.get_dimension_system().equivalent_dims(dim, syu.length)


@pytest.mark.parametrize(
    "quantity",
    [
        ufl.Circumradius,
        ufl.CellDiameter,
        ufl.MinCellEdgeLength,
        ufl.MaxCellEdgeLength,
        ufl.MinFacetEdgeLength,
        ufl.MaxFacetEdgeLength,
    ],
)
def test_length_quantities(mesh, quantity):
    """Every measure of cell or facet size scales like a single length."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dim = get_dimension(quantity(mesh), [length], mapping={mesh: length})
    assert syu.si.SI.get_dimension_system().equivalent_dims(dim, syu.length)


@pytest.mark.parametrize(
    "quantity", [ufl.FacetNormal, ufl.geometry.QuadratureWeight, ufl.geometry.CellOrientation]
)
def test_dimensionless_geometry_passes_through(mesh, quantity):
    """Reference-domain quantities, normals and orientations carry no length."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    transformed = transform(quantity(mesh) * length, {mesh: length})
    assert transformed is not None


@pytest.mark.parametrize(
    "quantity", [ufl.geometry.Jacobian, ufl.geometry.JacobianDeterminant, ufl.geometry.CellVertices]
)
def test_unhandled_geometry_raises(mesh, quantity):
    """A dimensional quantity without a rule is rejected, not silently left unscaled."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    with pytest.raises(NotImplementedError, match="no unit scaling rule"):
        transform(quantity(mesh), {mesh: length})


def test_cell_volume(mesh):
    """A cell volume scales with the topological dimension of the mesh."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    factorized = factorize(ufl.CellVolume(mesh), [length], mapping={mesh: length})
    assert factorized.factor is not None
    assert factorized.factor[0] == pytest.approx(2.0)  # triangle, tdim == 2


def test_facet_area(mesh):
    """A facet area scales one dimension lower than a cell volume."""
    length = Quantity(mesh, 1.0, syu.meter, "L")

    factorized = factorize(ufl.FacetArea(mesh), [length], mapping={mesh: length})
    assert factorized.factor is not None
    assert factorized.factor[0] == pytest.approx(1.0)  # triangle facet, tdim - 1 == 1


def test_div(mesh, V):
    """Divergence picks up an inverse length, like a gradient."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dim = get_dimension(ufl.div(ufl.grad(u)), [length], mapping={mesh: length})
    assert syu.si.SI.get_dimension_system().equivalent_dims(dim, 1 / syu.length**2)


@pytest.mark.parametrize(("measure", "exponent"), [("dx", 2.0), ("ds", 1.0)])
def test_measure_scaling(mesh, V, measure, exponent):
    """A cell measure scales with tdim, an exterior facet measure with tdim - 1."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    form = u * v * getattr(ufl, measure)
    factorized = factorize(form, [length], mapping={mesh: length})
    assert factorized.factor[0] == pytest.approx(exponent)


def test_interior_facet_measure(mesh, V):
    """An interior facet measure scales like an exterior one."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    form = ufl.avg(u) * ufl.avg(v) * ufl.dS
    factorized = factorize(form, [length], mapping={mesh: length})
    assert factorized.factor[0] == pytest.approx(1.0)


def test_transform_invalid_type(mesh):
    """Only expressions, forms and dictionaries of those can be transformed."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    with pytest.raises(TypeError, match="Unsupported type for unit transformation"):
        transform("not an expression", {mesh: length})


def test_transform_dict(mesh, V):
    """A dictionary is transformed value by value, keeping its keys."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    terms = {"mass": u * v * ufl.dx, "stiffness": ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx}
    transformed = transform(terms, {mesh: length})

    assert set(transformed) == {"mass", "stiffness"}
    assert all(isinstance(f, ufl.Form) for f in transformed.values())
