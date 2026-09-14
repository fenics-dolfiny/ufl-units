import basix.ufl
import ufl
from ufl.measure import point_integral_types

import pytest
import sympy as sy
import sympy.physics.units as syu
from ufl_units import (
    Quantity,
    UnitTransformer,
    collect_quantities,
    factorize,
    get_dimension,
    transform,
)
from ufl_units.transform import _dimensionless_geometry, _integral_type_codim, _measure_dim


@pytest.fixture(scope="module")
def mesh_3d():
    """A tetrahedral mesh, for the quantities whose scaling depends on `tdim`."""
    return ufl.Mesh(basix.ufl.element("P", "tetrahedron", 1, shape=(3,)))


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


def _geometry_exponent(mesh, quantity):
    """Length exponent `quantity` picks up when transformed on `mesh`."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    return factorize(quantity(mesh), [length], mapping={mesh: length}).factor[0]


@pytest.mark.parametrize(
    "quantity",
    [
        # Measures of cell or facet size
        ufl.Circumradius,
        ufl.CellDiameter,
        ufl.MinCellEdgeLength,
        ufl.MaxCellEdgeLength,
        ufl.MinFacetEdgeLength,
        ufl.MaxFacetEdgeLength,
        # Physical coordinates, and differences of them
        ufl.SpatialCoordinate,
        ufl.geometry.CellOrigin,
        ufl.geometry.FacetOrigin,
        ufl.geometry.RidgeOrigin,
        ufl.geometry.CellVertices,
        ufl.geometry.CellEdgeVectors,
        ufl.geometry.FacetEdgeVectors,
        # Jacobians, against dimensionless reference coordinates
        ufl.geometry.Jacobian,
        ufl.geometry.FacetJacobian,
        ufl.geometry.RidgeJacobian,
    ],
    ids=lambda cls: cls.__name__,
)
def test_length_quantities(mesh_3d, quantity):
    """Everything built from a physical coordinate scales like a single length."""
    assert _geometry_exponent(mesh_3d, quantity) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "quantity",
    [
        ufl.geometry.JacobianInverse,
        ufl.geometry.FacetJacobianInverse,
        ufl.geometry.RidgeJacobianInverse,
    ],
    ids=lambda cls: cls.__name__,
)
def test_inverse_length_quantities(mesh_3d, quantity):
    """The (pseudo-)inverse of a Jacobian undoes its length."""
    assert _geometry_exponent(mesh_3d, quantity) == pytest.approx(-1.0)


@pytest.mark.parametrize("quantity", _dimensionless_geometry, ids=lambda cls: cls.__name__)
def test_dimensionless_geometry_passes_through(mesh_3d, quantity):
    """Everything listed as dimensionless really does come through unscaled."""
    assert _geometry_exponent(mesh_3d, quantity) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("quantity", "exponent"),
    [
        # tdim == 3, so the cell, facet and ridge exponents are distinct
        (ufl.CellVolume, 3),
        (ufl.geometry.JacobianDeterminant, 3),
        (ufl.FacetArea, 2),
        (ufl.geometry.FacetJacobianDeterminant, 2),
        (ufl.geometry.RidgeJacobianDeterminant, 1),
    ],
    ids=lambda arg: arg.__name__ if isinstance(arg, type) else str(arg),
)
def test_volume_quantities(mesh_3d, quantity, exponent):
    """A volume, and the (pseudo-)determinant mapping onto it, scale with its dimension."""
    assert _geometry_exponent(mesh_3d, quantity) == pytest.approx(exponent)


def test_every_geometric_quantity_has_a_rule(mesh_3d):
    """No geometric quantity UFL ships falls through to the NotImplementedError."""
    length = Quantity(mesh_3d, 1.0, syu.meter, "L")

    for cls in vars(ufl.geometry).values():
        if isinstance(cls, type) and issubclass(cls, ufl.geometry.GeometricQuantity):
            if not cls._ufl_is_abstract_:
                transform(cls(mesh_3d), {mesh_3d: length})


def test_unhandled_geometry_raises(mesh):
    """A dimensional quantity without a rule is rejected, not silently left unscaled.

    Every quantity UFL ships now has one, so the fallback is reached directly, standing in
    for a geometric quantity a future UFL adds.
    """
    length = Quantity(mesh, 1.0, syu.meter, "L")
    transformer = UnitTransformer({mesh: length})

    with pytest.raises(NotImplementedError, match="no unit scaling rule"):
        transformer.geometric_quantity(ufl.SpatialCoordinate(mesh))


def test_div(mesh, V):
    """Divergence picks up an inverse length, like a gradient."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dim = get_dimension(ufl.div(ufl.grad(u)), [length], mapping={mesh: length})
    assert syu.si.SI.get_dimension_system().equivalent_dims(dim, 1 / syu.length**2)


@pytest.mark.parametrize("integrand", ["u", "L*u", "L*u**2"])
def test_coordinate_derivative(mesh, V, W, integrand):
    """A shape derivative sits one length below the form, as the Hadamard formula does."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    u_ref = Quantity(mesh, 2.0, syu.kelvin, "u_ref")
    direction = ufl.TestFunction(W)

    f = {"u": u, "L*u": length * u, "L*u**2": length * u**2}[integrand]
    quantities = [length, u_ref]
    mapping = {mesh: length, u: u_ref * u}

    hadamard = (ufl.dot(ufl.grad(f), direction) + f * ufl.div(direction)) * ufl.dx
    lazy = ufl.derivative(f * ufl.dx, ufl.SpatialCoordinate(mesh), direction)

    expected = factorize(hadamard, quantities, mapping=mapping).factor
    form_factor = factorize(f * ufl.dx, quantities, mapping=mapping).factor

    assert factorize(lazy, quantities, mapping=mapping).factor == pytest.approx(expected)
    # One length lower than the form, and the same power of u_ref
    assert expected == pytest.approx(form_factor - [1.0, 0.0])


@pytest.mark.parametrize("integrand", ["L*u*v", "L*u**2*v", "u*v"])
def test_coefficient_derivative(mesh, V, integrand):
    """Differentiating against a mapped coefficient divides its reference value out."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    u_ref = Quantity(mesh, 2.0, syu.kelvin, "u_ref")

    f = {"L*u*v": length * u * v, "L*u**2*v": length * u**2 * v, "u*v": u * v}[integrand]
    quantities = [length, u_ref]
    mapping = {mesh: length, u: u_ref * u}

    form_factor = factorize(f * ufl.dx, quantities, mapping=mapping).factor
    derivative_factor = factorize(ufl.derivative(f * ufl.dx, u), quantities, mapping=mapping).factor

    assert derivative_factor == pytest.approx(form_factor - [0.0, 1.0])


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


_supported_integral_types = sorted(_integral_type_codim) + list(point_integral_types)
_unsupported_integral_types = sorted(
    set(ufl.measure.integral_type_to_measure_name) - set(_supported_integral_types)
)


def _measure_exponent(mesh, integral_type):
    """Length exponent the measure of `integral_type` contributes on `mesh`."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    form = ufl.Constant(mesh) * ufl.Measure(integral_type, domain=mesh)
    return factorize(form, [length], mapping={mesh: length}).factor[0]


@pytest.mark.parametrize("integral_type", _supported_integral_types)
def test_measure_scaling_by_integral_type(mesh_3d, integral_type):
    """Each measure carries the length of the domain it integrates over."""
    assert _measure_exponent(mesh_3d, integral_type) == pytest.approx(
        _measure_dim(integral_type, tdim=3)
    )


@pytest.mark.parametrize("integral_type", _unsupported_integral_types)
def test_unsupported_integral_type_raises(mesh_3d, integral_type):
    """An integral type with no codimension is rejected, not silently taken as a cell."""
    length = Quantity(mesh_3d, 1.0, syu.meter, "L")
    form = ufl.Constant(mesh_3d) * ufl.Measure(integral_type, domain=mesh_3d)

    with pytest.raises(NotImplementedError, match="no measure scaling rule"):
        transform(form, {mesh_3d: length})


def test_transform_invalid_type(mesh):
    """Only expressions, forms and dictionaries of those can be transformed."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    with pytest.raises(TypeError, match="Unsupported type for unit transformation"):
        transform("not an expression", {mesh: length})


def test_transform_interpolate(mesh, V):
    """An Interpolate is a BaseForm and an Expr, and is descended into as the latter."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    u_ref = Quantity(mesh, 2.0, syu.kelvin, "u_ref")

    transformed = transform(ufl.Interpolate(u, V), {mesh: length, u: u_ref * u})

    assert isinstance(transformed, ufl.Interpolate)
    assert set(collect_quantities(transformed)) == {u_ref}


@pytest.mark.parametrize("build", [lambda V: ufl.Cofunction(V.dual()), lambda V: ufl.Matrix(V, V)])
def test_transform_base_form_terminal(mesh, V, build):
    """A dual object holds no operands: a mapping replaces it whole, or it passes through."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    terminal = build(V)

    assert transform(terminal, {mesh: length}) is terminal
    assert transform(terminal, {mesh: length, terminal: length * terminal}) is not terminal


def test_transform_form_sum(mesh, V):
    """Both the components of a sum of base forms and its weights are transformed."""
    u = ufl.Coefficient(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")
    u_ref = Quantity(mesh, 2.0, syu.kelvin, "u_ref")

    form_sum = ufl.FormSum((ufl.Interpolate(u, V), 1.0), (ufl.Cofunction(V.dual()), 1.0))
    transformed = transform(form_sum, {mesh: length, u: u_ref * u})

    assert isinstance(transformed, ufl.FormSum)
    assert set(collect_quantities(transformed)) == {u_ref}


def test_collect_form_sum_weights(mesh, V):
    """Scaling a dual object parks the quantity in a weight, outside the expression DAG."""
    length = Quantity(mesh, 1.0, syu.meter, "L")
    cofunction = ufl.Cofunction(V.dual())

    scaled = length * cofunction
    assert isinstance(scaled, ufl.FormSum)
    assert scaled.weights() == [length]
    assert collect_quantities(scaled) == [length]


def test_transform_dict(mesh, V):
    """A dictionary is transformed value by value, keeping its keys."""
    u, v = ufl.Coefficient(V), ufl.TestFunction(V)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    terms = {"mass": u * v * ufl.dx, "stiffness": ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx}
    transformed = transform(terms, {mesh: length})

    assert set(transformed) == {"mass", "stiffness"}
    assert all(isinstance(f, ufl.Form) for f in transformed.values())


@pytest.mark.parametrize(
    ("plain", "nabla"),
    [(ufl.grad, ufl.nabla_grad), (ufl.div, ufl.nabla_div)],
)
def test_nabla_scaling(mesh, W, plain, nabla):
    """The nabla spellings carry the reference length of the mesh, as grad and div do."""
    u = ufl.Coefficient(W)
    length = Quantity(mesh, 1.0, syu.meter, "L")

    dimsys = syu.si.SI.get_dimension_system()
    mapping = {mesh: length}
    plain_dim = get_dimension(ufl.inner(plain(u), plain(u)), [length], mapping=mapping)
    nabla_dim = get_dimension(ufl.inner(nabla(u), nabla(u)), [length], mapping=mapping)

    assert dimsys.equivalent_dims(plain_dim, 1 / syu.length**2)
    assert dimsys.equivalent_dims(nabla_dim, plain_dim)
