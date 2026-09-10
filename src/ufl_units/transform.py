import logging

import ufl
import ufl.geometry as ufl_geometry
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain
from ufl.form import Form

from ufl_units.quantity import QuantityMixin

logger = logging.getLogger(__name__)

# Geometric quantities that carry no length: quantities of the reference cell, facet or
# ridge, the reference-to-reference Jacobians between them, unit normals, orientations
# and quadrature weights. Everything else is either scaled by a handler below or
# rejected by `UnitTransformer.geometric_quantity`.
_DIMENSIONLESS_GEOMETRY = (
    ufl_geometry.CellCoordinate,
    ufl_geometry.FacetCoordinate,
    ufl_geometry.RidgeCoordinate,
    ufl_geometry.CellFacetJacobian,
    ufl_geometry.CellFacetJacobianDeterminant,
    ufl_geometry.CellFacetJacobianInverse,
    ufl_geometry.CellRidgeJacobian,
    ufl_geometry.CellRidgeJacobianDeterminant,
    ufl_geometry.CellRidgeJacobianInverse,
    ufl_geometry.FacetRidgeJacobian,
    ufl_geometry.CellFacetOrigin,
    ufl_geometry.CellRidgeOrigin,
    ufl_geometry.CellNormal,
    ufl_geometry.FacetNormal,
    ufl_geometry.ReferenceNormal,
    ufl_geometry.CellOrientation,
    ufl_geometry.FacetOrientation,
    ufl_geometry.QuadratureWeight,
    ufl_geometry.ReferenceCellVolume,
    ufl_geometry.ReferenceFacetVolume,
    ufl_geometry.ReferenceRidgeVolume,
    ufl_geometry.ReferenceCellEdgeVectors,
    ufl_geometry.ReferenceFacetEdgeVectors,
)


class UnitTransformer(MultiFunction):
    """Substitute quantities into an expression and rescale geometric quantities.

    The mapping must contain exactly one :class:`ufl.Mesh` key, whose value is the
    reference length of the mesh coordinates. All geometric quantities are rescaled
    with it, so that a form written on a unit-less mesh carries the dimensions of the
    physical domain.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
        meshes = [mesh for mesh in mapping.keys() if isinstance(mesh, ufl.Mesh)]
        if len(meshes) != 1:
            raise ValueError("Mapping must contain exactly one Mesh.")
        self._mesh_scale = self.mapping[meshes[0]]

        if not isinstance(self._mesh_scale, QuantityMixin):
            raise TypeError("Mesh scale must be a Quantity.")

        super().__init__()

    def ufl_type(self, o, *args):
        return self.mapping.get(o, self.reuse_if_untouched(o, *args))

    def grad(self, o, a):
        return o._ufl_expr_reconstruct_(a) / self._mesh_scale

    def spatial_coordinate(self, o, *ops):
        return o * self._mesh_scale

    def cell_volume(self, o, *ops):
        # Note that topological_dimension is a property, not a method
        tdim = extract_unique_domain(o).topological_dimension
        return o * (self._mesh_scale**tdim)

    def facet_area(self, o, *ops):
        tdim = extract_unique_domain(o).topological_dimension
        return o * (self._mesh_scale ** (tdim - 1))

    def geometric_quantity(self, o, *ops):
        """Reject geometric quantities that have no scaling rule.

        Reached only by quantities without a handler of their own. Those listed in
        `_DIMENSIONLESS_GEOMETRY` pass through untouched. Anything else carries a length
        and would silently yield a wrong dimension, so it is rejected instead.
        """
        if isinstance(o, _DIMENSIONLESS_GEOMETRY):
            return self.reuse_if_untouched(o, *ops)

        raise NotImplementedError(
            f"{type(o).__name__} has no unit scaling rule. Add a handler to "
            "UnitTransformer, or list it in _DIMENSIONLESS_GEOMETRY if it carries no unit."
        )

    div = grad
    curl = grad

    # All handler names below scale like a single length. Note that UFL dispatches these
    # as min/max_cell_edge_length and min/max_facet_edge_length, not as min/max_edge_length.
    circumradius = spatial_coordinate
    cell_diameter = spatial_coordinate
    min_cell_edge_length = spatial_coordinate
    max_cell_edge_length = spatial_coordinate
    min_facet_edge_length = spatial_coordinate
    max_facet_edge_length = spatial_coordinate


def _transform_expr(expr: Expr, mapping: dict):
    transformer = UnitTransformer(mapping)
    return map_expr_dag(transformer, expr)


def _transform_form(form: Form, mapping: dict) -> Form:
    _integral_type_codim = {"cell": 0, "interior_facet": 1, "exterior_facet": 1}

    transformer = UnitTransformer(mapping)
    transformed_integrals = []

    for integral in form.integrals():
        # Transform the integrand
        transformed_integrand = map_expr_dag(transformer, integral.integrand())

        # Scale by measure change
        tdim = integral.ufl_domain().topological_dimension
        measure_dim = tdim - _integral_type_codim[integral.integral_type()]
        scaled_integrand = transformed_integrand * (transformer._mesh_scale**measure_dim)

        transformed_integrals.append(integral.reconstruct(scaled_integrand))

    return Form(transformed_integrals)


def transform(expr: Expr | Form | dict, mapping: dict):
    """Transform expressions or forms by applying unit mapping and mesh scaling.

    Parameters
    ----------
    expr
        Expression, form, or dictionary to transform
    mapping
        Mapping of quantities to their replacements

    Returns
    -------
    Transformed expression, form, or dictionary

    """
    if isinstance(expr, dict):
        return {key: transform(value, mapping) for key, value in expr.items()}
    if isinstance(expr, Form):
        return _transform_form(expr, mapping)
    if isinstance(expr, Expr):
        return _transform_expr(expr, mapping)

    raise TypeError(f"Unsupported type for unit transformation: {type(expr).__name__}")


def collect_quantities(expr, mapping: dict | None = None) -> list[QuantityMixin]:
    """Collect all Quantity instances from a UFL expression.

    The quantities are returned in the order they were constructed in. Since the
    position of a quantity in this list fixes its column of the dimension matrix, and
    hence the basis of dimensionless groups that is reported, the order has to be
    reproducible rather than that of the set the quantities are gathered in.
    """
    if mapping is not None:
        expr = transform(expr, mapping)

    quantities = set()

    class QuantityCollector(MultiFunction):
        def ufl_type(self, o, *args):
            return self.reuse_if_untouched(o, *args)

        def constant(self, o, *ops):
            if isinstance(o, QuantityMixin):
                quantities.add(o)
            return self.reuse_if_untouched(o, *ops)

    if isinstance(expr, Form):
        for integral in expr.integrals():
            map_expr_dag(QuantityCollector(), integral.integrand())
    elif isinstance(expr, Expr):
        map_expr_dag(QuantityCollector(), expr)
    else:
        raise TypeError(f"Unsupported type for collecting quantities: {type(expr).__name__}")

    return sorted(quantities, key=lambda q: q.count())
