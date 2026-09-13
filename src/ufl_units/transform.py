import logging

import ufl
import ufl.geometry as ufl_geometry
from ufl.argument import Coargument
from ufl.coefficient import Cofunction
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm
from ufl.matrix import Matrix
from ufl.measure import point_integral_types

from ufl_units.quantity import QuantityMixin

logger = logging.getLogger(__name__)

# Dual and operator-valued forms that are not UFL expressions and hold no operands to
# descend into. A mapping may replace one.
_base_form_terminals = (Cofunction, Coargument, Matrix, ZeroBaseForm)

# Geometric quantities that carry no length: reference cell, facet and ridge quantities,
# the Jacobians between them, unit normals, orientations and quadrature weights.
_dimensionless_geometry = (
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

    def jacobian_inverse(self, o, *ops):
        return o / self._mesh_scale

    def cell_volume(self, o, *ops):
        # Note that topological_dimension is a property, not a method
        tdim = extract_unique_domain(o).topological_dimension
        return o * (self._mesh_scale**tdim)

    def facet_area(self, o, *ops):
        tdim = extract_unique_domain(o).topological_dimension
        return o * (self._mesh_scale ** (tdim - 1))

    def ridge_jacobian_determinant(self, o, *ops):
        tdim = extract_unique_domain(o).topological_dimension
        return o * (self._mesh_scale ** (tdim - 2))

    def geometric_quantity(self, o, *ops):
        """Reject geometric quantities that have no scaling rule, rather than assume one."""
        if isinstance(o, _dimensionless_geometry):
            return self.reuse_if_untouched(o, *ops)

        raise NotImplementedError(
            f"{type(o).__name__} has no unit scaling rule. Add a handler to "
            "UnitTransformer, or list it in _dimensionless_geometry if it carries no unit."
        )

    div = grad
    curl = grad
    nabla_grad = grad
    nabla_div = grad

    # Scale like a single length. Note the handler names UFL dispatches these to.
    circumradius = spatial_coordinate
    cell_diameter = spatial_coordinate
    min_cell_edge_length = spatial_coordinate
    max_cell_edge_length = spatial_coordinate
    min_facet_edge_length = spatial_coordinate
    max_facet_edge_length = spatial_coordinate
    cell_origin = spatial_coordinate
    facet_origin = spatial_coordinate
    ridge_origin = spatial_coordinate
    cell_vertices = spatial_coordinate
    cell_edge_vectors = spatial_coordinate
    facet_edge_vectors = spatial_coordinate
    # A Jacobian differentiates spatial coordinates with respect to reference ones, which
    # are dimensionless, so it too carries exactly one length.
    jacobian = spatial_coordinate
    facet_jacobian = spatial_coordinate
    ridge_jacobian = spatial_coordinate

    # The (pseudo-)inverses undo one length.
    facet_jacobian_inverse = jacobian_inverse
    ridge_jacobian_inverse = jacobian_inverse

    # A (pseudo-)determinant scales like the volume of the entity its Jacobian maps onto:
    # the cell (tdim), a facet (tdim - 1) or a ridge (tdim - 2).
    jacobian_determinant = cell_volume
    facet_jacobian_determinant = facet_area


# Codimension of the domain each UFL integral type integrates over, relative to the cell.
# The measure then scales as ``mesh_scale ** (tdim - codim)``.
_integral_type_codim = {
    "cell": 0,
    "exterior_facet": 1,
    "interior_facet": 1,
    "ridge": 2,
    # Extruded meshes: horizontal and vertical facets are facets like any other.
    "exterior_facet_bottom": 1,
    "exterior_facet_top": 1,
    "exterior_facet_vert": 1,
    "interior_facet_horiz": 1,
    "interior_facet_vert": 1,
}


def _measure_dim(integral_type: str, tdim: int) -> int:
    """Dimension of the domain an integral type integrates over, on a cell of dim ``tdim``."""
    if integral_type in point_integral_types:
        return 0

    if integral_type not in _integral_type_codim:
        raise NotImplementedError(
            f"Integral type '{integral_type}' has no measure scaling rule. "
            "Add its codimension to _integral_type_codim."
        )

    return tdim - _integral_type_codim[integral_type]


def _transform_expr(expr: Expr, mapping: dict):
    transformer = UnitTransformer(mapping)
    return map_expr_dag(transformer, expr)


def _transform_form(form: Form, mapping: dict) -> Form:
    transformer = UnitTransformer(mapping)
    transformed_integrals = []

    for integral in form.integrals():
        # Transform the integrand
        transformed_integrand = map_expr_dag(transformer, integral.integrand())

        # Scale by measure change
        tdim = integral.ufl_domain().topological_dimension
        measure_dim = _measure_dim(integral.integral_type(), tdim)
        scaled_integrand = transformed_integrand * (transformer._mesh_scale**measure_dim)

        transformed_integrals.append(integral.reconstruct(scaled_integrand))

    return Form(transformed_integrals)


def _transform_form_sum(form_sum: FormSum, mapping: dict) -> FormSum:
    """Transform each term of a sum of base forms, weights included.

    Scaling a dual object by a quantity produces a :class:`ufl.form.FormSum` carrying the
    quantity as a *weight*, outside the expression DAG, so the weights have to be
    transformed alongside the components.
    """
    components = [transform(component, mapping) for component in form_sum.components()]
    weights = [
        transform(weight, mapping) if isinstance(weight, Expr) else weight
        for weight in form_sum.weights()
    ]
    return FormSum(*zip(components, weights))


def transform(expr: Expr | BaseForm | dict, mapping: dict):
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
    if isinstance(expr, FormSum):
        return _transform_form_sum(expr, mapping)
    # A BaseFormOperator such as `ufl.Interpolate` is both a BaseForm and an Expr, and it
    # belongs here: its operands are an expression DAG to descend into.
    if isinstance(expr, Expr):
        return _transform_expr(expr, mapping)
    if isinstance(expr, _base_form_terminals):
        return mapping.get(expr, expr)

    raise TypeError(f"Unsupported type for unit transformation: {type(expr).__name__}")


def collect_quantities(expr, mapping: dict | None = None) -> list[QuantityMixin]:
    """Collect all Quantity instances from a UFL expression.

    Returned in construction order: a quantity's position fixes its column of the
    dimension matrix, so the order has to be reproducible.
    """
    if mapping is not None:
        expr = transform(expr, mapping)

    quantities: set[QuantityMixin] = set()

    class QuantityCollector(MultiFunction):
        def ufl_type(self, o, *args):
            # By mixin type, not handler name, so any backend's terminal is collected
            if isinstance(o, QuantityMixin):
                quantities.add(o)
            return self.reuse_if_untouched(o, *args)

    def collect(expr) -> None:
        if isinstance(expr, Form):
            for integral in expr.integrals():
                map_expr_dag(QuantityCollector(), integral.integrand())
        elif isinstance(expr, FormSum):
            for component, weight in zip(expr.components(), expr.weights()):
                collect(component)
                if isinstance(weight, Expr):
                    collect(weight)
        elif isinstance(expr, Expr):
            map_expr_dag(QuantityCollector(), expr)
        elif isinstance(expr, _base_form_terminals):
            pass  # Holds no operands; a quantity can only reach it through the mapping.
        else:
            raise TypeError(f"Unsupported type for collecting quantities: {type(expr).__name__}")

    collect(expr)

    return sorted(quantities, key=lambda q: q.count())
