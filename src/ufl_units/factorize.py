import fractions
import logging
import math
from collections.abc import Sequence
from typing import NamedTuple, overload

from ufl.algorithms.map_integrands import map_integrands
from ufl.conditional import Condition
from ufl.constantvalue import Zero, as_ufl
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.form import BaseForm, Form, FormSum

import numpy as np
import sympy as sy
from ufl_units.exceptions import InconsistentDimensions
from ufl_units.quantity import QuantityMixin, to_base_units
from ufl_units.table import print_table
from ufl_units.transform import _base_form_terminals, transform

logger = logging.getLogger(__name__)


def _zeros(n: int) -> np.ndarray:
    """Return a zero exponent vector of length ``n``, holding exact fractions."""
    return np.full(n, fractions.Fraction(0), dtype=object)


def _is_zero(exponents, atol: float = 1e-12) -> bool:
    """Whether all exponents vanish, up to the rounding of float exponents.

    UFL holds a literal exponent such as 1/3 as the nearest float, so exponents built from
    it agree up to rounding only, e.g. 3 * (1/3) is 1 + 2**-54.
    """
    return all(math.isclose(e, 0, abs_tol=atol) for e in exponents)


def _same_dimension(
    fa: np.ndarray, fb: np.ndarray, quantities: Sequence[QuantityMixin], atol: float = 1e-12
) -> bool:
    """Whether two exponent vectors carry the same dimension, compared in base dimensions."""
    difference: dict = {}
    for q, e in zip(quantities, fb - fa):
        for dim, power in q.dimensional_dependencies.items():
            difference[dim] = difference.get(dim, 0) + e * fractions.Fraction(power)
    return _is_zero(difference.values(), atol=atol)


class FactorizedExpr(NamedTuple):
    expr: Expr | BaseForm
    factor: np.ndarray | None
    dimless_numbers: tuple[np.ndarray, ...] = ()


class QuantityFactorizer(MultiFunction):
    """Pull the quantities out of an expression, tracking their exponents per node.

    Each node gets an exponent vector over ``quantities``, and the quantities are replaced
    by one, leaving the dimensionless remainder and the factor pulled out. Operands that
    agree dimensionally but carry different factors keep the dimensionless number they
    differ by, which is also collected in ``dimless_numbers``.
    """

    factors: dict[Expr, np.ndarray]
    dimless_numbers: list[np.ndarray]

    def __init__(self, quantities: Sequence[QuantityMixin]):
        self._quantities = quantities
        if any(q.ufl_shape != () for q in quantities):
            raise NotImplementedError("Factorization of non-scalar quantities is not implemented.")
        self.factors = {}
        self.dimless_numbers = []
        super().__init__()

    def product(self, o, *ops):
        a, b = o.ufl_operands
        self.factors[o] = self.factors[a] + self.factors[b]
        return self.reuse_if_untouched(o, *ops)

    def division(self, o, *ops):
        a, b = o.ufl_operands
        self.factors[o] = self.factors[a] - self.factors[b]
        return self.reuse_if_untouched(o, *ops)

    def power(self, o, *ops):
        a, n = o.ufl_operands
        self.factors[o] = fractions.Fraction(float(n)) * self.factors[a]
        return self.reuse_if_untouched(o, *ops)

    def sqrt(self, o, *ops):
        a = o.ufl_operands[0]
        self.factors[o] = self.factors[a] / 2
        return self.reuse_if_untouched(o, *ops)

    def independent(self, o, *ops):
        if isinstance(o, QuantityMixin):
            raise NotImplementedError(
                f"{type(o).__name__} is a quantity dispatched as "
                f"'{o._ufl_handler_name_}', which QuantityFactorizer does not handle. "
                "Alias that handler name to `quantity_terminal`."
            )

        self.factors.setdefault(o, _zeros(len(self._quantities)))
        return self.reuse_if_untouched(o, *ops)

    def _constraining_factors(self, operands) -> list[np.ndarray]:
        """Operand factors that constrain the dimension.

        Zero fits any dimension, and a condition is a boolean, carrying none.
        """
        return [
            self.factors[op]
            for op in operands
            if op in self.factors and not isinstance(op, (Zero, Condition))
        ]

    def _jointly_homogeneous(self, o, ops, degree: int):
        # Jointly homogeneous nodes of degree k, f(a*x, a*y) = a**k * f(x, y), require
        # operands of one dimension. The first operand gives the common factor, the others
        # are rescaled to it, and the node carries k times the common factor.
        factors = self._constraining_factors(o.ufl_operands)

        if len(factors) > 0:
            ops = self._check_operands(o, ops, factors[0])

        if degree == 0:
            self.factors[o] = _zeros(len(self._quantities))
        elif len(factors) > 0:
            self.factors[o] = factors[0]

        return self.reuse_if_untouched(o, *ops)

    def jointly_homogeneous_degree_1(self, o, *ops):
        # E.g. sum, grad: the node carries the common factor of its operands.
        return self._jointly_homogeneous(o, ops, 1)

    def jointly_homogeneous_degree_0(self, o, *ops):
        # E.g. a comparison: the common factor drops out of the result.
        return self._jointly_homogeneous(o, ops, 0)

    def nonhomogeneous(self, o, *ops):
        # Nonhomogeneous nodes (e.g. sin(x), exp(x), etc.) must have dimensionless
        # operands. No scale can be pulled out of a nonlinear function, so the operands
        # keep their factors, which are dimensionless numbers.
        factors = self._constraining_factors(o.ufl_operands)

        if len(factors) > 0:
            zero = _zeros(len(self._quantities))
            ops = self._check_operands(o, ops, zero)
            self.factors[o] = zero
        else:
            self.factors.setdefault(o, _zeros(len(self._quantities)))

        return self.reuse_if_untouched(o, *ops)

    def determinant(self, o, *ops):
        a = o.ufl_operands[0]
        self.factors[o] = (a.ufl_shape[0] if a.ufl_shape else 1) * self.factors[a]
        return self.reuse_if_untouched(o, *ops)

    def cofactor(self, o, *ops):
        a = o.ufl_operands[0]
        self.factors[o] = (a.ufl_shape[0] - 1) * self.factors[a]
        return self.reuse_if_untouched(o, *ops)

    def inverse(self, o, *ops):
        a = o.ufl_operands[0]
        self.factors[o] = -self.factors[a]
        return self.reuse_if_untouched(o, *ops)

    def multi_index(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def label(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def quantity_terminal(self, o, *ops):
        """Handle a terminal that may be one of the quantities being factorized out."""
        if o in self._quantities:
            idx = self._quantities.index(o)
            self.factors[o] = _zeros(len(self._quantities))
            self.factors[o][idx] = fractions.Fraction(1)
            # A UFL node, not a plain int: some operators reject non-UFL operands.
            return as_ufl(1)
        else:
            self.factors.setdefault(o, _zeros(len(self._quantities)))
            return self.reuse_if_untouched(o, *ops)

    def expr_list(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def expr_mapping(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def coefficient_derivative(self, o, *ops):
        """Divide by what is differentiated against, multiply by the direction."""
        integrand, variables, directions = o.ufl_operands[:3]
        zero = _zeros(len(self._quantities))

        factor = self.factors.get(integrand, zero).copy()
        for variable in variables.ufl_operands:
            factor -= self.factors.get(variable, zero)
        for direction in directions.ufl_operands:
            factor += self.factors.get(direction, zero)

        self.factors[o] = factor
        return self.reuse_if_untouched(o, *ops)

    def _check_operands(self, o, ops, reference_factor):
        r"""Check that the operands of the expression are consistent with a reference factor.

        Raises if an operand differs from the reference dimensionally.
        An operand that differs only by a dimensionless number is multiplied by it, so the
        mapped operands returned stay exact once the reference factor is pulled out.
        """
        ops = list(ops)

        for i, op in enumerate(o.ufl_operands):
            if op not in self.factors or isinstance(op, (Zero, Condition)):
                continue  # See _constraining_factors.

            fb = self.factors[op]

            if not _same_dimension(reference_factor, fb, self._quantities):
                f0_symbol = expand(reference_factor, [q.symbol for q in self._quantities])
                fb_symbol = expand(fb, [q.symbol for q in self._quantities])
                f0_expr = expand(reference_factor, [q.dimension for q in self._quantities])
                fb_expr = expand(fb, [q.dimension for q in self._quantities])
                raise InconsistentDimensions(
                    f"Inconsistent dimensions in operands of {o.__class__.__name__}.\n"
                    f"dim({f0_symbol}) != dim({fb_symbol}), (i.e. {f0_expr} != {fb_expr}).\n"
                )

            dimless_number = fb - reference_factor

            if not _is_zero(dimless_number):
                self.dimless_numbers.append(dimless_number)
                ops[i] = ops[i] * expand(dimless_number, self._quantities)

        return ops

    terminal = independent

    # The terminal a backend builds a quantity from: ufl.Constant (UFL, DOLFINx), a
    # Coefficient (function on a real space), or Firedrake's own registered type.
    constant = quantity_terminal
    coefficient = quantity_terminal
    firedrake_constant = quantity_terminal

    sum = jointly_homogeneous_degree_1
    indexed = jointly_homogeneous_degree_1
    grad = jointly_homogeneous_degree_1
    div = jointly_homogeneous_degree_1
    curl = jointly_homogeneous_degree_1
    nabla_grad = jointly_homogeneous_degree_1
    nabla_div = jointly_homogeneous_degree_1
    reference_value = jointly_homogeneous_degree_1
    reference_grad = jointly_homogeneous_degree_1
    reference_div = jointly_homogeneous_degree_1
    reference_curl = jointly_homogeneous_degree_1
    conj = jointly_homogeneous_degree_1
    real = jointly_homogeneous_degree_1
    imag = jointly_homogeneous_degree_1
    abs = jointly_homogeneous_degree_1
    index_sum = jointly_homogeneous_degree_1
    transposed = jointly_homogeneous_degree_1
    deviatoric = jointly_homogeneous_degree_1
    sym = jointly_homogeneous_degree_1
    skew = jointly_homogeneous_degree_1
    perp = jointly_homogeneous_degree_1
    trace = jointly_homogeneous_degree_1
    variable = jointly_homogeneous_degree_1
    component_tensor = jointly_homogeneous_degree_1
    list_tensor = jointly_homogeneous_degree_1
    restricted = jointly_homogeneous_degree_1
    cell_avg = jointly_homogeneous_degree_1
    facet_avg = jointly_homogeneous_degree_1
    max_value = jointly_homogeneous_degree_1
    min_value = jointly_homogeneous_degree_1
    # The condition is skipped, only the two branches carry a dimension.
    conditional = jointly_homogeneous_degree_1
    # Comparing requires operands of one dimension, but the result is a boolean.
    condition = jointly_homogeneous_degree_0
    # atan2(a*y, a*x) = atan2(y, x) for a > 0, so the result is dimensionless.
    atan2 = jointly_homogeneous_degree_0

    # Differentiates against the mesh coordinates, which the mapping has already scaled.
    coordinate_derivative = coefficient_derivative
    base_form_coordinate_derivative = coefficient_derivative
    base_form_operator_coordinate_derivative = coefficient_derivative

    variable_derivative = division

    inner = product
    dot = product
    cross = product
    outer = product

    interpolate = jointly_homogeneous_degree_1

    expr = nonhomogeneous


def _root_factor(factorizer: QuantityFactorizer, root_expr: Expr) -> np.ndarray | None:
    """Return the factor of the mapped root expression, with a traversal-order fallback."""
    if root_expr in factorizer.factors:
        return factorizer.factors[root_expr]

    fallback_root_expr = next(reversed(factorizer.factors), None)
    return factorizer.factors[fallback_root_expr] if fallback_root_expr is not None else None


def _check_consistent_factors(
    factors: Sequence[np.ndarray | None],
    quantities: Sequence[QuantityMixin],
    context: str,
) -> list[np.ndarray | None]:
    """Check that the terms of a form all carry the same dimension as the first one.

    ``context`` names the kind of term being compared, for the error message: the
    integrals of a Form, or the components of a FormSum. Returns, per term, the
    dimensionless number it differs from the first term by, or None if it does not.
    """
    dimless_numbers: list[np.ndarray | None] = [None] * len(factors)
    if not quantities or not factors:
        return dimless_numbers  # Nothing was factorized out, so every term trivially agrees.

    fa = factors[0]

    for i, fb in enumerate(factors):
        if fa is None or fb is None:
            continue

        if not _same_dimension(fa, fb, quantities):
            fa_expr = expand(fa, [q.dimension for q in quantities])
            fb_expr = expand(fb, [q.dimension for q in quantities])
            fa_symbol = expand(fa, [q.symbol for q in quantities])
            fb_symbol = expand(fb, [q.symbol for q in quantities])
            raise InconsistentDimensions(
                f"Inconsistent dimensions across {context}. \n"
                f"Scales: {fa_symbol} != {fb_symbol}. \n"
                f"{fa_expr} != {fb_expr}."
            )

        dimless_number = fb - fa

        if not _is_zero(dimless_number):
            dimless_numbers[i] = dimless_number

    return dimless_numbers


@overload
def factorize(
    expr: dict,
    quantities: Sequence[QuantityMixin],
    mapping: dict | None = None,
) -> dict[str, FactorizedExpr]: ...


@overload
def factorize(
    expr: Expr | BaseForm,
    quantities: Sequence[QuantityMixin],
    mapping: dict | None = None,
) -> FactorizedExpr: ...


def factorize(
    expr: Expr | BaseForm | dict,
    quantities: Sequence[QuantityMixin],
    mapping: dict | None = None,
) -> FactorizedExpr | dict[str, FactorizedExpr]:
    """Factorize expressions, forms, or dictionaries to extract dimensional factors.

    Raises :class:`InconsistentDimensions` if the expression is dimensionally inconsistent.
    Terms that agree dimensionally but carry different factors keep the dimensionless
    number they differ by, so the expression is always ``expand(factor) * expr``.

    Parameters
    ----------
    expr
        Expression, form, or dictionary to factorize
    quantities
        List of quantities to use for factorization
    mapping
        Optional mapping for unit transformation

    Returns
    -------
    FactorizedExpr | dict
        Factorized expression with dimensional factors, or dict of factorized items

    """
    if mapping is not None:
        expr = transform(expr, mapping)

    if isinstance(expr, dict):
        return {key: factorize(value, quantities) for key, value in expr.items()}
    factorized_expression: Expr | BaseForm

    dimless_numbers: list[np.ndarray] = []

    if isinstance(expr, Form):
        factorized_integrals = []
        factors: list[np.ndarray | None] = []
        for integral in expr.integrals():
            factorizer = QuantityFactorizer(quantities)
            integrand = integral.integrand()
            factorized_integrand = map_expr_dag(factorizer, integrand)
            factorized_integrals.append(integral.reconstruct(factorized_integrand))
            factors.append(_root_factor(factorizer, integrand))
            dimless_numbers += factorizer.dimless_numbers

        groups = _check_consistent_factors(factors, quantities, "integrals in Form")
        factorized_integrals = [
            itg if g is None else itg.reconstruct(itg.integrand() * expand(g, quantities))
            for itg, g in zip(factorized_integrals, groups)
        ]
        dimless_numbers += [g for g in groups if g is not None]

        factorized_expression = Form(factorized_integrals)
        root_factor = factors[0] if factors else None
    elif isinstance(expr, FormSum):
        factorized_expression, root_factor, dimless_numbers = _factorize_form_sum(expr, quantities)
    # A BaseFormOperator such as `ufl.Interpolate` is both a BaseForm and an Expr, and it
    # belongs here: its operands are an expression DAG to descend into.
    elif isinstance(expr, Expr):
        factorizer = QuantityFactorizer(quantities)
        factorized_expression = map_expr_dag(factorizer, expr)
        root_factor = _root_factor(factorizer, expr)
        dimless_numbers = factorizer.dimless_numbers
    elif isinstance(expr, _base_form_terminals):
        # Holds no operands, so nothing can be pulled out of it and it is dimensionless
        # unless a mapping has already scaled it, which shows up as a FormSum weight.
        factorized_expression = expr
        root_factor = _zeros(len(quantities))
    else:
        raise TypeError(f"Unsupported type for factorization: {type(expr).__name__}")

    return FactorizedExpr(factorized_expression, root_factor, tuple(dimless_numbers))


def _factorize_form_sum(
    form_sum: FormSum, quantities: Sequence[QuantityMixin]
) -> tuple[FormSum, np.ndarray | None, list[np.ndarray]]:
    """Factorize each term of a sum of base forms, weights included.

    Scaling a dual object by a quantity produces a :class:`ufl.form.FormSum` carrying the
    quantity as a *weight*, outside the expression DAG, so a term's factor is the one of
    its component plus the one of its weight.
    """
    components: list[Expr | BaseForm] = []
    weights: list[Expr | complex] = []
    factors: list[np.ndarray | None] = []
    dimless_numbers: list[np.ndarray] = []

    for component, weight in zip(form_sum.components(), form_sum.weights()):
        component, component_factor, component_groups = factorize(component, quantities)
        dimless_numbers += component_groups

        if isinstance(weight, Expr):
            weight, weight_factor, weight_groups = factorize(weight, quantities)
            dimless_numbers += weight_groups
        else:
            weight_factor = _zeros(len(quantities))

        components.append(component)
        weights.append(weight)
        if component_factor is None or weight_factor is None:
            factors.append(None)
        else:
            factors.append(component_factor + weight_factor)

    groups = _check_consistent_factors(factors, quantities, "components of FormSum")
    weights = [w if g is None else w * expand(g, quantities) for w, g in zip(weights, groups)]
    dimless_numbers += [g for g in groups if g is not None]

    return (
        FormSum(*zip(components, weights)),
        (factors[0] if factors else None),
        dimless_numbers,
    )


def expand(factor: np.ndarray | Sequence, quantities: Sequence) -> sy.Expr:
    """Expand factor array into symbolic expression using quantities as base.

    Parameters
    ----------
    factor
        Array of exponents for each quantity
    quantities
        Quantities/symbols to use as base

    Returns
    -------
    sy.Expr
        Symbolic expression with quantities raised to corresponding powers

    """
    return math.prod(q ** fractions.Fraction(f) for q, f in zip(quantities, factor))


def get_dimension(
    expr: Expr | BaseForm, quantities: Sequence[QuantityMixin], mapping: dict | None = None
) -> sy.Expr:
    """Get the physical dimension of an expression.

    Parameters
    ----------
    expr
        The expression to analyze for dimensional consistency.
    quantities
        List of quantities with their associated dimensions.
    mapping
        Optional mapping for variable substitution. Default is None.

    Returns
    -------
    unit
        The simplified physical dimension of the expression.

    """
    factor = factorize(expr, quantities, mapping=mapping)
    assert isinstance(factor, FactorizedExpr)
    assert factor.factor is not None, "Factorized expression must have a non-None factor"

    unit = expand(factor.factor, [q.dimension for q in quantities]).simplify()
    return unit


def normalize(
    expr_dict: dict[str, FactorizedExpr],
    reference_key: str,
    quantities: Sequence[QuantityMixin],
) -> dict[str, Expr | BaseForm]:
    """Normalize expressions or forms with respect to a reference expression.

    Parameters
    ----------
    expr_dict
        Dictionary of expressions or forms to normalize
    reference_key
        Key of the reference expression for normalization
    quantities
        List of quantities to use for factorization

    Returns
    -------
        Dictionary of normalized expressions or forms

    """
    if reference_key not in expr_dict:
        raise KeyError(f"Reference key '{reference_key}' not found in expression dictionary")

    # Validate that expr_dict contains factorized expressions
    for key, value in expr_dict.items():
        if not isinstance(value, FactorizedExpr):
            raise TypeError(
                f"Expression '{key}' must be already factorized (FactorizedExpr),"
                f"got {type(value).__name__}."
            )
        if value.factor is None:
            raise ValueError(
                f"Expression '{key}' has a None factor. "
                "All expressions must have valid factors for normalization."
            )

    # Get reference factor
    ref_factor = expr_dict[reference_key].factor
    assert ref_factor is not None  # Already validated above

    logger.info("")
    logger.info("=" * 50)
    logger.info(f'Terms after normalization with "{reference_key}"')
    logger.info("=" * 50)

    # Print reference factor information
    ref_factor_sym = expand(ref_factor, [q.symbol for q in quantities])
    ref_factor_expr = to_base_units(expand(ref_factor, [q.expr for q in quantities]))

    logger.info(f"Reference factor from '{reference_key}':")
    ref_rows = [[reference_key, str(ref_factor_sym), ref_factor_expr.simplify().n(4)]]
    print_table(ref_rows, ["Term", "Factor", "Value (in base units)"])
    logger.info("")

    # Create table showing original and normalized factors
    rows = []
    normalized_dict = {}

    for key, factorized_expr in expr_dict.items():
        original_factor = factorized_expr.factor
        assert original_factor is not None  # Already validated above
        normalized_factor = original_factor - ref_factor

        ratio_sym = expand(normalized_factor, [q.symbol for q in quantities])
        ratio_expr = to_base_units(expand(normalized_factor, [q.expr for q in quantities]))

        rows.append([key, str(ratio_sym), ratio_expr.simplify().n(4)])

        # Create normalized expression by applying the ratio F_original/F_reference.
        # The factorized expression has had F_original stripped out, so multiplying
        # by this ratio leaves the term scaled relative to the reference factor.
        ratio_value = expand(normalized_factor, quantities)
        if isinstance(factorized_expr.expr, Form):
            normalized_dict[key] = map_integrands(lambda x: x * ratio_value, factorized_expr.expr)
        else:
            normalized_dict[key] = factorized_expr.expr * ratio_value

    print_table(rows, ["Term", "Factor", "Value (in base units)"])
    logger.info("=" * 50)

    return normalized_dict
