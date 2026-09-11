import fractions
import logging
import math
from typing import NamedTuple, overload

from ufl.algorithms.map_integrands import map_integrands
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.form import Form

import numpy as np
import sympy as sy
from ufl_units.quantity import QuantityMixin, to_base_units
from ufl_units.table import print_table
from ufl_units.transform import transform

logger = logging.getLogger(__name__)


class FactorizedExpr(NamedTuple):
    expr: Expr | Form
    factor: np.ndarray | None


class QuantityFactorizer(MultiFunction):
    """Pull the quantities out of an expression, tracking their exponents per node.

    Each node gets an exponent vector over ``quantities``. In ``factorize`` mode the
    quantities are replaced by one, leaving the dimensionless remainder and the factor
    pulled out. ``check`` mode only verifies dimensional consistency of the operands.
    """

    factors: dict[Expr, np.ndarray]

    def __init__(self, quantities: list[QuantityMixin], mode="factorize"):
        self._quantities = quantities

        if mode not in ("factorize", "check"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'factorize' or 'check'.")

        self._mode = mode
        if any(q.ufl_shape != () for q in quantities):
            raise NotImplementedError("Factorization of non-scalar quantities is not implemented.")
        self.factors = {}
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
        self.factors[o] = float(n) * self.factors[a]
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

        self.factors.setdefault(o, np.zeros(len(self._quantities)))
        return self.reuse_if_untouched(o, *ops)

    def linear(self, o, *ops):
        # Linear nodes whose operands have the same factor (e.g. sum, grad, etc.)
        # can be assigned that factor. In check mode, only dimensional equivalence
        # is enforced, so the first operand is used as a dimension witness.
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        if len(factors) > 0:
            self._check_operands(o, factors[0])
            self.factors[o] = factors[0]

        return self.reuse_if_untouched(o, *ops)

    def inhomogeneous(self, o, *ops):
        # Inhomogeneous nodes (e.g. sin(x), exp(x), etc.) must have dimensionless
        # operands. In factorize mode, these operands must also have trivial factor,
        # because no multiplicative scale can be pulled out of a nonlinear function.
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        if len(factors) > 0:
            zero = np.zeros_like(factors[0])
            self._check_operands(o, zero)
            self.factors[o] = zero
        else:
            self.factors.setdefault(o, np.zeros(len(self._quantities)))

        return self.reuse_if_untouched(o, *ops)

    def multi_index(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def label(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def quantity_terminal(self, o, *ops):
        """Handle a terminal that may be one of the quantities being factorized out."""
        if o in self._quantities:
            idx = self._quantities.index(o)
            self.factors[o] = np.zeros(len(self._quantities))
            self.factors[o][idx] = 1
            return 1
        else:
            self.factors.setdefault(o, np.zeros(len(self._quantities)))
            return self.reuse_if_untouched(o, *ops)

    def expr_list(self, o, *ops):
        return self.reuse_if_untouched(o, *ops)

    def _check_operands(self, o, reference_factor):
        r"""Check that all operands of the expression are consistent with a reference factor.

        Verifies that all operands of a given expression have compatible units/dimensions
        and, in factorize mode, identical factors.
        """
        factors = [self.factors[op] for op in o.ufl_operands if op in self.factors]

        f0_expr = expand(reference_factor, [q.dimension for q in self._quantities]).simplify()
        f0_symbol = expand(reference_factor, [q.symbol for q in self._quantities])

        for fb in factors:
            fb_symbol = expand(fb, [q.symbol for q in self._quantities])

            dimsys = self._quantities[0].unit_system.get_dimension_system()
            fb_expr = expand(fb, [q.dimension for q in self._quantities]).simplify()
            if not dimsys.equivalent_dims(f0_expr, fb_expr):
                raise RuntimeError(
                    f"Inconsistent dimensions in operands of {o.__class__.__name__}.\n"
                    f"dim({f0_symbol}) != dim({fb_symbol}), (i.e. {f0_expr} != {fb_expr}).\n"
                )

            if self._mode == "factorize" and not np.allclose(reference_factor, fb):
                raise RuntimeError(
                    f"Inconsistent factors in operands of {o.__class__.__name__}.\n"
                    f"{f0_symbol} != {fb_symbol}."
                )

    terminal = independent

    # The terminal a backend builds a quantity from: ufl.Constant (UFL, DOLFINx), a
    # Coefficient (function on a real space), or Firedrake's own registered type.
    constant = quantity_terminal
    coefficient = quantity_terminal
    firedrake_constant = quantity_terminal

    sum = linear
    indexed = linear
    grad = linear
    div = linear
    conj = linear
    index_sum = linear
    transposed = linear
    deviatoric = linear
    sym = linear
    trace = linear
    variable = linear
    coefficient_derivative = linear
    component_tensor = linear

    variable_derivative = division

    inner = product
    dot = product
    cross = product

    expr = inhomogeneous


def _root_factor(factorizer: QuantityFactorizer, root_expr: Expr) -> np.ndarray | None:
    """Return the factor of the mapped root expression, with a traversal-order fallback."""
    if root_expr in factorizer.factors:
        return factorizer.factors[root_expr]

    fallback_root_expr = next(reversed(factorizer.factors), None)
    return factorizer.factors[fallback_root_expr] if fallback_root_expr is not None else None


@overload
def factorize(
    expr: dict,
    quantities: list[QuantityMixin],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> dict[str, FactorizedExpr]: ...


@overload
def factorize(
    expr: Expr | Form,
    quantities: list[QuantityMixin],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> FactorizedExpr: ...


def factorize(
    expr: Expr | Form | dict,
    quantities: list[QuantityMixin],
    mode: str = "factorize",
    mapping: dict | None = None,
) -> FactorizedExpr | dict[str, FactorizedExpr]:
    """Factorize expressions, forms, or dictionaries to extract dimensional factors.

    Parameters
    ----------
    expr
        Expression, form, or dictionary to factorize
    quantities
        List of quantities to use for factorization
    mode
        Factorization mode: "factorize" or "check" (default: "factorize")
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
        return {key: factorize(value, quantities, mode=mode) for key, value in expr.items()}
    if mode not in ("factorize", "check"):
        raise RuntimeError(f"{mode} is not a valid factorisation mode.")
    if isinstance(expr, Form):
        factorized_integrals = []
        factors = []
        for integral in expr.integrals():
            factorizer = QuantityFactorizer(quantities, mode=mode)
            integrand = integral.integrand()
            factorized_integrand = map_expr_dag(factorizer, integrand)
            factorized_integrals.append(integral.reconstruct(factorized_integrand))
            factors.append(_root_factor(factorizer, integrand))

        for i in range(len(factors) - 1):
            unit_system = quantities[0].unit_system
            dimsys = unit_system.get_dimension_system()
            fa_expr = expand(factors[i], [q.dimension for q in quantities]).simplify()  # type: ignore[arg-type]
            fb_expr = expand(factors[i + 1], [q.dimension for q in quantities]).simplify()  # type: ignore[arg-type]
            fa_symbol = expand(factors[i], [q.symbol for q in quantities])  # type: ignore[arg-type]
            fb_symbol = expand(factors[i + 1], [q.symbol for q in quantities])  # type: ignore[arg-type]
            if dimsys.equivalent_dims(fa_expr, fb_expr) is False:
                raise RuntimeError(
                    "Inconsistent dimensions across integrals in Form. \n"
                    f"Scales: {fa_symbol} != {fb_symbol}. \n"
                    f"{fa_expr} != {fb_expr}."
                )

            if mode == "factorize":
                if not np.allclose(factors[i], factors[i + 1]):  # type: ignore[arg-type]
                    raise RuntimeError(
                        "Inconsistent factors across integrals in Form. \n"
                        f"{fa_symbol} != {fb_symbol}."
                    )

        factorized_expression = Form(factorized_integrals)
        root_factor = factors[0] if factors else None
    elif isinstance(expr, Expr):
        factorizer = QuantityFactorizer(quantities, mode=mode)
        factorized_expression = map_expr_dag(factorizer, expr)
        root_factor = _root_factor(factorizer, expr)
    else:
        raise TypeError(f"Unsupported type for factorization: {type(expr).__name__}")

    return FactorizedExpr(factorized_expression, root_factor)


def expand(factor: np.ndarray | list, quantities: list) -> sy.Expr:
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
    expr: Expr | Form, quantities: list[QuantityMixin], mapping: dict | None = None
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
    factor = factorize(expr, quantities, mode="check", mapping=mapping)
    assert isinstance(factor, FactorizedExpr)
    assert factor.factor is not None, "Factorized expression must have a non-None factor"

    unit = expand(factor.factor, [q.dimension for q in quantities]).simplify()
    return unit


def normalize(
    expr_dict: dict[str, FactorizedExpr],
    reference_key: str,
    quantities: list[QuantityMixin],
) -> dict[str, Expr | Form]:
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
