"""Physical units and dimensional analysis for UFL expressions and forms.

The package attaches sympy units to UFL constants (:class:`Quantity`), substitutes them
into expressions and forms (:func:`transform`), pulls the resulting scales back out
(:func:`factorize`, :func:`normalize`) and reports the dimensionless groups of a problem
(:func:`buckingham_pi_analysis`).

It depends on UFL only, so quantities carry no values. Bind values by mixing
:class:`QuantityMixin` into the constant type of a form compiler backend.
"""

from ufl_units.analysis import buckingham_pi_analysis, dimension_matrix
from ufl_units.factorize import (
    FactorizedExpr,
    QuantityFactorizer,
    expand,
    factorize,
    get_dimension,
    normalize,
)
from ufl_units.quantity import Quantity, QuantityMixin, get_factor, to_base_units
from ufl_units.table import print_table
from ufl_units.transform import UnitTransformer, collect_quantities, transform

__all__ = [
    "FactorizedExpr",
    "Quantity",
    "QuantityFactorizer",
    "QuantityMixin",
    "UnitTransformer",
    "buckingham_pi_analysis",
    "collect_quantities",
    "dimension_matrix",
    "expand",
    "factorize",
    "get_dimension",
    "get_factor",
    "normalize",
    "print_table",
    "to_base_units",
    "transform",
]
