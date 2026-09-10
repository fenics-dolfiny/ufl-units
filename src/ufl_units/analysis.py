import logging
from typing import Any

import numpy as np
import sympy as sy
from sympy.physics.units import UnitSystem
from sympy.physics.units.dimensions import Dimension
from ufl_units.factorize import expand
from ufl_units.quantity import QuantityMixin, to_base_units
from ufl_units.table import print_table

logger = logging.getLogger(__name__)


def dimension_matrix(
    quantities: list[QuantityMixin],
    unit_system: UnitSystem = sy.physics.units.systems.SI,
) -> tuple[sy.Matrix, list[Dimension]]:
    """Build dimension matrix for Buckingham Pi analysis.

    Returns matrix where each column represents the exponents of a quantity with
    respect to the base dimensions.
    """
    dimsys = unit_system.get_dimension_system()
    base_dims: list[Dimension] = dimsys.base_dims

    # Build matrix: each column is the exponents of base_dims for a quantity
    rows: list[list[int]] = []
    for dim in base_dims:
        row: list[int] = []
        for q in quantities:
            deps: dict[Dimension, int] = q.dimensional_dependencies
            row.append(deps.get(dim, 0))
        rows.append(row)

    matrix = sy.Matrix(rows)
    return matrix, base_dims


def buckingham_pi_analysis(
    quantities: list[QuantityMixin],
    unit_system: UnitSystem = sy.physics.units.systems.SI,
    outlier_threshold: float = 1e-16,
) -> tuple[sy.Matrix, list[Dimension], list[sy.Matrix]]:
    """Perform Buckingham Pi analysis to find dimensionless groups.

    Parameters
    ----------
    quantities
        List of physical quantities to analyze
    unit_system
        Unit system for analysis (default: SI)
    outlier_threshold
        Threshold for detecting outlier values

    Returns
    -------
        Dimension matrix, base dimensions, and Pi groups

    """
    dim_matrix, base_dims = dimension_matrix(quantities, unit_system)

    logger.info("")
    logger.info("=" * 50)
    logger.info("Buckingham Pi Analysis")
    logger.info("=" * 50)

    # Print quantities in a table
    rows: list[list[Any]] = []
    for q in quantities:
        rows.append(
            [
                str(q.symbol),
                str(q.expr.simplify().n(4)),
                f"{to_base_units(q.expr, unit_system).simplify().n(4)}",
            ]
        )

    print_table(rows, ["Symbol", "Expression", "Value (in base units)"])
    logger.info("")
    logger.info(f"Dimension matrix ({len(base_dims)} x {len(quantities)}):")
    dim_array = np.array(dim_matrix).astype(float)

    # Create header with quantity symbols
    header = ["Dimension"] + [str(q.symbol) for q in quantities]

    # Create rows with dimension names and their exponents for each quantity
    rows = []
    for i, dim in enumerate(base_dims):
        row = [str(dim.name)] + [f"{int(dim_array[i, j])}" for j in range(len(quantities))]
        rows.append(row)

    print_table(rows, header)
    logger.info("")

    pi_groups = dim_matrix.nullspace()

    logger.info(f"Dimensionless groups ({len(pi_groups)}):")
    rows.clear()
    outliers: list[tuple[int, sy.Expr, float]] = []

    group_values = [
        np.prod(
            [q.factor ** float(pi_group[j]) for j, q in enumerate(quantities) if pi_group[j] != 0]
        )
        for pi_group in pi_groups
    ]
    # Guarded, as a set of dimensionally independent quantities has no groups to average
    average_group_value = np.mean(group_values) if group_values else np.nan

    for i, pi_group in enumerate(pi_groups):
        expr = sy.simplify(expand(pi_group, [q.symbol for q in quantities]))
        numerical_value = np.prod(
            [q.factor ** float(pi_group[j]) for j, q in enumerate(quantities) if pi_group[j] != 0]
        )

        # Check if numerical value is an outlier (too small or too large)
        is_outlier = (
            numerical_value < average_group_value * outlier_threshold
            or numerical_value > average_group_value / outlier_threshold
        )

        rows.append([f"Pi_{i + 1}", expr, f"{numerical_value:.3g}"])
        if is_outlier:
            outliers.append((i + 1, expr, float(numerical_value)))

    # Print the dimensionless groups in a table
    print_table(rows, ["Group", "Expression", "Value"])

    if outliers:
        logger.warning("\nThe following dimensionless groups have outlier values:")
        for pi_num, expr, value in outliers:
            logger.warning(f"  Pi_{pi_num:2d} = {value:.3g}")

    logger.info("=" * 50)
    return dim_matrix, base_dims, pi_groups
