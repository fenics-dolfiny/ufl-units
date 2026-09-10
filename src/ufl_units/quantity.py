import logging
from typing import TYPE_CHECKING

import ufl

import numpy as np
import sympy as sy
from sympy.physics.units import UnitSystem
from sympy.physics.units.dimensions import Dimension

logger = logging.getLogger(__name__)


class QuantityMixin:
    """Unit-carrying behaviour of a scalar quantity, independent of any value backend.

    A quantity couples a numerical ``scale`` with a sympy ``unit`` and a symbolic
    ``symbol`` used for reporting. The scale/unit pair is reduced to a plain
    ``factor`` expressed in the base units of ``unit_system``, which is the number a
    value backend is expected to expose to the form compiler.

    This class holds no reference to a UFL node, so it can be mixed into any
    constant-like class. Concrete flavours combine it with a UFL terminal, see
    :class:`Quantity` for the symbolic one. Backends that carry a mutable value
    override :meth:`_update_value` to keep that value in sync with ``scale``.
    """

    _scale: float | int
    _unit: sy.Expr
    _symbol: sy.Symbol
    _unit_system: UnitSystem
    _factor: float
    _dimension: Dimension
    _dimensional_dependencies: dict[Dimension, int]

    if TYPE_CHECKING:
        # Provided by the UFL terminal this mixin is combined with, declared read-only
        # here so that it does not clash with the property of that terminal.
        @property
        def ufl_shape(self) -> tuple[int, ...]: ...

    def _init_units(
        self,
        scale: float | int,
        unit: sy.Expr | None,
        symbol: str | sy.Symbol,
        unit_system: UnitSystem = sy.physics.units.si.SI,
    ) -> None:
        """Validate and store the scale/unit/symbol triple, then reduce it to a factor."""
        if not isinstance(scale, int | float):
            raise TypeError(f"Scale must be a numeric type, got {type(scale).__name__}.")

        if np.asarray(scale).shape != ():
            raise ValueError("Quantity supports only scalar values.")

        if unit is None:
            unit = sy.sympify(1)

        if not isinstance(unit, sy.Expr):
            raise TypeError(f"Unit must be a sympy expression, got {type(unit).__name__}.")

        self._scale = scale
        self._unit = unit

        if not isinstance(symbol, str | sy.Symbol):
            raise TypeError(
                f"Symbol must be a string or sympy.Symbol, got {type(symbol).__name__}."
            )

        self._symbol = (
            sy.Symbol(symbol, positive=True, real=True) if isinstance(symbol, str) else symbol
        )

        self._unit_system = unit_system

        self._factor, self._dimension, self._dimensional_dependencies = get_factor(
            self._scale, self._unit, self._unit_system
        )

    def _update_value(self, factor: float) -> None:
        """Propagate a recomputed factor to the value backend.

        The default implementation does nothing, which is correct for purely symbolic
        quantities. Backends holding a mutable value must override this.
        """

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    @property
    def dimensional_dependencies(self) -> dict[Dimension, int]:
        return self._dimensional_dependencies

    @property
    def factor(self) -> float:
        """Numerical value of ``scale * unit`` expressed in the base units."""
        return float(self._factor)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        self._factor, self._dimension, self._dimensional_dependencies = get_factor(
            self._scale, self._unit, self._unit_system
        )
        self._update_value(self.factor)

    @property
    def expr(self) -> sy.Expr:
        return self._scale * self._unit

    @property
    def unit(self) -> sy.Expr:
        return self._unit

    @property
    def unit_system(self) -> UnitSystem:
        return self._unit_system

    @property
    def symbol(self) -> sy.Symbol:
        return self._symbol

    def __str__(self):
        return str(self._symbol)

    def __repr__(self):
        return (
            f"{type(self).__name__}(scale={self._scale}, unit={self._unit}, "
            f"symbol={self._symbol}, unit_system={self._unit_system})"
        )


class Quantity(ufl.Constant, QuantityMixin):
    """Symbolic quantity, a UFL constant on ``domain`` carrying units.

    Note the order of the bases: form compilers resolve the handler of a terminal by
    walking ``__bases__[0]`` up to a type they know, so the UFL base has to come first.
    The two dunders the UFL base shadows are re-bound below.

    Parameters
    ----------
    domain
        Mesh the constant is defined on.
    scale
        Numerical scaling factor.
    unit
        Sympy unit expression, or ``None`` for a dimensionless quantity.
    symbol
        Symbol used when reporting factors and dimensionless groups.
    unit_system
        Unit system the scale/unit pair is reduced in. Default is SI.

    """

    def __init__(
        self,
        domain,
        scale: float | int,
        unit: sy.Expr | None,
        symbol: str | sy.Symbol,
        unit_system: UnitSystem = sy.physics.units.si.SI,
        **kwargs,
    ):
        self._init_units(scale, unit, symbol, unit_system)
        ufl.Constant.__init__(self, domain, **kwargs)

    __str__ = QuantityMixin.__str__
    __repr__ = QuantityMixin.__repr__


def to_base_units(expr: sy.Expr, unit_system: UnitSystem = sy.physics.units.systems.SI) -> sy.Expr:
    """Convert expression to base units of the given unit system.

    Parameters
    ----------
    expr
        Expression containing units to be converted.
    unit_system
        Unit system to use for conversion. Default is SI system.

    Returns
    -------
        Expression with units converted to base units of the specified system.

    """
    base_units = unit_system._base_units
    return sy.physics.units.convert_to(expr, base_units)


def get_factor(
    scale: float | int,
    unit: sy.physics.units.Unit,
    unit_system: sy.physics.units.UnitSystem = sy.physics.units.si.SI,
) -> tuple[
    float, sy.physics.units.dimensions.Dimension, dict[sy.physics.units.dimensions.Dimension, int]
]:
    """Extract numerical factor, dimension, and dimensional dependencies from a scaled unit.

    Parameters
    ----------
    scale
        Numerical scaling factor
    unit
        Unit to analyze
    unit_system
        Unit system for conversion (default: SI)

    Returns
    -------
        (factor, dimension, dimensional_dependencies)

    """
    base_units = unit_system._base_units
    base_value = sy.physics.units.convert_to(scale * unit, base_units)
    strip_map = {unit: 1 for unit in base_units}
    factor = base_value.subs(strip_map)

    _, dimension = unit_system._collect_factor_and_dimension(unit)
    dimensional_dependencies = unit_system.get_dimension_system().get_dimensional_dependencies(
        dimension
    )

    if not factor.is_number:
        raise ValueError(f"Cannot convert {scale * unit} to base units {base_units}.")

    return factor, dimension, dimensional_dependencies
