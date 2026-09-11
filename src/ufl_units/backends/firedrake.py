"""Quantity backed by a Firedrake constant."""

import firedrake
import sympy as sy
from sympy.physics.units import UnitSystem
from ufl_units.quantity import QuantityMixin


class Quantity(firedrake.Constant, QuantityMixin):
    """Quantity whose factor is held by a :class:`firedrake.Constant`.

    The Firedrake base comes first because form compilers resolve a terminal's handler by
    walking ``__bases__[0]``; here that is the ``firedrake_constant`` handler name
    Firedrake registers on its own constant type. The two dunders it shadows are re-bound
    below.

    Unlike DOLFINx, a Firedrake constant is not tied to a mesh, so the constructor takes
    no domain; the measure the quantity is integrated against carries the domain instead.

    Parameters
    ----------
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
        scale: float | int,
        unit: sy.Expr | None,
        symbol: str | sy.Symbol,
        unit_system: UnitSystem = sy.physics.units.si.SI,
        **kwargs,
    ):
        self._init_units(scale, unit, symbol, unit_system)
        firedrake.Constant.__init__(self, self.factor, **kwargs)

    def _update_value(self, factor: float) -> None:
        """Keep the constant's value in sync with the scale."""
        self.assign(factor)

    __str__ = QuantityMixin.__str__
    __repr__ = QuantityMixin.__repr__
