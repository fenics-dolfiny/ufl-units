"""Quantity backed by a dolfinx constant."""

import dolfinx

import numpy as np
import sympy as sy
from sympy.physics.units import UnitSystem
from ufl_units.quantity import QuantityMixin


class Quantity(dolfinx.fem.Constant, QuantityMixin):
    """Quantity whose factor is held by a :class:`dolfinx.fem.Constant`.

    The dolfinx base comes first because form compilers resolve a terminal's handler by
    walking ``__bases__[0]``. The two dunders it shadows are re-bound below.

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
        dolfinx.fem.Constant.__init__(self, domain, self.factor, **kwargs)

    def _update_value(self, factor: float) -> None:
        """Keep the constant's value in sync with the scale."""
        self.value = np.asarray(factor, dtype=self.dtype)

    __str__ = QuantityMixin.__str__
    __repr__ = QuantityMixin.__repr__
