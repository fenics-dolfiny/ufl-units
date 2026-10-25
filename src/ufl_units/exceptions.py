"""Errors raised when an expression cannot carry consistent units."""


class InconsistentDimensions(Exception):
    """Operands or terms that must share a dimension carry different ones.

    Adding a length to a time is meaningless, so no factor can be pulled out of it.
    """
