"""Quantity types bound to the constants of a form compiler backend.

The package itself depends on UFL only, so its quantities carry no values. Each module
here mixes :class:`ufl_units.QuantityMixin` into the constant type of one backend and is
importable only when that backend is installed. Import the module explicitly, e.g.
``from ufl_units.backends.dolfinx import Quantity``; nothing here is re-exported from
:mod:`ufl_units`.
"""
