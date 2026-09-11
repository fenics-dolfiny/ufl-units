import basix.ufl
import ufl

import pytest
import sympy.physics.units as syu


@pytest.fixture(scope="module")
def mesh():
    return ufl.Mesh(basix.ufl.element("P", "triangle", 1, shape=(2,)))


@pytest.fixture(scope="module")
def V(mesh):
    return ufl.FunctionSpace(mesh, basix.ufl.element("P", "triangle", 1))


@pytest.fixture(scope="module")
def W(mesh):
    return ufl.FunctionSpace(mesh, basix.ufl.element("P", "triangle", 1, shape=(2,)))


@pytest.fixture(scope="session")
def structural():
    """A non-SI unit system with GPa, metre and second as base units.

    Session scope: registering a unit mutates sympy's global scale factors.
    """
    GPa = syu.Quantity("gigapascal", abbrev="GPa")
    GPa.set_global_dimension(syu.pressure)
    GPa.set_global_relative_scale_factor(1e9, syu.pascal)

    dimension_system = syu.DimensionSystem(base_dims=[syu.pressure, syu.length, syu.time])
    unit_system = syu.UnitSystem(
        base_units=[GPa, syu.meter, syu.second], dimension_system=dimension_system
    )
    return unit_system, GPa
