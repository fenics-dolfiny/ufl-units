import logging

import sympy as sy
import sympy.physics.units as syu
from ufl_units import Quantity, buckingham_pi_analysis, dimension_matrix


def test_dimension_matrix(mesh):
    """Each column holds the exponents of the base dimensions for one quantity."""
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol="L")
    time = Quantity(mesh, 1.0, unit=syu.second, symbol="T")
    velocity = Quantity(mesh, 1.0, unit=syu.meter / syu.second, symbol="V")

    matrix, base_dims = dimension_matrix([length, time, velocity])

    assert matrix.shape == (len(base_dims), 3)

    rows = {str(dim.name): i for i, dim in enumerate(base_dims)}
    assert matrix[rows["length"], 0] == 1  # L is a length
    assert matrix[rows["time"], 0] == 0
    assert matrix[rows["time"], 1] == 1  # T is a time
    assert matrix[rows["length"], 2] == 1  # V is length / time
    assert matrix[rows["time"], 2] == -1


def test_buckingham_pi(mesh):
    """Test Buckingham Pi analysis for dimensional reduction of three physical quantities."""
    # Define three quantities: length (L), time (T), and velocity (L/T)
    L = sy.Symbol("L")
    T = sy.Symbol("T")
    V = sy.Symbol("V")
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol=L)
    time = Quantity(mesh, 1.0, unit=syu.second, symbol=T)
    velocity = Quantity(mesh, 1.0, unit=syu.meter / syu.second, symbol=V)

    # Buckingham Pi theorem: for 3 quantities, 2 fundamental units (L, T), expect 1 Pi group
    _dim_matrix, _base_dims, pi_groups = buckingham_pi_analysis([length, time, velocity])
    assert isinstance(pi_groups, list)
    assert len(pi_groups) == 1
    # The pi group should be a sympy matrix representing the dimensionless combination
    assert pi_groups[0] is not None


def test_buckingham_pi_no_groups(mesh):
    """Dimensionally independent quantities admit no dimensionless group."""
    length = Quantity(mesh, 1.0, unit=syu.meter, symbol="L")
    time = Quantity(mesh, 1.0, unit=syu.second, symbol="T")

    _, _, pi_groups = buckingham_pi_analysis([length, time])
    assert pi_groups == []


def test_custom_unit_system(mesh, structural):
    """The analysis is carried out in the base dimensions of the given system."""
    unit_system, GPa = structural

    mu = Quantity(mesh, 100, GPa, "mu", unit_system)
    length = Quantity(mesh, 2.0, syu.meter, "L", unit_system)
    force = Quantity(mesh, 50, GPa * syu.meter**2, "F", unit_system)

    quantities = [mu, length, force]
    matrix, base_dims = dimension_matrix(quantities, unit_system)

    # The system declares three base dimensions, not the seven of SI
    assert {str(d.name) for d in base_dims} == {"pressure", "length", "time"}
    assert matrix.shape == (3, 3)

    _, analysed_dims, pi_groups = buckingham_pi_analysis(quantities, unit_system)
    assert analysed_dims == base_dims
    # F / (mu L^2) is the only dimensionless group
    assert len(pi_groups) == 1


def test_outliers(mesh, caplog):
    """Dimensionless groups far from the average are reported as outliers."""
    L = Quantity(mesh, 1.0, syu.meter, "L")
    T = Quantity(mesh, 1.0, syu.second, "T")
    V = Quantity(mesh, 1e6, syu.meter / syu.second, "V")
    A = Quantity(mesh, 1e-6, syu.meter / syu.second**2, "A")

    with caplog.at_level(logging.WARNING, logger="ufl_units.analysis"):
        _, _, pi_groups = buckingham_pi_analysis([L, T, V, A], outlier_threshold=0.5)

    assert len(pi_groups) == 2
    assert "outlier values" in caplog.text
