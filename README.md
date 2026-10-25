# ufl-units

Physical units and dimensional analysis for [UFL](https://github.com/FEniCS/ufl) expressions and forms.

`ufl-units` lets you attach sympy units to the constants of a variational form, check that the form
is dimensionally consistent, non-dimensionalise it, and report the dimensionless groups that govern
the problem. It depends on UFL, sympy and numpy only — no form compiler and no assembly backend.

# Usage

```python
import basix.ufl
import ufl
import sympy.physics.units as syu

from ufl_units import Quantity

mesh = ufl.Mesh(basix.ufl.element("P", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, basix.ufl.element("P", "triangle", 1))
T, v = ufl.Coefficient(V), ufl.TestFunction(V)

kappa = Quantity(mesh, 1.0, syu.W / (syu.K * syu.m), "kappa")
l_ref = Quantity(mesh, 0.1, syu.m, "l_ref")
T_ref = Quantity(mesh, 300.0, syu.K, "T_ref")

form = ufl.inner(kappa * ufl.grad(T), ufl.grad(v)) * ufl.dx

# The mapping carries the reference length of the mesh and the scale of each field
mapping = {mesh: l_ref, T: T_ref * T, v: T_ref * v}

quantities = ufl_units.collect_quantities(form, mapping=mapping)
ufl_units.buckingham_pi_analysis(quantities)

print(ufl_units.get_dimension(form, quantities, mapping=mapping))

# Strip the dimensional scale off the form, leaving the dimensionless remainder. Terms
# may carry different scales: their ratios are the dimensionless groups of the problem,
# kept in the remainder and reported as exponent vectors over `quantities`
factorized = ufl_units.factorize(form, quantities, mapping=mapping)
print(factorized.expr, factorized.factor, factorized.dimless_numbers)
```

# Binding values

A `Quantity` is a `ufl.Constant` and therefore carries no value. To use quantities in an assembled
problem, mix `QuantityMixin` into the constant type of your backend and override `_update_value` so
that changes of `scale` reach the value:

```python
import dolfinx


class Quantity(dolfinx.fem.Constant, ufl_units.QuantityMixin):
    def __init__(self, mesh, scale, unit, symbol, unit_system=syu.si.SI, **kwargs):
        self._init_units(scale, unit, symbol, unit_system)
        dolfinx.fem.Constant.__init__(self, mesh, self.factor, **kwargs)

    def _update_value(self, factor):
        self.value = factor

    # The backend base shadows these, so re-bind them
    __str__ = ufl_units.QuantityMixin.__str__
    __repr__ = ufl_units.QuantityMixin.__repr__
```

The dolfinx flavour above ships as `ufl_units.backends.dolfinx`, and the Firedrake one as
`ufl_units.backends.firedrake`; each is importable wherever its backend is installed:

```python
from ufl_units.backends.dolfinx import Quantity  # Quantity(mesh, scale, unit, symbol)
from ufl_units.backends.firedrake import Quantity  # Quantity(scale, unit, symbol)
```

A Firedrake constant is not tied to a mesh, so that flavour takes no domain; the measure it is
integrated against carries the domain instead.

# Demos

The `demo/` folder holds the examples of the accompanying manuscript. Both need dolfinx, so
run them in the `ufl-units (dolfinx)` devcontainer (see below), or in any environment with
dolfinx and `ufl-units` installed:

```bash
cd demo
python3 navier_stokes.py
python3 neo_hooke.py
```

- `navier_stokes.py` prints the Buckingham Pi analysis and the normalized terms of the
  incompressible Navier-Stokes equations. It then sweeps the Euler number and writes the
  condition number of the lid-driven cavity Jacobian to `navier_stokes_condition_number.txt`,
  the data of the conditioning figure. The sweep is set with `--eu-min`, `--eu-max` and
  `--num-eu`, and the output file with `--output`. Run it on a single MPI rank, since the
  condition number is computed from the dense matrix.
- `neo_hooke.py` prints the Buckingham Pi analysis of Neo-Hooke hyperelasticity, the
  dimensionless numbers of the standard energy, and the normalized terms of its expansion in
  the Green-Lagrange strain.

# Development

Two devcontainers are provided, so that backend integration can be worked on against
either finite element library. Pick one when reopening the repository in a container:

- `ufl-units (dolfinx)`, on `dolfinx/dolfinx`
- `ufl-units (firedrake)`, on `firedrakeproject/firedrake-vanilla-default`

Both install `ufl-units` with `--no-deps`, leaving the image's own `fenics-ufl` in place.
The tests themselves need neither library, only `fenics-basix` for a mesh to hang
function spaces off, and run in any environment with `pytest -n auto test/`.

# License

`ufl-units` is free software distributed under the terms of the MIT License. See [LICENSE](LICENSE) for the full text.
