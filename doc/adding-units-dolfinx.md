# Adding units to a DOLFINx code

You have a working DOLFINx solver and want to attach units to it. This takes four steps,
shown here on the
[elasticity demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_elasticity.html)
of DOLFINx. The complete result is [demo/elasticity.py](../demo/elasticity.py). For
Firedrake, see [adding-units-firedrake.md](adding-units-firedrake.md).

## The original code

```python
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
    (16, 16, 16),
    CellType.tetrahedron,
)

ω, ρ = 300.0, 10.0
x = ufl.SpatialCoordinate(msh)
f = ufl.as_vector((ρ * ω**2 * x[0], ρ * ω**2 * x[1], 0.0))

E = 1.0e9
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))


def σ(v):
    return 2.0 * μ * ufl.sym(ufl.grad(v)) + λ * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v))


V = functionspace(msh, ("Lagrange", 1, (gdim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form(ufl.inner(σ(u), ufl.grad(v)) * ufl.dx)
L = form(ufl.inner(f, v) * ufl.dx)
```

`E` is a Young's modulus, `ρ` a density and `ω` an angular velocity, but the code does not
say in which units. We read the box in metres and everything else in SI units.

## 1. Replace constants by quantities

A `Quantity` is a DOLFINx constant with a unit. It takes the mesh, like `fem.Constant`,
and replaces the number one for one:

```python
import sympy.physics.units as syu
from ufl_units.backends.dolfinx import Quantity

ω = Quantity(msh, 300.0, 1 / syu.second, "ω")
ρ = Quantity(msh, 10.0, syu.kilogram / syu.meter**3, "rho")
E = Quantity(msh, 1.0e9, syu.pascal, "E")
```

> [!WARNING]
> The number a `Quantity` holds, which the solver sees and `q.factor` returns, is its value
> in the **base units of its unit system**, SI by default. The scale and unit you pass are
> converted: `Quantity(1, syu.kilometer, "L")` holds `1000.0`, and
> `Quantity(1, syu.minute, "dt")` holds `60.0`. If your code works in other units, say a mesh
> in millimetres and moduli in GPa, a quantity holds a different number than the constant it
> replaces, and the results change. Either keep the code in SI base units, or pass
> `unit_system=` a system whose base units are the ones your code uses.

The Poisson ratio `ν` is dimensionless and stays a number. `μ` and `λ` are computed from `E`,
so they carry its units without further changes.

Also add one quantity for the unit of the mesh and one for each unknown:

```python
l_ref = Quantity(msh, 1.0, syu.meter, "l_ref")
u_ref = Quantity(msh, 1.0, syu.meter, "u_ref")
```

`ufl_units` works on UFL forms, so analyse `a` and `L` before passing them to `form`:

```python
a = ufl.inner(σ(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
# ... steps 2 and 3 ...
a = form(a)
L = form(L)
```

## 2. Write the mapping

The mapping says what the numbers in the mesh and in the functions stand for:

```python
mapping = {msh.ufl_domain(): l_ref, u: u_ref * u, v: u_ref * v}
```

`u: u_ref * u` means "the displacement is `u_ref` times the number stored in `u`". The mesh
key is the UFL domain, `msh.ufl_domain()`. Map every function that holds a physical quantity,
and the test functions too. Scale a test function like the unknown it tests: `v` is a virtual
displacement, so both forms become a virtual work. A function that is not mapped is
dimensionless.

## 3. Analyse the form

Print the tables through the `ufl_units` logger:

```python
import logging

import ufl_units

logging.basicConfig(format="%(message)s")
logging.getLogger("ufl_units").setLevel(logging.INFO)
```

**Dimensionless groups.** The list of quantities is analysed in order, so put the reference
scales first:

```python
quantities = [l_ref, ω, E, ρ, u_ref]
ufl_units.buckingham_pi_analysis(quantities)
```

```
Group | Expression          | Value
------+---------------------+-------
Pi_1  | l_ref**2*rho*ω**2/E | 0.0009
Pi_2  | u_ref/l_ref         | 1
```

`Pi_1` is the strain the rotation causes.

**Consistency.** Both forms must have the same units, here an energy:

```python
dimension = ufl_units.get_dimension(a - L, quantities, mapping=mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(dimension, syu.energy)
```

If they don't, `InconsistentDimensions` is raised. With `ω, ρ = 300.0, 10.0` left plain
numbers:

```
Inconsistent dimensions across integrals in Form.
Scales: E*l_ref*u_ref**2 != l_ref**4*u_ref.
Dimension(length**3*pressure) != Dimension(length**5).
```

**Relative size of the terms.** Put the terms in a dictionary, factorize it, and normalize by
one term:

```python
terms = ufl_units.factorize({"a": a, "L": L}, quantities, mapping=mapping)
terms = ufl_units.normalize(terms, "a", quantities)
```

```
Term | Factor                      | Value (in base units)
-----+-----------------------------+----------------------
a    | 1                           | 1.000
L    | l_ref**3*rho*ω**2/(E*u_ref) | 0.0009000
```

The load is 9·10⁻⁴ of the stiffness at a displacement of 1 m, so the displacements are about
a millimetre.

## 4. Solve

Compile and solve `a` and `L` as before; they are the same forms, so the solution is
unchanged. Python code that needs a constant's value as a number uses `q.factor`, the value
in base units.

## Tips

- Numbers left in the form are dimensionless. Turn every physical constant into a quantity.
- Arguments of `exp`, `sin`, `ln` etc. must be dimensionless.
- Only scalar quantities are supported: write `g * ufl.as_vector([0, -1])`, with `g` a
  quantity.
- Setting `q.scale` changes the value used by the solver, without recompiling the form.
- More complete examples: [demo/navier_stokes.py](../demo/navier_stokes.py) and
  [demo/neo_hooke.py](../demo/neo_hooke.py).
