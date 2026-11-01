# Adding units to a Firedrake code

You have a working Firedrake solver and want to attach units to it. This takes four steps,
shown here on the [Burgers demo](https://www.firedrakeproject.org/demos/burgers.py.html) of
Firedrake. The complete result is [demo/burgers.py](../demo/burgers.py). For DOLFINx, see
[adding-units-dolfinx.md](adding-units-dolfinx.md).

## The original code

```python
from firedrake import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)
u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")
v = TestFunction(V)

nu = 0.0001
timestep = 1.0 / n

F = (
    inner((u - u_) / timestep, v) + inner(dot(u, nabla_grad(u)), v) + nu * inner(grad(u), grad(v))
) * dx
```

`nu` is a viscosity and `timestep` a time, but the code does not say in which units. We read
the domain as 1 m wide, the velocity in m/s and time in seconds.

## 1. Replace constants by quantities

A `Quantity` is a Firedrake constant with a unit. It replaces the number one for one:

```python
import sympy.physics.units as syu
from ufl_units.backends.firedrake import Quantity

nu = Quantity(0.0001, syu.meter**2 / syu.second, "nu")
timestep = Quantity(1.0 / n, syu.second, "dt")
```

> [!WARNING]
> The number a `Quantity` holds, which the solver sees and `q.factor` returns, is its value
> in the **base units of its unit system**, SI by default. The scale and unit you pass are
> converted: `Quantity(1, syu.kilometer, "L")` holds `1000.0`, and
> `Quantity(1, syu.minute, "dt")` holds `60.0`. If your code works in other units, say a mesh
> in millimetres and moduli in GPa, a quantity holds a different number than the constant it
> replaces, and the results change. Either keep the code in SI base units, or pass
> `unit_system=` a system whose base units are the ones your code uses.

Also add one quantity for the unit of the mesh and one for each unknown:

```python
l_ref = Quantity(1.0, syu.meter, "l_ref")
u_ref = Quantity(1.0, syu.meter / syu.second, "u_ref")
```

Finally, write the form as a dictionary of its terms. This one dictionary is analysed in
step 3 and summed back into the form you solve in step 4:

```python
terms = {
    "time": inner((u - u_) / timestep, v) * dx,
    "advection": inner(dot(u, nabla_grad(u)), v) * dx,
    "viscous": nu * inner(grad(u), grad(v)) * dx,
}
```

## 2. Write the mapping

The mapping says what the numbers in the mesh and in the functions stand for:

```python
mapping = {mesh: l_ref, u: u_ref * u, u_: u_ref * u_, v: u_ref * v}
```

`u: u_ref * u` means "the velocity is `u_ref` times the number stored in `u`". Map every
function that holds a physical quantity, and the test functions too. Scale a test function
like the unknown it tests: `v` tests the momentum equation, so it is a velocity, and each
term becomes a rate of work. A function that is not mapped is dimensionless.

## 3. Analyse the form

Print the tables through the `ufl_units` logger:

```python
from logging import INFO, basicConfig, getLogger

import ufl_units

basicConfig(format="%(message)s")
getLogger("ufl_units").setLevel(INFO)
```

**Dimensionless groups.** The list of quantities is analysed in order, so put the reference
scales first:

```python
quantities = [l_ref, u_ref, nu, timestep]
ufl_units.buckingham_pi_analysis(quantities)
```

```
Group | Expression       | Value
------+------------------+-------
Pi_1  | nu/(l_ref*u_ref) | 0.0001
Pi_2  | dt*u_ref/l_ref   | 0.0333
```

`Pi_1` is the inverse Reynolds number.

**Consistency.** Every term must have the same units, here an acceleration times a velocity
times an area:

```python
dimension = ufl_units.get_dimension(sum(terms.values(), Form([])), quantities, mapping=mapping)
assert syu.si.SI.get_dimension_system().equivalent_dims(dimension, syu.length**4 / syu.time**3)
```

If they don't, `InconsistentDimensions` is raised. With `nu = 0.0001` left a plain number:

```
Inconsistent dimensions across integrals in Form.
Scales: l_ref**2*u_ref**2/dt != u_ref**2.
Dimension(length**4/time**3) != Dimension(length**2/time**2).
```

**Normalization.** Factorize the terms and divide them all by the scale of one of them:

```python
terms = ufl_units.factorize(terms, quantities, mapping=mapping)
terms = ufl_units.normalize(terms, "advection", quantities)
```

```
Term      | Factor           | Value (in base units)
----------+------------------+----------------------
time      | l_ref/(dt*u_ref) | 30.00
advection | 1                | 1.000
viscous   | nu/(l_ref*u_ref) | 0.0001000
```

The table shows the size of each term relative to advection.

## 4. Solve

Sum the normalized terms back into the residual and solve it:

```python
F = sum(terms.values(), Form([]))

while t <= end:
    solve(F == 0, u)
    u_.assign(u)
    t += timestep.factor
```

`F` is the original residual divided by the scale of the advection term. The mapping
describes the numbers the code holds (a mesh in metres, velocities in m/s), so the solution
is the one of the original code. Python code that needs a constant's value as a number uses
`q.factor`, the value in base units, as in `t += timestep.factor`.

## Tips

- Numbers left in the form are dimensionless. Turn every physical constant into a quantity.
- Arguments of `exp`, `sin`, `ln` etc. must be dimensionless.
- Only scalar quantities are supported: write `g * as_vector([0, -1])`, with `g` a quantity.
- Setting `q.scale` changes the value used by the solver, without rebuilding the form.
- With `from firedrake import *`, import `basicConfig` and `getLogger` by name: Firedrake
  exports a different `logging`.
