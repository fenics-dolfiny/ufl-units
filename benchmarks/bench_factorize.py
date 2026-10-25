"""Time `factorize` against the number of nodes in the expression DAG."""

import time

import basix.ufl
import ufl
from ufl.corealg.traversal import unique_pre_traversal

import sympy.physics.units as syu
from ufl_units import Quantity, factorize

mesh = ufl.Mesh(basix.ufl.element("P", "triangle", 1, shape=(2,)))
V = ufl.FunctionSpace(mesh, basix.ufl.element("P", "triangle", 1))
u = ufl.Coefficient(V)

length = Quantity(mesh, 1.0, syu.meter, "L")
time_ = Quantity(mesh, 1.0, syu.second, "T")
quantities = [length, time_]


def build(n: int) -> ufl.core.expr.Expr:
    """Chain of `n` dimensionally consistent steps, each adding a few nodes."""
    expr = u * length / time_
    for _ in range(n):
        expr = expr * time_ / time_ + u * length / time_
    return expr


print(f"{'nodes':>10} {'time [s]':>10}")
for n in [10, 100, 1_000, 10_000]:
    expr = build(n)
    nodes = sum(1 for _ in unique_pre_traversal(expr))
    start = time.perf_counter()
    factorize(expr, quantities)
    print(f"{nodes:>10} {time.perf_counter() - start:>10.4f}")
