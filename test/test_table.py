import sympy as sy
from ufl_units import print_table


def test_alignment():
    """Every column is padded to its widest entry, header included."""
    lines: list[str] = []
    print_table([["a", "1"], ["bbbb", "22"]], ["h1", "h2"], outstream=lines.append)

    header, separator, *rows = lines
    assert len(lines) == 4
    assert len(separator) == len(header)
    assert all(len(row) == len(header) for row in rows)
    assert header.startswith("h1")
    assert rows[0].startswith("a   ")  # padded to the width of "bbbb"


def test_no_rows():
    """A table with no rows still prints its header, sized to the headers."""
    lines: list[str] = []
    print_table([], ["Group", "Value"], outstream=lines.append)

    assert lines == ["Group | Value", "------+------"]


def test_stringify():
    """Non-string cells are converted, so sympy expressions can be tabulated."""
    lines: list[str] = []
    print_table([[1, sy.Symbol("x") ** 2]], ["n", "expr"], outstream=lines.append)

    assert "1" in lines[-1]
    assert "x**2" in lines[-1]
