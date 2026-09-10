import logging
from typing import Any

logger = logging.getLogger(__name__)


def print_table(rows: list[list[Any]], headers: list[str], outstream=logger.info) -> None:
    """Pretty-print a list of rows as a table with the given headers.

    Parameters
    ----------
    rows
        List of rows, each row is a list of values.
    headers
        List of column header names.
    outstream
        Output stream function, defaults to `logger.info`.

    """
    # Convert all items to strings
    str_rows = [[str(item) for item in row] for row in rows]

    # Widest entry per column. The single list argument to max keeps a table
    # without rows working, falling back to the header widths.
    col_widths = [
        max([len(headers[i]), *(len(row[i]) for row in str_rows)]) for i in range(len(headers))
    ]
    # Build format strings
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    separator = "-+-".join("-" * w for w in col_widths)

    outstream(row_fmt.format(*headers))
    outstream(separator)

    for row in str_rows:
        outstream(row_fmt.format(*row))
