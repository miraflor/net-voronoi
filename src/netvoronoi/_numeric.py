"""Small numerical helpers shared by all modules.

Two kinds of help are collected here, so that each rule is written once:

* ``_tol`` is the single tolerance rule for comparing floating-point
  distances and positions;
* ``_first_appearance_ids`` and ``_count_distinct`` find equal rows in an
  array without a Python loop. The network builder uses them to find shared
  road vertices and repeated road segments, and the snapping code uses them
  to count distinct nearest locations.
"""

from __future__ import annotations

import numpy as np

_ABS_TOL = 1e-9
_REL_TOL = 1e-12


def _tol(*values):
    """Tolerance for comparing numbers of the magnitude of ``values``.

    Returns ``max(1e-9, 1e-12 * m)``, where ``m`` is the largest finite
    absolute value among the arguments (at least 1).

    Why two parts: a floating-point number of magnitude ``m`` is stored with a
    relative error of about ``1e-16``, and sums of many such numbers collect
    more error, so the relative part ``1e-12 * m`` grows with the numbers
    compared. The absolute part ``1e-9`` is a floor for small numbers. For
    example, two distances of about 1,000,000 m are treated as equal when they
    differ by at most 1e-6 m; two distances of about 10 m when they differ by
    at most 1e-9 m.

    Array arguments give an elementwise result; non-finite values are ignored.
    """
    scale = np.asarray(1.0)
    for value in values:
        a = np.abs(np.asarray(value, dtype=float))
        scale = np.maximum(scale, np.where(np.isfinite(a), a, 0.0))
    return np.maximum(_ABS_TOL, _REL_TOL * scale)


def _first_appearance_ids(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give equal rows the same id, numbering the ids in order of first appearance.

    ``rows`` is an ``(n, k)`` array. Two rows are equal when all ``k`` values
    are exactly equal (``0.0`` and ``-0.0`` count as equal). Returns
    ``(ids, first_rows)``: ``ids[i]`` is the id of row ``i``, and
    ``first_rows[j]`` is the position of the first row with id ``j``.

    Example::

        rows = [[5, 1],        ids        = [0, 1, 0, 2]
                [2, 0],        first_rows = [0, 1, 3]
                [5, 1],
                [7, 7]]

    Method, without a Python loop over rows:

    1. sort the rows, so that equal rows become neighbours (the row position
       is the last sort key, so equal rows keep their input order);
    2. mark each sorted row that differs from the row before it: it starts a
       new group of equal rows;
    3. number the groups, then renumber them so that the group whose first
       row comes first in the input gets id 0, the next one id 1, and so on.
    """
    n = len(rows)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Step 1. np.lexsort sorts by its last key first, so the columns are given
    # in reverse order; the row position np.arange(n) breaks remaining ties.
    columns = tuple(rows[:, j] for j in reversed(range(rows.shape[1])))
    order = np.lexsort((np.arange(n),) + columns)
    sorted_rows = rows[order]

    # Step 2. True where a sorted row starts a new group of equal rows.
    starts_group = np.r_[True, np.any(sorted_rows[1:] != sorted_rows[:-1], axis=1)]
    group_of_sorted_row = np.cumsum(starts_group) - 1  # 0, 0, 1, 2, 2, ...

    # Step 3. The first row of each group in ``order`` is also its first row
    # in the input, because equal rows kept their input order in step 1.
    first_row_of_group = order[starts_group]
    id_of_group = np.empty(len(first_row_of_group), dtype=np.int64)
    id_of_group[np.argsort(first_row_of_group)] = np.arange(len(first_row_of_group))

    ids = np.empty(n, dtype=np.int64)
    ids[order] = id_of_group[group_of_sorted_row]
    return ids, np.sort(first_row_of_group)


def _count_distinct(group: np.ndarray, key: np.ndarray, n_groups: int) -> np.ndarray:
    """For each group ``0 .. n_groups - 1``, count the distinct ``key`` values in it.

    Example: ``group = [0, 0, 0, 1]`` and ``key = [7, 7, 9, 7]`` give
    ``[2, 1]``: group 0 contains the keys 7 and 9, group 1 only the key 7.
    """
    pairs = np.c_[group, key]
    _, first_rows = _first_appearance_ids(pairs)  # one row per distinct (group, key) pair
    return np.bincount(pairs[first_rows, 0], minlength=n_groups)
