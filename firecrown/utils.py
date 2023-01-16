"""Some utility functions for patterns common in Firecrown.
"""


def upper_triangle_indices(n: int):  # pylint: disable-msg=invalid-name
    """generator that yields a sequence of tuples that carry the indices for an
    (n x n) upper-triangular matrix. This is a replacement for the nested loops:

    for i in range(n):
      for j in range(i, n):
        ...
    """
    for i in range(n):
        for j in range(i, n):
            yield i, j
