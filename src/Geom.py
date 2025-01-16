import numpy as np


def is_inside_triangle(A, B, C, P):
    denominator = ((B[1] - C[1]) * (A[0] - C[0]) +
                   (C[0] - B[0]) * (A[1] - C[1]))

    a = ((B[1] - C[1]) * (P[0] - C[0]) +
         (C[0] - B[0]) * (P[1] - C[1])) / denominator
    b = ((C[1] - A[1]) * (P[0] - C[0]) +
         (A[0] - C[0]) * (P[1] - C[1])) / denominator
    c = 1 - a - b

    return np.where(np.logical_and(
        np.logical_and(np.logical_or(np.isclose(a, 0.), a > 0), np.logical_or(np.isclose(b, 0.), b > 0.)),
        np.logical_or(np.isclose(c, 0.), c > 0.)), True, False)


def get_triangle_area(A, B, C):
    return abs((A[0] - C[0]) * (B[1] - A[1]) - (A[0] - B[0]) * (C[1] - A[1])) / 2
