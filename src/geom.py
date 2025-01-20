"""Module contains helper functions for 2D geometry."""

from typing import TypeAlias

import numpy as np

Point2d: TypeAlias = tuple[float, float]


def is_inside_triangle(
    vertex1: Point2d, vertex2: Point2d, vertex3: Point2d, point: Point2d
) -> np.typing.NDArray[np.bool_]:
    """
    Determine whether a given point lies inside a triangle defined by three vertices
    in a 2-dimensional space. The function uses a barycentric coordinate system to
    determine the position of the point with respect to the triangle. It returns
    an array of boolean values where True indicates the point lies inside the
    triangle or on its edges, and False otherwise.

    :param vertex1: The first vertex of the triangle, specified as a 2D point with x
                  and y coordinates.
    :param vertex2: The second vertex of the triangle, specified as a 2D point with
                  x and y coordinates.
    :param vertex3: The third vertex of the triangle, specified as a 2D point with x
                  and y coordinates.
    :param point: The point to be tested, also specified as a 2D point with x and
                  y coordinates.
    :return: A NumPy boolean array indicating whether the point lies inside the
             triangle or not.
    """

    def is_positive_or_close(value):
        """Check if a value is close to 0 or positive."""
        return np.logical_or(np.isclose(value, 0.0), value > 0.0)

    denominator = (vertex2[1] - vertex3[1]) * (vertex1[0] - vertex3[0]) + (
            vertex3[0] - vertex2[0]
    ) * (vertex1[1] - vertex3[1])

    alpha = (
                    (vertex2[1] - vertex3[1]) * (point[0] - vertex3[0])
                    + (vertex3[0] - vertex2[0]) * (point[1] - vertex3[1])
    ) / denominator

    beta = (
                   (vertex3[1] - vertex1[1]) * (point[0] - vertex3[0])
                   + (vertex1[0] - vertex3[0]) * (point[1] - vertex3[1])
    ) / denominator

    gamma = 1 - alpha - beta

    return np.logical_and.reduce(
        (
            is_positive_or_close(alpha),
            is_positive_or_close(beta),
            is_positive_or_close(gamma),
        )
    )


def get_triangle_area(vertex1: Point2d, vertex2: Point2d, vertex3: Point2d) -> float:
    """
    Calculate the area of a triangle given its vertices.

    :param vertex1: Coordinates of the first vertex of the triangle.
    :param vertex2: Coordinates of the second vertex of the triangle.
    :param vertex3: Coordinates of the third vertex of the triangle.
    :return: The absolute area of the triangle defined by the three vertices.
    """

    return (
        abs(
            (vertex1[0] - vertex3[0]) * (vertex2[1] - vertex1[1])
            - (vertex1[0] - vertex2[0]) * (vertex3[1] - vertex1[1])
        )
        / 2
    )