"""Module provides implementations of different quadratures algorithms."""
import scipy
from collections.abc import Callable

from src.geom import Point2d, get_triangle_area


def evaluate_2d_function_at_midpoint(
    fun: Callable[[Point2d], float], point1: Point2d, point2: Point2d
) -> float:
    """
    Calculate the value of a provided function at the midpoint between two 2D points.

    The function takes three arguments: a callable function that processes a 2D
    point and returns a float, and two 2D points. It computes the midpoint of the
    two given points and evaluates the function at this midpoint, returning the
    result.

    :param fun: Callable function that takes a 2D point as input and returns a float.
    :param point1: First 2D point represented as a tuple of two floats.
    :param point2: Second 2D point represented as a tuple of two floats.
    :return: The value of the function evaluated at the midpoint between `point1`
        and `point2`.
    """
    midpoint = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
    return fun(midpoint)


def midpoint_2d_quadrature(
    fun: Callable[[Point2d], float], point1: Point2d, point2: Point2d, point3: Point2d
) -> float:
    """
    Calculate the integral of a function over a triangle using the midpoint quadrature rule.

    This function performs numerical integration over a triangle using the midpoint
    quadrature rule. The integral is approximated by evaluating the function at
    the midpoints of the triangle's edges and combining these values with the
    triangle's area.

    :param fun: Callable that represents the function to integrate. The function
        must accept a Point2d object as its single argument.
    :param point1: First vertex of the triangle as a Point2d object.
    :param point2: Second vertex of the triangle
    :param point3: Third vertex of the triangle"""
    quadrature_result = 0.0
    triangle_area = get_triangle_area(point1, point2, point3)
    quadrature_result += evaluate_2d_function_at_midpoint(fun, point1, point2)
    quadrature_result += evaluate_2d_function_at_midpoint(fun, point2, point3)
    quadrature_result += evaluate_2d_function_at_midpoint(fun, point3, point1)
    quadrature_result *= triangle_area / 3
    return quadrature_result

def trapezoidal_1d_quadrature(fun:Callable[[float], float], beginning :float, end:float)->float:
    interval_length = end - beginning
    value_at_beginning = fun(beginning)
    value_at_end = fun(end)
    print(value_at_beginning, value_at_end)
    return (value_at_beginning + value_at_end) * interval_length / 2

def scipy_1d_quadrature_wrapper(fun, beginning, end):
    return scipy.integrate.quad(fun, beginning, end)[0]