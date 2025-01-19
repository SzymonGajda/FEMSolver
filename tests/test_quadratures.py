import pytest
import numpy as np
from src.geom import Point2d, get_triangle_area
from src.quadratures import evaluate_2d_function_at_midpoint, midpoint_2d_quadrature, trapezoidal_1d_quadrature


def test_evaluate_function_at_midpoint_linear_function():
    def linear_function(point: Point2d) -> float:
        return 2 * point[0] + 3 * point[1]

    point1 = (0, 0)
    point2 = (4, 4)
    result = evaluate_2d_function_at_midpoint(linear_function, point1, point2)
    expected = 10.0
    assert result == pytest.approx(expected)


def test_evaluate_function_at_midpoint_constant_function():
    def constant_function(point: Point2d) -> float:
        return 5.0

    point1 = (2, 3)
    point2 = (6, 7)
    result = evaluate_2d_function_at_midpoint(constant_function, point1, point2)
    assert result == pytest.approx(5.0)


def test_evaluate_function_at_midpoint_quadratic_function():
    def quadratic_function(point: Point2d) -> float:
        return point[0] ** 2 + point[1] ** 2

    point1 = (1, 1)
    point2 = (3, 3)
    result = evaluate_2d_function_at_midpoint(quadratic_function, point1, point2)
    expected = 8.0
    assert result == pytest.approx(expected)


def test_evaluate_function_at_midpoint_negative_coordinates():
    def linear_function(point: Point2d) -> float:
        return 4 * point[0] - 3 * point[1]

    point1 = (-4, -2)
    point2 = (0, -6)
    result = evaluate_2d_function_at_midpoint(linear_function, point1, point2)
    expected = 4.0
    assert result == pytest.approx(expected)


def test_evaluate_function_at_midpoint_zero_midpoint():
    def function(point: Point2d) -> float:
        return point[0] + point[1]

    point1 = (-1, 1)
    point2 = (1, -1)
    result = evaluate_2d_function_at_midpoint(function, point1, point2)
    expected = 0.0
    assert result == pytest.approx(expected)

def sample_function(point: Point2d) -> float:
    return point[0] + point[1]

def test_midpoint_quadrature_regular_triangle():
    point1 = (0, 0)
    point2 = (1, 0)
    point3 = (0, 1)
    result = midpoint_2d_quadrature(sample_function, point1, point2, point3)
    expected_value = 1/3
    assert pytest.approx(result, rel=1e-6) == expected_value

def test_midpoint_quadrature_constant_function():
    def constant_function(_: Point2d) -> float:
        return 5
    point1 = (0, 0)
    point2 = (2, 0)
    point3 = (0, 2)
    result = midpoint_2d_quadrature(constant_function, point1, point2, point3)
    expected_area = get_triangle_area(point1, point2, point3)
    expected_value = expected_area * 5
    assert pytest.approx(result, rel=1e-6) == expected_value

def test_midpoint_quadrature_zero_area_triangle():
    point1 = (0, 0)
    point2 = (1, 1)
    point3 = (2, 2)
    result = midpoint_2d_quadrature(sample_function, point1, point2, point3)
    assert result == 0.0

def test_trapezoidal_constant_function():
    fun = lambda x: 5
    result = trapezoidal_1d_quadrature(fun, 0, 10)
    assert np.isclose(result, 50)

def test_trapezoidal_linear_function():
    fun = lambda x: 2 * x
    result = trapezoidal_1d_quadrature(fun, 0, 5)
    expected = (fun(0) + fun(5)) * 5 / 2
    assert np.isclose(result, expected)

def test_trapezoidal_quadratic_function():
    fun = lambda x: x**2
    result = trapezoidal_1d_quadrature(fun, 0, 2)
    expected = (fun(0) + fun(2)) * 2 / 2
    assert np.isclose(result, expected)

def test_trapezoidal_negative_interval():
    fun = lambda x: x + 1
    result = trapezoidal_1d_quadrature(fun, 3, 1)
    expected = (fun(3) + fun(1)) * (1 - 3) / 2
    assert np.isclose(result, expected)

def test_trapezoidal_zero_interval():
    fun = lambda x: np.sin(x)
    result = trapezoidal_1d_quadrature(fun, 3, 3)
    assert np.isclose(result, 0)