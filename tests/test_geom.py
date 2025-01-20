from src.geom import get_triangle_area, is_inside_triangle
import numpy as np

def test_point_inside_triangle():
    node1 = (0, 0)
    node2 = (5, 0)
    node3 = (0, 5)
    point = (2, 2)
    assert np.all(is_inside_triangle(node1, node2, node3, point))


def test_point_on_vertex_of_triangle():
    node1 = (0, 0)
    node2 = (3, 0)
    node3 = (0, 4)
    point = (0, 0)
    assert np.all(is_inside_triangle(node1, node2, node3, point))


def test_point_on_edge_of_triangle():
    node1 = (0, 0)
    node2 = (6, 0)
    node3 = (0, 6)
    point = (3, 0)
    assert np.all(is_inside_triangle(node1, node2, node3, point))


def test_point_outside_triangle():
    node1 = (0, 0)
    node2 = (5, 0)
    node3 = (0, 5)
    point = (6, 6)
    assert not np.all(is_inside_triangle(node1, node2, node3, point))


def test_point_on_extended_line_of_triangle():
    node1 = (0, 0)
    node2 = (4, 0)
    node3 = (0, 4)
    point = (5, 0)
    assert not np.all(is_inside_triangle(node1, node2, node3, point))


def test_triangle_with_negative_coordinates_point_inside():
    node1 = (-5, -5)
    node2 = (0, -5)
    node3 = (-5, 0)
    point = (-4, -4)
    assert np.all(is_inside_triangle(node1, node2, node3, point))


def test_triangle_with_negative_coordinates_point_outside():
    node1 = (-3, -3)
    node2 = (0, -3)
    node3 = (-3, 0)
    point = (1, 1)
    assert not np.all(is_inside_triangle(node1, node2, node3, point))

def test_triangle_area_with_positive_vertices():
    node1 = (0, 0)
    node2 = (4, 0)
    node3 = (0, 3)
    expected_area = 6.0
    assert get_triangle_area(node1, node2, node3) == expected_area


def test_triangle_area_with_negative_vertices():
    node1 = (-3, -5)
    node2 = (0, -5)
    node3 = (-3, -1)
    expected_area = 6.0
    assert get_triangle_area(node1, node2, node3) == expected_area


def test_triangle_area_with_collinear_points():
    node1 = (0, 0)
    node2 = (1, 1)
    node3 = (2, 2)
    expected_area = 0.0
    assert get_triangle_area(node1, node2, node3) == expected_area


def test_triangle_area_with_zero_vertices():
    node1 = (0, 0)
    node2 = (0, 0)
    node3 = (0, 0)
    expected_area = 0.0
    assert get_triangle_area(node1, node2, node3) == expected_area


def test_triangle_area_with_large_coordinates():
    node1 = (1e9, 0)
    node2 = (0, 1e9)
    node3 = (0, 0)
    expected_area = 5e17
    assert get_triangle_area(node1, node2, node3) == expected_area