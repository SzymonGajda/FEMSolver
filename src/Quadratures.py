from src.Geom import get_triangle_area


def get_midpoint_value(fun, point1, point2):
    return fun((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def get_midpoint_quadrature(fun, point1, point2, point3):
    res = 0
    area = get_triangle_area(point1, point2, point3)
    res += get_midpoint_value(fun, point1, point2)
    res += get_midpoint_value(fun, point2, point3)
    res += get_midpoint_value(fun, point3, point1)
    res *= area / 3
    return res
