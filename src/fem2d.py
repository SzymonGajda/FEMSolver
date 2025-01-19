import numpy as np

from src.geom import get_triangle_area, is_inside_triangle
from src.quadratures import midpoint_2d_quadrature


def hat_function_2d(point, main_node, node_2, node_3, coefficient):
    denominator = (
        main_node[1] * node_2[0]
        - main_node[0] * node_2[1]
        + main_node[0] * node_3[1]
        - node_2[0] * node_3[1]
        - main_node[1] * node_3[0]
        + node_2[1] * node_3[0]
    )
    a = -node_2[1] + node_3[1]
    b = node_2[0] - node_3[0]
    c = -node_2[0] * node_3[1] + node_2[1] * node_3[0]
    return np.where(
        is_inside_triangle(main_node, node_2, node_3, point),
        coefficient * ((point[0] * a + point[1] * b + c) / denominator),
        0,
    )


def hat_function_2d_derivative(main_node, node_2, node_3):
    denominator = (
        main_node[1] * node_2[0]
        - main_node[0] * node_2[1]
        + main_node[0] * node_3[1]
        - node_2[0] * node_3[1]
        - main_node[1] * node_3[0]
        + node_2[1] * node_3[0]
    )
    return (
        (-node_2[1] + node_3[1]) / denominator,
        (node_2[0] - node_3[0]) / denominator,
    )


def get_stiffness_matrix_2d(coefficient_function, vertices, triangles, integrate_fun):
    stiffness_matrix = np.zeros((len(vertices), len(vertices)))
    for triangle in triangles:
        for i in range(3):
            for j in range(3):
                i_derivative = hat_function_2d_derivative(
                    vertices[triangle[i]],
                    vertices[list(set(triangle) - {triangle[i]})[0]],
                    vertices[list(set(triangle) - {triangle[i]})[1]],
                )
                j_derivative = hat_function_2d_derivative(
                    vertices[triangle[j]],
                    vertices[list(set(triangle) - {triangle[j]})[0]],
                    vertices[list(set(triangle) - {triangle[j]})[1]],
                )
                dot_product = (
                    i_derivative[0] * j_derivative[0]
                    + i_derivative[1]
                    + j_derivative[1]
                )
                stiffness_matrix[triangle[i]][triangle[j]] += integrate_fun(
                    lambda point_2d: dot_product * coefficient_function(point_2d),
                    vertices[triangle[0]],
                    vertices[triangle[1]],
                    vertices[triangle[2]],
                )
    return stiffness_matrix


def get_piecewise_linear_function_2d(vertices, triangles, coefficients):
    def res_fun(x):
        res = 0
        for triangle in triangles:
            triangle_res = 0
            for vertice in triangle:
                triangle_res += hat_function_2d(
                    x,
                    vertices[vertice],
                    vertices[list(set(triangle) - {vertice})[0]],
                    vertices[list(set(triangle) - {vertice})[1]],
                    coefficients[vertice],
                )
            res = np.where(triangle_res != 0, triangle_res, res)
        return res

    return res_fun


def get_mass_matrix_2d(vertices, triangles):
    mass_matrix = np.zeros((len(vertices), len(vertices)))
    for triangle in triangles:
        area = get_triangle_area(
            vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        )
        for i in range(3):
            for j in range(3):
                if i == j:
                    mass_matrix[triangle[i]][triangle[i]] += area / 6.0
                else:
                    mass_matrix[triangle[i]][triangle[j]] += area / 12.0
    return mass_matrix


def get_load_vector_2d(fun, vertices, triangles, integrate_fun, t):
    load_vector = np.zeros(len(vertices))
    for triangle in triangles:
        for vertice in triangle:
            fun_to_integrate = lambda point_2d: fun(point_2d, t) * hat_function_2d(
                point_2d,
                vertices[vertice],
                vertices[list(set(triangle) - {vertice})[0]],
                vertices[list(set(triangle) - {vertice})[1]],
                1,
            )
            load_vector[vertice] += integrate_fun(
                fun_to_integrate,
                vertices[triangle[0]],
                vertices[triangle[1]],
                vertices[triangle[2]],
            )
    return load_vector


def compute_l2_projection_2d(fun, vertices, triangles, t=0):
    mass_matrix = get_mass_matrix_2d(vertices, triangles)
    load_vector = get_load_vector_2d(
        fun, vertices, triangles, midpoint_2d_quadrature, t
    )
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (
        get_piecewise_linear_function_2d(vertices, triangles, projection_coefficients),
        projection_coefficients,
    )


def get_time_dependent_wave_equation_2d_solution(
    triangles, vertices, vertex_markers, time_partition
):
    res = []
    stiffness_matrix = get_stiffness_matrix_2d(
        lambda x: 1, vertices, triangles, midpoint_2d_quadrature
    )
    mass_matrix = get_mass_matrix_2d(vertices, triangles)
    load_vector = get_load_vector_2d(
        lambda x, t: 0, vertices, triangles, midpoint_2d_quadrature, 0
    )
    x0 = np.zeros(len(vertices) * 2)
    res.append(x0)

    boundary_points = []
    move_points = []
    for i in range(len(vertex_markers)):
        if vertex_markers[i] == 1:
            boundary_points.append(i)
        if vertices[i][0] <= -0.3:
            move_points.append(i)

    for i in range(1, len(time_partition)):
        print(f"Liczenie rozwiązania: iteracja {i} z {len(time_partition) - 1}")
        k = time_partition[i] - time_partition[i - 1]
        lhs = np.block(
            [
                [mass_matrix, -0.5 * k * mass_matrix],
                [0.5 * k * stiffness_matrix, mass_matrix],
            ]
        )
        rhs = (
            np.block(
                [
                    [mass_matrix, 0.5 * k * mass_matrix],
                    [-0.5 * k * stiffness_matrix, mass_matrix],
                ]
            )
            @ res[-1]
        )
        solution = np.linalg.solve(lhs, rhs)
        for boundary_point in boundary_points:
            solution[boundary_point] = 0
        for move_point in move_points:
            solution[move_point] = 0.1 * np.sin(8 * np.pi * time_partition[i])
        # solution[53] = 0.1 * np.sin(8 * np.pi * time_partition[i])
        res.append(solution)

    res2 = []
    for sol in res:
        res2.append(sol[0 : len(vertices)])
    return res2
