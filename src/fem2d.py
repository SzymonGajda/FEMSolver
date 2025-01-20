"""Main module for 2d fem solving. Provides methods for handling hat functions, mass
matrix, stiffness matrix, load vector, and projection coefficients."""

from collections.abc import Callable

import numpy as np

from src.geom import Point2d, get_triangle_area, is_inside_triangle
from src.quadratures import midpoint_2d_quadrature


def get_hat_function_denominator(
    main_vertex: Point2d, vertex2: Point2d, vertex3: Point2d
) -> float:
    """
    Computes the denominator of the hat function for three vertices of a triangle.

    This function uses the coordinates of the three vertices to calculate a value
    specific to hat functions, which are commonly used in finite element
    approximations. The calculation involves performing a determinant-based
    operation, which is essential for further computations in the given context.

    :param main_vertex: The first vertex of the triangle, represented as a
        `Point2d` object.
    :param vertex2: The second vertex of the triangle, represented as a
        `Point2d` object.
    :param vertex3: The third vertex of the triangle, represented as a
        `Point2d` object.

    :return: The computed denominator for the hat function, as a float.
    """
    denominator = (
        main_vertex[1] * vertex2[0]
        - main_vertex[0] * vertex2[1]
        + main_vertex[0] * vertex3[1]
        - vertex2[0] * vertex3[1]
        - main_vertex[1] * vertex3[0]
        + vertex2[1] * vertex3[0]
    )
    return denominator


def hat_function_2d(
    point: Point2d,
    main_vertex: Point2d,
    vertex2: Point2d,
    vertex3: Point2d,
    coefficient: float,
) -> np.ndarray:
    """
    Computes the 2D hat function value at a given point within a triangle specified
    by three vertices and a given coefficient. This is used in finite element
    methods to compute shape functions or basis functions.

    :param point: The point at which the hat function is to be evaluated.
    :param main_vertex: The main vertex of the triangle associated with the
        current hat function.
    :param vertex2: The second vertex of the triangle.
    :param vertex3: The third vertex of the triangle.
    :param coefficient: Scalar value to scale the computed hat function value.
    :return: An array containing the computed value of the hat function at the
        given point or zero if the point lies outside the triangle.
    """
    denominator = get_hat_function_denominator(main_vertex, vertex2, vertex3)
    a = -vertex2[1] + vertex3[1]
    b = vertex2[0] - vertex3[0]
    c = -vertex2[0] * vertex3[1] + vertex2[1] * vertex3[0]
    return np.where(
        is_inside_triangle(main_vertex, vertex2, vertex3, point),
        coefficient * ((point[0] * a + point[1] * b + c) / denominator),
        0,
    )


def hat_function_2d_derivative(
    main_vertex: Point2d, vertex2: Point2d, vertex3: Point2d
) -> tuple[float, float]:
    """
    Computes the derivative of the 2D hat function for a triangle element defined
    by three vertices. The derivative is returned as a tuple containing the
    x and y partial derivatives.

    :param main_vertex: The primary vertex of the triangle for which the
        derivative of the hat function is being computed.
    :param vertex2: The second vertex of the triangle.
    :param vertex3: The third vertex of the triangle.
    :return: A tuple containing the partial derivatives with respect to x and y.
    """
    denominator = get_hat_function_denominator(main_vertex, vertex2, vertex3)
    return (
        (-vertex2[1] + vertex3[1]) / denominator,
        (vertex2[0] - vertex3[0]) / denominator,
    )


def get_stiffness_matrix_2d(
    coefficient_function: Callable[[Point2d], float],
    vertices: list[Point2d],
    triangles: list[tuple[int, int, int]],
    integrate_fun: Callable[
        [Callable[[Point2d], float], Point2d, Point2d, Point2d], float
    ],
) -> np.ndarray:
    """
    Computes the stiffness matrix for a two-dimensional domain discretized using
    a triangular mesh. The stiffness matrix is assembled based on the given
    coefficient function, nodal positions, and integration function.

    The function iterates over provided triangular elements, calculates derivatives
    of basis (hat) functions, and integrates over triangles to compute contributions
    to the stiffness matrix. These contributions are summed to form the global
    stiffness matrix.

    :param coefficient_function: Function that computes coefficients at a given
        2D point, used in the integration to account for material properties or
        other spatially varying factors.
    :param vertices: List of 2D points representing the vertices of the domain.
    :param triangles: List of tuples, each containing three indices that define
        a triangle by referencing the vertices in the provided vertex list.
    :param integrate_fun: Function that performs integration over a triangle,
        given the derived function and vertices of the triangle to integrate over.
    :return: The assembled stiffness matrix as a 2D NumPy array. Each entry
        represents the stiffness contribution between corresponding nodes.
    """
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


def get_piecewise_linear_function_2d(
    vertices: list[Point2d],
    triangles: list[tuple[int, int, int]],
    coefficients: np.ndarray,
) -> Callable[[Point2d], float]:
    """
    Constructs a piecewise linear function in 2D space based on given vertices,
    triangles, and coefficients. The function interpolates over a 2D mesh defined
    by the vertices and their connectivity specified in the triangles.

    :param vertices:
        A list of 2D points representing the vertices of the mesh.
    :param triangles:
        A list of tuples, where each tuple contains three integers representing
        the indices of vertices forming a triangle.
    :param coefficients:
        A numpy array containing coefficients for the basis functions associated
        with each vertex.
    :return:
        A callable function that takes a 2D point (Point2d) as input and returns
        the interpolated value at that point using a piecewise linear basis.
    """

    def piecewise_linear_fun(x):
        value_at_point = 0
        for triangle in triangles:
            value_on_triangle = 0
            for vertice in triangle:
                value_on_triangle += hat_function_2d(
                    x,
                    vertices[vertice],
                    vertices[list(set(triangle) - {vertice})[0]],
                    vertices[list(set(triangle) - {vertice})[1]],
                    coefficients[vertice],
                )
            value_at_point = np.where(
                value_on_triangle != 0, value_on_triangle, value_at_point
            )
        return value_at_point

    return piecewise_linear_fun


def get_mass_matrix_2d(
    vertices: list[Point2d], triangles: list[tuple[int, int, int]]
) -> np.ndarray:
    """
    Computes the mass matrix for a given 2D mesh defined by vertices and triangles.
    Each triangle's contribution is based on its area and the indices of its vertices.

    :param vertices: A list of 2D points representing the vertices of the mesh.
        Each vertex is an instance of `Point2d`.
    :param triangles: A list of tuples where each tuple represents a triangle in the
        mesh. Each tuple consists of three integer indices corresponding to vertices
        in the `vertices` list.
    :return: A 2D NumPy array representing the computed mass matrix.
    """
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


def get_load_vector_2d(
    fun: Callable[[Point2d], float],
    vertices: list[Point2d],
    triangles: list[tuple[int, int, int]],
    integrate_fun: Callable[
        [Callable[[Point2d], float], Point2d, Point2d, Point2d], float
    ],
) -> np.ndarray:
    """
    Calculate the load vector for a given 2D finite element problem.

    This function computes the load vector associated with a set of 2D triangular
    finite elements. The load vector is computed by integrating a given function
    over each triangular element while taking into account specific shape functions.

    :param fun: Function to be integrated on the domain.
    :param vertices: List of 2D points representing the vertices of the mesh.
    :param triangles: List of indices representing triangular elements, where each
        triangle is defined by a tuple of three vertex indices referring to the
        ``vertices`` list.
    :param integrate_fun: Function responsible for performing integration over a
        triangular region. It takes as parameters the function to be integrated and
        the three vertices of the triangle.
    :return: A NumPy array representing the computed load vector, where each entry
        corresponds to a vertex in the mesh.
    """
    load_vector = np.zeros(len(vertices))
    for triangle in triangles:
        for vertice in triangle:
            fun_to_integrate = lambda point_2d: fun(point_2d) * hat_function_2d(
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


def compute_l2_projection_2d(
    fun: Callable[[Point2d], float],
    vertices: list[Point2d],
    triangles: list[tuple[int, int, int]],
) -> tuple[Callable[[Point2d], float], np.ndarray]:
    """
    Compute the L2 projection of a given 2D function onto a piecewise linear finite element space.

    This function takes as input a callable to represent a real-valued function of a 2D
    point, the list of vertices of a finite element mesh, and the list of triangles
    defined by indices of the vertices. It computes the L2 projection of the function
    onto the piecewise linear finite element basis functions defined on this triangular
    mesh. The result is a new piecewise linear function and the coefficients of the basis
    functions.

    :param fun: A callable representing the input function to be projected, which takes
        a 2D point (represented as ``Point2d``) and returns a float value.
    :param vertices: A list of 2D points (``Point2d``) representing the vertices of
        the triangular mesh to be used for the projection.
    :param triangles: A list of tuples (each containing three integers), where each
        tuple represents triangle connectivity by referencing the vertex indices from
        the ``vertices`` list.
    :return: A tuple consisting of:
        - A callable representing the projected piecewise linear function over the mesh.
        - A NumPy array containing the coefficients of the piecewise linear basis
            functions corresponding to the projection.
    """
    mass_matrix = get_mass_matrix_2d(vertices, triangles)
    load_vector = get_load_vector_2d(fun, vertices, triangles, midpoint_2d_quadrature)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (
        get_piecewise_linear_function_2d(vertices, triangles, projection_coefficients),
        projection_coefficients,
    )


def get_time_dependent_wave_equation_2d_solution(
    triangles: list[tuple[int, int, int]],
    vertices: list[Point2d],
    vertex_markers: list[int],
    time_partition: list[float],
) -> list[np.ndarray]:
    """
    Calculates the time-dependent solution of the wave equation in 2D using the finite
    element method. It involves constructing stiffness and mass matrices, assembling
    the load vector, applying boundary conditions, and iteratively solving for the
    coefficients over the time domain. The solution is returned as a list of matrices,
    each corresponding to a time step.

    :param triangles: A list of tuples where each tuple contains three integers
        representing the vertex indices of a triangle in the 2D finite element mesh.
    :param vertices: A list of Point2d objects representing the coordinates of
        the vertices in the 2D finite element mesh.
    :param vertex_markers: A list of integers where each entry denotes the marker
        (e.g., boundary condition type or identification) of the corresponding vertex.
    :param time_partition: A list of floats representing the time discretization points
        for which the solution is computed.
    :return: A list of numpy.ndarray objects, where each array contains the solution
        at a particular time step for all nodes in the mesh.
    """
    coefficients = []
    stiffness_matrix = get_stiffness_matrix_2d(
        lambda x: 1, vertices, triangles, midpoint_2d_quadrature
    )
    mass_matrix = get_mass_matrix_2d(vertices, triangles)
    load_vector = get_load_vector_2d(
        lambda x: 0, vertices, triangles, midpoint_2d_quadrature
    )
    x0 = np.zeros(len(vertices) * 2)
    coefficients.append(x0)

    boundary_points = []
    moving_points = []
    for i in range(len(vertex_markers)):
        if vertex_markers[i] == 1:
            boundary_points.append(i)
        if vertices[i][0] <= -0.3:
            moving_points.append(i)

    for i in range(1, len(time_partition)):
        time_step = time_partition[i] - time_partition[i - 1]
        lhs = np.block(
            [
                [mass_matrix, -0.5 * time_step * mass_matrix],
                [0.5 * time_step * stiffness_matrix, mass_matrix],
            ]
        )
        rhs = (
            np.block(
                [
                    [mass_matrix, 0.5 * time_step * mass_matrix],
                    [-0.5 * time_step * stiffness_matrix, mass_matrix],
                ]
            )
            @ coefficients[-1]
        )
        solution = np.linalg.solve(lhs, rhs)
        # Enforcing boundary conditions
        for boundary_point in boundary_points:
            solution[boundary_point] = 0
        for move_point in moving_points:
            solution[move_point] = 0.1 * np.sin(8 * np.pi * time_partition[i])

        coefficients.append(solution)

    result = []
    for coefficient in coefficients:
        result.append(coefficient[0 : len(vertices)])
    return result
