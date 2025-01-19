"""Main module for 1d fem solving. Provides methods for handling hat functions, mass
matrix, stiffness matrix, load vector, and projection coefficients."""

from typing import Callable

import numpy as np
from numpy.ma import logical_or

from src import fdm


def one_sided_hat_function_1d(
    x: float, beginning: float, end: float, coefficient: float
) -> np.ndarray:
    """
    Computes a one-sided hat function in 1D using the given parameters.

    The one-sided hat function is defined as a piecewise linear function that returns
    values calculated based on the input `x` for a specified range between `beginning`
    and `end`. It outputs 0 outside the specified range. Within the range, the function
    scales the input `x` linearly using the given `coefficient`.

    :param x: The input value for the computation of the hat function.
    :param beginning: The starting point of the interval used for the one-sided hat
        function.
    :param end: The ending point of the interval used for the one-sided hat
        function.
    :param coefficient: The scaling value applied to the linear transformation
        within the interval.

    :return: The computed values of the one-sided hat function for the given
        parameters as a NumPy array.
    """
    return np.where(
        (logical_or(x < min(beginning, end), x > max(beginning, end))),
        0,
        x * coefficient / (beginning - end) - end * coefficient / (beginning - end),
    )


def two_sided_hat_function_1d(
    x: float, beginning: float, node: float, end: float, coefficient: float
) -> np.ndarray:
    """
    Evaluates a two-sided hat function in 1D space. The function is defined by two
    halves of a one-sided hat function mirrored around the node. It calculates
    values piecewise based on the position of the input `x` relative to the
    specified `node`.

    :param x: Input scalar value or array of values for which the function will be
        evaluated.
    :param beginning: The starting point of the one-sided hat function for the
        left side of the node.
    :param node: The node or reference point that divides the function into two
        segments.
    :param end: The ending point of the one-sided hat function for the right side
        of the node.
    :param coefficient: Scaling coefficient to adjust the amplitude of the
        one-sided hat function on both sides.
    :return: An array filled with the evaluated output values of the two-sided
        hat function for all elements in `x`.
    """
    return np.where(
        x < node,
        one_sided_hat_function_1d(x, node, beginning, coefficient),
        one_sided_hat_function_1d(x, node, end, coefficient),
    )


def one_sided_hat_function_derivative(
    x: float, beginning: float, end: float, coefficient: float
) -> np.ndarray:
    """
    Compute the derivative of a one-sided hat function.

    This function calculates the derivative of a one-sided hat function, based on the
    provided interval defined by `beginning` and `end`. It returns an array where the
    computed derivative is non-zero within the interval and zero outside.

    :param x: The point(s) at which the derivative is to be evaluated.
    :param beginning: The starting point of the interval for the one-sided hat function.
    :param end: The ending point of the interval for the one-sided hat function.
    :param coefficient: A constant coefficient applied to the derivative.
    :return: An array of derivative values evaluated at the given points.
    """
    return np.where(
        (logical_or(x < min(beginning, end), x > max(beginning, end))),
        0,
        coefficient / (beginning - end),
    )


def two_sided_hat_function_derivative(
    x: float, beginning: float, node: float, end: float, coefficient: float
) -> np.ndarray:
    """
    Calculates the derivative of a two-sided hat function based on its parameters.

    This function computes the derivative of a two-sided hat function
    by determining the appropriate side (left or right of the given node)
    and using the derivative of a one-sided hat function for that side.

    :param x: The input values for which the derivative of the hat function
              is to be computed.
    :param beginning: The starting point of the first linear segment.
    :param node: The node or peak point of the hat function.
    :param end: The ending point of the second linear segment.
    :param coefficient: The coefficient that scales the derivative of the
                        one-sided hat function.
    :return: An array containing the derivative values of the two-sided
             hat function for the input `x`.
    """
    return np.where(
        x < node,
        one_sided_hat_function_derivative(x, node, beginning, coefficient),
        one_sided_hat_function_derivative(x, node, end, coefficient),
    )


def get_mass_matrix(mesh: list[float]) -> np.ndarray:
    """
    Generates the mass matrix for a given 1D discretized mesh.

    This function computes the mass matrix for a one-dimensional mesh,
    assuming linear basis functions. The mass matrix is a symmetric matrix
    that integrates the basis functions over the domain, weighted by the
    length of each interval.

    :param mesh: The discretized 1D mesh as a list of floats, where each element
                 represents a node's position in increasing order.
    :return: The computed mass matrix as a symmetric 2D NumPy array.
    """
    mass_matrix = np.zeros((len(mesh), len(mesh)))
    for i in range(len(mesh) - 1):
        interval_length = mesh[i + 1] - mesh[i]
        mass_matrix[i, i] += 1 / 3 * interval_length
        mass_matrix[i + 1, i + 1] += 1 / 3 * interval_length
        mass_matrix[i, i + 1] += 1 / 6 * interval_length
        mass_matrix[i + 1, i] += 1 / 6 * interval_length
    return mass_matrix


def get_stiffness_matrix(
    mesh: list[float],
    integrate_fun: Callable[[Callable[[float], np.ndarray], float, float], float],
    coefficient_function: Callable[[float], float],
) -> np.ndarray:
    """
    Computes the stiffness matrix for a given finite element mesh using specified
    integral and coefficient functions. The stiffness matrix is computed based on
    hat-function derivatives and their interactions at nodal points of the mesh.

    :param mesh: List of nodal points defining the finite element mesh.
    :param integrate_fun: Function used to numerically integrate over a segment
        of the mesh.
    :param coefficient_function: A function defining the coefficient applied to the
        integrals of the hat-function derivatives.
    :return: A 2D numpy array representing the stiffness matrix, with dimensions
        len(mesh) x len(mesh). The matrix is symmetric and contains the
        contributions for all nodes based on the specified integration.
    """
    stiffness_matrix = np.zeros((len(mesh), len(mesh)))
    stiffness_matrix[0, 0] += integrate_fun(
        lambda x: (
            coefficient_function(x)
            * one_sided_hat_function_derivative(x, mesh[0], mesh[1], 1)
            * one_sided_hat_function_derivative(x, mesh[0], mesh[1], 1)
        ),
        mesh[0],
        mesh[1],
    )
    stiffness_matrix[-1, -1] += integrate_fun(
        lambda x: (
            coefficient_function(x)
            * one_sided_hat_function_derivative(x, mesh[-1], mesh[-2], 1)
            * one_sided_hat_function_derivative(x, mesh[-1], mesh[-2], 1)
        ),
        mesh[-2],
        mesh[-1],
    )

    for i in range(1, len(mesh)):
        if i < len(mesh) - 1:
            stiffness_matrix[i, i] += integrate_fun(
                lambda x: (
                    coefficient_function(x)
                    * two_sided_hat_function_derivative(
                        x, mesh[i - 1], mesh[i], mesh[i + 1], 1
                    )
                    * two_sided_hat_function_derivative(
                        x, mesh[i - 1], mesh[i], mesh[i + 1], 1
                    )
                ),
                mesh[i - 1],
                mesh[i + 1],
            )
        stiffness_matrix[i, i - 1] += integrate_fun(
            lambda x: (
                coefficient_function(x)
                * one_sided_hat_function_derivative(x, mesh[i - 1], mesh[i], 1)
                * one_sided_hat_function_derivative(x, mesh[i], mesh[i - 1], 1)
            ),
            mesh[i - 1],
            mesh[i],
        )
        stiffness_matrix[i - 1, i] = stiffness_matrix[i, i - 1]
    return stiffness_matrix


def get_load_vector(
    fun: Callable[[float], float],
    mesh: list[float],
    integrate_fun: Callable[[Callable[[float], np.ndarray], float, float], float],
) -> np.ndarray:
    """
    Computes the load vector based on the given function, mesh, and integration function. The load vector represents
    the integral of the product of the supplied function and the associated 1D hat functions over the provided mesh.

    :param fun: A callable that specifies the function to be integrated.
    :param mesh: A list of floats representing the mesh points where the load vector is computed.
    :param integrate_fun: A callable that specifies the integration function taking a function and integration limits
                          as inputs.
    :return: A NumPy ndarray representing the computed load vector.
    """
    load_vector = np.zeros(len(mesh))
    load_vector[0] = integrate_fun(
        lambda x: (fun(x) * one_sided_hat_function_1d(x, mesh[0], mesh[1], 1)),
        mesh[0],
        mesh[1],
    )
    load_vector[-1] = integrate_fun(
        lambda x: (fun(x) * one_sided_hat_function_1d(x, mesh[-1], mesh[-2], 1)),
        mesh[-2],
        mesh[-1],
    )
    for i in range(len(mesh) - 1):
        load_vector[i] += integrate_fun(
            lambda x: (
                fun(x)
                * two_sided_hat_function_1d(x, mesh[i - 1], mesh[i], mesh[i + 1], 1)
            ),
            mesh[i - 1],
            mesh[i + 1],
        )
    return load_vector


def get_piecewise_linear_function(
    mesh: list[float], coefficients: np.ndarray
) -> Callable[[float], float]:
    """
    Creates and returns a piecewise linear function based on the provided mesh
    and coefficients.

    The piecewise linear function is constructed using one-sided hat functions
    for the endpoints of the mesh and two-sided hat functions for internal
    points of the mesh. The resulting function is a linear combination of these
    hat functions, where the coefficients provided in the input determine the
    scaling of each hat function.

    :param mesh: List or array of numerical values that define the discretized
        mesh points. These points represent the "nodes" of the piecewise linear
        function, and their order matters for the construction of the function.
        The mesh must include at least two points.
    :param coefficients: 1D array of coefficients applied to each hat function.
        There must be a coefficient corresponding to each mesh point. The
        scaling of one-sided or two-sided hat functions is determined by these
        coefficients.
    :return: A function representing the piecewise linear combination of hat
        functions, which takes a scalar x as input and outputs the corresponding
        value of the piecewise linear function.
    """

    def piecewise_linear_function(x):
        value = 0
        value += one_sided_hat_function_1d(x, mesh[0], mesh[1], coefficients[0])
        value += one_sided_hat_function_1d(x, mesh[-1], mesh[-2], coefficients[-1])
        for i in range(1, len(mesh) - 1):
            value += two_sided_hat_function_1d(
                x, mesh[i - 1], mesh[i], mesh[i + 1], coefficients[i]
            )
        return value

    return piecewise_linear_function


def compute_l2_projection(
    fun: Callable[[float], float],
    mesh: list[float],
    integrate_fun: Callable[[Callable[[float], np.ndarray], float, float], float],
) -> tuple[Callable[[float], float], np.ndarray]:
    """
    Compute the L2 projection of a given function onto a piecewise linear function space
    defined by a provided mesh. The method involves constructing the mass matrix and
    the load vector for the mesh, solving the linear system of equations, and returning
    both the piecewise linear function and the coefficients of the projection.

    :param fun: The target function to project.
    :param mesh: A list of points in the domain that define the mesh.
    :param integrate_fun: A function to numerically integrate another function over a
        given interval.
    :return: A tuple consisting of the piecewise linear function approximating the
        input function and the projection coefficients.
    """
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(fun, mesh, integrate_fun)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (
        get_piecewise_linear_function(mesh, projection_coefficients),
        projection_coefficients,
    )


def add_boundary_conditions(
    stiffness_matrix: np.ndarray,
    load_vector: np.ndarray,
    kappa_0: float,
    kappa_n: float,
    g_0: float,
    g_n: float,
):
    """
    Adds boundary conditions to the stiffness matrix and load vector. This function modifies the given stiffness matrix and
    load vector to incorporate boundary conditions defined by boundary parameters `kappa_0`, `kappa_n`, `g_0`, and `g_n`.
    The modified stiffness matrix and load vector are then returned.

    :param stiffness_matrix: 2D NumPy array representing the stiffness matrix to be modified.
    :param load_vector: 1D NumPy array representing the load vector to be modified.
    :param kappa_0: Scalar value for the boundary condition coefficient at the start of the domain.
    :param kappa_n: Scalar value for the boundary condition coefficient at the end of the domain.
    :param g_0: Scalar value representing the boundary value at the start of the domain.
    :param g_n: Scalar value representing the boundary value at the end of the domain.
    :return: A tuple containing the modified stiffness matrix and load vector as (stiffness_matrix, load_vector).
    """
    load_vector[0] += kappa_0 * g_0
    load_vector[-1] += kappa_n * g_n
    stiffness_matrix[0, 0] += kappa_0
    stiffness_matrix[-1, -1] += kappa_n
    return stiffness_matrix, load_vector


def get_heat_equation_solution(
    heat_source: Callable[[float], float],
    kappa_0: float,
    kappa_n: float,
    g_0: float,
    g_n: float,
    mesh: list[float],
    cross_section_area_fun: Callable[[float], float],
    conductivity_fun: Callable[[float], float],
    integrate_fun: Callable[[Callable[[float], np.ndarray], float, float], float],
) -> Callable[[float], float]:
    """
    Computes the solution of the heat equation with specified boundary conditions,
    source term, and material properties over a given mesh.

    This function assembles a finite element system to solve the heat equation. It
    builds a stiffness matrix and load vector using numerical integration, applies
    boundary conditions to the system, and solves the resulting linear system. The
    solution is returned as a piecewise linear function.

    :param heat_source: The heat source term, represented as a callable function
        of one variable (position).
    :param kappa_0: The coefficient for the left boundary condition.
    :param kappa_n: The coefficient for the right boundary condition.
    :param g_0: The value of the boundary condition at the left boundary.
    :param g_n: The value of the boundary condition at the right boundary.
    :param mesh: The spatial discretization of the domain, represented as a list
        of coordinates.
    :param cross_section_area_fun: A callable function of one variable that
        defines the cross-sectional area of the material at a given position.
    :param conductivity_fun: A callable function of one variable that defines the
        thermal conductivity of the material at a given position.
    :param integrate_fun: A callable function that performs numerical integration,
        given a function to integrate and the integration bounds.
    :return: A callable function representing the heat equation solution as a
        piecewise linear function.
    """
    coefficient_function = lambda x: cross_section_area_fun(x) * conductivity_fun(x)
    stiffness_matrix = get_stiffness_matrix(mesh, integrate_fun, coefficient_function)
    load_vector = get_load_vector(heat_source, mesh, integrate_fun)
    (stiffness_matrix, load_vector) = add_boundary_conditions(
        stiffness_matrix, load_vector, kappa_0, kappa_n, g_0, g_n
    )
    projection_coefficients = np.linalg.solve(stiffness_matrix, load_vector)
    return get_piecewise_linear_function(mesh, projection_coefficients)


def get_time_dependent_heat_equation_solution(
    heat_source: Callable[[float, float], float],
    kappa_0: float,
    kappa_n: float,
    g_0: float,
    g_n: float,
    mesh: list[float],
    time_partition: list[float],
    cross_section_area: Callable[[float], float],
    conductivity: Callable[[float], float],
    initial_temperature: Callable[[float], float],
    integrate_fun: Callable[[Callable[[float], np.ndarray], float, float], float],
) -> list[np.ndarray]:
    """
    Computes the solution of the time-dependent heat equation using the Finite
    Difference Method (FDM). The function requires the problem's boundary
    conditions, mesh information, time discretization partitions, and functions
    defining heat source, cross-sectional area, conductivity, and initial
    temperature.

    :param heat_source: A callable that represents the heat source as a function
        of spatial position `x` and time `t`.
    :param kappa_0: The boundary condition coefficient at the starting point of
        the domain boundary.
    :param kappa_n: The boundary condition coefficient at the ending point of
        the domain boundary.
    :param g_0: The value of the Dirichlet or Neumann boundary condition at the
        starting point of the domain boundary.
    :param g_n: The value of the Dirichlet or Neumann boundary condition at the
        ending point of the domain boundary.
    :param mesh: A list containing the spatial discretization points.
    :param time_partition: A list defining the time discretization intervals.
    :param cross_section_area: A callable that represents the cross-sectional
        area of the domain as a function of spatial position `x`.
    :param conductivity: A callable that represents the material's thermal
        conductivity as a function of spatial position `x`.
    :param initial_temperature: A callable that represents the initial
        temperature of the domain as a function of spatial position `x`.
    :param integrate_fun: A callable that performs numerical integration in
        space, given a function and integration bounds.
    :return: A list where each element is a numpy array representing the
        spatial temperature profile at a specific time step over the
        discretized domain.
    """
    solution = []
    variable_coefficient = lambda x: cross_section_area(x) * conductivity(x)
    stiffness_matrix = get_stiffness_matrix(mesh, integrate_fun, variable_coefficient)
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(
        lambda x: heat_source(x, time_partition[0]), mesh, integrate_fun
    )
    (stiffness_matrix, load_vector) = add_boundary_conditions(
        stiffness_matrix, load_vector, kappa_0, kappa_n, g_0, g_n
    )
    initial_conditions = compute_l2_projection(
        initial_temperature, mesh, integrate_fun
    )[1]
    solution.append(initial_conditions)
    for i in range(1, len(time_partition)):
        time_step = time_partition[i] - time_partition[i - 1]
        solution.append(
            fdm.backward_euler_one_iteration(
                mass_matrix,
                stiffness_matrix,
                load_vector,
                solution[-1],
                time_step,
            )
        )
    return solution
