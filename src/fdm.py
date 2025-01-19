"""Module provides implementations of different finite difference methods for ODE solving."""

import numpy as np


def backward_euler_one_iteration(
    mass_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
    load_vector: np.ndarray,
    current_state: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """
    Performs one iteration of the backward Euler method for solving a system of
    linear ordinary differential equations (ODEs).

    This method approximates the next state of the system using implicit
    discretization by solving a linear system derived from the ODE.

    :param mass_matrix: Coefficient matrix representing system mass or generalized mass
              (typically square matrix with compatible dimensions).
    :param stiffness_matrix: Coefficient matrix representing system dynamics or stiffness
              (typically square matrix with dimensions matching `M`).
    :param load_vector: External forcing vector or constant term (dimension compatible
              with the number of rows in `M` and `A`).
    :param current_state: Initial state vector at the current time step
               (dimension compatible with the rows in `M` and `A`).
    :param step_size: Discretization time step size for the numerical method.
    :return: Approximated state vector at the next time step based on the backward
             Euler method.
    """

    lhs = mass_matrix + step_size * stiffness_matrix
    rhs = mass_matrix @ current_state + step_size * load_vector
    return np.linalg.solve(lhs, rhs)


def backward_euler_method(
    mass_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
    load_vector: np.ndarray,
    initial_state: np.ndarray,
    time_partition: list[float],
) -> list[np.ndarray]:
    r"""
    Solves a system of first-order linear ordinary differential equations of the form
    M * x\'(t) = A * x(t) + b(t) using the Backward Euler method for a given time
    partition. This function iteratively applies the Backward Euler method to
    compute the numerical solutions at each step in the time partition.

    :param mass_matrix: Coefficient matrix representing the system dynamics
    :param stiffness_matrix: State transition matrix
    :param load_vector: Constant vector term in the system
    :param initial_state: Initial state vector
    :param time_partition: List of time points for discretization
    :return: List of state vectors computed at each time step
    """
    result = [initial_state]
    for i in range(1, len(time_partition)):
        time_step = time_partition[i] - time_partition[i - 1]
        result.append(
            backward_euler_one_iteration(
                mass_matrix, stiffness_matrix, load_vector, result[-1], time_step
            )
        )
    return result