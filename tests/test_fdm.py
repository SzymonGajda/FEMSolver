import numpy as np
import pytest
from src.fdm import backward_euler_one_iteration, backward_euler_method


def test_backward_euler_one_iteration_basic():
    mass_matrix = np.array([[2]])
    stiffness_matrix = np.array([[1]])
    load_vector = np.array([1])
    current_state = np.array([0])
    step_size = 0.1
    result = backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, current_state, step_size)
    expected = np.linalg.solve(mass_matrix + step_size * stiffness_matrix,
                               mass_matrix @ current_state + step_size * load_vector)
    assert np.allclose(result, expected)

def test_backward_euler_one_iteration_multiple_variables():
    mass_matrix = np.array([[2, 0], [0, 3]])
    stiffness_matrix = np.array([[1, 0], [0, 2]])
    load_vector = np.array([1, 2])
    current_state = np.array([1, 1])
    step_size = 0.1
    result = backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, current_state, step_size)
    expected = np.linalg.solve(mass_matrix + step_size * stiffness_matrix,
                               mass_matrix @ current_state + step_size * load_vector)
    assert np.allclose(result, expected)

def test_backward_euler_one_iteration_zero_load_vector():
    mass_matrix = np.array([[1, 0], [0, 1]])
    stiffness_matrix = np.array([[1, 0], [0, 1]])
    load_vector = np.array([0, 0])
    current_state = np.array([1, 2])
    step_size = 0.2
    result = backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, current_state, step_size)
    expected = np.linalg.solve(mass_matrix + step_size * stiffness_matrix,
                               mass_matrix @ current_state + step_size * load_vector)
    assert np.allclose(result, expected)

def test_backward_euler_one_iteration_identity_matrices():
    mass_matrix = np.eye(3)
    stiffness_matrix = np.eye(3)
    load_vector = np.array([1, 1, 1])
    current_state = np.array([0, 0, 0])
    step_size = 0.5
    result = backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, current_state, step_size)
    expected = np.linalg.solve(mass_matrix + step_size * stiffness_matrix,
                               mass_matrix @ current_state + step_size * load_vector)
    assert np.allclose(result, expected)

def test_backward_euler_one_iteration_large_step_size():
    mass_matrix = np.array([[1, 0], [0, 2]])
    stiffness_matrix = np.array([[1, 1], [1, 3]])
    load_vector = np.array([1, 0])
    current_state = np.array([0, 1])
    step_size = 10.0
    result = backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, current_state, step_size)
    expected = np.linalg.solve(mass_matrix + step_size * stiffness_matrix,
                               mass_matrix @ current_state + step_size * load_vector)
    assert np.allclose(result, expected)

def test_backward_euler_basic_case():
    mass_matrix = np.array([[1]])
    stiffness_matrix = np.array([[2]])
    load_vector = np.array([0])
    initial_state = np.array([1])
    time_partition = [0, 1, 2]
    result = backward_euler_method(
        mass_matrix, stiffness_matrix, load_vector, initial_state, time_partition
    )
    assert len(result) == len(time_partition)
    assert np.isclose(result[-1][0], -1 / 3, atol=1e-6)


def test_backward_euler_nonzero_load():
    mass_matrix = np.array([[2]])
    stiffness_matrix = np.array([[1]])
    load_vector = np.array([1])
    initial_state = np.array([0])
    time_partition = [0, 1]
    result = backward_euler_method(
        mass_matrix, stiffness_matrix, load_vector, initial_state, time_partition
    )
    assert len(result) == len(time_partition)
    assert np.isclose(result[-1][0], 0.333333, atol=1e-6)


def test_backward_euler_multiple_dimensions():
    mass_matrix = np.eye(2)
    stiffness_matrix = np.array([[1, 1], [-1, 1]])
    load_vector = np.array([0, 0])
    initial_state = np.array([1, -1])
    time_partition = [0, 0.5, 1.0]
    result = backward_euler_method(
        mass_matrix, stiffness_matrix, load_vector, initial_state, time_partition
    )
    assert len(result) == len(time_partition)
    assert np.isclose(result[-1][0], 0.455, atol=1e-3)
    assert np.isclose(result[-1][1], -0.455, atol=1e-3)


def test_backward_euler_varying_time_step():
    mass_matrix = np.array([[1]])
    stiffness_matrix = np.array([[1]])
    load_vector = np.array([0])
    initial_state = np.array([1])
    time_partition = [0, 0.1, 0.4, 1.0]
    result = backward_euler_method(
        mass_matrix, stiffness_matrix, load_vector, initial_state, time_partition
    )
    assert len(result) == len(time_partition)
    assert np.isclose(result[-1][0], 0.367879, atol=1e-6)


def test_backward_euler_stability():
    mass_matrix = np.array([[1]])
    stiffness_matrix = np.array([[-100]])
    load_vector = np.array([0])
    initial_state = np.array([1])
    time_partition = [0, 0.01, 0.02, 0.03]
    result = backward_euler_method(
        mass_matrix, stiffness_matrix, load_vector, initial_state, time_partition
    )
    assert len(result) == len(time_partition)
    assert all(state[0] >= 0 for state in result)
