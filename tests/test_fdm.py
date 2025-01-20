import numpy as np
from src.fdm import backward_euler_one_iteration

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
