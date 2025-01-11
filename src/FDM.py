import numpy as np

def backward_euler_one_iteration(M, A, b, x0, step_size):
    return np.linalg.solve(M + step_size * A, M @ x0 + step_size * b)

def backward_euler_method(M, A, b, x0, time_partition):
    r"""Solve ODE system Mx\'(T) + Ax(t) = b(T) using backward Euler Method.

    Keyword arguments:
    time_partition -- list [t0, t1, ..., tn] of time points to use

    Return value: list [x0, x1, ..., xn] where each element is a numpy array being an ODE solution approximation at given point"""
    res = [x0]
    for i in range(1, len(time_partition)):
        res.append(backward_euler_one_iteration(M, A, b, res[-1], time_partition[i] - time_partition[i-1]))
    return res

if __name__ == "__main__":
    M = np.array([[1,2,3], [2,3,1], [3,2,1]])
    A = np.array([[4,5,6],[5,6,4],[6,4,5]])
    b = np.array([8,9,7])
    x0 = np.array([1,2,3])
    time_partition = [1,2,3,4, 5, 6, 7]
    print(backward_euler_method(M, A, b, x0, time_partition))