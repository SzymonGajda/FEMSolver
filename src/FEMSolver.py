import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from numpy.ma.core import logical_or
import scipy.integrate as integrate


class Boundary:
    pass

class OneDimensionalBoundary(Boundary):
    def __init__(self, beginning: float, end: float):
        self.beginning = beginning
        self.end = end

class Mesh:
    def __init__(self, boundary: OneDimensionalBoundary, h: float = 1, mesh=None):
        self.boundary = boundary
        self.h = h
        if mesh is not None:
            self.mesh = mesh
        else:
            self.mesh = self.generate_mesh()

    def generate_mesh(self):
        num_of_subintervals = int(np.ceil((self.boundary.end - self.boundary.beginning) / self.h)) + 1
        return np.linspace(self.boundary.beginning, self.boundary.end, num_of_subintervals)


def one_sided_hat_function(x: float, beginning: np.float64, end: np.float64, coeff: np.float64):
    return np.where((logical_or(x < min(beginning, end), x > max(beginning, end))), 0,
                    x * coeff / (beginning - end) - end * coeff / (beginning - end))


def two_sided_hat_function(x: float, beginning: np.float64, node: np.float64, end: np.float64, coeff: np.float64):
    return np.where(x < node, one_sided_hat_function(x, node, beginning, coeff),
                    one_sided_hat_function(x, node, end, coeff))


def get_mass_matrix(mesh):
    res = np.zeros((mesh.mesh.size, mesh.mesh.size))
    for i in range(mesh.mesh.size - 1):
        h = mesh.mesh[i + 1] - mesh.mesh[i]
        res[i, i] += 1 / 3 * h
        res[i + 1, i + 1] += 1 / 3 * h
        res[i, i + 1] += 1 / 6 * h
        res[i + 1, i] += 1 / 6 * h
    return res


def get_load_vector(fun, mesh):
    res = np.zeros(mesh.mesh.size)
    res[0] = integrate.quad(lambda x: (fun(x) * one_sided_hat_function(x, mesh.mesh[0], mesh.mesh[1], 1)), mesh.mesh[0],
                            mesh.mesh[1])[0]
    res[-1] = integrate.quad(lambda x: (fun(x) * one_sided_hat_function(x, mesh.mesh[-1], mesh.mesh[-2], 1)),
                             mesh.mesh[-2], mesh.mesh[-1])[0]
    for i in range(mesh.mesh.size - 1):
        res[i] += integrate.quad(
            lambda x: (fun(x) * two_sided_hat_function(x, mesh.mesh[i - 1], mesh.mesh[i], mesh.mesh[i + 1], 1)),
            mesh.mesh[i - 1], mesh.mesh[i + 1])[0]
    return res


def get_piecewise_linear_function(mesh: Mesh, coeffs: np.ndarray):
    def res_fun(x):
        res = 0
        res += one_sided_hat_function(x, mesh.mesh[0], mesh.mesh[1], coeffs[0])
        res += one_sided_hat_function(x, mesh.mesh[-1], mesh.mesh[-2], coeffs[-1])
        for i in range(1, len(mesh.mesh) - 1):
            res += two_sided_hat_function(x, mesh.mesh[i - 1], mesh.mesh[i], mesh.mesh[i + 1], coeffs[i])
        return res

    return res_fun


def compute_l2_projection(fun: Callable[[float], float], mesh: Mesh):
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(fun, mesh)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return get_piecewise_linear_function(mesh, projection_coefficients)


if __name__ == "__main__":
    boundary = OneDimensionalBoundary(0, 20)
    mesh = Mesh(boundary, h=1)
    np.set_printoptions(linewidth=np.inf)
    projection = compute_l2_projection(np.sin, mesh)
    lin = np.linspace(0, 20, 1000)
    plt.plot(lin, projection(lin))
    plt.show()
