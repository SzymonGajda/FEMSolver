import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Callable

from matplotlib import animation
from numpy.ma.core import logical_or
import scipy.integrate as integrate
import FDM


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


def one_sided_hat_function_derivative(x: float, beginning: np.float64, end: np.float64, coeff: np.float64):
    return np.where((logical_or(x < min(beginning, end), x > max(beginning, end))), 0, coeff / (beginning - end))


def two_sided_hat_function_derivative(x: float, beginning: np.float64, node: np.float64, end: np.float64,
                                      coeff: np.float64):
    return np.where(x < node, one_sided_hat_function_derivative(x, node, beginning, coeff),
                    one_sided_hat_function_derivative(x, node, end, coeff))


def get_rode_cross_section_area(x):
    # return 0.1
    return 1


def get_rode_conductivity(x):
    return 0.1 * (5 - 0.6 * x)


# return 1


def get_heat_source_function(x, t):
    return 2 * x
    # if logical_or(x < 2,  x > 4):
    #    return 0
    # return 1
    # return np.where(logical_or(x < 1, x > 2), 0, 0.1)


def get_initial_temperature(x, t=0):
    return 0.5 - abs(x - 0.5)


def get_mass_matrix(mesh):
    res = np.zeros((mesh.mesh.size, mesh.mesh.size))
    for i in range(mesh.mesh.size - 1):
        h = mesh.mesh[i + 1] - mesh.mesh[i]
        res[i, i] += 1 / 3 * h
        res[i + 1, i + 1] += 1 / 3 * h
        res[i, i + 1] += 1 / 6 * h
        res[i + 1, i] += 1 / 6 * h
    return res


def get_stiffness_matrix(mesh, integrate_fun):
    res = np.zeros((mesh.mesh.size, mesh.mesh.size))
    res[0, 0] += integrate_fun(lambda x:
                               (get_rode_cross_section_area(x) * get_rode_conductivity(x) *
                                one_sided_hat_function_derivative(x, mesh.mesh[0], mesh.mesh[1], 1) *
                                one_sided_hat_function_derivative(x, mesh.mesh[0], mesh.mesh[1], 1)), mesh.mesh[0],
                               mesh.mesh[1])[0]
    res[-1, -1] += integrate_fun(lambda x:
                                 (get_rode_cross_section_area(x) * get_rode_conductivity(x) *
                                  one_sided_hat_function_derivative(x, mesh.mesh[-1], mesh.mesh[-2], 1) *
                                  one_sided_hat_function_derivative(x, mesh.mesh[-1], mesh.mesh[-2], 1)),
                                 mesh.mesh[-2], mesh.mesh[-1])[0]

    for i in range(1, mesh.mesh.size):
        if i < mesh.mesh.size - 1:
            res[i, i] += integrate_fun(lambda x:
                                       (get_rode_cross_section_area(x) * get_rode_conductivity(x) *
                                        two_sided_hat_function_derivative(x, mesh.mesh[i - 1], mesh.mesh[i],
                                                                          mesh.mesh[i + 1], 1) *
                                        two_sided_hat_function_derivative(x, mesh.mesh[i - 1], mesh.mesh[i],
                                                                          mesh.mesh[i + 1], 1)), mesh.mesh[i - 1],
                                       mesh.mesh[i + 1])[0]
        res[i, i - 1] += integrate_fun(lambda x: (get_rode_cross_section_area(x) * get_rode_conductivity(x) *
                                                  one_sided_hat_function_derivative(x, mesh.mesh[i - 1], mesh.mesh[i],
                                                                                    1) *
                                                  one_sided_hat_function_derivative(x, mesh.mesh[i], mesh.mesh[i - 1],
                                                                                    1)
                                                  ), mesh.mesh[i - 1], mesh.mesh[i])[0]
        res[i - 1, i] = res[i, i - 1]
    return res


def get_load_vector(fun, mesh, integrate_fun, t=0):
    res = np.zeros(mesh.mesh.size)
    res[0] = \
    integrate_fun(lambda x: (fun(x, t) * one_sided_hat_function(x, mesh.mesh[0], mesh.mesh[1], 1)), mesh.mesh[0],
                  mesh.mesh[1])[0]
    res[-1] = integrate_fun(lambda x: (fun(x, t) * one_sided_hat_function(x, mesh.mesh[-1], mesh.mesh[-2], 1)),
                            mesh.mesh[-2], mesh.mesh[-1])[0]
    for i in range(mesh.mesh.size - 1):
        res[i] += integrate_fun(
            lambda x: (fun(x, t) * two_sided_hat_function(x, mesh.mesh[i - 1], mesh.mesh[i], mesh.mesh[i + 1], 1)),
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


def compute_l2_projection(fun: Callable[[float], float], mesh: Mesh, t=0):
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(fun, mesh, integrate.quad, t)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (get_piecewise_linear_function(mesh, projection_coefficients), projection_coefficients)


def add_boundary_conditions(stiffness_matrix, load_vector, kappa_0, kappa_n, g_0, g_n):
    load_vector[0] += kappa_0 * g_0
    load_vector[-1] += kappa_n * g_n
    stiffness_matrix[0, 0] += kappa_0
    stiffness_matrix[-1, -1] += kappa_n
    # stiffness_matrix[0, 0] = 1
    # stiffness_matrix[0, 1] = 0
    # stiffness_matrix[-1, -1] = 1
    # stiffness_matrix[-1, -2] = 0
    # load_vector[0] = 0
    # load_vector[-1] = 0
    return stiffness_matrix, load_vector


def get_heat_equation_solution(f, kappa_0, kappa_n, g_0, g_n, mesh):
    stiffness_matrix = get_stiffness_matrix(mesh, integrate.quad)
    load_vector = get_load_vector(f, mesh, integrate.quad)
    (stiffness_matrix, load_vector) = add_boundary_conditions(stiffness_matrix, load_vector, kappa_0, kappa_n, g_0, g_n)
    projection_coefficients = np.linalg.solve(stiffness_matrix, load_vector)
    return get_piecewise_linear_function(mesh, projection_coefficients)


def get_time_dependent_heat_equation_solution(f, kappa_0, kappa_n, g_0, g_n, mesh, time_partition):
    res = []
    stiffness_matrix = get_stiffness_matrix(mesh, integrate.quad)
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(f, mesh, integrate.quad, time_partition[0])
    (stiffness_matrix, load_vector) = add_boundary_conditions(stiffness_matrix, load_vector, kappa_0, kappa_n, g_0, g_n)
    x0 = compute_l2_projection(get_initial_temperature, mesh, time_partition[0])[1]
    res.append(x0)
    for i in range(1, len(time_partition)):
        res.append(FDM.backward_euler_one_iteration(mass_matrix, stiffness_matrix, load_vector, res[-1],
                                                    time_partition[i] - time_partition[i - 1]))
    return res


def animate_solution(functions, start, stop, num_of_points):
    fig, ax = plt.subplots()
    artists = []
    lin = np.linspace(start, stop, num_of_points)
    for i in range(len(functions)):
        container = ax.plot(lin, get_piecewise_linear_function(mesh, functions[i])(lin), color="blue")
        artists.append(container)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani.save(filename="/home/szymon/PycharmProjects/FEMSolver/animation.gif", writer="pillow")


if __name__ == "__main__":
    boundary = OneDimensionalBoundary(0, 1)
    mesh = Mesh(boundary, h=0.01)
    np.set_printoptions(linewidth=np.inf)
    lin = np.linspace(0, 1, 100)
    rod_temps = get_time_dependent_heat_equation_solution(get_heat_source_function, 1000000, 1000000, 0, 0, mesh, lin)
    animate_solution(rod_temps, 0, 1, 50)
