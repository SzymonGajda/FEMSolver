import numpy as np
import matplotlib
import matplotlib.pyplot as plt, mpld3
from collections.abc import Callable

from matplotlib import animation
from numpy.ma.core import logical_or
import scipy.integrate as integrate
import FDM
import triangle as tr

matplotlib.use("TkAgg")


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


def is_inside_triangle(A, B, C, P):
    denominator = ((B[1] - C[1]) * (A[0] - C[0]) +
                   (C[0] - B[0]) * (A[1] - C[1]))

    a = ((B[1] - C[1]) * (P[0] - C[0]) +
         (C[0] - B[0]) * (P[1] - C[1])) / denominator
    b = ((C[1] - A[1]) * (P[0] - C[0]) +
         (A[0] - C[0]) * (P[1] - C[1])) / denominator
    c = 1 - a - b

    return np.where(np.logical_and(
        np.logical_and(np.logical_or(np.isclose(a, 0.), a > 0), np.logical_or(np.isclose(b, 0.), b > 0.)),
        np.logical_or(np.isclose(c, 0.), c > 0.)), True, False)


def get_triangle_area(A, B, C):
    return abs((A[0] - C[0]) * (B[1] - A[1]) - (A[0] - B[0]) * (C[1] - A[1])) / 2


def one_sided_hat_function_1D(x: float, beginning: np.float64, end: np.float64, coeff: np.float64):
    return np.where((logical_or(x < min(beginning, end), x > max(beginning, end))), 0,
                    x * coeff / (beginning - end) - end * coeff / (beginning - end))


def two_sided_hat_function_1D(x: float, beginning: np.float64, node: np.float64, end: np.float64, coeff: np.float64):
    return np.where(x < node, one_sided_hat_function_1D(x, node, beginning, coeff),
                    one_sided_hat_function_1D(x, node, end, coeff))


def hat_function_2D(point, main_node, node_2, node_3, coeff):
    return np.where(is_inside_triangle(main_node, node_2, node_3, point), coeff * (
                point[0] * (-node_2[1] + node_3[1]) + point[1] * (node_2[0] - node_3[0]) + (
                    -node_2[0] * node_3[1] + node_2[1] * node_3[0])) / (
                                main_node[1] * node_2[0] - main_node[0] * node_2[1] + main_node[0] * node_3[1] - node_2[
                            0] * node_3[1] -
                                main_node[1] * node_3[0] + node_2[1] * node_3[0]), 0)


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
        integrate_fun(lambda x: (fun(x, t) * one_sided_hat_function_1D(x, mesh.mesh[0], mesh.mesh[1], 1)), mesh.mesh[0],
                      mesh.mesh[1])[0]
    res[-1] = integrate_fun(lambda x: (fun(x, t) * one_sided_hat_function_1D(x, mesh.mesh[-1], mesh.mesh[-2], 1)),
                            mesh.mesh[-2], mesh.mesh[-1])[0]
    for i in range(mesh.mesh.size - 1):
        res[i] += integrate_fun(
            lambda x: (fun(x, t) * two_sided_hat_function_1D(x, mesh.mesh[i - 1], mesh.mesh[i], mesh.mesh[i + 1], 1)),
            mesh.mesh[i - 1], mesh.mesh[i + 1])[0]
    return res


def get_piecewise_linear_function(mesh: Mesh, coeffs: np.ndarray):
    def res_fun(x):
        res = 0
        res += one_sided_hat_function_1D(x, mesh.mesh[0], mesh.mesh[1], coeffs[0])
        res += one_sided_hat_function_1D(x, mesh.mesh[-1], mesh.mesh[-2], coeffs[-1])
        for i in range(1, len(mesh.mesh) - 1):
            res += two_sided_hat_function_1D(x, mesh.mesh[i - 1], mesh.mesh[i], mesh.mesh[i + 1], coeffs[i])
        return res

    return res_fun


def get_piecewise_linear_function_2D(vertices, triangles, coeffs):
    def res_fun(x):
        res = 0
        for triangle in triangles:
            triangle_res = 0
            for vertice in triangle:
                triangle_res += hat_function_2D(x, vertices[vertice], vertices[list(set(triangle) - {vertice})[0]],
                                                vertices[list(set(triangle) - {vertice})[1]], coeffs[vertice])
            res = np.where(triangle_res != 0, triangle_res, res)
        return res

    return res_fun


def compute_l2_projection(fun: Callable[[float], float], mesh: Mesh, t=0):
    mass_matrix = get_mass_matrix(mesh)
    load_vector = get_load_vector(fun, mesh, integrate.quad, t)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (get_piecewise_linear_function(mesh, projection_coefficients), projection_coefficients)


def get_mass_matrix_2D(vertices, triangles):
    mass_matrix = np.zeros((len(vertices), len(vertices)))
    for triangle in triangles:
        area = get_triangle_area(vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]])
        for i in range(3):
            for j in range(3):
                if i == j:
                    mass_matrix[triangle[i]][triangle[i]] += area / 6.
                else:
                    mass_matrix[triangle[i]][triangle[j]] += area / 12.
    return mass_matrix


def get_load_vector_2D(fun, vertices, triangles, integrate_fun, t):
    load_vector = np.zeros(len(vertices))
    for triangle in triangles:
        for vertice in triangle:
            fun_to_integrate = lambda x, y: fun((x, y), t) * hat_function_2D((x, y), vertices[vertice], vertices[
                list(set(triangle) - {vertice})[0]], vertices[list(set(triangle) - {vertice})[1]], 1)
            load_vector[vertice] += integrate_fun(fun_to_integrate, vertices[triangle[0]], vertices[triangle[1]],
                                                  vertices[triangle[2]])
    return load_vector


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


def compute_l2_projection_2D(fun, vertices, triangles, t=0):
    mass_matrix = get_mass_matrix_2D(vertices, triangles)
    load_vector = get_load_vector_2D(fun, vertices, triangles, get_midpoint_quadrature, t)
    projection_coefficients = np.linalg.solve(mass_matrix, load_vector)
    return (get_piecewise_linear_function_2D(vertices, triangles, projection_coefficients), projection_coefficients)


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


def animate_solution(functions, start, stop, num_of_points, mesh):
    fig, ax = plt.subplots()
    artists = []
    lin = np.linspace(start, stop, num_of_points)
    for i in range(len(functions)):
        container = ax.plot(lin, get_piecewise_linear_function(mesh, functions[i])(lin), color="blue")
        artists.append(container)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani.save(filename="/home/szymon/PycharmProjects/FEMSolver/animation.gif", writer="pillow")


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    # lin = np.linspace(0, 1, 100)
    # rod_temps = get_time_dependent_heat_equation_solution(get_heat_source_function, 1000000, 1000000, 0, 0, mesh, lin)
    # animate_solution(rod_temps, 0, 1, 50, mesh)
    data = tr.get_data('double_hex2')
    triangulation = tr.triangulate(data, "pqa.1")
    vertices = triangulation["vertices"]
    triangles = triangulation["triangles"]
    projection = \
    compute_l2_projection_2D(lambda x, t: np.sin(x[0] * 2 * np.pi) + np.cos(x[1] * np.pi), vertices, triangles)[0]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.linspace(0, 1, 500)
    Y = np.linspace(0, 1, 500)
    X, Y = np.meshgrid(X, Y)
    plot = ax.plot_surface(X, Y, projection((X, Y)))
    plt.show()
