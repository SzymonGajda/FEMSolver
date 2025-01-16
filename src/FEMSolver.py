import numpy as np
import matplotlib

import triangle as tr

from src.FEM2D import get_time_dependent_wave_equation_2D_solution
from src.Plotting import animate_2D

matplotlib.use("TkAgg")


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


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    # lin = np.linspace(0, 1, 100)
    # rod_temps = get_time_dependent_heat_equation_solution(get_heat_source_function, 1000000, 1000000, 0, 0, mesh, lin)
    # animate_solution(rod_temps, 0, 1, 50, mesh)
    #data = tr.get_data('double_hex2')
    #data = tr.get_data('box.1')
    #triangulation = tr.triangulate(data, "pqa.001")
    v = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.2], [0, 0.3], [0,0.7], [0,0.8], [-0.3,0.2], [-0.3, 0.3], [-0.3,0.7], [-0.3,0.8]]
    s = [[0, 1], [1, 2], [2, 3], [3, 7], [7, 11], [11, 10], [10, 6], [6, 5], [5, 9], [9, 8], [8, 4], [4, 0]]
    triangulation = tr.triangulate({'vertices': v, 'segments': s}, 'pq30a.001')
    #triangulation = tr.triangulate({'vertices': v}, 'q30a.001')
    vertices = triangulation["vertices"]
    triangles = triangulation["triangles"]
    vertex_markers = triangulation["vertex_markers"]
    #print(triangulation)
    time_partition = np.linspace(0, 3, 500)
    res = get_time_dependent_wave_equation_2D_solution(triangles, vertices, vertex_markers, time_partition)
    #for r in res:
    #    print(r)
    #res_fun = get_piecewise_linear_function_2D(vertices, triangles, res[99])
    animate_2D(res, -0.3, 0, 1, 1, 100, vertices, triangles)
    #animate_solution_2D(res, 0, 0, 1, 1, 100, vertices, triangles)
    # projection = \
    #    compute_l2_projection_2D(lambda x, t: np.sin(x[0] * 2 * np.pi) + np.cos(x[1] * np.pi), vertices, triangles)[0]
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #X = np.linspace(0, 1, 50)
    #Y = np.linspace(0, 1, 50)
    #X, Y = np.meshgrid(X, Y)
    #plot = ax.plot_surface(X, Y, res_fun((X, Y)))
    #plt.show()
