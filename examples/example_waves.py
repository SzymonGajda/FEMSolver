import triangle as tr
import numpy as np

from src.fem2d import (
    get_time_dependent_wave_equation_2d_solution,
    get_piecewise_linear_function_2d,
)
from src.plotting import animate_2d


def generate_plane_geometry():
    vertices = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0.2],
        [0, 0.3],
        [0, 0.7],
        [0, 0.8],
        [-0.3, 0.2],
        [-0.3, 0.3],
        [-0.3, 0.7],
        [-0.3, 0.8],
    ]
    segments = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 7],
        [7, 11],
        [11, 10],
        [10, 6],
        [6, 5],
        [5, 9],
        [9, 8],
        [8, 4],
        [4, 0],
    ]
    return dict(vertices=vertices, segments=segments)


if __name__ == "__main__":
    geometry_data = generate_plane_geometry()
    triangulation = tr.triangulate(geometry_data, "pq30a.001")
    vertices = triangulation["vertices"]
    triangles = triangulation["triangles"]
    vertex_markers = triangulation["vertex_markers"]
    time_partition = np.linspace(0, 5, 500)

    solution_coefficients = get_time_dependent_wave_equation_2d_solution(
        triangles, vertices, vertex_markers, time_partition
    )
    piecewise_linear_functions = []
    for coefficients in solution_coefficients:
        piecewise_linear_functions.append(
            get_piecewise_linear_function_2d(vertices, triangles, coefficients)
        )

    animate_2d(
        piecewise_linear_functions,
        -0.3,
        0,
        1,
        1,
        100,
        vertices,
        triangles,
        path_to_save="/home/FEMAnimations/waves.gif",
    )
