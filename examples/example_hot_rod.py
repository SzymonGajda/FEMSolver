import numpy as np

from src.fem1d import (
    get_time_dependent_heat_equation_solution,
    get_piecewise_linear_function,
)
from src.plotting import animate
from src.quadratures import scipy_1d_quadrature_wrapper


def rode_cross_section_area(x):
    return 1


def rode_conductivity(x):
    return 0.1 * (5 - 0.6 * x)


def heat_source(x, t):
    return np.where(np.logical_or(x < 0.2, x > 0.4), 0, 3)


def initial_temperature(x):
    return 0.5 - abs(x - 0.5)


if __name__ == "__main__":
    mesh = list(np.linspace(0, 1, 100))
    time_partition = list(np.linspace(0, 3, 400))
    solution_coefficients = get_time_dependent_heat_equation_solution(
        heat_source,
        1000000,
        1000000,
        0,
        0,
        mesh,
        time_partition,
        rode_cross_section_area,
        rode_conductivity,
        initial_temperature,
        scipy_1d_quadrature_wrapper,
    )
    piecewise_linear_functions = []
    for coefficients in solution_coefficients:
        piecewise_linear_functions.append(
            get_piecewise_linear_function(mesh, coefficients)
        )
    animate(
        piecewise_linear_functions,
        0,
        1,
        100,
        mesh,
        "/home/FEMAnimations/hot_rod.gif",
    )
