"""Module provides methods for visualising FEM solutions and meshes."""

import matplotlib
import numpy as np
import triangle as tr
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(functions, x0, xn, num_of_points, mesh, path_to_save):
    """
    Generates and saves an animation by visualizing multiple one
    dimensional piecewise linear functions over a specified grid.

    This function takes a sequence of piecewise linear functions, applies each
    of them to a linearly spaced grid, and visualizes their results as an
    animated sequence of plots. The animation is saved to a specified file.

    :param functions: A list of piecewise linear functions that will be visualized.
    :param x0: The starting value of the linear space grid.
    :param xn: The ending value of the linear space grid.
    :param num_of_points: The number of points to be generated in the linear space grid.
    :param mesh: A collection of support points defining piecewise linear functions.
    :param path_to_save: File path where the animation will be saved.
    :return: This function does not return any value.
    """
    fig, ax = plt.subplots()
    artists = []
    grid = np.linspace(x0, xn, num_of_points)
    for function in functions:
        container = ax.plot(grid, function(grid), color="blue")
        artists.append(container)
    anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)
    anim.save(filename=path_to_save, writer="pillow")


def animate_2d(
    functions, x0, y0, xn, yn, num_of_points, vertices, triangles, path_to_save
):
    """
    Generates and saves a 2D animation based on provided functions over a specified
    grid. It creates a 3D plot for each frame, representing the provided functions
    interpolated over a triangular mesh. The animation is saved in the specified
    location in GIF format.

    :param functions: A list of 2D piecewise linear functions evaluated at each frame
                      of the animation.
    :param x0: The starting x-coordinate of the grid.
    :param y0: The starting y-coordinate of the grid.
    :param xn: The ending x-coordinate of the grid.
    :param yn: The ending y-coordinate of the grid.
    :param num_of_points: The number of points in each dimension of the grid
                          representing the piecewise linear functions.
    :param vertices: An array or list defining the vertices of the triangular mesh
                     used for interpolation.
    :param triangles: A collection representing the indices of the vertices
                      defining the triangular elements of the mesh.
    :param path_to_save: The file path where the animation will be saved.
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x_grid = np.linspace(x0, xn, num_of_points)
    y_grid = np.linspace(y0, yn, num_of_points)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1, clip=False)

    def plot_frame(n):
        print(f"Animowanie: iteracja {n} z {len(functions) - 1}")
        ax.cla()
        z = functions[n]((x_grid, y_grid))
        ax.plot_surface(
            x_grid, y_grid, z, norm=norm, cmap=cm.coolwarm, antialiased=True
        )
        ax.set_zlim(-0.5, 0.5)
        return (fig,)

    anim = FuncAnimation(
        fig=fig, func=plot_frame, frames=len(functions), interval=100, repeat=False
    )
    anim.save(path_to_save, writer="pillow")


def plot_triangulation(triangulation: dict):
    """
    Plots the given triangulation on a subplot.

    This function accepts a dictionary containing the triangulation
    data. It initializes a subplot, plots the triangulated data using
    the provided dictionary, and displays the plot.

    :param triangulation: A dictionary that defines the configuration
        data required for plotting the triangulated data.
    :return: None. The function produces a visual plot and does not
        return any value.

    """
    ax = plt.subplot()
    tr.plot(ax, **triangulation)
    plt.show()
