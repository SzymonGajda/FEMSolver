import matplotlib
import numpy as np
import triangle as tr
from matplotlib import pyplot as plt, animation, cm
from matplotlib.animation import FuncAnimation

from src.FEM1D import get_piecewise_linear_function
from src.FEM2D import get_piecewise_linear_function_2D


def animate_solution(functions, start, stop, num_of_points, mesh):
    fig, ax = plt.subplots()
    artists = []
    lin = np.linspace(start, stop, num_of_points)
    for i in range(len(functions)):
        container = ax.plot(lin, get_piecewise_linear_function(mesh, functions[i])(lin), color="blue")
        artists.append(container)
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani.save(filename="/home/szymon/PycharmProjects/FEMSolver/animation.gif", writer="pillow")


def animate_2D(functions, x0, y0, xn, yn, num_of_points, vertices, triangles, path="/home/szymon/PycharmProjects/FEMSolver/animation10.gif"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(x0, xn, num_of_points)
    Y = np.linspace(y0, yn, num_of_points)
    X, Y = np.meshgrid(X, Y)
    norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1, clip=False)

    def animate(n):
        print(f"Animowanie: iteracja {n} z {len(functions) - 1}")
        ax.cla()
        Z = get_piecewise_linear_function_2D(vertices, triangles, functions[n])((X, Y))

        ax.plot_surface(X, Y, Z, norm = norm, cmap = cm.coolwarm, antialiased=True)
        ax.set_zlim(-0.5, 0.5)

        return fig,

    anim = FuncAnimation(fig=fig, func=animate, frames=len(functions), interval=100, repeat=False)
    anim.save(path, writer='pillow')

def plot_triangulation(triangulation):
    ax = plt.subplot()
    tr.plot(ax, **triangulation)
    plt.show()