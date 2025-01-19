import triangle as tr

from src.plotting import plot_triangulation


def generate_wing_geometry():
    plane_vertices = [(-30, -15), (30, -15), (30, 15), (-30, 15)]
    wing_vertices = [
        (17.7218, 1.5737),
        (16.0116, 1.6675),
        (9.0610, 1.3668),
        (-0.5759, -0.1102),
        (-9.5198, -1.8942),
        (-15.6511, -2.5938),
        (-18.1571, -1.7234),
        (-16.9459, 0.2051),
        (-12.4137, 2.2238),
        (-5.4090, 3.4543),
        (2.8155, 3.5046),
        (10.6777, 2.6664),
        (16.3037, 1.7834),
    ]
    vertices = plane_vertices + wing_vertices
    segments = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 4),
    ]
    holes = [(-10, 0)]
    return dict(vertices=vertices, segments=segments, holes=holes)


if __name__ == "__main__":
    data = generate_wing_geometry()
    plot_triangulation(data)
