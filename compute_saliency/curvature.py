import numpy as np
from trimesh import util
from trimesh.curvature import line_ball_intersection

def mod_discrete_mean_curvature_measure(mesh, points, radius):
    """MODIFIED
    Return the discrete mean curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.

    This is the sum of the angle at all edges contained in the
    sphere for each point.

    Parameters
    ----------
    points : (n, 3) float
      Points in space
    radius : float
      Sphere radius which should typically be greater than zero

    Returns
    --------
    mean_curvature : (n,) float
      Discrete mean curvature measure.
    """

    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    # axis aligned bounds
    bounds = np.column_stack((points - radius,
                              points + radius))

    # line segments that intersect axis aligned bounding box
    candidates = [list(mesh.face_adjacency_tree.intersection(b))
                  for b in bounds]

    mean_curv = np.empty(len(points))
    num_neighbours = np.empty(len(points))
    for i, (x, x_candidates) in enumerate(zip(points, candidates)):
        endpoints = mesh.vertices[mesh.face_adjacency_edges[x_candidates]]
        lengths = line_ball_intersection(
            endpoints[:, 0],
            endpoints[:, 1],
            center=x,
            radius=radius)
        angles = mesh.face_adjacency_angles[x_candidates]
        signs = np.where(mesh.face_adjacency_convex[x_candidates], 1, -1)
        mean_curv[i] = (lengths * angles * signs).sum() / 2
        num_neighbours[i] = np.shape(lengths)[0]

    return mean_curv, num_neighbours

