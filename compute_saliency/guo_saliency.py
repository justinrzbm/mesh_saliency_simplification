import numpy as np
import open3d as o3d
import trimesh
from knn import kneighbors_all


def saliency_covariance_descriptors(points, max_k=16, r_scale=10):
    # max r relative to scale of point cloud
    curvature_r = (points.max()-points.min()) / 100 * r_scale
    ball_r = (points.max()-points.min()) / 100 * r_scale / 1000

    # Convert to o3d PointCloud and estimate normal vectors
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_k))
    pc = pc.normalize_normals()
    normals = np.asarray(pc.normals)

    # compute curvature
    curvature = get_curvature(pc)
    # # Convert point cloud to mesh
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, ball_r)
    # # discard (second?) return item
    # if type(mesh)==tuple:
    #     mesh = mesh[0]
    #     assert type(mesh)==o3d.geometry.TriangleMesh
    # # Convert Open3D mesh to trimesh
    # mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
    # # Compute Gaussian curvature using trimesh  (slow)
    # curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, radius=curvature_r)
    # curvature = curvature/max(curvature)    # normalize

    # Make feature vector at each point (Nx, Ny, Nz, K)
    F = np.zeros((len(pc.points), 4))
    for i in range(len(F)):
        F[i, 0:3] = normals[i]
        F[i, 3] = curvature[i]

    pass

def get_curvature(pc):
    knn = kneighbors_all(pc)
    # define normal at this point as mean of neighbouring normals for stability
    guassian_curvature = np.empty(len(pc.points), dtype=np.double)
    normals = np.asarray(pc.normals)
    for i in range(len(pc.points)):
        C = np.cov(normals[knn[i]].T)
        eigenvalues = np.linalg.eig(C)[0]
        guassian_curvature[i] = np.prod(eigenvalues)

    guassian_curvature /= np.max(guassian_curvature)    # normalize these values
    return guassian_curvature