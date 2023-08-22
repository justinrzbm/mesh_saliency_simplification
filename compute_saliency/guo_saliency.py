import numpy as np
import open3d as o3d
import trimesh
from knn import kneighbors_all
from sklearn import decomposition

def saliency_covariance_descriptors(points, max_k=16, r_scale=10):
    # max r relative to scale of point cloud
    curvature_r = (points.max()-points.min()) / 100 * r_scale
    ball_r = (points.max()-points.min()) / 100 * r_scale / 1000


    # Convert to o3d PointCloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Estimate normal vectors
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=max_k))
    pc = pc.normalize_normals()
    normals = np.asarray(pc.normals)

    # Get k-NN
    knn = kneighbors_all(pc)

    # compute curvature
    curvature = get_curvature(pc, knn)
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

    # Compute covariance descriptor at each point
    C = np.zeros((len(pc.points), 4, 4))
    for i in range(len(F)):
        local_mean = np.mean(F[knn[i]], axis=0)
        C[i,:,:] = np.outer((F[i]-local_mean), (F[i]-local_mean))

    # in the paper, the covariance for this point is defined as the
    # average covariance matrix over the neighborhood
    C_ = np.zeros_like(C)
    for i in range(len(F)):
        C_[i] = 1/(len(knn[i])) * np.sum(C[knn[i]], axis=0)
    C = C_

    # Compute Sigma Set
    S = np.zeros((len(C), 8, 4))
    alpha = np.sqrt(4)
    for i in range(len(C)):
        M = np.linalg.cholesky(C[i])
        for j in range(4):
            S[i,j,:] = alpha * M[:,j]
        for j in range(4):
            S[i,j+4,:] = -alpha * M[:,j]


    # PCA
    # The paper is a bit confusing on how PCA is applied to the collection of sigma sets.
    # What should be implemented, a PCA dimensionality reduciton across all points down to 8 principal components,
    # (each the size of one point feature 8x4)
    # Then, the L1 norm should be taken of the feature of a given point in the global PCA coordinate system to 
    # calculate the saliency of that point.
    # We do this distance calculation by flattening the feature into a 1x32 vector, but the paper does not explain
    # how they specifically did this comparison.
    # In addition, we are assuming that sklearn.decomposition.PCA calculates the point average feature as mean(S, axis=0)
    feature_vectors = S.reshape((len(C), 32))
    pca = decomposition.PCA(n_components=8)
    pca.fit(feature_vectors)
    transformed_feature = pca.transform(feature_vectors)
    sal = np.sum(np.abs(transformed_feature), axis=1)

    return sal

def get_curvature(pc, knn):
    # define normal at this point as mean of neighbouring normals for stability
    guassian_curvature = np.empty(len(pc.points), dtype=np.double)
    normals = np.asarray(pc.normals)
    for i in range(len(pc.points)):
        C = np.cov(normals[knn[i]].T)
        eigenvalues = np.linalg.eig(C)[0]
        guassian_curvature[i] = np.prod(eigenvalues)

    guassian_curvature /= np.max(guassian_curvature)    # normalize these values
    return guassian_curvature
