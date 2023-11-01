import trimesh
import trimesh.curvature
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from compute_saliency.curvature import mod_discrete_mean_curvature_measure
from compute_saliency.pca_saliency import local_curvature_pca
from compute_saliency.guo_saliency import saliency_covariance_descriptors

# filename = 'compute_saliency/bunny.obj'
# filename = './compute_saliency/object/bunny.obj'
# filename = './models/young_boy_head_obj.obj'
# filename = './models/cube.obj'
# filename = './models/dragon.obj'
filename = './models/bunny.obj'
# filename = './models/bunnysmall.obj'

def plot_3d_point_cloud(X):
    # Ensure the input is of shape (n, 3)
    assert X.shape[1] == 3, "Input should be of shape (n, 3)!"
    # Extract x, y, and z coordinates
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    z_coords = X[:, 2]
    # Create a new figure
    fig = plt.figure()
    # Add 3D subplot
    ax = fig.add_subplot(111, projection='3d')   
    # Scatter plot
    size = 20
    ax.view_init(elev=110., azim=-90)
    ax.scatter(x_coords, y_coords, z_coords, s=size, ec='w', c='blue')
    # Plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    # Show
    plt.show(block=True)

def fit_plane(points_subset):
    center = points_subset.mean(dim=0)
    covariance = torch.mm((points_subset - center).T, (points_subset - center))
    _, _, V = torch.svd(covariance)
    normal = V[:, -1]
    return normal, center

def get_neighborhood(point, points, radius=0.1):
    distances = torch.norm(points - point, dim=1)
    return points[distances < radius]

def get_neighborhood_k(point, points, k=32):
    distances = torch.norm(points - point, dim=1)
    _, indices = torch.topk(distances, int(k), largest=False)
    return points[indices]

def fit_plane(points_subset):
    center = points_subset.mean(dim=0)
    covariance = torch.mm((points_subset - center).T, (points_subset - center))
    _, _, V = torch.svd(covariance)
    normal = V[:, -1]
    return normal, center

def get_neighborhood(point, points, radius=0.1):
    distances = torch.norm(points - point, dim=1)
    return points[distances < radius]

def compute_qem(point, normal, point_on_plane):
    d = -torch.dot(normal, point_on_plane)
    qem = torch.dot(normal, point) + d
    return qem**2

def main():
    # Load mesh object
    # with open(filename, 'r') as f:
    mesh = trimesh.load(filename, file_type='obj')
    # # consider as point cloud, compute KNN to compute neighbors -- this will not work since mesh is not convex
    # cloud = trimesh.points.PointCloud(mesh.vertices)
    # mesh = cloud.convex_hull
    start_time = time.time()
    print(f"{filename}: Processing {len(mesh.vertices)} vertices")

    radius = 0.1
    k = 32
    coef = 0.2
    simplification_ratio = 0.1
    # Saliency by Sigma Set
    points = mesh.vertices
    sal = saliency_covariance_descriptors(points, r_scale=10)
    sal = sal / sal.max()
    # QEM error
    points = torch.from_numpy(points.view(np.ndarray))
    point_errors = torch.zeros(points.size(0))
    for i, point in enumerate(points):
        # neighborhood = get_neighborhood(point, points, radius)
        neighborhood = get_neighborhood_k(point, points, k)
        normal, point_on_plane = fit_plane(neighborhood)
        point_errors[i] = compute_qem(point, normal, point_on_plane)
    point_errors = point_errors / point_errors.max()
    point_errors -= coef * sal
    _, indices = torch.topk(point_errors, int(points.size(0) * simplification_ratio), largest=False)
    simplified_points = points[indices]

    print(f"Plotting {simplified_points.shape[0]} points")
    print("--- %.6s seconds runtime ---" % (time.time() - start_time))
    plot_3d_point_cloud(simplified_points)



if __name__=='__main__':
    main()