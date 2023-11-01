import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import pyvista as pv
import vtk
pv.global_theme.window_backend = 'x'

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

def plot_3d_point_cloud_mesh(points):
    # Create a PolyData from the numpy array
    cloud = pv.PolyData(points)

    # Perform triangulation using ball pivoting
    reconstructed_mesh = ball_pivoting(cloud)

    # Visualize the reconstructed mesh
    # reconstructed_mesh.plot(show_edges=True, color="white")
    file_name = "reconstructed_mesh.vtk"
    reconstructed_mesh.save(file_name)

def ball_pivoting(cloud, radius=None):
    """
    Perform the Ball Pivoting Algorithm for meshing.
    """
    if radius is None:
        radius = cloud.length / 100
    bpa = vtk.vtkBallPivotFilter()
    bpa.SetInputData(cloud)
    bpa.SetRadius(radius)
    bpa.Update()
    return pv.wrap(bpa.GetOutput())

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
    device = torch.device("cuda:9" if (torch.cuda.is_available()) else "cpu")
    # Load mesh object
    # filename = 'compute_saliency/bunny.obj'
    filename = './compute_saliency/object/bunny.obj'
    # filename = './models/young_boy_head_obj.obj'
    # filename = './models/cube.obj'
    # filename = './models/dragon.obj'
    # filename = './models/bunny.obj'
    # filename = './models/bunnysmall.obj'
    mesh = trimesh.load(filename, file_type='obj')
    points = mesh.vertices
    points = torch.from_numpy(points.view(np.ndarray))

    start_time = time.time()
    radius = 0.1
    k = 32
    simplify_ratio = 0.1
    point_errors = torch.zeros(points.size(0))
    for i, point in enumerate(points):
        # neighborhood = get_neighborhood(point, points, radius)
        neighborhood = get_neighborhood_k(point, points, k)
        normal, point_on_plane = fit_plane(neighborhood)
        point_errors[i] = compute_qem(point, normal, point_on_plane)

    _, indices = torch.topk(point_errors, int(points.size(0) * simplify_ratio), largest=False)
    simplified_points = points[indices]
    print(simplified_points.shape)
    plot_3d_point_cloud(simplified_points)

    print("--- %.6s seconds runtime ---" % (time.time() - start_time))

if __name__=='__main__':
    main()