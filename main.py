import trimesh
import trimesh.curvature
import numpy as np
import matplotlib.pyplot as plt
import time
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
CURVATURE_R = 4
ENTROPY_R = 8
NBINS = 8

def scatter3d_twoplots(points, title='saliency', cmap=None, cmap2=None):
    if cmap is not None:
        assert len(cmap.shape)==1 and cmap.shape[0] == points.shape[0], \
            f"cmap of size {cmap.shape} does not match size of x: {points.shape}"

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection="3d")

    size = 20
    edgec = 'none'
    ax.view_init(elev=110., azim=-90)
    ax.set_title(f'Curvature; r={CURVATURE_R}')
    if cmap is not None:
        ax.scatter(*points.T, s=size, c=cmap, cmap='viridis')
    else:
        ax.scatter(*points.T, s=size, ec='w', c='blue')

    if cmap2 is not None:
        assert len(cmap.shape)==1 and cmap.shape[0] == points.shape[0], \
            f"cmap of size {cmap.shape} does not match size of x: {points.shape}"
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.view_init(elev=110., azim=-90)
        ax2.set_title(f'LCE; r={ENTROPY_R}')
        if cmap is not None:
            ax2.scatter(*points.T, s=size, c=cmap2, cmap='viridis')
        else:
            ax2.scatter(*points.T, s=size, ec='w', c='blue')
    plt.savefig(f"compute_saliency/results/{title}_lce_cr{CURVATURE_R}_er{ENTROPY_R}_nb{NBINS}.png")
    plt.show()

def scatter3d(points, title='saliency', cmap=None):
    if cmap is not None:
        assert len(cmap.shape)==1 and cmap.shape[0] == points.shape[0], \
            f"cmap of size {cmap.shape} does not match size of x: {points.shape}"

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection="3d")

    size = 20
    edgec = 'none'
    ax.view_init(elev=110., azim=-90)
    ax.set_title(title)
    if cmap is not None:
        ax.scatter(*points.T, s=size, c=cmap, cmap='viridis')
    else:
        ax.scatter(*points.T, s=size, ec='w', c='blue')

    plt.savefig(f"compute_saliency/saliency_results/{title}.png")
    # plt.show()

def plot_3d_point_cloud(X):
    """
    Plots a 3D point cloud.

    Parameters:
        X (numpy array): A NumPy array of shape (n, 3) containing the x, y, z coordinates of the points.
    """

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
    # ax = plt.gca(projection='3d')
    
    # Scatter plot
    size = 20
    ax.view_init(elev=110., azim=-90)
    ax.scatter(x_coords, y_coords, z_coords, s=size, ec='w', c='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show(block=True)

def main():
    # Load mesh object
    # with open(filename, 'r') as f:
    mesh = trimesh.load(filename, file_type='obj')
    # # consider as point cloud, compute KNN to compute neighbors -- this will not work since mesh is not convex
    # cloud = trimesh.points.PointCloud(mesh.vertices)
    # mesh = cloud.convex_hull
    start_time = time.time()
    print(f"{filename}: Processing {len(mesh.vertices)} vertices")

    # curvature = local_curvature_pca(mesh, curvature_r=CURVATURE_R, entropy_r=ENTROPY_R, nbins=NBINS, compute_entropy=False)
    # lce = local_curvature_pca(mesh, curvature_r=CURVATURE_R, entropy_r=ENTROPY_R, nbins=NBINS, compute_entropy=True)
    points = mesh.vertices
    sal = saliency_covariance_descriptors(points, r_scale=10)

    print(sal)
    simplification_ratio = 0.25
    sampled_points = mesh.vertices[sal > np.percentile(sal, (1 - simplification_ratio) * 100)]
    print(f"Plotting {sampled_points.shape[0]} points")

    print("--- %.6s seconds runtime ---" % (time.time() - start_time))
    title='guo_flattened'
    # scatter3d(mesh.vertices, title=title, cmap=sal)
    # scatter3d(mesh.vertices, title=title)
    np.save(f"compute_saliency/{title}", sal)

    plot_3d_point_cloud(sampled_points)

if __name__=='__main__':
    main()