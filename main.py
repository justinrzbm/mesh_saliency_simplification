import trimesh
import trimesh.curvature
import numpy as np
import matplotlib.pyplot as plt
import time
from compute_saliency.curvature import mod_discrete_mean_curvature_measure
from compute_saliency.pca_saliency import local_curvature_pca
from compute_saliency.guo_saliency import saliency_covariance_descriptors

filename = 'compute_saliency/bunny.obj'
CURVATURE_R = 4
ENTROPY_R = 8
NBINS = 8

def scatter3d(points, cmap=None, cmap2=None):
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
    plt.savefig(f"compute_saliency/results/lce_cr{CURVATURE_R}_er{ENTROPY_R}_nb{NBINS}.png")
    plt.show()

def main():
    # Load mesh object
    mesh = trimesh.load(filename)
    # # consider as point cloud, compute KNN to compute neighbors -- this will not work since mesh is not convex
    # cloud = trimesh.points.PointCloud(mesh.vertices)
    # mesh = cloud.convex_hull
    start_time = time.time()
    print(f"{filename}: Processing {len(mesh.vertices)} vertices")

    # curvature = local_curvature_pca(mesh, curvature_r=CURVATURE_R, entropy_r=ENTROPY_R, nbins=NBINS, compute_entropy=False)
    # lce = local_curvature_pca(mesh, curvature_r=CURVATURE_R, entropy_r=ENTROPY_R, nbins=NBINS, compute_entropy=True)
    points = mesh.vertices
    sal = saliency_covariance_descriptors(points)

    print("--- %.6s seconds runtime ---" % (time.time() - start_time))
    # scatter3d(mesh.vertices, cmap=curvature, cmap2=lce)
    # np.save("/compute_saliency/bunny_sal", sal)

if __name__=='__main__':
    main()