import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import random

# filename = 'compute_saliency/bunny.obj'
# filename = './compute_saliency/object/bunny.obj'
# filename = './models/young_boy_head_obj.obj'
# filename = './models/cube.obj'
# filename = './models/dragon.obj'
# filename = './models/bunny.obj'
# filename = './models/bunnysmall.obj'

def gaussian_kernel_density_estimation(x, points, bandwidth=1.0):
    pairwise_distance = torch.cdist(x, points)
    densities = torch.exp(-pairwise_distance**2 / (2 * bandwidth**2)).mean(dim=1)
    print(densities.shape)
    return densities

def kl_divergence(point_cloud_p, point_cloud_q, bandwidth=1.0, epsilon=1e-10):
    # We estimate the densities of point_cloud_p based on its own set of points
    p_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_p, bandwidth)
    
    # Then, we estimate the densities of point_cloud_p based on point_cloud_q
    q_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_q, bandwidth)
    
    kl = (torch.log(p_density + epsilon) - torch.log(q_density + epsilon)).mean()
    return kl

def main():
    # device = torch.device("cuda:9" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:9")
    # Load mesh object
    # point cloud 1
    filename1 = './models/bunny.obj'
    mesh1 = trimesh.load(filename1, file_type='obj')
    points1 = mesh1.vertices
    points_np1 = points1.view(np.ndarray) # N, 3 (x, y, z)
    # point cloud 2
    filename2 = './models/dragon.obj'
    mesh2 = trimesh.load(filename2, file_type='obj')
    points2 = mesh2.vertices
    points_np2 = points2.view(np.ndarray) # N, 3 (x, y, z)
    np.random.shuffle(points_np2)
    points_np2 = points_np2[:35000]

    start_time = time.time()
    kl = kl_divergence(torch.from_numpy(points_np1), torch.from_numpy(points_np2))
    print(kl)
    print("--- %.6s seconds runtime ---" % (time.time() - start_time))

if __name__=='__main__':
    main()