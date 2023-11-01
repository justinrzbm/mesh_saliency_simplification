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
    return densities

def kl_divergence_pc(point_cloud_p, point_cloud_q, bandwidth=1.0, epsilon=1e-10):
    # We estimate the densities of point_cloud_p based on its own set of points
    p_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_p, bandwidth)
    
    # Then, we estimate the densities of point_cloud_p based on point_cloud_q
    q_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_q, bandwidth)
    
    kl = (torch.log(p_density + epsilon) - torch.log(q_density + epsilon)).mean()
    return kl

def read_off_file(filename):
    """
    Read a .off file and return a 3D point cloud as a torch.tensor.

    Args:
        filename (str): Path to the .off file.

    Returns:
        torch.Tensor: A tensor containing the 3D point cloud.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Parse the header
    num_vertices, num_faces, _ = map(int, lines[1].strip().split())

    # Read vertices
    vertices = []
    for line in lines[2:2 + num_vertices]:
        x, y, z = map(float, line.strip().split())
        vertices.append([x, y, z])

    # Read faces and convert them to a list of vertex indices
    faces = []
    for line in lines[2 + num_vertices:]:
        parts = list(map(int, line.strip().split()))
        num_face_vertices = parts[0]
        face_indices = parts[1:]
        faces.append(face_indices)

    # Convert the data to a torch.tensor
    vertices = torch.tensor(vertices, dtype=torch.float32)
    return vertices

def main():
    # device = torch.device("cuda:9" if (torch.cuda.is_available()) else "cpu")
    device = torch.device("cuda:9")
    # Load mesh object
    start_time = time.time()
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
    
    kl = kl_divergence_pc(torch.from_numpy(points_np1), torch.from_numpy(points_np2))
    print(kl)
    print("--- %.6s seconds runtime ---" % (time.time() - start_time))

if __name__=='__main__':
    main()