import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# filename = 'compute_saliency/bunny.obj'
# filename = './compute_saliency/object/bunny.obj'
# filename = './models/young_boy_head_obj.obj'
filename = './models/dragon.obj'
# filename = './models/bunnysmall.obj'

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
    plt.show()

def scatter2d(points, title='points'):
    fig = plt.figure(figsize=(5,5))
    plt.title('points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(*points.T)
    plt.savefig(f"compute_saliency/projected/{title}.png")
    plt.show()


def main():
    # Load mesh object
    mesh = trimesh.load(filename, file_type='obj')
    start_time = time.time()
    print(f"{filename}: Processing {len(mesh.vertices)} vertices")

    points = mesh.vertices
    # parse points
    points_np = points.view(np.ndarray) # N, 3 (x, y, z)
    np.random.shuffle(points_np)
    sampled_points = points_np[:1250]
    
    scatter2d(sampled_points[:,[1, 2]])

    print("--- %.6s seconds runtime ---" % (time.time() - start_time))
    scatter3d(mesh.vertices, title='original_points')
    scatter3d(sampled_points, title='sampled_points')

if __name__=='__main__':
    main()