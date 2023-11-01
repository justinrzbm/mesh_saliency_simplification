import trimesh
import matplotlib.pyplot as plt
import numpy as np
import argparse

def scatter3d(input_filepath:str, output_filepath:str, cmap=None):
    if input_filepath[-4:]=='.obj':
        mesh = trimesh.load_mesh(input_filepath)
        points = mesh.vertices
    elif input_filepath[-4:]=='.npy':
        points = np.load(input_filepath)
    else: raise NotImplementedError

    if cmap is not None:
        cmap = np.load(cmap)
        print(cmap)
        assert len(cmap.shape)==1 and cmap.shape[0] == points.shape[0], \
            f"cmap of size {cmap.shape} does not match size of x: {points.shape}"

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d")

    size = 20
    edgec = 'none'
    ax.view_init(elev=110., azim=-90)
    if cmap is not None:
        ax.scatter(*points.T, s=size, c=cmap, cmap='viridis')
    else:
        ax.scatter(*points.T, s=size, ec='w', c='blue')

    plt.savefig(output_filepath)
    plt.show()
    plt.close()


def plotmesh(mesh_file:str, file_out:str):
    mesh = trimesh.load_mesh(mesh_file)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
    ax.view_init(elev=110., azim=-90)

    plt.savefig(file_out)
    plt.show()
    plt.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Plot mesh')
    parser.add_argument('-i', type=str, default=None, help='Please provide the input file path of an existing 3d model.')
    parser.add_argument('-s', type=str, default=None, help='Please provide the saliency file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default=None, help='Please provide the output file path of the plot image.')
    parser.add_argument('-mesh', action='store_true', help='Flag to plot as a mesh or point cloud')
    args=parser.parse_args()
    input_filepath=args.i
    saliency_filepath = args.s
    output_filepath=args.o

    if args.mesh:
        plotmesh(input_filepath, output_filepath)
    else:   # point cloud
        scatter3d(input_filepath, output_filepath, saliency_filepath)

# test_sal = np.ones(2503, dtype=np.double)   # hardcoded n_vertices for bunnysmall

