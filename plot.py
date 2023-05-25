import trimesh
import matplotlib.pyplot as plt
import numpy as np
import argparse

def scatter3d(points, cmap=None):
    if cmap is not None:
        assert len(cmap.shape)==1 and cmap.shape[0] == points.shape[0], \
            f"cmap of size {cmap.shape} does not match size of x: {points.shape}"

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection="3d")

    size = 20
    edgec = 'none'
    # ax.view_init(elev=110., azim=-90)
    ax.set_title(f'')
    if cmap is not None:
        ax.scatter(*points.T, s=size, c=cmap, cmap='viridis')
    else:
        ax.scatter(*points.T, s=size, ec='w', c='blue')

    plt.savefig(f"results/test.png")
    plt.show()


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
    parser.add_argument('-o', type=str, default=None, help='Please provide the output file path of the plot image.')
    args=parser.parse_args()
    input_filepath=args.i
    output_filepath=args.o

    plotmesh(input_filepath, output_filepath)

# test_sal = np.ones(2503, dtype=np.double)   # hardcoded n_vertices for bunnysmall

