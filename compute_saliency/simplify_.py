from quad_mesh_simplify import simplify_mesh
import trimesh
import matplotlib.pyplot as plt
import numpy as np

def plotmesh(mesh):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
    plt.show()

filename = 'bunny.obj'
mesh = trimesh.load(filename)
test_sal = np.zeros(len(mesh.vertices), dtype=np.double)
new_pos, new_face = simplify_mesh(np.array(mesh.vertices), np.array(mesh.faces, dtype=np.uint32), test_sal, len(mesh.vertices)//16)
new_mesh = trimesh.Trimesh(vertices=new_pos, faces=new_face)
print(f"#nodes from {len(mesh.vertices)} to {len(new_mesh.vertices)}")
plotmesh(new_mesh)