
import pymesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from saliency_zhang import *

# Compute the mesh saliency values.
mesh = pymesh.load_mesh("bunny.obj")
saliency = compute_mesh_saliency(mesh)

# Plot the mesh saliency map.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces, cmap="jet", linewidth=0.2,
                antialiased=True, edgecolor="grey", alpha=1.0,
                facecolors=plt.cm.jet(saliency))
ax.set_axis_off()
plt.show()