import numpy as np
from scipy.stats import entropy


def local_pos_pca(mesh, entropy_r=8, nbins=16):

    neighbors_dict = {}

    for i, vertex in enumerate(mesh.vertices):
        # Compute covariance matrix of neighboring face normals
        # TODO trimesh has vertex_normals function to average adjacent face normals

        entropy_neighborhood = set()
        entropy_neighborhood.update([i])
        # get r-geodesic neighborhood
        for j in range(entropy_r):
            for k in entropy_neighborhood.copy():
                entropy_neighborhood.update(mesh.vertex_neighbors[k])

        # put unique neighbors for this node in top level dictionary
        neighbors_dict[i] = []
        for neighbor in entropy_neighborhood:
            neighbors_dict[i].append(neighbor)

        num_neighbors = len(entropy_neighborhood)
        
        X = np.zeros((num_neighbors, 3))
        for j, point_idx in  enumerate(entropy_neighborhood):
            X[j, :] = mesh.vertices[point_idx]
        cov = np.cov(X.T)

        # Compute eigenvalues of covariance matrix
        eigenvalues, _ = np.linalg.eig(cov)


        # Positive eigenvalues mean convex in that principal direction, negative means concave
        # define saliency to be the sum of absolute values of each eigenvalue
        curvature_norm[i] = np.sum(np.abs(eigenvalues))
        # entropy[i] = - np.sum((eigenvalues / lambda_sum) * np.log2(eigenvalues / lambda_sum))

    # Normalize saliency values
    curvature_norm /= np.max(curvature_norm)

    # now that curvature is computed for all points, iterate again to compute LCE
    lce = np.empty(len(mesh.vertices), dtype=np.float64)
    for i, vertex in enumerate(mesh.vertices):
        local_curvatures = curvature_norm[neighbors_dict[i]]
        # Calculate Shannon Entropy on curvature distribution for this neighborhood
        pk = np.histogram(local_curvatures, nbins)[0] 
        pk = np.float64(pk) / pk.sum()  # normalize distribution to a PDF
        lce[i] = entropy(pk)            # shannon entropy
    
    lce /= np.max(lce)  # Normalize
    return lce


