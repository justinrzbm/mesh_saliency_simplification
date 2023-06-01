import numpy as np
from scipy.stats import entropy


def local_curvature_pca(mesh, curvature_r=4, entropy_r=8, compute_entropy=True, nbins=16):
    curvature_norm = np.zeros(len(mesh.vertices))
    neighbors_dict = {}

    for i, vertex in enumerate(mesh.vertices):
        # Compute covariance matrix of neighboring face normals
        # TODO trimesh has vertex_normals function to average adjacent face normals
        neighbors = set()   # much better time complexity, and ensure uniqueness
        neighbors.update([i])
        entropy_neighborhood = set()
        entropy_neighborhood.update([i])
        assert entropy_r >= curvature_r
        for j in range(entropy_r):
            if j < curvature_r:
                for k in neighbors.copy():
                    neighbors.update(mesh.vertex_neighbors[k])
            for k in entropy_neighborhood.copy():
                entropy_neighborhood.update(mesh.vertex_neighbors[k])

        # put unique neighbors for this node in top level dictionary
        neighbors_dict[i] = []
        for neighbor in entropy_neighborhood:
            neighbors_dict[i].append(neighbor)

        num_neighbors = len(neighbors)
        assert num_neighbors>0, "not fully connected mesh"
        

        X = np.zeros((num_neighbors, 3))
        for j, point_idx in  enumerate(neighbors):
            X[j, :] = mesh.vertex_normals[point_idx]
        cov = np.cov(X.T)

        # Compute eigenvalues of covariance matrix
        eigenvalues, _ = np.linalg.eig(cov)


        # Positive eigenvalues mean convex in that principal direction, negative means concave
        # define saliency to be the sum of absolute values of each eigenvalue
        curvature_norm[i] = np.sum(np.abs(eigenvalues))
        # entropy[i] = - np.sum((eigenvalues / lambda_sum) * np.log2(eigenvalues / lambda_sum))

    # Normalize saliency values
    curvature_norm /= np.max(curvature_norm)

    if compute_entropy:
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
        
    return curvature_norm

