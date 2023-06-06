import numpy as np
import open3d as o3d

def euclidian_distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))


# Modified naive vectorized KNN algorithm from Levent Baş (https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1)
'''
k nearest neighbor indices. Distance is the euclidean distance by default
input: 
    `X`: input pointcloud array (nx3)
    `k`: number of neighbors (int)
returns:
    Neighbor indices array(n, k)
'''
def kneighbors_all_naive(X, k:int=5):
    
    nn_idx = []
    
    point_dist = [euclidian_distance(x_, X) for x_ in X]

    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]

        ind_list = [tup[0] for tup in sorted_neigh]

        nn_idx.append(ind_list)
    
    return np.array(nn_idx)

'''
Takes o3d KDTree and a 3d point as input and computes KNN on one vertex
'''
def kneighbors(KDTree, point3d, k):
    [k_, indices, _] = KDTree.search_knn_vector_3d(point3d, k)
    return indices


'''
k nearest neighbor indices. Distance is the euclidean distance by default
input: 
    `X`: input pointcloud array (nx3)
    `k`: number of neighbors (int)
returns:
    Neighbor indices array(n, k)
'''
def kneighbors_all(pointcloud, k:int=5):
    # KDTree must be created newly from this point cloud
    kdtree = o3d.geometry.KDTreeFlann(pointcloud)
    nn_idx = []
    for point in pointcloud.points:
        indices = kneighbors(kdtree, point, k)
        nn_idx.append(indices)
    nn_idx = np.array(nn_idx)
    return nn_idx
