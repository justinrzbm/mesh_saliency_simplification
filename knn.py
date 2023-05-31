import numpy as np

# Modified naive vectorized KNN algorithm from Levent Baş (https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1)

def euclidian_distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))


'''
k nearest neighbor indices. Distance is the euclidean distance by default
input: 
    `X`: input array
    `k`: number of neighbors
returns:
    Neighbor indices array(n, k)

'''
def kneighbors(X, k=5):
    
    k = k 
    neigh_ind = []
    
    point_dist = [euclidian_distance(x_, X) for x_ in X]

    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]

        ind_list = [tup[0] for tup in sorted_neigh]

        neigh_ind.append(ind_list)
    
    
    return np.array(neigh_ind)
