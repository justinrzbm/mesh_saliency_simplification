import torch

def batch_pairwise_dist(x, y):
    """
    Computes the pairwise distance between two batches of point clouds.
    
    Args:
        x: Tensor of shape (batch_size, num_points, dims) representing the first batch of point clouds.
        y: Tensor of shape (batch_size, num_points, dims) representing the second batch of point clouds.
        
    Returns:
        Tensor of shape (batch_size, num_points_x, num_points_y) with the pairwise distances.
    """
    xx = torch.sum(x**2, dim=2).unsqueeze(2)
    yy = torch.sum(y**2, dim=2).unsqueeze(1)
    xy = torch.bmm(x, y.transpose(2, 1))
    
    return xx + yy - 2 * xy

def chamfer_distance(x, y):
    """
    Computes the chamfer distance between two batches of point clouds.
    
    Args:
        x: Tensor of shape (batch_size, num_points, dims) representing the first batch of point clouds.
        y: Tensor of shape (batch_size, num_points, dims) representing the second batch of point clouds.
        
    Returns:
        Tensor of shape (batch_size,) with the chamfer distances for each batch.
    """
    dists = batch_pairwise_dist(x, y)
    
    min_dists_x, _ = torch.min(dists, dim=2)
    min_dists_y, _ = torch.min(dists, dim=1)
    
    chamfer_dist = torch.mean(min_dists_x, dim=1) + torch.mean(min_dists_y, dim=1)
    
    return chamfer_dist

# Example usage:
A = torch.randn(32, 100, 3)  # 32 point clouds with 100 points each in 3D
B = torch.randn(32, 100, 3)

dist = chamfer_distance(A, B)
print(dist)
