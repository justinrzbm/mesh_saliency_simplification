import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ToTensor, ModelNet10

def generate_point_clouds(vae, num_samples, latent_dim):
    # Sample from the latent space
    z_samples = torch.randn(num_samples, latent_dim)
    # Generate point clouds
    generated_point_clouds = vae.decoder(z_samples)
    return generated_point_clouds

class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PointCloudEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling over points
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim, num_points):
        super(PointCloudDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 3 * num_points)  # Output 3D coordinates

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(3, -1)

class VAE(nn.Module):
    def __init__(self, latent_dim, num_points):
        super(VAE, self).__init__()
        # self.encoder_mean = PointCloudEncoder(latent_dim)
        # self.encoder_logvar = PointCloudEncoder(latent_dim)
        self.encoder = PointCloudEncoder(latent_dim)
        self.decoder = PointCloudDecoder(latent_dim, num_points)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + epsilon * std
        return z

    def forward(self, x):
        # B, P, 3
        batch_size, channel, num_points = x.shape
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar   

def chamfer_distance(cloud1, cloud2):
    with torch.no_grad():
        distances = get_distances(cloud1, cloud2)
        indices = torch.argmin(distances, dim=0)
    points = cloud2[indices]
    loss = torch.sum((cloud1 - points) ** 2)

    with torch.no_grad():
        distances = torch.transpose(distances, 0, 1)
        indices = torch.argmin(distances, dim=0)
    points = cloud1[indices]
    loss += torch.sum((cloud2 - points) ** 2)

    return loss

def get_distances(cloud1, cloud2):
    distances = []
    for i in range(len(cloud2)):
        dists = (cloud1 - cloud2[i]) ** 2
        distances.append(torch.sum(dists, dim=1))
    return torch.stack(distances)

def gaussian_kernel_density_estimation(x, points, bandwidth=1.0):
    pairwise_distance = torch.cdist(x, points)
    densities = torch.exp(-pairwise_distance**2 / (2 * bandwidth**2)).mean(dim=1)
    return densities

def kl_divergence(point_cloud_p, point_cloud_q, bandwidth=1.0, epsilon=1e-10):
    # We estimate the densities of point_cloud_p based on its own set of points
    p_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_p, bandwidth)
    
    # Then, we estimate the densities of point_cloud_p based on point_cloud_q
    q_density = gaussian_kernel_density_estimation(point_cloud_p, point_cloud_q, bandwidth)
    
    kl = (torch.log(p_density + epsilon) - torch.log(q_density + epsilon)).mean()
    return kl

def loss_function(recon_x, x, mu, logvar):
    cd = chamfer_distance(recon_x, x)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
    alpha = 0.5
    return cd + alpha * kl

def main():
    latent_dim = 256
    num_epochs = 100
    batch_size = 1 # 32
    learning_rate = 0.01
    num_points = 30000

    device = torch.device("cuda:9")

    # Data
    dataset_dir = "/home/ztan4/Downloads/ModelNet10"
    dataset = ModelNet10(dataset_dir, transform=ToTensor(), num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize your VAE
    vae = VAE(latent_dim, num_points).to(device)

    # Set up an optimizer (e.g., Adam)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch["points"].permute(0, 2, 1).to(device)
            # Forward pass
            recon_batch, mu, logvar = vae(batch)

            # Compute the loss
            loss = loss_function(recon_batch, batch, mu, logvar)
            total_loss += loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} | Loss: {total_loss}")

if __name__ == "__main__":
    main()