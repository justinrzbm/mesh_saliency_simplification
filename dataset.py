import os
import torch
from torch.utils.data import Dataset
import trimesh

class ModelNet10(Dataset):
    def __init__(self, root_dir, transform=None, num_points=3000):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []
        self.labels = []
        self.num_points = num_points

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for model_filename in os.listdir(os.path.join(class_dir, 'train')):
                model_path = os.path.join(class_dir, 'train', model_filename)
                # mesh = trimesh.load(model_path)
                # points = mesh.sample(self.num_points)  # Sample points from the 3D model
                # self.data.append(points)
                self.data.append(model_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        mesh = trimesh.load(data_path)
        points = mesh.vertices
        # points = mesh.sample(self.num_points)  # Sample points from the 3D model      
        label = self.labels[idx]

        sample = {'points': points, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        points, label = sample['points'], sample['label']

        # Convert points to a PyTorch tensor
        points = torch.tensor(points, dtype=torch.float32)

        return {'points': points, 'label': label}
