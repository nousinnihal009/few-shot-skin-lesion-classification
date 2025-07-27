# utils/data_loader.py

import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
import torchvision.transforms as transforms

class FewShotDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def get_transform(img_size=224):
    """Image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def create_episode(dataset, n_way, k_shot, q_queries):
    """
    Creates a single few-shot learning episode
    - n_way: number of classes
    - k_shot: number of support images per class
    - q_queries: number of query images per class
    """
    # Randomly sample N classes
    classes = list(set(dataset.labels))
    sampled_classes = random.sample(classes, n_way)

    support_images, query_images = [], []
    support_labels, query_labels = [], []

    for label_idx, cls in enumerate(sampled_classes):
        # All indices of this class
        cls_indices = [i for i, y in enumerate(dataset.labels) if y == cls]

        # Sample k_shot + q_queries images
        selected = random.sample(cls_indices, k_shot + q_queries)

        support_images += [dataset.image_paths[i] for i in selected[:k_shot]]
        query_images += [dataset.image_paths[i] for i in selected[k_shot:]]
        support_labels += [label_idx] * k_shot
        query_labels += [label_idx] * q_queries

    return support_images, support_labels, query_images, query_labels
