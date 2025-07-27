# train_few_shot.py

import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.data_loader import FewShotDataset, get_transform, create_episode
from utils.model import ProtoNet, euclidean_distance
from config import Config

def train_protonet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # ✅ 1. Load dataset paths
    # -------------------------
    # TODO: Replace this with actual ISIC dataset paths + labels
    image_paths = []  # e.g., ["data/img1.jpg", "data/img2.jpg", ...]
    labels = []       # e.g., ["melanoma", "nevus", ...]

    if len(image_paths) == 0:
        print("⚠️ No dataset found! Please add ISIC dataset paths.")
        return

    dataset = FewShotDataset(image_paths, labels, transform=get_transform(Config.image_size))
    
    # -------------------------
    # ✅ 2. Initialize model
    # -------------------------
    model = ProtoNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # -------------------------
    # ✅ 3. Episodic Training Loop
    # -------------------------
    for episode in range(Config.num_episodes):
        # Sample an episode (support + query sets)
        support_imgs, support_labels, query_imgs, query_labels = create_episode(
            dataset, Config.n_way, Config.k_shot, Config.q_queries
        )
        
        # Load images into tensors
        support_tensors = torch.stack([dataset.transform(Image.open(p).convert("RGB")) for p in support_imgs]).to(device)
        query_tensors = torch.stack([dataset.transform(Image.open(p).convert("RGB")) for p in query_imgs]).to(device)
        support_labels = torch.tensor(support_labels).to(device)
        query_labels = torch.tensor(query_labels).to(device)
        
        # Compute embeddings
        support_embeddings = model(support_tensors)
        query_embeddings = model(query_tensors)
        
        # Compute class prototypes (mean embedding per class)
        prototypes = []
        for c in range(Config.n_way):
            prototypes.append(support_embeddings[support_labels == c].mean(0))
        prototypes = torch.stack(prototypes)
        
        # Compute distances (query → prototypes)
        distances = euclidean_distance(query_embeddings, prototypes)
        log_p_y = F.log_softmax(-distances, dim=1)
        
        # Compute loss
        loss = F.nll_loss(log_p_y, query_labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode + 1) % 50 == 0:
            print(f"[Episode {episode+1}/{Config.num_episodes}] Loss: {loss.item():.4f}")

    # -------------------------
    # ✅ 4. Save trained model
    # -------------------------
    os.makedirs(os.path.dirname(Config.save_model_path), exist_ok=True)
    torch.save(model.state_dict(), Config.save_model_path)
    print(f"✅ Model saved to {Config.save_model_path}")

if __name__ == "__main__":
    train_protonet()
