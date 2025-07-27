# test_few_shot.py

import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loader import FewShotDataset, get_transform, create_episode
from utils.model import ProtoNet, euclidean_distance
from config import Config

def test_few_shot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # ✅ 1. Load trained model
    # -------------------------
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(Config.save_model_path, map_location=device))
    model.eval()
    print("✅ Loaded trained model.")

    # -------------------------
    # ✅ 2. Load test dataset
    # -------------------------
    # TODO: Replace with unseen lesion classes from ISIC
    test_image_paths = []  # e.g., ["data/test_img1.jpg", ...]
    test_labels = []       # e.g., ["rare_lesion_1", ...]

    if len(test_image_paths) == 0:
        print("⚠️ No test dataset found! Please add unseen lesion test data.")
        return

    dataset = FewShotDataset(test_image_paths, test_labels, transform=get_transform(Config.image_size))

    # -------------------------
    # ✅ 3. Sample an episode (simulate unseen lesion classification)
    # -------------------------
    support_imgs, support_labels, query_imgs, query_labels = create_episode(
        dataset, Config.n_way, Config.k_shot, Config.q_queries
    )

    # Load images into tensors
    support_tensors = torch.stack([dataset.transform(Image.open(p).convert("RGB")) for p in support_imgs]).to(device)
    query_tensors = torch.stack([dataset.transform(Image.open(p).convert("RGB")) for p in query_imgs]).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_labels = torch.tensor(query_labels).to(device)

    # -------------------------
    # ✅ 4. Compute embeddings & prototypes
    # -------------------------
    with torch.no_grad():
        support_embeddings = model(support_tensors)
        query_embeddings = model(query_tensors)

        prototypes = []
        for c in range(Config.n_way):
            prototypes.append(support_embeddings[support_labels == c].mean(0))
        prototypes = torch.stack(prototypes)

        # Compute distances → predicted class
        distances = euclidean_distance(query_embeddings, prototypes)
        pred_labels = (-distances).softmax(dim=1).argmax(dim=1)

        # Calculate accuracy
        acc = (pred_labels == query_labels).float().mean().item() * 100

    print(f"✅ Few-Shot Accuracy on unseen lesion episode: {acc:.2f}%")

if __name__ == "__main__":
    test_few_shot()
