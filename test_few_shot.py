# test_few_shot.py

import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loader import FewShotDataset, get_transform, create_episode
from utils.model import ProtoNet, euclidean_distance
from config import Config


def test_few_shot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device in use: {device}")

    # ------------------------------------------------------
    # 1. Load the trained model
    # ------------------------------------------------------
    model = ProtoNet().to(device)
    model.load_state_dict(torch.load(Config.SAVE_MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Trained ProtoNet model loaded.")

    # ------------------------------------------------------
    # 2. Load unseen test dataset (user-defined paths)
    # ------------------------------------------------------
    test_image_paths = []  # Fill with: ["data/demo_test/img1.jpg", ...]
    test_labels = []       # Fill with: ["class1", "class1", "class2", ...]

    if not test_image_paths:
        print("⚠️ No test images found. Please add image paths to test_image_paths.")
        return

    dataset = FewShotDataset(test_image_paths, test_labels, transform=get_transform(Config.IMAGE_SIZE))

    # ------------------------------------------------------
    # 3. Sample an N-way K-shot episode
    # ------------------------------------------------------
    support_imgs, support_labels, query_imgs, query_labels = create_episode(
        dataset, Config.N_WAY, Config.K_SHOT, Config.Q_QUERIES
    )

    support_tensors = torch.stack([
        dataset.transform(Image.open(p).convert("RGB")) for p in support_imgs
    ]).to(device)
    query_tensors = torch.stack([
        dataset.transform(Image.open(p).convert("RGB")) for p in query_imgs
    ]).to(device)

    support_labels = torch.tensor(support_labels).to(device)
    query_labels = torch.tensor(query_labels).to(device)

    # ------------------------------------------------------
    # 4. Compute embeddings and evaluate
    # ------------------------------------------------------
    with torch.no_grad():
        support_embeddings = model(support_tensors)
        query_embeddings = model(query_tensors)

        # Compute class prototypes
        prototypes = []
        for c in range(Config.N_WAY):
            class_embed = support_embeddings[support_labels == c]
            prototypes.append(class_embed.mean(0))
        prototypes = torch.stack(prototypes)

        # Compute distances and predictions
        distances = euclidean_distance(query_embeddings, prototypes)
        pred_labels = (-distances).softmax(dim=1).argmax(dim=1)

        accuracy = (pred_labels == query_labels).float().mean().item() * 100

    print(f"🎯 Few-shot classification accuracy: {accuracy:.2f}%\n")


if __name__ == "__main__":
    test_few_shot()
