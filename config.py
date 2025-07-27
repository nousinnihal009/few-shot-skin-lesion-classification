# config.py

class Config:
    # Dataset paths
    data_path = "data/"        # path where images are stored
    save_model_path = "models/proto_net.pth"
    results_path = "results/"

    # Few-shot setup
    n_way = 5        # 5 classes per episode
    k_shot = 1       # 1 support image per class
    q_queries = 5    # 5 query images per class

    # Image processing
    image_size = 224

    # Training setup
    meta_batch_size = 32
    num_episodes = 1000
    learning_rate = 1e-3

    # Device
    device = "cuda"  # "cpu" if no GPU
