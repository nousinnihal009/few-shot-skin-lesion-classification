# config.py

class Config:
    """
    Configuration for few-shot skin lesion classification.
    """

    # Dataset paths
    DATA_PATH = "data/"                     # Folder containing all images
    SAVE_MODEL_PATH = "models/proto_net.pth"
    RESULTS_PATH = "results/"

    # Few-shot learning setup
    N_WAY = 5                               # Number of classes per episode
    K_SHOT = 1                              # Number of support samples per class
    Q_QUERIES = 5                           # Number of query samples per class

    # Image processing
    IMAGE_SIZE = 224                        # Resize all images to this size

    # Training hyperparameters
    META_BATCH_SIZE = 32
    NUM_EPISODES = 1000
    LEARNING_RATE = 1e-3

    # Device configuration
    DEVICE = "cuda"                         # Change to "cpu" if no GPU available
