import os 
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images/Image")
CAPTION_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

IMAGE_SIZE  = 224       # CNN input size
CHANNELS    = 3

MAX_LENGTH  = 35        # max caption length
VOCAB_THRESHOLD = 5 

EMBED_DIM   = 256
LSTM_UNITS  = 512
CNN_FEAT_DIM = 4096     # custom CNN output dimension
DROPOUT     = 0.4

BATCH_SIZE  = 32        # safe for 4GB VRAM
EPOCHS      = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

BEAM_WIDTH  = 3  

SEED = 42