# Image Caption Generator — V1

A deep learning system that automatically generates natural language captions
for images using a custom CNN encoder and LSTM decoder trained on the Flickr8k dataset.

---

## How It Works

### Architecture Overview




### CNN Encoder

The CNN encoder is built from scratch using 5 convolutional blocks.
Each block follows the pattern Conv2D → BatchNorm → ReLU → MaxPool.
The final AdaptiveAvgPool layer reduces spatial dimensions to 1×1,
and a Dense projector maps the features to a 4096-dimensional vector
that captures the visual semantics of the image.

### LSTM Decoder

The LSTM decoder takes the image feature vector and uses it to
initialize the hidden state of a 2-layer LSTM. At each timestep,
the LSTM takes the previous word embedding as input and predicts
the next word in the caption. This continues until the model
generates an `<end>` token or reaches the maximum caption length.

### Beam Search

Instead of greedily picking the highest probability word at each step,
beam search maintains the top-3 candidate sequences simultaneously,
resulting in more fluent and accurate captions.

### Training Strategy

- **Loss:** CrossEntropyLoss (padding tokens ignored)
- **Optimizer:** Adam with weight decay (L2 regularization)
- **LR Scheduler:** ReduceLROnPlateau — halves LR if val loss stagnates for 3 epochs
- **Mixed Precision:** fp16 AMP training for RTX GPU memory efficiency
- **Gradient Clipping:** max norm 5.0 — prevents LSTM exploding gradients
- **Best Model:** saved based on validation loss, not training loss

---

## Dataset

This project uses the **Flickr8k** dataset:
- 8,090 images
- 5 human-written captions per image
- Split: 80% train / 10% val / 10% test

Download from: https://www.kaggle.com/datasets/adityajn105/flickr8k

Expected folder structure after download:


---

## Setup

### 1. Clone & Create Environment

```bash
git clone <your-repo-url>
cd image_caption_generator

python -m venv caption_env
source caption_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install nltk pillow tqdm matplotlib scikit-learn
```

### 3. Download Dataset

Download Flickr8k and place files as shown in the Dataset section above.

---

## Usage

### Train

```bash
python3 main.py --mode train
```

Trains for 20 epochs. Best model saved to `checkpoints/best_model.pt`.
Loss curve saved to `outputs/loss_curve.png`.

### Evaluate (BLEU Score)

```bash
python3 main.py --mode evaluate
```

Runs evaluation on the test set and prints BLEU-1 through BLEU-4 scores.

### Inference — Single Image

```bash
python3 main.py --mode inference --image path/to/your/image.jpg
```

Generates and displays a caption for the given image.
Result saved to `outputs/<image_name>_caption.png`.

### Inference — Batch (Random Images)

```bash
python3 main.py --mode inference --batch 10
```

Generates captions for 10 random images from the dataset.
Grid visualization saved to `outputs/batch_captions.png`.

### Use a Specific Checkpoint

```bash
python3 main.py --mode evaluate --checkpoint checkpoints/final_model.pt
python3 main.py --mode inference --checkpoint checkpoints/final_model.pt --image img.jpg
```

---

## Results — V1 Baseline

Trained for 20 epochs on Flickr8k with custom CNN + LSTM from scratch.

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.2275 |
| BLEU-2 | 0.0997 |
| BLEU-3 | 0.0521 |
| BLEU-4 | 0.0297 |

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 3.9331    | 3.5096   |
| 5     | 3.0692    | 3.1148   |
| 10    | 2.9327    | 3.0459   |
| 15    | 2.8816    | 3.0314   |
| 20    | 2.8516    | 3.0119   |

---

## Roadmap

| Version | Architecture | Expected BLEU-4 | Status |
|---------|-------------|-----------------|--------|
| V1 | Custom CNN + LSTM | ~0.03 | ✅ Complete |
| V2 | Pretrained ResNet50 + LSTM | ~0.20 | 🔜 Planned |
| V3 | Pretrained CNN + Attention + LSTM | ~0.28 | 🔜 Planned |

---

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| IMAGE_SIZE | 224 | CNN input resolution |
| MAX_LENGTH | 35 | Maximum caption length |
| VOCAB_THRESHOLD | 5 | Minimum word frequency |
| EMBED_DIM | 256 | Word embedding dimension |
| LSTM_UNITS | 512 | LSTM hidden state size |
| CNN_FEAT_DIM | 4096 | CNN output feature size |
| BATCH_SIZE | 32 | Training batch size |
| EPOCHS | 20 | Training epochs |
| LEARNING_RATE | 0.001 | Initial learning rate |
| BEAM_WIDTH | 3 | Beam search width |
| USE_AMP | True | Mixed precision training |