import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.dataset import Vocabulary, get_transforms
from src.model import ImageCaptioner


# ── Load Model ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, vocab_size):
    model = ImageCaptioner(vocab_size).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded — epoch: {checkpoint['epoch']} | "
          f"val_loss: {checkpoint['val_loss']:.4f}")
    return model


# ── Preprocess Image ───────────────────────────────────────────────────────────

def preprocess_image(image_path):
    transform = get_transforms(train=False)
    image     = Image.open(image_path).convert("RGB")
    tensor    = transform(image).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    return image, tensor


# ── Predict Caption ────────────────────────────────────────────────────────────

def predict(model, vocab, image_path):
    original_image, tensor = preprocess_image(image_path)
    caption = model.generate(tensor, vocab)
    return original_image, caption


# ── Visualize ──────────────────────────────────────────────────────────────────

def visualize(image_path, caption, save=True):
    original_image, _ = preprocess_image(image_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title(caption, fontsize=13, wrap=True,
              bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", alpha=0.7))

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_caption.png")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.show()
    plt.close()


# ── Batch Inference ────────────────────────────────────────────────────────────

def predict_batch(model, vocab, image_dir, n=5):
    """Run inference on n random images from image_dir"""
    import random
    all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    selected   = random.sample(all_images, min(n, len(all_images)))

    results = []
    for img_file in selected:
        img_path = os.path.join(image_dir, img_file)
        _, caption = predict(model, vocab, img_path)
        results.append((img_path, caption))
        print(f"  {img_file} → {caption}")

    return results


# ── Grid Visualization ─────────────────────────────────────────────────────────

def visualize_grid(results, save=True):
    """Show multiple image-caption pairs in a grid"""
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (img_path, caption) in zip(axes, results):
        image = Image.open(img_path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(caption, fontsize=10, wrap=True)

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "batch_captions.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"\nGrid saved → {out_path}")

    plt.show()
    plt.close()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load vocab & model
    vocab = Vocabulary.load(os.path.join(CHECKPOINT_DIR, "vocab.json"))
    model = load_model(
        os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        vocab_size=len(vocab)
    )

    print("\n── Single Image Inference ────────────────────")
    # pick one image from test set
    sample_image = os.path.join(IMAGE_DIR, "1000268201_693b08cb0e.jpg")
    _, caption   = predict(model, vocab, sample_image)
    print(f"Caption: {caption}")
    visualize(sample_image, caption)

    print("\n── Batch Inference (5 random images) ────────")
    results = predict_batch(model, vocab, IMAGE_DIR, n=5)
    visualize_grid(results)