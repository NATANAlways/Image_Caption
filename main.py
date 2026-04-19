import os
import sys
import argparse
import torch

from config import *
from src.dataset import load_captions, get_dataloaders, Vocabulary
from src.model import ImageCaptioner, count_parameters
from src.train import train
from src.evaluate import evaluate, compute_bleu
from src.inference import load_model, predict, visualize, predict_batch, visualize_grid


# ── Argument Parser ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Image Caption Generator")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "inference"],
                        help="Mode to run")

    parser.add_argument("--image", type=str, default=None,
                        help="Path to image for inference")

    parser.add_argument("--batch", type=int, default=5,
                        help="Number of random images for batch inference")

    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_model.pt"),
                        help="Path to model checkpoint")

    return parser.parse_args()


# ── Train Mode ─────────────────────────────────────────────────────────────────

def run_train():
    print("\n═══════════════════════════════════════")
    print("         IMAGE CAPTION GENERATOR        ")
    print("              TRAINING MODE             ")
    print("═══════════════════════════════════════\n")

    # load captions
    mapping = load_captions(CAPTION_FILE)

    # build vocabulary
    all_captions = [c for caps in mapping.values() for c in caps]
    vocab = Vocabulary()
    vocab.build(all_captions)
    vocab.save(os.path.join(CHECKPOINT_DIR, "vocab.json"))

    # dataloaders
    train_loader, val_loader, _ = get_dataloaders(mapping, vocab, IMAGE_DIR)

    # model
    model = ImageCaptioner(len(vocab)).to(DEVICE)
    count_parameters(model)

    # train
    train(model, train_loader, val_loader, vocab)
    print("\n✅ Training complete.")


# ── Evaluate Mode ──────────────────────────────────────────────────────────────

def run_evaluate(checkpoint_path):
    print("\n═══════════════════════════════════════")
    print("         IMAGE CAPTION GENERATOR        ")
    print("            EVALUATION MODE             ")
    print("═══════════════════════════════════════\n")

    # load captions & vocab
    mapping = load_captions(CAPTION_FILE)
    vocab   = Vocabulary.load(os.path.join(CHECKPOINT_DIR, "vocab.json"))

    # dataloaders
    _, _, test_loader = get_dataloaders(mapping, vocab, IMAGE_DIR)

    # load model
    model = load_model(checkpoint_path, vocab_size=len(vocab))

    # evaluate
    actual, predict_list = evaluate(model, test_loader, vocab)
    compute_bleu(actual, predict_list)
    print("✅ Evaluation complete.")


# ── Inference Mode ─────────────────────────────────────────────────────────────

def run_inference(checkpoint_path, image_path=None, batch_n=5):
    print("\n═══════════════════════════════════════")
    print("         IMAGE CAPTION GENERATOR        ")
    print("            INFERENCE MODE              ")
    print("═══════════════════════════════════════\n")

    # load vocab & model
    vocab = Vocabulary.load(os.path.join(CHECKPOINT_DIR, "vocab.json"))
    model = load_model(checkpoint_path, vocab_size=len(vocab))

    if image_path:
        # single image inference
        print(f"\nImage : {image_path}")
        _, caption = predict(model, vocab, image_path)
        print(f"Caption: {caption}")
        visualize(image_path, caption)
    else:
        # batch inference on random images
        print(f"\nRunning batch inference on {batch_n} random images...")
        results = predict_batch(model, vocab, IMAGE_DIR, n=batch_n)
        visualize_grid(results)

    print("\n✅ Inference complete.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        run_train()

    elif args.mode == "evaluate":
        run_evaluate(args.checkpoint)

    elif args.mode == "inference":
        run_inference(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            batch_n=args.batch
        )