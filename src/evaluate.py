import os
import sys
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.dataset import load_captions, get_dataloaders, Vocabulary
from src.model import ImageCaptioner


# ── Generate Captions for Test Set ────────────────────────────────────────────

def evaluate(model, test_loader, vocab):
    model.eval()
    actual  = []   # list of list of reference captions (tokenized)
    predict = []   # list of predicted captions (tokenized)

    with torch.no_grad():
        for images, captions in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)

            for i in range(images.size(0)):
                image   = images[i].unsqueeze(0)        # (1, C, H, W)
                caption = model.generate(image, vocab)  # predicted string

                # decode reference captions from token ids
                ref_tokens = captions[i].tolist()
                ref_caption = vocab.decode(ref_tokens)

                # tokenize for BLEU
                actual.append([ref_caption.split()])    # [[ref words]]
                predict.append(caption.split())         # [pred words]

    return actual, predict


# ── BLEU Score ─────────────────────────────────────────────────────────────────

def compute_bleu(actual, predict):
    smoother = SmoothingFunction().method1

    bleu1 = corpus_bleu(actual, predict,
                        weights=(1.0, 0, 0, 0),
                        smoothing_function=smoother)
    bleu2 = corpus_bleu(actual, predict,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoother)
    bleu3 = corpus_bleu(actual, predict,
                        weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smoother)
    bleu4 = corpus_bleu(actual, predict,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoother)

    print("\n── BLEU Scores ───────────────────────────────")
    print(f"  BLEU-1 : {bleu1:.4f}")
    print(f"  BLEU-2 : {bleu2:.4f}")
    print(f"  BLEU-3 : {bleu3:.4f}")
    print(f"  BLEU-4 : {bleu4:.4f}")
    print("─────────────────────────────────────────────\n")

    return bleu1, bleu2, bleu3, bleu4


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load captions & vocab
    mapping = load_captions(CAPTION_FILE)
    vocab   = Vocabulary.load(os.path.join(CHECKPOINT_DIR, "vocab.json"))

    # dataloaders — only need test_loader
    _, _, test_loader = get_dataloaders(mapping, vocab, IMAGE_DIR)

    # load best model
    model = ImageCaptioner(len(vocab)).to(DEVICE)
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state'])
    print(f"Model loaded — best epoch: {checkpoint['epoch']} | "
          f"val_loss: {checkpoint['val_loss']:.4f}")

    # evaluate
    actual, predict = evaluate(model, test_loader, vocab)
    compute_bleu(actual, predict)