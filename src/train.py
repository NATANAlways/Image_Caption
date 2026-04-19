import os
import sys
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.dataset import load_captions, get_dataloaders, Vocabulary
from src.model import ImageCaptioner, count_parameters


# ── Loss ───────────────────────────────────────────────────────────────────────

def compute_loss(logits, captions, criterion, pad_idx=0):
    """
    logits  : (B, seq_len-1, vocab_size)
    captions: (B, seq_len)
    target is captions[:, 1:] — everything after <start>
    """
    B, seq_len_minus1, vocab_size = logits.shape

    target = captions[:, 1:]                          # (B, seq_len-1)
    logits = logits.reshape(-1, vocab_size)           # (B*(seq_len-1), vocab_size)
    target = target.reshape(-1)                       # (B*(seq_len-1),)

    loss = criterion(logits, target)
    return loss


# ── Train One Epoch ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for images, captions in tqdm(loader, desc="Train", leave=False):
        images   = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()

        if USE_AMP:
            with autocast("cuda"):
                logits = model(images, captions)
                loss   = compute_loss(logits, captions, criterion)
            scaler.scale(loss).backward()
            # gradient clipping — prevents exploding gradients in LSTM
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, captions)
            loss   = compute_loss(logits, captions, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Validate One Epoch ─────────────────────────────────────────────────────────

def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Val", leave=False):
            images   = images.to(DEVICE)
            captions = captions.to(DEVICE)

            if USE_AMP:
                with autocast("cuda"):
                    logits = model(images, captions)
                    loss   = compute_loss(logits, captions, criterion)
            else:
                logits = model(images, captions)
                loss   = compute_loss(logits, captions, criterion)

            total_loss += loss.item()

    return total_loss / len(loader)


# ── Plot Loss ──────────────────────────────────────────────────────────────────

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved → {OUTPUT_DIR}/loss_curve.png")


# ── Main Train Loop ────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, vocab):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5,
                    patience=3)
    scaler    = GradScaler("cuda", enabled=USE_AMP)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nTraining on {DEVICE} | AMP: {USE_AMP}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss   = val_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch'     : epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss'  : val_loss,
                'vocab_size': len(vocab),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

    # save final model
    torch.save({
        'epoch'     : EPOCHS,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_loss'  : val_losses[-1],
        'vocab_size': len(vocab),
    }, os.path.join(CHECKPOINT_DIR, "final_model.pt"))
    print(f"\nFinal model saved → {CHECKPOINT_DIR}/final_model.pt")

    plot_loss(train_losses, val_losses)
    return train_losses, val_losses


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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