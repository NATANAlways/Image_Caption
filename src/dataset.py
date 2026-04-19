import os
import re
import json
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self):
        # special tokens
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.freq     = Counter()

    def build(self, captions, threshold=VOCAB_THRESHOLD):
        for caption in captions:
            self.freq.update(caption.split())

        idx = len(self.word2idx)
        for word, count in self.freq.items():
            if count >= threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
                idx += 1

        print(f"Vocabulary built — {len(self.word2idx)} words "
              f"(threshold={threshold})")

    def encode(self, caption):
        """caption string → list of token ids"""
        return [
            self.word2idx.get(w, self.word2idx["<unk>"])
            for w in caption.split()
        ]

    def decode(self, indices):
        """list of token ids → caption string"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<unk>")
            if word in ("<start>", "<pad>"):
                continue
            if word == "<end>":
                break
            words.append(word)
        return " ".join(words)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx,
                       "idx2word": {str(k): v
                                    for k, v in self.idx2word.items()}}, f)
        print(f"Vocabulary saved → {path}")

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        print(f"Vocabulary loaded — {len(vocab.word2idx)} words")
        return vocab

    def __len__(self):
        return len(self.word2idx)


# ── Caption Cleaning ────────────────────────────────────────────────────────────

def clean_caption(caption):
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z\s]", "", caption)   # remove non-alpha (regex fix)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


# ── Load & Parse Captions ──────────────────────────────────────────────────────

def load_captions(caption_file):
    """
    Returns dict: { image_id: [caption1, caption2, ...] }
    Skips header line automatically.
    """
    mapping = {}

    with open(caption_file, "r") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            # split only on first comma — captions may contain commas
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue

            image_file, caption = parts
            image_id = image_file.split(".")[0]
            caption  = clean_caption(caption)

            if not caption:
                continue

            mapping.setdefault(image_id, []).append(caption)

    print(f"Loaded captions for {len(mapping)} images")
    return mapping


# ── Train / Val / Test Split ───────────────────────────────────────────────────

def split_data(mapping, train_ratio=0.80, val_ratio=0.10, seed=SEED):
    import random
    random.seed(seed)

    ids = list(mapping.keys())
    random.shuffle(ids)

    n         = len(ids)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_ids = ids[:train_end]
    val_ids   = ids[train_end:val_end]
    test_ids  = ids[val_end:]

    print(f"Split — Train: {len(train_ids)} | "
          f"Val: {len(val_ids)} | Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# ── Dataset ────────────────────────────────────────────────────────────────────

class CaptionDataset(Dataset):
    def __init__(self, image_ids, mapping, vocab, image_dir, transform=None):
        self.image_dir = image_dir
        self.vocab     = vocab
        self.transform = transform
        self.samples   = []  # list of (image_id, caption_str)

        for img_id in image_ids:
            for caption in mapping[img_id]:
                self.samples.append((img_id, caption))

        print(f"Dataset ready — {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]

        # load image
        img_path = os.path.join(self.image_dir, image_id + ".jpg")
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # encode caption: <start> + tokens + <end>
        tokens  = (["<start>"] + caption.split() + ["<end>"])[:MAX_LENGTH]
        encoded = [self.vocab.word2idx.get(w, self.vocab.word2idx["<unk>"])
                   for w in tokens]

        # pad to MAX_LENGTH
        padded  = encoded + [self.vocab.word2idx["<pad>"]] * (MAX_LENGTH - len(encoded))
        caption_tensor = torch.tensor(padded, dtype=torch.long)

        return image, caption_tensor


# ── Collate & DataLoader ───────────────────────────────────────────────────────

def collate_fn(batch):
    images, captions = zip(*batch)
    images   = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    return images, captions


def get_dataloaders(mapping, vocab, image_dir):
    mapping = {k: v for k, v in mapping.items()
               if os.path.exists(os.path.join(image_dir, k + ".jpg"))}
    print(f"Images available after filter: {len(mapping)}")
    train_ids, val_ids, test_ids = split_data(mapping)

    train_ds = CaptionDataset(train_ids, mapping, vocab, image_dir,
                               transform=get_transforms(train=True))
    val_ds   = CaptionDataset(val_ids,   mapping, vocab, image_dir,
                               transform=get_transforms(train=False))
    test_ds  = CaptionDataset(test_ids,  mapping, vocab, image_dir,
                               transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader