import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ── Custom CNN Encoder ─────────────────────────────────────────────────────────

class CNNBlock(nn.Module):
    """Conv → BN → ReLU → Pool"""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, feat_dim=CNN_FEAT_DIM, dropout=DROPOUT):
        super().__init__()

        self.features = nn.Sequential(
            CNNBlock(3,   32),   # 224 → 112
            CNNBlock(32,  64),   # 112 → 56
            CNNBlock(64,  128),  # 56  → 28
            CNNBlock(128, 256),  # 28  → 14
            CNNBlock(256, 512, pool=False),  # 14 → 14 (no pool)
        )

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))  # → (512, 1, 1)
        self.flatten = nn.Flatten()                   # → (512,)

        self.projector = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)   # (B, 512, 14, 14)
        x = self.pool(x)        # (B, 512, 1, 1)
        x = self.flatten(x)     # (B, 512)
        x = self.projector(x)   # (B, feat_dim)
        return x


# ── LSTM Decoder ───────────────────────────────────────────────────────────────

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 lstm_units=LSTM_UNITS, feat_dim=CNN_FEAT_DIM,
                 max_length=MAX_LENGTH, dropout=DROPOUT):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        # project image features to lstm_units dimension
        self.img_proj  = nn.Linear(feat_dim, lstm_units)

        self.lstm      = nn.LSTM(embed_dim, lstm_units,
                                 batch_first=True, num_layers=2,
                                 dropout=dropout)

        self.fc        = nn.Linear(lstm_units, vocab_size)

        self.max_length = max_length

    def forward(self, features, captions):
        """
        features : (B, feat_dim)   — from CNN encoder
        captions : (B, max_length) — token ids, includes <start>, excludes last
        """
        # image feature → initial hidden state
        # shape: (num_layers, B, lstm_units)
        h0 = self.img_proj(features).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)

        # embed captions (drop last token — we don't need <end> as input)
        embeds = self.embedding(captions[:, :-1])  # (B, seq_len-1, embed_dim)
        embeds = self.dropout(embeds)

        # lstm forward
        out, _ = self.lstm(embeds, (h0, c0))       # (B, seq_len-1, lstm_units)

        # project to vocab
        logits = self.fc(out)                       # (B, seq_len-1, vocab_size)
        return logits

    def generate(self, feature, vocab, max_length=None, beam_width=BEAM_WIDTH):
        """
        Beam search caption generation for a single image.
        feature : (1, feat_dim) — single image
        Returns : caption string
        """
        self.eval()
        max_length = max_length or self.max_length

        start_idx = vocab.word2idx["<start>"]
        end_idx   = vocab.word2idx["<end>"]
        pad_idx   = vocab.word2idx["<pad>"]

        # initial hidden state
        h = self.img_proj(feature).unsqueeze(0).repeat(2, 1, 1)
        c = torch.zeros_like(h)

        # beam: list of (score, token_ids, h, c)
        beams = [(0.0, [start_idx], h, c)]
        completed = []

        for _ in range(max_length):
            new_beams = []
            for score, tokens, h, c in beams:
                if tokens[-1] == end_idx:
                    completed.append((score, tokens))
                    continue

                last_token = torch.tensor([[tokens[-1]]],
                                          device=feature.device)
                embed = self.dropout(self.embedding(last_token))  # (1,1,embed)
                out, (h_new, c_new) = self.lstm(embed, (h, c))
                logits = self.fc(out.squeeze(1))                  # (1, vocab)
                log_probs = torch.log_softmax(logits, dim=-1)

                topk_probs, topk_ids = log_probs.topk(beam_width)

                for i in range(beam_width):
                    new_score = score + topk_probs[0, i].item()
                    new_tokens = tokens + [topk_ids[0, i].item()]
                    new_beams.append((new_score, new_tokens, h_new, c_new))

            if not new_beams:
                break

            # keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # pick best completed or best beam
        if completed:
            best = max(completed, key=lambda x: x[0])
        else:
            best = max(beams, key=lambda x: x[0])

        return vocab.decode(best[1])


# ── Full Model ─────────────────────────────────────────────────────────────────

class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)       # (B, feat_dim)
        logits   = self.decoder(features, captions)  # (B, seq_len-1, vocab_size)
        return logits

    def generate(self, image, vocab):
        """image: (1, C, H, W) tensor"""
        with torch.no_grad():
            feature = self.encoder(image)
            caption = self.decoder.generate(feature, vocab)
        return caption


# ── Parameter Count ────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    return trainable


if __name__ == "__main__":
    # quick sanity check
    vocab_size = 5000
    model = ImageCaptioner(vocab_size).to(DEVICE)
    count_parameters(model)

    dummy_img     = torch.randn(4, 3, 224, 224).to(DEVICE)
    dummy_caption = torch.randint(0, vocab_size, (4, MAX_LENGTH)).to(DEVICE)

    logits = model(dummy_img, dummy_caption)
    print(f"Output shape: {logits.shape}")  
    # expected: (4, MAX_LENGTH-1, vocab_size)