"""
Experiment 4 — Threshold Sensitivity Analysis

Sweeps the sigmoid threshold from 0.1 to 0.9 on the trained IngredientNet
and records precision, recall, and F1 at each point.

No retraining needed — loads checkpoints/ingredient_net.pt directly.

Usage:
    python experiment4.py
"""

import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from PIL import Image as PILImage

from finetune_ing import (
    FoodIngredientDataset, MultiLabelHead,
    DATA_DIR, val_transform,
    precision_recall_f1,
)

BASE       = Path(__file__).parent
CKPT       = BASE / "checkpoints/ingredient_net.pt"
LABELS     = BASE / "ingredient_labels_clean.json"
BATCH_SIZE  = 64
NUM_WORKERS = 0
THRESHOLDS = [round(t, 2) for t in torch.arange(0.1, 1.0, 0.1).tolist()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Build vocab and val loader from the main ingredient_labels.json
# ---------------------------------------------------------------------------
labels = json.loads(LABELS.read_text())

freq  = Counter(ing for ings in labels.values() for ing in ings)
vocab = sorted(freq.keys())
ing2idx = {ing: i for i, ing in enumerate(vocab)}

import finetune_ing as fi
fi._IMAGE_LABELS    = labels
fi.INGREDIENT_VOCAB = vocab
fi.NUM_INGREDIENTS  = len(vocab)
fi.ING2IDX          = ing2idx

full_ds = FoodIngredientDataset(DATA_DIR, split="train", transform=val_transform)
val_n   = int(0.2 * len(full_ds))
indices = torch.randperm(len(full_ds),
                         generator=torch.Generator().manual_seed(42)).tolist()
val_ds  = torch.utils.data.Subset(full_ds, indices[:val_n])
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
print(f"Val images: {len(val_ds)}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if not CKPT.exists():
    raise FileNotFoundError(f"{CKPT} not found — train IngredientNet first.")

class IngredientNet(nn.Module):
    def __init__(self, num_ingredients):
        super().__init__()
        backbone = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
        self.head = MultiLabelHead(1280, 256, num_ingredients)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))

model = IngredientNet(len(vocab)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
print(f"Loaded {CKPT}")

# ---------------------------------------------------------------------------
# Collect all logits and targets in one pass
# ---------------------------------------------------------------------------
print("Running forward pass over val set...")
all_logits, all_targets = [], []
with torch.inference_mode():
    for X, y in tqdm(val_loader, leave=False):
        all_logits.append(model(X.to(device)).cpu())
        all_targets.append(y)

logits  = torch.cat(all_logits)   # [N, vocab_size]
targets = torch.cat(all_targets)  # [N, vocab_size]

# ---------------------------------------------------------------------------
# Sweep thresholds
# ---------------------------------------------------------------------------
print("\nThreshold sweep:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 42)

results = []
for t in THRESHOLDS:
    p, r, f1 = precision_recall_f1(logits, targets, threshold=t)
    print(f"{t:>10.2f} {p:>10.4f} {r:>8.4f} {f1:>8.4f}")
    results.append({"threshold": t, "precision": round(p, 4),
                    "recall": round(r, 4), "f1": round(f1, 4)})

# best F1
best = max(results, key=lambda x: x["f1"])
print(f"\nBest F1 {best['f1']:.4f} at threshold {best['threshold']}")

(BASE / "exp4_results.json").write_text(json.dumps(results, indent=2))
print("Saved to exp4_results.json")
