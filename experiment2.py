"""
Experiment 2 — Effect of Pseudo-Label Cleaning on IngredientNet

Config A — raw BLIP output, no post-processing
Config B — cleaned (dedup + hallucination filter)

Both configs train on their own label file but validate against
ingredient_labels_clean.json as the shared ground truth.

Usage:
    python experiment2.py
"""

import os
import json
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from PIL import Image as PILImage

from finetune_ing import (
    FoodIngredientDataset, MultiLabelHead,
    train_one_epoch,
    DATA_DIR, train_transform, val_transform,
    precision_recall_f1,
)

BASE = Path(__file__).parent

CONFIGS = {
    "A_raw":     {"labels_file": BASE / "ingredient_labels_raw.json",   "min_freq": 0},
    "B_cleaned": {"labels_file": BASE / "ingredient_labels_clean.json", "min_freq": 0},
}

BATCH_SIZE  = 32
NUM_WORKERS = 2
EPOCHS      = 50
PATIENCE    = 10
LR          = 1e-3
HIDDEN_UNIT = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


def build_vocab(labels: dict, min_freq: int = 0) -> tuple[list, dict]:
    freq = Counter()
    for ings in labels.values():
        freq.update(ings)
    if min_freq > 0:
        vocab = sorted(ing for ing, cnt in freq.items() if cnt >= min_freq)
    else:
        vocab = sorted(freq.keys())
    return vocab, {ing: i for i, ing in enumerate(vocab)}


class _CapturedDataset(torch.utils.data.Dataset):
    """Self-contained dataset — labels captured at construction, no module globals."""
    def __init__(self, paths, labels, ing2idx, num_ingredients, transform):
        self.paths           = paths
        self.labels          = labels
        self.ing2idx         = ing2idx
        self.num_ingredients = num_ingredients
        self.transform       = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = PILImage.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        vec = torch.zeros(self.num_ingredients)
        key = f"{img_path.parent.name}/{img_path.stem}"
        for ing in self.labels.get(key, []):
            if ing in self.ing2idx:
                vec[self.ing2idx[ing]] = 1.0
        return img, vec


class IngredientNetExp(nn.Module):
    def __init__(self, num_ingredients):
        super().__init__()
        backbone = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
        self.head = MultiLabelHead(1280, HIDDEN_UNIT, num_ingredients)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


def _project_and_eval(model, loader, src_indices, clean_indices, clean_vocab_size):
    """Run val loader, project logits to clean vocab space, return preds and targets."""
    model.eval()
    src_t   = torch.tensor(src_indices)
    clean_t = torch.tensor(clean_indices)
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for X, y in loader:
            logits    = model(X.to(device)).cpu()
            projected = torch.full((logits.shape[0], clean_vocab_size), -10.0)
            projected[:, clean_t] = logits[:, src_t]
            all_preds.append(projected)
            all_targets.append(y)
    return torch.cat(all_preds), torch.cat(all_targets)


def run_config(name: str, labels_file: Path, min_freq: int) -> dict:
    print(f"\n{'='*60}")
    print(f"Config {name}  |  labels: {labels_file.name}  |  min_freq: {min_freq}")
    print(f"{'='*60}")

    if not labels_file.exists():
        print(f"  [skip] {labels_file} not found")
        return {"config": name, "vocab_size": 0, "precision": 0, "recall": 0, "f1": 0}

    train_labels = json.loads(labels_file.read_text())
    train_vocab, train_ing2idx = build_vocab(train_labels, min_freq)
    print(f"Train vocab size: {len(train_vocab)}")

    clean_labels = json.loads((BASE / "ingredient_labels_clean.json").read_text())
    clean_vocab, clean_ing2idx = build_vocab(clean_labels)
    print(f"Clean vocab size: {len(clean_vocab)}")

    # image paths — use clean labels as reference so both configs see the same images
    import finetune_ing as fi
    fi._IMAGE_LABELS = clean_labels
    ref_ds    = FoodIngredientDataset(DATA_DIR, split="train", transform=None)
    all_paths = ref_ds.paths

    val_n   = int(0.2 * len(all_paths))
    indices = torch.randperm(len(all_paths),
                             generator=torch.Generator().manual_seed(42)).tolist()
    train_paths = [all_paths[i] for i in indices[val_n:]]
    val_paths   = [all_paths[i] for i in indices[:val_n]]
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = _CapturedDataset(train_paths, train_labels, train_ing2idx,
                                len(train_vocab), train_transform)
    val_ds   = _CapturedDataset(val_paths,   clean_labels, clean_ing2idx,
                                len(clean_vocab), val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # projection indices: for each slot in clean_vocab, find its position in train_vocab
    proj         = [(ci, train_ing2idx[ing]) for ci, ing in enumerate(clean_vocab) if ing in train_ing2idx]
    clean_indices = [ci for ci, _ in proj]
    src_indices   = [si for _, si in proj]

    model     = IngredientNetExp(len(train_vocab)).to(device)
    loss_fn   = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=LR)

    best_f1    = 0.0
    no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in tqdm(range(EPOCHS), desc=name):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        preds, targets = _project_and_eval(model, val_loader,
                                           src_indices, clean_indices, len(clean_vocab))
        _, _, vl_f1 = precision_recall_f1(preds, targets, threshold=0.2)

        if vl_f1 > best_f1:
            best_f1    = vl_f1
            no_improve = 0
            torch.save(model.state_dict(), f"checkpoints/exp2_{name}.pt")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}  best_f1={best_f1:.4f}")
            break

    # final metrics on best checkpoint
    model.load_state_dict(torch.load(f"checkpoints/exp2_{name}.pt",
                                     map_location=device, weights_only=True))
    preds, targets = _project_and_eval(model, val_loader,
                                       src_indices, clean_indices, len(clean_vocab))
    p, r, f1 = precision_recall_f1(preds, targets, threshold=0.2)
    print(f"  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    return {"config": name, "vocab_size": len(train_vocab),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Run all configs and print table
# ---------------------------------------------------------------------------
results = []
for name, cfg in CONFIGS.items():
    results.append(run_config(name, cfg["labels_file"], cfg["min_freq"]))

print("\n\n" + "="*60)
print("EXPERIMENT 2 RESULTS  (val scored against clean labels)")
print("="*60)
print(f"{'Config':<12} {'Vocab':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-"*60)
for r in results:
    print(f"{r['config']:<12} {r['vocab_size']:>6} "
          f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")
print("="*60)

(BASE / "exp2_results.json").write_text(json.dumps(results, indent=2))
print("\nResults saved to exp2_results.json")
