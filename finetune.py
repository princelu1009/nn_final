import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image

BATCH_SIZE  = 32
NUM_WORKERS = 2
EPOCHS      = 1000
PATIENCE    = 15
DATA_DIR    = Path("/share/nas165/princelu/nn_final/data/food-101")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Datasets — food-101 uses meta/train.txt to define the train split
# double-load so each split gets its own transform
# ---------------------------------------------------------------------------
train_full = datasets.ImageFolder(DATA_DIR / "images", transform=train_transforms)
val_full   = datasets.ImageFolder(DATA_DIR / "images", transform=val_transforms)

# filter to only food-101 official train images
_train_allowed = set((DATA_DIR / "meta/train.txt").read_text().splitlines())
train_indices  = [
    i for i, (path, _) in enumerate(train_full.samples)
    if f"{Path(path).parent.name}/{Path(path).stem}" in _train_allowed
]

# 80/20 split within the train set for train/val
val_n     = int(0.2 * len(train_indices))
_perm     = torch.randperm(len(train_indices),
                           generator=torch.Generator().manual_seed(42)).tolist()
train_idx = [train_indices[i] for i in _perm[val_n:]]
val_idx   = [train_indices[i] for i in _perm[:val_n]]

train_dataset = torch.utils.data.Subset(train_full, train_idx)
val_dataset   = torch.utils.data.Subset(val_full,   val_idx)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
valid_dataloader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

NUM_CLASSES = len(train_full.classes)
print(f"Classes  : {NUM_CLASSES}")
print(f"Train    : {len(train_dataset)} | Val: {len(val_dataset)}")

# ---------------------------------------------------------------------------
# Model — EfficientNet-B0
#   features[0..5] frozen  (low-level edges/textures — ImageNet is fine)
#   features[6..8] unfrozen with low LR
#   classifier head         with normal LR
# ---------------------------------------------------------------------------
cls = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")

for param in cls.features.parameters():
    param.requires_grad = False

for block in [cls.features[6], cls.features[7], cls.features[8]]:
    for param in block.parameters():
        param.requires_grad = True

cls.classifier[1] = nn.Linear(1280, NUM_CLASSES)
cls = cls.to(device)

# ---------------------------------------------------------------------------
# Loss & optimiser
# ---------------------------------------------------------------------------
loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam([
    {"params": cls.features[6].parameters(), "lr": 1e-4},
    {"params": cls.features[7].parameters(), "lr": 1e-4},
    {"params": cls.features[8].parameters(), "lr": 1e-4},
    {"params": cls.classifier.parameters(),  "lr": 1e-3},
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------
def one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = total_acc = 0
    for X, y in tqdm(dataloader, leave=False):
        X, y   = X.to(device), y.to(device)
        y_pred = model(X)
        loss   = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc  += (y_pred.argmax(1) == y).float().mean().item()
    n = len(dataloader)
    return total_loss / n, total_acc / n


def val_epoch(model, dataloader, loss_fn):
    model.eval()
    total_loss = total_acc = 0
    with torch.inference_mode():
        for X, y in tqdm(dataloader, leave=False):
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            total_acc  += (y_pred.argmax(1) == y).float().mean().item()
    n = len(dataloader)
    return total_loss / n, total_acc / n

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
CLASS_NAMES = {str(i): name for i, name in enumerate(train_full.classes)}

_infer_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint: str = "checkpoints/best.pt") -> nn.Module:
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded classifier from {checkpoint}  ({NUM_CLASSES} classes)")
    return model


def predict_class(model: nn.Module, image_path: str) -> str:
    img    = Image.open(image_path).convert("RGB")
    tensor = _infer_tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        idx = model(tensor).argmax(1).item()
    return CLASS_NAMES[str(idx)].replace("_", " ")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    best_acc   = float("-inf")
    no_improve = 0
    start_epoch = 0
    print(device)

    # resume from last checkpoint if available
    # resume.pt has full state; best.pt has model weights only (for inference)
    resume_path = Path("checkpoints/resume.pt")
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        cls.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc    = ckpt["best_acc"]
        no_improve  = ckpt["no_improve"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}  (best_acc={best_acc:.4f})")
    elif Path("checkpoints/best.pt").exists():
        # best.pt exists but no resume.pt — load weights only, restart optimizer
        cls.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
        print("Loaded weights from best.pt — optimizer and scheduler start fresh")

    for epoch in tqdm(range(start_epoch, EPOCHS)):
        tr_loss, tr_acc = one_epoch(cls, train_dataloader, loss_fn, optimizer)
        vl_loss, vl_acc = val_epoch(cls, valid_dataloader, loss_fn)

        scheduler.step()

        if vl_acc > best_acc:
            best_acc   = vl_acc
            no_improve = 0
            torch.save(cls.state_dict(), "checkpoints/best.pt")
            print(f"  Saved best checkpoint (val_acc={best_acc:.4f})")
        else:
            no_improve += 1

        # save full training state after every epoch for resuming
        torch.save({
            "epoch":     epoch,
            "model":     cls.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc":  best_acc,
            "no_improve": no_improve,
        }, "checkpoints/resume.pt")

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss {tr_loss:.4f}  train_acc {tr_acc:.4f} | "
            f"val_loss {vl_loss:.4f}  val_acc {vl_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping — no improvement for {PATIENCE} epochs.")
            break
