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
PATIENCE    = 15   # stop if val_acc doesn't improve for this many epochs
DATA_DIR    = Path("/share/nas165/princelu/nn_final/food11/food11")

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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Datasets & dataloaders
# ---------------------------------------------------------------------------
# Load the same folder twice with different transforms so each split
# gets the correct augmentation without sharing the same dataset object.
train_full = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_full   = datasets.ImageFolder(DATA_DIR / "train", transform=val_transforms)

val_size   = int(0.2 * len(train_full))
train_size = len(train_full) - val_size
indices    = torch.randperm(len(train_full),
                            generator=torch.Generator().manual_seed(42)).tolist()

train_dataset = torch.utils.data.Subset(train_full, indices[val_size:])
val_dataset   = torch.utils.data.Subset(val_full,   indices[:val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
valid_dataloader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

NUM_CLASSES = len(train_full.classes)
print(f"Classes ({NUM_CLASSES}): {train_full.classes}")
print(f"Train: {train_size} | Val: {val_size}")

# ---------------------------------------------------------------------------
# Model — EfficientNet-B0
#   features[0..5] frozen  (low-level edges, textures — ImageNet is fine)
#   features[6..8] unfrozen with low LR  (high-level food features)
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
# Loss & optimiser — differential learning rates
# ---------------------------------------------------------------------------
loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam([
    {"params": cls.features[6].parameters(), "lr": 1e-4},
    {"params": cls.features[7].parameters(), "lr": 1e-4},
    {"params": cls.features[8].parameters(), "lr": 1e-4},
    {"params": cls.classifier.parameters(),  "lr": 1e-3},
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------
def one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = total_acc = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
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
        for X, y in tqdm(dataloader):
            X, y  = X.to(device), y.to(device)
            y_pred = model(X)
            total_loss += loss_fn(y_pred, y).item()
            total_acc  += (y_pred.argmax(1) == y).float().mean().item()
    n = len(dataloader)
    return total_loss / n, total_acc / n

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
CLASS_NAMES = {str(i): name for i, name in enumerate(val_full.classes)}

_infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint: str = "checkpoints/best.pt") -> nn.Module:
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded classifier from {checkpoint}  ({NUM_CLASSES} classes: {train_full.classes})")
    return model


def predict_class(model: nn.Module, image_path: str) -> str:
    img    = Image.open(image_path).convert("RGB")
    tensor = _infer_tf(img).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        idx = model(tensor).argmax(1).item()
    return CLASS_NAMES[str(idx)]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    best_acc    = float("-inf")
    no_improve  = 0

    for epoch in tqdm(range(EPOCHS)):
        tr_loss, tr_acc = one_epoch(cls, train_dataloader, loss_fn, optimizer)
        vl_loss, vl_acc = val_epoch(cls, valid_dataloader, loss_fn)

        scheduler.step(vl_acc)

        if vl_acc > best_acc:
            os.makedirs("checkpoints", exist_ok=True)
            best_acc   = vl_acc
            no_improve = 0
            torch.save(cls.state_dict(), "checkpoints/best.pt")
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss {tr_loss:.4f} | train_acc {tr_acc:.4f} | "
            f"val_loss {vl_loss:.4f} | val_acc {vl_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping — no improvement for {PATIENCE} epochs.")
            break
