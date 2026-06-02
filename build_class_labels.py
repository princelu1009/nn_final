"""
Experiment 3 — Class-level label generator.

Runs BLIP once per food class using the food name in the prompt:
    "the ingredients in this {food_name} are"

All training images of a class receive the same ingredient list.
Saves to ingredient_labels_class.json.

Run before experiment3.py:
    python build_class_labels.py
"""

import json
import re
import torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

IMAGES_DIR = Path(__file__).parent / "data/food-101/images"
META_FILE  = Path(__file__).parent / "data/food-101/meta/train.txt"
OUT_PATH   = Path(__file__).parent / "ingredient_labels_class.json"

# ---------------------------------------------------------------------------
# Build per-class image lists from train split
# ---------------------------------------------------------------------------
allowed = set(META_FILE.read_text().splitlines())
by_class: dict[str, list] = defaultdict(list)
for entry in allowed:
    cls = entry.split("/")[0]
    img_path = IMAGES_DIR / f"{entry}.jpg"
    if img_path.exists():
        by_class[cls].append(img_path)

classes = sorted(by_class.keys())
print(f"Classes: {len(classes)}")

# ---------------------------------------------------------------------------
# Load BLIP
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True,
).to(device)
model.eval()

# ---------------------------------------------------------------------------
# Ingredient parsing (same rules as build_image_labels.py)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "of", "in", "on", "is", "are",
    "it", "this", "that", "made", "dish", "food", "served", "topped",
    "filled", "covered", "cooked", "fresh", "some", "various", "different",
}

def _parse(text: str) -> list[str]:
    parts = re.split(r"[,;&]|\band\b", text, flags=re.IGNORECASE)
    result = []
    for p in parts:
        words = [w for w in p.strip().lower().split() if w not in _STOPWORDS]
        phrase = " ".join(words).strip(".")
        if len(phrase) >= 3:
            result.append(phrase)
    return result

# ---------------------------------------------------------------------------
# Run BLIP once per class — use first sorted image as representative
# ---------------------------------------------------------------------------
labels: dict[str, list] = {}

for i, cls_name in enumerate(classes, 1):
    food_name = cls_name.replace("_", " ")
    prompt    = f"the ingredients in this {food_name} are"
    img_path  = sorted(by_class[cls_name])[0]

    try:
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(img, prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=60,
                                 num_beams=4, early_stopping=True)
        text = processor.decode(out[0], skip_special_tokens=True)
        if text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip()
        class_ingredients = _parse(text)
    except Exception as e:
        print(f"  [skip] {cls_name}: {e}")
        class_ingredients = []

    # every training image of this class gets the same label list
    for p in by_class[cls_name]:
        key = f"{p.parent.name}/{p.stem}"
        labels[key] = class_ingredients

    print(f"  [{i:03d}/{len(classes)}] {cls_name}: {class_ingredients}")

OUT_PATH.write_text(json.dumps(labels, indent=2))
print(f"\nDone. {OUT_PATH} — {len(labels)} image labels saved.")
