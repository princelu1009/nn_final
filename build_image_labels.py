"""
One-time script: runs BLIP on every food-101 training image and saves
per-image ingredient labels to ingredient_labels.json.

The prompt never mentions the food type so BLIP must reason visually.
Labels are keyed by "class_name/image_stem", e.g. "apple_pie/1001116".

Run once before training IngredientNet:
    python build_image_labels.py
"""

import json
import re
import torch
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

IMAGES_DIR   = Path(__file__).parent / "data/food-101/images"
META_FILE    = Path(__file__).parent / "data/food-101/meta/train.txt"
OUT_PATH     = Path(__file__).parent / "ingredient_labels.json"
IMAGES_PER_CLASS = 750   # food-101 has 750 train images per class — label all of them

# ---------------------------------------------------------------------------
# Load the food-101 train split (so we only label training images)
# ---------------------------------------------------------------------------
allowed = set(META_FILE.read_text().splitlines())   # "apple_pie/1001116"
all_images = [
    IMAGES_DIR / f"{entry}.jpg"
    for entry in allowed
    if (IMAGES_DIR / f"{entry}.jpg").exists()
]

# limit to IMAGES_PER_CLASS images per class
from collections import defaultdict
by_class: dict[str, list] = defaultdict(list)
for p in all_images:
    by_class[p.parent.name].append(p)

images_to_label = []
for cls, paths in by_class.items():
    images_to_label.extend(sorted(paths)[:IMAGES_PER_CLASS])

print(f"Images to label: {len(images_to_label)} "
      f"({IMAGES_PER_CLASS} per class × {len(by_class)} classes)")

# ---------------------------------------------------------------------------
# Load existing labels so we can resume if the script was interrupted
# ---------------------------------------------------------------------------
if OUT_PATH.exists():
    labels: dict[str, list] = json.loads(OUT_PATH.read_text())
    print(f"Resuming — {len(labels)} labels already saved")
else:
    labels = {}

# ---------------------------------------------------------------------------
# Load BLIP
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                use_safetensors=True,
            ).to(device)
model.eval()

# ---------------------------------------------------------------------------
# Ingredient parsing
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
# Main loop
# ---------------------------------------------------------------------------
PROMPT = "the ingredients visible in this dish are"
SAVE_EVERY = 50

for i, img_path in enumerate(images_to_label, 1):
    key = f"{img_path.parent.name}/{img_path.stem}"

    if key in labels:   # already done
        continue

    try:
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(img, PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=60,
                                 num_beams=4, early_stopping=True)
        text = processor.decode(out[0], skip_special_tokens=True)

        if text.lower().startswith(PROMPT.lower()):
            text = text[len(PROMPT):].strip()

        labels[key] = _parse(text)

    except Exception as e:
        print(f"  [skip] {key}: {e}")
        labels[key] = []

    if i % SAVE_EVERY == 0:
        OUT_PATH.write_text(json.dumps(labels, indent=2))
        print(f"  [{i}/{len(images_to_label)}] saved checkpoint")

OUT_PATH.write_text(json.dumps(labels, indent=2))
print(f"\nDone. {OUT_PATH} — {len(labels)} image labels saved.")
