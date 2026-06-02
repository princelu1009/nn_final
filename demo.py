"""
Demo — runs the full pipeline on 2 images per class across 10 food classes.

Output directory: demo_output/
  - <class>_<stem>.mp3   one audio file per image
  - results.json         all predictions + report text in one file

Usage:
    python demo.py
"""

import json
import asyncio
import random
import torch
import edge_tts
from pathlib import Path
from PIL import Image

from finetune import load_classifier, predict_class
import finetune_ing as fi
from collections import Counter as _Counter
import json as _json

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data/food-101/images"
META_FILE = BASE / "data/food-101/meta/test.txt"
OUT_DIR   = BASE / "demo_output"
OUT_DIR.mkdir(exist_ok=True)

IMAGES_PER_CLASS = 10
CHOSEN_CLASSES   = ["sashimi", "ramen"]
THRESHOLD        = 0.2   # optimal threshold from Experiment 4

# patch vocab from clean labels before anything reads NUM_INGREDIENTS
_clean = _json.loads((BASE / "ingredient_labels_clean.json").read_text())
_freq  = _Counter(ing for ings in _clean.values() for ing in ings)
_vocab = sorted(_freq.keys())
fi._IMAGE_LABELS    = _clean
fi.INGREDIENT_VOCAB = _vocab
fi.NUM_INGREDIENTS  = len(_vocab)
fi.ING2IDX          = {ing: i for i, ing in enumerate(_vocab)}

from finetune_ing import IngredientNet, NUM_INGREDIENTS, predict_ingredients




device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
cls_ckpt = BASE / "checkpoints/best.pt"
ing_ckpt = BASE / "checkpoints/ingredient_net.pt"

if not cls_ckpt.exists():
    raise FileNotFoundError(f"{cls_ckpt} not found — train the food classifier first.")
if not ing_ckpt.exists():
    raise FileNotFoundError(f"{ing_ckpt} not found — train IngredientNet first.")

classifier       = load_classifier(str(cls_ckpt))
ingredient_model = IngredientNet(NUM_INGREDIENTS).to(device)
ingredient_model.load_state_dict(
    torch.load(ing_ckpt, map_location=device, weights_only=True)
)
ingredient_model.eval()
print(f"Loaded classifier  ({cls_ckpt.name})")
print(f"Loaded IngredientNet  ({ing_ckpt.name},  {NUM_INGREDIENTS} ingredients)")

# ---------------------------------------------------------------------------
# Sample images — 2 per class from 10 randomly chosen classes
# ---------------------------------------------------------------------------
allowed = set(META_FILE.read_text().splitlines())

from collections import defaultdict
by_class: dict[str, list] = defaultdict(list)
for entry in allowed:
    cls = entry.split("/")[0]
    p   = DATA_DIR / f"{entry}.jpg"
    if p.exists():
        by_class[cls].append(p)

selected: list[tuple[str, Path]] = []
for cls in CHOSEN_CLASSES:
    paths = sorted(by_class[cls])
    for p in random.sample(paths, min(IMAGES_PER_CLASS, len(paths))):
        selected.append((cls, p))

print(f"\nSelected {len(selected)} images across {len(CHOSEN_CLASSES)} classes:")
for cls, p in selected:
    print(f"  {cls:30s}  {p.name}")

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def generate_report(food_type: str, ingredients: list) -> str:
    ing_str = ", ".join(ingredients) if ingredients else "various ingredients"
    return f"This is {food_type}. It contains the following ingredients: {ing_str}."


async def _save_audio(text: str, path: Path):
    communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    await communicate.save(str(path))


def synthesize_audio(text: str, path: Path) -> bool:
    try:
        asyncio.run(_save_audio(text, path))
        return True
    except Exception as e:
        print(f"  [edge-tts failed: {e}]")
        return False

# ---------------------------------------------------------------------------
# Run pipeline on each image
# ---------------------------------------------------------------------------
all_results = []

for true_class, img_path in selected:
    print(f"\n--- {true_class}/{img_path.name} ---")

    predicted_class = predict_class(classifier, str(img_path))
    ingredients     = predict_ingredients(ingredient_model, str(img_path),
                                          threshold=THRESHOLD)
    report_text     = generate_report(predicted_class, ingredients)

    audio_name = f"{true_class}_{img_path.stem}.mp3"
    audio_path = OUT_DIR / audio_name
    audio_ok   = synthesize_audio(report_text, audio_path)

    # copy image into demo_output/
    import shutil
    img_copy = OUT_DIR / f"{true_class}_{img_path.stem}{img_path.suffix}"
    shutil.copy2(img_path, img_copy)

    result = {
        "image":            str(img_path),
        "image_copy":       str(img_copy),
        "true_class":       true_class,
        "predicted_class":  predicted_class,
        "correct":          predicted_class.replace(" ", "_") == true_class,
        "ingredients":      ingredients,
        "report":           report_text,
        "audio_file":       audio_name if audio_ok else None,
    }
    all_results.append(result)

    print(f"  True class  : {true_class}")
    print(f"  Predicted   : {predicted_class}  {'✓' if result['correct'] else '✗'}")
    print(f"  Ingredients : {', '.join(ingredients) or '(none)'}")
    print(f"  Audio       : {audio_name if audio_ok else 'failed'}")

# ---------------------------------------------------------------------------
# Save results.json
# ---------------------------------------------------------------------------
summary = {
    "num_images":   len(all_results),
    "num_classes":  len(CHOSEN_CLASSES),
    "threshold":    THRESHOLD,
    "accuracy":     round(sum(r["correct"] for r in all_results) / len(all_results), 4),
    "results":      all_results,
}

out_json = OUT_DIR / "results.json"
out_json.write_text(json.dumps(summary, indent=2))
print(f"\nDone. {len(all_results)} images processed.")
print(f"Accuracy on demo set: {summary['accuracy']:.1%}")
print(f"Output saved to: {OUT_DIR}/")
print(f"  results.json + {len(all_results)} .mp3 files")
