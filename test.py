import json
import torch
from pathlib import Path
from PIL import Image
from finetune import predict_class
from finetune import load_classifier

IMG_PTH="/share/nas165/princelu/nn_final/food11/food11/train/chicken_curry/253336.jpg"
_DB_PATH      = Path(__file__).parent / "nutrition_db.json"
_NUTRITION_DB = json.loads(_DB_PATH.read_text())
print(_NUTRITION_DB)

for db_key, values in _NUTRITION_DB.items():
    print(db_key)


def analyze_food(image_path: str, classifier=None) -> str:
    """Returns food_type using the fine-tuned EfficientNet-B0 classifier."""
    if classifier is not None:
        return predict_class(classifier, image_path)
    # Fallback: derive food type from the parent folder name if running on
    # dataset images (e.g. food11/train/chicken_curry/img.jpg)
    return Path(image_path).parent.name.replace("_", " ")

classifier = None
checkpoint = Path(__file__).parent / "checkpoints" / "best.pt"
if checkpoint.exists():

    classifier = load_classifier(str(checkpoint))
    print(f"Using fine-tuned EfficientNet-B0 from {checkpoint}")
else:
    print("No checkpoint found — falling back to folder name as food type.")

print(analyze_food(IMG_PTH,classifier))