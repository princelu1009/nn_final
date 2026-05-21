"""
Food Calorie & Nutrition Estimation Pipeline
Image -> food_type (EfficientNet-B0) -> knowledge DB lookup
      -> ingredient + nutrition report -> audio summary
"""

import json
import torch
from pathlib import Path
from PIL import Image
from finetune import load_classifier
import asyncio, edge_tts

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Nutrition + ingredient knowledge base
# ---------------------------------------------------------------------------
_DB_PATH      = Path(__file__).parent / "nutrition_db.json"
_NUTRITION_DB = json.loads(_DB_PATH.read_text())

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _db_lookup(food_type: str) -> dict | None:
    def norm(s: str) -> str:
        return s.lower().strip().replace("_", " ")
    #key is food type name : e.g. pepperoni,pizza
    key = norm(food_type)
    for db_key, values in _NUTRITION_DB.items():
        if norm(db_key) == key:
            return values
    #a slice of pizza -> pizza
    for db_key, values in _NUTRITION_DB.items():
        if norm(db_key) in key:
            return values
    return None


def generate_report(food_type: str, nutrition: dict) -> str:
    """Build a natural language report from food_type + DB knowledge."""
    if "error" in nutrition:
        return f"This appears to be {food_type}, but no nutritional data was found."

    """
        food_type = "chicken curry"
        nutrition = {"portion_g": 300, "calories": 450.0, "protein_g": 36.0, ...}
    """
    base        = _db_lookup(food_type)
    """
        base = _db_lookup("chicken curry")
     → {"ingredients": ["chicken", "curry paste", ...], "allergens": [], "diet_tags": ["gluten-free", ...]}
    """
    ingredients = base.get("ingredients", []) if base else []
    allergens   = base.get("allergens",   []) if base else []
    diet_tags   = base.get("diet_tags",   []) if base else []

    """
    ing_str     = "chicken, curry paste, coconut milk, onion, garlic, ginger, tomato"
    allergy_str = ""                              # skipped — no allergens
    diet_str    = " Suitable for: gluten-free, dairy-free."
    """

    ing_str     = ", ".join(ingredients) if ingredients else "various ingredients"
    allergy_str = (f" Allergens: {', '.join(allergens)}." if allergens else "")
    diet_str    = (f" Suitable for: {', '.join(diet_tags)}." if diet_tags else "")

    return (
        f"This is {food_type}. "
        f"It typically contains {ing_str}."
        f"{allergy_str}"
        f"{diet_str} "
        f"This portion of {nutrition['portion_g']}g contains "
        f"{nutrition['calories']} calories, "
        f"{nutrition['protein_g']}g of protein, "
        f"{nutrition['carbs_g']}g of carbohydrates, "
        f"and {nutrition['fat_g']}g of fat."
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_food(image_path: str, classifier=None) -> str:
    """Returns food_type using the fine-tuned EfficientNet-B0 classifier."""
    if classifier is not None:
        from finetune import predict_class
        return predict_class(classifier, image_path)
    # Fallback: derive food type from the parent folder name if running on
    # dataset images (e.g. food11/train/chicken_curry/img.jpg)
    return Path(image_path).parent.name.replace("_", " ")


def _usda_lookup(food_type: str) -> dict | None:
    """Fetch nutrition per 100g from USDA FoodData Central API."""
    try:
        import requests
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": food_type, "pageSize": 1, "api_key": "DEMO_KEY"},
            timeout=5,
        )
        resp.raise_for_status()
        foods = resp.json().get("foods", [])
        if not foods:
            return None
        nutrients = {n["nutrientName"]: n["value"] for n in foods[0]["foodNutrients"]}
        result = {
            "calories_per_100g": nutrients.get("Energy", 0),
            "protein_g":         nutrients.get("Protein", 0),
            "carbs_g":           nutrients.get("Carbohydrate, by difference", 0),
            "fat_g":             nutrients.get("Total lipid (fat)", 0),
            "fiber_g":           nutrients.get("Fiber, total dietary", 0),
            "default_portion_g": 150,
            "ingredients":       [],
            "allergens":         [],
            "diet_tags":         [],
            "_source":           "USDA FoodData Central",
        }
        print(f"[USDA] fetched nutrition for '{food_type}'")
        return result
    except Exception as e:
        print(f"[USDA] lookup failed: {e}")
        return None


def estimate_nutrition(food_type: str) -> dict:
    """
    Looks up nutrition from local DB first.
    Falls back to USDA FoodData Central API if not found locally.
    """
    base = _db_lookup(food_type) or _usda_lookup(food_type)
    if base is None:
        return {
            "error": f"No nutrition data found for '{food_type}'",
            "tip":   "Add an entry to nutrition_db.json or check your internet connection.",
        }
    grams = base.get("default_portion_g", 150)
    ratio = grams / 100.0
    return {
        "portion_g": grams,
        "calories":  round(base["calories_per_100g"] * ratio, 1),
        "protein_g": round(base["protein_g"]         * ratio, 1),
        "carbs_g":   round(base["carbs_g"]           * ratio, 1),
        "fat_g":     round(base["fat_g"]             * ratio, 1),
        "fiber_g":   round(base["fiber_g"]           * ratio, 1),
    }


def synthesize_audio(text: str, output_path: str = "food_report.mp3") -> str:
    try:
        async def _speak():
            communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
            await communicate.save(output_path)
        asyncio.run(_speak())
        print(f"[edge-tts] Audio saved to {output_path}")
    except Exception as e:
        print(f"[edge-tts failed: {e}] falling back to pyttsx3")
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        print(f"[pyttsx3] Audio saved to {output_path}")
    return output_path


def run_pipeline(image_path: str, classifier=None, save_json: bool = True) -> dict:
    """Full pipeline: classify -> nutrition lookup -> report -> JSON + audio."""
    print(f"\nAnalyzing: {image_path}")

    food_type = analyze_food(image_path, classifier=classifier)
    nutrition = estimate_nutrition(food_type)
    report    = generate_report(food_type, nutrition)

    base = _db_lookup(food_type) or {}
    output = {
        "image":       image_path,
        "food_type":   food_type,
        "ingredients": base.get("ingredients", []),
        "allergens":   base.get("allergens",   []),
        "diet_tags":   base.get("diet_tags",   []),
        "nutrition":   nutrition,
        "report":      report,
    }

    print(f"Food type   : {food_type}")
    print(f"Ingredients : {', '.join(output['ingredients']) or 'unknown'}")
    print(f"Nutrition   : {json.dumps(nutrition, indent=2)}")
    print(f"Report      : {report}")

    if save_json:
        json_path = Path(image_path).with_suffix(".json")
        json_path.write_text(json.dumps(output, indent=2))
        print(f"Saved       : {json_path}")
        output["json_path"] = str(json_path)

    output["audio"] = synthesize_audio(report)
    return output


if __name__ == "__main__":
    import sys
    image_file = (
        sys.argv[1] if len(sys.argv) > 1
        else "/share/nas165/princelu/nn_final/food11/food11/train/chicken_curry/253336.jpg"
    )
    classifier = None
    checkpoint = Path(__file__).parent / "checkpoints" / "best.pt"
    classifier = load_classifier(str(checkpoint))
    print(f"Using fine-tuned EfficientNet-B0 from {checkpoint}")
    run_pipeline(image_file, classifier=classifier)


