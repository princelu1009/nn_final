"""
Food Recognition Pipeline
Image -> food_type (EfficientNet-B0)
      -> ingredients (IngredientNet)
      -> natural language report -> audio summary
"""

import json
import asyncio
import edge_tts
import torch
from pathlib import Path
from PIL import Image
from finetune import load_classifier
from finetune_ing import IngredientNet, NUM_INGREDIENTS, predict_ingredients

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# IngredientNet loader
# ---------------------------------------------------------------------------
def load_ingredient_net(checkpoint: str) -> IngredientNet:
    model = IngredientNet(NUM_INGREDIENTS)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded IngredientNet from {checkpoint}  ({NUM_INGREDIENTS} ingredients)")
    return model

# ---------------------------------------------------------------------------
# Food classification
# ---------------------------------------------------------------------------
def analyze_food(image_path: str, classifier=None) -> str:
    """Returns food_type using the fine-tuned EfficientNet-B0 classifier."""
    if classifier is not None:
        from finetune import predict_class
        return predict_class(classifier, image_path)
    return Path(image_path).parent.name.replace("_", " ")

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(food_type: str, ingredients: list) -> str:
    ing_str = ", ".join(ingredients) if ingredients else "various ingredients"
    return (
        f"This is {food_type}. "
        f"It contains the following ingredients: {ing_str}."
    )

# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(image_path: str, classifier=None, ingredient_model=None, save_json: bool = True) -> dict:
    print(f"\nAnalyzing: {image_path}")

    food_type = analyze_food(image_path, classifier=classifier)

    if ingredient_model is not None:
        ingredients     = predict_ingredients(ingredient_model, image_path)
        ingredients_src = "IngredientNet"
    else:
        ingredients     = []
        ingredients_src = "none"

    report = generate_report(food_type, ingredients)

    output = {
        "image":           image_path,
        "food_type":       food_type,
        "ingredients":     ingredients,
        "ingredients_src": ingredients_src,
        "report":          report,
    }

    print(f"Food type   : {food_type}")
    print(f"Ingredients : {', '.join(ingredients) or 'unknown'}")
    print(f"Report      : {report}")

    if save_json:
        json_path = Path(image_path).with_suffix(".json")
        json_path.write_text(json.dumps(output, indent=2))
        print(f"Saved       : {json_path}")
        output["json_path"] = str(json_path)

    output["audio"] = synthesize_audio(report)
    return output

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    image_file = (
        sys.argv[1] if len(sys.argv) > 1
        else "/share/nas165/princelu/nn_final/data/food-101/images/sushi/2323447.jpg"
    )

    checkpoint = Path(__file__).parent / "checkpoints" / "best.pt"
    classifier = load_classifier(str(checkpoint))
    print(f"Using EfficientNet-B0 from {checkpoint}")

    ingredient_model = None
    ing_ckpt = Path(__file__).parent / "checkpoints" / "ingredient_net.pt"
    if ing_ckpt.exists():
        ingredient_model = load_ingredient_net(str(ing_ckpt))
    else:
        print(f"[warning] {ing_ckpt} not found — ingredients will be empty")

    run_pipeline(image_file, classifier=classifier, ingredient_model=ingredient_model)
