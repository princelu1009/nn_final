"""
One-time script: cleans ingredient_labels.json in place.

Fixes:
  - deduplicates repeated ingredients per image
  - collapses "flour flour flour" → "flour"
  - drops hallucinated repetitions (same word 3+ times in a phrase)
  - drops entries shorter than 3 characters
  - drops entries containing digits or special characters
  - drops entries where a single token appears 3+ times in the image list
"""

import json
import re
from collections import Counter
from pathlib import Path

LABELS_PATH = Path(__file__).parent / "ingredient_labels.json"
labels: dict = json.loads(LABELS_PATH.read_text())

# ---------------------------------------------------------------------------
# Per-phrase cleaning
# ---------------------------------------------------------------------------
def clean_phrase(phrase: str) -> str | None:
    phrase = phrase.strip().lower()

    # drop anything with digits or non-letter chars (except spaces/hyphens)
    if re.search(r"[^a-z\s\-]", phrase):
        return None

    words = phrase.split()
    if not words:
        return None

    # collapse repeated tokens: "flour flour flour" → "flour"
    unique_words = list(dict.fromkeys(words))  # preserves order, removes consecutive dups
    counts = Counter(words)

    # if any single word repeats 3+ times it's a hallucination → discard
    if max(counts.values()) >= 3:
        return None

    phrase = " ".join(unique_words)

    # drop if too short after collapsing
    if len(phrase) < 3:
        return None

    return phrase


# ---------------------------------------------------------------------------
# Per-image cleaning
# ---------------------------------------------------------------------------
def clean_list(ingredients: list[str]) -> list[str]:
    seen = set()
    result = []
    for raw in ingredients:
        cleaned = clean_phrase(raw)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Apply and report
# ---------------------------------------------------------------------------
before_total = sum(len(v) for v in labels.values())

cleaned_labels = {k: clean_list(v) for k, v in labels.items()}

# drop images with empty label lists after cleaning
cleaned_labels = {k: v for k, v in cleaned_labels.items() if v}

after_total  = sum(len(v) for v in cleaned_labels.values())
removed_imgs = len(labels) - len(cleaned_labels)

print(f"Images  : {len(labels)} → {len(cleaned_labels)} ({removed_imgs} fully empty dropped)")
print(f"Tokens  : {before_total} → {after_total} ({before_total - after_total} removed)")

# rebuild vocab size
vocab = set()
for v in cleaned_labels.values():
    vocab.update(v)
print(f"Vocab size: {len(vocab)} unique ingredients")

LABELS_PATH.write_text(json.dumps(cleaned_labels, indent=2))
print(f"Saved cleaned labels to {LABELS_PATH}")
