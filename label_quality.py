"""
Experiment 2 — Label Quality Analysis (no model training needed)

Directly compares ingredient_labels_raw.json vs ingredient_labels_clean.json
across several quality dimensions to show that cleaning produces better labels.

Usage:
    python label_quality.py
"""

import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent

raw_labels   = json.loads((BASE / "ingredient_labels_raw.json").read_text())
clean_labels = json.loads((BASE / "ingredient_labels_clean.json").read_text())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def vocab_freq(labels: dict) -> Counter:
    freq = Counter()
    for ings in labels.values():
        freq.update(ings)
    return freq

def stats(labels: dict, freq: Counter) -> dict:
    counts_per_image = [len(v) for v in labels.values()]
    total_tokens     = sum(counts_per_image)
    n_images         = len(labels)
    vocab_size       = len(freq)
    singletons       = sum(1 for c in freq.values() if c == 1)
    stable           = sum(1 for c in freq.values() if c >= 10)
    empty_images     = sum(1 for c in counts_per_image if c == 0)

    return {
        "images":          n_images,
        "vocab_size":      vocab_size,
        "total_tokens":    total_tokens,
        "avg_per_image":   round(total_tokens / n_images, 2),
        "min_per_image":   min(counts_per_image),
        "max_per_image":   max(counts_per_image),
        "empty_images":    empty_images,
        "singleton_labels": singletons,           # appear in only 1 image — likely noise
        "stable_labels":   stable,                # appear in >= 10 images
        "stable_pct":      round(100 * stable / vocab_size, 1),
    }

raw_freq   = vocab_freq(raw_labels)
clean_freq = vocab_freq(clean_labels)
raw_stats  = stats(raw_labels,   raw_freq)
cln_stats  = stats(clean_labels, clean_freq)

# ---------------------------------------------------------------------------
# Overlap — what % of raw tokens survived cleaning
# ---------------------------------------------------------------------------
raw_vocab   = set(raw_freq.keys())
clean_vocab = set(clean_freq.keys())
removed     = raw_vocab - clean_vocab
retained    = raw_vocab & clean_vocab

overlap_pct      = round(100 * len(retained)  / len(raw_vocab), 1)
removed_pct      = round(100 * len(removed)   / len(raw_vocab), 1)

# token-level noise rate: of all raw label tokens across images, what % were noise?
raw_token_total   = raw_stats["total_tokens"]
noise_tokens      = sum(
    sum(1 for ing in ings if ing not in clean_vocab)
    for ings in raw_labels.values()
)
noise_token_pct   = round(100 * noise_tokens / raw_token_total, 1)

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
print("\n" + "="*58)
print("LABEL QUALITY ANALYSIS — Raw vs Cleaned")
print("="*58)
print(f"{'Metric':<30} {'Raw':>12} {'Cleaned':>12}")
print("-"*58)

rows = [
    ("Images",                raw_stats["images"],           cln_stats["images"]),
    ("Vocab size",            raw_stats["vocab_size"],        cln_stats["vocab_size"]),
    ("Total label tokens",    raw_stats["total_tokens"],      cln_stats["total_tokens"]),
    ("Avg labels / image",    raw_stats["avg_per_image"],     cln_stats["avg_per_image"]),
    ("Min labels / image",    raw_stats["min_per_image"],     cln_stats["min_per_image"]),
    ("Max labels / image",    raw_stats["max_per_image"],     cln_stats["max_per_image"]),
    ("Images with 0 labels",  raw_stats["empty_images"],      cln_stats["empty_images"]),
    ("Singleton labels",      raw_stats["singleton_labels"],  cln_stats["singleton_labels"]),
    ("Stable labels (>=10)",  raw_stats["stable_labels"],     cln_stats["stable_labels"]),
    ("Stable label %",        f"{raw_stats['stable_pct']}%",  f"{cln_stats['stable_pct']}%"),
]
for label, raw_val, cln_val in rows:
    print(f"{label:<30} {str(raw_val):>12} {str(cln_val):>12}")

print("="*58)
print("\nCleaning impact:")
print(f"  Vocab tokens removed   : {len(removed):,}  ({removed_pct}% of raw vocab)")
print(f"  Vocab tokens retained  : {len(retained):,}  ({overlap_pct}% of raw vocab)")
print(f"  Noisy token instances  : {noise_tokens:,}  ({noise_token_pct}% of all raw label tokens)")

# ---------------------------------------------------------------------------
# Examples of removed noise tokens
# ---------------------------------------------------------------------------
print("\nExamples of removed noise tokens (raw only, sorted by frequency):")
removed_freq = {ing: raw_freq[ing] for ing in removed}
top_removed  = sorted(removed_freq.items(), key=lambda x: -x[1])[:20]
for ing, cnt in top_removed:
    print(f"  {cnt:>5}x  {ing!r}")

# ---------------------------------------------------------------------------
# Stable vocab comparison
# ---------------------------------------------------------------------------
print("\nTop 20 stable ingredients in cleaned labels (freq >= 10):")
top_clean = sorted(((ing, cnt) for ing, cnt in clean_freq.items() if cnt >= 10),
                   key=lambda x: -x[1])[:20]
for ing, cnt in top_clean:
    print(f"  {cnt:>5}x  {ing}")

# ---------------------------------------------------------------------------
# Save summary for report
# ---------------------------------------------------------------------------
summary = {
    "raw":   raw_stats,
    "clean": cln_stats,
    "overlap": {
        "vocab_tokens_removed":  len(removed),
        "vocab_tokens_retained": len(retained),
        "removed_pct":           removed_pct,
        "noise_token_instances": noise_tokens,
        "noise_token_pct":       noise_token_pct,
    }
}
(BASE / "label_quality_results.json").write_text(json.dumps(summary, indent=2))
print("\nSaved to label_quality_results.json")
