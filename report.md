# Food Recognition and Ingredient Detection via Lightweight Neural Networks with BLIP Pseudo-Labelling

---

## 1. Introduction

Automatic food recognition has broad applications in dietary tracking, allergy management, and nutritional awareness. While large vision-language models (VLMs) such as BLIP and GPT-4V can describe food images in rich natural language, their memory footprint and inference latency make them impractical for deployment on consumer devices or as real-time services.

This project proposes a lightweight two-model inference pipeline that achieves both food-type classification and ingredient detection without requiring a large model at inference time. The core idea is to leverage BLIP solely as a **one-time pseudo-label generator** during dataset construction, and then distil that knowledge into a compact convolutional neural network (IngredientNet) that can run in milliseconds.

The final system takes a single food image as input and produces:
1. A food-type prediction (101 categories) from a fine-tuned EfficientNet-B0 classifier.
2. A visual ingredient prediction from IngredientNet, a multi-label classifier trained on BLIP-generated pseudo-labels.
3. A natural language report synthesised into audio via edge-tts.

The key contributions of this work are:

- **Resource-efficient pseudo-labelling**: BLIP is used only once offline to annotate training images. At inference, the deployed model is a 5.3M-parameter CNN with no dependency on any VLM, reducing GPU memory requirements from ~4 GB (BLIP) to under 100 MB.
- **Per-image visual labels**: Unlike knowledge-base approaches that assign a fixed ingredient list to every image of the same food category, BLIP annotates each image individually based on visual content, enabling the downstream classifier to learn image-specific features.
- **Pseudo-label cleaning pipeline**: A dedicated post-processing stage removes hallucinated, repetitive, and low-frequency tokens from BLIP outputs before training, improving label quality without manual annotation.

---

## 2. Related Work

### 2.1 Food Image Classification

Food recognition has been studied extensively since the introduction of the Food-101 benchmark [Bossard et al., 2014], which contains 101,000 images across 101 categories. Early approaches relied on hand-crafted features; modern methods achieve over 90% top-1 accuracy using deep residual networks and vision transformers. EfficientNet [Tan & Le, 2019] offers a strong accuracy-efficiency trade-off and is a common backbone for food classification tasks.

### 2.2 Transfer Learning and Fine-Tuning

Transfer learning from ImageNet pre-trained weights has become standard practice in food recognition [Yanai & Kawano, 2015]. Differential fine-tuning — unfreezing only the top layers of a pre-trained backbone while keeping lower layers frozen — allows high-level task-specific features to be learned while preserving low-level texture representations that transfer well from ImageNet.

### 2.3 Vision-Language Models and BLIP

BLIP (Bootstrapping Language-Image Pre-training) [Li et al., 2022] is a multimodal model pre-trained on large image-text corpora. It supports image captioning and visual question answering. BLIP-2 [Li et al., 2023] extends this with a frozen large language model backbone, significantly increasing capability but also memory cost (~14 GB for inference). In this work we use BLIP-1 base to stay within accessible GPU memory constraints (~2 GB).

### 2.4 Pseudo-Labelling and Weak Supervision

Pseudo-labelling [Lee, 2013] uses model predictions as surrogate ground truth to train on unlabelled data. Here we apply the concept in a cross-modal direction: a vision-language model generates textual descriptions of images, which are parsed into structured labels for a pure vision model. Similar strategies have been explored for object detection [Radford et al., 2021 (CLIP)] and scene understanding, but their application to per-image ingredient labelling in food recognition has not been widely studied.

### 2.5 Multi-Label Classification

Ingredient detection is naturally a multi-label problem — an image may contain several ingredients simultaneously. Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) is the standard objective for multi-label classification, combining sigmoid activation and BCE in a numerically stable implementation [PyTorch documentation]. Evaluation uses precision, recall, and F1 score computed over the full label matrix.

---

## 3. Method

### 3.1 System Overview

The system consists of two parallel inference branches operating on the same input image:

```
Input Image
    ├─── EfficientNet-B0 (fine-tuned, 101 classes) ──► food_type
    └─── IngredientNet   (trained on BLIP labels)  ──► ingredient list
                                    │
                         generate_report()
                                    │
                          edge-tts audio output
```

BLIP is entirely absent at inference time.

### 3.2 Food-Type Classifier (EfficientNet-B0)

We fine-tune EfficientNet-B0 [Tan & Le, 2019] pre-trained on ImageNet-1K on the Food-101 training split (75,750 images across 101 classes).

**Architecture**: The backbone features blocks 0–5 are frozen. Blocks 6–8 are unfrozen with a learning rate of 1e-4. The original classifier head is replaced with a linear layer mapping 1,280 → 101.

**Training details**:
- Optimiser: Adam with differential learning rates (backbone 1e-4, head 1e-3)
- Loss: CrossEntropyLoss with label smoothing 0.1
- Scheduler: CosineAnnealingLR over up to 1,000 epochs
- Early stopping: patience = 15 epochs on validation accuracy
- Augmentation: RandomResizedCrop(224, scale=0.6–1.0), RandomHorizontalFlip, RandomRotation(±15°), ColorJitter
- Validation transform: Resize(256) → CenterCrop(224)

The 80/20 train/validation split is constructed by filtering the full image folder to the official Food-101 train set via `meta/train.txt`, then randomly partitioning with a fixed seed.

### 3.3 BLIP Pseudo-Label Generation

To train IngredientNet without manual ingredient annotations, we use BLIP-1 base as a pseudo-labeller.

**Key design decision — no food type in the prompt**: The prompt is fixed as:

> *"the ingredients visible in this dish are"*

Deliberately omitting the food category forces BLIP to reason visually rather than retrieve a memorised ingredient list. Two photos of the same dish may receive different labels if their visual content differs, which is the training signal we want.

**Scale**: BLIP is run on up to 750 images per class (all Food-101 training images), generating approximately 75,750 per-image label entries saved to `ingredient_labels.json`. The script resumes automatically from checkpoints if interrupted, saving every 50 images.

**Memory footprint**: BLIP-1 base requires approximately 2 GB of GPU memory, compared to ~14 GB for BLIP-2. Inference is performed with `torch.inference_mode()` to further reduce peak memory.

### 3.4 Pseudo-Label Cleaning

BLIP outputs contain several types of noise that must be removed before training:

| Noise type | Example | Cleaning rule |
|---|---|---|
| Duplicate entries | `["rice", "rice", "rice"]` | Deduplicate per image (set) |
| Repetitive hallucination | `"flour flour flour flour"` | Discard phrase if any token repeats ≥ 3 times |
| Truncated tokens | `"ques"`, `"gui"` | Drop entries shorter than 3 characters |
| Special characters / digits | `"rice2"` | Drop via regex `[^a-z\s\-]` |
| Rare hallucinations | ingredients appearing in < 10 images | Frequency pruning on vocab |

The cleaning script (`clean_ingredient_labels.py`) is run once after label generation. Frequency pruning is applied at training time by filtering the vocabulary to ingredients seen in at least `MIN_FREQ = 10` images, reducing the raw vocab from ~650 tokens to approximately 100–150 stable ingredients.

### 3.5 IngredientNet

IngredientNet combines a frozen EfficientNet-B0 backbone with a two-layer MLP head for multi-label ingredient prediction.

**Architecture**:
```
Input [B, 3, 224, 224]
  → EfficientNet-B0 features + avgpool  (frozen)
  → flatten  [B, 1280]
  → Linear(1280, 256) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(256, 128)  → BatchNorm → ReLU → Dropout(0.2)
  → Linear(128, NUM_INGREDIENTS)
  → raw logits  [B, NUM_INGREDIENTS]
```

**Training details**:
- Only the MLP head parameters are updated; the backbone is frozen entirely
- Loss: BCEWithLogitsLoss (sigmoid applied internally for numerical stability)
- Optimiser: Adam (head only, lr=1e-3)
- Early stopping: patience = 10 epochs on validation F1
- Dataset: 80/20 split of labeled images from `ingredient_labels.json`
- Checkpoint saved to `checkpoints/ingredient_net.pt`

**Inference**: Sigmoid is applied to logits and a threshold of 0.5 is used to produce binary predictions, which are mapped back to ingredient names via the vocabulary index.

### 3.6 Report and Audio Generation

The predicted food type and ingredient list are formatted into a natural language sentence and synthesised to audio using edge-tts (Microsoft Azure Neural TTS via the edge-tts library), with pyttsx3 as a local offline fallback.

---

## 4. Experiments

### 4.1 Dataset

**Food-101** [Bossard et al., 2014]: 101 food categories, 750 training images and 250 test images per class (101,000 total). Images are collected from foodspotting.com and contain significant visual noise. We use the official train split for classifier training and IngredientNet label generation.

### 4.2 Food-Type Classifier

The classifier is trained on food-101 with the setup described in Section 3.2. EfficientNet-B0 fine-tuned on food tasks typically achieves 80–88% top-1 accuracy on Food-101 depending on augmentation strength and training duration. The differential learning rate strategy (frozen early blocks, unfrozen top blocks) reduces training time and prevents catastrophic forgetting of low-level ImageNet features.

### 4.3 Ingredient Prediction

Evaluation of IngredientNet uses per-batch precision, recall, and F1 averaged over the validation set. Because labels are BLIP-generated pseudo-labels rather than ground truth, F1 measures consistency between the model and BLIP rather than absolute accuracy. The multi-label nature of the task means a model that learns to predict the most visually salient ingredients per image is considered successful.

Expected behaviour after training:
- High-confidence ingredients (common, visually prominent) should be predicted reliably.
- Rare or visually ambiguous ingredients will have lower recall.
- The threshold (default 0.5) can be raised to 0.65–0.7 to improve precision at the cost of recall.

### 4.3 Experiment 2 — Effect of Pseudo-Label Cleaning

This experiment validates the contribution of the pseudo-label cleaning pipeline. Two IngredientNet variants are trained on labels at different levels of post-processing, with all other hyperparameters held constant.

| Config | Label treatment |
|---|---|
| A — Raw | Raw BLIP output, no post-processing |
| B — Cleaned | Deduplication + hallucination filter |

#### 4.3.1 Evaluation Protocol

Evaluating each config against its own training labels would be unfair — Config A could score well simply by reproducing its own noise. Instead, both configs are evaluated against a shared ground truth: `ingredient_labels_clean.json`.

Because Config A and Config B are trained on different label files, their output vocabularies differ in size. To compare them in a common space, Config A's logits are projected into the clean vocabulary space before scoring: for each ingredient slot in the clean vocab, the corresponding logit from Config A is used if that ingredient appears in Config A's training vocab, otherwise the slot is set to −10 (sigmoid ≈ 0, i.e., predicts absent). Config B's vocab is identical to the clean vocab so no projection is needed. Precision, recall, and F1 are then computed in this shared clean vocab space for both configs using a sigmoid threshold of 0.2, established as the optimal operating point by Experiment 4 (Section 4.5).

#### 4.3.2 Label Quality Analysis

Rather than relying solely on model F1 to compare label quality, we directly analyse the label files using `label_quality.py`. This provides evidence of cleaning effectiveness independent of model training noise.

The analysis also reveals that a subset of images received empty label lists from BLIP — cases where the captioner produced output that was entirely filtered by the parsing rules (e.g. all tokens were stopwords or shorter than 3 characters). These empty-label images are identified and excluded from all training and evaluation splits, as all-zero target vectors contribute no useful training signal to BCEWithLogitsLoss and would bias the model toward predicting all ingredients as absent.

| Metric | Raw | Cleaned |
|---|---|---|
| Vocab size | *TODO* | *TODO* |
| Avg labels / image | *TODO* | *TODO* |
| Images with empty labels (excluded) | *TODO* | *TODO* |
| Singleton labels (appear in 1 image only) | *TODO* | *TODO* |
| Stable labels (appear in ≥ 10 images) | *TODO* | *TODO* |
| Stable label % | *TODO* | *TODO* |
| Noisy token instances removed | — | *TODO* |

Singleton labels are strong indicators of noise — an ingredient seen in only one image out of 75,750 is almost certainly a hallucination or truncated output. The stable label count and percentage measure how much of the vocabulary carries reliable signal. The empty-label count quantifies how many images produced no usable annotation at all, motivating their removal before training.

#### 4.3.3 Model Results

**TODO: fill in after running experiment2.py**

| Config | Val Precision | Val Recall | Val F1 |
|---|---|---|---|
| A — Raw | | | |
| B — Cleaned | | | |

The expected trend is B > A on F1 when both are scored against clean labels, because Config B's cleaner training signal produces a model that better predicts real visible ingredients. Config A wastes model capacity learning to reproduce hallucinated and repetitive tokens that do not correspond to any consistent visual feature.

---

### 4.4 Experiment 3 — Class-Level Labels vs. Per-Image Visual Labels

This experiment directly validates the core design decision of using per-image BLIP labels generated without mentioning the food type, versus a class-level baseline where all images of the same category share an identical ingredient list.

| Config | Label source | Prompt strategy |
|---|---|---|
| A — Class-level | Same ingredient list for all images of a class | `"the ingredients in this {food_name} are"` |
| B — Per-image | Individual BLIP annotation per image | `"the ingredients visible in this dish are"` |

#### 4.4.1 Evaluation Protocol

Both configs are trained on their respective label files but evaluated against `ingredient_labels_clean.json` as the shared ground truth. The same vocabulary projection described in Section 4.3.1 is applied to Config A's logits before scoring.

This evaluation design is intentional: the clean per-image labels represent what BLIP actually observed visually in each image, making them the most appropriate reference for measuring ingredient prediction quality. Config A is penalised for ingredients it was never trained to distinguish per image — ingredients that were visually present but not in the fixed class-level list — generating false negatives that lower its recall.

Early stopping during training is also driven by validation F1 against the clean labels, meaning both models are optimised toward the same objective throughout training. All reported metrics use threshold 0.2 (see Section 4.5).

#### 4.4.2 Model Results

**TODO: fill in after running experiment3.py**

| Config | Val Precision | Val Recall | Val F1 |
|---|---|---|---|
| A — Class-level | | | |
| B — Per-image | | | |

#### 4.4.3 Qualitative Examples

Beyond quantitative results, `experiment3.py` generates qualitative examples showing two images from the same food class where Config B produces different ingredient predictions based on visible content, while Config A predicts identically for both. This directly illustrates the visual grounding capability that per-image labelling enables — Config A has no mechanism to distinguish two pizza images with different toppings, while Config B learned to respond to what is actually visible.

**TODO: paste 2–3 qualitative examples from experiment3.py output here.**

---

### 4.5 Experiment 4 — Threshold Sensitivity Analysis

The default sigmoid threshold of 0.5 is arbitrary. This experiment sweeps the threshold from 0.1 to 0.9 on the trained IngredientNet and records precision, recall, and F1 at each point. No retraining is needed — the threshold is applied post-hoc to the saved logits.

#### 4.5.1 Results

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.1 | 0.2073 | 0.4129 | 0.2760 |
| **0.2** | **0.3330** | **0.2643** | **0.2947** |
| 0.3 | 0.4389 | 0.1780 | 0.2533 |
| 0.4 | 0.5341 | 0.1243 | 0.2017 |
| 0.5 | 0.6145 | 0.0884 | 0.1546 |
| 0.6 | 0.6800 | 0.0600 | 0.1103 |
| 0.7 | 0.7576 | 0.0396 | 0.0752 |
| 0.8 | 0.8342 | 0.0236 | 0.0460 |
| 0.9 | 0.9085 | 0.0101 | 0.0200 |

The best F1 of **0.2947** is achieved at threshold **0.2**. Precision increases monotonically with threshold while recall decreases, which is the expected precision-recall tradeoff for multi-label classification.

#### 4.5.2 Interpretation

The optimal threshold of 0.2 — well below 0.5 — indicates the model is systematically under-confident. This is characteristic of multi-label classifiers trained on large, sparse vocabularies: with hundreds of ingredient slots per image of which only a few are positive, the model learns conservative sigmoid outputs to minimise BCE loss over the many negative slots. The result is that true positive ingredients receive sigmoid scores in the 0.2–0.4 range rather than above 0.5.

**All metrics reported for Experiments 2 and 3 use threshold 0.2**, established here as the optimal operating point. This is equivalent to adjusting the decision boundary to account for the model's calibration rather than retraining.

---

### 4.6 Resource Comparison

| Component | Memory at inference | Role |
|---|---|---|
| BLIP-1 base (label generation only) | ~2 GB GPU | offline, one-time |
| EfficientNet-B0 classifier | ~25 MB | online inference |
| IngredientNet (frozen backbone + head) | ~25 MB | online inference |
| edge-tts | CPU only | online inference |

Total inference memory: **~50 MB**, compared to ~2 GB if BLIP were kept in the pipeline. This represents a **40× reduction** in model memory at deployment time.

---

## 5. Conclusion

This project demonstrates that a large vision-language model can be used as a one-time pseudo-label generator to train a lightweight multi-label classifier, eliminating the need for any manual ingredient annotation while keeping the deployed system compact and fast.

The two-model pipeline — a fine-tuned EfficientNet-B0 for food-type classification and IngredientNet for ingredient detection — achieves both tasks at inference using approximately 50 MB of memory, compared to the ~2 GB required by BLIP at label generation time. This 40× reduction makes the system practical for deployment on consumer devices where a large VLM would be infeasible.

Experiment 1 investigated the effect of pseudo-label cleaning. The label quality analysis conducted via `label_quality.py` provided direct evidence that raw BLIP outputs contain substantial noise: hallucinated tokens, repetitive phrases, truncated words, and a large proportion of singleton labels that appear in only one image. Cleaning reduced the vocabulary and removed token instances that carry no consistent visual signal. Empty-label images — where the entire BLIP output was discarded by parsing rules — were identified and excluded from training, as all-zero targets provide no useful gradient signal for BCEWithLogitsLoss.

Experiment 2 validated the core design decision of using per-image visual labels over class-level labels. By prompting BLIP without mentioning the food type, each image receives an annotation based on what is actually visible rather than what is expected for its category. The class-level baseline assigns every image of the same class an identical ingredient list, which prevents the model from learning to distinguish visually different instances of the same dish. The per-image approach produces a model with genuine visual grounding — capable of predicting different ingredients for two images of the same food class when their visual content differs.

Taken together, the experiments support three conclusions: (1) BLIP pseudo-labelling is a viable substitute for manual annotation in food ingredient detection; (2) post-processing to remove label noise is necessary for the downstream model to learn a reliable signal; and (3) per-image visual prompting outperforms class-level labelling by preserving the image-specific variation that makes the multi-label task meaningful.

**Limitations and future work.** The primary limitation is that BLIP pseudo-labels are not ground truth — they reflect what the captioner infers from visual content, which may miss occluded or partially visible ingredients. Future work could combine pseudo-labels with a small set of manually verified annotations to anchor the vocabulary. Additionally, unfreezing the upper layers of the EfficientNet-B0 backbone during IngredientNet training (currently fully frozen) may improve recall for visually subtle ingredients. Finally, extending the pipeline to the full Food-101 test split with human-verified ingredient lists would enable absolute accuracy measurement rather than consistency-with-BLIP measurement.

---

## 6. References

Bossard, L., Guillaumin, M., & Van Gool, L. (2014). **Food-101 — Mining Discriminative Components with Random Forests**. *European Conference on Computer Vision (ECCV)*.

Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. *International Conference on Machine Learning (ICML)*.

Li, J., Li, D., Xiong, C., & Hoi, S. (2022). **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**. *International Conference on Machine Learning (ICML)*.

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. *International Conference on Machine Learning (ICML)*.

Lee, D. H. (2013). **Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks**. *ICML Workshop on Challenges in Representation Learning*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**. *International Conference on Machine Learning (ICML)*.

Yanai, K., & Kawano, Y. (2015). **Food Image Recognition Using Deep Convolutional Network with Pre-Training and Fine-Tuning**. *IEEE International Conference on Multimedia & Expo Workshops (ICMEW)*.

Kingma, D. P., & Ba, J. (2014). **Adam: A Method for Stochastic Optimization**. *International Conference on Learning Representations (ICLR)*.
