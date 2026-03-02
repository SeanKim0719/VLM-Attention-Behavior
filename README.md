# VLM Attention Behavior Analysis

An empirical study on **how CLIP decides what something is** — which part of an image actually drives its classification, and whether that region is visually meaningful.

---

## Motivation

CLIP frequently achieves high classification accuracy based on spurious correlations rather than semantic visual features. A model that identifies a wolf because of the snowy background, not the wolf itself, is fundamentally unreliable. This project attempts to make that gap visible and measurable.

---

## Experiment Design

### Categories

**Cat, Dog, Fox, Wolf, Tiger** — selected for deliberate visual overlap. Coarse-grained categories (car, bicycle) were excluded in early pilots: they induced ceiling effects in confidence scores, obscuring the model's internal uncertainty.

### Stage 1 — Variant-Level Analysis

Each image is converted into 7 versions by removing different types of visual information:

| Variant | What is removed |
|---------|----------------|
| Full | Nothing (baseline) |
| Center 50% | Peripheral context |
| Top / Bottom Half | Spatial regions |
| Edges Only | Center subject |
| Blurry | Texture and fine detail |
| Silhouette | Color and texture; outline only |

Key metric: **Silhouette Drop Rate** — how much confidence falls when only the outline remains. A significant confidence degradation indicates a Bottom-Up dependency, where the model fails to perform robust recognition without fine-grained texture information.

### Stage 2 — Grid Importance Analysis

Each image is divided into a **4×4 grid**. Each cell is individually masked with neutral gray, and the resulting confidence change is recorded as an importance score.

| Score | Meaning |
|-------|---------|
| High positive | This region is critical for recognition |
| Near zero | This region is not contributing |
| Negative | This region actively suppresses correct classification |

Images are classified into four patterns based on importance distribution:

- **Local-Focused**: top 3 regions account for >70% of total importance
- **Global-Distributed**: importance spread evenly across the grid
- **Mixed**: neither clearly local nor global
- **Interference-Present**: significant negative values dominate

### Stage 3 — Attention Rollout

Attention flow is traced across all 12 ViT-B/32 transformer layers using Attention Rollout to visualize where the model looks when forming its final representation.

---

## Results

### Variant-Level: Silhouette Drop by Category

| Category | Mean Drop | Range | Pattern |
|----------|-----------|-------|---------|
| CAT | 8% | −1% ~ 33% | Top-Down |
| DOG | 32% | 1% ~ 68% | Hybrid |
| FOX | 73% | 33% ~ 96% | Bottom-Up |
| WOLF | 75% | 11% ~ 98% | Bottom-Up |
| TIGER | 63% | 0% ~ 91% | Bottom-Up |

Cat and Dog maintain recognition from outline alone. Fox, Wolf, and Tiger collapse without color and texture — their silhouettes are not discriminative enough to separate them from each other.

### Grid-Level: Per-Image Pattern Distribution

| Category | Local-Focused | Global-Distributed | Mixed | Interference | Dominant |
|----------|:---:|:---:|:---:|:---:|---------|
| CAT | 0% | 20% | 60% | 20% | Mixed |
| DOG | 40% | 20% | 0% | 40% | Local-Focused |
| FOX | 20% | 20% | 40% | 20% | Mixed |
| WOLF | 60% | 0% | 40% | 0% | Local-Focused |
| TIGER | 0% | 20% | 80% | 0% | Mixed |

**Notable cases:**

- **Wolf #2**: one grid cell (head/neck area) shows an importance drop of 0.170, far exceeding all others. The model's wolf/dog distinction hinges almost entirely on the face region.
- **Cat #4**: all top importance values are negative (peak: −0.291). The model classified correctly, but specific regions were actively working against it — likely an occlusion such as a blanket covering part of the subject.
- **Fox #5**: full confidence is only 0.283 with strong negative regions in the upper grid. The model is near classification failure, likely due to background competing with the subject.

### Attention Map Summary

Across 25 attention visualizations, roughly half showed the model attending to background or peripheral regions rather than the subject itself. Correctly classified images do not reliably correspond to well-grounded attention.

---

## Key Findings

**1. Recognition strategy shifts with category discriminability.**
When categories are visually distinct, CLIP uses shape. When they overlap, it shifts to texture and color. This is not a fixed model property — it changes depending on what the model needs to distinguish.

**2. The same category produces different strategies per image.**
Dog silhouette drop ranges from 1% to 68%. CLIP does not apply a single recognition strategy per category; it adapts to the composition of each image.

**3. Interference is a real and measurable phenomenon.**
Negative importance scores are not noise. They correlate with lower overall confidence and harder classification cases, and represent regions that actively mislead the model.

**4. Variant drop and grid concentration measure different things.**
Fox shows a massive silhouette drop but low per-cell importance. Its recognition depends on distributed texture — removing any single region leaves enough signal to recover, but removing all texture simultaneously causes collapse. The two analysis methods are complementary, not redundant.

**5. Correct label does not imply correct grounding.**
Multiple images were classified correctly while attention was concentrated on background or irrelevant regions. Accuracy alone does not indicate that the model is looking at the right thing.


---

## Limitations

- 5 images per category — observed patterns are not statistically conclusive
- 4×4 grid cannot localize features smaller than ~56×56 pixels; finer-grained masking may reveal clearer signals
- Results are specific to CLIP ViT-B/32 and may not generalize to other architectures
- Attention Rollout is an approximation of internal attention flow, not a ground-truth explanation
