# Project Summary — Solar Panel Anomaly Detection

> English summary kept for backup and as context for AI prompts. It explains the project root and
> what each notebook does. Source notebooks are the ground truth; this file mirrors their current state.

## Goal
Detect and classify anomalies in solar PV modules from **infrared (thermal) imagery**. The project
compares two approaches to the same problem on the same dataset:

- **Approach A — Classical Computer Vision** (`visao_classica.ipynb`)
- **Approach B — Deep Learning** (`visao_deep_learning_4.ipynb`, also exported as `visao_deep_learning_4.py`)

## Dataset
- Source: **InfraredSolarModules** (RaptorMaps), published at ICLR 2020 (AI for Earth Sciences workshop).
  Lives in the sibling folder `../InfraredSolarModules/` and is downloaded at runtime from GitHub.
- **20,000** grayscale infrared images, **24 x 40 px** each.
- **12 classes**: Cell, Cell-Multi, Cracking, Hot-Spot, Hot-Spot-Multi, Shadowing, Diode,
  Diode-Multi, Vegetation, Soiling, Offline-Module, No-Anomaly.
- **Heavily imbalanced**: No-Anomaly alone is 10,000 images (half the dataset); rarest classes
  (Diode-Multi, Soiling, Hot-Spot-Multi) have ~175–250.
- Labels come from `module_metadata.json` mapping `<image_number>` → `{image_filepath, anomaly_class}`.

---

## Approach A — Classical Computer Vision (`visao_classica.ipynb`)
Pure image-processing heuristic, **no learning**.

Pipeline:
1. Download + unzip dataset; load `module_metadata.json` into a pandas DataFrame.
2. Visualize grayscale samples per class.
3. Per-image processing (OpenCV):
   - Grayscale read.
   - **Thresholding** (binarize, threshold ~200–220) to isolate hot/bright regions.
   - **Mathematical morphology** — erosion + dilation, and `MORPH_OPEN` (opening) to clean noise.
     Small 2x2 kernel because images are tiny (24x40).
   - **Contour detection** (`findContours`, external), then take the **largest blob area**.
4. Hand-tuned classification rule on largest blob area:
   - `< 5` → **No-Anomaly**
   - `> 800` → **Offline-Module**
   - otherwise → **Cell**
5. Evaluation: accuracy + confusion matrix over the dataset.

**Limitation:** only 3 of the 12 classes are modeled; classification is a fixed area threshold,
so most anomaly types collapse into the wrong bucket.

---

## Approach B — Deep Learning (`visao_deep_learning_4.ipynb`) — current accuracy ~67%
Transfer learning with **fastai + ResNet34**.

Pipeline:
1. Download + unzip dataset; load JSON into a DataFrame.
2. **Class consolidation** (merge redundant variants):
   `Cell-Multi → Cell`, `Diode-Multi → Diode`, `Hot-Spot-Multi → Hot-Spot`.
3. **Balancing**: down-sample No-Anomaly to **1,800**, keep all other (anomaly) rows,
   concat + shuffle with `random_state=42`.
4. **DataLoaders**: `ImageDataLoaders.from_df`, `Resize(224)`,
   `aug_transforms(do_flip=True, max_rotate=0.0, max_lighting=0.0, max_warp=0.0)` — flip only.
5. **Training**:
   - `vision_learner(dls, resnet34, metrics=accuracy)`, `fit_one_cycle(4)` (frozen backbone) → save weights.
   - `lr_find()` to locate the best learning-rate band.
   - `unfreeze()` + `fit_one_cycle(10, lr_max=slice(1e-5, 1e-3))` — discriminative fine-tuning.
6. **Diagnostics**: confusion matrix, `plot_top_losses` (where the model was confidently wrong).
7. **Export**: `learn.export('modelo_termografico_final.pkl')` (architecture + weights) for inference.

**Known open issues (noted in the notebook):**
- Apply **class weights** so every class has equal training impact (balancing is only partial).
- **Reproducibility**: results vary between runs despite `set_seed(42)`.

---

## Comparison
| | Classical CV | Deep Learning |
|---|---|---|
| Learns from data | No (hand-tuned rules) | Yes (ResNet34 transfer learning) |
| Classes covered | 3 (No-Anomaly / Offline-Module / Cell) | Multiple (after consolidation) |
| Interpretability | High (explicit pipeline) | Lower (CNN) |
| Setup cost | Low, no GPU | Needs GPU (Colab T4) |
| Accuracy | Low / narrow | ~67% |
| Main weakness | Rigid area thresholds | Imbalance + reproducibility |

---

## Possible Improvements / Next Steps
**Deep learning**
- Add **class weights** (e.g. weighted loss / weighted sampling) for true balance.
- Make the run **fully reproducible** (seed every source of randomness end-to-end).
- Try **ResNet50** or progressive resizing; consider light real augmentation (small rotations are fine for thermal).
- Report **per-class precision/recall/F1**, not just global accuracy.
- Use a proper **held-out test split** (or k-fold), separate from validation.

**Classical CV**
- It only models 3 classes — either scope it down explicitly, or add features
  (blob count, position, aspect ratio, intensity stats) to reach more anomaly types.
- Tune threshold/area per target class instead of one global rule.

**Project-level**
- Put both approaches on **one shared train/val/test split** so they are comparable on identical data.
- Add an environment/`requirements` note (fastai, OpenCV, pandas, scikit-learn, seaborn, tqdm).
- Document the dataset path/download step once.
