# Fashion Feature Extraction Pipeline — Project Brief

## 1. Executive Summary

Pipeline end-to-end de extracción de atributos visuales de prendas de moda. Lee una imagen, segmenta la prenda, extrae color, y clasifica 7 atributos detallados (color_family, pattern, fit, fabric, style, waist_rise/neck_style). Produce JSON estructurado para alimentar un sistema de recomendación downstream.

**Contexto Master's:** Trabajo práctico de Visión Computarizada (Universidad de San Andrés). Prototipo end-to-end sin GPU local, entrenado en Google Colab (T4 gratuito).

---

## 2. Data & Domain

### Datasets
- **Pants CSV** (Abercrombie): 41,868 registros con URL de imagen + 7 atributos en inglés
- **Tops CSV** (Macy's): 78,686 registros con URL de imagen + 7 atributos en inglés
- **Prototype subset**: ~15k pants + ~15k tops para entrenar (descargadas on-demand en Colab)

### Atributos
| Atributo | Clases (ejemplo) |  Pants | Tops | Peso |
|----------|-----------------|--------|------|------|
| color_family | neutrals_dark, blues, reds, greens, ... | ✓ | ✓ | 1.0 |
| pattern | liso, rayas, cuadros, estampado, ... | ✓ | ✓ | 1.0 |
| fit_silhouette | slim, regular, relaxed, wide, oversized | ✓ | ✓ | 0.9 |
| fabric_content | algodón, denim, punto, sintético, seda | ✓ | ✓ | 0.8 |
| dressing_style | minimalista, clásico, casual, sport, formal | ✓ | ✓ | 0.7 |
| waist_rise | mid_rise, high_rise, low_rise | ✓ |   | 0.6 |
| neck_style | redondo, en_V, polo, cuello_alto, barco |   | ✓ | 0.6 |

### Key Challenges
- **Long-tail imbalance**: algunas clases <1% del dataset → agrupadas como "otro"
- **Vocabulary inconsistency**: Abercrombie y Macy's usan etiquetas diferentes en inglés
- **Missing data**: ~15-20% de los registros no tienen valor en ciertos atributos
- **Datos crudos**: JSON array con strings en inglés (ej: "Neutrals-Dark" → "neutrals_dark")

---

## 3. Architecture

### Pipeline Flow

```
IMAGE
  ↓
[1. SEGMENTATION]
  SegFormer B2 (mattmdjaga/segformer_b2_clothes)
  → Salida: máscara binaria + categoría dominante (upper_body/lower_body)
  ↓
[2. COLOR EXTRACTION]
  K-Means en espacio LAB + color naming
  → Salida: hex dominante, palette, pattern (liso/rayas/cuadros/estampado)
  ↓
[3. ROUTING LOGIC]
  IF upper_body → TOPS CLASSIFIER
  ELSE → PANTS CLASSIFIER
  ↓
[4. MULTI-TASK CLASSIFICATION]
  ViT-base backbone + 6 task heads (LoRA-friendly)
  → Salida: 7 atributos con confidence
  ↓
JSON OUTPUT
  {
    "image_path": "...",
    "segmentation": { "dominant_category": "upper_body", "mask_area_px": 50000 },
    "color": { "dominant": { "hex": "#2E2E2E", "name": "black" }, "palette": [...], "pattern": {...} },
    "garment_type": "tops",
    "attributes": {
      "color_family": { "label": "neutrals_dark", "confidence": 0.92 },
      "pattern": { "label": "liso", "confidence": 0.88 },
      ...
    }
  }
```

### Models
1. **SegFormer B2**: segmentación semántica (8 categorías + background). Checkpoint pre-entrenado.
2. **ViT-base (pants)**: 224×224, backbone congelado → decongelado, 6 heads independientes
3. **ViT-base (tops)**: igual, pero con `neck_style` en lugar de `waist_rise`
4. **K-Means + colores**: No ML; reglas clásicas (HSV + CSS color names)

### No Incluido (POC)
- **Brand classification**: Logo-2K+ no tiene logos sobre prendas reales; fuera de scope
- **Fine-grained attributes**: tallas exactas, tipo de tela en detalle
- **Demographic bias mitigation**: future work

---

## 4. Technical Stack

### Core ML
- **PyTorch 2.0+**
- **Transformers (HuggingFace)**: ViT, SegFormer, AutoTokenizer
- **TorchVision**: transforms, pretrained models
- **scikit-learn**: stratified splits, confusion matrices

### Infrastructure
- **Local**: Python 3.10+, pip
- **Cloud**: Google Colab (T4 GPU, ~12h free/month)
- **Storage**: Google Drive para CSVs, modelos, resultados

### Data Processing
- **Pandas**: CSV load/merge/splits
- **PIL**: image ops (resize, crop, mask)
- **NumPy**: arrays, normalization
- **Concurrent.futures**: parallel image downloads (8 workers)

### Testing & Validation
- **pytest**: unit tests (sin GPU, sin red)
- **nbformat**: notebook validation
- **YAML**: single config file, env-agnostic

---

## 5. Workflow

### Phase 1: EDA (Local, ~1h)
**Notebook:** `01_eda_csv_datasets.ipynb`

1. Load CSVs, check nulls/duplicates
2. String consistency: fuzzy matching (SequenceMatcher)
3. Cardinality + Shannon entropy → identify long-tail
4. Cramér's V heatmaps (attribute correlation)
5. Image validation: download stratified sample (200), compute HSV stats
6. **Outputs:** class_weights dict, EN→ES mapping, splits suggestions

### Phase 2: Data Prep (Colab, ~2-3h)
**Notebook:** `02_prepare_data_colab.ipynb`

1. Mount Google Drive
2. Clone repo + install dependencies
3. `download_images.py`: fetch 30k images (ThreadPoolExecutor, MD5 cache)
4. `prepare_splits.py`: stratified 70/15/15 splits, filter min_samples_per_class
5. **Outputs:** `data/splits/{pants,tops}_{train,val,test}.csv` en Drive

### Phase 3: Training (Colab T4, ~12h total)
**Notebooks:** `03_train_pants_colab.ipynb`, `04_train_tops_colab.ipynb`

1. Load config → instantiate `MultiTaskFashionClassifier.from_config()`
2. Prepare dataloaders (`CSVImageDataset` + transforms)
3. **Training (2 phases):**
   - Phase 1: Backbone frozen, only heads trainable (3-5 epochs)
   - Phase 2: Unfreeze backbone, discriminative LRs (10 epochs)
4. Mixed precision (AMP), class weights, early stopping (F1-macro)
5. Save checkpoint → Drive
6. **Outputs:** `models/best_{pants,tops}.pth`, training curves

### Phase 4: Inference (Local or Colab, <1s per image)
**Notebook:** `06_full_pipeline_demo.ipynb` or CLI

```bash
python -m src.pipeline --config config/pipeline_config.yaml --image photo.jpg
```

---

## 6. Key Design Decisions

### 1. Separate Models (Pants vs Tops) over Unified
- **Rationale:** Atributos exclusivos (`waist_rise` ≠ `neck_style`), dataset asimétrico (2:1)
- **Trade-off:** +complexity vs. +accuracy (each learns its domain tighter)

### 2. Routing via SegFormer (no binary classifier)
- **Rationale:** SegFormer ya distingue `upper_body` vs `lower_body`; reusar signal
- **Trade-off:** Fails gracefully on ambiguous cases (full_body → fallback a tops)

### 3. Multi-task Learning (6 heads on shared backbone)
- **Rationale:** Atributos correlacionados (color+pattern+fabric frecuentemente juntos)
- **Trade-off:** Shared head puede ser bottleneck vs. individual heads más especializados

### 4. Class Weights & `ignore_index=-1`
- **Rationale:** Long-tail imbalance + missing data manejos por atributo
- **Trade-off:** Loss computation más lenta vs. Correcta ponderación

### 5. Colab-First Design (no GPU local)
- **Rationale:** Acceso T4 gratis + reproducibilidad en cloud
- **Trade-off:** Offline development más lento (debug en Colab)

### 6. Cache Idempotente (MD5 filenames)
- **Rationale:** Retryable downloads, resume sin perder progreso
- **Trade-off:** Harder to correlate failures back to source URL

---

## 7. Evaluation Metrics

### Per-Attribute
- **Accuracy**: (TP+TN) / total
- **F1 (macro)**: mean of per-class F1
- **Confusion Matrix**: detect class confusion patterns
- **Class-weighted Accuracy**: per-attribute weight (dari config)

### System-Level
- **End-to-end latency**: segmentation + color + classification per image
- **Failure modes**: missing attributes, low confidence (<0.5)
- **Downstream impact**: TBD (system de recomendación)

### Baselines
- **Null**: assign most frequent class
- **Simple CNN (ResNet-18)**: no ViT overhead
- **Single unified model**: all 7 attributes in one head

---

## 8. Project Structure

```
fashion-feature-extraction/
│
├── config/
│   └── pipeline_config.yaml           # Single source of truth (device, batch_size, class lists, etc.)
│
├── data/
│   ├── preprocessed/
│   │   ├── pants_1.csv               # 41.868 pants (Abercrombie)
│   │   └── tops_1.csv                # 78.686 tops (Macy's)
│   ├── images/
│   │   ├── pants/                    # ~15k cached PNG/JPG
│   │   └── tops/                     # ~15k cached PNG/JPG
│   └── splits/
│       ├── pants_train.csv
│       ├── pants_val.csv
│       ├── pants_test.csv
│       ├── tops_train.csv
│       ├── tops_val.csv
│       └── tops_test.csv
│
├── models/
│   ├── best_pants.pth               # SegFormer (pre-trained) + ViT pants (fine-tuned)
│   └── best_tops.pth                # ViT tops (fine-tuned)
│
├── src/
│   ├── data/
│   │   ├── csv_dataset.py           # PyTorch Dataset (CSV + image loading + EN→ES mapping)
│   │   ├── downloader.py            # Parallel image downloader + cache
│   │   ├── mappings.py              # EN→ES translation, long-tail grouping
│   │   ├── splits.py                # Stratified splits generator
│   │   └── colab.py                 # Mount Drive, rewrite paths, setup
│   │
│   ├── segmentation/
│   │   ├── segmenter.py             # SegFormer inference
│   │   └── postprocess.py           # Mask cleanup, crop-to-content
│   │
│   ├── color/
│   │   ├── extractor.py             # K-Means LAB + color naming
│   │   └── names.py                 # CSS color name mapping
│   │
│   ├── classification/
│   │   ├── model.py                 # MultiTaskFashionClassifier (ViT + heads)
│   │   └── train.py                 # Training loop (two-phase, AMP, early stopping)
│   │
│   └── pipeline.py                  # Orquestador (load models → process image → JSON)
│
├── scripts/
│   ├── download_images.py           # CLI para descargar batch desde CSV
│   ├── prepare_splits.py            # CLI para generar splits
│   ├── train_pants.py               # Entry point training pants
│   ├── train_tops.py                # Entry point training tops
│   └── evaluate.py                  # Confusion matrices, F1 vs ground truth
│
├── notebooks/
│   ├── 01_eda_csv_datasets.ipynb    # EDA profundo (local)
│   ├── 02_prepare_data_colab.ipynb  # Data prep (Colab)
│   ├── 03_train_pants_colab.ipynb   # Training pants (Colab T4)
│   ├── 04_train_tops_colab.ipynb    # Training tops (Colab T4)
│   └── 06_full_pipeline_demo.ipynb  # Inference demo (Colab or local)
│
├── tests/
│   ├── test_csv_dataset.py
│   ├── test_downloader.py
│   ├── test_mappings.py
│   ├── test_splits.py
│   ├── test_color.py
│   ├── test_segmentation.py
│   └── conftest.py
│
├── requirements.txt                 # Dependencies
├── setup.py                         # pip install -e .
├── README.md                        # Workflow documentation
└── PROJECT_BRIEF.md                 # This document
```

---

## 9. Deliverables & Checkpoints

| Phase | Deliverable | Status | Checkpoint |
|-------|-------------|--------|-----------|
| EDA | `01_eda_csv_datasets.ipynb` + class_weights + EN→ES mapping | ✓ | Cramér's V heatmap, entropy analysis |
| Config | `pipeline_config.yaml` with pants/tops sections | ✓ | YAML validates, no syntax errors |
| Data Layer | `CSVImageDataset`, `ImageDownloader`, `mappings.py` | ✓ | 6 tests passing (no GPU/net) |
| Segmentation | SegFormer integration + postprocessing | ✓ | Smoke test on sample image |
| Color | K-Means + color naming | ✓ | Palette extraction verified |
| Models | `MultiTaskFashionClassifier` architecture | ✓ | 87.6M params, forward pass OK |
| Training (Pants) | `03_train_pants_colab.ipynb`, checkpoint saved | 🟡 | Trained in Colab (user-run) |
| Training (Tops) | `04_train_tops_colab.ipynb`, checkpoint saved | 🟡 | Trained in Colab (user-run) |
| Pipeline | End-to-end inference, JSON output | ✓ | smoke test: config loads, image processes |
| Evaluation | `evaluate.py`, confusion matrices | ✓ | Script ready, CSV format |
| Tests | 100% coverage of data layer | ✓ | 6+ tests, no GPU/net required |
| Documentation | README + this brief | ✓ | Architecture diagram, workflow |

---

## 10. Known Limitations & Future Work

### Limitations
- **No brand classification**: POC scope (Logo-2K+ unsuitable for real prendas)
- **Single backbone**: ViT-base only; no EfficientNet/ResNet experiments yet
- **No semi-supervised**: all training is supervised
- **Limited augmentation**: basic transforms (flip, color jitter, crop, rotate)
- **No edge cases**: handles "missing" as -1, but no outlier detection

### Future Work
1. **Hyperparameter sweep**: Optuna + WandB for LR, batch size, warmup
2. **Ensemble methods**: vote between pants/tops classifiers
3. **Few-shot learning**: support new clothing types with <100 examples
4. **Mobile deployment**: ONNX export, TensorFlow Lite quantization
5. **Brand detection**: if dataset with bbox labels becomes available
6. **Bias analysis**: intersection of color × style × fit across demographics
7. **A/B testing**: offline eval vs. online recommendation metrics

---

## 11. Running Locally vs Colab

### Local (Development)
```bash
# EDA only (no downloads)
jupyter notebook notebooks/01_eda_csv_datasets.ipynb

# Smoke test (tiny batch)
python scripts/download_images.py --csv data/preprocessed/pants_1.csv \
  --output data/images/pants --sample 50

python scripts/prepare_splits.py --csv data/preprocessed/pants_1.csv \
  --stratify color_family --output data/splits

# Pipeline inference (if checkpoints exist)
python -m src.pipeline --image sample.jpg --config config/pipeline_config.yaml
```

### Colab (Production)
```
1. Upload to Drive: CSVs + code
2. Run notebooks sequentially: 02 → 03 → 04 → 06
3. Download checkpoints from Drive
4. (Optional) Run locally with --config pointing to Drive paths
```

---

## 12. Contact & Changelog

**Project**: Master's Thesis Work (Visión Computarizada, UDESA)
**Author**: Damian Ilkow
**Last Updated**: 2026-04-25
**Status**: POC Complete (pants/tops classification, no brand)

### Key Changes (v0.2.0)
- Removed brand classification module (Logo-2K+ unsuitable)
- Refactored to CSV-based data (pants_1.csv + tops_1.csv)
- Colab-first workflow (Drive mounting, path rewriting)
- Two-phase training (frozen → unfrozen backbone)
- Multi-task learning with per-attribute class weights

---

## Appendix A: Config Example

```yaml
device: "cuda"  # auto-fallback to cpu
seed: 42
image_size: [224, 224]

data:
  pants_csv: "data/preprocessed/pants_1.csv"
  tops_csv: "data/preprocessed/tops_1.csv"
  download:
    workers: 8
    timeout: 15
    max_retries: 3

pants:
  backbone: "google/vit-base-patch16-224"
  checkpoint: "models/best_pants.pth"
  heads:
    color_family:
      classes: ["neutrals_dark", "neutrals_light", "blues", "reds", ..., "otro"]
      weight: 1.0
    pattern:
      classes: ["liso", "rayas", "cuadros", "estampado", ..., "otro"]
      weight: 1.0
    # ... (fit_silhouette, fabric_content, dressing_style, waist_rise)
  training:
    epochs: 15
    batch_size: 32
    learning_rate: 3.0e-4
    freeze_backbone_epochs: 3
    use_class_weights: true

# Similar structure for `tops` (with neck_style instead of waist_rise)

pipeline:
  run_segmentation: true
  run_color: true
  run_classification: true
```

---

## Appendix B: JSON Output Example

```json
{
  "image_path": "data/sample.jpg",
  "pipeline_version": "0.2.0",
  "segmentation": {
    "dominant_category": "upper_body",
    "mask_area_px": 45230
  },
  "color": {
    "dominant": {
      "hex": "#1a1a1a",
      "name": "charcoal"
    },
    "palette": ["#1a1a1a", "#4a4a4a", "#8a8a8a", ...],
    "pattern": {
      "label": "liso",
      "confidence": 0.94
    }
  },
  "garment_type": "tops",
  "attributes": {
    "color_family": {
      "label": "neutrals_dark",
      "confidence": 0.91
    },
    "pattern": {
      "label": "liso",
      "confidence": 0.88
    },
    "fit_silhouette": {
      "label": "regular",
      "confidence": 0.73
    },
    "fabric_content": {
      "label": "algodon",
      "confidence": 0.82
    },
    "dressing_syle": {
      "label": "casual",
      "confidence": 0.79
    },
    "neck_style": {
      "label": "redondo",
      "confidence": 0.85
    }
  },
  "processing_time_ms": 1240
}
```
