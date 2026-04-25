# Fashion Feature Extraction Pipeline

Pipeline de extracción de atributos visuales de prendas de moda. Recibe una imagen y produce un JSON estructurado con tipo, color, estilo, fit, material y otros atributos. Pensado para alimentar un sistema de recomendación downstream.

## Arquitectura

```
Imagen
   |
   v
[1. Segmentación]  SegFormer -> máscara + categoría dominante
   |
   +-------+--------+
   v                v
[2. Color]   [3a. Pants] / [3b. Tops]
 K-Means      ViT multi-task (routing por categoría)
 LAB
   |          |
   +----------+
              v
        JSON estructurado
```

El **routing entre `pants` y `tops`** se decide en base a la categoría dominante (upper_body / lower_body) detectada por SegFormer.

## Datasets

- **`data/preprocessed/pants_1.csv`** — 41.868 pantalones de Abercrombie con URL + 7 atributos
- **`data/preprocessed/tops_1.csv`** — 78.686 prendas superiores de Macy's con URL + 7 atributos

Atributos comunes: `color_family`, `pattern`, `fit_silhouette`, `fabric_content`, `dressing_syle`. Exclusivos: `waist_rise` (pants), `neck_style` (tops). Los CSV traen los labels en inglés; el módulo `src/data/mappings.py` los traduce a la ontología en español del config.

## Estructura

```
fashion-feature-extraction/
├── config/pipeline_config.yaml         # Configuración centralizada
├── data/                               # CSVs, imágenes, splits
├── models/                             # Checkpoints
├── src/
│   ├── data/                           # CSV dataset, downloader, mappings, splits, colab helpers
│   ├── segmentation/                   # SegFormer
│   ├── color/                          # K-Means en LAB + naming
│   ├── classification/                 # ViT multi-task (pants y tops)
│   └── pipeline.py                     # Orquestador con auto-routing
├── scripts/
│   ├── download_images.py              # Descarga paralela desde CSV
│   ├── prepare_splits.py               # Splits train/val/test estratificados
│   ├── train_pants.py / train_tops.py  # Entrenamiento clasificadores
│   └── evaluate.py                     # Evaluación contra ground truth
├── tests/                              # pytest sin GPU ni red
└── notebooks/
    ├── 01_eda_csv_datasets.ipynb       # EDA profundo
    ├── 02_prepare_data_colab.ipynb     # Drive + descarga + splits
    ├── 03_train_pants_colab.ipynb      # Training pants
    ├── 04_train_tops_colab.ipynb       # Training tops
    └── 06_full_pipeline_demo.ipynb     # Demo end-to-end
```

## Instalación local

```bash
cd fashion-feature-extraction
pip install -r requirements.txt
pip install -e .
```

## Flujo de trabajo

### 1. EDA (local)

```bash
jupyter notebook notebooks/01_eda_csv_datasets.ipynb
```

Genera class weights, splits sugeridos, mapeos EN→ES y detecta long-tail.

### 2. Preparación de datos (Colab)

Subí a Drive: `data/preprocessed/{pants_1.csv, tops_1.csv}` y el código del repo.

Ejecutá `notebooks/02_prepare_data_colab.ipynb`. Descarga ~30k imágenes en paralelo (cache idempotente con hash MD5 de URLs) y arma splits estratificados.

### 3. Entrenamiento (Colab, T4 ~12h gratuitas)

Ejecutá los dos notebooks en orden:
- `03_train_pants_colab.ipynb` — clasificador ViT multi-task para pantalones (6 heads)
- `04_train_tops_colab.ipynb` — idem para prendas superiores (6 heads)

Cada training:
- Dos fases: backbone congelado → descongelado con learning rates discriminativos
- AMP en GPU + class weights inverse-frequency
- Early stopping sobre F1-macro
- Checkpoints a Drive

### 4. Demo end-to-end

```bash
# Local con modelos descargados de Drive
python -m src.pipeline --config config/pipeline_config.yaml --image foto.jpg
```

O en Colab: `notebooks/06_full_pipeline_demo.ipynb`.

### 5. Evaluación

```bash
python scripts/evaluate.py \
    --predictions outputs/results.json \
    --csv data/preprocessed/pants_1.csv \
    --dataset pants
```

Compara predicciones contra ground truth (con mapeo EN→ES aplicado).

## Tests

```bash
pytest tests/ -v
```

Los tests no requieren GPU ni red (usan fixtures dummy). Cubren mappings, downloader, csv_dataset, splits, color y segmentación post-process.

## Configuración Colab

`src/data/colab.py` provee `setup_colab(config_path)` que:
1. Detecta entorno Colab y monta Drive
2. Reescribe los paths del config para apuntar a `/content/drive/MyDrive/master_ia/fashion-extraction/`
3. Crea las carpetas necesarias

Estructura sugerida en Drive:

```
/content/drive/MyDrive/master_ia/fashion-extraction/
├── code/                    # Copia del repo (o clonalo desde GitHub)
├── data/preprocessed/       # CSVs
├── data/images/{pants,tops}/  # Cache de descargas
├── data/splits/             # CSVs train/val/test
├── models/                  # Checkpoints
└── outputs/                 # Predicciones + métricas
```

## Decisiones de diseño

- **Dos modelos separados** (pants vs tops): cada uno con un head exclusivo (`waist_rise` / `neck_style`) y heads compartidas. Más simple que un modelo unificado y aprovecha la asimetría del dataset.
- **Routing por segmentación**: SegFormer ya distingue upper_body vs lower_body, así que reusamos esa señal en lugar de un clasificador binario adicional.
- **Marca fuera del scope del POC**: el dataset disponible (Logo-2K+) tiene logos aislados, no logos sobre prendas, y sin bounding boxes. Clasificar marca confiable sobre prendas reales requiere otro pipeline; queda como follow-up si aparecen datos con bboxes.
- **Cache idempotente**: descargas con `md5(url).jpg` como filename. Permite reanudar sin perder progreso.
- **`ignore_index=-1` en la loss**: las imágenes con labels faltantes o no mapeables no contribuyen al gradient de esa head, pero sí a las otras.
