# Fashion Feature Extraction Pipeline

Pipeline de extraccion de atributos visuales de prendas de moda. Recibe una imagen de prenda y produce un JSON estructurado con tipo, color, estilo, fit, material, marca y otros atributos.

## Arquitectura

```
Imagen de prenda
       |
       v
[1. Segmentacion]  SegFormer -> mascara + PNG recortado
       |
  +---------+-----------+
  v         v           v
[2. Color] [3. Clasif.] [4. Marca]
 K-Means    ViT multi-   YOLO +
 en LAB     task heads   classifier
  +----+----+           |
       v                |
  [JSON estructurado] <-+
```

## Modulos

1. **Segmentacion** (`src/segmentation/`): SegFormer para aislar prenda del fondo
2. **Color** (`src/color/`): K-Means en espacio LAB + naming en espanol
3. **Clasificacion** (`src/classification/`): ViT multi-task (tipo, estilo, fit, cuello, manga, material)
4. **Marca** (`src/brand/`): YOLOv8 + EfficientNet-B0 para deteccion de logos

## Instalacion

```bash
cd fashion-feature-extraction
pip install -r requirements.txt
pip install -e .
```

## Uso rapido

```bash
# Procesar una imagen
python -m src.pipeline --config config/pipeline_config.yaml --image path/to/image.jpg

# Procesar un directorio
python -m src.pipeline --config config/pipeline_config.yaml --dir path/to/images/
```

## Tests

```bash
pytest tests/ -v
```

## Notebooks

| Notebook | Descripcion |
|----------|-------------|
| `01_eda_modanet.ipynb` | Analisis exploratorio de ModaNet |
| `02_segmentation_eval.ipynb` | Evaluacion del modelo de segmentacion |
| `03_color_extraction.ipynb` | Demo de extraccion de color |
| `04_classification_experiments.ipynb` | Experimentos de clasificacion multi-task |
| `05_full_pipeline_demo.ipynb` | Demo end-to-end del pipeline completo |
