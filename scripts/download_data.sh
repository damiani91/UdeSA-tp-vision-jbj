#!/bin/bash
# Fashion Feature Extraction - Script de descarga de datos.
#
# Uso:
#   bash scripts/download_data.sh modanet
#   bash scripts/download_data.sh deepfashion
#   bash scripts/download_data.sh logo2kplus
#   bash scripts/download_data.sh demo         # sample pequeno para dev
#
# NOTA: DeepFashion y Logo-2K+ requieren acceso academico. Este script
# crea la estructura de directorios y documenta los URLs. Para descarga
# automatica se requieren credenciales propias.

set -e

DATASET="${1:-demo}"
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

case "$DATASET" in
  modanet)
    echo "=== ModaNet ==="
    echo "ModaNet usa imagenes de Paperdoll + anotaciones de eBay."
    echo "URLs de referencia:"
    echo "  - Anotaciones: https://github.com/eBay/modanet"
    echo "  - Imagenes: se obtienen via script de Paperdoll"
    mkdir -p "$DATA_DIR/modanet/images" "$DATA_DIR/modanet/annotations"
    echo "Estructura creada en $DATA_DIR/modanet/"
    ;;

  deepfashion)
    echo "=== DeepFashion ==="
    echo "DeepFashion Category & Attribute Prediction Benchmark."
    echo "URL: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html"
    echo "Requiere aceptar licencia academica."
    mkdir -p "$DATA_DIR/deepfashion/images" "$DATA_DIR/deepfashion/annotations"
    ;;

  logo2kplus)
    echo "=== Logo-2K+ ==="
    echo "URL: https://github.com/msn199959/Logo-2k-plus-Dataset"
    mkdir -p "$DATA_DIR/logo2kplus"
    ;;

  demo)
    echo "=== Demo dataset ==="
    echo "Descargando muestra pequena de imagenes publicas..."
    mkdir -p "$DATA_DIR/demo"
    # Descarga de unas pocas imagenes publicas de HuggingFace
    if command -v huggingface-cli >/dev/null 2>&1; then
      echo "Podes usar: huggingface-cli download <dataset> para obtener imagenes."
    else
      echo "Instala huggingface_hub: pip install huggingface_hub"
    fi
    echo "Alternativamente, copia manualmente imagenes JPG/PNG a $DATA_DIR/demo/"
    ;;

  *)
    echo "Dataset desconocido: $DATASET"
    echo "Opciones: modanet | deepfashion | logo2kplus | demo"
    exit 1
    ;;
esac

echo ""
echo "Listo. Directorios creados bajo $DATA_DIR/"
