"""Helpers para correr el pipeline en Google Colab.

Uso tipico al comienzo de un notebook Colab:

    >>> from src.data.colab import setup_colab
    >>> config = setup_colab("config/pipeline_config.yaml")

`setup_colab` detecta si esta en Colab, monta Drive si aplica, ajusta
los paths del config para apuntar a Drive y retorna el config ya parseado.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """True si estamos corriendo en Google Colab."""
    return "google.colab" in sys.modules


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Monta Google Drive en Colab. No-op fuera de Colab.

    Returns:
        True si se monto exitosamente, False si no estaba en Colab.
    """
    if not is_colab():
        logger.info("No estamos en Colab, skip mount.")
        return False
    from google.colab import drive  # type: ignore

    drive.mount(mount_point, force_remount=False)
    logger.info("Drive montado en %s", mount_point)
    return True


def _rewrite_paths(config: dict, drive_root: str | Path) -> dict:
    """Reescribe los paths del config para que apunten a Drive."""
    drive_root = Path(drive_root)
    if "paths" not in config:
        return config
    rewrites = {
        "raw_data": "data/raw",
        "preprocessed_data": "data/preprocessed",
        "images_pants": "data/images/pants",
        "images_tops": "data/images/tops",
        "splits": "data/splits",
        "models": "models",
        "outputs": "outputs",
    }
    for key, rel in rewrites.items():
        config["paths"][key] = str(drive_root / rel)

    if "data" in config:
        data = config["data"]
        for key in ["pants_csv", "tops_csv"]:
            if key in data:
                rel = Path(data[key])
                data[key] = str(drive_root / rel)

    for key in ["pants", "tops"]:
        if key in config and "checkpoint" in config[key]:
            rel = Path(config[key]["checkpoint"])
            config[key]["checkpoint"] = str(drive_root / rel)

    return config


def setup_colab(
    config_path: str | Path,
    drive_root: Optional[str] = None,
    mount: bool = True,
) -> dict:
    """Setup completo: monta Drive, ajusta paths y retorna el config.

    Args:
        config_path: Path al YAML de config (relativo al repo).
        drive_root: Override del root en Drive. Si None, usa `colab.drive_root`
            del config.
        mount: Si False, asume que Drive ya esta montado.

    Returns:
        Dict de config con paths ajustados.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not is_colab():
        logger.info("No estamos en Colab, retornando config sin modificar.")
        return config

    if mount:
        mount_drive()

    drive_root = drive_root or config.get("colab", {}).get(
        "drive_root", "/content/drive/MyDrive/master_ia/fashion-extraction"
    )

    config = _rewrite_paths(config, drive_root)

    for key in ["paths"]:
        if key in config:
            for sub in config[key].values():
                Path(sub).mkdir(parents=True, exist_ok=True)

    logger.info("Setup Colab completo. Drive root: %s", drive_root)
    return config


def ensure_repo_in_path(repo_dir: str | Path) -> None:
    """Agrega el repo al sys.path si no esta."""
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        logger.info("Agregado al PYTHONPATH: %s", repo_dir)


def setup_gcp(
    config_path: str | Path,
    bucket: str = "gs://jbj-vision",
    local_cache_root: str = "/content/cache",
    authenticate: bool = True,
) -> dict:
    """Setup para Colab Enterprise / Vertex AI con GCS como almacenamiento.

    Reescribe los paths del config para que apunten al bucket GCS, excepto
    los directorios de imagenes (que quedan locales para performance de training).

    Args:
        config_path: Path al YAML de config (relativo al repo).
        bucket: URI del bucket GCS, sin slash final.
        local_cache_root: Directorio local para cache de imagenes.
        authenticate: Si True, intenta autenticar via google.colab.auth.

    Returns:
        Dict de config con paths ajustados.
    """
    if authenticate:
        try:
            from google.colab import auth  # type: ignore
            auth.authenticate_user()
            logger.info("Autenticacion GCP exitosa.")
        except ImportError:
            logger.info("google.colab.auth no disponible, asumiendo auth via service account.")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    bucket = bucket.rstrip("/")

    if "paths" in config:
        gcs_paths = {
            "raw_data": f"{bucket}/data/raw",
            "preprocessed_data": f"{bucket}/data/preprocessed",
            "splits": f"{bucket}/data/splits",
            "models": f"{bucket}/models",
            "outputs": f"{bucket}/outputs",
        }
        local_paths = {
            "images_pants": f"{local_cache_root}/images/pants",
            "images_tops": f"{local_cache_root}/images/tops",
        }
        config["paths"].update(gcs_paths)
        config["paths"].update(local_paths)

    if "data" in config:
        stem = config["data"].get("pants_csv", "data/preprocessed/pants_1.csv")
        config["data"]["pants_csv"] = f"{bucket}/data/preprocessed/{Path(stem).name}"
        stem = config["data"].get("tops_csv", "data/preprocessed/tops_1.csv")
        config["data"]["tops_csv"] = f"{bucket}/data/preprocessed/{Path(stem).name}"

    for key in ["pants", "tops"]:
        if key in config:
            config[key]["checkpoint"] = f"{bucket}/models/best_{key}.pth"

    for local_path in local_paths.values():
        Path(local_path).mkdir(parents=True, exist_ok=True)

    logger.info("Setup GCP completo. Bucket: %s, cache local: %s", bucket, local_cache_root)
    return config


def setup_local(
    config_path: str | Path = "config/pipeline_config.yaml",
    repo_root: str | Path | None = None,
) -> dict:
    """Setup para ejecución local sin Colab ni GCS.

    Lee el YAML de config y reescribe todos los paths para que apunten
    al filesystem local usando ``repo_root`` como base.

    Args:
        config_path: Path al YAML de config (relativo al repo_root).
        repo_root: Raiz del repositorio. Si None, se detecta
            automáticamente subiendo desde este archivo.

    Returns:
        Dict de config con paths locales absolutos.
    """
    if repo_root is None:
        # src/data/colab.py → subimos 3 niveles para llegar a la raíz del repo
        repo_root = Path(__file__).resolve().parent.parent.parent

    repo_root = Path(repo_root).resolve()
    config_full = repo_root / config_path

    with open(config_full) as f:
        config = yaml.safe_load(f)

    # Reescribir paths como absolutos locales
    if "paths" in config:
        rewrites = {
            "raw_data": "data/raw",
            "preprocessed_data": "data/preprocessed",
            "images_pants": "data/images/pants",
            "images_tops": "data/images/tops",
            "splits": "data/splits",
            "models": "models",
            "outputs": "outputs",
        }
        for key, rel in rewrites.items():
            config["paths"][key] = str(repo_root / rel)

    # Reescribir CSV paths
    if "data" in config:
        for key in ["pants_csv", "tops_csv"]:
            if key in config["data"]:
                config["data"][key] = str(repo_root / config["data"][key])

    # Reescribir checkpoint paths
    for key in ["pants", "tops"]:
        if key in config and "checkpoint" in config[key]:
            config[key]["checkpoint"] = str(repo_root / config[key]["checkpoint"])

    # Crear directorios
    if "paths" in config:
        for sub in config["paths"].values():
            Path(sub).mkdir(parents=True, exist_ok=True)

    logger.info("Setup local completo. Repo root: %s", repo_root)
    return config
