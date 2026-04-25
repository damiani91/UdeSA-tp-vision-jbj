"""Helpers de I/O transparentes para paths locales y gs:// (GCS).

Uso:
    from src.data.io import open_any, save_torch_any, load_torch_any

    with open_any("gs://bucket/file.json", "r") as f:
        data = json.load(f)

    save_torch_any(state_dict, "gs://bucket/models/best.pth")
    state = load_torch_any("gs://bucket/models/best.pth")
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import IO


def _is_gcs(path: str | Path) -> bool:
    return str(path).startswith("gs://")


def open_any(path: str | Path, mode: str = "r") -> IO:
    """Abre un archivo local o en GCS. Retorna un file-like object.

    Para GCS requiere que gcsfs esté instalado.
    Para escritura en texto, usa mode='w'; para binario, 'wb' o 'rb'.
    """
    if _is_gcs(path):
        import fsspec
        return fsspec.open(str(path), mode).__enter__()
    return open(path, mode)


def save_torch_any(state: object, path: str | Path) -> None:
    """Guarda un state dict con torch.save en local o GCS."""
    import torch

    if _is_gcs(path):
        import fsspec
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        with fsspec.open(str(path), "wb") as f:
            f.write(buf.read())
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def load_torch_any(path: str | Path, map_location: str = "cpu") -> object:
    """Carga un checkpoint con torch.load desde local o GCS."""
    import torch

    if _is_gcs(path):
        import fsspec
        with fsspec.open(str(path), "rb") as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, map_location=map_location)
    return torch.load(path, map_location=map_location)
