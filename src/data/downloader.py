"""Descarga paralela de imagenes desde URLs con cache en disco."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def url_to_filename(url: str, ext: str = ".jpg") -> str:
    """Genera nombre de archivo deterministico a partir de la URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest() + ext


class ImageDownloader:
    """Descarga imagenes en paralelo con cache, retry y validacion.

    El cache es un directorio donde cada imagen se guarda con nombre
    `md5(url).jpg`. Si el archivo ya existe y se puede abrir con PIL,
    se considera valido y se saltea.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        workers: int = 8,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        """Inicializa el downloader.

        Args:
            cache_dir: Directorio donde guardar las imagenes.
            workers: Numero de threads paralelos.
            timeout: Timeout por request en segundos.
            max_retries: Reintentos por URL fallida.
            backoff_factor: Multiplicador de espera entre reintentos.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def cache_path(self, url: str) -> Path:
        """Path en cache para una URL."""
        return self.cache_dir / url_to_filename(url)

    def is_cached(self, url: str) -> bool:
        """True si la imagen ya esta en cache y se puede abrir."""
        path = self.cache_path(url)
        if not path.exists():
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            path.unlink(missing_ok=True)
            return False

    def _download_one(self, url: str) -> tuple[str, bool, Optional[str]]:
        """Descarga una imagen. Retorna (url, exito, error_msg)."""
        if self.is_cached(url):
            return url, True, None

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(self.cache_path(url), "JPEG", quality=88)
                return url, True, None
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                wait = self.backoff_factor ** attempt
                time.sleep(wait)
        return url, False, last_err

    def download_many(
        self, urls: Iterable[str], desc: str = "Descargando"
    ) -> pd.DataFrame:
        """Descarga un iterable de URLs en paralelo.

        Args:
            urls: Iterable de URLs.
            desc: Descripcion para la barra de progreso.

        Returns:
            DataFrame con columnas [url, success, error, cache_path].
        """
        urls = list(urls)
        results = []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self._download_one, u): u for u in urls}
            with tqdm(total=len(urls), desc=desc) as pbar:
                for fut in as_completed(futures):
                    url, ok, err = fut.result()
                    results.append({
                        "url": url,
                        "success": ok,
                        "error": err,
                        "cache_path": str(self.cache_path(url)) if ok else None,
                    })
                    pbar.update(1)
        df = pd.DataFrame(results)
        n_ok = int(df["success"].sum())
        logger.info("Descarga: %d/%d OK (%.1f%%)", n_ok, len(df), 100 * n_ok / max(1, len(df)))
        return df


def download_csv_images(
    csv_path: str | Path,
    cache_dir: str | Path,
    url_col: str = "image_url",
    sample: Optional[int] = None,
    workers: int = 8,
    timeout: int = 15,
    seed: int = 42,
    log_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Descarga las imagenes referenciadas por una columna de URLs en un CSV.

    Args:
        csv_path: Path al CSV.
        cache_dir: Directorio destino para imagenes.
        url_col: Nombre de la columna con URLs.
        sample: Si se provee, hace `df.sample(n=sample, random_state=seed)`.
        workers: Threads paralelos.
        timeout: Timeout en segundos.
        seed: Semilla para sampling.
        log_path: Si se provee, guarda el log de descarga como CSV ahi.

    Returns:
        DataFrame con resultados (url, success, error, cache_path).
    """
    df = pd.read_csv(csv_path)
    if url_col not in df.columns:
        raise ValueError(f"CSV no tiene columna '{url_col}'")
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    urls = df[url_col].dropna().unique().tolist()
    downloader = ImageDownloader(cache_dir, workers=workers, timeout=timeout)
    log = downloader.download_many(urls, desc=f"Descargando {Path(csv_path).stem}")

    if log_path is not None:
        if not str(log_path).startswith("gs://"):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
        logger.info("Log de descarga: %s", log_path)
    return log
