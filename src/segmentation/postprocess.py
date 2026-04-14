"""Post-procesamiento de mascaras de segmentacion."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Aplica una mascara binaria a la imagen produciendo un PNG con transparencia.

    Args:
        image: Imagen PIL en modo RGB.
        mask: Mascara binaria (H, W) con valores 0 o 1.

    Returns:
        PIL.Image en modo RGBA donde el canal alpha viene de la mascara.
    """
    image_rgba = image.convert("RGBA")
    arr = np.array(image_rgba)

    # Normalizar mascara a 0-255 uint8
    if mask.dtype != np.uint8:
        mask_uint8 = (mask.astype(bool).astype(np.uint8)) * 255
    else:
        mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Resize si es necesario
    if mask_uint8.shape[:2] != arr.shape[:2]:
        mask_uint8 = cv2.resize(
            mask_uint8, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    arr[:, :, 3] = mask_uint8
    return Image.fromarray(arr, mode="RGBA")


def crop_to_content(image: Image.Image, padding: int = 10) -> Image.Image:
    """Recorta la imagen al bounding box del contenido no-transparente.

    Args:
        image: Imagen PIL en modo RGBA.
        padding: Pixels de padding alrededor del bounding box.

    Returns:
        PIL.Image recortada.
    """
    if image.mode != "RGBA":
        logger.warning("crop_to_content requiere RGBA; convirtiendo automaticamente.")
        image = image.convert("RGBA")

    arr = np.array(image)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        logger.warning("No hay contenido no-transparente, retornando imagen original.")
        return image

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(arr.shape[1], int(xs.max()) + padding)
    y2 = min(arr.shape[0], int(ys.max()) + padding)

    return image.crop((x1, y1, x2, y2))


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Limpia la mascara con operaciones morfologicas.

    Args:
        mask: Mascara binaria (H, W).
        kernel_size: Tamano del kernel para operaciones morfologicas.

    Returns:
        Mascara binaria limpia (H, W) como uint8 (0 o 1).
    """
    mask_uint8 = (mask.astype(bool).astype(np.uint8)) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Opening elimina ruido pequeno, closing rellena huecos
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return (closed > 0).astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Retiene solo el componente conectado mas grande.

    Args:
        mask: Mascara binaria (H, W).

    Returns:
        Mascara con solo el componente mas grande (uint8, 0 o 1).
    """
    mask_uint8 = (mask.astype(bool).astype(np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        return mask_uint8

    # Excluir el fondo (label 0), tomar el componente con mayor area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    return (labels == largest_label).astype(np.uint8)
