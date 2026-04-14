"""Diccionario de nombres de color en espanol y utilidades de matching.

Los colores estan definidos en sRGB y se convierten a LAB al cargar el modulo.
El matching usa Delta E CIE2000 para maxima precision perceptual.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Diccionario de referencia: nombre -> RGB (0-255)
# Organizado por familias cromaticas.
COLOR_REFERENCES_RGB: dict[str, tuple[int, int, int]] = {
    # Blancos y cremas
    "blanco": (255, 255, 255),
    "blanco roto": (245, 245, 235),
    "crema": (255, 253, 208),
    "marfil": (255, 255, 240),
    "hueso": (242, 231, 210),
    # Negros y grises
    "negro": (0, 0, 0),
    "gris oscuro": (64, 64, 64),
    "gris": (128, 128, 128),
    "gris claro": (192, 192, 192),
    "gris perla": (220, 220, 220),
    # Rojos
    "rojo": (220, 20, 20),
    "rojo oscuro": (139, 0, 0),
    "borgona": (128, 0, 32),
    "bordo": (108, 20, 38),
    "carmesi": (220, 20, 60),
    "coral": (255, 127, 80),
    "terracota": (204, 78, 92),
    "salmon": (250, 128, 114),
    # Naranjas
    "naranja": (255, 140, 0),
    "naranja claro": (255, 180, 90),
    "durazno": (255, 203, 164),
    "mandarina": (242, 133, 0),
    # Amarillos
    "amarillo": (255, 215, 0),
    "amarillo claro": (255, 239, 150),
    "mostaza": (204, 153, 0),
    "ocre": (204, 119, 34),
    # Verdes
    "verde": (34, 139, 34),
    "verde oscuro": (0, 100, 0),
    "verde oliva": (128, 128, 0),
    "verde militar": (78, 83, 42),
    "verde menta": (152, 255, 152),
    "verde agua": (127, 222, 201),
    "verde lima": (191, 255, 0),
    "verde esmeralda": (80, 200, 120),
    # Azules
    "azul": (30, 80, 200),
    "azul oscuro": (0, 0, 139),
    "azul marino": (0, 0, 80),
    "azul claro": (135, 206, 235),
    "celeste": (135, 206, 250),
    "turquesa": (64, 224, 208),
    "aguamarina": (127, 255, 212),
    "denim": (70, 100, 150),
    # Violetas y purpuras
    "violeta": (138, 43, 226),
    "morado": (128, 0, 128),
    "lavanda": (230, 230, 250),
    "lila": (200, 162, 200),
    # Rosas
    "rosa": (255, 105, 180),
    "rosa claro": (255, 182, 193),
    "rosa palido": (250, 218, 221),
    "fucsia": (255, 0, 255),
    "magenta": (220, 50, 150),
    # Marrones
    "marron": (139, 69, 19),
    "marron oscuro": (92, 51, 23),
    "marron claro": (181, 101, 29),
    "camel": (193, 154, 107),
    "caramelo": (196, 128, 66),
    "chocolate": (78, 52, 46),
    "cafe": (111, 78, 55),
    # Beiges y tierras
    "beige": (245, 222, 179),
    "beige oscuro": (210, 180, 140),
    "arena": (194, 178, 128),
    "kaki": (195, 176, 145),
    "taupe": (139, 125, 107),
    "nude": (238, 213, 183),
    # Metalicos (aproximados)
    "dorado": (255, 215, 0),
    "plateado": (192, 192, 192),
    "cobre": (184, 115, 51),
}


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    """Convierte un triplete sRGB (0-255) a LAB usando skimage.

    Args:
        rgb: Tupla (R, G, B) con valores 0-255.

    Returns:
        Array (3,) con valores LAB.
    """
    from skimage.color import rgb2lab

    arr = np.array(rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)
    return lab[0, 0]


# Conversion precomputada al importar el modulo.
COLOR_REFERENCES_LAB: dict[str, np.ndarray] = {
    name: _rgb_to_lab(rgb) for name, rgb in COLOR_REFERENCES_RGB.items()
}


def _delta_e_cie2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calcula Delta E CIE2000 entre dos colores LAB.

    Usa colormath si esta disponible, si no fallback a Euclidean en LAB.

    Args:
        lab1: Array (3,) con valores L, a, b.
        lab2: Array (3,) con valores L, a, b.

    Returns:
        Distancia perceptual.
    """
    try:
        from colormath.color_diff import delta_e_cie2000
        from colormath.color_objects import LabColor

        c1 = LabColor(lab_l=lab1[0], lab_a=lab1[1], lab_b=lab1[2])
        c2 = LabColor(lab_l=lab2[0], lab_a=lab2[1], lab_b=lab2[2])
        return float(delta_e_cie2000(c1, c2))
    except Exception as exc:
        logger.debug("Fallback a distancia euclidiana en LAB: %s", exc)
        return float(np.linalg.norm(lab1 - lab2))


def find_nearest_color_name(
    lab_value: np.ndarray, references: Optional[dict[str, np.ndarray]] = None
) -> tuple[str, float]:
    """Encuentra el nombre de color perceptualmente mas cercano.

    Args:
        lab_value: Array (3,) con valores LAB del color a identificar.
        references: Diccionario opcional de referencias (default: COLOR_REFERENCES_LAB).

    Returns:
        Tupla (nombre_color, delta_e).
    """
    refs = references if references is not None else COLOR_REFERENCES_LAB
    best_name = None
    best_delta = float("inf")
    for name, ref_lab in refs.items():
        delta = _delta_e_cie2000(lab_value, ref_lab)
        if delta < best_delta:
            best_delta = delta
            best_name = name

    return best_name or "desconocido", best_delta


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convierte RGB (0-255) a string hexadecimal.

    Args:
        rgb: Tupla (R, G, B).

    Returns:
        String hexadecimal en mayusculas, e.g. '#FF0000'.
    """
    r, g, b = (int(max(0, min(255, c))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"
