"""Mapeos EN-ES y agrupado de clases long-tail.

Los diccionarios `MAPPING_EN_ES` traducen los labels en ingles de los CSV
(Abercrombie/Macy's) a la ontologia en espanol definida en
`config/pipeline_config.yaml`. Las clases con representacion menor a
`LONG_TAIL_THRESHOLD` se agrupan en "otro".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

LONG_TAIL_THRESHOLD: float = 0.01
DEFAULT_OTHER: str = "otro"

MAPPING_EN_ES: dict[str, dict[str, str]] = {
    "color_family": {
        "Neutrals-Dark": "neutrals_dark",
        "Neutrals-Light": "neutrals_light",
        "Blues": "blues",
        "Reds": "reds",
        "Pinks": "reds",
        "Greens": "greens",
        "Yellows": "yellows",
        "Purples": "purples",
        "Oranges": "oranges",
        "Browns": "browns",
        "Metallics": "metallics",
    },
    "pattern": {
        "Non-Pattern/Solid": "liso",
        "No-Print/No-Pattern": "liso",
        "Solid": "liso",
        "Stripes": "rayas",
        "Striped": "rayas",
        "Pinstripe": "rayas",
        "Check": "cuadros",
        "Checkered": "cuadros",
        "Plaid": "cuadros",
        "Gingham": "cuadros",
        "Printed": "estampado",
        "Print": "estampado",
        "Floral": "floral",
        "Geometric": "geometrico",
        "Abstract": "estampado",
        "Graphic": "estampado",
    },
    "fit_silhouette": {
        "Slim-Fit": "slim",
        "Slim-Fit-Pant": "slim",
        "Skinny-Fit": "skinny",
        "Skinny-Fit-Pant": "skinny",
        "Regular-Fit": "regular",
        "Regular-Fit-Pant": "regular",
        "Straight-Fit": "regular",
        "Relaxed-Fit": "relaxed",
        "Relaxed-Fit-Pant": "relaxed",
        "Loose-Fit": "relaxed",
        "Wide-Leg": "wide",
        "Wide-Leg-Pant": "wide",
        "Bootcut": "wide",
        "Flare": "wide",
        "Oversized": "oversized",
        "Fitted": "fitted",
        "Cropped": "fitted",
    },
    "fabric_content": {
        "Polyester": "sintetico",
        "Nylon": "sintetico",
        "Spandex": "sintetico",
        "Elastane": "sintetico",
        "Rayon": "sintetico",
        "Acrylic": "sintetico",
        "Cotton": "algodon",
        "Modal": "algodon",
        "Tencel": "algodon",
        "Jersey-Knit": "punto",
        "Knit": "punto",
        "Cashmere": "punto",
        "Linen": "lino",
        "Silk": "seda",
        "Wool": "lana",
        "Leather": "cuero",
        "Suede": "cuero",
        "Denim": "denim",
        "Corduroy": "algodon",
    },
    "dressing_syle": {
        "Minimalist": "minimalista",
        "Classic": "clasico",
        "Athleisure": "sport",
        "Athletic": "sport",
        "Casual": "casual",
        "Formal": "formal",
        "Streetwear": "streetwear",
        "Urban": "streetwear",
        "Bohemian": "bohemio",
        "Boho": "bohemio",
        "Romantic": "romantico",
        "Preppy": "clasico",
    },
    "waist_rise": {
        "Mid-Rise": "mid_rise",
        "High-Rise": "high_rise",
        "Low-Rise": "low_rise",
        "Ultra-High-Rise": "high_rise",
    },
    "neck_style": {
        "Crew-Neck": "redondo",
        "Round-Neck": "redondo",
        "Scoop-Neck": "redondo",
        "V-Neck": "en_v",
        "Deep-V-Neck": "en_v",
        "Collared-Neck": "polo",
        "Polo": "polo",
        "Polo-Neck": "polo",
        "Henley": "polo",
        "Turtleneck": "cuello_alto",
        "Mock-Neck": "cuello_alto",
        "Square-Neck": "cuadrado",
        "Boat-Neck": "barco",
        "Off-Shoulder": "sin_cuello",
        "Strapless": "sin_cuello",
        "Halter": "sin_cuello",
    },
}


def apply_mapping(
    value: Optional[str],
    mapping: dict[str, str],
    default: Optional[str] = DEFAULT_OTHER,
) -> Optional[str]:
    """Aplica un mapeo EN -> ES con fallback configurable.

    Args:
        value: Valor crudo del CSV.
        mapping: Diccionario EN -> ES.
        default: Valor a retornar si no esta mapeado. None deja pasar None.

    Returns:
        Valor mapeado, o default si no se encontro, o None si el input era None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = str(value).strip()
    return mapping.get(key, default)


def map_series(series: pd.Series, attribute: str) -> pd.Series:
    """Aplica `MAPPING_EN_ES[attribute]` a toda una columna.

    Valores no mapeados quedan como `DEFAULT_OTHER`. Nulos se preservan.
    """
    if attribute not in MAPPING_EN_ES:
        raise KeyError(f"No hay mapeo definido para '{attribute}'")
    mapping = MAPPING_EN_ES[attribute]
    return series.apply(lambda v: apply_mapping(v, mapping))


def group_long_tail(
    series: pd.Series,
    threshold: float = LONG_TAIL_THRESHOLD,
    other_label: str = DEFAULT_OTHER,
) -> pd.Series:
    """Reemplaza por `other_label` toda clase con frecuencia menor a `threshold`.

    Args:
        series: Serie categorica.
        threshold: Fraccion minima para conservar la clase (ej. 0.01 = 1%).
        other_label: Label para las clases agrupadas.

    Returns:
        Nueva serie con las clases minoritarias agrupadas.
    """
    counts = series.value_counts(normalize=True)
    keep = set(counts[counts >= threshold].index.tolist())
    return series.apply(lambda v: v if v in keep or pd.isna(v) else other_label)
