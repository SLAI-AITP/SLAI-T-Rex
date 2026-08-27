from __future__ import annotations

import random
from typing import Any


DEFAULT_BUSINESS_DECIMALS = 2


def round_business_float(value: Any, digits: int = DEFAULT_BUSINESS_DECIMALS) -> float:
    """Round seed-level business coefficients before they enter solver artifacts."""
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def uniform_rounded(low: float, high: float, digits: int = DEFAULT_BUSINESS_DECIMALS) -> float:
    return round_business_float(random.uniform(float(low), float(high)), digits)


def range_uniform_rounded(value: Any, digits: int = DEFAULT_BUSINESS_DECIMALS) -> float:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return uniform_rounded(float(value[0]), float(value[1]), digits)
    return round_business_float(value, digits)


def reciprocal_rounded(value: Any, digits: int = DEFAULT_BUSINESS_DECIMALS) -> float:
    return round_business_float(1.0 / float(value), digits)


def count_long_decimals(text: str, max_decimal_places: int = DEFAULT_BUSINESS_DECIMALS) -> int:
    import re

    if max_decimal_places < 0:
        return 0
    pattern = re.compile(rf"(?<![A-Za-z0-9_])[-+]?\d+\.\d{{{max_decimal_places + 1},}}(?![A-Za-z0-9_])")
    return len(pattern.findall(text or ""))
