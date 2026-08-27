from __future__ import annotations

from or_cpt_engine.schemas.common import ObjectiveComparison


def compare_objective(
    reference: float | int | None,
    generated: float | int | None,
    *,
    abs_tolerance: float = 1e-4,
    rel_tolerance: float = 1e-4,
) -> ObjectiveComparison:
    if reference is None or generated is None:
        return ObjectiveComparison(
            is_correct=False,
            abs_error=None,
            rel_error=None,
            tolerance_abs=abs_tolerance,
            tolerance_rel=rel_tolerance,
        )
    ref = float(reference)
    gen = float(generated)
    abs_error = abs(ref - gen)
    denominator = max(abs(ref), 1.0)
    rel_error = abs_error / denominator
    return ObjectiveComparison(
        is_correct=abs_error <= abs_tolerance or rel_error <= rel_tolerance,
        abs_error=abs_error,
        rel_error=rel_error,
        tolerance_abs=abs_tolerance,
        tolerance_rel=rel_tolerance,
    )
