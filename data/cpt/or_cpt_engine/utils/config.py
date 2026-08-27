from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from or_cpt_engine.schemas.common import (
    EngineConfig,
    GenerationPlanConfig,
    LLMConfig,
    PathsConfig,
    ProjectConfig,
    QualityThresholdsConfig,
    RenderingConfig,
    TrainValSplitConfig,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_engine_config(
    *,
    project_path: str | Path = "engine_configs/project.yaml",
    paths_path: str | Path = "engine_configs/paths.yaml",
    llm_path: str | Path = "engine_configs/llm.yaml",
    generation_plan_path: str | Path = "engine_configs/generation_plan.yaml",
    quality_thresholds_path: str | Path = "engine_configs/quality_thresholds.yaml",
    rendering_path: str | Path = "engine_configs/cpt_rendering.yaml",
    split_path: str | Path = "engine_configs/train_val_split.yaml",
    family_contracts_path: str | Path = "engine_configs/family_contracts.yaml",
) -> EngineConfig:
    paths_config = PathsConfig.model_validate(_load_yaml(paths_path))
    resolved_family_contracts_path = Path(family_contracts_path)
    if family_contracts_path == "engine_configs/family_contracts.yaml" and paths_config.family_contracts_path:
        resolved_family_contracts_path = Path(paths_config.family_contracts_path)
    family_contracts = _load_yaml(resolved_family_contracts_path) if resolved_family_contracts_path.exists() else {}
    return EngineConfig(
        project=ProjectConfig.model_validate(_load_yaml(project_path)),
        paths=paths_config,
        llm=LLMConfig.model_validate(_load_yaml(llm_path)),
        generation_plan=GenerationPlanConfig.model_validate(_load_yaml(generation_plan_path)),
        quality_thresholds=QualityThresholdsConfig.model_validate(_load_yaml(quality_thresholds_path)),
        rendering=RenderingConfig.model_validate(_load_yaml(rendering_path)),
        train_val_split=TrainValSplitConfig.model_validate(_load_yaml(split_path)),
        family_contracts=family_contracts,
    )
