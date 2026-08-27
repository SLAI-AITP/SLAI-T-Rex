from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EngineBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ProjectConfig(EngineBaseModel):
    project_name: str = "or_cpt_data_engine"
    version: str = "0.1.0"
    mode: str = "cold_start_optmath"
    random_seed: int = 42
    default_language: str = "en"
    target: dict[str, Any] = Field(default_factory=dict)
    data_mix: dict[str, float] = Field(default_factory=lambda: {
        "final_model_report_ratio": 0.70,
        "modeling_rationale_ratio": 0.30,
    })
    run: dict[str, Any] = Field(default_factory=dict)


class PathsConfig(EngineBaseModel):
    external_optmath_root: str = "external/OptMATH"
    optmath_generators_dir: str = "or_cpt_engine/generators/optmath_seed"
    internal_optmath_generators_dir: str = "or_cpt_engine/generators/optmath_seed"
    scenario_catalog_path: str = "engine_configs/scenario_catalog.yaml"
    family_contracts_path: str = "engine_configs/family_contracts.yaml"
    generator_profiles_path: str = "engine_configs/generator_profiles.yaml"
    production_targets_path: str = "engine_configs/production_targets.yaml"
    distribution_policy_path: str = "engine_configs/distribution_policy.yaml"
    quality_gate_path: str = "engine_configs/quality_gate.yaml"
    data_root: str = "data"
    runs_root: str = "data/runs"
    registry_root: str = "data/registry"
    final_root: str = "data/final"
    logs_root: str = "logs"
    reports_root: str = "reports"


class LLMProviderConfig(EngineBaseModel):
    type: str = "openai_compatible"
    endpoint_pool_path: str | None = None
    base_url_env: str = "OR_LLM_BASE_URL"
    api_key_env: str = "OR_LLM_API_KEY"
    model: str = "deepseek-v3"
    timeout_sec: int = 120
    max_retries: int = 3
    trust_env: bool = False
    concurrency_per_endpoint: int = 1
    endpoint_failure_threshold: int = 2
    endpoint_cooldown_seconds: int = 60


class LLMStageConfig(EngineBaseModel):
    provider: str = "default"
    temperature: float = 0.2
    max_tokens: int = 4096
    num_candidates: int = 1


class LLMConfig(EngineBaseModel):
    providers: dict[str, LLMProviderConfig] = Field(default_factory=lambda: {"default": LLMProviderConfig()})
    backtranslation: LLMStageConfig = Field(default_factory=lambda: LLMStageConfig(temperature=0.8, max_tokens=4096, num_candidates=2))
    forward_modeling: LLMStageConfig = Field(default_factory=lambda: LLMStageConfig(temperature=0.2, max_tokens=8192))
    rendering: LLMStageConfig = Field(default_factory=lambda: LLMStageConfig(temperature=0.3, max_tokens=8192))


class GenerationPlanConfig(EngineBaseModel):
    run_name: str = "optmath_cold_start_10k"
    source: dict[str, Any] = Field(default_factory=lambda: {"type": "optmath_generators"})
    generation: dict[str, Any] = Field(default_factory=lambda: {
        "num_instances_per_generator": 200,
        "max_iter_per_generator": 200,
        "num_workers": 8,
    })
    filters: dict[str, Any] = Field(default_factory=lambda: {
        "var_num_max": 200,
        "constraint_num_max": 500,
        "solve_timeout_sec": 60,
    })
    difficulty_mix: dict[str, float] = Field(default_factory=dict)


class QualityThresholdsConfig(EngineBaseModel):
    instance_generation: dict[str, Any] = Field(default_factory=dict)
    backtranslation: dict[str, Any] = Field(default_factory=dict)
    forward_modeling_eval: dict[str, Any] = Field(default_factory=lambda: {
        "allowed_execution_status": ["SUCCESS"],
        "objective_abs_tolerance": 1e-4,
        "objective_rel_tolerance": 1e-4,
    })
    forward_repair: dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "max_attempts": 1,
        "repairable_failure_families": [
            "objective_mismatch",
            "solver_status_failure",
            "execution_failure",
        ],
    })
    cpt_rendering: dict[str, Any] = Field(default_factory=dict)


class RenderingConfig(EngineBaseModel):
    final_model_report_ratio: float = 0.70
    modeling_rationale_ratio: float = 0.30
    doc_type_policy: dict[str, Any] = Field(default_factory=dict)
    max_views_per_seed_instance: int = 12
    language: str = "en"
    numeric_rendering: dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "max_decimal_places": 2,
        "apply_to_code_blocks": False,
    })


class TrainValSplitConfig(EngineBaseModel):
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    random_seed: int = 42


class EngineConfig(EngineBaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation_plan: GenerationPlanConfig = Field(default_factory=GenerationPlanConfig)
    quality_thresholds: QualityThresholdsConfig = Field(default_factory=QualityThresholdsConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    train_val_split: TrainValSplitConfig = Field(default_factory=TrainValSplitConfig)
    family_contracts: dict[str, Any] = Field(default_factory=dict)


class GeneratorRecord(EngineBaseModel):
    generator_id: str
    source: str = "optmath"
    name: str
    path: str
    python_files: list[str] = Field(default_factory=list)
    metadata_path: str | None = None
    readme_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    readme_excerpt: str | None = None
    domain: list[str] = Field(default_factory=list)
    model_type: str | None = None
    optimization_sense: str | None = None
    has_metadata: bool = False
    has_readme: bool = False
    has_python_generator: bool = False
    status: str = "REGISTERED"
    errors: list[str] = Field(default_factory=list)
    upstream_generator_name: str | None = None
    upstream_path: str | None = None
    internal_path: str | None = None
    profile_status: str = "UNKNOWN"
    local_modified: bool = False


class AdapterSmokeResult(EngineBaseModel):
    generator_id: str
    load_status: str
    generate_status: str
    seed: int
    difficulty_level: str
    num_variables: int | None = None
    num_constraints: int | None = None
    solver_status: str | None = None
    objective_value: float | None = None
    elapsed_sec: float = 0.0
    errors: list[str] = Field(default_factory=list)


class InstanceRecord(EngineBaseModel):
    instance_id: str
    run_id: str
    source: str = "optmath"
    generator_id: str
    generator_name: str | None = None
    seed: int
    difficulty_level: str
    generator_profile_version: str | None = None
    sub_family: str | None = None
    canonical_math_signature: dict[str, Any] = Field(default_factory=dict)
    concept_tags: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    source_compact_data: dict[str, Any] = Field(default_factory=dict)
    model_type: str | None = None
    optimization_sense: str | None = None
    num_variables: int | None = None
    num_constraints: int | None = None
    lp_text: str | None = None
    math_formula: str | None = None
    gurobi_code: str | None = None
    initial_solver_status: str | None = None
    initial_objective_value: float | None = None
    generation_status: str = "SUCCESS"
    errors: list[str] = Field(default_factory=list)


class SolverValidationRecord(EngineBaseModel):
    instance_id: str
    generator_id: str
    model_type: str | None = None
    difficulty_level: str | None = None
    generator_profile_version: str | None = None
    sub_family: str | None = None
    canonical_math_signature: dict[str, Any] = Field(default_factory=dict)
    concept_tags: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    source_compact_data: dict[str, Any] = Field(default_factory=dict)
    lp_text: str | None = None
    math_formula: str | None = None
    gurobi_code: str | None = None
    solver_validation: dict[str, Any] = Field(default_factory=dict)
    reference_answer: dict[str, Any] = Field(default_factory=dict)
    rejection_stage: str | None = None
    rejection_reason: str | None = None


class BacktranslationCandidate(EngineBaseModel):
    bt_id: str
    instance_id: str
    generator_id: str
    candidate_index: int = 1
    language: str = "en"
    problem_background: str = ""
    problem_statement: str
    structured_problem_data: dict[str, Any] = Field(default_factory=dict)
    reference_answer: dict[str, Any] = Field(default_factory=dict)
    source_lp_text: str | None = None
    source_math_formula: str | None = None
    source_compact_data: dict[str, Any] = Field(default_factory=dict)
    generator_profile_version: str | None = None
    difficulty_level: str | None = None
    sub_family: str | None = None
    canonical_math_signature: dict[str, Any] = Field(default_factory=dict)
    concept_tags: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    llm_metadata: dict[str, Any] = Field(default_factory=dict)
    bt_status: str = "SUCCESS"
    errors: list[str] = Field(default_factory=list)


class ForwardModelingOutput(EngineBaseModel):
    fm_id: str
    bt_id: str
    instance_id: str
    generator_id: str
    problem_statement: str
    structured_problem_data: dict[str, Any] = Field(default_factory=dict)
    source_lp_text: str | None = None
    source_math_formula: str | None = None
    source_compact_data: dict[str, Any] = Field(default_factory=dict)
    generated_answer: dict[str, Any] = Field(default_factory=dict)
    code_path: str | None = None
    code_preparation: dict[str, Any] = Field(default_factory=dict)
    reference_answer: dict[str, Any] = Field(default_factory=dict)
    generator_profile_version: str | None = None
    difficulty_level: str | None = None
    sub_family: str | None = None
    canonical_math_signature: dict[str, Any] = Field(default_factory=dict)
    concept_tags: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    family_contract_version: str | None = None
    family_contract_id: str | None = None
    family_contract: dict[str, Any] = Field(default_factory=dict)
    llm_metadata: dict[str, Any] = Field(default_factory=dict)
    forward_repair: dict[str, Any] = Field(default_factory=dict)
    fm_status: str = "SUCCESS"
    errors: list[str] = Field(default_factory=list)


class ObjectiveComparison(EngineBaseModel):
    is_correct: bool
    abs_error: float | None = None
    rel_error: float | None = None
    tolerance_abs: float = 1e-4
    tolerance_rel: float = 1e-4


class AcceptedPair(EngineBaseModel):
    pair_id: str
    fm_id: str
    bt_id: str
    instance_id: str
    generator_id: str
    problem_statement: str
    reference: dict[str, Any] = Field(default_factory=dict)
    generated: dict[str, Any] = Field(default_factory=dict)
    source_compact_data: dict[str, Any] = Field(default_factory=dict)
    generator_profile_version: str | None = None
    difficulty_level: str | None = None
    sub_family: str | None = None
    canonical_math_signature: dict[str, Any] = Field(default_factory=dict)
    concept_tags: list[str] = Field(default_factory=list)
    variant_id: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    family_contract_version: str | None = None
    family_contract_id: str | None = None
    family_contract: dict[str, Any] = Field(default_factory=dict)
    correctness: ObjectiveComparison
    modeling_answer: dict[str, Any] = Field(default_factory=dict)
    forward_repair: dict[str, Any] = Field(default_factory=dict)
    acceptance_status: str = "ACCEPTED"


class CPTDocument(EngineBaseModel):
    doc_id: str
    pair_id: str
    instance_id: str
    generator_id: str
    source: str = "optmath_seed_generator"
    doc_type: str
    language: str = "en"
    model_type: str | None = None
    difficulty_level: str | None = None
    text: str
    token_count: int = 0
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def ensure_stage_dir(output: str | Path) -> Path:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    return path
