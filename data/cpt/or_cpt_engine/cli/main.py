from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from or_cpt_engine.backtranslation.backtranslate import _compact_source_data as _backtranslation_compact_source_data
from or_cpt_engine.backtranslation.backtranslate import backtranslate_instances
from or_cpt_engine.backtranslation.nl_quality_filter import filter_nl_candidates
from or_cpt_engine.bootstrap.environment import run_bootstrap_checks
from or_cpt_engine.contracts import render_family_contract_inventory
from or_cpt_engine.evaluation.forward_eval import evaluate_forward_outputs
from or_cpt_engine.export.train_val_export import export_train_val_test
from or_cpt_engine.forward_modeling.generate_forward_model import forward_model_candidates
from or_cpt_engine.generation.generator_profiles import profile_generators
from or_cpt_engine.generation.instance_generator import generate_instances, smoke_test_generators
from or_cpt_engine.generation.production_planner import plan_production
from or_cpt_engine.generation.quality_analyzer import analyze_generator_quality
from or_cpt_engine.generators.import_optmath import import_optmath_generators
from or_cpt_engine.pipeline.cpt_streaming_pipeline import run_cpt_streaming_pipeline
from or_cpt_engine.prompt_optimizer.engine import optimize_prompts_for_run
from or_cpt_engine.quality.instance_quality import validate_instance_quality
from or_cpt_engine.registry.scan_optmath_generators import scan_optmath_generators
from or_cpt_engine.reporting.diversity_dashboard import analyze_run_diversity
from or_cpt_engine.reporting.run_quality_dashboard import analyze_run_quality
from or_cpt_engine.rendering.cpt_renderer import render_cpt_documents
from or_cpt_engine.solver.solver_validation import validate_solver
from or_cpt_engine.utils.config import load_engine_config
from or_cpt_engine.utils.io import count_jsonl_records, read_jsonl, write_text
from or_cpt_engine.utils.logging import configure_logging
from or_cpt_engine.utils.run_id import make_run_id
from cpt_cleaner.utils.throughput import write_throughput_summary

app = typer.Typer(help="OR-CPT data factory CLI.")

_COMPACT_FACT_HIGH_RISK_GENERATORS = {
    "optmath_clsp_expand_capacity",
    "optmath_uncapacitatedlotsizing",
    "optmath_uncapacitatedlotsizingbacklogging",
    "optmath_singlelevelsmallbucket",
    "optmath_marketshare",
    "optmath_structure_based_assignment",
    "optmath_steel4",
    "optmath_steel3",
    "optmath_net1",
    "optmath_netasgn",
    "optmath_electrical_power",
    "optmath_aircraftlanding",
    "optmath_aircraftassignment",
}

ProjectPath = Annotated[Path, typer.Option(help="Path to project.yaml")]
PathsPath = Annotated[Path, typer.Option(help="Path to paths.yaml")]
LLMPath = Annotated[Path, typer.Option(help="Path to llm.yaml")]
GenerationPlanPath = Annotated[Path, typer.Option(help="Path to generation_plan.yaml")]
QualityPath = Annotated[Path, typer.Option(help="Path to quality_thresholds.yaml")]
RenderingPath = Annotated[Path, typer.Option(help="Path to cpt_rendering.yaml")]
SplitPath = Annotated[Path, typer.Option(help="Path to train_val_split.yaml")]


def _load_config(
    project: Path = Path("engine_configs/project.yaml"),
    paths: Path = Path("engine_configs/paths.yaml"),
    llm: Path = Path("engine_configs/llm.yaml"),
    generation_plan: Path = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: Path = Path("engine_configs/quality_thresholds.yaml"),
    rendering: Path = Path("engine_configs/cpt_rendering.yaml"),
    split: Path = Path("engine_configs/train_val_split.yaml"),
):
    return load_engine_config(
        project_path=project,
        paths_path=paths,
        llm_path=llm,
        generation_plan_path=generation_plan,
        quality_thresholds_path=quality_thresholds,
        rendering_path=rendering,
        split_path=split,
    )


@app.command("bootstrap")
def bootstrap_cmd(
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    output: Annotated[Path | None, typer.Option(help="Output directory for bootstrap reports.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    run_id = make_run_id(config.generation_plan.run_name)
    output_dir = output or Path(config.paths.runs_root) / run_id / "00_bootstrap"
    report = run_bootstrap_checks(config, output_dir)
    typer.echo(f"bootstrap status={report['status']} output={output_dir}")


@app.command("import-optmath-generators")
def import_optmath_generators_cmd(
    source: Annotated[Path, typer.Option(help="Upstream OptMATH generators directory.")] = Path("external/OptMATH/generators"),
    output: Annotated[Path, typer.Option(help="Project-local internal generator directory.")] = Path("or_cpt_engine/generators/optmath_seed"),
    dry_run: Annotated[bool, typer.Option(help="Preview import without copying files.")] = False,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing internal generator files. Use carefully after local edits.")] = False,
) -> None:
    result = import_optmath_generators(source, output, dry_run=dry_run, overwrite=overwrite)
    typer.echo(
        "import-optmath-generators "
        f"generators={result['generators']} planned_files={result['planned_files']} "
        f"copied_files={result['copied_files']} dry_run={result['dry_run']} output={result['output']}"
    )


@app.command("scan-generators")
def scan_generators_cmd(
    generators_dir: Annotated[Path, typer.Option(help="OptMATH generators directory.")] = Path("or_cpt_engine/generators/optmath_seed"),
    output: Annotated[Path, typer.Option(help="Output directory for generator registry.")] = Path("data/runs/manual/01_generator_registry"),
    source_label: Annotated[str | None, typer.Option(help="Registry source label, e.g. optmath_internal or optmath.")] = None,
    profiles: Annotated[Path | None, typer.Option(help="Optional generator_profiles.yaml for profile coverage status.")] = Path("engine_configs/generator_profiles.yaml"),
) -> None:
    result = scan_optmath_generators(generators_dir, output, source_label=source_label, profiles_path=profiles)
    typer.echo(f"scan-generators scanned={result['scanned']} registered={result['registered']} invalid={result['invalid']} output={output}")


@app.command("profile-generators")
def profile_generators_cmd(
    registry: Annotated[Path, typer.Option(help="Path to generator_registry.jsonl.")],
    profiles_output: Annotated[Path, typer.Option(help="Output path for generator_profiles.yaml.")] = Path("engine_configs/generator_profiles.yaml"),
    report_output: Annotated[Path, typer.Option(help="Output path for generator profile inventory markdown.")] = Path("data/registry/generator_profile_inventory.md"),
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing generator profile file.")] = False,
) -> None:
    result = profile_generators(registry, profiles_output, report_output=report_output, overwrite=overwrite)
    typer.echo(f"profile-generators profiles={result['profiles']} output={profiles_output} report={report_output}")


@app.command("smoke-test-generators")
def smoke_test_generators_cmd(
    registry: Annotated[Path, typer.Option(help="Path to generator_registry.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for smoke report.")],
    seed: Annotated[int, typer.Option(help="Smoke-test random seed.")] = 42,
    difficulty_level: Annotated[str, typer.Option(help="Difficulty label passed to adapters.")] = "level_1",
    limit: Annotated[int | None, typer.Option(help="Maximum registered generators to smoke test.")] = None,
    extract_model_info: Annotated[bool, typer.Option(help="Also extract/solve model info during smoke; slower.")] = False,
    generator_ids: Annotated[str | None, typer.Option("--generator-ids", help="Comma-separated generator ids to include.")] = None,
) -> None:
    selected_generator_ids = _parse_generator_ids(generator_ids)
    result = smoke_test_generators(
        registry,
        output,
        seed=seed,
        difficulty_level=difficulty_level,
        limit=limit,
        extract_model_info=extract_model_info,
        generator_ids=selected_generator_ids,
    )
    typer.echo(f"smoke-test-generators total={result['total']} passed={result['passed']} failed={result['failed']} output={output}")


@app.command("generate-instances")
def generate_instances_cmd(
    registry: Annotated[Path, typer.Option(help="Path to generator_registry.jsonl.")],
    smoke_report: Annotated[Path, typer.Option(help="Path to adapter_smoke_report.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for generated instances.")],
    run_id: Annotated[str, typer.Option(help="Run id recorded into generated instances.")] = "manual",
    num_instances_per_generator: Annotated[int, typer.Option(help="Number of instances to request per smoke-passing generator.")] = 1,
    limit: Annotated[int | None, typer.Option(help="Maximum total generated attempts.")] = None,
    generator_ids: Annotated[str | None, typer.Option("--generator-ids", help="Comma-separated generator ids to include.")] = None,
    use_profiles: Annotated[bool, typer.Option(help="Use generator profiles for difficulty/concept metadata.")] = True,
    profiles: Annotated[Path, typer.Option(help="Path to generator_profiles.yaml.")] = Path("engine_configs/generator_profiles.yaml"),
    difficulty_mix: Annotated[str | None, typer.Option(help="Comma-separated difficulty mix, e.g. level_1=0.5,level_2=0.3.")] = None,
    extract_solve_timeout_sec: Annotated[int, typer.Option(help="Gurobi solve timeout during model info extraction (seconds).")] = 60,
) -> None:
    selected_generator_ids = _parse_generator_ids(generator_ids)
    result = generate_instances(
        registry,
        smoke_report,
        output,
        run_id=run_id,
        num_instances_per_generator=num_instances_per_generator,
        limit=limit,
        generator_ids=selected_generator_ids,
        profiles_path=profiles if use_profiles else None,
        difficulty_mix=_parse_difficulty_mix(difficulty_mix),
        extract_solve_timeout_sec=extract_solve_timeout_sec,
    )
    typer.echo(f"generate-instances generated={result['generated']} rejected={result['rejected']} output={output}")


@app.command("analyze-generator-quality")
def analyze_generator_quality_cmd(
    run_dir: Annotated[Path, typer.Option(help="Completed or partially completed OR-CPT run directory.")],
    output: Annotated[Path | None, typer.Option(help="Output directory for generator quality reports.")] = None,
    profiles: Annotated[Path, typer.Option(help="Path to generator_profiles.yaml.")] = Path("engine_configs/generator_profiles.yaml"),
) -> None:
    output_dir = output or run_dir / "reports" / "generator_quality"
    result = analyze_generator_quality(run_dir, output_dir, profiles_path=profiles)
    typer.echo(f"analyze-generator-quality generators={result['generators']} output={output_dir}")


@app.command("analyze-diversity")
def analyze_diversity_cmd(
    run_dir: Annotated[Path, typer.Option(help="Completed or partially completed OR-CPT CPT-factory run directory.")],
    output: Annotated[Path | None, typer.Option(help="Output directory for diversity dashboard artifacts.")] = None,
    near_duplicate_threshold: Annotated[float, typer.Option(help="Jaccard threshold used for sampled near-duplicate detection.")] = 0.72,
    max_rows_per_similarity_group: Annotated[int, typer.Option(help="Maximum rows sampled per generator/scenario/doc_type group.")] = 80,
    max_reported_pairs: Annotated[int, typer.Option(help="Maximum near-duplicate example pairs written to JSONL.")] = 100,
) -> None:
    output_dir = output or run_dir / "reports" / "diversity"
    result = analyze_run_diversity(
        run_dir,
        output_dir,
        near_duplicate_threshold=near_duplicate_threshold,
        max_rows_per_similarity_group=max_rows_per_similarity_group,
        max_reported_pairs=max_reported_pairs,
    )
    typer.echo(f"analyze-diversity stages={result['stages']} output={output_dir}")


@app.command("analyze-run-quality")
def analyze_run_quality_cmd(
    run_dir: Annotated[Path, typer.Option(help="Completed or partially completed OR-CPT CPT-factory run directory.")],
    output: Annotated[Path | None, typer.Option(help="Output directory for run quality dashboard artifacts.")] = None,
) -> None:
    output_dir = output or run_dir / "reports"
    result = analyze_run_quality(run_dir, output_dir)
    typer.echo(
        "analyze-run-quality "
        f"train={result['train_count']} rejection_rows={result['rejection_rows']} output={output_dir}"
    )


@app.command("family-contract-inventory")
def family_contract_inventory_cmd(
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    output: Annotated[Path, typer.Option(help="Output markdown path for family contract inventory.")] = Path("engine_runs/reports/family_contract_inventory.md"),
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    write_text(output, render_family_contract_inventory(config.family_contracts))
    typer.echo(f"family-contract-inventory output={output}")


@app.command("audit-generators")
def audit_generators_cmd(
    registry: Annotated[Path, typer.Option(help="Path to generator_registry.jsonl.")],
    smoke_report: Annotated[Path, typer.Option(help="Path to adapter_smoke_report.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for generator audit artifacts.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    samples_per_generator: Annotated[int, typer.Option(help="Number of candidate instances generated per selected generator.")] = 200,
    run_id: Annotated[str, typer.Option(help="Run id recorded into audit instances.")] = "generator_audit",
    generator_ids: Annotated[str | None, typer.Option("--generator-ids", help="Comma-separated generator ids to include.")] = None,
    difficulty_mix: Annotated[str | None, typer.Option(help="Comma-separated difficulty mix, e.g. level_1=0.5,level_2=0.3.")] = None,
    solver_concurrency: Annotated[int | None, typer.Option("--solver-concurrency", help="Concurrent solver subprocesses for stage 04. Defaults to filters.solver_concurrency or generation.num_workers.")] = None,
    solver_threads_per_process: Annotated[int | None, typer.Option("--solver-threads-per-process", help="Gurobi Threads per solver subprocess. Defaults to filters.solver_threads_per_process or 1.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    selected_generator_ids = _parse_generator_ids(generator_ids)
    audit_dir = Path(output)
    instances_dir = audit_dir / "03_instance_generation"
    solver_dir = audit_dir / "04_solver_validation"
    quality_dir = audit_dir / "04b_instance_quality"
    gen_result = generate_instances(
        registry,
        smoke_report,
        instances_dir,
        run_id=run_id,
        num_instances_per_generator=samples_per_generator,
        generator_ids=selected_generator_ids,
        profiles_path=config.paths.generator_profiles_path,
        difficulty_mix=_parse_difficulty_mix(difficulty_mix) or config.generation_plan.difficulty_mix,
        extract_solve_timeout_sec=int(config.generation_plan.filters.get("solve_timeout_sec", 60)),
    )
    solver_result = validate_solver(
        instances_dir / "instances_generated.jsonl",
        solver_dir,
        timeout_seconds=int(config.generation_plan.filters.get("solve_timeout_sec", 60)),
        concurrency=_resolve_solver_concurrency(config, solver_concurrency),
        threads_per_process=_resolve_solver_threads_per_process(config, solver_threads_per_process),
    )
    quality_result = validate_instance_quality(
        solver_dir / "solver_validated_instances.jsonl",
        quality_dir,
        thresholds=config.quality_thresholds.instance_generation,
        profiles_path=config.paths.generator_profiles_path,
    )
    analysis_result = analyze_generator_quality(
        audit_dir,
        audit_dir / "reports",
        profiles_path=config.paths.generator_profiles_path,
    )
    typer.echo(
        "audit-generators "
        f"generated={gen_result['generated']} solver_validated={solver_result['validated']} "
        f"quality_pass={quality_result['accepted']} review={quality_result['review']} "
        f"rejected={quality_result['rejected']} analyzed_generators={analysis_result['generators']} "
        f"output={audit_dir}"
    )


@app.command("plan-production")
def plan_production_cmd(
    target_accepted: Annotated[int, typer.Option(help="Target number of accepted CPT documents.")],
    output: Annotated[Path, typer.Option(help="Output directory for production_plan.json/md.")] = Path("data/registry/production_plan"),
    profiles: Annotated[Path, typer.Option(help="Path to generator_profiles.yaml.")] = Path("engine_configs/generator_profiles.yaml"),
    quality_report: Annotated[Path | None, typer.Option(help="Optional generator_quality_report.jsonl.")] = None,
    min_expected_acceptance_rate: Annotated[float, typer.Option(help="Floor used when estimating planned attempts.")] = 0.05,
) -> None:
    result = plan_production(
        target_accepted=target_accepted,
        profiles_path=profiles,
        output_dir=output,
        quality_report_path=quality_report,
        min_expected_acceptance_rate=min_expected_acceptance_rate,
    )
    typer.echo(
        "plan-production "
        f"target_accepted={result['target_accepted']} generators={result['generators']} "
        f"planned_attempts={result['planned_attempts']} output={result['output']}"
    )


@app.command("validate-solver")
def validate_solver_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to instances_generated.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for solver validation.")],
    timeout_seconds: Annotated[int, typer.Option(help="Per-instance solver execution timeout.")] = 60,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Concurrent solver subprocesses. Use 1 for serial execution.")] = 1,
    threads_per_process: Annotated[int, typer.Option("--threads-per-process", help="Gurobi Threads per solver subprocess.")] = 1,
) -> None:
    result = validate_solver(
        input_path,
        output,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        threads_per_process=threads_per_process,
    )
    typer.echo(f"validate-solver validated={result['validated']} rejected={result['rejected']} output={output}")


@app.command("validate-instance-quality")
def validate_instance_quality_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to solver_validated_instances.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for 04b instance quality gate.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = validate_instance_quality(
        input_path,
        output,
        thresholds=config.quality_thresholds.instance_generation,
        profiles_path=config.paths.generator_profiles_path,
    )
    typer.echo(
        "validate-instance-quality "
        f"accepted={result['accepted']} review={result['review']} rejected={result['rejected']} output={output}"
    )


@app.command("backtranslate")
def backtranslate_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to solver_validated_instances.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for backtranslation.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    limit: Annotated[int | None, typer.Option(help="Maximum input rows.")] = None,
    mock_llm: Annotated[bool, typer.Option(help="Use deterministic mock LLM output.")] = False,
    llm_concurrency: Annotated[int | None, typer.Option(help="Per-endpoint LLM concurrency override.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = backtranslate_instances(
        input_path,
        output,
        config,
        limit=limit,
        mock=mock_llm,
        concurrency_per_endpoint=llm_concurrency,
    )
    typer.echo(f"backtranslate candidates={result['candidates']} rejected={result['rejected']} output={output}")


@app.command("filter-nl")
def filter_nl_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to backtranslation_candidates.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for NL quality filter.")],
    min_chars: Annotated[int, typer.Option(help="Minimum problem statement characters.")] = 160,
    max_chars: Annotated[int, typer.Option(help="Maximum problem statement characters.")] = 4000,
    min_number_coverage: Annotated[float, typer.Option(help="Minimum source-number coverage ratio.")] = 0.4,
) -> None:
    result = filter_nl_candidates(input_path, output, min_chars=min_chars, max_chars=max_chars, min_number_coverage=min_number_coverage)
    typer.echo(f"filter-nl accepted={result['accepted']} rejected={result['rejected']} output={output}")


@app.command("forward-model")
def forward_model_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to nl_validated_candidates.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for forward modeling.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    limit: Annotated[int | None, typer.Option(help="Maximum input rows.")] = None,
    mock_llm: Annotated[bool, typer.Option(help="Use deterministic mock LLM output.")] = False,
    llm_concurrency: Annotated[int | None, typer.Option(help="Per-endpoint LLM concurrency override.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = forward_model_candidates(
        input_path,
        output,
        config,
        limit=limit,
        mock=mock_llm,
        concurrency_per_endpoint=llm_concurrency,
    )
    typer.echo(f"forward-model outputs={result['outputs']} rejected={result['rejected']} output={output}")


@app.command("forward-eval")
def forward_eval_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to forward_modeling_outputs.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for forward evaluation.")],
    timeout_seconds: Annotated[int, typer.Option(help="Per-code execution timeout.")] = 60,
    abs_tolerance: Annotated[float, typer.Option(help="Objective absolute tolerance.")] = 1e-4,
    rel_tolerance: Annotated[float, typer.Option(help="Objective relative tolerance.")] = 1e-4,
) -> None:
    result = evaluate_forward_outputs(
        input_path,
        output,
        timeout_seconds=timeout_seconds,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    typer.echo(f"forward-eval accepted={result['accepted']} rejected={result['rejected']} output={output}")


@app.command("render-cpt")
def render_cpt_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to accepted_pairs.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for CPT rendering.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    limit: Annotated[int | None, typer.Option(help="Maximum input rows.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = render_cpt_documents(input_path, output, config, limit=limit)
    typer.echo(f"render-cpt documents={result['documents']} rejected={result['rejected']} output={output}")


@app.command("export")
def export_cmd(
    input_path: Annotated[Path, typer.Option("--input", help="Path to cpt_documents.jsonl.")],
    output: Annotated[Path, typer.Option(help="Output directory for train-only export.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    exact_dedup: Annotated[bool, typer.Option(help="Enable exact text dedup.")] = True,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = export_train_val_test(input_path, output, config, exact_dedup=exact_dedup)
    typer.echo(
        "export "
        f"mode={result.get('mode', 'train_only_streaming')} "
        f"train={result['train']} "
        f"duplicates_removed={result.get('duplicates_removed', 0)} "
        f"output={output}"
    )


@app.command("optimize-prompts")
def optimize_prompts_cmd(
    run_dir: Annotated[Path, typer.Option(help="Completed or partially completed OR-CPT run directory.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    auto_promote: Annotated[bool, typer.Option(help="Write the optimized prompt back and bump prompt_versions.yaml.")] = False,
    force: Annotated[bool, typer.Option(help="Generate a candidate even if no clear prompt-owned failures are found.")] = False,
    mock_llm: Annotated[bool, typer.Option(help="Use deterministic mock optimizer output.")] = False,
    sample_limit: Annotated[int, typer.Option(help="Rejected examples included in optimizer diagnostics.")] = 5,
    llm_concurrency: Annotated[int | None, typer.Option(help="Per-endpoint LLM concurrency override for optimizer call.")] = None,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    result = optimize_prompts_for_run(
        run_dir,
        config,
        auto_promote=auto_promote,
        force=force,
        mock=mock_llm,
        sample_limit=sample_limit,
        concurrency_per_endpoint=llm_concurrency,
    )
    typer.echo(
        "optimize-prompts "
        f"status={result.get('status')} "
        f"target_prompt={result.get('target_prompt') or (result.get('diagnostics') or {}).get('target_prompt')} "
        f"run_dir={run_dir}"
    )


@app.command("run-seed-factory")
def run_seed_factory_cmd(
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    run_id: Annotated[str | None, typer.Option(help="Optional fixed seed-factory run id.")] = None,
    num_instances: Annotated[int | None, typer.Option("--num-instances", help="Total seed attempts. With --production-plan, scale the plan to this total.")] = None,
    production_plan: Annotated[Path | None, typer.Option("--production-plan", help="Optional production_plan.json consumed by the seed factory.")] = None,
    generator_ids: Annotated[str | None, typer.Option("--generator-ids", help="Comma-separated generator ids to include.")] = None,
    solver_concurrency: Annotated[int | None, typer.Option("--solver-concurrency", help="Concurrent solver subprocesses for stage 04. Defaults to filters.solver_concurrency or generation.num_workers.")] = None,
    solver_threads_per_process: Annotated[int | None, typer.Option("--solver-threads-per-process", help="Gurobi Threads per solver subprocess. Defaults to filters.solver_threads_per_process or 1.")] = None,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR).")] = "INFO",
    force: Annotated[bool, typer.Option(help="Remove an existing run directory before starting.")] = False,
) -> None:
    """Run the non-LLM seed factory: 00 -> 04b plus seed reports."""
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    selected_generator_ids = _parse_generator_ids(generator_ids)
    resolved_run_id = run_id or make_run_id(f"{config.generation_plan.run_name}_seed")
    run_dir = Path(config.paths.runs_root) / resolved_run_id
    _prepare_run_dir(run_dir, force=force)
    stage_dirs = _stage_dirs(run_dir)
    wall_start = time.perf_counter()
    logger, run_log_path, _ = _configure_run_logging(run_dir, resolved_run_id, log_level)
    logger.info("run-seed-factory started run_id=%s run_dir=%s", resolved_run_id, run_dir)

    try:
        result = _run_seed_factory_stages(
            config=config,
            run_dir=run_dir,
            stage_dirs=stage_dirs,
            run_id=resolved_run_id,
            selected_generator_ids=selected_generator_ids,
            production_plan=production_plan,
            num_instances=num_instances,
            solver_concurrency=_resolve_solver_concurrency(config, solver_concurrency),
            solver_threads_per_process=_resolve_solver_threads_per_process(config, solver_threads_per_process),
            logger=logger,
        )
        wall_secs = time.perf_counter() - wall_start
        seed_manifest = _write_seed_manifest(
            run_dir=run_dir,
            run_id=resolved_run_id,
            config=config,
            stage_results=result["stage_results"],
            per_generator_counts=result["per_generator_counts"],
            production_plan_usage=result["production_plan_usage"],
            wall_seconds=wall_secs,
        )
        logger.info("run-seed-factory complete run_dir=%s seed_manifest=%s wall_seconds=%.1f", run_dir, seed_manifest, wall_secs)
        typer.echo(f"run-seed-factory complete run_dir={run_dir}")
        typer.echo(f"Seed input: {stage_dirs['instance_quality'] / 'quality_validated_instances.jsonl'}")
        typer.echo(f"Seed manifest: {seed_manifest}")
    finally:
        _write_run_throughput_summary(run_dir, run_log_path, time.perf_counter() - wall_start, logger)

    typer.echo(f"Task log file: {run_log_path}")


@app.command("run-cpt-from-seeds")
def run_cpt_from_seeds_cmd(
    seed_input: Annotated[Path, typer.Option("--seed-input", help="Path to quality_validated_instances.jsonl from run-seed-factory.")],
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    run_id: Annotated[str | None, typer.Option(help="Optional fixed CPT-factory run id.")] = None,
    seed_manifest: Annotated[Path | None, typer.Option("--seed-manifest", help="Optional seed_manifest.json copied into the CPT run reports.")] = None,
    limit: Annotated[int | None, typer.Option(help="Maximum seed rows consumed by backtranslation.")] = None,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR).")] = "INFO",
    mock_llm: Annotated[bool, typer.Option(help="Use deterministic mock LLM output for LLM stages.")] = False,
    llm_concurrency: Annotated[int | None, typer.Option(help="Per-endpoint LLM concurrency override for LLM stages.")] = None,
    pipeline_mode: Annotated[str, typer.Option("--pipeline-mode", help="CPT factory execution mode: streaming or staged.")] = "streaming",
    forward_eval_concurrency: Annotated[int | None, typer.Option("--forward-eval-concurrency", help="Concurrent solver subprocesses for stage 08 in streaming mode. Defaults to filters.solver_concurrency or generation.num_workers.")] = None,
    solver_threads_per_process: Annotated[int | None, typer.Option("--solver-threads-per-process", help="Gurobi Threads per stage-08 solver subprocess in streaming mode. Defaults to filters.solver_threads_per_process or 1.")] = None,
    pipeline_queue_size: Annotated[int | None, typer.Option("--pipeline-queue-size", help="Bounded queue size between streaming stages. Defaults to max(1024, total_llm_capacity * 8).")] = None,
    auto_optimize_prompts: Annotated[bool, typer.Option(help="Run prompt optimization analysis after CPT factory completes.")] = False,
    auto_promote_prompts: Annotated[bool, typer.Option(help="Allow post-run optimizer to write prompts and bump versions.")] = False,
    force: Annotated[bool, typer.Option(help="Remove an existing run directory before starting.")] = False,
) -> None:
    """Run the LLM CPT factory from frozen seeds: 05 -> 10."""
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    resolved_run_id = run_id or make_run_id(f"{config.generation_plan.run_name}_cpt")
    run_dir = Path(config.paths.runs_root) / resolved_run_id
    _prepare_run_dir(run_dir, force=force)
    stage_dirs = _stage_dirs(run_dir)
    wall_start = time.perf_counter()
    logger, run_log_path, _ = _configure_run_logging(run_dir, resolved_run_id, log_level)
    resolved_pipeline_mode = pipeline_mode.strip().lower()
    if resolved_pipeline_mode not in {"streaming", "staged"}:
        raise typer.BadParameter("--pipeline-mode must be either streaming or staged")
    logger.info(
        "run-cpt-from-seeds started run_id=%s run_dir=%s seed_input=%s pipeline_mode=%s",
        resolved_run_id,
        run_dir,
        seed_input,
        resolved_pipeline_mode,
    )
    seed_preflight_result = _audit_seed_input_for_cpt(seed_input, run_dir, limit=limit)
    logger.info(
        "stage=seed_input_preflight status=completed seed_rows=%s compact_available=%s high_risk_missing_compact=%s report=%s",
        seed_preflight_result.get("seed_rows"),
        seed_preflight_result.get("compact_available"),
        seed_preflight_result.get("high_risk_missing_compact"),
        seed_preflight_result.get("report_path"),
    )
    if seed_preflight_result.get("high_risk_missing_compact"):
        logger.warning(
            "stage=seed_input_preflight status=warning high_risk_missing_compact=%s missing_by_generator=%s",
            seed_preflight_result.get("high_risk_missing_compact"),
            seed_preflight_result.get("missing_compact_by_generator"),
        )

    try:
        if resolved_pipeline_mode == "streaming":
            _record_seed_lineage(run_dir, seed_manifest)
            stage_results = run_cpt_streaming_pipeline(
                seed_input=seed_input,
                stage_dirs=stage_dirs,
                config=config,
                limit=limit,
                mock_llm=mock_llm,
                llm_concurrency=llm_concurrency,
                forward_eval_concurrency=_resolve_solver_concurrency(config, forward_eval_concurrency),
                solver_threads_per_process=_resolve_solver_threads_per_process(config, solver_threads_per_process),
                queue_size=pipeline_queue_size,
                logger=logger,
            )
            stage_results["seed_input_preflight"] = seed_preflight_result
            if auto_optimize_prompts:
                logger.info("stage=prompt_optimization status=started")
                optimize_result = optimize_prompts_for_run(
                    run_dir,
                    config,
                    auto_promote=auto_promote_prompts,
                    mock=mock_llm,
                    concurrency_per_endpoint=llm_concurrency,
                )
                stage_results["prompt_optimization"] = optimize_result
                logger.info(
                    "stage=prompt_optimization status=%s target_prompt=%s",
                    optimize_result.get("status"),
                    optimize_result.get("target_prompt") or (optimize_result.get("diagnostics") or {}).get("target_prompt"),
                )
                typer.echo(
                    "post-run prompt optimization "
                    f"status={optimize_result.get('status')} "
                    f"target_prompt={optimize_result.get('target_prompt') or (optimize_result.get('diagnostics') or {}).get('target_prompt')}"
                )
        else:
            if forward_eval_concurrency is not None or solver_threads_per_process is not None or pipeline_queue_size is not None:
                logger.warning(
                    "pipeline_mode=staged ignores --forward-eval-concurrency, --solver-threads-per-process, and --pipeline-queue-size"
                )
            stage_results = _run_cpt_factory_stages(
                config=config,
                run_dir=run_dir,
                stage_dirs=stage_dirs,
                seed_input=seed_input,
                seed_manifest=seed_manifest,
                limit=limit,
                mock_llm=mock_llm,
                llm_concurrency=llm_concurrency,
                auto_optimize_prompts=auto_optimize_prompts,
                auto_promote_prompts=auto_promote_prompts,
                logger=logger,
            )
            stage_results["seed_input_preflight"] = seed_preflight_result
        # The run quality dashboard reads the throughput summary. Write a
        # provisional copy before report generation, then refresh it in finally.
        _write_run_throughput_summary(run_dir, run_log_path, time.perf_counter() - wall_start, logger)
        logger.info("stage=diversity_dashboard status=started")
        try:
            diversity_result = analyze_run_diversity(run_dir, run_dir / "reports" / "diversity")
            stage_results["diversity_dashboard"] = diversity_result
            logger.info(
                "stage=diversity_dashboard status=completed stages=%s output=%s",
                diversity_result.get("stages"),
                diversity_result.get("output_dir"),
            )
        except Exception:
            logger.warning("stage=diversity_dashboard status=failed", exc_info=True)
        logger.info("stage=run_quality_dashboard status=started")
        try:
            quality_dashboard_result = analyze_run_quality(run_dir, run_dir / "reports")
            stage_results["run_quality_dashboard"] = quality_dashboard_result
            logger.info(
                "stage=run_quality_dashboard status=completed train=%s rejection_rows=%s output=%s",
                quality_dashboard_result.get("train_count"),
                quality_dashboard_result.get("rejection_rows"),
                quality_dashboard_result.get("output_dir"),
            )
        except Exception:
            logger.warning("stage=run_quality_dashboard status=failed", exc_info=True)
        wall_secs = time.perf_counter() - wall_start
        cpt_manifest = _write_cpt_run_manifest(
            run_dir=run_dir,
            run_id=resolved_run_id,
            seed_input=seed_input,
            seed_manifest=seed_manifest,
            stage_results=stage_results,
            wall_seconds=wall_secs,
        )
        logger.info("run-cpt-from-seeds complete run_dir=%s cpt_manifest=%s wall_seconds=%.1f", run_dir, cpt_manifest, wall_secs)
        typer.echo(f"run-cpt-from-seeds complete run_dir={run_dir}")
        typer.echo(f"Train output: {stage_dirs['export'] / 'train.jsonl'}")
        typer.echo(f"CPT manifest: {cpt_manifest}")
    finally:
        _write_run_throughput_summary(run_dir, run_log_path, time.perf_counter() - wall_start, logger)

    typer.echo(f"Task log file: {run_log_path}")


@app.command("run-full")
def run_full_cmd(
    project: ProjectPath = Path("engine_configs/project.yaml"),
    paths: PathsPath = Path("engine_configs/paths.yaml"),
    llm: LLMPath = Path("engine_configs/llm.yaml"),
    generation_plan: GenerationPlanPath = Path("engine_configs/generation_plan.yaml"),
    quality_thresholds: QualityPath = Path("engine_configs/quality_thresholds.yaml"),
    rendering: RenderingPath = Path("engine_configs/cpt_rendering.yaml"),
    split: SplitPath = Path("engine_configs/train_val_split.yaml"),
    run_id: Annotated[str | None, typer.Option(help="Optional fixed run id.")] = None,
    num_instances: Annotated[int | None, typer.Option("--num-instances", help="Total number of instances to generate, evenly distributed across selected generators.")] = None,
    production_plan: Annotated[Path | None, typer.Option("--production-plan", help="Optional production_plan.json. When provided, instance counts follow planned_attempts; --num-instances scales the plan proportionally.")] = None,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR).")] = "INFO",
    mock_llm: Annotated[bool, typer.Option(help="Use deterministic mock LLM output for LLM stages.")] = False,
    generator_ids: Annotated[str | None, typer.Option("--generator-ids", help="Comma-separated generator ids to include, for example optmath_knapsack.")] = None,
    solver_concurrency: Annotated[int | None, typer.Option("--solver-concurrency", help="Concurrent solver subprocesses for stage 04. Defaults to filters.solver_concurrency or generation.num_workers.")] = None,
    solver_threads_per_process: Annotated[int | None, typer.Option("--solver-threads-per-process", help="Gurobi Threads per solver subprocess. Defaults to filters.solver_threads_per_process or 1.")] = None,
    llm_concurrency: Annotated[int | None, typer.Option(help="Per-endpoint LLM concurrency override for LLM stages.")] = None,
    auto_optimize_prompts: Annotated[bool, typer.Option(help="Run prompt optimization analysis after run-full completes.")] = False,
    auto_promote_prompts: Annotated[bool, typer.Option(help="Allow post-run optimizer to write prompts and bump versions.")] = False,
) -> None:
    config = _load_config(project, paths, llm, generation_plan, quality_thresholds, rendering, split)
    selected_generator_ids = _parse_generator_ids(generator_ids)
    resolved_run_id = run_id or make_run_id(config.generation_plan.run_name)
    run_dir = Path(config.paths.runs_root) / resolved_run_id
    stage_dirs = _stage_dirs(run_dir)
    wall_start = time.perf_counter()

    logger, run_log_path, _ = _configure_run_logging(run_dir, resolved_run_id, log_level)
    logger.info("run-full started run_id=%s run_dir=%s", resolved_run_id, run_dir)

    try:
        seed_result = _run_seed_factory_stages(
            config=config,
            run_dir=run_dir,
            stage_dirs=stage_dirs,
            run_id=resolved_run_id,
            selected_generator_ids=selected_generator_ids,
            production_plan=production_plan,
            num_instances=num_instances,
            solver_concurrency=_resolve_solver_concurrency(config, solver_concurrency),
            solver_threads_per_process=_resolve_solver_threads_per_process(config, solver_threads_per_process),
            logger=logger,
        )
        seed_manifest = _write_seed_manifest(
            run_dir=run_dir,
            run_id=resolved_run_id,
            config=config,
            stage_results=seed_result["stage_results"],
            per_generator_counts=seed_result["per_generator_counts"],
            production_plan_usage=seed_result["production_plan_usage"],
            wall_seconds=time.perf_counter() - wall_start,
        )
        logger.info("seed_manifest written to %s", seed_manifest)

        cpt_stage_results = _run_cpt_factory_stages(
            config=config,
            run_dir=run_dir,
            stage_dirs=stage_dirs,
            seed_input=stage_dirs["instance_quality"] / "quality_validated_instances.jsonl",
            seed_manifest=seed_manifest,
            limit=None,
            mock_llm=mock_llm,
            llm_concurrency=llm_concurrency,
            auto_optimize_prompts=auto_optimize_prompts,
            auto_promote_prompts=auto_promote_prompts,
            logger=logger,
        )
        cpt_manifest = _write_cpt_run_manifest(
            run_dir=run_dir,
            run_id=resolved_run_id,
            seed_input=stage_dirs["instance_quality"] / "quality_validated_instances.jsonl",
            seed_manifest=seed_manifest,
            stage_results=cpt_stage_results,
            wall_seconds=time.perf_counter() - wall_start,
        )
        logger.info("cpt_run_manifest written to %s", cpt_manifest)

        wall_secs = time.perf_counter() - wall_start
        logger.info("run-full complete run_dir=%s wall_seconds=%.1f", run_dir, wall_secs)
        typer.echo(f"run-full complete run_dir={run_dir}")
    finally:
        _write_run_throughput_summary(run_dir, run_log_path, time.perf_counter() - wall_start, logger)

    typer.echo(f"Task log file: {run_log_path}")


def _configure_run_logging(run_dir: Path, resolved_run_id: str, log_level: str) -> tuple[logging.Logger, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_run_id)
    run_log_path = run_dir / "logs" / f"{safe_run_id}_{timestamp}.log"
    model_failure_log_path = run_log_path.with_name(f"{run_log_path.stem}_model_failures.jsonl")
    configure_logging(
        log_level,
        log_file=str(run_log_path),
        model_failure_log_file=str(model_failure_log_path),
    )
    return logging.getLogger("or_cpt_engine"), run_log_path, model_failure_log_path


def _prepare_run_dir(run_dir: Path, *, force: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not force:
            raise typer.BadParameter(f"run directory already exists and is not empty: {run_dir}. Use --force to overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _write_run_throughput_summary(run_dir: Path, run_log_path: Path, wall_seconds: float, logger: logging.Logger) -> None:
    report_dir = run_dir / "reports"
    if not run_log_path.exists():
        return
    try:
        write_throughput_summary(
            report_dir,
            run_log_path,
            wall_seconds=wall_seconds,
            npu_count=0,  # or_cpt_engine doesn't track NPU count; set 0 to skip kNPU metrics
        )
        logger.info("throughput_summary written to %s", report_dir)
    except Exception:
        logger.warning("Failed to write throughput summary", exc_info=True)


def _run_seed_factory_stages(
    *,
    config,
    run_dir: Path,
    stage_dirs: dict[str, Path],
    run_id: str,
    selected_generator_ids: set[str] | None,
    production_plan: Path | None,
    num_instances: int | None,
    solver_concurrency: int,
    solver_threads_per_process: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    stage_results: dict[str, dict[str, Any]] = {}

    logger.info("stage=00_bootstrap status=started")
    stage_results["bootstrap"] = run_bootstrap_checks(config, stage_dirs["bootstrap"])
    logger.info("stage=00_bootstrap status=completed")

    logger.info("stage=01_generator_registry status=started")
    scan_result = scan_optmath_generators(
        config.paths.optmath_generators_dir,
        stage_dirs["registry"],
        source_label="optmath_internal",
        profiles_path=config.paths.generator_profiles_path,
    )
    stage_results["scan"] = scan_result
    logger.info(
        "stage=01_generator_registry status=completed scanned=%s registered=%s invalid=%s",
        scan_result.get("scanned"),
        scan_result.get("registered"),
        scan_result.get("invalid"),
    )

    logger.info("stage=02_adapter_smoke_test status=started")
    smoke_result = smoke_test_generators(
        stage_dirs["registry"] / "generator_registry.jsonl",
        stage_dirs["smoke"],
        generator_ids=selected_generator_ids,
    )
    stage_results["smoke"] = smoke_result
    logger.info(
        "stage=02_adapter_smoke_test status=completed total=%s passed=%s failed=%s",
        smoke_result.get("total"),
        smoke_result.get("passed"),
        smoke_result.get("failed"),
    )

    smoke_report_path = stage_dirs["smoke"] / "adapter_smoke_report.jsonl"
    per_gen_count, production_plan_usage = _resolve_per_generator_counts(
        smoke_report_path=smoke_report_path,
        selected_generator_ids=selected_generator_ids,
        production_plan_path=production_plan,
        num_instances=num_instances,
        yaml_per_gen=int(config.generation_plan.generation.get("num_instances_per_generator", 1)),
    )
    if production_plan_usage:
        _record_production_plan_usage(run_dir, production_plan, production_plan_usage)
        logger.info("production_plan_path=%s scaled_total=%s", production_plan, num_instances)
    logger.info("per_generator_counts=%s total_planned=%s", per_gen_count, sum(per_gen_count.values()))

    logger.info("stage=03_instance_generation status=started")
    gen_result = generate_instances(
        stage_dirs["registry"] / "generator_registry.jsonl",
        smoke_report_path,
        stage_dirs["instances"],
        run_id=run_id,
        num_instances_per_generator=per_gen_count,
        generator_ids=selected_generator_ids,
        profiles_path=config.paths.generator_profiles_path,
        difficulty_mix=config.generation_plan.difficulty_mix,
        extract_solve_timeout_sec=int(config.generation_plan.filters.get("solve_timeout_sec", 60)),
    )
    stage_results["generation"] = gen_result
    logger.info("stage=03_instance_generation status=completed generated=%s rejected=%s", gen_result.get("generated"), gen_result.get("rejected"))

    logger.info("stage=04_solver_validation status=started")
    solver_result = validate_solver(
        stage_dirs["instances"] / "instances_generated.jsonl",
        stage_dirs["solver"],
        timeout_seconds=int(config.generation_plan.filters.get("solve_timeout_sec", 60)),
        concurrency=solver_concurrency,
        threads_per_process=solver_threads_per_process,
    )
    stage_results["solver"] = solver_result
    logger.info(
        "stage=04_solver_validation status=completed validated=%s rejected=%s concurrency=%s threads_per_process=%s",
        solver_result.get("validated"),
        solver_result.get("rejected"),
        solver_concurrency,
        solver_threads_per_process,
    )

    logger.info("stage=04b_instance_quality status=started")
    quality_result = validate_instance_quality(
        stage_dirs["solver"] / "solver_validated_instances.jsonl",
        stage_dirs["instance_quality"],
        thresholds=config.quality_thresholds.instance_generation,
        profiles_path=config.paths.generator_profiles_path,
    )
    stage_results["instance_quality"] = quality_result
    logger.info(
        "stage=04b_instance_quality status=completed accepted=%s review=%s rejected=%s",
        quality_result.get("accepted"),
        quality_result.get("review"),
        quality_result.get("rejected"),
    )

    logger.info("stage=generator_quality_analysis status=started")
    analysis_result = analyze_generator_quality(
        run_dir,
        run_dir / "reports",
        profiles_path=config.paths.generator_profiles_path,
    )
    stage_results["generator_quality_analysis"] = analysis_result
    logger.info("stage=generator_quality_analysis status=completed generators=%s", analysis_result.get("generators"))

    return {
        "stage_results": stage_results,
        "per_generator_counts": per_gen_count,
        "production_plan_usage": production_plan_usage,
    }


def _audit_seed_input_for_cpt(seed_input: Path, run_dir: Path, *, limit: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(seed_input)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    explicit_compact = 0
    fallback_compact = 0
    legacy_reconstructable = 0
    compact_available = 0
    high_risk_rows = 0
    high_risk_missing = 0
    missing_by_generator: dict[str, int] = {}
    fallback_by_generator: dict[str, int] = {}
    legacy_by_generator: dict[str, int] = {}

    for row in rows:
        generator_id = str(row.get("generator_id") or "unknown")
        has_explicit = _row_has_explicit_source_compact(row)
        has_fallback = _row_has_generation_param_compact(row)
        has_legacy = False
        if not (has_explicit or has_fallback):
            has_legacy = _row_has_legacy_reconstructable_compact(row)
        has_any = has_explicit or has_fallback or has_legacy
        if has_explicit:
            explicit_compact += 1
        elif has_fallback:
            fallback_compact += 1
            fallback_by_generator[generator_id] = fallback_by_generator.get(generator_id, 0) + 1
        elif has_legacy:
            legacy_reconstructable += 1
            legacy_by_generator[generator_id] = legacy_by_generator.get(generator_id, 0) + 1
        if has_any:
            compact_available += 1
        if generator_id in _COMPACT_FACT_HIGH_RISK_GENERATORS:
            high_risk_rows += 1
            if not has_any:
                high_risk_missing += 1
                missing_by_generator[generator_id] = missing_by_generator.get(generator_id, 0) + 1

    result = {
        "seed_rows": len(rows),
        "explicit_source_compact": explicit_compact,
        "fallback_generation_param_compact": fallback_compact,
        "legacy_reconstructable_compact": legacy_reconstructable,
        "compact_available": compact_available,
        "compact_missing": len(rows) - compact_available,
        "high_risk_rows": high_risk_rows,
        "high_risk_missing_compact": high_risk_missing,
        "missing_compact_by_generator": dict(sorted(missing_by_generator.items())),
        "fallback_compact_by_generator": dict(sorted(fallback_by_generator.items())),
        "legacy_reconstructable_by_generator": dict(sorted(legacy_by_generator.items())),
        "report_path": str(reports_dir / "seed_input_preflight.md"),
    }
    (reports_dir / "seed_input_preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_text(reports_dir / "seed_input_preflight.md", _render_seed_input_preflight_report(result, seed_input))
    return result


def _row_has_explicit_source_compact(row: dict[str, Any]) -> bool:
    source_compact = row.get("source_compact_data")
    return isinstance(source_compact, dict) and source_compact.get("available") is True


def _row_has_generation_param_compact(row: dict[str, Any]) -> bool:
    generation_params = row.get("generation_params")
    return isinstance(generation_params, dict) and any(str(key).startswith("compact_") for key in generation_params)


def _row_has_legacy_reconstructable_compact(row: dict[str, Any]) -> bool:
    compact = _backtranslation_compact_source_data(row)
    return isinstance(compact, dict) and compact.get("available") is True


def _render_seed_input_preflight_report(result: dict[str, Any], seed_input: Path) -> str:
    lines = [
        "# Seed Input Preflight",
        "",
        f"- Seed input: `{seed_input}`",
        f"- Rows inspected: `{result.get('seed_rows', 0)}`",
        f"- Explicit `source_compact_data`: `{result.get('explicit_source_compact', 0)}`",
        f"- Fallback compact facts in `generation_params`: `{result.get('fallback_generation_param_compact', 0)}`",
        f"- Legacy reconstructable compact facts: `{result.get('legacy_reconstructable_compact', 0)}`",
        f"- Any compact facts available: `{result.get('compact_available', 0)}`",
        f"- Compact facts missing: `{result.get('compact_missing', 0)}`",
        f"- High-risk rows inspected: `{result.get('high_risk_rows', 0)}`",
        f"- High-risk rows missing compact facts: `{result.get('high_risk_missing_compact', 0)}`",
        "",
        "## Missing Compact Facts By Generator",
        "",
        "| generator_id | count |",
        "|---|---:|",
    ]
    missing = result.get("missing_compact_by_generator") or {}
    if missing:
        for generator_id, count in missing.items():
            lines.append(f"| `{generator_id}` | {count} |")
    else:
        lines.append("| n/a | 0 |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `explicit source_compact_data` is preferred for new seed-factory runs.",
            "- `fallback generation_params` is acceptable, but indicates the seed artifact is using the older implicit handoff.",
            "- `legacy reconstructable` means 05 can rebuild compact facts from older generator fields, but a fresh seed-factory run is still cleaner.",
            "- High-risk rows with no compact facts are allowed to continue, but they are more likely to drift in 05/07 and should usually be regenerated with the current seed factory.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_cpt_factory_stages(
    *,
    config,
    run_dir: Path,
    stage_dirs: dict[str, Path],
    seed_input: Path,
    seed_manifest: Path | None,
    limit: int | None,
    mock_llm: bool,
    llm_concurrency: int | None,
    auto_optimize_prompts: bool,
    auto_promote_prompts: bool,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    stage_results: dict[str, dict[str, Any]] = {}
    _record_seed_lineage(run_dir, seed_manifest)

    logger.info("stage=05_backtranslation status=started seed_input=%s", seed_input)
    bt_result = backtranslate_instances(
        seed_input,
        stage_dirs["backtranslation"],
        config,
        limit=limit,
        mock=mock_llm,
        concurrency_per_endpoint=llm_concurrency,
    )
    stage_results["backtranslation"] = bt_result
    logger.info("stage=05_backtranslation status=completed candidates=%s rejected=%s", bt_result.get("candidates"), bt_result.get("rejected"))

    logger.info("stage=06_nl_quality_filter status=started")
    bt_thresholds = config.quality_thresholds.backtranslation
    nl_result = filter_nl_candidates(
        stage_dirs["backtranslation"] / "backtranslation_candidates.jsonl",
        stage_dirs["nl_filter"],
        min_chars=int(bt_thresholds.get("min_chars", 160)),
        max_chars=int(bt_thresholds.get("max_chars", 4000)),
        min_number_coverage=float(bt_thresholds.get("min_number_coverage", 0.4)),
    )
    stage_results["nl_filter"] = nl_result
    logger.info("stage=06_nl_quality_filter status=completed accepted=%s rejected=%s", nl_result.get("accepted"), nl_result.get("rejected"))

    logger.info("stage=07_forward_modeling status=started")
    fm_result = forward_model_candidates(
        stage_dirs["nl_filter"] / "nl_validated_candidates.jsonl",
        stage_dirs["forward_modeling"],
        config,
        mock=mock_llm,
        concurrency_per_endpoint=llm_concurrency,
    )
    stage_results["forward_modeling"] = fm_result
    logger.info("stage=07_forward_modeling status=completed outputs=%s rejected=%s", fm_result.get("outputs"), fm_result.get("rejected"))

    logger.info("stage=08_forward_eval status=started")
    eval_thresholds = config.quality_thresholds.forward_modeling_eval
    fe_result = evaluate_forward_outputs(
        stage_dirs["forward_modeling"] / "forward_modeling_outputs.jsonl",
        stage_dirs["forward_eval"],
        timeout_seconds=int(config.generation_plan.filters.get("solve_timeout_sec", 60)),
        abs_tolerance=float(eval_thresholds.get("objective_abs_tolerance", 1e-4)),
        rel_tolerance=float(eval_thresholds.get("objective_rel_tolerance", 1e-4)),
    )
    stage_results["forward_eval"] = fe_result
    logger.info("stage=08_forward_eval status=completed accepted=%s rejected=%s", fe_result.get("accepted"), fe_result.get("rejected"))

    logger.info("stage=09_cpt_rendering status=started")
    render_result = render_cpt_documents(stage_dirs["forward_eval"] / "accepted_pairs.jsonl", stage_dirs["rendering"], config)
    stage_results["rendering"] = render_result
    logger.info("stage=09_cpt_rendering status=completed documents=%s rejected=%s", render_result.get("documents"), render_result.get("rejected"))

    logger.info("stage=10_train_export status=started mode=train_only")
    export_result = export_train_val_test(stage_dirs["rendering"] / "cpt_documents.jsonl", stage_dirs["export"], config)
    stage_results["export"] = export_result
    logger.info(
        "stage=10_train_export status=completed train=%s duplicates_removed=%s",
        export_result.get("train"),
        export_result.get("duplicates_removed"),
    )

    if auto_optimize_prompts:
        logger.info("stage=prompt_optimization status=started")
        optimize_result = optimize_prompts_for_run(
            run_dir,
            config,
            auto_promote=auto_promote_prompts,
            mock=mock_llm,
            concurrency_per_endpoint=llm_concurrency,
        )
        stage_results["prompt_optimization"] = optimize_result
        logger.info(
            "stage=prompt_optimization status=%s target_prompt=%s",
            optimize_result.get("status"),
            optimize_result.get("target_prompt") or (optimize_result.get("diagnostics") or {}).get("target_prompt"),
        )
        typer.echo(
            "post-run prompt optimization "
            f"status={optimize_result.get('status')} "
            f"target_prompt={optimize_result.get('target_prompt') or (optimize_result.get('diagnostics') or {}).get('target_prompt')}"
        )

    return stage_results


def _stage_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        "bootstrap": run_dir / "00_bootstrap",
        "registry": run_dir / "01_generator_registry",
        "smoke": run_dir / "02_adapter_smoke_test",
        "instances": run_dir / "03_instance_generation",
        "solver": run_dir / "04_solver_validation",
        "instance_quality": run_dir / "04b_instance_quality",
        "backtranslation": run_dir / "05_backtranslation",
        "nl_filter": run_dir / "06_nl_quality_filter",
        "forward_modeling": run_dir / "07_forward_modeling",
        "forward_eval": run_dir / "08_forward_eval",
        "rendering": run_dir / "09_cpt_rendering",
        "export": run_dir / "10_train_val_export",
    }


def _parse_generator_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def _parse_difficulty_mix(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    result: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise typer.BadParameter(f"difficulty mix item must look like level_1=0.5: {item}")
        key, raw_number = item.split("=", 1)
        result[key.strip()] = float(raw_number.strip())
    return result or None


def _resolve_solver_concurrency(config, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    configured = config.generation_plan.filters.get("solver_concurrency")
    if configured is not None:
        return max(1, int(configured))
    return max(1, int(config.generation_plan.generation.get("num_workers", 1)))


def _resolve_solver_threads_per_process(config, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    configured = config.generation_plan.filters.get("solver_threads_per_process")
    if configured is not None:
        return max(1, int(configured))
    return 1


def _resolve_per_generator_counts(
    *,
    smoke_report_path: Path,
    selected_generator_ids: set[str] | None,
    production_plan_path: Path | None,
    num_instances: int | None,
    yaml_per_gen: int,
) -> tuple[dict[str, int], dict[str, Any] | None]:
    if production_plan_path is None:
        per_gen_count = _build_per_generator_count(
            smoke_report_path=smoke_report_path,
            selected_generator_ids=selected_generator_ids,
            num_instances=num_instances,
            yaml_per_gen=yaml_per_gen,
        )
        return per_gen_count, None

    smoke_pass_ids = _load_smoke_pass_ids(smoke_report_path)
    selected_smoke_pass_ids = smoke_pass_ids
    if selected_generator_ids:
        selected_smoke_pass_ids = {gid for gid in smoke_pass_ids if gid in selected_generator_ids}

    plan_rows = _load_production_plan_rows(production_plan_path)
    original_plan_counts: dict[str, int] = {}
    for row in plan_rows:
        generator_id = str(row.get("generator_id") or "")
        if not generator_id:
            continue
        count = _as_nonnegative_int(row.get("planned_attempts"))
        if count <= 0:
            count = _as_nonnegative_int(row.get("target_accepted"))
        if count > 0:
            original_plan_counts[generator_id] = count

    per_gen_count = _build_per_generator_count_from_plan(
        smoke_report_path=smoke_report_path,
        selected_generator_ids=selected_generator_ids,
        production_plan_path=production_plan_path,
        num_instances=num_instances,
    )
    skipped_generator_ids = sorted(gid for gid in original_plan_counts if gid not in selected_smoke_pass_ids)
    filtered_generator_ids = sorted(
        gid for gid in original_plan_counts if selected_generator_ids and gid not in selected_generator_ids
    )
    zero_after_scaling_ids = sorted(
        gid for gid, original_count in original_plan_counts.items()
        if original_count > 0 and gid in per_gen_count and per_gen_count.get(gid, 0) == 0
    )

    usage = {
        "source_production_plan": str(production_plan_path),
        "requested_num_instances": num_instances,
        "plan_scaled": num_instances is not None,
        "planned_attempts_original": sum(original_plan_counts.values()),
        "planned_attempts_used": sum(per_gen_count.values()),
        "generator_count_original": len(original_plan_counts),
        "generator_count_used": sum(1 for count in per_gen_count.values() if count > 0),
        "selected_generator_ids": sorted(selected_generator_ids) if selected_generator_ids else [],
        "filtered_generator_ids": filtered_generator_ids,
        "skipped_generator_ids": skipped_generator_ids,
        "zero_after_scaling_generator_ids": zero_after_scaling_ids,
        "per_generator_counts": dict(sorted(per_gen_count.items())),
    }
    return per_gen_count, usage


def _record_production_plan_usage(run_dir: Path, production_plan_path: Path | None, usage: dict[str, Any] | None) -> Path | None:
    if production_plan_path is None or usage is None:
        return None
    output_dir = run_dir / "reports" / "production_plan_used"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(production_plan_path, output_dir / "production_plan.json")
    candidate_md = production_plan_path.with_suffix(".md")
    if candidate_md.exists():
        shutil.copy2(candidate_md, output_dir / "production_plan.md")
    _write_json(output_dir / "production_plan_usage.json", usage)
    write_text(output_dir / "production_plan_usage.md", _render_production_plan_usage(usage))
    return output_dir / "production_plan_usage.json"


def _render_production_plan_usage(usage: dict[str, Any]) -> str:
    lines = [
        "# Production Plan Usage",
        "",
        f"- Source plan: `{usage.get('source_production_plan')}`",
        f"- Requested num instances: `{usage.get('requested_num_instances')}`",
        f"- Plan scaled: `{usage.get('plan_scaled')}`",
        f"- Original planned attempts: `{usage.get('planned_attempts_original')}`",
        f"- Used planned attempts: `{usage.get('planned_attempts_used')}`",
        f"- Original generator count: `{usage.get('generator_count_original')}`",
        f"- Used generator count: `{usage.get('generator_count_used')}`",
        "",
        "## Warnings",
        "",
        f"- Filtered by `--generator-ids`: {', '.join(usage.get('filtered_generator_ids') or []) or 'none'}",
        f"- Skipped because not smoke-pass / unavailable: {', '.join(usage.get('skipped_generator_ids') or []) or 'none'}",
        f"- Zero after scaling: {', '.join(usage.get('zero_after_scaling_generator_ids') or []) or 'none'}",
        "",
        "## Per-Generator Counts",
        "",
        "| generator_id | count |",
        "| --- | ---: |",
    ]
    for generator_id, count in (usage.get("per_generator_counts") or {}).items():
        lines.append(f"| `{generator_id}` | {count} |")
    return "\n".join(lines) + "\n"


def _write_seed_manifest(
    *,
    run_dir: Path,
    run_id: str,
    config,
    stage_results: dict[str, Any],
    per_generator_counts: dict[str, int],
    production_plan_usage: dict[str, Any] | None,
    wall_seconds: float,
) -> Path:
    seed_output = run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl"
    manifest = {
        "manifest_type": "or_cpt_seed_manifest",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "seed_output": str(seed_output),
        "seed_count": count_jsonl_records(seed_output),
        "review_count": count_jsonl_records(run_dir / "04b_instance_quality" / "quality_review_instances.jsonl"),
        "rejected_count": count_jsonl_records(run_dir / "04b_instance_quality" / "quality_rejected_instances.jsonl"),
        "generator_profiles_path": str(config.paths.generator_profiles_path),
        "difficulty_mix": config.generation_plan.difficulty_mix,
        "per_generator_counts": dict(sorted(per_generator_counts.items())),
        "production_plan_usage": production_plan_usage,
        "stage_results": stage_results,
        "wall_seconds": round(wall_seconds, 3),
    }
    manifest_path = run_dir / "reports" / "seed_manifest.json"
    _write_json(manifest_path, manifest)
    write_text(run_dir / "reports" / "seed_manifest.md", _render_seed_manifest(manifest))
    return manifest_path


def _render_seed_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        "# Seed Manifest",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Seed output: `{manifest['seed_output']}`",
        f"- Seed count: `{manifest['seed_count']}`",
        f"- Review count: `{manifest['review_count']}`",
        f"- Rejected count: `{manifest['rejected_count']}`",
        f"- Wall seconds: `{manifest['wall_seconds']}`",
        "",
        "## Production Plan",
        "",
    ]
    usage = manifest.get("production_plan_usage")
    if usage:
        lines.extend(
            [
                f"- Source plan: `{usage.get('source_production_plan')}`",
                f"- Plan scaled: `{usage.get('plan_scaled')}`",
                f"- Used attempts: `{usage.get('planned_attempts_used')}`",
            ]
        )
    else:
        lines.append("- Source plan: none; counts came from CLI/YAML defaults.")
    lines.extend(["", "## Stage Results", ""])
    for name, result in (manifest.get("stage_results") or {}).items():
        lines.append(f"- `{name}`: `{json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)}`")
    return "\n".join(lines) + "\n"


def _record_seed_lineage(run_dir: Path, seed_manifest: Path | None) -> None:
    if seed_manifest is None:
        return
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if seed_manifest.exists():
        shutil.copy2(seed_manifest, reports_dir / "seed_manifest_used.json")


def _write_cpt_run_manifest(
    *,
    run_dir: Path,
    run_id: str,
    seed_input: Path,
    seed_manifest: Path | None,
    stage_results: dict[str, Any],
    wall_seconds: float,
) -> Path:
    export_dir = run_dir / "10_train_val_export"
    manifest = {
        "manifest_type": "or_cpt_cpt_run_manifest",
        "export_mode": "train_only_streaming",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "seed_input": str(seed_input),
        "seed_manifest": str(seed_manifest) if seed_manifest else None,
        "seed_input_count": count_jsonl_records(seed_input),
        "train_count": count_jsonl_records(export_dir / "train.jsonl"),
        "export_manifest": str(export_dir / "manifest.json"),
        "stage_results": stage_results,
        "wall_seconds": round(wall_seconds, 3),
    }
    manifest_path = run_dir / "reports" / "cpt_run_manifest.json"
    _write_json(manifest_path, manifest)
    write_text(run_dir / "reports" / "cpt_run_manifest.md", _render_cpt_run_manifest(manifest))
    return manifest_path


def _render_cpt_run_manifest(manifest: dict[str, Any]) -> str:
    lines = [
        "# CPT Run Manifest",
        "",
        f"- Run id: `{manifest['run_id']}`",
        f"- Seed input: `{manifest['seed_input']}`",
        f"- Seed manifest: `{manifest.get('seed_manifest') or 'none'}`",
        f"- Seed input count: `{manifest['seed_input_count']}`",
        f"- Export mode: `{manifest.get('export_mode', 'train_only_streaming')}`",
        f"- Train count: `{manifest['train_count']}`",
        f"- Wall seconds: `{manifest['wall_seconds']}`",
        "",
        "## Stage Results",
        "",
    ]
    for name, result in (manifest.get("stage_results") or {}).items():
        lines.append(f"- `{name}`: `{json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)}`")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _load_smoke_pass_ids(smoke_report_path: Path) -> set[str]:
    return {
        row["generator_id"]
        for row in read_jsonl(smoke_report_path)
        if row.get("load_status") == "PASS" and row.get("generate_status") == "PASS"
    }


def _build_per_generator_count(
    *,
    smoke_report_path: Path,
    selected_generator_ids: set[str] | None,
    num_instances: int | None,
    yaml_per_gen: int,
) -> dict[str, int]:
    """Build a ``{generator_id: count}`` mapping for instance generation.

    When ``--num-instances`` is given on the CLI it is evenly distributed across
    smoke-pass generators (remainder assigned to the first generators in sorted
    order).  Otherwise every smoke-pass generator gets the YAML default.
    """
    smoke_pass_ids: list[str] = sorted(
        row["generator_id"]
        for row in read_jsonl(smoke_report_path)
        if row.get("load_status") == "PASS" and row.get("generate_status") == "PASS"
    )
    if selected_generator_ids:
        smoke_pass_ids = [gid for gid in smoke_pass_ids if gid in selected_generator_ids]

    if not smoke_pass_ids:
        return {}

    if num_instances is not None:
        base = num_instances // len(smoke_pass_ids)
        remainder = num_instances % len(smoke_pass_ids)
    else:
        base = yaml_per_gen
        remainder = 0

    per_gen: dict[str, int] = {}
    for idx, gid in enumerate(smoke_pass_ids):
        per_gen[gid] = base + (1 if idx < remainder else 0)
    return per_gen


def _build_per_generator_count_from_plan(
    *,
    smoke_report_path: Path,
    selected_generator_ids: set[str] | None,
    production_plan_path: Path,
    num_instances: int | None,
) -> dict[str, int]:
    """Build per-generator counts from a production plan.

    If ``--num-instances`` is provided, the planned attempts are scaled down/up
    proportionally while preserving the requested total. This makes small pilot
    runs follow the same generator distribution as a larger production plan.
    """
    smoke_pass_ids = {
        row["generator_id"]
        for row in read_jsonl(smoke_report_path)
        if row.get("load_status") == "PASS" and row.get("generate_status") == "PASS"
    }
    if selected_generator_ids:
        smoke_pass_ids = {gid for gid in smoke_pass_ids if gid in selected_generator_ids}

    plan_rows = _load_production_plan_rows(production_plan_path)
    plan_counts: dict[str, int] = {}
    for row in plan_rows:
        generator_id = str(row.get("generator_id") or "")
        if not generator_id or generator_id not in smoke_pass_ids:
            continue
        count = _as_nonnegative_int(row.get("planned_attempts"))
        if count <= 0:
            count = _as_nonnegative_int(row.get("target_accepted"))
        if count > 0:
            plan_counts[generator_id] = count

    if not plan_counts:
        raise typer.BadParameter(
            f"production plan has no positive planned_attempts matching smoke-passing generators: {production_plan_path}"
        )

    if num_instances is None:
        return dict(sorted(plan_counts.items()))
    if num_instances < 0:
        raise typer.BadParameter("--num-instances must be non-negative")
    if num_instances == 0:
        return {gid: 0 for gid in sorted(plan_counts)}
    return _scale_counts_to_total(plan_counts, num_instances)


def _load_production_plan_rows(production_plan_path: Path) -> list[dict]:
    if not production_plan_path.exists():
        raise FileNotFoundError(f"production plan not found: {production_plan_path}")
    payload = json.loads(production_plan_path.read_text(encoding="utf-8"))
    rows = payload.get("generators") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise typer.BadParameter(f"production plan must contain a generators list: {production_plan_path}")
    return [row for row in rows if isinstance(row, dict)]


def _scale_counts_to_total(counts: dict[str, int], target_total: int) -> dict[str, int]:
    total = sum(max(0, int(value)) for value in counts.values())
    if total <= 0:
        raise typer.BadParameter("cannot scale production plan with zero total planned attempts")
    scaled: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    allocated = 0
    for generator_id, count in sorted(counts.items()):
        exact = target_total * max(0, int(count)) / total
        base = int(exact)
        scaled[generator_id] = base
        allocated += base
        fractions.append((exact - base, generator_id))
    remainder = target_total - allocated
    for _, generator_id in sorted(fractions, key=lambda item: (-item[0], item[1]))[:remainder]:
        scaled[generator_id] += 1
    return scaled


def _as_nonnegative_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    app()
