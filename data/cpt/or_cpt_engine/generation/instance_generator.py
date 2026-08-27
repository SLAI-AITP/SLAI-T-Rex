from __future__ import annotations

from pathlib import Path
from typing import Any

from or_cpt_engine.adapters.adapter_factory import build_adapter
from or_cpt_engine.adapters.optmath_adapter import AdapterUnsupportedError
from or_cpt_engine.generation.generator_profiles import (
    build_difficulty_config,
    load_generator_profiles,
    select_difficulty_level,
)
from or_cpt_engine.schemas.common import AdapterSmokeResult, InstanceRecord
from or_cpt_engine.utils.io import append_jsonl, read_jsonl, write_jsonl, write_text


def smoke_test_generators(
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    difficulty_level: str = "level_1",
    limit: int | None = None,
    extract_model_info: bool = False,
    generator_ids: set[str] | None = None,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(registry_path)
    if generator_ids:
        records = [record for record in records if record.get("generator_id") in generator_ids]
    if limit is not None:
        records = records[:limit]
    results: list[AdapterSmokeResult] = []

    for record in records:
        generator_id = record["generator_id"]
        started_errors: list[str] = []
        load_status = "FAIL"
        generate_status = "FAIL"
        model_info: dict[str, Any] = {}
        try:
            adapter = build_adapter(record)
            adapter.load()
            load_status = "PASS"
            raw = adapter.generate(seed, {"difficulty_level": difficulty_level})
            if extract_model_info:
                model_info = adapter.extract_model_info(raw)
            generate_status = "PASS"
        except AdapterUnsupportedError as exc:
            started_errors.append(f"ADAPTER_UNSUPPORTED: {exc}")
        except Exception as exc:  # noqa: BLE001
            started_errors.append(f"{exc.__class__.__name__}: {exc}")

        results.append(
            AdapterSmokeResult(
                generator_id=generator_id,
                load_status=load_status,
                generate_status=generate_status,
                seed=seed,
                difficulty_level=difficulty_level,
                num_variables=model_info.get("num_variables"),
                num_constraints=model_info.get("num_constraints"),
                solver_status=model_info.get("initial_solver_status"),
                objective_value=model_info.get("initial_objective_value"),
                errors=started_errors,
            )
        )

    write_jsonl(output / "adapter_smoke_report.jsonl", results)
    write_text(output / "adapter_smoke_summary.md", render_smoke_summary(results))
    return {
        "total": len(results),
        "passed": sum(1 for item in results if item.load_status == "PASS" and item.generate_status == "PASS"),
        "failed": sum(1 for item in results if item.generate_status != "PASS"),
    }


def render_smoke_summary(results: list[AdapterSmokeResult]) -> str:
    passed = sum(1 for item in results if item.load_status == "PASS" and item.generate_status == "PASS")
    lines = [
        "# Adapter Smoke Summary",
        "",
        f"- Total generators: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        "",
        "| generator_id | load | generate | errors |",
        "|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item.generator_id} | {item.load_status} | {item.generate_status} | {'; '.join(item.errors)} |"
        )
    return "\n".join(lines) + "\n"


def generate_instances(
    registry_path: str | Path,
    smoke_report_path: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    num_instances_per_generator: int | dict[str, int] = 1,
    limit: int | None = None,
    seed_start: int = 42,
    generator_ids: set[str] | None = None,
    profiles_path: str | Path | None = None,
    difficulty_mix: dict[str, float] | None = None,
    extract_solve_timeout_sec: int = 60,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registry_by_id = {row["generator_id"]: row for row in read_jsonl(registry_path)}
    smoke_pass_ids = {
        row["generator_id"]
        for row in read_jsonl(smoke_report_path)
        if row.get("load_status") == "PASS" and row.get("generate_status") == "PASS"
    }
    if generator_ids:
        smoke_pass_ids = {generator_id for generator_id in smoke_pass_ids if generator_id in generator_ids}
    profile_payload = load_generator_profiles(profiles_path)
    profile_version = str(profile_payload.get("version") or "none")
    profiles = profile_payload.get("profiles") or {}

    # Resolve per-generator count: dict takes precedence, int is broadcast.
    if isinstance(num_instances_per_generator, dict):
        per_gen_count = num_instances_per_generator
    else:
        per_gen_count = {gid: num_instances_per_generator for gid in smoke_pass_ids}

    # Streaming: write each instance immediately to disk via append.
    raw_path = output / "instances_raw.jsonl"
    accepted_path = output / "instances_generated.jsonl"
    rejected_path = output / "instances_generation_rejected.jsonl"
    difficulty_counts: dict[str, int] = {}
    raw_count = 0
    accepted_count = 0
    rejected_count = 0
    produced = 0

    for generator_id in sorted(smoke_pass_ids):
        if limit is not None and produced >= limit:
            break
        count = per_gen_count.get(generator_id, 0)
        if count <= 0:
            continue
        profile = profiles.get(generator_id) if isinstance(profiles, dict) else None
        if isinstance(profile, dict) and profile.get("enabled") is False:
            continue
        record = registry_by_id[generator_id]
        adapter = build_adapter(record)
        try:
            adapter.load()
        except Exception as exc:  # noqa: BLE001
            append_jsonl(rejected_path, {
                "generator_id": generator_id,
                "generation_status": "REJECTED",
                "rejection_stage": "instance_generation",
                "rejection_reason": f"ADAPTER_LOAD_ERROR: {exc}",
            })
            rejected_count += 1
            continue

        for local_index in range(count):
            if limit is not None and produced >= limit:
                break
            seed = seed_start + produced
            instance_id = f"inst_{generator_id}_{produced + 1:08d}"
            try:
                profile_dict = profile if isinstance(profile, dict) else {}
                difficulty_level = select_difficulty_level(
                    generator_id=generator_id,
                    seed=seed,
                    difficulty_mix=difficulty_mix,
                    profile=profile,
                )
                difficulty_config = build_difficulty_config(profile, difficulty_level, seed)
                raw = adapter.generate(seed, difficulty_config)
                info = adapter.extract_model_info(raw, solve_timeout_sec=extract_solve_timeout_sec)
                generation_params = dict(info.pop("generation_params") or {})
                generation_params.update(
                    {
                        "difficulty_config": difficulty_config,
                        "profile_priority": profile_dict.get("priority"),
                    }
                )
                source_compact_data = _source_compact_data_from_generation_params(generation_params)
                instance = InstanceRecord(
                    instance_id=instance_id,
                    run_id=run_id,
                    generator_id=generator_id,
                    generator_name=record.get("name"),
                    seed=seed,
                    difficulty_level=difficulty_level,
                    generator_profile_version=profile_version if profile else None,
                    sub_family=profile_dict.get("sub_family"),
                    canonical_math_signature=profile_dict.get("canonical_math_signature") or {},
                    concept_tags=list(profile_dict.get("modeling_concepts") or []),
                    variant_id=profile_dict.get("default_variant_id"),
                    generation_params=generation_params,
                    source_compact_data=source_compact_data,
                    **info,
                )
                append_jsonl(raw_path, instance)
                raw_count += 1
                difficulty_counts[difficulty_level] = difficulty_counts.get(difficulty_level, 0) + 1
                if instance.gurobi_code or instance.lp_text or instance.initial_objective_value is not None:
                    append_jsonl(accepted_path, instance)
                    accepted_count += 1
                else:
                    append_jsonl(rejected_path, {
                        **instance.model_dump(mode="json"),
                        "generation_status": "REJECTED",
                        "rejection_stage": "instance_generation",
                        "rejection_reason": "MISSING_MODEL_ARTIFACTS",
                    })
                    rejected_count += 1
            except Exception as exc:  # noqa: BLE001
                append_jsonl(rejected_path, {
                    "instance_id": instance_id,
                    "generator_id": generator_id,
                    "seed": seed,
                    "generation_status": "REJECTED",
                    "rejection_stage": "instance_generation",
                    "rejection_reason": f"{exc.__class__.__name__}: {exc}",
                })
                rejected_count += 1
            produced += 1

    write_text(output / "instance_generation_report.md",
               render_generation_report(raw_count, accepted_count, rejected_count, difficulty_counts))
    return {"raw": raw_count, "generated": accepted_count, "rejected": rejected_count}


def _source_compact_data_from_generation_params(generation_params: dict[str, Any]) -> dict[str, Any]:
    """Expose compact generator facts explicitly in seed artifacts.

    Downstream LLM stages treat these compact tables as the authoritative
    source facts. Keeping them as a first-class field avoids relying on each
    later stage to rediscover them from the much larger generation_params blob.
    """
    if not isinstance(generation_params, dict):
        return {"available": False}
    selected = {
        str(key): value
        for key, value in generation_params.items()
        if str(key).startswith("compact_")
    }
    if not selected:
        return {"available": False}
    return {"available": True, "truncated": False, "value": selected}


def render_generation_report(
    raw_count: int,
    accepted_count: int,
    rejected_count: int,
    difficulty_counts: dict[str, int],
) -> str:
    return "\n".join(
        [
            "# Instance Generation Report",
            "",
            f"- Raw generation attempts: {raw_count}",
            f"- Generated instances: {accepted_count}",
            f"- Rejected instances: {rejected_count}",
            "",
            "## Difficulty Distribution",
            "",
            "| difficulty | count |",
            "|---|---:|",
            *[
                f"| {difficulty} | {count} |"
                for difficulty, count in sorted(difficulty_counts.items())
            ],
            "",
        ]
    )
