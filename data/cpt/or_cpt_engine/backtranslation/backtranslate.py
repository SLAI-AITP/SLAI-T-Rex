from __future__ import annotations

import asyncio
import hashlib
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from or_cpt_engine.generation.generator_profiles import load_generator_profiles
from or_cpt_engine.llm.openai_compatible_client import (
    AsyncOpenAICompatibleClient,
    extract_json_object,
    is_retryable_llm_error,
)
from or_cpt_engine.schemas.common import BacktranslationCandidate, EngineConfig, LLMStageConfig
from or_cpt_engine.backtranslation.scenario_catalog import scenario_guidance_for_generator
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text
from or_cpt_engine.utils.prompt_registry import PromptSpec, load_default_prompt_registry


PROMPT_NAME = "backtranslation_prompt"
COMPACT_PROMPT_NAME = "backtranslation_compact_prompt"
AIRCRAFT_PROMPT_NAME = "backtranslation_aircraft_prompt"
TRANSPORTATION_PROMPT_NAME = "backtranslation_transportation_prompt"
CFLP_PROMPT_NAME = "backtranslation_cflp_prompt"
REVENUE_MANAGEMENT_PROMPT_NAME = "backtranslation_revenue_management_prompt"
CAR_SELECTION_PROMPT_NAME = "backtranslation_car_selection_prompt"
CELL_TOWER_PROMPT_NAME = "backtranslation_cell_tower_prompt"
SOURCE_FACT_PROMPT_NAME = "backtranslation_source_fact_prompt"
SOURCE_FACT_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_clsp_expand_capacity",
    "optmath_uncapacitatedlotsizing",
    "uncapacitatedlotsizingbacklogging",
    "optmath_uncapacitatedlotsizingbacklogging",
    "optmath_net1",
    "net1",
    "optmath_structure_based_assignment",
    "structure_based_assignment",
    "optmath_steel4",
    "steel4",
    "optmath_steel3",
    "steel3",
    "optmath_electrical_power",
    "electrical_power",
    "optmath_marketshare",
    "marketshare",
    "optmath_singlelevelsmallbucket",
    "singlelevelsmallbucket",
)
AIRCRAFT_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "aircraftassignment",
    "aircraftlanding",
)
TRANSPORTATION_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_transp",
)
CFLP_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_cflp",
)
REVENUE_MANAGEMENT_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_revenue_management",
    "optmath_revenue",
)
CAR_SELECTION_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_carselection",
)
CELL_TOWER_BACKTRANSLATION_GENERATOR_PATTERNS = (
    "optmath_cell_tower",
)
# Keep the broad compact-prompt router disabled by default. In cpt_factory_1kv2_v2
# it caused many generators to emit structure summaries without numeric tables,
# which then failed NL number-coverage checks. Re-enable per generator only after
# adding deterministic table-preservation tests or statement-rewrite fallbacks.
COMPACT_BACKTRANSLATION_GENERATOR_PATTERNS: tuple[str, ...] = ()
_MAX_BACKTRANSLATION_PROMPT_CHARS = 20_000
_TARGET_BACKTRANSLATION_PROMPT_CHARS = 16_000
_SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS = 18_500
_MAX_BACKTRANSLATION_QUALITY_ATTEMPTS = 2
_MAX_GENERATION_PARAMS_CHARS = 8_000
_MAX_LP_PREVIEW_CHARS = 3_000
_MAX_CODE_PREVIEW_CHARS = 1_500
_EMERGENCY_BACKTRANSLATION_TEMPLATE = """You are converting a verified optimization seed into a concise English business problem.

Return exactly one JSON object with:
- `problem_background`: one short sentence with organization, trigger, and decision owner.
- `problem_statement`: a self-contained problem statement.
- `structured_problem_data`: compact sets, parameters, and units used in the statement.

Rules:
- Preserve the source structure in `generator_profile.canonical_math_signature`.
- Use `source_artifacts.compact_source_data` or `source_artifacts.generation_params` as the numeric source.
- Do not invent missing values. If a table is truncated, state only available values and dimensions.
- Do not reveal solver status, objective value, solution values, LP/code text, or audit/training commentary.
- Keep the statement concise and business-grounded using `scenario_guidance`.

Emergency compact seed:

```json
{{INSTANCE_JSON}}
```
"""
_LAST_RESORT_BACKTRANSLATION_TEMPLATE = """Convert this verified optimization seed into one concise English OR business problem.

Return exactly one JSON object with `problem_background`, `problem_statement`, and `structured_problem_data`.
Use only the compact seed facts below. Do not mention solver status, objective value, source code, LP text, audit, or training data.

```json
{{INSTANCE_JSON}}
```
"""


def backtranslate_instances(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    limit: int | None = None,
    mock: bool = False,
    concurrency_per_endpoint: int | None = None,
) -> dict[str, int]:
    return asyncio.run(
        backtranslate_instances_async(
            input_path,
            output_dir,
            config,
            limit=limit,
            mock=mock,
            concurrency_per_endpoint=concurrency_per_endpoint,
        )
    )


async def backtranslate_instances_async(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    limit: int | None = None,
    mock: bool = False,
    concurrency_per_endpoint: int | None = None,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompt_registry = load_default_prompt_registry()

    candidate_path = output / "backtranslation_candidates.jsonl"
    rejected_path = output / "backtranslation_rejected.jsonl"
    raw_response_path = output / "backtranslation_raw_responses.jsonl"
    write_lock = asyncio.Lock()
    candidate_count = 0
    rejected_count = 0
    requeued_count = 0

    client = None if mock else AsyncOpenAICompatibleClient(
        config.llm,
        config.llm.backtranslation,
        concurrency_per_endpoint=concurrency_per_endpoint,
    )
    worker_count = 1 if mock else max(1, client.total_concurrency_capacity() if client is not None else 1)

    # Queue items: (row, candidate_index, requeue_count)
    # When an LLM call fails with a transient error the worker re-queues the item
    # instead of permanently rejecting it, mirroring cpt_cleaner's RequeueDirective.
    async def _worker(queue: asyncio.Queue) -> None:
        nonlocal candidate_count, rejected_count, requeued_count
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            row, candidate_index, requeue_count = item
            prompt_name, prompt_spec = select_backtranslation_prompt_spec(prompt_registry, row)
            result = await _backtranslate_one(
                row,
                candidate_index,
                prompt_spec,
                config,
                client=client,
                mock=mock,
                prompt_name=prompt_name,
            )
            if result["kind"] == "requeue":
                requeued_count += 1
                await queue.put((row, candidate_index, requeue_count + 1))
                queue.task_done()
                continue
            async with write_lock:
                if result["kind"] == "candidate":
                    append_jsonl(candidate_path, result["candidate"])
                    append_jsonl(raw_response_path, result["raw_response"])
                    candidate_count += 1
                else:
                    if result.get("raw_response") is not None:
                        append_jsonl(raw_response_path, result["raw_response"])
                    append_jsonl(rejected_path, result["rejected"])
                    rejected_count += 1
            queue.task_done()

    # Stream rows from input file, push work items directly into queue.
    queue: asyncio.Queue = asyncio.Queue()
    workers = [asyncio.create_task(_worker(queue)) for _ in range(worker_count)]
    produced = 0
    for row in (iter_jsonl(input_path) or []):
        if limit is not None and produced >= limit:
            break
        for candidate_index in range(1, config.llm.backtranslation.num_candidates + 1):
            await queue.put((row, candidate_index, 0))
            produced += 1

    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
    if client is not None:
        await client.close()

    write_text(output / "backtranslation_report.md",
               render_backtranslation_report(candidate_count, rejected_count, requeued_count, candidate_path))
    return {"candidates": candidate_count, "rejected": rejected_count}


async def _backtranslate_one(
    row: dict[str, Any],
    candidate_index: int,
    prompt_spec: PromptSpec,
    config: EngineConfig,
    *,
    client: AsyncOpenAICompatibleClient | None,
    mock: bool,
    stage_config: LLMStageConfig | None = None,
    prompt_name: str = PROMPT_NAME,
    _quality_attempt: int = 1,
    _quality_retry_reasons: list[str] | None = None,
    _quality_retry_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    row = _profile_normalized_row(row, config)
    quality_retry_reasons = list(_quality_retry_reasons or [])
    quality_retry_events = list(_quality_retry_events or [])
    bt_id = f"bt_{_short_hash(row['instance_id'], candidate_index)}"
    scenario_guidance = scenario_guidance_for_generator(
        row.get("generator_id"),
        instance_id=row.get("instance_id"),
        candidate_index=candidate_index,
        random_seed=config.project.random_seed,
        catalog_path=config.paths.scenario_catalog_path,
    )
    profile_normalization_metadata = {
        key: (row.get("llm_metadata") or {}).get(key)
        for key in ("profile_normalized", "profile_normalized_from", "profile_normalized_to")
        if (row.get("llm_metadata") or {}).get(key) is not None
    }
    raw_response: dict[str, Any] | None = None
    raw_content: Any = None
    llm_metadata: dict[str, Any] = {}
    try:
        if mock:
            payload = _mock_backtranslation(row)
            llm_metadata = {"mock": True}
            raw_content = payload
        else:
            if client is None:
                raise RuntimeError("async client is not initialized")
            prompt = _render_prompt(prompt_spec.text, row, candidate_index, scenario_guidance)
            prompt = _with_backtranslation_quality_retry_directive(prompt, quality_retry_reasons)
            response = await client.chat(
                prompt,
                log_context={"instance_id": row.get("instance_id", ""), "prompt_name": prompt_name},
                stage_config=stage_config,
            )
            llm_metadata = {
                "request_id": response["request_id"],
                "latency_sec": response["latency_sec"],
                "model": response["model"],
                "endpoint": response.get("endpoint"),
                "attempt": response["attempt"],
            }
            raw_content = response["content"]
            payload = extract_json_object(response["content"])
        llm_metadata.update(
            {
                "prompt_name": prompt_name,
                "prompt_version": prompt_spec.version,
                "prompt_hash": prompt_spec.hash,
                "quality_attempt": _quality_attempt,
                "quality_max_attempts": 1 if mock else _MAX_BACKTRANSLATION_QUALITY_ATTEMPTS,
                "quality_retries": len(quality_retry_reasons),
                "quality_retry_reasons": quality_retry_reasons,
                "quality_retry_events": quality_retry_events,
                "scenario_id": scenario_guidance.get("scenario_id"),
                "base_scenario_id": scenario_guidance.get("base_scenario_id"),
                "scenario_variant_id": scenario_guidance.get("scenario_variant_id"),
                "scenario_source": scenario_guidance.get("scenario_source"),
                "task_family": scenario_guidance.get("task_family"),
                "business_trigger": scenario_guidance.get("selected_business_trigger"),
                "industry_lens_id": (scenario_guidance.get("scenario_variant") or {}).get("industry_lens_id"),
                "narrative_angle_id": (scenario_guidance.get("scenario_variant") or {}).get("narrative_angle_id"),
                "organization_profile_id": (scenario_guidance.get("scenario_variant") or {}).get("organization_profile_id"),
                "planning_horizon_id": (scenario_guidance.get("scenario_variant") or {}).get("planning_horizon_id"),
                "entity_naming_style_id": (scenario_guidance.get("scenario_variant") or {}).get("entity_naming_style_id"),
                **profile_normalization_metadata,
            }
        )
        raw_response = {
            "bt_id": bt_id,
            "instance_id": row["instance_id"],
            "candidate_index": candidate_index,
            "raw_response": raw_content,
            "llm_metadata": llm_metadata,
            "scenario_guidance": scenario_guidance,
        }
        source_compact_data = _compact_source_data(row, max_chars=20_000)
        structured_problem_data = _payload_mapping(
            payload,
            "structured_problem_data",
            "structured_data",
            "problem_data",
            "instance_data",
            "data",
            "parameters",
        )
        background = _payload_text(
            payload,
            "problem_background",
            "background",
            "business_background",
            "business_context",
            "scenario_context",
            "context",
            "setting",
        )
        statement = _combine_background_and_statement(
            background,
            _payload_text(
                payload,
                "problem_statement",
                "problem",
                "problem_text",
                "problem_description",
                "business_problem",
                "question",
                "task",
                "natural_language_problem",
            ),
        )
        if not statement:
            reason = "ValueError: missing problem_statement"
            if _should_retry_backtranslation_generation(reason, mock=mock, quality_attempt=_quality_attempt):
                return await _backtranslate_one(
                    row,
                    candidate_index,
                    prompt_spec,
                    config,
                    client=client,
                    mock=mock,
                    stage_config=stage_config,
                    prompt_name=prompt_name,
                    _quality_attempt=_quality_attempt + 1,
                    _quality_retry_reasons=quality_retry_reasons + [reason],
                    _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
                )
            raise ValueError("missing problem_statement")
        statement = _ensure_required_problem_statement_tables(
            row,
            statement,
            source_compact_data=source_compact_data,
        )
        structured_problem_data = _structured_problem_data_from_compact_source(
            row,
            source_compact_data=source_compact_data,
            fallback=structured_problem_data,
        )
        candidate = BacktranslationCandidate(
            bt_id=bt_id,
            instance_id=row["instance_id"],
            generator_id=row.get("generator_id", ""),
            candidate_index=candidate_index,
            language=config.rendering.language,
            problem_background=background,
            problem_statement=statement,
            structured_problem_data=structured_problem_data,
            reference_answer=row.get("reference_answer") or {},
            source_lp_text=_artifact_preview(row.get("lp_text"), max_chars=_MAX_LP_PREVIEW_CHARS).get("preview"),
            source_math_formula=row.get("math_formula"),
            source_compact_data=source_compact_data,
            generator_profile_version=row.get("generator_profile_version"),
            difficulty_level=row.get("difficulty_level"),
            sub_family=row.get("sub_family"),
            canonical_math_signature=row.get("canonical_math_signature") or {},
            concept_tags=list(row.get("concept_tags") or []),
            variant_id=row.get("variant_id"),
            llm_metadata=llm_metadata,
        )
        return {"kind": "candidate", "candidate": candidate, "raw_response": raw_response}
    except Exception as exc:  # noqa: BLE001
        if is_retryable_llm_error(exc):
            return {"kind": "requeue"}
        reason = f"{exc.__class__.__name__}: {exc}"
        if _should_retry_backtranslation_generation(reason, mock=mock, quality_attempt=_quality_attempt):
            return await _backtranslate_one(
                row,
                candidate_index,
                prompt_spec,
                config,
                client=client,
                mock=mock,
                stage_config=stage_config,
                prompt_name=prompt_name,
                _quality_attempt=_quality_attempt + 1,
                _quality_retry_reasons=quality_retry_reasons + [reason],
                _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
            )
        rejection_metadata = dict(llm_metadata)
        rejection_metadata.setdefault("quality_attempt", _quality_attempt)
        rejection_metadata.setdefault("quality_max_attempts", 1 if mock else _MAX_BACKTRANSLATION_QUALITY_ATTEMPTS)
        rejection_metadata.setdefault("quality_retries", len(quality_retry_reasons))
        rejection_metadata.setdefault("quality_retry_reasons", quality_retry_reasons)
        rejection_metadata.setdefault("quality_retry_events", quality_retry_events)
        if raw_response is None and raw_content is not None:
            raw_response = {
                "bt_id": bt_id,
                "instance_id": row.get("instance_id"),
                "candidate_index": candidate_index,
                "raw_response": raw_content,
                "llm_metadata": rejection_metadata,
                "scenario_guidance": scenario_guidance,
            }
        return {
            "kind": "rejected",
            "raw_response": raw_response,
            "rejected": {
                "bt_id": bt_id,
                "instance_id": row.get("instance_id"),
                "generator_id": row.get("generator_id"),
                "candidate_index": candidate_index,
                "llm_metadata": rejection_metadata,
                "bt_status": "REJECTED",
                "rejection_stage": "backtranslation",
                "rejection_reason": reason,
            },
        }


def _should_retry_backtranslation_generation(reason: str, *, mock: bool, quality_attempt: int) -> bool:
    if mock or quality_attempt >= _MAX_BACKTRANSLATION_QUALITY_ATTEMPTS:
        return False
    lowered = reason.lower()
    if "backtranslation prompt exceeds safety limit" in lowered:
        return False
    retry_markers = (
        "missing problem_statement",
        "missing executable",
        "json",
        "extract",
        "parse",
        "schema",
        "missing required",
    )
    return any(marker in lowered for marker in retry_markers)


def _quality_retry_event(reason: str, llm_metadata: dict[str, Any], quality_attempt: int) -> dict[str, Any]:
    endpoint = llm_metadata.get("endpoint") if isinstance(llm_metadata, dict) else None
    if isinstance(endpoint, dict):
        endpoint_summary = {
            key: endpoint.get(key)
            for key in ("name", "base_url", "model")
            if endpoint.get(key) is not None
        }
    else:
        endpoint_summary = endpoint
    return {
        "quality_attempt": quality_attempt,
        "reason": reason,
        "request_id": llm_metadata.get("request_id") if isinstance(llm_metadata, dict) else None,
        "model": llm_metadata.get("model") if isinstance(llm_metadata, dict) else None,
        "endpoint": endpoint_summary,
    }


def _with_backtranslation_quality_retry_directive(prompt: str, retry_reasons: list[str]) -> str:
    reasons = [str(reason).strip() for reason in retry_reasons if str(reason).strip()]
    if not reasons:
        return prompt
    reason_summary = "; ".join(_short_text(reason, max_chars=180) for reason in reasons[-3:])
    directive = (
        "QUALITY RETRY DIRECTIVE:\n"
        f"The previous backtranslation response was rejected for: {reason_summary}.\n"
        "Return exactly one JSON object with non-empty string fields `problem_background` and `problem_statement`, "
        "plus `structured_problem_data` when useful. Do not return only tables or `structured_problem_data`. "
        "The `problem_statement` must be a self-contained natural-language OR business problem, not a math-only summary.\n"
    )
    candidate = f"{directive}\n{prompt}"
    if len(candidate) <= _MAX_BACKTRANSLATION_PROMPT_CHARS:
        return candidate
    short_directive = (
        "QUALITY RETRY DIRECTIVE: previous response was rejected. Return one JSON object with a non-empty "
        "`problem_statement`, `problem_background`, and optional `structured_problem_data`.\n"
    )
    candidate = f"{short_directive}\n{prompt}"
    if len(candidate) <= _MAX_BACKTRANSLATION_PROMPT_CHARS:
        return candidate
    return prompt


def select_backtranslation_prompt_spec(prompt_registry: Any, row: dict[str, Any]) -> tuple[str, PromptSpec]:
    generator_id = str(row.get("generator_id") or "").lower()
    if any(pattern in generator_id for pattern in SOURCE_FACT_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return SOURCE_FACT_PROMPT_NAME, prompt_registry.get(SOURCE_FACT_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in CFLP_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return CFLP_PROMPT_NAME, prompt_registry.get(CFLP_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in REVENUE_MANAGEMENT_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return REVENUE_MANAGEMENT_PROMPT_NAME, prompt_registry.get(REVENUE_MANAGEMENT_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in CAR_SELECTION_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return CAR_SELECTION_PROMPT_NAME, prompt_registry.get(CAR_SELECTION_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in CELL_TOWER_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return CELL_TOWER_PROMPT_NAME, prompt_registry.get(CELL_TOWER_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in TRANSPORTATION_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return TRANSPORTATION_PROMPT_NAME, prompt_registry.get(TRANSPORTATION_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in AIRCRAFT_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return AIRCRAFT_PROMPT_NAME, prompt_registry.get(AIRCRAFT_PROMPT_NAME)
        except KeyError:
            pass
    if any(pattern in generator_id for pattern in COMPACT_BACKTRANSLATION_GENERATOR_PATTERNS):
        try:
            return COMPACT_PROMPT_NAME, prompt_registry.get(COMPACT_PROMPT_NAME)
        except KeyError:
            pass
    return PROMPT_NAME, prompt_registry.get(PROMPT_NAME)


def _profile_normalized_row(row: dict[str, Any], config: EngineConfig) -> dict[str, Any]:
    """Refresh profile-derived fields for old frozen seed files.

    Seed factories can be expensive because they run solver validation and
    instance quality checks. When a generator profile is refined after a seed
    factory run, the CPT stage should use the current canonical signature and
    concept tags instead of carrying stale family-level hints forward.
    """
    generator_id = str(row.get("generator_id") or "").strip()
    if not generator_id:
        return row
    profile_payload = _cached_profile_payload(str(config.paths.generator_profiles_path))
    profiles = profile_payload.get("profiles") or {}
    profile = profiles.get(generator_id) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        return row
    normalized = dict(row)
    normalized["generator_profile_version"] = str(profile_payload.get("version") or normalized.get("generator_profile_version") or "")
    normalized["sub_family"] = profile.get("sub_family") or normalized.get("sub_family")
    normalized["canonical_math_signature"] = profile.get("canonical_math_signature") or normalized.get("canonical_math_signature") or {}
    normalized["concept_tags"] = list(profile.get("modeling_concepts") or normalized.get("concept_tags") or [])
    normalized["variant_id"] = profile.get("default_variant_id") or normalized.get("variant_id")
    metadata = dict(normalized.get("llm_metadata") or {})
    metadata["profile_normalized"] = True
    metadata["profile_normalized_from"] = row.get("generator_profile_version")
    metadata["profile_normalized_to"] = normalized["generator_profile_version"]
    normalized["llm_metadata"] = metadata
    return normalized


@lru_cache(maxsize=8)
def _cached_profile_payload(profiles_path: str) -> dict[str, Any]:
    return load_generator_profiles(profiles_path)


def render_backtranslation_report(candidate_count: int, rejected_count: int, requeued_count: int, candidate_path: Path) -> str:
    task_family_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    for row in iter_jsonl(candidate_path) or []:
        metadata = row.get("llm_metadata") or {}
        task_family_counts[str(metadata.get("task_family") or "unknown")] += 1
        scenario_counts[str(metadata.get("scenario_id") or "unknown")] += 1
        trigger_counts[str(metadata.get("business_trigger") or "unknown")] += 1
        prompt_counts[str(metadata.get("prompt_name") or "unknown")] += 1
    lines = [
        "# Backtranslation Report",
        "",
        f"- Candidates: {candidate_count}",
        f"- Rejected during generation/parsing: {rejected_count}",
        f"- Re-queued (transient LLM errors, retried): {requeued_count}",
        "",
        "## Task Families",
        "",
        "| Task family | Count |",
        "| --- | ---: |",
    ]
    for task_family, count in sorted(task_family_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {task_family} | {count} |")
    lines.extend(["", "## Scenarios", "", "| Scenario | Count |", "| --- | ---: |"])
    for scenario_id, count in sorted(scenario_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {scenario_id} | {count} |")
    lines.extend(["", "## Business Triggers", "", "| Business trigger | Count |", "| --- | ---: |"])
    for trigger, count in sorted(trigger_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {trigger} | {count} |")
    lines.extend(["", "## Prompt Usage", "", "| Prompt | Count |", "| --- | ---: |"])
    for prompt_name, count in sorted(prompt_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {prompt_name} | {count} |")
    return "\n".join(lines) + "\n"


def _render_prompt(
    template: str,
    row: dict[str, Any],
    candidate_index: int,
    scenario_guidance: dict[str, Any],
) -> str:
    target_chars = _target_backtranslation_prompt_chars(row)
    payload = _compact_backtranslation_payload(row, candidate_index, scenario_guidance)
    prompt = template.replace("{{INSTANCE_JSON}}", _stable_json(payload))
    if len(prompt) <= target_chars:
        return prompt

    payload["source_artifacts"]["lp_summary"] = _artifact_preview(row.get("lp_text"), max_chars=800)
    payload["source_artifacts"]["gurobi_code_summary"] = _artifact_preview(row.get("gurobi_code"), max_chars=400)
    payload["math_formula"] = _artifact_preview(row.get("math_formula"), max_chars=300)
    for compact_source_chars in (4_500, 2_500, 1_200):
        payload["source_artifacts"]["compact_source_data"] = _compact_source_data(row, max_chars=compact_source_chars)
        if payload["source_artifacts"]["compact_source_data"].get("available"):
            payload["source_artifacts"]["generation_params"] = {
                "available": False,
                "omitted_because": "compact_source_data is present and preferred for prompt safety",
            }
        else:
            payload["source_artifacts"]["generation_params"] = _compact_json_artifact(row.get("generation_params"), max_chars=1_000)
        prompt = template.replace("{{INSTANCE_JSON}}", _stable_json(payload))
        if len(prompt) <= target_chars:
            return prompt

    # Final safety net for large structured generators. At this point even the
    # compact payload is too large, so drop all LP/code/formula previews and keep
    # only the source-faithful fact card plus minimal scenario/signature context.
    for compact_source_chars in (2_500, 1_600, 900, 600, 350):
        ultra_payload = _ultra_compact_backtranslation_payload(
            row,
            candidate_index,
            scenario_guidance,
            compact_source_chars=compact_source_chars,
        )
        prompt = template.replace("{{INSTANCE_JSON}}", _stable_json(ultra_payload))
        if len(prompt) <= target_chars:
            return prompt
    for compact_source_chars in (900, 600, 350, 200, 120, 80, 0):
        ultra_payload = _ultra_compact_backtranslation_payload(
            row,
            candidate_index,
            scenario_guidance,
            compact_source_chars=compact_source_chars,
        )
        ultra_payload["prompt_safety_mode"] = "emergency_compact_template"
        prompt = _EMERGENCY_BACKTRANSLATION_TEMPLATE.replace("{{INSTANCE_JSON}}", _stable_json(ultra_payload))
        if len(prompt) <= target_chars:
            return prompt
    last_resort_payload = _last_resort_backtranslation_payload(row, candidate_index, scenario_guidance)
    prompt = _LAST_RESORT_BACKTRANSLATION_TEMPLATE.replace("{{INSTANCE_JSON}}", _stable_json(last_resort_payload))
    if len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS:
        return prompt
    raise ValueError(
        "backtranslation prompt exceeds safety limit after compaction: "
        f"{len(prompt)} chars > {_MAX_BACKTRANSLATION_PROMPT_CHARS}"
    )


def _target_backtranslation_prompt_chars(row: dict[str, Any]) -> int:
    """Use shorter prompts by default while preserving source-fact fidelity.

    Drift-prone generators need the compact fact card to survive prompt
    compaction. Empirically, 18.5k preserves all required source numbers for
    the seed_factory_1kv2 audit set; ordinary generators can use a tighter 16k
    target to reduce timeout and throughput pressure.
    """
    generator_id = str(row.get("generator_id") or "").lower()
    source_fact_patterns = SOURCE_FACT_BACKTRANSLATION_GENERATOR_PATTERNS + (
        "optmath_aircraftlanding",
        "aircraftlanding",
    )
    if any(pattern in generator_id for pattern in source_fact_patterns):
        return _SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS
    return _TARGET_BACKTRANSLATION_PROMPT_CHARS


def _compact_backtranslation_payload(
    row: dict[str, Any],
    candidate_index: int,
    scenario_guidance: dict[str, Any],
) -> dict[str, Any]:
    compact_source_data = _compact_source_data(row, max_chars=12_000)
    if compact_source_data.get("available"):
        generation_params = {
            "available": False,
            "omitted_because": "compact_source_data is present and preferred for prompt safety",
        }
    else:
        generation_params = _compact_json_artifact(row.get("generation_params"), max_chars=_MAX_GENERATION_PARAMS_CHARS)
    return {
        "instance_id": row.get("instance_id"),
        "generator_id": row.get("generator_id"),
        "candidate_index": candidate_index,
        "scenario_guidance": _compact_scenario_guidance(scenario_guidance),
        "generator_profile": {
            "difficulty_level": row.get("difficulty_level"),
            "sub_family": row.get("sub_family"),
            "concept_tags": row.get("concept_tags") or [],
            "variant_id": row.get("variant_id"),
            "canonical_math_signature": row.get("canonical_math_signature") or {},
        },
        "source_dimensions": {
            "model_type": row.get("model_type"),
            "optimization_sense": row.get("optimization_sense"),
            "num_variables": row.get("num_variables") or (row.get("solver_validation") or {}).get("num_variables"),
            "num_constraints": row.get("num_constraints") or (row.get("solver_validation") or {}).get("num_constraints"),
        },
        "math_formula": _artifact_preview(row.get("math_formula"), max_chars=2_000),
        "source_artifacts": {
            "compact_source_data": compact_source_data,
            "generation_params": generation_params,
            "lp_summary": _artifact_preview(row.get("lp_text"), max_chars=_MAX_LP_PREVIEW_CHARS),
            "gurobi_code_summary": _artifact_preview(row.get("gurobi_code"), max_chars=_MAX_CODE_PREVIEW_CHARS),
        },
        "source_artifact_policy": (
            "Large LP/code artifacts are summarized for prompt safety. Prefer structured generation_params when present. "
            "Do not mention solver status or objective value."
        ),
    }


def _ultra_compact_backtranslation_payload(
    row: dict[str, Any],
    candidate_index: int,
    scenario_guidance: dict[str, Any],
    *,
    compact_source_chars: int,
) -> dict[str, Any]:
    """Smallest safe payload used only when normal compacting still exceeds the prompt limit."""
    signature = row.get("canonical_math_signature") or {}
    compact_source_data = _compact_source_data(row, max_chars=compact_source_chars)
    generation_params = {"available": False, "omitted_because": "ultra_compact_payload"}
    if not compact_source_data.get("available"):
        generation_params = _compact_json_artifact(row.get("generation_params"), max_chars=compact_source_chars)
    return {
        "instance_id": row.get("instance_id"),
        "generator_id": row.get("generator_id"),
        "candidate_index": candidate_index,
        "scenario_guidance": _ultra_compact_scenario_guidance(scenario_guidance),
        "generator_profile": {
            "difficulty_level": row.get("difficulty_level"),
            "sub_family": _short_text(row.get("sub_family"), max_chars=120),
            "concept_tags": _short_list(row.get("concept_tags") or [], limit=8, max_chars=120),
            "canonical_math_signature": _ultra_compact_signature(signature),
        },
        "source_dimensions": {
            "model_type": row.get("model_type"),
            "optimization_sense": row.get("optimization_sense"),
            "num_variables": row.get("num_variables") or (row.get("solver_validation") or {}).get("num_variables"),
            "num_constraints": row.get("num_constraints") or (row.get("solver_validation") or {}).get("num_constraints"),
        },
        "source_artifacts": {
            "compact_source_data": compact_source_data,
            "generation_params": generation_params,
            "lp_summary": {"available": False, "omitted_because": "ultra_compact_payload"},
            "gurobi_code_summary": {"available": False, "omitted_because": "ultra_compact_payload"},
        },
        "math_formula": {"available": False, "omitted_because": "ultra_compact_payload"},
        "source_artifact_policy": (
            "Ultra-compact prompt safety mode. Use compact_source_data or generation_params only. "
            "Do not mention solver status, objective value, source code, LP text, or audit/training commentary."
        ),
    }


def _last_resort_backtranslation_payload(
    row: dict[str, Any],
    candidate_index: int,
    scenario_guidance: dict[str, Any],
) -> dict[str, Any]:
    compact_source_data = _compact_source_data(row, max_chars=120)
    generation_params = {"available": False, "omitted_because": "last_resort_compact_payload"}
    if not compact_source_data.get("available"):
        generation_params = _compact_json_artifact(row.get("generation_params"), max_chars=120)
    signature = row.get("canonical_math_signature") or {}
    return {
        "prompt_safety_mode": "last_resort_compact_template",
        "instance_id": _short_text(row.get("instance_id"), max_chars=80),
        "generator_id": _short_text(row.get("generator_id"), max_chars=80),
        "candidate_index": candidate_index,
        "scenario": {
            "task_family": _short_text(scenario_guidance.get("task_family"), max_chars=80),
            "scenario_id": _short_text(scenario_guidance.get("scenario_id"), max_chars=100),
            "trigger": _short_text(scenario_guidance.get("selected_business_trigger"), max_chars=120),
        },
        "generator_profile": {
            "sub_family": _short_text(row.get("sub_family"), max_chars=100),
            "concept_tags": _short_list(row.get("concept_tags") or [], limit=5, max_chars=80),
            "canonical_math_signature": {
                "objective_sense": _short_text(signature.get("objective_sense"), max_chars=60) if isinstance(signature, dict) else None,
                "variable_types": _short_list(signature.get("variable_types") if isinstance(signature, dict) else [], limit=4, max_chars=80),
                "core_constraints": _short_list(signature.get("core_constraints") if isinstance(signature, dict) else [], limit=5, max_chars=80),
                "forbidden_changes": _short_list(signature.get("forbidden_changes") if isinstance(signature, dict) else [], limit=4, max_chars=100),
            },
        },
        "source_dimensions": {
            "model_type": _short_text(row.get("model_type"), max_chars=40),
            "optimization_sense": _short_text(row.get("optimization_sense"), max_chars=40),
            "num_variables": row.get("num_variables") or (row.get("solver_validation") or {}).get("num_variables"),
            "num_constraints": row.get("num_constraints") or (row.get("solver_validation") or {}).get("num_constraints"),
        },
        "source_artifacts": {
            "compact_source_data": compact_source_data,
            "generation_params": generation_params,
        },
        "source_artifact_policy": "Last-resort prompt safety mode. Prefer compact_source_data; otherwise use generation_params preview only.",
    }


def _ultra_compact_signature(signature: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(signature, dict):
        return {}
    return {
        "variable_types": _short_list(signature.get("variable_types") or [], limit=6, max_chars=140),
        "objective_sense": _short_text(signature.get("objective_sense"), max_chars=80),
        "objective_terms": _short_list(signature.get("objective_terms") or [], limit=6, max_chars=140),
        "core_constraints": _short_list(signature.get("core_constraints") or [], limit=8, max_chars=140),
        "required_tables": _short_list(signature.get("required_tables") or [], limit=8, max_chars=140),
        "forbidden_changes": _short_list(signature.get("forbidden_changes") or [], limit=6, max_chars=160),
    }


def _ultra_compact_scenario_guidance(scenario_guidance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(scenario_guidance, dict):
        return {}
    selected_scenario = scenario_guidance.get("selected_scenario") or {}
    applicability = scenario_guidance.get("applicability") or {}
    return {
        "generator_id": _short_text(scenario_guidance.get("generator_id"), max_chars=100),
        "task_family": _short_text(scenario_guidance.get("task_family"), max_chars=100),
        "scenario_id": _short_text(scenario_guidance.get("scenario_id"), max_chars=140),
        "selected_business_trigger": _short_text(scenario_guidance.get("selected_business_trigger"), max_chars=180),
        "selected_scenario": {
            "industry": _short_text(selected_scenario.get("industry"), max_chars=120),
            "background": _short_text(selected_scenario.get("background") or selected_scenario.get("base_background"), max_chars=220),
            "entities": _short_list(selected_scenario.get("entities") or [], limit=8, max_chars=100),
        },
        "required_parameter_presentation": _short_list(
            scenario_guidance.get("required_parameter_presentation") or [],
            limit=8,
            max_chars=160,
        ),
        "forbidden_misframings": _short_list(scenario_guidance.get("forbidden_misframings") or [], limit=6, max_chars=160),
        "incompatible_source_features": _short_list(
            applicability.get("incompatible_source_features") or [],
            limit=6,
            max_chars=140,
        ),
        "scenario_policy": (
            "Use selected_scenario as the business wrapper. Do not replace its entities with a generic scenario variant."
        ),
    }


def _short_text(value: Any, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _short_list(values: Any, *, limit: int, max_chars: int) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        shortened
        for value in list(values)[:limit]
        if (shortened := _short_text(value, max_chars=max_chars))
    ]


def _compact_scenario_guidance(scenario_guidance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(scenario_guidance, dict):
        return {}
    variant = scenario_guidance.get("scenario_variant") or {}
    selected_scenario = scenario_guidance.get("selected_scenario") or {}
    industry_lens = variant.get("industry_lens") or {}
    narrative_angle = variant.get("narrative_angle") or {}
    organization_profile = variant.get("organization_profile") or {}
    planning_horizon = variant.get("planning_horizon") or {}
    entity_naming_style = variant.get("entity_naming_style") or {}
    applicability = scenario_guidance.get("applicability") or {}
    return {
        "catalog_version": scenario_guidance.get("catalog_version"),
        "generator_id": scenario_guidance.get("generator_id"),
        "task_family": scenario_guidance.get("task_family"),
        "scenario_id": scenario_guidance.get("scenario_id"),
        "base_scenario_id": scenario_guidance.get("base_scenario_id"),
        "scenario_variant_id": scenario_guidance.get("scenario_variant_id"),
        "selected_business_trigger": scenario_guidance.get("selected_business_trigger"),
        "selected_scenario": {
            "industry": selected_scenario.get("industry"),
            "background": selected_scenario.get("background") or selected_scenario.get("base_background"),
            "entities": selected_scenario.get("entities"),
        },
        "scenario_variant": {
            "industry_lens": {
                "industry": industry_lens.get("industry"),
                "context": industry_lens.get("context"),
                "entities": industry_lens.get("entities"),
            },
            "narrative_angle": {"description": narrative_angle.get("description")},
            "organization_profile": {"description": organization_profile.get("description")},
            "planning_horizon": {"description": planning_horizon.get("description")},
            "entity_naming_style": {"description": entity_naming_style.get("description")},
        },
        "required_parameter_presentation": scenario_guidance.get("required_parameter_presentation") or [],
        "forbidden_misframings": scenario_guidance.get("forbidden_misframings") or [],
        "modeling_notes": scenario_guidance.get("modeling_notes") or [],
        "incompatible_source_features": applicability.get("incompatible_source_features") or [],
    }


def _compact_source_data(row: dict[str, Any], *, max_chars: int = 12_000) -> dict[str, Any]:
    existing = row.get("source_compact_data")
    if isinstance(existing, dict) and existing.get("available"):
        if existing.get("truncated") and "value" not in existing:
            return existing
        value = existing.get("value") if isinstance(existing.get("value"), dict) else {
            key: val for key, val in existing.items() if str(key).startswith("compact_")
        }
        if isinstance(value, dict) and value:
            return _compact_json_artifact(value, max_chars=max_chars)
    generation_params = row.get("generation_params")
    if not isinstance(generation_params, dict):
        generation_params = {}
    preferred_keys = (
        "compact_facility_location_tables",
        "compact_cflp_tables",
        "compact_car_selection_tables",
        "compact_cell_tower_tables",
        "compact_revenue_tables",
        "compact_revenue_management_tables",
        "compact_marketshare_tables",
        "compact_electrical_power_tables",
        "compact_fleet_flow_tables",
        "compact_portfolio_tables",
        "compact_factory_planning_tables",
        "compact_cutting_stock_tables",
        "compact_contract_allocation_tables",
        "compact_farm_planning_tables",
        "compact_team_formulation_tables",
        "compact_multi_factory_schedule_tables",
        "compact_multisetcover_tables",
        "compact_p_dispersion_tables",
        "compact_mcnd_tables",
        "compact_supplychain_tables",
        "compact_transportation_tables",
        "compact_multi_commodity_transportation_tables",
        "compact_netmcol_tables",
        "compact_network_flow_tables",
        "compact_net1_tables",
        "compact_project_assignment_tables",
        "compact_scooter_location_tables",
        "compact_shortest_path_tables",
        "compact_tsp_tables",
        "compact_aircraft_assignment_tables",
        "compact_aircraft_landing_tables",
        "compact_staffing_tables",
        "compact_flowshop_tables",
        "compact_structure_assignment_tables",
        "compact_steel_product_mix_tables",
        "compact_lotsizing_tables",
    )
    selected = {
        key: generation_params[key]
        for key in preferred_keys
        if key in generation_params
    }
    legacy_selected = _legacy_compact_source_data(
        str(row.get("generator_id") or ""),
        generation_params,
        lp_text=str(row.get("lp_text") or row.get("source_lp_text") or ""),
    )
    for key, value in legacy_selected.items():
        selected.setdefault(key, value)
    if not selected:
        return {"available": False}
    return _compact_json_artifact(selected, max_chars=max_chars)


def _legacy_compact_source_data(
    generator_id: str,
    generation_params: dict[str, Any],
    *,
    lp_text: str = "",
) -> dict[str, Any]:
    """Rebuild compact fact cards for seed runs produced before compact_* fields.

    This is deliberately conservative: it only reconstructs formats when the
    legacy generation_params already contain the complete source tables.
    """
    lowered = generator_id.lower()
    selected: dict[str, Any] = {}
    if ("optmath_steel4" in lowered or "steel4" in lowered) and _has_keys(
        generation_params,
        ("products", "stages", "processing_hours_per_unit", "available", "profit", "commit", "market"),
    ):
        selected["compact_steel_product_mix_tables"] = _legacy_steel4_compact_tables(generation_params)
    if ("optmath_steel3" in lowered or "steel3" in lowered) and _has_keys(
        generation_params,
        ("products", "production_rates", "processing_hours_per_unit", "profits", "min_sold", "max_sold", "available_hours"),
    ):
        selected["compact_steel_product_mix_tables"] = _legacy_steel3_compact_tables(generation_params)
    if ("optmath_electrical_power" in lowered or "electrical_power" in lowered) and _has_keys(
        generation_params,
        (
            "generator_types",
            "time_periods",
            "demand",
            "min_output",
            "max_output",
            "base_cost",
            "per_mw_cost",
            "startup_cost",
            "generators_available",
            "on_start",
        ),
    ):
        selected["compact_electrical_power_tables"] = _legacy_electrical_power_compact_tables(generation_params)
    if "clsp_expand_capacity" in lowered:
        clsp_tables = _legacy_clsp_compact_tables_from_lp(lp_text)
        if clsp_tables:
            selected["compact_lotsizing_tables"] = clsp_tables
    is_ulsb = "uncapacitatedlotsizingbacklogging" in lowered or "uncapacitated_lot_sizing_with_backlogging" in lowered
    is_uls = (
        "uncapacitatedlotsizing" in lowered
        or "uncapacitated_lot_sizing" in lowered
    ) and not is_ulsb
    if is_ulsb:
        ulsb_tables = _legacy_ulsb_compact_tables_from_lp(lp_text)
        if ulsb_tables:
            selected["compact_lotsizing_tables"] = ulsb_tables
        elif isinstance(generation_params.get("period_cost_table"), list):
            selected["compact_lotsizing_tables"] = _legacy_ulsb_compact_tables(generation_params)
    elif is_uls and isinstance(generation_params.get("period_cost_table"), list):
        selected["compact_lotsizing_tables"] = _legacy_uls_compact_tables(generation_params)
    elif is_uls:
        uls_tables = _legacy_uls_compact_tables_from_lp(lp_text)
        if uls_tables:
            selected["compact_lotsizing_tables"] = uls_tables
    if "singlelevelsmallbucket" in lowered:
        smallbucket_tables = _legacy_smallbucket_compact_tables_from_lp(lp_text)
        if smallbucket_tables:
            selected["compact_lotsizing_tables"] = smallbucket_tables
    if "marketshare" in lowered:
        marketshare_tables = _legacy_marketshare_compact_tables_from_lp(lp_text)
        if marketshare_tables:
            selected["compact_marketshare_tables"] = marketshare_tables
    if ("optmath_net1" in lowered or lowered.endswith("net1") or lowered == "net1") and _has_keys(
        generation_params,
        ("cities", "links", "supply", "demand", "shipping_cost", "capacity"),
    ):
        selected["compact_network_flow_tables"] = _legacy_net1_compact_tables(generation_params)
    if ("optmath_netasgn" in lowered or lowered.endswith("netasgn") or lowered == "netasgn") and _has_keys(
        generation_params,
        ("people", "projects", "supply_hours", "demand_hours", "cost_per_hour", "max_contribution_hours"),
    ):
        selected["compact_project_assignment_tables"] = _legacy_netasgn_compact_tables(generation_params)
    if "structure_based_assignment" in lowered:
        structure_tables = _legacy_structure_assignment_compact_tables_from_lp(lp_text)
        if structure_tables:
            selected["compact_structure_assignment_tables"] = structure_tables
    return selected


def _legacy_steel4_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    products = list(params.get("products") or [])
    stages = list(params.get("stages") or [])
    product_table = params.get("product_table") if isinstance(params.get("product_table"), dict) else {}
    stage_capacity_table = (
        params.get("stage_capacity_table") if isinstance(params.get("stage_capacity_table"), dict) else {}
    )
    processing = params.get("processing_hours_per_unit") if isinstance(params.get("processing_hours_per_unit"), dict) else {}
    profit = params.get("profit") if isinstance(params.get("profit"), dict) else {}
    commit = params.get("commit") if isinstance(params.get("commit"), dict) else {}
    market = params.get("market") if isinstance(params.get("market"), dict) else {}
    available = params.get("available") if isinstance(params.get("available"), dict) else {}
    return {
        "sets": {
            "products": products,
            "production_stages": stages,
        },
        "product_table": {
            "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
            "rows": [
                [
                    product,
                    _nested_or(product_table, [product, "profit_per_ton"], profit.get(product)),
                    _nested_or(product_table, [product, "minimum_commitment_tons"], commit.get(product)),
                    _nested_or(product_table, [product, "maximum_market_tons"], market.get(product)),
                ]
                for product in products
            ],
        },
        "stage_capacity_table": {
            "columns": ["stage", "available_hours"],
            "rows": [
                [
                    stage,
                    _nested_or(stage_capacity_table, [stage, "available_hours"], available.get(stage)),
                ]
                for stage in stages
            ],
        },
        "processing_hours_per_ton_matrix": {
            "columns": stages,
            "rows": [
                {
                    "product": product,
                    "values": [
                        processing.get(f"{product}|{stage}")
                        for stage in stages
                    ],
                }
                for product in products
            ],
        },
        "source_model_note": (
            "Legacy generation_params fact card. Stage capacity is "
            "sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]. "
            "Product market upper bounds are separate from stage available hours."
        ),
    }


def _legacy_ulsb_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in params.get("period_cost_table") or [] if isinstance(row, dict)]
    periods = list(params.get("periods") or [row.get("period") for row in rows])
    total_demand = params.get("big_m_order_upper_bound")
    if total_demand is None:
        total_demand = sum(_safe_number(row.get("demand")) for row in rows)
    return {
        "model_family": "uncapacitated_lot_sizing_with_backlogging",
        "periods": periods,
        "period_cost_table": rows,
        "total_demand_big_m": total_demand,
        "initial_inventory": params.get("initial_inventory", 0),
        "initial_backlog": params.get("initial_backlog", 0),
        "required_final_inventory": params.get("required_final_inventory", 0),
        "required_final_backlog": params.get("required_final_backlog", 0),
        "balance_transition_by_period": params.get("balance_transition_by_period") or [],
        "setup_linking_by_period": params.get("setup_linking_by_period") or {},
        "source_model_note": (
            "Legacy generation_params fact card. Demand values are period-by-period demand, not cumulative demand. "
            "Net inventory is EndingInventory[t] - BackloggedAmount[t]. There is no production capacity constraint."
        ),
    }


def _legacy_ulsb_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    objective = _lp_objective_section(lp_text)
    if not objective:
        return {}
    order_costs = _lp_single_index_coefficients(objective, "OrderedAmount")
    holding_costs = _lp_single_index_coefficients(objective, "EndingInventory")
    fixed_costs = _lp_single_index_coefficients(objective, "OrderIsPlaced")
    backlog_penalties = _lp_single_index_coefficients(objective, "BackloggedAmount")
    if not order_costs or not holding_costs or not fixed_costs or not backlog_penalties:
        return {}

    demand_by_period: dict[str, float] = {}
    big_m_by_period: dict[str, float] = {}
    required_final_inventory: float | None = None
    required_final_backlog: float | None = None
    for label, body in _iter_lp_constraints(lp_text):
        if label.startswith("FlowBalance_"):
            period = label.removeprefix("FlowBalance_")
            rhs = _lp_rhs_number(body, "=")
            if rhs is None:
                return {}
            demand_by_period[period] = abs(rhs)
        elif label.startswith("OrderedUpperBound_"):
            period = label.removeprefix("OrderedUpperBound_")
            match = re.search(
                rf"-\s*(?P<big_m>{_LP_NUMBER_RE})\s+OrderIsPlaced\[{re.escape(period)}\]\s*<=\s*0",
                body,
            )
            if match:
                big_m_by_period[period] = abs(float(match.group("big_m")))
        elif label in {"EndingInventoryZero", "FinalInventory", "EndingInventory"}:
            match = re.search(rf"EndingInventory\[(?P<period>[^\]]+)\]\s*=\s*(?P<rhs>{_LP_NUMBER_RE})", body)
            if match:
                required_final_inventory = float(match.group("rhs"))
        elif label in {"EndingBacklogZero", "FinalBacklog", "EndingBacklog"}:
            match = re.search(rf"BackloggedAmount\[(?P<period>[^\]]+)\]\s*=\s*(?P<rhs>{_LP_NUMBER_RE})", body)
            if match:
                required_final_backlog = float(match.group("rhs"))

    periods = sorted(demand_by_period, key=_period_label_sort_key)
    if not periods:
        return {}
    required_maps = (order_costs, holding_costs, fixed_costs, backlog_penalties)
    if any(set(periods) - set(cost_map) for cost_map in required_maps):
        return {}
    total_demand = sum(demand_by_period[period] for period in periods)
    if big_m_by_period:
        parsed_big_ms = [big_m_by_period.get(period) for period in periods]
        if any(value is None for value in parsed_big_ms):
            return {}
        parsed_values = [float(value) for value in parsed_big_ms if value is not None]
        first_big_m = parsed_values[0]
        if not all(_lp_numbers_close(value, first_big_m) for value in parsed_values):
            return {}
        if not _lp_numbers_close(first_big_m, total_demand):
            return {}
        total_demand_big_m = first_big_m
    else:
        total_demand_big_m = total_demand
    if required_final_inventory is None:
        required_final_inventory = 0.0
    if required_final_backlog is None:
        required_final_backlog = 0.0
    rows = [
        {
            "period": period,
            "demand": _lp_clean_number(demand_by_period[period]),
            "fixed_ordering_cost": _lp_clean_number(fixed_costs[period]),
            "unit_order_cost": _lp_clean_number(order_costs[period]),
            "unit_holding_cost": _lp_clean_number(holding_costs[period]),
            "unit_backlog_penalty": _lp_clean_number(backlog_penalties[period]),
        }
        for period in periods
    ]
    balance_transition_by_period = []
    for index, period in enumerate(periods):
        if index == 0:
            previous_state = "initial_inventory - initial_backlog = 0"
            equation = f"EndingInventory[{period}] - BackloggedAmount[{period}] = OrderedAmount[{period}] - demand[{period}]"
        else:
            previous = periods[index - 1]
            previous_state = f"EndingInventory[{previous}] - BackloggedAmount[{previous}]"
            equation = (
                f"EndingInventory[{period}] - BackloggedAmount[{period}] = "
                f"EndingInventory[{previous}] - BackloggedAmount[{previous}] + OrderedAmount[{period}] - demand[{period}]"
            )
        balance_transition_by_period.append(
            {
                "period": period,
                "previous_net_inventory_state": previous_state,
                "demand": _lp_clean_number(demand_by_period[period]),
                "equation": equation,
            }
        )
    return {
        "model_family": "uncapacitated_lot_sizing_with_backlogging",
        "periods": periods,
        "period_cost_table": rows,
        "total_demand_big_m": _lp_clean_number(total_demand_big_m),
        "initial_inventory": 0,
        "initial_backlog": 0,
        "required_final_inventory": _lp_clean_number(required_final_inventory),
        "required_final_backlog": _lp_clean_number(required_final_backlog),
        "balance_transition_by_period": balance_transition_by_period,
        "setup_linking_by_period": {
            period: f"OrderedAmount[{period}] <= {_lp_clean_number(total_demand_big_m)} * OrderIsPlaced[{period}]"
            for period in periods
        },
        "source_model_note": (
            "Legacy LP-derived fact card. Demand and costs are reconstructed from FlowBalance, objective, "
            "OrderedUpperBound, final-inventory, and final-backlog constraints. Demand values are period-by-period, "
            "and net inventory is EndingInventory[t] - BackloggedAmount[t]."
        ),
    }


def _legacy_steel3_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    products = list(params.get("products") or [])
    rates = params.get("production_rates") if isinstance(params.get("production_rates"), dict) else {}
    processing = params.get("processing_hours_per_unit") if isinstance(params.get("processing_hours_per_unit"), dict) else {}
    profits = params.get("profits") if isinstance(params.get("profits"), dict) else {}
    min_sold = params.get("min_sold") if isinstance(params.get("min_sold"), dict) else {}
    max_sold = params.get("max_sold") if isinstance(params.get("max_sold"), dict) else {}
    available_hours = params.get("available_hours")
    return {
        "model_family": "single_stage_continuous_product_mix",
        "sets": {"products": products},
        "product_table": {
            "columns": [
                "product",
                "profit_per_ton",
                "production_rate_tons_per_hour",
                "processing_hours_per_ton",
                "minimum_commitment_tons",
                "maximum_market_tons",
            ],
            "rows": [
                [
                    product,
                    profits.get(product),
                    rates.get(product),
                    processing.get(product),
                    min_sold.get(product),
                    max_sold.get(product),
                ]
                for product in products
            ],
        },
        "time_capacity": {
            "available_hours": available_hours,
            "constraint": "sum_p processing_hours_per_ton[p] * Production[p] <= available_hours",
        },
        "source_model_note": (
            "This is a single-stage continuous product-mix LP. It has one shared production-hour "
            "capacity, product minimum commitments, and product maximum market bounds. It has no "
            "inventory, setup, binary product-selection, sequencing, or multi-stage capacity matrix."
        ),
    }


def _legacy_electrical_power_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    generator_types = list(params.get("generator_types") or [])
    time_periods = list(params.get("time_periods") or [])
    demand = params.get("demand") if isinstance(params.get("demand"), dict) else {}
    min_output = params.get("min_output") if isinstance(params.get("min_output"), dict) else {}
    max_output = params.get("max_output") if isinstance(params.get("max_output"), dict) else {}
    base_cost = params.get("base_cost") if isinstance(params.get("base_cost"), dict) else {}
    per_mw_cost = params.get("per_mw_cost") if isinstance(params.get("per_mw_cost"), dict) else {}
    startup_cost = params.get("startup_cost") if isinstance(params.get("startup_cost"), dict) else {}
    available = params.get("generators_available") if isinstance(params.get("generators_available"), dict) else {}
    on_start = params.get("on_start") if isinstance(params.get("on_start"), dict) else {}
    reserve_margin = params.get("reserve_margin", 1.15)
    return {
        "model_family": "unit_commitment_electrical_power",
        "sets": {
            "generator_types": generator_types,
            "time_periods": time_periods,
        },
        "generator_type_table": [
            {
                "generator_type": generator_type,
                "base_operating_cost_per_period": base_cost.get(generator_type),
                "per_mw_generation_cost": per_mw_cost.get(generator_type),
                "startup_cost": startup_cost.get(generator_type),
                "minimum_output_mw": min_output.get(generator_type),
                "maximum_output_mw": max_output.get(generator_type),
                "generators_available": available.get(generator_type),
                "generators_on_initially": on_start.get(generator_type),
            }
            for generator_type in generator_types
        ],
        "period_demand_table": [
            {
                "period": period,
                "demand_mw": demand.get(period),
                "reserve_required_capacity_mw": _round_if_number(_safe_multiply(reserve_margin, demand.get(period)), digits=4),
            }
            for period in time_periods
        ],
        "decision_variables": {
            "NumGenerators[t,p]": "nonnegative integer number of generators of type t that are on in period p",
            "PowerOutput[t,p]": "continuous nonnegative total MW output from generator type t in period p",
            "NumStart[t,p]": "nonnegative integer number of generators of type t started in period p",
        },
        "constraints": [
            "NumGenerators[t,p] <= generators_available[t]",
            "sum_t PowerOutput[t,p] >= demand[p]",
            "PowerOutput[t,p] >= minimum_output_mw[t] * NumGenerators[t,p]",
            "PowerOutput[t,p] <= maximum_output_mw[t] * NumGenerators[t,p]",
            "sum_t maximum_output_mw[t] * NumGenerators[t,p] >= reserve_margin * demand[p]",
            "NumGenerators[t,first_period] <= generators_on_initially[t] + NumStart[t,first_period]",
            "NumGenerators[t,p] <= NumGenerators[t,previous_period] + NumStart[t,p]",
        ],
        "objective": "minimize base operating cost, per-MW generation cost, and startup cost",
        "source_contract_note": (
            "This is a deterministic unit-commitment style MILP with integer on-count and startup-count "
            "variables by generator type and period. It has no battery storage, power-flow network, "
            "fuel inventory, emissions cap, unit-specific identity, or minimum up/down time constraints."
        ),
    }


def _safe_multiply(left: Any, right: Any) -> float | None:
    try:
        return float(left) * float(right)
    except (TypeError, ValueError):
        return None


def _round_if_number(value: Any, *, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _legacy_uls_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in params.get("period_cost_table") or [] if isinstance(row, dict)]
    periods = list(params.get("periods") or [row.get("period") for row in rows])
    total_demand = params.get("big_m_order_upper_bound")
    if total_demand is None:
        total_demand = sum(_safe_number(row.get("demand")) for row in rows)
    return {
        "model_family": "uncapacitated_lot_sizing_without_backlog",
        "periods": periods,
        "period_cost_table": rows,
        "total_demand_big_m": total_demand,
        "initial_inventory": params.get("initial_inventory", 0),
        "required_final_inventory": params.get("required_final_inventory", 0),
        "source_model_note": (
            "Legacy generation_params fact card. Demand values are period-by-period demand. "
            "There are no backlog variables and no production capacity constraint."
        ),
    }


def _legacy_uls_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    objective = _lp_objective_section(lp_text)
    if not objective:
        return {}
    order_costs = _lp_single_index_coefficients(objective, "OrderedAmount")
    holding_costs = _lp_single_index_coefficients(objective, "EndingInventory")
    fixed_costs = _lp_single_index_coefficients(objective, "OrderIsPlaced")
    if not order_costs or not holding_costs or not fixed_costs:
        return {}

    demand_by_period: dict[str, float] = {}
    big_m_by_period: dict[str, float] = {}
    required_final_inventory: float | None = None
    for label, body in _iter_lp_constraints(lp_text):
        if label.startswith("FlowBalance_"):
            period = label.removeprefix("FlowBalance_")
            rhs = _lp_rhs_number(body, "=")
            if rhs is None:
                return {}
            demand_by_period[period] = rhs
        elif label.startswith("OrderedUpperBound_"):
            period = label.removeprefix("OrderedUpperBound_")
            match = re.search(
                rf"-\s*(?P<big_m>{_LP_NUMBER_RE})\s+OrderIsPlaced\[{re.escape(period)}\]\s*<=\s*0",
                body,
            )
            if match:
                big_m_by_period[period] = abs(float(match.group("big_m")))
        elif label == "EndingInventory":
            match = re.search(rf"EndingInventory\[(?P<period>[^\]]+)\]\s*=\s*(?P<rhs>{_LP_NUMBER_RE})", body)
            if match:
                required_final_inventory = float(match.group("rhs"))

    periods = sorted(demand_by_period, key=_period_label_sort_key)
    if not periods:
        return {}
    if set(periods) - set(order_costs) or set(periods) - set(holding_costs) or set(periods) - set(fixed_costs):
        return {}
    total_demand = sum(demand_by_period[period] for period in periods)
    if big_m_by_period:
        parsed_big_ms = [big_m_by_period.get(period) for period in periods]
        if any(value is None for value in parsed_big_ms):
            return {}
        parsed_values = [float(value) for value in parsed_big_ms if value is not None]
        first_big_m = parsed_values[0]
        if not all(_lp_numbers_close(value, first_big_m) for value in parsed_values):
            return {}
        if not _lp_numbers_close(first_big_m, total_demand):
            return {}
        total_demand_big_m = first_big_m
    else:
        total_demand_big_m = total_demand
    if required_final_inventory is None:
        required_final_inventory = 0.0
    rows = [
        {
            "period": period,
            "demand": _lp_clean_number(demand_by_period[period]),
            "fixed_ordering_cost": _lp_clean_number(fixed_costs[period]),
            "unit_order_cost": _lp_clean_number(order_costs[period]),
            "unit_holding_cost": _lp_clean_number(holding_costs[period]),
        }
        for period in periods
    ]
    return {
        "model_family": "uncapacitated_lot_sizing_without_backlog",
        "periods": periods,
        "period_cost_table": rows,
        "total_demand_big_m": _lp_clean_number(total_demand_big_m),
        "initial_inventory": 0,
        "required_final_inventory": _lp_clean_number(required_final_inventory),
        "source_model_note": (
            "Legacy LP-derived fact card. Demand and costs are reconstructed from FlowBalance, objective, "
            "OrderedUpperBound, and final-inventory constraints. This is no-backlog uncapacitated lot sizing."
        ),
    }


def _legacy_clsp_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    objective = _lp_objective_section(lp_text)
    if not objective:
        return {}
    production_costs = _lp_two_index_coefficients(objective, "Production")
    setup_costs = _lp_two_index_coefficients(objective, "Setup")
    inventory_costs = _lp_underscore_two_index_coefficients(objective, "Inventory")
    if not production_costs or not setup_costs or not inventory_costs:
        return {}

    cumulative_demand: dict[tuple[int, int], float] = {}
    capacities: dict[int, float] = {}
    capacity_consumption_samples: dict[int, list[float]] = {}
    setup_big_m: dict[tuple[int, int], float] = {}
    for label, body in _iter_lp_constraints(lp_text):
        inventory_match = re.fullmatch(r"InventoryBalance_(\d+)_(\d+)", label)
        if inventory_match:
            product = int(inventory_match.group(1))
            period = int(inventory_match.group(2))
            rhs = _lp_rhs_number(body, "=")
            if rhs is None:
                return {}
            cumulative_demand[(product, period)] = abs(rhs)
            continue
        capacity_match = re.fullmatch(r"Capacity_(\d+)", label)
        if capacity_match:
            period = int(capacity_match.group(1))
            rhs = _lp_rhs_number(body, "<=")
            if rhs is None:
                return {}
            capacities[period] = rhs
            for term in re.finditer(
                rf"(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+Production\[(?P<product>\d+),(?P<period>\d+)\]",
                body,
            ):
                if int(term.group("period")) != period:
                    return {}
                product = int(term.group("product"))
                capacity_consumption_samples.setdefault(product, []).append(abs(float(term.group("coef").replace(" ", ""))))
            continue
        setup_match = re.fullmatch(r"Setup_(\d+)_(\d+)", label)
        if setup_match:
            product = int(setup_match.group(1))
            period = int(setup_match.group(2))
            match = re.search(
                rf"Production\[{product},{period}\]\s*-\s*(?P<big_m>{_LP_NUMBER_RE})\s+Setup\[{product},{period}\]\s*<=\s*0",
                body,
            )
            if match:
                setup_big_m[(product, period)] = abs(float(match.group("big_m")))

    products = sorted({product for product, _ in cumulative_demand})
    periods = sorted({period for _, period in cumulative_demand})
    if not products or not periods:
        return {}
    expected_pairs = {(product, period) for product in products for period in periods}
    required_pair_maps = [cumulative_demand, production_costs, setup_costs, inventory_costs, setup_big_m]
    if any(expected_pairs - set(mapping) for mapping in required_pair_maps):
        return {}
    if set(periods) - set(capacities):
        return {}

    capacity_consumption: dict[int, float] = {}
    for product in products:
        samples = capacity_consumption_samples.get(product) or []
        if not samples or len(samples) != len(periods):
            return {}
        first = samples[0]
        if not all(_lp_numbers_close(value, first) for value in samples):
            return {}
        capacity_consumption[product] = first

    demand_rows = []
    for product in products:
        previous_cumulative = 0.0
        for period in periods:
            cumulative = cumulative_demand[(product, period)]
            period_demand = cumulative - previous_cumulative
            if period_demand < -1e-7:
                return {}
            previous_cumulative = cumulative
            demand_rows.append(
                {
                    "product": product,
                    "period": period,
                    "period_demand": _lp_clean_number(period_demand),
                    "cumulative_demand_through_period": _lp_clean_number(cumulative),
                    "setup_cost": _lp_clean_number(setup_costs[(product, period)]),
                    "unit_production_cost": _lp_clean_number(production_costs[(product, period)]),
                    "unit_holding_cost": _lp_clean_number(inventory_costs[(product, period)]),
                    "setup_big_m_remaining_demand": _lp_clean_number(setup_big_m[(product, period)]),
                }
            )

    return {
        "model_family": "capacitated_lot_sizing_without_backlog",
        "products": products,
        "periods": periods,
        "period_demand_table": demand_rows,
        "period_capacity_table": [
            {"period": period, "capacity": _lp_clean_number(capacities[period])}
            for period in periods
        ],
        "capacity_consumption_per_product": [
            {
                "product": product,
                "capacity_consumption_per_unit": _lp_clean_number(capacity_consumption[product]),
            }
            for product in products
        ],
        "source_model_note": (
            "Legacy LP-derived fact card. Cumulative demand, costs, setup big-M, fixed period capacity, "
            "and per-product capacity consumption are reconstructed from the verified CLSP LP. "
            "The source model has fixed capacity, no backlog, and no capacity-expansion decision."
        ),
    }


def _legacy_smallbucket_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    objective = _lp_objective_section(lp_text)
    if not objective:
        return {}
    production_costs = _lp_three_index_coefficients(objective, "Production")
    startup_costs = _lp_three_index_coefficients(objective, "Startup")
    holding_costs = _lp_two_index_coefficients(objective, "Stock")
    backlog_costs = _lp_two_index_coefficients(objective, "Backlog")
    if not production_costs or not startup_costs or not holding_costs or not backlog_costs:
        return {}

    items = sorted({item for item, _, _ in production_costs})
    machines = sorted({machine for _, machine, _ in production_costs})
    periods = sorted({period for _, _, period in production_costs})
    if not items or not machines or not periods:
        return {}
    expected_three = {(item, machine, period) for item in items for machine in machines for period in periods}
    if expected_three - set(production_costs) or expected_three - set(startup_costs):
        return {}
    setup_cost_values = list(production_costs.values())
    startup_cost_values = list(startup_costs.values())
    setup_cost = setup_cost_values[0]
    startup_cost = startup_cost_values[0]
    if not all(_lp_numbers_close(value, setup_cost) for value in setup_cost_values):
        return {}
    if not all(_lp_numbers_close(value, startup_cost) for value in startup_cost_values):
        return {}

    demand: dict[tuple[int, int], float] = {}
    machine_capacity_samples: dict[int, list[float]] = {}
    machine_startup_samples: dict[int, list[float]] = {}
    for _label, body in _iter_lp_constraints(lp_text):
        rhs_eq = _lp_rhs_number(body, "=")
        if rhs_eq is not None:
            amount_terms = [
                (int(match.group("item")), int(match.group("machine")), int(match.group("period")))
                for match in re.finditer(
                    r"Amount\[(?P<item>\d+),(?P<machine>\d+),(?P<period>\d+)\]",
                    body,
                )
            ]
            if amount_terms:
                flow_items = {item for item, _, _ in amount_terms}
                flow_periods = {period for _, _, period in amount_terms}
                flow_machines = {machine for _, machine, _ in amount_terms}
                if len(flow_items) != 1 or len(flow_periods) != 1:
                    return {}
                if flow_machines != set(machines):
                    return {}
                demand[(next(iter(flow_items)), next(iter(flow_periods)))] = rhs_eq
            continue

        if _lp_rhs_number(body, "<=") is None:
            continue
        production_match = re.search(
            rf"(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+Production\[(?P<item>\d+),(?P<machine>\d+),(?P<period>\d+)\]",
            body,
        )
        startup_match = re.search(
            rf"(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+Startup\[(?P<item>\d+),(?P<machine>\d+),(?P<period>\d+)\]",
            body,
        )
        amount_match = re.search(
            r"Amount\[(?P<item>\d+),(?P<machine>\d+),(?P<period>\d+)\]",
            body,
        )
        if not production_match or not startup_match or not amount_match:
            continue
        prod_index = (
            int(production_match.group("item")),
            int(production_match.group("machine")),
            int(production_match.group("period")),
        )
        startup_index = (
            int(startup_match.group("item")),
            int(startup_match.group("machine")),
            int(startup_match.group("period")),
        )
        amount_index = (
            int(amount_match.group("item")),
            int(amount_match.group("machine")),
            int(amount_match.group("period")),
        )
        if prod_index != startup_index or prod_index != amount_index:
            return {}
        _, machine, _ = prod_index
        production_coefficient = float(production_match.group("coef").replace(" ", ""))
        startup_coefficient = float(startup_match.group("coef").replace(" ", ""))
        if production_coefficient >= 0 or startup_coefficient <= 0:
            return {}
        machine_capacity_samples.setdefault(machine, []).append(abs(production_coefficient))
        machine_startup_samples.setdefault(machine, []).append(startup_coefficient)

    expected_two = {(item, period) for item in items for period in periods}
    if expected_two - set(demand) or expected_two - set(holding_costs) or expected_two - set(backlog_costs):
        return {}

    holding_by_item: dict[int, float] = {}
    backlog_by_item: dict[int, float] = {}
    for item in items:
        holding_values = [holding_costs[(item, period)] for period in periods]
        backlog_values = [backlog_costs[(item, period)] for period in periods]
        if not all(_lp_numbers_close(value, holding_values[0]) for value in holding_values):
            return {}
        if not all(_lp_numbers_close(value, backlog_values[0]) for value in backlog_values):
            return {}
        holding_by_item[item] = holding_values[0]
        backlog_by_item[item] = backlog_values[0]

    capacity_by_machine: dict[int, float] = {}
    startup_time_by_machine: dict[int, float] = {}
    expected_capacity_samples = len(items) * len(periods)
    for machine in machines:
        capacities = machine_capacity_samples.get(machine) or []
        startup_times = machine_startup_samples.get(machine) or []
        if len(capacities) != expected_capacity_samples or len(startup_times) != expected_capacity_samples:
            return {}
        if not all(_lp_numbers_close(value, capacities[0]) for value in capacities):
            return {}
        if not all(_lp_numbers_close(value, startup_times[0]) for value in startup_times):
            return {}
        capacity_by_machine[machine] = capacities[0]
        startup_time_by_machine[machine] = startup_times[0]

    return {
        "sets": {
            "items": items,
            "machines": machines,
            "periods": periods,
        },
        "global_costs": {
            "setup_cost_per_active_item_machine_period": _lp_clean_number(setup_cost),
            "startup_cost_per_start_event": _lp_clean_number(startup_cost),
            "omitted_objective_terms": ["per-unit production cost"],
            "source_objective_note": "There is no per-unit production cost term in this source model.",
        },
        "item_cost_table": {
            "columns": ["item", "holding_cost", "backlog_cost"],
            "rows": [
                [
                    item,
                    _lp_clean_number(holding_by_item[item]),
                    _lp_clean_number(backlog_by_item[item]),
                ]
                for item in items
            ],
        },
        "machine_table": {
            "columns": ["machine", "capacity", "startup_time"],
            "rows": [
                [
                    machine,
                    _lp_clean_number(capacity_by_machine[machine]),
                    _lp_clean_number(startup_time_by_machine[machine]),
                ]
                for machine in machines
            ],
        },
        "demand_matrix": {
            "columns": periods,
            "rows": [
                {
                    "item": item,
                    "values": [_lp_clean_number(demand[(item, period)]) for period in periods],
                }
                for item in items
            ],
        },
        "source_model_note": (
            "Legacy LP-derived fact card. Demand, scalar setup/startup costs, item holding/backlog costs, "
            "and machine capacity/startup-time values are reconstructed from the verified small-bucket LP. "
            "The source objective has no per-unit production cost term."
        ),
    }


def _legacy_net1_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    cities = list(params.get("cities") or [])
    supply = params.get("supply") if isinstance(params.get("supply"), dict) else {}
    demand = params.get("demand") if isinstance(params.get("demand"), dict) else {}
    net_balance = params.get("net_balance") if isinstance(params.get("net_balance"), dict) else {}
    costs = params.get("shipping_cost") if isinstance(params.get("shipping_cost"), dict) else {}
    capacities = params.get("capacity") if isinstance(params.get("capacity"), dict) else {}
    arc_rows = []
    for link in params.get("links") or []:
        if not isinstance(link, (list, tuple)) or len(link) < 2:
            continue
        start, end = str(link[0]), str(link[1])
        key = f"{start}|{end}"
        arc_rows.append(
            {
                "from": start,
                "to": end,
                "unit_arc_flow_cost": costs.get(key),
                "arc_capacity": capacities.get(key),
            }
        )
    return {
        "model_family": "capacitated_min_cost_network_flow",
        "cities": cities,
        "node_balance_table": [
            {
                "city": city,
                "supply": supply.get(city, 0),
                "demand": demand.get(city, 0),
                "net_supply_minus_demand": net_balance.get(city, _safe_number(supply.get(city)) - _safe_number(demand.get(city))),
            }
            for city in cities
        ],
        "directed_arc_table": arc_rows,
        "total_supply": params.get("total_supply", sum(_safe_number(value) for value in supply.values())),
        "total_demand": params.get("total_demand", sum(_safe_number(value) for value in demand.values())),
        "source_model_note": (
            "Legacy generation_params fact card. Use supply[node] + inbound_flow[node] = "
            "demand[node] + outbound_flow[node] on declared directed arcs only."
        ),
    }


def _legacy_netasgn_compact_tables(params: dict[str, Any]) -> dict[str, Any]:
    people = list(params.get("people") or [])
    projects = list(params.get("projects") or [])
    supply_hours = params.get("supply_hours") if isinstance(params.get("supply_hours"), dict) else {}
    demand_hours = params.get("demand_hours") if isinstance(params.get("demand_hours"), dict) else {}
    cost_per_hour = params.get("cost_per_hour") if isinstance(params.get("cost_per_hour"), dict) else {}
    max_contribution = (
        params.get("max_contribution_hours")
        if isinstance(params.get("max_contribution_hours"), dict)
        else {}
    )
    return {
        "model_family": "continuous_resource_assignment_hours",
        "sets": {
            "people": people,
            "projects": projects,
        },
        "person_supply_table": [
            {"person": person, "available_hours": supply_hours.get(person)}
            for person in people
        ],
        "project_demand_table": [
            {"project": project, "required_hours": demand_hours.get(project)}
            for project in projects
        ],
        "cost_per_hour_matrix": {
            "columns": projects,
            "rows": [
                {
                    "person": person,
                    "values": [
                        (cost_per_hour.get(person) or {}).get(project)
                        for project in projects
                    ],
                }
                for person in people
            ],
        },
        "max_contribution_hours_matrix": {
            "columns": projects,
            "rows": [
                {
                    "person": person,
                    "values": [
                        (max_contribution.get(person) or {}).get(project)
                        for project in projects
                    ],
                }
                for person in people
            ],
        },
        "total_supply_hours": params.get("total_supply_hours") or sum(
            _safe_number(value) for value in supply_hours.values()
        ),
        "total_demand_hours": params.get("total_demand_hours") or sum(
            _safe_number(value) for value in demand_hours.values()
        ),
        "source_model_note": (
            "Continuous nonnegative assignment hours. Person supply and project demand are equalities; "
            "every person-project upper bound is enforced. No binary one-to-one matching."
        ),
    }


def _legacy_structure_assignment_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    if not lp_text.strip() or "total_assignments" not in lp_text or "NOE_" not in lp_text:
        return {}
    objective_match = re.search(r"Minimize\s+(.*?)\nSubject To", lp_text, flags=re.DOTALL | re.IGNORECASE)
    if not objective_match:
        return {}
    costs: dict[tuple[int, int], float] = {}
    for match in re.finditer(
        r"(?:(?P<coef>[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+))\s+|(?P<sign>[+-])\s*)?x\[(?P<peak>\d+),(?P<acid>\d+)\]",
        objective_match.group(1),
    ):
        if match.group("coef") is not None:
            coefficient = float(match.group("coef").replace(" ", ""))
        else:
            coefficient = -1.0 if match.group("sign") == "-" else 1.0
        peak = int(match.group("peak"))
        acid = int(match.group("acid"))
        costs[(peak, acid)] = coefficient
    if not costs:
        return {}
    all_variables = set(costs)
    binaries_match = re.search(r"Binaries\s+(.*?)\nEnd", lp_text, flags=re.DOTALL | re.IGNORECASE)
    if binaries_match:
        for match in re.finditer(r"x\[(\d+),(\d+)\]", binaries_match.group(1)):
            all_variables.add((int(match.group(1)), int(match.group(2))))
    peaks = sorted({peak for peak, _ in all_variables})
    acids = sorted({acid for _, acid in all_variables})
    if len(all_variables) != len(peaks) * len(acids):
        return {}
    for peak in peaks:
        for acid in acids:
            costs.setdefault((peak, acid), 0.0)

    assignment_match = re.search(r"total_assignments:.*?=\s*(\d+)", lp_text, flags=re.DOTALL)
    if not assignment_match:
        return {}
    required_assignment_count = int(assignment_match.group(1))

    compatibility: dict[tuple[int, int], int] = {(acid, acid): 1 for acid in acids}
    noe_pairs: set[tuple[int, int]] = set()
    for match in re.finditer(
        r"NOE_(\d+)_(\d+)_(\d+)_(\d+):[^\n]*?<=\s*([12](?:\.0)?)",
        lp_text,
    ):
        peak = int(match.group(1))
        related_peak = int(match.group(2))
        acid = int(match.group(3))
        other_acid = int(match.group(4))
        rhs = float(match.group(5))
        if peak not in peaks or related_peak not in peaks or acid not in acids or other_acid not in acids:
            return {}
        noe_pairs.add((peak, related_peak))
        compatibility[(acid, other_acid)] = int(round(rhs - 1))

    if noe_pairs:
        missing_pairs = [
            (acid, other_acid)
            for acid in acids
            for other_acid in acids
            if acid != other_acid and (acid, other_acid) not in compatibility
        ]
        if missing_pairs:
            return {}

    return {
        "sets": {
            "peaks": peaks,
            "amino_acids": acids,
        },
        "required_assignment_count": required_assignment_count,
        "distance_threshold": "not available in legacy LP; compatibility matrix is authoritative",
        "assignment_cost_matrix": {
            "columns": acids,
            "rows": [
                {
                    "peak": peak,
                    "values": [costs[(peak, acid)] for acid in acids],
                }
                for peak in peaks
            ],
        },
        "acid_compatibility_matrix": {
            "columns": acids,
            "rows": [
                {
                    "amino_acid": acid,
                    "values": [compatibility[(acid, other_acid)] for other_acid in acids],
                }
                for acid in acids
            ],
        },
        "noe_relation_pairs": [[peak, related_peak] for peak, related_peak in sorted(noe_pairs)],
        "source_model_note": (
            "Legacy LP-derived fact card. Assignment costs, exact assignment count, NOE peak pairs, "
            "and amino-acid compatibility are reconstructed from the verified LP text. Distances are not available; "
            "use the compatibility matrix as authoritative."
        ),
    }


def _legacy_marketshare_compact_tables_from_lp(lp_text: str) -> dict[str, Any]:
    objective = _lp_objective_section(lp_text)
    profit = _lp_three_index_coefficients(objective, "supply")
    if not profit:
        return {}
    demand: dict[tuple[int, int], float] = {}
    for label, body in _iter_lp_constraints(lp_text):
        lowered_label = label.lower()
        if not lowered_label.startswith("demand_"):
            continue
        rhs = _lp_rhs_number(body, "=")
        if rhs is None:
            continue
        match = re.match(r"demand_(?P<market>\d+)_(?P<product>\d+)$", lowered_label)
        if match:
            demand[(int(match.group("market")), int(match.group("product")))] = rhs
            continue
        supply_terms = _lp_three_index_coefficients(body, "supply")
        if supply_terms:
            _, market, product = next(iter(supply_terms))
            demand[(market, product)] = rhs
    companies = sorted({company for company, _, _ in profit})
    markets = sorted({market for _, market, _ in profit} | {market for market, _ in demand})
    products = sorted({product for _, _, product in profit} | {product for _, product in demand})
    if not companies or not markets or not products or not demand:
        return {}
    demand_matrix = {
        "columns": products,
        "rows": [
            {
                "market": market,
                "values": [_lp_clean_number(demand.get((market, product), 0.0)) for product in products],
            }
            for market in markets
        ],
    }
    unit_profit_matrices_by_company = {
        "columns": products,
        "rows": [
            {
                "company": company,
                "market": market,
                "values": [_lp_clean_number(profit.get((company, market, product), 0.0)) for product in products],
            }
            for company in companies
            for market in markets
        ],
    }
    return {
        "model_family": "integer_market_share_allocation",
        "sets": {
            "companies": companies,
            "markets": markets,
            "products": products,
        },
        "demand_matrix": demand_matrix,
        "unit_profit_matrices_by_company": unit_profit_matrices_by_company,
        "demand_table": [
            {
                "market": market,
                "product": product,
                "demand": _lp_clean_number(demand[market, product]),
            }
            for market, product in sorted(demand)
        ],
        "profit_table_format": "see unit_profit_matrices_by_company; each row is a company-market pair with product columns",
        "decision_variables": [
            "Supply[i,j,k] is the nonnegative integer quantity supplied by company i to market j for product k",
        ],
        "objective": "maximize sum_{i,j,k} unit_profit[i,j,k] * Supply[i,j,k]",
        "constraints": [
            "for every market-product pair (j,k), sum_i Supply[i,j,k] == demand[j,k]",
            "Supply[i,j,k] is nonnegative integer for every declared company-market-product triple",
        ],
        "source_contract_note": (
            "This source model has exact market-product demand fulfillment and no resource capacity, "
            "budget, channel capacity, eligibility, or nonlinear market-response constraints."
        ),
    }


def _has_keys(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in mapping for key in keys)


def _nested_or(mapping: dict[str, Any], path: list[str], fallback: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return fallback
        current = current[key]
    return current


def _safe_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


_LP_NUMBER_RE = r"(?:\d+(?:\.\d*)?|\.\d+)"


def _lp_objective_section(lp_text: str) -> str:
    match = re.search(r"(?:Minimize|Maximize)\s+(.*?)\nSubject To", lp_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _iter_lp_constraints(lp_text: str) -> list[tuple[str, str]]:
    match = re.search(
        r"\nSubject To\s+(.*?)(?:\nBounds|\nBinaries|\nGenerals|\nEnd)",
        lp_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return []
    constraints: list[tuple[str, str]] = []
    current_label = ""
    current_parts: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label_match = re.match(r"(?P<label>[A-Za-z_][A-Za-z0-9_\.]*):\s*(?P<body>.*)", line)
        if label_match:
            if current_label:
                constraints.append((current_label, " ".join(current_parts)))
            current_label = label_match.group("label")
            current_parts = [label_match.group("body").strip()]
        elif current_label:
            current_parts.append(line)
    if current_label:
        constraints.append((current_label, " ".join(current_parts)))
    return constraints


def _lp_rhs_number(body: str, sense: str) -> float | None:
    if sense == "=":
        pattern = rf"(?<![<>])=\s*(?P<rhs>[+-]?{_LP_NUMBER_RE})"
    else:
        pattern = rf"{re.escape(sense)}\s*(?P<rhs>[+-]?{_LP_NUMBER_RE})"
    match = re.search(pattern, body)
    if not match:
        return None
    return float(match.group("rhs"))


def _lp_match_coefficient(match: re.Match[str]) -> float:
    if match.group("coef") is not None:
        return float(match.group("coef").replace(" ", ""))
    return -1.0 if match.group("sign") == "-" else 1.0


def _lp_single_index_coefficients(section: str, variable_name: str) -> dict[str, float]:
    pattern = re.compile(
        rf"(?:(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+|(?P<sign>[+-])\s*)?"
        rf"{re.escape(variable_name)}\[(?P<index>[^\]]+)\]"
    )
    return {
        match.group("index"): _lp_match_coefficient(match)
        for match in pattern.finditer(section)
    }


def _lp_two_index_coefficients(section: str, variable_name: str) -> dict[tuple[int, int], float]:
    pattern = re.compile(
        rf"(?:(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+|(?P<sign>[+-])\s*)?"
        rf"{re.escape(variable_name)}\[(?P<first>\d+),(?P<second>\d+)\]"
    )
    return {
        (int(match.group("first")), int(match.group("second"))): _lp_match_coefficient(match)
        for match in pattern.finditer(section)
    }


def _lp_three_index_coefficients(section: str, variable_name: str) -> dict[tuple[int, int, int], float]:
    pattern = re.compile(
        rf"(?:(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+|(?P<sign>[+-])\s*)?"
        rf"{re.escape(variable_name)}\[(?P<first>\d+),(?P<second>\d+),(?P<third>\d+)\]"
    )
    return {
        (int(match.group("first")), int(match.group("second")), int(match.group("third"))): _lp_match_coefficient(match)
        for match in pattern.finditer(section)
    }


def _lp_underscore_two_index_coefficients(section: str, variable_prefix: str) -> dict[tuple[int, int], float]:
    pattern = re.compile(
        rf"(?:(?P<coef>[+-]?\s*{_LP_NUMBER_RE})\s+|(?P<sign>[+-])\s*)?"
        rf"{re.escape(variable_prefix)}_(?P<first>\d+)_(?P<second>\d+)"
    )
    return {
        (int(match.group("first")), int(match.group("second"))): _lp_match_coefficient(match)
        for match in pattern.finditer(section)
    }


def _period_label_sort_key(period: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(period))
    if not match:
        return (10**9, str(period))
    return (int(match.group(1)), str(period))


def _lp_numbers_close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _lp_clean_number(value: float) -> int | float:
    rounded = round(value)
    if _lp_numbers_close(value, float(rounded)):
        return int(rounded)
    return round(value, 6)


def _compact_json_artifact(value: Any, *, max_chars: int) -> dict[str, Any]:
    if value in (None, {}, []):
        return {"available": False}
    text = _stable_json(value if isinstance(value, dict) else {"value": value})
    if len(text) <= max_chars:
        return {"available": True, "truncated": False, "value": value}
    return {
        "available": True,
        "truncated": True,
        "original_chars": len(text),
        "preview": text[:max_chars],
    }


def _combine_background_and_statement(background: str, statement: str) -> str:
    """Keep the business setting in the text consumed by later stages."""
    clean_background = " ".join(background.split())
    clean_statement = statement.strip()
    if not clean_background:
        return clean_statement
    if clean_background.lower() in " ".join(clean_statement.lower().split()):
        return clean_statement
    if not clean_statement:
        return clean_background
    return f"{clean_background}\n\n{clean_statement}"


def _payload_text(payload: dict[str, Any], *keys: str) -> str:
    direct = _payload_text_from_mapping(payload, keys)
    if direct:
        return direct
    for nested in _payload_nested_mappings(payload):
        nested_text = _payload_text_from_mapping(nested, keys)
        if nested_text:
            return nested_text
    return ""


def _payload_text_from_mapping(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _payload_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    direct = _payload_mapping_from_mapping(payload, keys)
    if direct:
        return direct
    for nested in _payload_nested_mappings(payload):
        nested_mapping = _payload_mapping_from_mapping(nested, keys)
        if nested_mapping:
            return nested_mapping
    return {}


def _payload_mapping_from_mapping(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _payload_nested_mappings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nested: list[dict[str, Any]] = []
    for key in ("problem", "task", "output", "result", "response"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested.append(value)
    return nested


def _ensure_required_problem_statement_tables(
    row: dict[str, Any],
    statement: str,
    *,
    source_compact_data: dict[str, Any],
) -> str:
    generator_id = str(row.get("generator_id") or "").lower()
    if "optmath_clsp_expand_capacity" in generator_id:
        statement = _compact_clsp_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if (
        "optmath_uncapacitatedlotsizingbacklogging" in generator_id
        or "uncapacitatedlotsizingbacklogging" in generator_id
        or "optmath_uncapacitatedlotsizing" in generator_id
    ):
        statement = _compact_uncapacitated_lotsizing_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_net1" in generator_id or generator_id.endswith("net1") or generator_id == "net1":
        statement = _compact_net1_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_netasgn" in generator_id or generator_id.endswith("netasgn") or generator_id == "netasgn":
        statement = _compact_netasgn_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_electrical_power" in generator_id or "electrical_power" in generator_id:
        statement = _compact_electrical_power_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_marketshare" in generator_id or "marketshare" in generator_id:
        statement = _compact_marketshare_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_singlelevelsmallbucket" in generator_id or "singlelevelsmallbucket" in generator_id:
        statement = _compact_smallbucket_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_steel4" in generator_id or "steel4" in generator_id:
        statement = _compact_steel4_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_steel3" in generator_id or "steel3" in generator_id:
        statement = _compact_steel3_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_structure_based_assignment" in generator_id or "structure_based_assignment" in generator_id:
        statement = _compact_structure_assignment_statement_if_needed(
            statement,
            source_compact_data=source_compact_data,
        )
    if "optmath_aircraftlanding" not in generator_id and "aircraftlanding" not in generator_id:
        return statement
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    landing_tables = source_value.get("compact_aircraft_landing_tables")
    if not isinstance(landing_tables, dict):
        return statement
    required_numbers = _aircraft_landing_table_numbers(landing_tables)
    if not required_numbers:
        return statement
    statement_numbers = _canonical_number_set(statement)
    if len(required_numbers & statement_numbers) / max(1, len(required_numbers)) >= 0.9:
        return statement
    table_text = _format_aircraft_landing_tables(landing_tables)
    if not table_text:
        return statement
    if "Aircraft landing data:" in statement:
        return statement
    return f"{statement.rstrip()}\n\n{table_text}"


def _compact_clsp_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_lotsizing_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_clsp_statement(tables)
    return compact or statement


def _compact_uncapacitated_lotsizing_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_lotsizing_tables")
    if not isinstance(tables, dict):
        return statement
    model_family = str(tables.get("model_family") or "").lower()
    if model_family == "uncapacitated_lot_sizing_with_backlogging":
        return _format_ulsb_statement(tables) or statement
    if model_family == "uncapacitated_lot_sizing_without_backlog":
        return _format_uls_statement(tables) or statement
    return statement


def _format_uls_statement(tables: dict[str, Any]) -> str:
    periods = tables.get("periods") or []
    rows = tables.get("period_cost_table") or []
    total_demand_big_m = tables.get("total_demand_big_m")
    initial_inventory = tables.get("initial_inventory", 0)
    required_final_inventory = tables.get("required_final_inventory", 0)
    if not rows:
        return ""
    lines = [
        "A replenishment planner must create an uncapacitated lot-sizing plan without backlog after a forecast reset. "
        f"The planning periods are {periods}. The planner chooses continuous nonnegative OrderedAmount[t], "
        "continuous nonnegative EndingInventory[t], and binary OrderIsPlaced[t]. "
        "The objective is to minimize fixed ordering cost, unit order cost, and ending-inventory holding cost. "
        "There are no backlog variables, no backlog penalties, no lost sales, no unmet-demand slack, and no production capacity constraints.",
        "",
        f"Initial inventory is {initial_inventory}. Final inventory in the last period must be {required_final_inventory}. "
        f"Use total-demand big-M {total_demand_big_m} for OrderedAmount[t] <= big_M * OrderIsPlaced[t].",
        "",
        "Period rows use period, demand, fixed_ordering_cost, unit_order_cost, unit_holding_cost:",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"period={row.get('period')}, demand={row.get('demand')}, "
            f"fixed={row.get('fixed_ordering_cost')}, "
            f"unit_order={row.get('unit_order_cost')}, "
            f"holding={row.get('unit_holding_cost')}"
        )
    lines.append(
        "Constraints: EndingInventory[t] = previous EndingInventory + OrderedAmount[t] - demand[t], "
        "with previous inventory equal to the initial inventory in the first period; "
        "OrderedAmount[t] is linked to OrderIsPlaced[t] by the total-demand big-M; "
        "EndingInventory is nonnegative in every period and the final period inventory is fixed to the required final value."
    )
    return "\n".join(lines)


def _format_ulsb_statement(tables: dict[str, Any]) -> str:
    periods = tables.get("periods") or []
    rows = tables.get("period_cost_table") or []
    total_demand_big_m = tables.get("total_demand_big_m")
    initial_inventory = tables.get("initial_inventory", 0)
    initial_backlog = tables.get("initial_backlog", 0)
    required_final_inventory = tables.get("required_final_inventory", 0)
    required_final_backlog = tables.get("required_final_backlog", 0)
    if not rows:
        return ""
    lines = [
        "A replenishment planner must create an uncapacitated lot-sizing plan with carried backlog after a demand reset. "
        f"The planning periods are {periods}. The planner chooses continuous nonnegative OrderedAmount[t], "
        "continuous nonnegative EndingInventory[t], continuous nonnegative BackloggedAmount[t], and binary OrderIsPlaced[t]. "
        "The objective is to minimize fixed ordering cost, unit order cost, ending-inventory holding cost, and ending-backlog penalty. "
        "There is no production capacity constraint, no lost sales, and no per-period independent shortage slack.",
        "",
        f"Initial inventory is {initial_inventory} and initial backlog is {initial_backlog}. "
        f"Final inventory in the last period must be {required_final_inventory} and final backlog must be {required_final_backlog}. "
        f"Use total-demand big-M {total_demand_big_m} for OrderedAmount[t] <= big_M * OrderIsPlaced[t].",
        "",
        "Period rows use period, demand, fixed_ordering_cost, unit_order_cost, unit_holding_cost, unit_backlog_penalty:",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"period={row.get('period')}, demand={row.get('demand')}, "
            f"fixed={row.get('fixed_ordering_cost')}, "
            f"unit_order={row.get('unit_order_cost')}, "
            f"holding={row.get('unit_holding_cost')}, "
            f"backlog_penalty={row.get('unit_backlog_penalty')}"
        )
    lines.append(
        "Constraints: net inventory is EndingInventory[t] - BackloggedAmount[t]. "
        "For each period, net inventory equals previous net inventory plus OrderedAmount[t] minus demand[t], "
        "with previous net inventory equal to initial inventory minus initial backlog in the first period; "
        "OrderedAmount[t] is linked to OrderIsPlaced[t] by the total-demand big-M; "
        "EndingInventory and BackloggedAmount are nonnegative, and the final period inventory and backlog are fixed to the required final values."
    )
    return "\n".join(lines)


def _compact_net1_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_network_flow_tables") or source_value.get("compact_net1_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_net1_statement(tables)
    return compact or statement


def _format_net1_statement(tables: dict[str, Any]) -> str:
    cities = tables.get("cities") or []
    node_rows = tables.get("node_balance_table") or []
    arc_rows = tables.get("directed_arc_table") or []
    total_supply = tables.get("total_supply")
    total_demand = tables.get("total_demand")
    if not node_rows or not arc_rows:
        return ""
    lines = [
        "A logistics planner must solve a continuous capacitated minimum-cost network-flow problem after a network dispatch review. "
        f"The nodes are {cities}. Decision variable Ship[i,j] is the continuous nonnegative flow on declared directed arc i->j only. "
        "The objective is to minimize total arc unit cost times shipped flow. "
        "Use separate nonnegative supply and demand values from the node table; do not convert them into an ambiguous signed net-demand table. "
        "For every node, enforce supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node]. "
        "Every directed arc has its own hard capacity. Do not add binary activation, vehicles, routes, time windows, unmet-demand slack, disposal slack, or arcs not listed below.",
        "",
        f"Total supply is {total_supply} and total demand is {total_demand}.",
        "",
        "Node balance rows use node, supply, demand, net_supply_minus_demand:",
    ]
    for row in node_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"node={row.get('city')}, supply={row.get('supply')}, "
            f"demand={row.get('demand')}, "
            f"net={row.get('net_supply_minus_demand')}"
        )
    lines.append("")
    lines.append("Directed arc rows use from, to, unit_arc_flow_cost, arc_capacity:")
    for row in arc_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"from={row.get('from')}, to={row.get('to')}, "
            f"cost={row.get('unit_arc_flow_cost')}, "
            f"capacity={row.get('arc_capacity')}"
        )
    return "\n".join(lines)


def _compact_netasgn_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_project_assignment_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_netasgn_statement(tables)
    return compact or statement


def _format_netasgn_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") if isinstance(tables.get("sets"), dict) else {}
    people = sets.get("people") or []
    projects = sets.get("projects") or []
    person_rows = tables.get("person_supply_table") or []
    project_rows = tables.get("project_demand_table") or []
    cost_matrix = tables.get("cost_per_hour_matrix") if isinstance(tables.get("cost_per_hour_matrix"), dict) else {}
    limit_matrix = (
        tables.get("max_contribution_hours_matrix")
        if isinstance(tables.get("max_contribution_hours_matrix"), dict)
        else {}
    )
    if not people or not projects or not person_rows or not project_rows:
        return ""
    lines = [
        "A resource planning manager must solve a continuous project-assignment hours problem after an allocation review. "
        f"The people/resources are {people}; the projects/requests are {projects}. "
        "Decision variable Assign[i,j] is the continuous nonnegative number of hours assigned from person i to project j. "
        "The objective is to minimize cost_per_hour[i,j] * Assign[i,j] over every listed person-project pair. "
        "Every person's available hours must be fully allocated with equality. "
        "Every project's required hours must be exactly satisfied with equality. "
        "Every listed person-project maximum contribution is a hard upper bound. "
        "Do not model this as binary one-to-one matching, route scheduling, shift sequencing, or optional coverage.",
        "",
        f"Total available hours are {tables.get('total_supply_hours')} and total required hours are {tables.get('total_demand_hours')}.",
        "",
        "Person supply rows use person, available_hours:",
    ]
    for row in person_rows:
        if isinstance(row, dict):
            lines.append(f"- person={row.get('person')}, available_hours={row.get('available_hours')}")
    lines.append("")
    lines.append("Project demand rows use project, required_hours:")
    for row in project_rows:
        if isinstance(row, dict):
            lines.append(f"- project={row.get('project')}, required_hours={row.get('required_hours')}")
    cost_columns = cost_matrix.get("columns") or []
    if cost_columns:
        lines.append("")
        lines.append("Cost per assigned hour matrix columns: " + ", ".join(str(value) for value in cost_columns))
        for row in cost_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"- person={row.get('person')}: " + ", ".join(str(value) for value in row.get("values") or []))
    limit_columns = limit_matrix.get("columns") or []
    if limit_columns:
        lines.append("")
        lines.append("Maximum contribution hours matrix columns: " + ", ".join(str(value) for value in limit_columns))
        for row in limit_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"- person={row.get('person')}: " + ", ".join(str(value) for value in row.get("values") or []))
    return "\n".join(lines)


def _compact_electrical_power_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_electrical_power_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_electrical_power_statement(tables)
    return compact or statement


def _format_electrical_power_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    generator_types = sets.get("generator_types") or []
    time_periods = sets.get("time_periods") or []
    generator_rows = tables.get("generator_type_table") or []
    demand_rows = tables.get("period_demand_table") or []
    if not generator_rows or not demand_rows:
        return ""
    reserve_margin = None
    for row in demand_rows:
        if not isinstance(row, dict):
            continue
        demand = row.get("demand_mw")
        reserve = row.get("reserve_required_capacity_mw")
        try:
            if demand:
                reserve_margin = round(float(reserve) / float(demand), 4)
                break
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    reserve_text = reserve_margin if reserve_margin is not None else "the stated reserve margin"
    lines = [
        "A microgrid operations team must prepare a deterministic unit-commitment plan after a demand forecast update. "
        f"The generator types are {generator_types} and the time periods are {time_periods}. "
        "For each generator type and period, decide NumGenerators[t,p] as a nonnegative integer on-count, "
        "PowerOutput[t,p] as continuous nonnegative MW output, and NumStart[t,p] as a nonnegative integer startup count. "
        "The objective is to minimize base operating cost, per-MW generation cost, and startup cost. "
        "This source model has no battery storage, fuel inventory, emissions cap, power-flow network, unit-specific identity, or minimum up/down time constraints.",
        "",
        "Generator type data rows use: type, base_cost, per_mw_cost, startup_cost, min_output, max_output, available_count, initially_on:",
    ]
    for row in generator_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"type={row.get('generator_type')}, "
            f"base={row.get('base_operating_cost_per_period')}, "
            f"per_mw={row.get('per_mw_generation_cost')}, "
            f"startup={row.get('startup_cost')}, "
            f"min_output={row.get('minimum_output_mw')}, "
            f"max_output={row.get('maximum_output_mw')}, "
            f"available={row.get('generators_available')}, "
            f"initially_on={row.get('generators_on_initially')}"
        )
    lines.append("")
    lines.append("Period demand rows use: period, demand_mw, reserve_required_capacity_mw:")
    for row in demand_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"period={row.get('period')}, "
            f"demand={row.get('demand_mw')}, "
            f"reserve_required={row.get('reserve_required_capacity_mw')}"
        )
    lines.append(
        "Constraints: on-count cannot exceed available count; total output must meet each period demand; "
        "output for each generator type-period must lie between min_output * NumGenerators and max_output * NumGenerators; "
        f"reserve requires total online max capacity at least {reserve_text} times demand in each period; "
        "initial startup satisfies NumGenerators[t,first] <= initially_on[t] + NumStart[t,first]; "
        "later startup satisfies NumGenerators[t,p] <= NumGenerators[t,previous] + NumStart[t,p]."
    )
    return "\n".join(lines)


def _compact_marketshare_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_marketshare_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_marketshare_statement(tables)
    return compact or statement


def _format_marketshare_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    companies = sets.get("companies") or []
    markets = sets.get("markets") or []
    products = sets.get("products") or []
    demand_rows = tables.get("demand_table") or []
    demand_matrix = tables.get("demand_matrix") if isinstance(tables.get("demand_matrix"), dict) else {}
    profit_rows = tables.get("profit_table") or []
    profit_matrix = (
        tables.get("unit_profit_matrices_by_company")
        if isinstance(tables.get("unit_profit_matrices_by_company"), dict)
        else {}
    )
    if not products and isinstance(demand_matrix, dict):
        products = list(demand_matrix.get("columns") or [])
    if not markets and isinstance(demand_matrix, dict):
        markets = [
            row.get("market")
            for row in demand_matrix.get("rows") or []
            if isinstance(row, dict) and row.get("market") is not None
        ]
    if not companies and isinstance(profit_matrix, dict):
        companies = [
            row.get("company")
            for row in profit_matrix.get("rows") or []
            if isinstance(row, dict) and row.get("company") is not None
        ]
        companies = _dedupe_keep_order(companies)
    has_demand_data = bool(demand_rows) or bool(demand_matrix.get("rows"))
    has_profit_data = bool(profit_rows) or bool(profit_matrix.get("rows"))
    if not companies or not markets or not products or not has_demand_data or not has_profit_data:
        return ""
    demand_by_market: dict[Any, dict[Any, Any]] = {}
    if demand_rows:
        for row in demand_rows:
            if not isinstance(row, dict):
                continue
            demand_by_market.setdefault(row.get("market"), {})[row.get("product")] = row.get("demand")
    else:
        matrix_columns = demand_matrix.get("columns") or products
        for row in demand_matrix.get("rows") or []:
            if not isinstance(row, dict):
                continue
            market = row.get("market")
            values = row.get("values") or []
            demand_by_market[market] = {
                product: values[index] if index < len(values) else None
                for index, product in enumerate(matrix_columns)
            }
    profit_by_company_market: dict[tuple[Any, Any], dict[Any, Any]] = {}
    if profit_rows:
        for row in profit_rows:
            if not isinstance(row, dict):
                continue
            key = (row.get("company"), row.get("market"))
            profit_by_company_market.setdefault(key, {})[row.get("product")] = row.get("unit_profit")
    else:
        matrix_columns = profit_matrix.get("columns") or products
        for row in profit_matrix.get("rows") or []:
            if not isinstance(row, dict):
                continue
            key = (row.get("company"), row.get("market"))
            values = row.get("values") or []
            profit_by_company_market[key] = {
                product: values[index] if index < len(values) else None
                for index, product in enumerate(matrix_columns)
            }
    lines = [
        "A market-allocation planning team must decide integer supply quantities after a demand commitment review. "
        f"The companies are {companies}, markets are {markets}, and products are {products}. "
        "Decision variable Supply[i,j,k] is the nonnegative integer quantity supplied by company i to market j for product k. "
        "The objective is to maximize total unit profit times Supply. "
        "For every market-product pair, total supply over all companies must exactly equal the stated demand. "
        "This source model has no resource capacity, budget, channel capacity, eligibility, price, nonlinear demand-response, or optional market-coverage constraints.",
        "",
        "Demand table by market (product=demand):",
    ]
    for market in markets:
        values = demand_by_market.get(market, {})
        entries = ", ".join(f"{product}={values.get(product)}" for product in products)
        lines.append(f"- market={market}: {entries}")
    lines.append("")
    lines.append("Unit-profit table by company and market (product=unit_profit):")
    for company in companies:
        for market in markets:
            values = profit_by_company_market.get((company, market), {})
            entries = ", ".join(f"{product}={values.get(product)}" for product in products)
            lines.append(f"- company={company}, market={market}: {entries}")
    lines.append(
        "Formulate the MILP with exact demand fulfillment sum_i Supply[i,j,k] == demand[j,k] for every market-product pair, "
        "nonnegative integer Supply variables for every declared company-market-product triple, and objective "
        "maximize sum_i,j,k unit_profit[i,j,k] * Supply[i,j,k]."
    )
    return "\n".join(lines)


def _format_clsp_statement(tables: dict[str, Any]) -> str:
    products = tables.get("products") or []
    periods = tables.get("periods") or []
    demand_rows = tables.get("period_demand_table") or []
    capacity_rows = tables.get("period_capacity_table") or []
    consumption_rows = tables.get("capacity_consumption_per_product") or []
    if not demand_rows:
        return ""
    lines = [
        "A production planning team must create a capacitated lot-sizing plan after a demand forecast reset. "
        f"There are {len(products)} products indexed {products} and {len(periods)} periods indexed {periods}. "
        "For each product-period pair, the team chooses a nonnegative production quantity, a binary setup indicator, "
        "and a nonnegative ending inventory level. The objective is to minimize fixed setup, unit production, "
        "and inventory holding costs. This source model has fixed period capacities; it has no backlog variables, "
        "no backlog penalty, no lost sales, no capacity-expansion decision, and no separate zero-final-inventory hard constraint.",
        "",
        "Product-period data rows use: p, t, period_demand, cumulative_demand_through_t, setup_cost, unit_production_cost, unit_holding_cost, setup_big_M. "
        "If the model is written recursively, use period_demand in each one-period inventory balance. "
        "Use cumulative_demand_through_t only in a cumulative balance expression or as a remaining-demand/big-M helper.",
    ]
    for row in demand_rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            f"p={row.get('product')}, t={row.get('period')}, "
            f"period_demand={row.get('period_demand')}, "
            f"cumulative_demand_through_t={row.get('cumulative_demand_through_period')}, "
            f"setup={row.get('setup_cost')}, "
            f"prod={row.get('unit_production_cost')}, "
            f"hold={row.get('unit_holding_cost')}, "
            f"M={row.get('setup_big_m_remaining_demand')}"
        )
    if consumption_rows:
        lines.append("Capacity consumption per unit:")
        lines.append(
            "- "
            + "; ".join(
                f"p={row.get('product')}: {row.get('capacity_consumption_per_unit')}"
                for row in consumption_rows
                if isinstance(row, dict)
            )
        )
    if capacity_rows:
        lines.append("Fixed period capacities:")
        lines.append(
            "- "
            + "; ".join(
                f"t={row.get('period')}: {row.get('capacity')}"
                for row in capacity_rows
                if isinstance(row, dict)
            )
        )
    lines.append(
        "Constraints: either use recursive inventory balance with period_demand, or cumulative inventory balance where ending inventory equals cumulative production minus cumulative demand; "
        "for every period, total capacity consumption of production cannot exceed the fixed capacity; "
        "production in a period is limited by setup_big_M times the binary setup indicator; inventory is nonnegative."
    )
    return "\n".join(lines)


def _compact_smallbucket_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_lotsizing_tables")
    if not isinstance(tables, dict):
        return statement
    lowered = statement.lower()
    has_wrong_unit_cost = any(
        phrase in lowered
        for phrase in (
            "production cost per unit",
            "per-unit production cost",
            "unit production cost",
            "production costs",
            "prod_cost",
        )
    )
    has_placeholder_or_truncation = any(
        phrase in lowered
        for phrase in (
            "placeholder",
            "truncated",
            "not fully provided",
            "assumed",
            "dummy values",
        )
    )
    if len(statement) <= 3800 and not has_wrong_unit_cost and not has_placeholder_or_truncation:
        return statement
    compact = _format_smallbucket_statement(tables)
    return compact or statement


def _format_smallbucket_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    items = sets.get("items") or []
    machines = sets.get("machines") or []
    periods = sets.get("periods") or []
    global_costs = tables.get("global_costs") or {}
    item_cost_table = tables.get("item_cost_table") or {}
    machine_table = tables.get("machine_table") or {}
    demand_matrix = tables.get("demand_matrix") or {}
    item_rows = item_cost_table.get("rows") or []
    machine_rows = machine_table.get("rows") or []
    demand_rows = demand_matrix.get("rows") or []
    demand_columns = demand_matrix.get("columns") or periods
    if not item_rows or not machine_rows or not demand_rows:
        return ""
    lines = [
        "A production scheduling team must create a single-level small-bucket lot-sizing plan after a capacity disruption. "
        f"There are items {items}, machines {machines}, and periods {periods}. The team chooses continuous Amount[i,m,t], "
        "binary Production[i,m,t], binary Startup[i,m,t], continuous Stock[i,t], and continuous Backlog[i,t]. "
        "The objective is to minimize setup cost for active item-machine-period states, startup cost for start events, "
        "inventory holding cost, and backlog cost. There is no per-unit production cost, material cost, or revenue term in the source objective.",
        "",
        "Global costs:",
        f"- setup_cost_per_active_item_machine_period={global_costs.get('setup_cost_per_active_item_machine_period')}",
        f"- startup_cost_per_start_event={global_costs.get('startup_cost_per_start_event')}",
        "",
        "Item costs use rows item, holding_cost, backlog_cost:",
    ]
    for row in item_rows:
        if isinstance(row, list) and len(row) >= 3:
            lines.append(f"- item={row[0]}, holding={row[1]}, backlog={row[2]}")
    lines.append("Machine data use rows machine, capacity, startup_time:")
    for row in machine_rows:
        if isinstance(row, list) and len(row) >= 3:
            lines.append(f"- machine={row[0]}, capacity={row[1]}, startup_time={row[2]}")
    lines.append("Demand matrix columns are periods: " + ", ".join(str(value) for value in demand_columns))
    for row in demand_rows:
        if not isinstance(row, dict):
            continue
        values = ", ".join(str(value) for value in row.get("values") or [])
        lines.append(f"- item={row.get('item')}: {values}")
    lines.append(
        "Constraints: previous Stock minus previous Backlog plus total Amount across machines equals demand plus current Stock minus current Backlog; "
        "Amount[i,m,t] plus startup_time[m] times Startup[i,m,t] cannot exceed capacity[m] times Production[i,m,t]; "
        "each machine produces at most one item per period; Startup tracks transitions from not producing to producing the same item on the same machine."
    )
    return "\n".join(lines)


def _compact_steel4_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_steel_product_mix_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_steel4_statement(tables)
    return compact or statement


def _compact_steel3_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_steel_product_mix_tables")
    if not isinstance(tables, dict):
        return statement
    if str(tables.get("model_family") or "").lower() != "single_stage_continuous_product_mix":
        return statement
    compact = _format_steel3_statement(tables)
    return compact or statement


def _format_steel3_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    products = sets.get("products") or []
    product_table = tables.get("product_table") or {}
    time_capacity = tables.get("time_capacity") if isinstance(tables.get("time_capacity"), dict) else {}
    rows = product_table.get("rows") or []
    available_hours = time_capacity.get("available_hours")
    if not rows or available_hours is None:
        return ""
    lines = [
        "A steel production planning team must set a profit-maximizing single-stage product mix after a capacity review. "
        f"The products are {products}. Decision variable Production[p] is the continuous nonnegative tons of product p to produce. "
        "The objective is to maximize total profit, sum over products of profit_per_ton[p] * Production[p]. "
        f"There is one shared production-hour capacity of {available_hours} hours: "
        "sum_p processing_hours_per_ton[p] * Production[p] <= available_hours. "
        "Each product has a minimum commitment lower bound and a maximum market upper bound. "
        "There are no inventory, backlog, setup, binary product-selection, sequencing, time periods, or multi-stage capacity matrix.",
        "",
        "Product data rows use product, profit_per_ton, production_rate_tons_per_hour, processing_hours_per_ton, minimum_commitment_tons, maximum_market_tons:",
    ]
    for row in rows:
        if isinstance(row, list) and len(row) >= 6:
            lines.append(
                f"- product={row[0]}, profit={row[1]}, production_rate={row[2]}, "
                f"processing_hours_per_ton={row[3]}, min_commit={row[4]}, max_market={row[5]}"
            )
    return "\n".join(lines)


def _format_steel4_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    products = sets.get("products") or []
    stages = sets.get("production_stages") or []
    product_table = tables.get("product_table") or {}
    stage_capacity_table = tables.get("stage_capacity_table") or {}
    processing_matrix = tables.get("processing_hours_per_ton_matrix") or {}
    product_rows = product_table.get("rows") or []
    stage_rows = stage_capacity_table.get("rows") or []
    processing_rows = processing_matrix.get("rows") or []
    processing_columns = processing_matrix.get("columns") or stages
    if not product_rows or not stage_rows or not processing_rows:
        return ""
    lines = [
        "A steel production planning team must set a profit-maximizing product mix after a capacity review. "
        f"There are products {products} and shared production stages {stages}. Decision variable Production[p] is the continuous nonnegative tons of product p to produce. "
        "The objective is to maximize total profit, sum over products of profit_per_ton[p] * Production[p]. "
        "Each product has a minimum commitment lower bound and a maximum market upper bound. "
        "Each stage has its own available processing hours; do not copy product market bounds into stage capacities. "
        "Stage capacity is sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]. "
        "There are no inventory, backlog, setup, binary product-selection, sequencing, or time-period variables.",
        "",
        "Product data rows use product, profit_per_ton, minimum_commitment_tons, maximum_market_tons:",
    ]
    for row in product_rows:
        if isinstance(row, list) and len(row) >= 4:
            lines.append(f"- product={row[0]}, profit={row[1]}, min_commit={row[2]}, max_market={row[3]}")
    lines.append("")
    lines.append("Stage available-hour rows use stage, available_hours:")
    for row in stage_rows:
        if isinstance(row, list) and len(row) >= 2:
            lines.append(f"- stage={row[0]}, available_hours={row[1]}")
    lines.append("")
    lines.append("Processing-hours-per-ton matrix columns are stages: " + ", ".join(str(value) for value in processing_columns))
    for row in processing_rows:
        if not isinstance(row, dict):
            continue
        values = ", ".join(str(value) for value in row.get("values") or [])
        lines.append(f"- product={row.get('product')}: {values}")
    return "\n".join(lines)


def _compact_structure_assignment_statement_if_needed(statement: str, *, source_compact_data: dict[str, Any]) -> str:
    if source_compact_data.get("truncated"):
        return statement
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return statement
    tables = source_value.get("compact_structure_assignment_tables")
    if not isinstance(tables, dict):
        return statement
    compact = _format_structure_assignment_statement(tables)
    return compact or statement


def _format_structure_assignment_statement(tables: dict[str, Any]) -> str:
    sets = tables.get("sets") or {}
    peaks = sets.get("peaks") or []
    acids = sets.get("amino_acids") or []
    required_count = tables.get("required_assignment_count")
    threshold = tables.get("distance_threshold")
    cost_matrix = tables.get("assignment_cost_matrix") or {}
    compatibility_matrix = tables.get("acid_compatibility_matrix") or {}
    noe_pairs = tables.get("noe_relation_pairs") or []
    cost_rows = cost_matrix.get("rows") or []
    cost_columns = cost_matrix.get("columns") or acids
    compatibility_rows = compatibility_matrix.get("rows") or []
    compatibility_columns = compatibility_matrix.get("columns") or acids
    if not cost_rows or not compatibility_rows:
        return ""
    lines = [
        "An NMR spectroscopy analysis team must assign observed spectral peaks to amino acids after a protein-structure review. "
        f"The peaks are indexed {peaks}, the amino acids are indexed {acids}, and exactly {required_count} peak-amino-acid assignments must be selected. "
        "Use binary decision variable x[p,a] = 1 if peak p is assigned to amino acid a. Each peak can be assigned to at most one amino acid, "
        "and each amino acid can receive at most one peak. The objective is to minimize total assignment cost sum c[p,a] * x[p,a]. "
        f"For each listed NOE-related peak pair, the selected amino-acid pair must be compatible; compatibility[a,b] = 1 means the amino-acid distance is below threshold {threshold}, "
        "and compatibility[a,b] = 0 means the two corresponding assignments cannot both be selected. Do not assume missing compatibility values: the full compatibility matrix below is authoritative.",
        "",
        "Assignment cost matrix c[p,a] columns are amino acids: " + ", ".join(str(value) for value in cost_columns),
    ]
    for row in cost_rows:
        if not isinstance(row, dict):
            continue
        values = ", ".join(str(value) for value in row.get("values") or [])
        lines.append(f"- peak={row.get('peak')}: {values}")
    lines.append("")
    if noe_pairs:
        pair_text = "; ".join(f"({pair[0]},{pair[1]})" for pair in noe_pairs if isinstance(pair, list) and len(pair) >= 2)
    else:
        pair_text = "none"
    lines.append("NOE-related peak pairs: " + pair_text)
    lines.append("")
    lines.append("Amino-acid compatibility matrix columns are amino acids: " + ", ".join(str(value) for value in compatibility_columns))
    for row in compatibility_rows:
        if not isinstance(row, dict):
            continue
        values = ", ".join(str(value) for value in row.get("values") or [])
        lines.append(f"- amino_acid={row.get('amino_acid')}: {values}")
    lines.append(
        "Formulate and solve the binary MILP with the exact assignment-count constraint, at-most-one constraints for peaks and amino acids, "
        "and NOE compatibility constraints x[p,a] + x[q,b] <= compatibility[a,b] + 1 for every listed NOE pair (p,q) and amino-acid pair (a,b)."
    )
    return "\n".join(lines)


def _structured_problem_data_from_compact_source(
    row: dict[str, Any],
    *,
    source_compact_data: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    generator_id = str(row.get("generator_id") or "").lower()
    supports_rebuild = any(
        marker in generator_id
        for marker in (
            "optmath_structure_based_assignment",
            "structure_based_assignment",
            "optmath_steel4",
            "steel4",
            "optmath_steel3",
            "steel3",
            "optmath_clsp_expand_capacity",
            "optmath_uncapacitatedlotsizingbacklogging",
            "uncapacitatedlotsizingbacklogging",
            "optmath_uncapacitatedlotsizing",
            "optmath_net1",
            "optmath_electrical_power",
            "electrical_power",
            "optmath_marketshare",
            "marketshare",
        )
    )
    if not supports_rebuild:
        return fallback
    if source_compact_data.get("truncated"):
        return fallback
    source_value = source_compact_data.get("value")
    if not isinstance(source_value, dict):
        return fallback
    steel_tables = source_value.get("compact_steel_product_mix_tables")
    if ("optmath_steel3" in generator_id or "steel3" in generator_id) and isinstance(steel_tables, dict):
        sets = steel_tables.get("sets") or {}
        return {
            "entities": ["steel products", "single shared production-hour capacity"],
            "sets": {
                "P": sets.get("products") or [],
            },
            "parameters": {
                "product_table": "see source_compact_data.compact_steel_product_mix_tables.product_table",
                "time_capacity": "see source_compact_data.compact_steel_product_mix_tables.time_capacity",
            },
            "variables": {
                "Production[p]": "continuous nonnegative tons of product p to produce",
            },
            "constraints": {
                "time_capacity": "sum_p processing_hours_per_ton[p] * Production[p] <= available_hours",
                "minimum_commitment": "Production[p] >= minimum_commitment_tons[p] for every product p",
                "maximum_market_bound": "Production[p] <= maximum_market_tons[p] for every product p",
            },
            "objective": "maximize total profit_per_ton[p] * Production[p]",
            "forbidden_structures": ["inventory", "backlog", "setup", "binary product selection", "sequencing", "multi-stage capacity matrix"],
        }
    if ("optmath_steel4" in generator_id or "steel4" in generator_id) and isinstance(steel_tables, dict):
        sets = steel_tables.get("sets") or {}
        return {
            "entities": ["steel products", "shared production stages"],
            "sets": {
                "P": sets.get("products") or [],
                "S": sets.get("production_stages") or [],
            },
            "parameters": {
                "product_table": "see source_compact_data.compact_steel_product_mix_tables.product_table",
                "stage_capacity_table": "see source_compact_data.compact_steel_product_mix_tables.stage_capacity_table",
                "processing_hours_per_ton_matrix": "see source_compact_data.compact_steel_product_mix_tables.processing_hours_per_ton_matrix",
            },
            "variables": {
                "Production[p]": "continuous nonnegative tons of product p to produce",
            },
            "constraints": {
                "stage_time_capacity": "sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s] for every stage s",
                "minimum_commitment": "Production[p] >= minimum_commitment_tons[p] for every product p",
                "maximum_market_bound": "Production[p] <= maximum_market_tons[p] for every product p",
            },
            "objective": "maximize total profit_per_ton[p] * Production[p]",
            "forbidden_structures": ["inventory", "backlog", "setup", "binary product selection", "sequencing"],
        }
    lot_tables = source_value.get("compact_lotsizing_tables")
    if isinstance(lot_tables, dict):
        model_family = str(lot_tables.get("model_family") or "").lower()
        if model_family == "capacitated_lot_sizing_without_backlog":
            return {
                "entities": ["products", "planning periods"],
                "sets": {
                    "products": lot_tables.get("products") or [],
                    "periods": lot_tables.get("periods") or [],
                },
                "parameters": {
                    "period_demand_table": "see source_compact_data.compact_lotsizing_tables.period_demand_table",
                    "period_capacity_table": "see source_compact_data.compact_lotsizing_tables.period_capacity_table",
                    "capacity_consumption_per_product": "see source_compact_data.compact_lotsizing_tables.capacity_consumption_per_product",
                },
                "variables": {
                    "Production[i,t]": "continuous nonnegative production quantity",
                    "Setup[i,t]": "binary setup indicator",
                    "Inventory[i,t]": "continuous nonnegative ending inventory",
                },
                "constraints": {
                    "inventory_balance": "Inventory[i,t] is cumulative production through t minus cumulative demand through t; no backlog",
                    "capacity": "sum_i capacity_consumption[i] * Production[i,t] <= capacity[t]",
                    "setup_linking": "Production[i,t] <= remaining_demand_big_M[i,t] * Setup[i,t]",
                },
                "objective": "minimize setup, unit production, and inventory holding costs",
                "forbidden_structures": ["backlog", "lost sales", "capacity expansion", "zero final inventory hard constraint"],
            }
        if model_family == "uncapacitated_lot_sizing_without_backlog":
            return {
                "entities": ["planning periods", "orders", "ending inventory"],
                "sets": {"periods": lot_tables.get("periods") or []},
                "parameters": {
                    "period_cost_table": "see source_compact_data.compact_lotsizing_tables.period_cost_table",
                    "total_demand_big_m": lot_tables.get("total_demand_big_m"),
                    "initial_inventory": lot_tables.get("initial_inventory", 0),
                    "required_final_inventory": lot_tables.get("required_final_inventory", 0),
                },
                "variables": {
                    "OrderedAmount[t]": "continuous nonnegative order quantity",
                    "EndingInventory[t]": "continuous nonnegative ending inventory",
                    "OrderIsPlaced[t]": "binary order setup indicator",
                },
                "constraints": {
                    "inventory_balance": "EndingInventory[t] = previous EndingInventory + OrderedAmount[t] - demand[t]",
                    "setup_linking": "OrderedAmount[t] <= total_demand_big_M * OrderIsPlaced[t]",
                    "final_inventory": "final period EndingInventory equals required_final_inventory",
                },
                "objective": "minimize fixed ordering, unit order, and holding costs",
                "forbidden_structures": ["backlog", "lost sales", "capacity"],
            }
        if model_family == "uncapacitated_lot_sizing_with_backlogging":
            return {
                "entities": ["planning periods", "orders", "ending inventory", "carried backlog"],
                "sets": {"periods": lot_tables.get("periods") or []},
                "parameters": {
                    "period_cost_table": "see source_compact_data.compact_lotsizing_tables.period_cost_table",
                    "total_demand_big_m": lot_tables.get("total_demand_big_m"),
                    "initial_inventory": lot_tables.get("initial_inventory", 0),
                    "initial_backlog": lot_tables.get("initial_backlog", 0),
                    "required_final_inventory": lot_tables.get("required_final_inventory", 0),
                    "required_final_backlog": lot_tables.get("required_final_backlog", 0),
                },
                "variables": {
                    "OrderedAmount[t]": "continuous nonnegative order quantity",
                    "EndingInventory[t]": "continuous nonnegative ending inventory",
                    "BackloggedAmount[t]": "continuous nonnegative carried backlog",
                    "OrderIsPlaced[t]": "binary order setup indicator",
                },
                "constraints": {
                    "net_inventory_balance": "EndingInventory[t] - BackloggedAmount[t] equals previous net inventory plus OrderedAmount[t] minus demand[t]",
                    "setup_linking": "OrderedAmount[t] <= total_demand_big_M * OrderIsPlaced[t]",
                    "final_state": "final period EndingInventory and BackloggedAmount equal required final values",
                },
                "objective": "minimize fixed ordering, unit order, holding, and backlog penalty costs",
                "forbidden_structures": ["capacity", "lost sales", "independent shortage slack"],
            }
    net1_tables = source_value.get("compact_network_flow_tables") or source_value.get("compact_net1_tables")
    if ("optmath_net1" in generator_id or generator_id.endswith("net1") or generator_id == "net1") and isinstance(net1_tables, dict):
        return {
            "entities": ["network nodes", "declared directed arcs"],
            "sets": {
                "nodes": net1_tables.get("cities") or [],
                "directed_arcs": [
                    [row.get("from"), row.get("to")]
                    for row in net1_tables.get("directed_arc_table") or []
                    if isinstance(row, dict)
                ],
            },
            "parameters": {
                "node_balance_table": "see source_compact_data.compact_network_flow_tables.node_balance_table",
                "directed_arc_table": "see source_compact_data.compact_network_flow_tables.directed_arc_table",
                "total_supply": net1_tables.get("total_supply"),
                "total_demand": net1_tables.get("total_demand"),
                "flow_balance_convention": (
                    "for every node, supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node]"
                ),
            },
            "variables": {
                "Ship[i,j]": "continuous nonnegative flow on declared directed arc i->j",
            },
            "constraints": {
                "node_flow_balance": "supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node]",
                "arc_capacity": "Ship[i,j] <= arc_capacity[i,j] for every declared directed arc",
            },
            "objective": "minimize sum of unit_arc_flow_cost[i,j] * Ship[i,j]",
            "forbidden_structures": ["binary activation", "vehicles", "routes", "time windows", "unmet-demand slack", "unlisted arcs"],
        }
    netasgn_tables = source_value.get("compact_project_assignment_tables")
    if (
        "optmath_netasgn" in generator_id
        or generator_id.endswith("netasgn")
        or generator_id == "netasgn"
    ) and isinstance(netasgn_tables, dict):
        sets = netasgn_tables.get("sets") if isinstance(netasgn_tables.get("sets"), dict) else {}
        return {
            "entities": ["people/resources", "projects/requests"],
            "sets": {
                "people": sets.get("people") or [],
                "projects": sets.get("projects") or [],
            },
            "parameters": {
                "person_supply_table": "see source_compact_data.compact_project_assignment_tables.person_supply_table",
                "project_demand_table": "see source_compact_data.compact_project_assignment_tables.project_demand_table",
                "cost_per_hour_matrix": "see source_compact_data.compact_project_assignment_tables.cost_per_hour_matrix",
                "max_contribution_hours_matrix": (
                    "see source_compact_data.compact_project_assignment_tables.max_contribution_hours_matrix"
                ),
                "total_supply_hours": netasgn_tables.get("total_supply_hours"),
                "total_demand_hours": netasgn_tables.get("total_demand_hours"),
            },
            "variables": {
                "Assign[i,j]": "continuous nonnegative hours assigned from person/resource i to project/request j",
            },
            "constraints": {
                "person_supply_hours_equality": "sum_j Assign[i,j] = available_hours[i] for every person",
                "project_demand_hours_equality": "sum_i Assign[i,j] = required_hours[j] for every project",
                "person_project_contribution_upper_bound": "Assign[i,j] <= max_contribution_hours[i,j] for every pair",
            },
            "objective": "minimize sum cost_per_hour[i,j] * Assign[i,j]",
            "forbidden_structures": ["binary one-to-one matching", "routes", "time windows", "optional coverage"],
        }
    electrical_tables = source_value.get("compact_electrical_power_tables")
    if ("optmath_electrical_power" in generator_id or "electrical_power" in generator_id) and isinstance(electrical_tables, dict):
        sets = electrical_tables.get("sets") or {}
        return {
            "entities": ["generator types", "time periods"],
            "sets": {
                "generator_types": sets.get("generator_types") or [],
                "time_periods": sets.get("time_periods") or [],
            },
            "parameters": {
                "generator_type_table": "see source_compact_data.compact_electrical_power_tables.generator_type_table",
                "period_demand_table": "see source_compact_data.compact_electrical_power_tables.period_demand_table",
            },
            "variables": {
                "NumGenerators[t,p]": "nonnegative integer number of generators of type t on in period p",
                "PowerOutput[t,p]": "continuous nonnegative MW output from generator type t in period p",
                "NumStart[t,p]": "nonnegative integer number of generators of type t started in period p",
            },
            "constraints": {
                "availability": "NumGenerators[t,p] <= generators_available[t]",
                "demand": "sum_t PowerOutput[t,p] >= demand[p]",
                "min_generation": "PowerOutput[t,p] >= minimum_output[t] * NumGenerators[t,p]",
                "max_generation": "PowerOutput[t,p] <= maximum_output[t] * NumGenerators[t,p]",
                "reserve": "sum_t maximum_output[t] * NumGenerators[t,p] >= reserve_margin * demand[p]",
                "startup_initial": "NumGenerators[t,first] <= initially_on[t] + NumStart[t,first]",
                "startup_transition": "NumGenerators[t,p] <= NumGenerators[t,previous] + NumStart[t,p]",
            },
            "objective": "minimize base operating cost, per-MW generation cost, and startup cost",
            "forbidden_structures": ["battery storage", "fuel inventory", "emissions cap", "power-flow network", "minimum up/down time"],
        }
    marketshare_tables = source_value.get("compact_marketshare_tables")
    if ("optmath_marketshare" in generator_id or "marketshare" in generator_id) and isinstance(marketshare_tables, dict):
        sets = marketshare_tables.get("sets") or {}
        return {
            "entities": ["companies", "markets", "products"],
            "sets": {
                "companies": sets.get("companies") or [],
                "markets": sets.get("markets") or [],
                "products": sets.get("products") or [],
            },
            "parameters": {
                "demand_table": "see source_compact_data.compact_marketshare_tables.demand_matrix or demand_table",
                "unit_profit_table": "see source_compact_data.compact_marketshare_tables.unit_profit_matrices_by_company",
            },
            "variables": {
                "Supply[i,j,k]": "nonnegative integer quantity supplied by company i to market j for product k",
            },
            "constraints": {
                "market_product_demand": "sum_i Supply[i,j,k] == demand[j,k] for every market-product pair",
            },
            "objective": "maximize sum of unit_profit[i,j,k] * Supply[i,j,k]",
            "forbidden_structures": [
                "resource capacity",
                "budget limit",
                "channel capacity",
                "eligibility",
                "price decisions",
                "nonlinear demand response",
                "optional market coverage",
            ],
        }
    tables = source_value.get("compact_structure_assignment_tables")
    if not isinstance(tables, dict):
        return fallback
    sets = tables.get("sets") or {}
    return {
        "entities": ["NMR spectral peaks", "protein amino acids"],
        "sets": {
            "P": sets.get("peaks") or [],
            "A": sets.get("amino_acids") or [],
        },
        "parameters": {
            "required_assignment_count": tables.get("required_assignment_count"),
            "distance_threshold": tables.get("distance_threshold"),
            "assignment_cost_matrix": "see source_compact_data.compact_structure_assignment_tables.assignment_cost_matrix",
            "noe_relation_pairs": tables.get("noe_relation_pairs") or [],
            "acid_compatibility_matrix": "see source_compact_data.compact_structure_assignment_tables.acid_compatibility_matrix",
        },
        "variables": {
            "x[p,a]": "binary; 1 if peak p is assigned to amino acid a",
        },
        "constraints": {
            "peak_assignment": "each peak is assigned to at most one amino acid",
            "amino_acid_assignment": "each amino acid receives at most one peak",
            "total_assignments": "exactly required_assignment_count assignments are selected",
            "noe_compatibility": "for every NOE peak pair and amino-acid pair, enforce x[p,a] + x[q,b] <= compatibility[a,b] + 1",
        },
        "objective": "minimize total assignment cost c[p,a] * x[p,a]",
        "units": {
            "assignment_cost": "normalized NMR assignment cost",
            "compatibility": "binary parameter, 1 compatible and 0 incompatible",
        },
    }


def _aircraft_landing_table_numbers(landing_tables: dict[str, Any]) -> set[str]:
    numbers: set[str] = set()
    for row in landing_tables.get("aircraft_time_penalty_table") or []:
        if not isinstance(row, dict):
            continue
        for key in (
            "earliest_landing",
            "target_landing",
            "latest_landing",
            "early_penalty_per_minute",
            "late_penalty_per_minute",
        ):
            numbers.update(_canonical_number_set(str(row.get(key))))
    matrix = landing_tables.get("ordered_pair_separation_matrix") or {}
    if isinstance(matrix, dict):
        for row in matrix.get("rows") or []:
            values = row.get("values") if isinstance(row, dict) else []
            for value in values or []:
                if value is not None:
                    numbers.update(_canonical_number_set(str(value)))
    return numbers


def _format_aircraft_landing_tables(landing_tables: dict[str, Any]) -> str:
    time_rows = landing_tables.get("aircraft_time_penalty_table") or []
    matrix = landing_tables.get("ordered_pair_separation_matrix") or {}
    lines = [
        "Aircraft landing data:",
        "- Time and penalty table (minutes; penalty per minute):",
    ]
    for row in time_rows:
        if not isinstance(row, dict):
            continue
        aircraft = row.get("aircraft") or row.get("batch") or row.get("flight") or "aircraft"
        lines.append(
            "  - "
            f"{aircraft}: earliest={row.get('earliest_landing')}, "
            f"target={row.get('target_landing')}, "
            f"latest={row.get('latest_landing')}, "
            f"early_penalty={row.get('early_penalty_per_minute')}, "
            f"late_penalty={row.get('late_penalty_per_minute')}"
        )
    if isinstance(matrix, dict):
        columns = [str(value) for value in matrix.get("columns") or []]
        rows = matrix.get("rows") or []
        if columns and rows:
            lines.append(
                "- Separation-time matrix in minutes (row aircraft lands before column aircraft; '-' means same aircraft):"
            )
            lines.append("  - columns: " + ", ".join(columns))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                aircraft_before = row.get("aircraft_before") or row.get("batch_before") or row.get("flight_before")
                values = ["-" if value is None else str(value) for value in row.get("values") or []]
                lines.append(f"  - {aircraft_before}: " + ", ".join(values))
    return "\n".join(lines)


def _canonical_number_set(text: str) -> set[str]:
    values: set[str] = set()
    for match in re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?", str(text or "")):
        try:
            value = float(match)
        except ValueError:
            continue
        if value == 0:
            value = 0.0
        if abs(value) <= 1e8:
            values.add(f"{value:.8g}")
    return values


def _artifact_preview(value: Any, *, max_chars: int) -> dict[str, Any]:
    text = str(value or "")
    if not text:
        return {"available": False}
    line_count = text.count("\n") + 1
    return {
        "available": True,
        "truncated": len(text) > max_chars,
        "original_chars": len(text),
        "line_count": line_count,
        "preview": text[:max_chars],
    }


def _stable_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _short_hash(instance_id: str, candidate_index: int) -> str:
    return hashlib.sha1(f"{instance_id}:{candidate_index}".encode("utf-8")).hexdigest()[:16]


def _mock_backtranslation(row: dict[str, Any]) -> dict[str, Any]:
    objective_sense = row.get("optimization_sense") or "optimize"
    source_numbers = _mock_source_number_list(row)
    numeric_sentence = f" Source numeric values: {', '.join(source_numbers)}." if source_numbers else ""
    return {
        "problem_background": "A planning team needs to allocate limited resources across several activities.",
        "problem_statement": (
            "A planning team is preparing an optimization model. The exact coefficients and "
            "constraints are listed in the structured instance data. Formulate and solve the "
            f"corresponding {objective_sense} problem without using external files."
            f"{numeric_sentence}"
        ),
        "structured_problem_data": {
            "instance_id": row.get("instance_id"),
            "generator_id": row.get("generator_id"),
            "math_formula_available": bool(row.get("math_formula")),
            "lp_available": bool(row.get("lp_text")),
            "source_numeric_values": source_numbers,
        },
    }


def _mock_source_number_list(row: dict[str, Any]) -> list[str]:
    compact = _compact_source_data(row, max_chars=20_000)
    if not compact.get("available") or compact.get("truncated"):
        return []
    text = _stable_json(compact.get("value") or {})
    values: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            value = float(match)
        except ValueError:
            continue
        if abs(value) > 1e8:
            continue
        canonical = f"{value:.8g}"
        if canonical in seen:
            continue
        seen.add(canonical)
        values.append(canonical)
        if len(values) >= 140:
            break
    return values
