from __future__ import annotations

import asyncio
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from cpt_cleaner.utils.code_utils import extract_executable_code

from or_cpt_engine.contracts import resolve_family_contract
from or_cpt_engine.forward_modeling.code_quality import prepare_forward_code
from or_cpt_engine.generation.generator_profiles import load_generator_profiles
from or_cpt_engine.llm.openai_compatible_client import (
    AsyncOpenAICompatibleClient,
    extract_json_object,
    is_retryable_llm_error,
)
from or_cpt_engine.quality.semantic_coherence import contains_unnegated_phrase
from or_cpt_engine.schemas.common import EngineConfig, ForwardModelingOutput, LLMStageConfig
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text
from or_cpt_engine.utils.prompt_registry import PromptSpec, load_default_prompt_registry


PROMPT_NAME = "forward_modeling_prompt"
_MAX_FORWARD_GENERATION_QUALITY_ATTEMPTS = 2


def forward_model_candidates(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    limit: int | None = None,
    mock: bool = False,
    concurrency_per_endpoint: int | None = None,
) -> dict[str, int]:
    return asyncio.run(
        forward_model_candidates_async(
            input_path,
            output_dir,
            config,
            limit=limit,
            mock=mock,
            concurrency_per_endpoint=concurrency_per_endpoint,
        )
    )


async def forward_model_candidates_async(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    limit: int | None = None,
    mock: bool = False,
    concurrency_per_endpoint: int | None = None,
) -> dict[str, int]:
    output = Path(output_dir)
    code_dir = output / "extracted_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_default_prompt_registry().get(PROMPT_NAME)

    fm_output_path = output / "forward_modeling_outputs.jsonl"
    rejected_path = output / "forward_modeling_rejected.jsonl"
    raw_response_path = output / "forward_modeling_raw_responses.jsonl"
    write_lock = asyncio.Lock()
    output_count = 0
    rejected_count = 0
    requeued_count = 0

    client = None if mock else AsyncOpenAICompatibleClient(
        config.llm,
        config.llm.forward_modeling,
        concurrency_per_endpoint=concurrency_per_endpoint,
    )
    worker_count = 1 if mock else max(1, client.total_concurrency_capacity() if client is not None else 1)

    # Queue items: (row, requeue_count).
    # Transient LLM errors trigger requeue instead of permanent rejection.
    async def _worker(queue: asyncio.Queue) -> None:
        nonlocal output_count, rejected_count, requeued_count
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            row, requeue_count = item
            result = await _forward_model_one(
                row,
                prompt_spec,
                code_dir,
                client=client,
                mock=mock,
                family_contracts=config.family_contracts,
                config=config,
            )
            if result["kind"] == "requeue":
                requeued_count += 1
                await queue.put((row, requeue_count + 1))
                queue.task_done()
                continue
            async with write_lock:
                if result["kind"] == "output":
                    append_jsonl(fm_output_path, result["output"])
                    append_jsonl(raw_response_path, result["raw_response"])
                    output_count += 1
                else:
                    append_jsonl(rejected_path, result["rejected"])
                    rejected_count += 1
            queue.task_done()

    queue: asyncio.Queue = asyncio.Queue()
    workers = [asyncio.create_task(_worker(queue)) for _ in range(worker_count)]
    produced = 0
    for row in (iter_jsonl(input_path) or []):
        if limit is not None and produced >= limit:
            break
        await queue.put((row, 0))
        produced += 1

    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
    if client is not None:
        await client.close()

    write_text(output / "forward_modeling_report.md",
               render_forward_modeling_report(output_count, rejected_count, requeued_count))
    return {"outputs": output_count, "rejected": rejected_count}


async def _forward_model_one(
    row: dict[str, Any],
    prompt_spec: PromptSpec,
    code_dir: Path,
    *,
    client: AsyncOpenAICompatibleClient | None,
    mock: bool,
    stage_config: LLMStageConfig | None = None,
    family_contracts: dict[str, Any] | None = None,
    config: EngineConfig | None = None,
    _quality_attempt: int = 1,
    _quality_retry_reasons: list[str] | None = None,
    _quality_retry_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if config is not None:
        row = _profile_normalized_row(row, config)
    quality_retry_reasons = list(_quality_retry_reasons or [])
    quality_retry_events = list(_quality_retry_events or [])
    fm_id = f"fm_{_short_hash(row['bt_id'])}"
    llm_metadata: dict[str, Any] = {}
    try:
        source_metadata = _source_metadata(row)
        family_contract = resolve_family_contract(row, family_contracts)
        if mock:
            payload = _mock_forward_model(row)
            llm_metadata = {"mock": True}
            raw_content = payload
        else:
            if client is None:
                raise RuntimeError("async client is not initialized")
            prompt = _render_prompt_with_contract(prompt_spec.text, row, family_contract=family_contract)
            prompt = _with_forward_quality_retry_directive(prompt, quality_retry_reasons)
            response = await client.chat(
                prompt,
                log_context={"instance_id": row.get("instance_id", ""), "prompt_name": PROMPT_NAME},
                stage_config=stage_config,
            )
            payload = extract_json_object(response["content"])
            llm_metadata = {
                "request_id": response["request_id"],
                "latency_sec": response["latency_sec"],
                "model": response["model"],
                "endpoint": response.get("endpoint"),
                "attempt": response["attempt"],
            }
            raw_content = response["content"]
        llm_metadata.update(
            {
                "prompt_name": PROMPT_NAME,
                "prompt_version": prompt_spec.version,
                "prompt_hash": prompt_spec.hash,
                "quality_attempt": _quality_attempt,
                "quality_max_attempts": 1 if mock else _MAX_FORWARD_GENERATION_QUALITY_ATTEMPTS,
                "quality_retries": len(quality_retry_reasons),
                "quality_retry_reasons": quality_retry_reasons,
                "quality_retry_events": quality_retry_events,
            }
        )
        code = extract_executable_code(payload.get("gurobipy_code") or payload.get("code") or payload.get("python_code"))
        if not code:
            reason = "ValueError: missing executable gurobipy code"
            if _should_retry_forward_generation(reason, mock=mock, quality_attempt=_quality_attempt):
                return await _forward_model_one(
                    row,
                    prompt_spec,
                    code_dir,
                    client=client,
                    mock=mock,
                    stage_config=stage_config,
                    family_contracts=family_contracts,
                    config=config,
                    _quality_attempt=_quality_attempt + 1,
                    _quality_retry_reasons=quality_retry_reasons + [reason],
                    _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
                )
            raise ValueError("missing executable gurobipy code")
        code_preparation = prepare_forward_code(code)
        if not code_preparation.passed:
            reason = "STATIC_CODE_CHECK_FAILED:" + "; ".join(code_preparation.issues)
            if _should_retry_forward_generation(reason, mock=mock, quality_attempt=_quality_attempt):
                return await _forward_model_one(
                    row,
                    prompt_spec,
                    code_dir,
                    client=client,
                    mock=mock,
                    stage_config=stage_config,
                    family_contracts=family_contracts,
                    config=config,
                    _quality_attempt=_quality_attempt + 1,
                    _quality_retry_reasons=quality_retry_reasons + [reason],
                    _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
                )
            return {
                "kind": "rejected",
                "rejected": {
                    "fm_id": fm_id,
                    "bt_id": row.get("bt_id"),
                    "instance_id": row.get("instance_id"),
                    "generator_id": row.get("generator_id"),
                    "source_metadata": source_metadata,
                    "family_contract_version": family_contract.get("version"),
                    "family_contract_id": family_contract.get("contract_id"),
                    "family_contract": family_contract,
                    "llm_metadata": llm_metadata,
                    "generated_answer": {
                        "modeling_explanation": payload.get("modeling_explanation"),
                        "math_model": payload.get("math_model"),
                        "gurobipy_code": code_preparation.code,
                        "self_check": payload.get("self_check") or {},
                    },
                    "source_lp_text": row.get("source_lp_text"),
                    "source_math_formula": row.get("source_math_formula"),
                    "source_compact_data": row.get("source_compact_data") or {},
                    "structured_problem_data": row.get("structured_problem_data") or {},
                    "reference_answer": row.get("reference_answer") or {},
                    "generator_profile_version": row.get("generator_profile_version"),
                    "difficulty_level": row.get("difficulty_level"),
                    "sub_family": row.get("sub_family"),
                    "canonical_math_signature": row.get("canonical_math_signature") or {},
                    "concept_tags": list(row.get("concept_tags") or []),
                    "variant_id": row.get("variant_id"),
                    "code_preparation": code_preparation.to_dict(),
                    "fm_status": "REJECTED",
                    "rejection_stage": "forward_modeling",
                    "rejection_reason": reason,
                },
            }
        code = code_preparation.code
        static_issues = _static_signature_issues(row, payload, code, family_contract=family_contract)
        if static_issues:
            reason = "STATIC_SIGNATURE_MISMATCH:" + "; ".join(static_issues)
            if _should_retry_forward_generation(reason, mock=mock, quality_attempt=_quality_attempt):
                return await _forward_model_one(
                    row,
                    prompt_spec,
                    code_dir,
                    client=client,
                    mock=mock,
                    stage_config=stage_config,
                    family_contracts=family_contracts,
                    config=config,
                    _quality_attempt=_quality_attempt + 1,
                    _quality_retry_reasons=quality_retry_reasons + [reason],
                    _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
                )
            return {
                "kind": "rejected",
                "rejected": {
                    "fm_id": fm_id,
                    "bt_id": row.get("bt_id"),
                    "instance_id": row.get("instance_id"),
                    "generator_id": row.get("generator_id"),
                    "source_metadata": source_metadata,
                    "family_contract_version": family_contract.get("version"),
                    "family_contract_id": family_contract.get("contract_id"),
                    "family_contract": family_contract,
                    "llm_metadata": llm_metadata,
                    "generated_answer": {
                        "modeling_explanation": payload.get("modeling_explanation"),
                        "math_model": payload.get("math_model"),
                        "gurobipy_code": code,
                        "self_check": payload.get("self_check") or {},
                    },
                    "source_lp_text": row.get("source_lp_text"),
                    "source_math_formula": row.get("source_math_formula"),
                    "source_compact_data": row.get("source_compact_data") or {},
                    "structured_problem_data": row.get("structured_problem_data") or {},
                    "reference_answer": row.get("reference_answer") or {},
                    "generator_profile_version": row.get("generator_profile_version"),
                    "difficulty_level": row.get("difficulty_level"),
                    "sub_family": row.get("sub_family"),
                    "canonical_math_signature": row.get("canonical_math_signature") or {},
                    "concept_tags": list(row.get("concept_tags") or []),
                    "variant_id": row.get("variant_id"),
                    "code_preparation": code_preparation.to_dict(),
                    "fm_status": "REJECTED",
                    "rejection_stage": "forward_modeling",
                    "rejection_reason": reason,
                },
            }
        code_path = code_dir / f"{fm_id}.py"
        code_path.write_text(code, encoding="utf-8")
        raw_response = {
            "fm_id": fm_id,
            "bt_id": row["bt_id"],
            "instance_id": row["instance_id"],
            "raw_response": raw_content,
            "source_metadata": source_metadata,
            "family_contract_version": family_contract.get("version"),
            "family_contract_id": family_contract.get("contract_id"),
            "family_contract": family_contract,
            "llm_metadata": llm_metadata,
        }
        output = ForwardModelingOutput(
            fm_id=fm_id,
            bt_id=row["bt_id"],
            instance_id=row["instance_id"],
            generator_id=row.get("generator_id", ""),
            problem_statement=row.get("problem_statement") or "",
            structured_problem_data=row.get("structured_problem_data") or {},
            source_lp_text=row.get("source_lp_text"),
            source_math_formula=row.get("source_math_formula"),
            source_compact_data=row.get("source_compact_data") or {},
            generated_answer={
                "modeling_explanation": payload.get("modeling_explanation"),
                "math_model": payload.get("math_model"),
                "gurobipy_code": code,
                "self_check": payload.get("self_check") or {},
            },
            code_preparation=code_preparation.to_dict(),
            code_path=str(code_path),
            reference_answer=row.get("reference_answer") or {},
            generator_profile_version=row.get("generator_profile_version"),
            difficulty_level=row.get("difficulty_level"),
            sub_family=row.get("sub_family"),
            canonical_math_signature=row.get("canonical_math_signature") or {},
            concept_tags=list(row.get("concept_tags") or []),
            variant_id=row.get("variant_id"),
            source_metadata=source_metadata,
            family_contract_version=family_contract.get("version"),
            family_contract_id=family_contract.get("contract_id"),
            family_contract=family_contract,
            llm_metadata=llm_metadata,
        )
        return {"kind": "output", "output": output, "raw_response": raw_response}
    except Exception as exc:  # noqa: BLE001
        if is_retryable_llm_error(exc):
            return {"kind": "requeue"}
        reason = f"{exc.__class__.__name__}: {exc}"
        if _should_retry_forward_generation(reason, mock=mock, quality_attempt=_quality_attempt):
            return await _forward_model_one(
                row,
                prompt_spec,
                code_dir,
                client=client,
                mock=mock,
                stage_config=stage_config,
                family_contracts=family_contracts,
                config=config,
                _quality_attempt=_quality_attempt + 1,
                _quality_retry_reasons=quality_retry_reasons + [reason],
                _quality_retry_events=quality_retry_events + [_quality_retry_event(reason, llm_metadata, _quality_attempt)],
            )
        return {
            "kind": "rejected",
            "rejected": {
                "fm_id": fm_id,
                "bt_id": row.get("bt_id"),
                "instance_id": row.get("instance_id"),
                "generator_id": row.get("generator_id"),
                "source_metadata": _source_metadata(row),
                "source_lp_text": row.get("source_lp_text"),
                "source_math_formula": row.get("source_math_formula"),
                "source_compact_data": row.get("source_compact_data") or {},
                "structured_problem_data": row.get("structured_problem_data") or {},
                "reference_answer": row.get("reference_answer") or {},
                "generator_profile_version": row.get("generator_profile_version"),
                "difficulty_level": row.get("difficulty_level"),
                "sub_family": row.get("sub_family"),
                "canonical_math_signature": row.get("canonical_math_signature") or {},
                "concept_tags": list(row.get("concept_tags") or []),
                "variant_id": row.get("variant_id"),
                "llm_metadata": {
                    "quality_attempt": _quality_attempt,
                    "quality_max_attempts": 1 if mock else _MAX_FORWARD_GENERATION_QUALITY_ATTEMPTS,
                    "quality_retries": len(quality_retry_reasons),
                    "quality_retry_reasons": quality_retry_reasons,
                    "quality_retry_events": quality_retry_events,
                },
                "fm_status": "REJECTED",
                "rejection_stage": "forward_modeling",
                "rejection_reason": reason,
            },
        }


def render_forward_modeling_report(output_count: int, rejected_count: int, requeued_count: int = 0) -> str:
    return "\n".join(
        [
            "# Forward Modeling Report",
            "",
            f"- Outputs with extracted code: {output_count}",
            f"- Rejected during generation/parsing: {rejected_count}",
            f"- Re-queued (transient LLM errors, retried): {requeued_count}",
            "",
        ]
    )


def _should_retry_forward_generation(reason: str, *, mock: bool, quality_attempt: int) -> bool:
    if mock or quality_attempt >= _MAX_FORWARD_GENERATION_QUALITY_ATTEMPTS:
        return False
    lowered = reason.lower()
    retry_markers = (
        "missing executable gurobipy code",
        "static_code_check_failed",
        "static_signature_mismatch",
        "placeholder",
        "replace_with",
        "todo",
        "external_file",
        "syntax",
        "indentation",
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


def _with_forward_quality_retry_directive(prompt: str, retry_reasons: list[str]) -> str:
    reasons = [str(reason).strip() for reason in retry_reasons if str(reason).strip()]
    if not reasons:
        return prompt
    reason_summary = "; ".join(_short_text(reason, max_chars=220) for reason in reasons[-3:])
    hints = _forward_retry_issue_hints(reasons)
    hint_text = "\n".join(f"- {hint}" for hint in hints)
    directive = (
        "QUALITY RETRY DIRECTIVE:\n"
        f"The previous forward-modeling response was rejected for: {reason_summary}.\n"
        "Rewrite the answer from scratch and return exactly one JSON object with `modeling_explanation`, "
        "`math_model`, and executable `gurobipy_code`.\n"
        "Use `source_fact_digest`, `source_compact_data`, `family_contract`, and `modeling_guardrails` as authoritative. "
        "Do not introduce variables, constraints, costs, time periods, routing logic, backlog, or terminal conditions "
        "unless they are explicitly present in those source facts or the family contract.\n"
        "Remove all placeholders, TODO text, prose inside code, and external-file dependencies.\n"
    )
    if hint_text:
        directive += "Specific fixes required by the rejection reason:\n" + hint_text + "\n"
    return f"{directive}\n{prompt}"


def _forward_retry_issue_hints(retry_reasons: list[str]) -> list[str]:
    lowered = "\n".join(retry_reasons).lower()
    hints: list[str] = []
    if "placeholder" in lowered or "replace_with" in lowered or "todo" in lowered:
        hints.append("Replace placeholder/TODO code with real Gurobi variables, objectives, constraints, and `model.optimize()`.")
    if "missing executable gurobipy code" in lowered:
        hints.append("Return executable Python/Gurobi code in `gurobipy_code`; do not return pseudocode or prose-only formulas.")
    if "backlog_added" in lowered:
        hints.append("Do not add backlog, shortage, lost-sales, unmet-demand, or zero-final-backlog variables/constraints unless the source facts explicitly include backlog.")
    if "zero_final_inventory_added" in lowered:
        hints.append("Do not add a zero-final-inventory constraint unless the source facts explicitly require terminal inventory to be zero.")
    if "unit_production_cost_added" in lowered:
        hints.append("Do not invent unit production costs; preserve the source objective terms exactly.")
    if "nonnegative_integer_supply" in lowered:
        hints.append("Define supply/allocation quantity variables as nonnegative integer variables, e.g. `vtype=GRB.INTEGER`, not continuous variables.")
    if "vehicle_routing_structure_added" in lowered:
        hints.append("Do not transform assignment or flow tasks into vehicle routing; avoid route, time-window, sequencing, and subtour constraints unless the source model has them.")
    if "capacity_if_present" in lowered:
        hints.append("If the source facts include setup/capacity links, include the required capacity or big-M setup-capacity constraints.")
    if "material_balance" in lowered:
        hints.append("Include the source material/flow/balance constraints exactly, with the same direction and indexed sets.")
    if "demand_or_output_requirement_if_present" in lowered:
        hints.append("Include every demand, output, coverage, or fulfillment requirement present in the source facts.")
    if "arc_shipping_cost" in lowered:
        hints.append("Keep the network objective tied to source arc costs/latencies on the same arcs; do not invent a different cost basis.")
    if "objective_sense_mismatch" in lowered:
        hints.append("Use the objective sense from the family contract and source signature exactly.")
    if not hints:
        hints.append("Address the rejection reason directly and preserve the source mathematical structure without adding extra business assumptions.")
    return hints


def _short_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _render_prompt(template: str, row: dict[str, Any]) -> str:
    import json

    family_contract = resolve_family_contract(row, None)
    return _render_prompt_with_contract(template, row, family_contract=family_contract)


def _render_prompt_with_contract(template: str, row: dict[str, Any], *, family_contract: dict[str, Any]) -> str:
    import json

    source_fact_digest = _source_fact_digest(row)
    payload = {
        "bt_id": row.get("bt_id"),
        "instance_id": row.get("instance_id"),
        "problem_statement": row.get("problem_statement"),
        "structured_problem_data": row.get("structured_problem_data"),
        "source_compact_data": _forward_prompt_source_compact_data(row, source_fact_digest=source_fact_digest),
        "source_fact_digest": source_fact_digest,
        "generator_profile": {
            "difficulty_level": row.get("difficulty_level"),
            "sub_family": row.get("sub_family"),
            "concept_tags": row.get("concept_tags") or [],
            "variant_id": row.get("variant_id"),
            "canonical_math_signature": row.get("canonical_math_signature") or {},
        },
        "modeling_guardrails": _modeling_guardrails(row),
        "family_contract": family_contract,
    }
    return template.replace("{{PROBLEM_JSON}}", json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _forward_prompt_source_compact_data(row: dict[str, Any], *, source_fact_digest: list[str]) -> dict[str, Any]:
    source_compact = row.get("source_compact_data") or {}
    if not source_fact_digest or not isinstance(source_compact, dict) or not source_compact.get("available"):
        return source_compact if isinstance(source_compact, dict) else {}
    value = source_compact.get("value")
    value_keys = sorted(str(key) for key in value.keys()) if isinstance(value, dict) else []
    return {
        "available": True,
        "truncated_for_forward_prompt": True,
        "reason": (
            "source_fact_digest carries the authoritative compact facts for the forward-modeling prompt; "
            "the full source_compact_data is preserved in stage artifacts."
        ),
        "value_keys": value_keys,
    }


def _source_fact_digest(row: dict[str, Any], *, max_chars: int = 6_000) -> list[str]:
    source_compact = row.get("source_compact_data") or {}
    if not isinstance(source_compact, dict) or not source_compact.get("available") or source_compact.get("truncated"):
        return []
    source_value = source_compact.get("value")
    if not isinstance(source_value, dict):
        return []
    lines = [
        "AUTHORITATIVE_SOURCE_FACTS: Use these facts exactly. If problem_statement or structured_problem_data disagrees, source_compact_data wins.",
    ]
    if isinstance(source_value.get("compact_steel_product_mix_tables"), dict):
        lines.extend(_steel_product_mix_digest(source_value["compact_steel_product_mix_tables"]))
    if isinstance(source_value.get("compact_lotsizing_tables"), dict):
        lines.extend(_lotsizing_digest(source_value["compact_lotsizing_tables"]))
    if isinstance(source_value.get("compact_network_flow_tables"), dict):
        lines.extend(_network_flow_digest(source_value["compact_network_flow_tables"]))
    if isinstance(source_value.get("compact_project_assignment_tables"), dict):
        lines.extend(_project_assignment_digest(source_value["compact_project_assignment_tables"]))
    if isinstance(source_value.get("compact_structure_assignment_tables"), dict):
        lines.extend(_structure_assignment_digest(source_value["compact_structure_assignment_tables"]))
    if isinstance(source_value.get("compact_marketshare_tables"), dict):
        lines.extend(_marketshare_digest(source_value["compact_marketshare_tables"]))
    if isinstance(source_value.get("compact_electrical_power_tables"), dict):
        lines.extend(_electrical_power_digest(source_value["compact_electrical_power_tables"]))
    if isinstance(source_value.get("compact_aircraft_landing_tables"), dict):
        lines.extend(_aircraft_landing_digest(source_value["compact_aircraft_landing_tables"]))
    if len(lines) == 1:
        return []
    return _bounded_lines(lines, max_chars=max_chars)


def _steel_product_mix_digest(tables: dict[str, Any]) -> list[str]:
    model_family = str(tables.get("model_family") or "").lower()
    if model_family == "single_stage_continuous_product_mix":
        return _single_stage_steel_product_mix_digest(tables)
    lines = [
        "steel_product_mix: continuous Production[p]; maximize profit; no inventory, setup, sequencing, or time periods.",
        "steel_product_mix: stage capacity is sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]; do not use product market bounds as stage capacities.",
    ]
    product_rows = ((tables.get("product_table") or {}).get("rows") or [])
    stage_rows = ((tables.get("stage_capacity_table") or {}).get("rows") or [])
    processing_matrix = tables.get("processing_hours_per_ton_matrix") or {}
    processing_columns = processing_matrix.get("columns") or []
    processing_rows = processing_matrix.get("rows") or []
    if product_rows:
        lines.append("products product,profit,min_commit,max_market: " + "; ".join(_join_row(row) for row in product_rows))
    if stage_rows:
        lines.append("stage_available_hours stage,available: " + "; ".join(_join_row(row) for row in stage_rows))
    if processing_rows:
        lines.append("processing_matrix_columns: " + ", ".join(str(value) for value in processing_columns))
        for row in processing_rows:
            if isinstance(row, dict):
                lines.append(f"processing {row.get('product')}: " + ", ".join(str(value) for value in row.get("values") or []))
    return lines


def _single_stage_steel_product_mix_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "steel3_single_stage_product_mix: continuous Production[p]; maximize profit; no inventory, setup, sequencing, time periods, or multi-stage capacity matrix.",
    ]
    time_capacity = tables.get("time_capacity") if isinstance(tables.get("time_capacity"), dict) else {}
    lines.append(
        "steel3_time_capacity: "
        f"sum_p processing_hours_per_ton[p] * Production[p] <= available_hours={time_capacity.get('available_hours')}"
    )
    product_rows = ((tables.get("product_table") or {}).get("rows") or [])
    if product_rows:
        lines.append(
            "steel3_products product,profit,production_rate,processing_hours_per_ton,min_commit,max_market: "
            + "; ".join(_join_row(row) for row in product_rows)
        )
    return lines


def _electrical_power_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "electrical_power_unit_commitment: integer NumGenerators[type,period], continuous PowerOutput[type,period], integer NumStart[type,period]; minimize base operating cost + per-MW cost + startup cost.",
        "electrical_power: no battery storage, fuel inventory, emissions cap, power-flow network, unit-specific identity, or minimum up/down time constraints.",
    ]
    generator_rows = tables.get("generator_type_table") if isinstance(tables.get("generator_type_table"), list) else []
    if generator_rows:
        lines.append("generator_type_rows type,base_cost,per_mw,startup,min_output,max_output,available,initially_on:")
        for row in generator_rows:
            if isinstance(row, dict):
                lines.append(
                    f"type={row.get('generator_type')},base={row.get('base_operating_cost_per_period')},"
                    f"per_mw={row.get('per_mw_generation_cost')},startup={row.get('startup_cost')},"
                    f"min={row.get('minimum_output_mw')},max={row.get('maximum_output_mw')},"
                    f"available={row.get('generators_available')},initially_on={row.get('generators_on_initially')}"
                )
    demand_rows = tables.get("period_demand_table") if isinstance(tables.get("period_demand_table"), list) else []
    if demand_rows:
        lines.append("period_demand_rows period,demand_mw,reserve_required_capacity_mw:")
        for row in demand_rows:
            if isinstance(row, dict):
                lines.append(
                    f"period={row.get('period')},demand={row.get('demand_mw')},"
                    f"reserve_required={row.get('reserve_required_capacity_mw')}"
                )
    lines.append(
        "electrical_power_constraints: output meets demand; output is between min_output*NumGenerators and max_output*NumGenerators; "
        "online max capacity meets reserve_required; NumGenerators cannot exceed available; startup links first period to initially_on and later periods to previous NumGenerators."
    )
    return lines


def _aircraft_landing_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "aircraft_landing: static runway landing-time sequencing; continuous Landing[i], Early[i], Late[i], binary AircraftOrder[i,j].",
        "aircraft_landing_objective: minimize early_penalty_per_minute[i] * Early[i] + late_penalty_per_minute[i] * Late[i].",
        "aircraft_landing_constraints: one precedence order for each unordered pair; ordered-pair separation times; landing time windows; early/late deviation lower bounds.",
        "aircraft_landing_forbidden: do not convert to aircraft type assignment, route allocation, fleet routing, passenger capacity, gate assignment, makespan-only scheduling, or expose big-M as business data.",
    ]
    time_rows = tables.get("aircraft_time_penalty_table") if isinstance(tables.get("aircraft_time_penalty_table"), list) else []
    if time_rows:
        lines.append("aircraft_landing_time_penalty_rows aircraft,earliest,target,latest,early_penalty,late_penalty:")
        for row in time_rows:
            if isinstance(row, dict):
                lines.append(
                    f"aircraft={row.get('aircraft')},earliest={row.get('earliest_landing')},"
                    f"target={row.get('target_landing')},latest={row.get('latest_landing')},"
                    f"early_penalty={row.get('early_penalty_per_minute')},"
                    f"late_penalty={row.get('late_penalty_per_minute')}"
                )
    separation_matrix = tables.get("ordered_pair_separation_matrix") if isinstance(tables.get("ordered_pair_separation_matrix"), dict) else {}
    columns = separation_matrix.get("columns") or []
    if columns:
        lines.append("aircraft_landing_separation_columns: " + ", ".join(str(value) for value in columns))
    for row in separation_matrix.get("rows") or []:
        if isinstance(row, dict):
            values = ["-" if value is None else str(value) for value in row.get("values") or []]
            lines.append(f"aircraft_landing_separation before={row.get('aircraft_before')}: " + ", ".join(values))
    return lines


def _lotsizing_digest(tables: dict[str, Any]) -> list[str]:
    model_family = str(tables.get("model_family") or "").lower()
    lines: list[str] = []
    if model_family == "capacitated_lot_sizing_without_backlog":
        lines.append("clsp: capacitated lot sizing without backlog; use Production[p,t], EndingInventory[p,t], binary Setup[p,t].")
        lines.append("clsp: no Backlog, lost sales, capacity expansion, overtime, or separate zero-final-inventory hard constraint.")
        lines.append("clsp_demand_rule: period_demand is the only demand value for recursive inventory balance; cumulative_demand_through_period is only a helper for cumulative-balance form or setup big-M.")
        lines.append("clsp_balance_options: recursive form EndingInventory[p,prev] + Production[p,t] == period_demand[p,t] + EndingInventory[p,t]; cumulative form EndingInventory[p,t] == sum_{tau<=t} Production[p,tau] - cumulative_demand_through_period[p,t].")
        lines.append("clsp_forbidden_data_use: never put cumulative_demand_through_period as the RHS demand in a one-period recursive balance.")
        rows = tables.get("period_demand_table") or []
        if rows:
            lines.append("clsp_rows p,t,period_demand,cumulative_demand_through_period,setup_cost,unit_prod_cost,holding_cost,setup_big_m:")
            for row in rows:
                if isinstance(row, dict):
                    lines.append(
                        f"p={row.get('product')},t={row.get('period')},period_demand={row.get('period_demand')},"
                        f"cumulative_demand_through_period={row.get('cumulative_demand_through_period')},setup={row.get('setup_cost')},"
                        f"prod={row.get('unit_production_cost')},hold={row.get('unit_holding_cost')},"
                        f"M={row.get('setup_big_m_remaining_demand')}"
                    )
        capacity_rows = tables.get("period_capacity_table") or []
        if capacity_rows:
            lines.append("clsp_period_capacities: " + "; ".join(f"t={row.get('period')}: {row.get('capacity')}" for row in capacity_rows if isinstance(row, dict)))
        consumption_rows = tables.get("capacity_consumption_per_product") or []
        if consumption_rows:
            lines.append("clsp_capacity_consumption: " + "; ".join(f"p={row.get('product')}: {row.get('capacity_consumption_per_unit')}" for row in consumption_rows if isinstance(row, dict)))
        return lines
    if model_family == "uncapacitated_lot_sizing_with_backlogging":
        lines.append("ulsb: uncapacitated lot sizing with carried backlog state; demand values are period demand, not cumulative.")
        lines.append(
            f"ulsb_initial_final: initial_inventory={tables.get('initial_inventory', 0)}, initial_backlog={tables.get('initial_backlog', 0)}, "
            f"required_final_inventory={tables.get('required_final_inventory', 0)}, required_final_backlog={tables.get('required_final_backlog', 0)}, total_demand_big_m={tables.get('total_demand_big_m')}"
        )
        rows = tables.get("period_cost_table") or []
        if rows:
            lines.append("ulsb_rows period,demand,fixed,unit_order,holding,backlog_penalty:")
            for row in rows:
                if isinstance(row, dict):
                    lines.append(
                        f"period={row.get('period')},demand={row.get('demand')},fixed={row.get('fixed_ordering_cost')},"
                        f"unit_order={row.get('unit_order_cost')},holding={row.get('unit_holding_cost')},"
                        f"backlog_penalty={row.get('unit_backlog_penalty')}"
                    )
        return lines
    if model_family == "uncapacitated_lot_sizing_without_backlog":
        lines.append("uls: uncapacitated lot sizing without backlog; use ONLY OrderedAmount[t], EndingInventory[t], and binary OrderIsPlaced[t].")
        lines.append("uls_forbidden_variables: do not define B[t], Backlog[t], BackloggedAmount[t], Shortage[t], LostSales[t], UnmetDemand[t], backlog_penalty, or zero_final_backlog.")
        lines.append("uls_no_backlog_formula: first period OrderedAmount[t] + initial_inventory == demand[t] + EndingInventory[t]; later periods EndingInventory[prev] + OrderedAmount[t] == demand[t] + EndingInventory[t].")
        lines.append("uls_setup_link: OrderedAmount[t] <= total_demand_big_m * OrderIsPlaced[t]. There is no production capacity/resource constraint.")
        lines.append(
            f"uls_initial_final: initial_inventory={tables.get('initial_inventory', 0)}, required_final_inventory={tables.get('required_final_inventory', 0)}, total_demand_big_m={tables.get('total_demand_big_m')}"
        )
        rows = tables.get("period_cost_table") or []
        if rows:
            lines.append("uls_rows period,demand,fixed,unit_order,holding:")
            for row in rows:
                if isinstance(row, dict):
                    lines.append(
                        f"period={row.get('period')},demand={row.get('demand')},fixed={row.get('fixed_ordering_cost')},"
                        f"unit_order={row.get('unit_order_cost')},holding={row.get('unit_holding_cost')}"
                    )
        return lines
    if "demand_matrix" in tables and "machine_table" in tables:
        lines.append("smallbucket: item-machine-period small-bucket lot sizing with Amount, binary Production state, binary Startup, Stock, and Backlog.")
        lines.append("smallbucket: objective has setup cost, startup cost, holding cost, backlog cost only; no per-unit production/material/revenue term.")
        global_costs = tables.get("global_costs") or {}
        lines.append(
            f"smallbucket_global_costs: setup={global_costs.get('setup_cost_per_active_item_machine_period')}, startup={global_costs.get('startup_cost_per_start_event')}"
        )
        item_rows = ((tables.get("item_cost_table") or {}).get("rows") or [])
        if item_rows:
            lines.append("smallbucket_item_costs item,holding,backlog: " + "; ".join(_join_row(row) for row in item_rows))
        machine_rows = ((tables.get("machine_table") or {}).get("rows") or [])
        if machine_rows:
            lines.append("smallbucket_machines machine,capacity,startup_time: " + "; ".join(_join_row(row) for row in machine_rows))
        demand_matrix = tables.get("demand_matrix") or {}
        demand_columns = demand_matrix.get("columns") or []
        lines.append("smallbucket_demand_columns: " + ", ".join(str(value) for value in demand_columns))
        for row in demand_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"smallbucket_demand item={row.get('item')}: " + ", ".join(str(value) for value in row.get("values") or []))
        return lines
    return []


def _network_flow_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "network_flow: use only declared directed arcs. Balance convention is supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node].",
        f"network_totals: total_supply={tables.get('total_supply')}, total_demand={tables.get('total_demand')}",
    ]
    node_rows = tables.get("node_balance_table") or []
    if node_rows:
        lines.append("network_nodes city,supply,demand,net_supply_minus_demand:")
        for row in node_rows:
            if isinstance(row, dict):
                lines.append(f"city={row.get('city')},supply={row.get('supply')},demand={row.get('demand')},net={row.get('net_supply_minus_demand')}")
    arc_rows = tables.get("directed_arc_table") or []
    if arc_rows:
        lines.append("network_arcs from,to,cost,capacity:")
        for row in arc_rows:
            if isinstance(row, dict):
                lines.append(f"from={row.get('from')},to={row.get('to')},cost={row.get('unit_arc_flow_cost')},capacity={row.get('arc_capacity')}")
    return lines


def _project_assignment_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "netasgn_project_assignment: continuous Assign[person,project] hours; minimize cost_per_hour * Assign.",
        "netasgn_constraints: person supply equality, project demand equality, and Assign[person,project] <= max_contribution_hours[person,project].",
        "netasgn_forbidden: no binary one-to-one matching, no optional coverage, no route/time/shift sequencing, no hidden feasible allocation certificate.",
        f"netasgn_totals: total_supply_hours={tables.get('total_supply_hours')}, total_demand_hours={tables.get('total_demand_hours')}",
    ]
    person_rows = tables.get("person_supply_table") if isinstance(tables.get("person_supply_table"), list) else []
    if person_rows:
        lines.append(
            "netasgn_person_supply person,available_hours: "
            + "; ".join(
                f"{row.get('person')}={row.get('available_hours')}"
                for row in person_rows
                if isinstance(row, dict)
            )
        )
    project_rows = tables.get("project_demand_table") if isinstance(tables.get("project_demand_table"), list) else []
    if project_rows:
        lines.append(
            "netasgn_project_demand project,required_hours: "
            + "; ".join(
                f"{row.get('project')}={row.get('required_hours')}"
                for row in project_rows
                if isinstance(row, dict)
            )
        )
    cost_matrix = tables.get("cost_per_hour_matrix") if isinstance(tables.get("cost_per_hour_matrix"), dict) else {}
    columns = cost_matrix.get("columns") or []
    if columns:
        lines.append("netasgn_cost_columns: " + ", ".join(str(value) for value in columns))
        for row in cost_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"netasgn_cost person={row.get('person')}: " + ", ".join(str(value) for value in row.get("values") or []))
    limit_matrix = (
        tables.get("max_contribution_hours_matrix")
        if isinstance(tables.get("max_contribution_hours_matrix"), dict)
        else {}
    )
    limit_columns = limit_matrix.get("columns") or []
    if limit_columns:
        lines.append("netasgn_limit_columns: " + ", ".join(str(value) for value in limit_columns))
        for row in limit_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"netasgn_limit person={row.get('person')}: " + ", ".join(str(value) for value in row.get("values") or []))
    return lines


def _structure_assignment_digest(tables: dict[str, Any]) -> list[str]:
    lines = [
        "structure_assignment: NMR peak-to-amino-acid assignment, not staffing/vehicle/job assignment.",
        f"structure_assignment_count: exactly {tables.get('required_assignment_count')} peak-amino-acid assignments must be selected.",
        "structure_assignment: use declared assignment cost matrix, NOE relation pairs, and amino-acid compatibility matrix exactly; do not assume missing compatibility values.",
    ]
    cost_matrix = tables.get("assignment_cost_matrix") or {}
    cost_columns = cost_matrix.get("columns") or []
    if cost_columns:
        lines.append("structure_cost_columns amino_acids: " + ", ".join(str(value) for value in cost_columns))
    for row in cost_matrix.get("rows") or []:
        if isinstance(row, dict):
            lines.append(f"structure_cost peak={row.get('peak')}: " + ", ".join(str(value) for value in row.get("values") or []))
    compatibility = tables.get("acid_compatibility_matrix") or {}
    compatibility_columns = compatibility.get("columns") or []
    if compatibility_columns:
        lines.append("structure_compatibility_columns amino_acids: " + ", ".join(str(value) for value in compatibility_columns))
    for row in compatibility.get("rows") or []:
        if isinstance(row, dict):
            lines.append(f"structure_compatibility amino_acid={row.get('amino_acid')}: " + ", ".join(str(value) for value in row.get("values") or []))
    noe_pairs = tables.get("noe_relation_pairs") or []
    if noe_pairs:
        lines.append("structure_noe_pairs: " + "; ".join(_join_row(pair) for pair in noe_pairs))
    return lines


def _marketshare_digest(tables: dict[str, Any]) -> list[str]:
    sets = tables.get("sets") if isinstance(tables.get("sets"), dict) else {}
    lines = [
        "marketshare: nonnegative integer Supply[i,j,k] quantity by company/initiative i, market/region j, product k.",
        "marketshare: implement Supply[i,j,k] with lb=0 and vtype=GRB.INTEGER in Gurobi; continuous supply is not source-faithful.",
        "marketshare: maximize net profit exactly over the listed company-market-product triples.",
        "marketshare: for every market-product pair, sum_i Supply[i,j,k] == demand[j,k].",
        "marketshare: no resource capacity, budget, channel-capacity, eligibility, nonlinear-response, or optional-demand constraints are present unless listed in source.",
    ]
    if sets:
        lines.append(
            "marketshare_sets: companies={companies}; markets={markets}; products={products}".format(
                companies=", ".join(str(value) for value in sets.get("companies") or []),
                markets=", ".join(str(value) for value in sets.get("markets") or []),
                products=", ".join(str(value) for value in sets.get("products") or []),
            )
        )
    demand_rows = tables.get("demand_table") or []
    demand_matrix = tables.get("demand_matrix") if isinstance(tables.get("demand_matrix"), dict) else {}
    if demand_matrix.get("rows"):
        columns = demand_matrix.get("columns") or []
        lines.append("marketshare_demand_matrix product_columns: " + ", ".join(str(value) for value in columns))
        for row in demand_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(f"market={row.get('market')}: " + ", ".join(str(value) for value in row.get("values") or []))
    elif demand_rows:
        lines.append("marketshare_demand market,product,demand:")
        for row in demand_rows:
            lines.append(_join_row(row))
    profit_matrix = (
        tables.get("unit_profit_matrices_by_company")
        if isinstance(tables.get("unit_profit_matrices_by_company"), dict)
        else {}
    )
    if profit_matrix.get("rows"):
        columns = profit_matrix.get("columns") or []
        lines.append("marketshare_unit_profit_matrix product_columns: " + ", ".join(str(value) for value in columns))
        for row in profit_matrix.get("rows") or []:
            if isinstance(row, dict):
                lines.append(
                    f"company={row.get('company')},market={row.get('market')}: "
                    + ", ".join(str(value) for value in row.get("values") or [])
                )
    else:
        profit_rows = tables.get("profit_table") or []
        if profit_rows:
            lines.append("marketshare_profit company,market,product,unit_cost,unit_revenue,unit_profit:")
            for row in profit_rows:
                lines.append(_join_row(row))
    return lines


def _join_row(row: Any) -> str:
    if isinstance(row, dict):
        return ",".join(f"{key}={value}" for key, value in row.items())
    if isinstance(row, (list, tuple)):
        return ",".join(str(value) for value in row)
    return str(row)


def _bounded_lines(lines: list[str], *, max_chars: int) -> list[str]:
    result: list[str] = []
    total = 0
    for line in lines:
        projected = total + len(line) + 1
        if projected > max_chars:
            result.append("DIGEST_TRUNCATED: see source_compact_data.value for the remaining authoritative facts.")
            break
        result.append(line)
        total = projected
    return result


def _profile_normalized_row(row: dict[str, Any], config: EngineConfig) -> dict[str, Any]:
    generator_id = str(row.get("generator_id") or "").strip()
    if not generator_id:
        return row
    profile_payload = _cached_profile_payload(str(config.paths.generator_profiles_path))
    profiles = profile_payload.get("profiles") or {}
    profile = profiles.get(generator_id) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        return row
    normalized = dict(row)
    normalized["generator_profile_version"] = str(
        profile_payload.get("version") or normalized.get("generator_profile_version") or ""
    )
    normalized["sub_family"] = profile.get("sub_family") or normalized.get("sub_family")
    normalized["canonical_math_signature"] = (
        profile.get("canonical_math_signature") or normalized.get("canonical_math_signature") or {}
    )
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


def _modeling_guardrails(row: dict[str, Any]) -> list[str]:
    generator_id = str(row.get("generator_id") or "").lower()
    sub_family = str(row.get("sub_family") or "").lower()
    signature = row.get("canonical_math_signature") or {}
    concepts = {str(value).lower() for value in row.get("concept_tags") or []}
    core_constraints = {str(value).lower() for value in signature.get("core_constraints") or []}
    required_tables = {str(value).lower() for value in signature.get("required_tables") or []}
    hints: list[str] = []
    if "optmath_facility_location" in generator_id or "multi_commodity_distribution_center_selection" in sub_family:
        hints.append("Preserve the three facility-location decision layers exactly: Selected[d] binary, Served[d,z] binary, and Shipped[c,p,d,z] continuous nonnegative.")
        hints.append("Preserve exact served-demand equalities: sum_p Shipped[c,p,d,z] = Demand[c,z] * Served[d,z].")
        hints.append("Preserve fixed distribution-center selection cost, unit throughput cost, min/max throughput links, and Served[d,z] <= Selected[d].")
        hints.append("Do not add vehicles, routes, subtour constraints, time windows, or customer visit sequencing.")
    if "optmath_cflp" in generator_id or "capacitated_facility_location" == sub_family:
        hints.append("Model this as capacitated facility location with binary FacilityOpen[j] and continuous nonnegative ShippedAmount[i,j].")
        hints.append("Preserve fixed opening costs, exact customer demand satisfaction, ShippedAmount[i,j] <= demand[i] * FacilityOpen[j], and facility capacity linked to FacilityOpen[j].")
        hints.append("Do not reduce it to pure transportation and do not add vehicles, tours, time windows, subtour constraints, or route sequencing.")
    if "optmath_carselection" in generator_id:
        hints.append("Model this as maximum-cardinality bipartite matching with binary Assignments[p,c].")
        hints.append("Preserve the complete participant-car eligibility matrix, Assignments[p,c] <= eligibility[p,c], participant at-most-one, and car at-most-one constraints.")
        hints.append("Do not introduce assignment costs, compatibility scores, travel distances, routes, time windows, or exact full-assignment requirements.")
    if "optmath_cell_tower" in generator_id:
        hints.append("Model this as budgeted maximum coverage with binary Build[t] and binary Covered[r].")
        hints.append("Preserve tower setup costs, region populations, tower-region coverage matrix, budget constraint, and Covered[r] <= sum_t coverage[t,r] * Build[t].")
        hints.append("Do not require every region to be covered and do not reinterpret it as capacitated facility location, transportation, or route planning.")
    if "optmath_scooter_location" in generator_id or "micromobility_facility_location" in sub_family:
        hints.append("Model this as micromobility station selection and demand assignment with binary SelectedLocation[j], binary Assign[i,j], and integer NewScooters[j].")
        hints.append("Preserve demand-weighted walking distance, station opening cost, new-scooter deployment cost, exact demand assignment, assignment-to-selected-station linking, station capacity using existing plus new scooters, selected-station limit, total new-scooter budget, and NewScooters[j] <= total_demand * SelectedLocation[j].")
        hints.append("Do not reinterpret it as vehicle routing, TSP, shortest path, station visit sequencing, or pure transportation.")
    elif "facility_location" in generator_id or "fixed_cost_vector" in required_tables:
        hints.append("Preserve fixed opening/setup/operating costs as objective terms; do not reduce the instance to pure transportation.")
        hints.append("Include open-before-serve linking and capacity constraints for opened facilities or activated resources.")
    if "optmath_tsp" in generator_id or "traveling_salesperson" in sub_family:
        hints.append("Model this as a single traveling-salesperson tour with binary route[i,j] directed arc variables and MTZ subtour-elimination order variables.")
        hints.append("Preserve exactly one inbound and exactly one outbound selected arc for every city, complete directed distance costs, and one single tour returning to the depot.")
        hints.append("Do not add vehicle capacity, time windows, service times, multiple vehicles, delivery quantities, or convert it into shortest path or transportation.")
    if "optmath_the_shortest_path_problem" in generator_id or "shortest_path" in sub_family:
        hints.append("Model this as a binary shortest path over only the declared directed arcs.")
        hints.append("Preserve source/sink flow balance: outflow minus inflow equals 1 at the source, -1 at the sink, and 0 at intermediate nodes.")
        hints.append("Preserve arc cost minimization and do not add capacities, fixed charges, vehicles, subtours, time windows, facility openings, or complete-graph arcs.")
    if "optmath_fleet_routing" in generator_id or "time_expanded_fleet_flow" in sub_family:
        hints.append("Model this as time-expanded fleet flow with integer NumPlanes[v,i,t,j,h], idle fleet counts, and initial fleet placement counts.")
        hints.append("Demand exists only on listed active legs; inactive feasible legs must be forced to zero by leg_is_active route restrictions.")
        hints.append("Preserve initial fleet balance, later-period flow conservation, fleet availability, and active-leg demand satisfaction.")
        hints.append("Do not introduce binary route arcs, customer visit-once constraints, subtour elimination, customer time windows, or service-time propagation.")
    if "optmath_portfolio" in generator_id or "continuous_mean_variance_portfolio" in sub_family:
        hints.append("Model this as continuous mean-variance portfolio optimization: continuous weights, sum weights = 1, target return >= R, and weight bounds.")
        hints.append("Minimize the full covariance quadratic form; do not replace it with linear independent risk scores.")
        hints.append("Do not introduce binary asset-selection, cardinality, transaction-cost, minimum-lot, or buy/sell variables.")
    if "contractallocation" in generator_id or "split_award_contract_quantity_allocation" in sub_family:
        hints.append("Model this as split-award contract quantity allocation with continuous Generation[p,c] and binary GenerationIncidence[p,c].")
        hints.append("Preserve producer capacity, exact contract fulfillment, minimum contributors, and both minimum-delivery and upper-bound activation links.")
        hints.append("Do not add facility openings, routes, regions, time periods, or award-only binary decisions.")
    elif "activation" in concepts or "fixed_cost" in concepts:
        hints.append("Include every fixed open/activation/setup cost exactly once and link assignment or flow variables to the binary activation decision.")
    if "optmath_farmplanning" in generator_id or "continuous_farm_resource_planning" in sub_family:
        hints.append("Model this as a continuous farm resource LP with no binary crop-selection or fixed planting setup variables.")
        hints.append("Preserve monthly land, labor, and water limits, the annual water limit, crop yield split, and consumption bundle normalization.")
        hints.append("Maximize net earnings from crop sales minus family/permanent/temporary labor and water costs.")
    if "optmath_team_formulation" in generator_id or "team_skill_shortage_assignment" in sub_family:
        hints.append("Model this as team skill-shortage minimization, not compatibility-score maximization.")
        hints.append("Preserve Assign[p,c], AttainedSkill[c,s], SkillShortage[c,s], and MaxSkillShortage.")
        hints.append("Preserve exactly-one project assignment per person, at-least-one person per project, attained-skill balances, shortage gap constraints, and max-shortage bounds.")
    if "optmath_cut_edited" in generator_id or "integer_cutting_stock_pattern_count" in sub_family:
        hints.append("Model this as cutting-stock pattern-count planning with nonnegative integer Cut[j] roll-count variables.")
        hints.append("Preserve the stock roll width, complete cutting-pattern piece matrix, exact order-width fulfillment, and total raw-roll-count minimization objective.")
        hints.append("Do not reinterpret it as set cover, bin packing, routing, scheduling, or binary pattern selection.")
    if "optmath_multisetcover" in generator_id or "set_multicover" in sub_family:
        hints.append("Model this as binary set multicover with lower-bound coverage requirements; preserve the complete set-element coverage matrix.")
        hints.append("Do not downgrade multicover to ordinary one-cover set cover or change >= required coverage into exact equality.")
    if "optmath_the_p_dispersion_model" in generator_id or "p_dispersion_min_distance" in sub_family:
        hints.append("Model this as p-dispersion: select exactly p nodes and maximize the continuous minimum distance among selected pairs.")
        hints.append("Preserve binary node selection, binary pair selection, pairwise distance table, and min-distance upper bounds for selected pairs.")
        hints.append("Do not add fixed opening costs, demand, assignments, capacities, flows, or maxisum total-distance objective.")
    if "optmath_mcnd" in generator_id or "multi_commodity_network_design" in sub_family:
        hints.append("Model this as multicommodity capacitated network design with continuous x[k,i,j] flow fractions and nonnegative integer y[i,j] arc facility counts.")
        hints.append("Preserve commodity-specific origins, destinations, demands, routing costs, declared arcs, facility capacities, fixed facility costs, flow conservation, and demand-weighted capacity linking.")
        hints.append("Do not make y[i,j] binary and do not collapse commodities into one aggregate flow.")
    if "optmath_supplychain" in generator_id or "fixed_charge_capacitated_network_flow" in sub_family:
        hints.append("Model this as fixed-charge sparse directed network flow with binary arc activation y[i,j] and continuous flow x[i,j] only on declared arcs.")
        hints.append("Preserve fixed arc activation costs, unit flow costs, arc capacity activation linking, and node balance as inflow minus outflow equals demand minus supply.")
        hints.append("Do not replace arc activation with node facility opening, add inventory/time periods, or imply a complete graph.")
    if "optmath_multi" in generator_id or "balanced_multi_commodity_transportation" in sub_family:
        hints.append("Model this as balanced multi-commodity transportation with continuous Transport[i,j,p], exact origin-product supply, exact destination-product demand, and shared lane capacities.")
        hints.append("Do not collapse products, add binary lane activation, fixed costs, vehicles, service frequencies, inventory, or backlog.")
    if "optmath_nltrans" in generator_id or "capacitated_balanced_transportation" in sub_family:
        hints.append("Model this as capacitated balanced transportation with continuous Shipping[i,j], exact supply and demand equalities, lane costs, and lane capacity limits.")
        hints.append("Do not add vehicles, routes, subtours, time windows, unmet demand, or optional unused supply.")
    if "optmath_netmcol" in generator_id or "continuous_multi_commodity_flow" in sub_family:
        hints.append("Model this as continuous multi-commodity network flow with Ship[i,j,p], city-product flow balance, product-specific link capacity, and joint link capacity.")
        hints.append("Do not add binary activation, fixed costs, path selection, vehicle routing, or collapse products into one commodity.")
    if "optmath_the_multi_factory_schedule_problem" in generator_id or "fixed_charge_multi_factory_production" in sub_family:
        hints.append("Model this as fixed-charge monthly factory production with binary RunDecision[m,f] and continuous Production[m,f].")
        hints.append("Preserve fixed run costs, unit production costs, min/max production-if-running links, and monthly demand satisfaction.")
        hints.append("Do not reinterpret it as job sequencing, flow shop, inventory planning, routing, or product-level lot sizing.")
    if "optmath_revenue_management" in generator_id or "optmath_revenue" in generator_id:
        hints.append("Model this as static resource-capacity revenue management with nonnegative integer PackageSales[p].")
        hints.append("Preserve package revenue maximization, package demand upper bounds, resource capacities, and the complete package-resource usage matrix.")
        hints.append("Do not introduce dynamic pricing, booking periods, stochastic arrivals, nonlinear demand curves, bid-price controls, or cost minimization.")
    if "optmath_marketshare" in generator_id or "integer_supply_allocation" in concepts:
        hints.append("Model this as nonnegative integer company-market-product supply allocation, not revenue-management capacity allocation.")
        hints.append("In Gurobi, create Supply[i,j,k] with lb=0 and vtype=GRB.INTEGER (or gp.GRB.INTEGER); do not use continuous supply variables.")
        hints.append("Preserve exact market-product demand fulfillment: for each market-product pair, sum over companies/initiatives equals the listed demand.")
        hints.append("Maximize net profit from the listed company-market-product unit-profit table.")
        hints.append("Do not introduce resource capacity, budget, channel-capacity, eligibility, nonlinear market-response, or optional unmet-demand constraints.")
    if "supplychain" in generator_id or "inventory_or_shipment_balance" in concepts:
        hints.append("Preserve all echelon, period, flow, production, inventory, capacity, and demand-balance relationships from the statement.")
    if "net" in generator_id or "flow_balance" in core_constraints:
        hints.append("Preserve node or commodity flow-balance constraints and arc capacity/activation logic; do not replace them with independent assignments.")
    if "transp" in generator_id or "transportation_problem" in str(row.get("sub_family") or "").lower() or "balanced_transportation_lp" in sub_family:
        hints.append("Model this as a balanced transportation LP with continuous nonnegative Transport[i,j] shipment variables, exact origin supply equalities, exact destination demand equalities, and complete lane-cost minimization.")
        hints.append("Do not add vehicles, routes, subtours, time windows, lane capacities, fixed charges, binary lane activation, facility opening, unmet-demand slack, or optional unused supply.")
    if "netasgn" in generator_id:
        hints.append("Model this as continuous assignment hours from people to projects, not binary one-to-one matching.")
        hints.append("Preserve exact person supply-hour equality, exact project demand-hour equality, person-project upper bounds, and cost-per-hour minimization.")
    elif "binary_assignment" in concepts:
        hints.append("Preserve assignment balance, eligibility, exclusivity, and capacity constraints; do not introduce routing continuity.")
    if "netthru" in generator_id:
        hints.append("Preserve through-node conservation and any source/sink balance exactly; do not collapse throughput into independent source-destination choices.")
    if "portfolio" in generator_id or "risk_constraint" in concepts:
        hints.append("Preserve budget, return, risk, allocation bounds, and only add cardinality or selection structure if it is explicitly present in the source.")
    if "factory_planning" in generator_id or "multi_period_production" in concepts:
        hints.append("Preserve separate Production[t,p], Sales[t,p], and Inventory[t,p] variables; do not collapse the model into a static product mix.")
        hints.append("Preserve period-by-period inventory balance with zero initial inventory and required final inventory targets.")
        hints.append("Preserve machine-hour capacity constraints for every period-machine pair after maintenance.")
        hints.append("Preserve product-period sales upper bounds; do not replace them with one inferred total demand per product.")
        hints.append("Preserve inventory upper bounds and all sales revenue and inventory holding cost coefficients.")
    elif "optmath_steel4" in generator_id or "continuous_stage_capacity_product_mix" in sub_family:
        hints.append("Model this as a static continuous product-mix LP with one Production[p] variable per product.")
        hints.append("Maximize profit_per_ton[p] * Production[p]; do not minimize cost and do not add inventory, setup, sequencing, or time-period variables.")
        hints.append("Preserve every product lower commitment Production[p] >= commit[p] and upper market bound Production[p] <= market[p].")
        hints.append("Preserve every stage capacity as sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]. Do not confuse product market tons with stage available hours.")
    elif "product_mix" in concepts:
        hints.append("Preserve product-level production variables, resource capacity constraints, demand or sales bounds, and all objective coefficients.")
    if "optmath_clsp_expand_capacity" in generator_id or (
        "capacitated_lot_sizing" in sub_family and "uncapacitated" not in sub_family
    ):
        hints.append("Model this as capacitated lot sizing without backlog: continuous Production[p,t], continuous nonnegative EndingInventory[p,t], and binary Setup[p,t].")
        hints.append("Preserve cumulative inventory balance using declared period demand, setup big-M by remaining demand, unit production cost, fixed setup cost, holding cost, and every fixed period capacity limit.")
        hints.append("Do not add Backlog, lost sales, unmet-demand slack, capacity-expansion/overtime variables, expansion costs, or a separate zero-final-inventory hard constraint.")
        hints.append("Do not treat cumulative-demand helper values as independent period demands; use them only for setup big-M/remaining-demand logic.")
    if "optmath_uncapacitatedlotsizingbacklogging" in generator_id or "un_capacitated_lot_sizing_with_backlogging" in sub_family or "uncapacitated_lot_sizing_with_backlogging" in sub_family:
        hints.append("Model this as uncapacitated lot sizing with carried backlog state: OrderedAmount[t], EndingInventory[t], BackloggedAmount[t], and binary OrderIsPlaced[t].")
        hints.append("Preserve net inventory balance with EndingInventory[t] - BackloggedAmount[t], initial zero inventory/backlog, final zero inventory/backlog, total-demand setup linking, fixed ordering setup cost, unit order cost, holding cost, and backlog penalty.")
        hints.append("Do not add production capacity constraints, lost-sales variables, unmet-demand slack, independent shortage variables, or resource constraints.")
        hints.append("Do not drop BackloggedAmount or backlog penalty; backlog is a carried state across periods, not one-period lost demand.")
    elif "optmath_uncapacitatedlotsizing" in generator_id or "un_capacitated_lot_sizing" in sub_family or "uncapacitated_lot_sizing" in sub_family:
        hints.append("Model this as uncapacitated lot sizing without backlog: OrderedAmount[t], EndingInventory[t], and binary OrderIsPlaced[t].")
        hints.append("Preserve inventory balance, fixed ordering setup cost, unit order cost, holding cost, total-demand setup linking, initial inventory, and required final inventory.")
        hints.append("Do not add BackloggedAmount, backlog penalties, lost sales, unmet-demand slack, production capacity constraints, resource limits, or a zero-final-backlog constraint.")
    if "optmath_singlelevelsmallbucket" in generator_id or "single_level_small_bucket" in sub_family:
        hints.append("Model this as single-level small-bucket lot sizing with item-machine-period production states, startup events, production amounts, stock, and backlog.")
        hints.append("Preserve only the source objective terms: setup cost times binary production state, startup cost times binary startup event, holding cost times stock, and backlog cost times backlog.")
        hints.append("Do not add a per-unit production cost, material cost, revenue term, aggregate production variable, routing variable, or placeholder numeric table.")
        hints.append("Preserve machine-specific capacity, startup time consumption, one item per machine-period, initial startup equals production state, and period-to-period startup transition logic.")
    if "optmath_structure_based_assignment" in generator_id or "nmr_peak_amino_acid_structure_assignment" in sub_family:
        hints.append("Model this as NMR peak-to-amino-acid structure assignment, not as staff/job/equipment/vehicle assignment.")
        hints.append("Use binary Assign[peak, amino_acid], minimize declared assignment cost, preserve each peak at most one amino acid, each amino acid at most one peak, and exact total assignment count.")
        hints.append("Preserve NOE compatibility constraints using only declared NOE peak pairs and the declared amino-acid compatibility/distance matrix.")
        hints.append("Do not introduce vehicles, routes, workers, jobs, workloads, capacities, flow balance, travel distances, or optional assignment in place of the exact assignment-count requirement.")
    if "lotsizing" in generator_id or "inventory_balance" in concepts:
        hints.append("Preserve time-indexed inventory recursion, period demand, setup/activation variables if present, holding/backlog costs, and capacity limits.")
    if "schedulingproblem" in generator_id or "flowshop" in generator_id or "jopshop" in generator_id:
        hints.append("Preserve sequencing, machine/station assignment, processing times, precedence/non-overlap, and the exact makespan/lateness objective.")
    if "vrptw" in generator_id or "routing_binary_variable" in concepts:
        hints.append("Preserve route arc binaries, visit-once logic, route continuity, vehicle capacity, and time-window constraints when present.")
    if "setcover" in generator_id or "coverage_matrix" in concepts:
        hints.append("Preserve coverage multiplicity and every coverage matrix entry; do not weaken multi-cover requirements to simple coverage.")
    forbidden_changes = signature.get("forbidden_changes") or []
    for change in forbidden_changes:
        hints.append(f"Avoid forbidden change: {change}.")
    return _dedupe_keep_order(hints)


def _static_signature_issues(
    row: dict[str, Any],
    payload: dict[str, Any],
    code: str,
    *,
    family_contract: dict[str, Any] | None = None,
) -> list[str]:
    generator_id = str(row.get("generator_id") or "").lower()
    signature = row.get("canonical_math_signature") or {}
    concepts = {str(value).lower() for value in row.get("concept_tags") or []}
    core_constraints = {str(value).lower() for value in signature.get("core_constraints") or []}
    required_tables = {str(value).lower() for value in signature.get("required_tables") or []}
    combined = "\n".join(
        [
            str(payload.get("modeling_explanation") or ""),
            str(payload.get("math_model") or ""),
            code,
        ]
    ).lower()
    issues: list[str] = []

    expected_sense = _expected_objective_sense(str(row.get("problem_statement") or ""))
    code_sense = _code_objective_sense(code)
    if expected_sense and code_sense and expected_sense != code_sense:
        issues.append(f"OBJECTIVE_SENSE_MISMATCH expected={expected_sense} code={code_sense}")

    if ("fixed_cost_vector" in required_tables or "fixed_cost" in concepts) and not _contains_any(
        combined,
        ("fixed", "opening cost", "setup cost", "operating cost", "activation cost", "construction cost", "open_"),
    ):
        issues.append("MISSING_FIXED_COST_SIGNAL")
    if "open_before_serve_linking" in core_constraints and not _contains_any(
        combined,
        ("link", "only if", "open before serve", "open_", "activate", "activation"),
    ):
        issues.append("MISSING_OPEN_BEFORE_SERVE_LINKING_SIGNAL")
    if "flow_balance" in core_constraints and not _contains_any(combined, ("balance", "conservation", "flow")):
        issues.append("MISSING_FLOW_BALANCE_SIGNAL")
    if "visit_once" in core_constraints and not _contains_any(combined, ("visit", "once", "customer", "node", "tour")):
        issues.append("MISSING_VISIT_ONCE_SIGNAL")
    should_check_time_window = "vrptw" in generator_id or _contains_any(
        str(row.get("problem_statement") or "").lower(),
        ("capacity", "time window", "time_window", "window"),
    )
    if should_check_time_window and "capacity_or_time_window" in core_constraints and not _contains_any(
        combined,
        ("capacity", "time window", "time_window", "window"),
    ):
        issues.append("MISSING_CAPACITY_OR_TIME_WINDOW_SIGNAL")
    if "risk_constraint" in concepts and not _contains_any(combined, ("risk", "variance", "volatility")):
        issues.append("MISSING_RISK_SIGNAL")
    source_context = _source_context_text(row)
    issues.extend(
        _family_contract_issues(
            family_contract or {},
            combined,
            code,
            source_context=source_context,
            expected_sense=expected_sense,
        )
    )

    return issues


def _expected_objective_sense(problem_statement: str) -> str | None:
    lowered = problem_statement.lower()
    if re.search(r"objective\s+is\s+to\s+minimi[sz]e|goal\s+is\s+to\s+minimi[sz]e|minimi[sz]e\s+total", lowered):
        return "minimize"
    if re.search(r"objective\s+is\s+to\s+maximi[sz]e|goal\s+is\s+to\s+maximi[sz]e|maximi[sz]e\s+total", lowered):
        return "maximize"
    return None


def _code_objective_sense(code: str) -> str | None:
    lowered = code.lower()
    if "grb.minimize" in lowered:
        return "minimize"
    if "grb.maximize" in lowered:
        return "maximize"
    return None


def _family_contract_issues(
    family_contract: dict[str, Any],
    combined: str,
    code: str,
    *,
    source_context: str = "",
    expected_sense: str | None = None,
) -> list[str]:
    answer_contract = family_contract.get("answer_contract") if isinstance(family_contract, dict) else {}
    if not isinstance(answer_contract, dict):
        return []
    issues: list[str] = []
    code_sense = _code_objective_sense(code)
    sense_options = [str(value).lower() for value in answer_contract.get("objective_sense_options") or []]
    if len(sense_options) == 1 and code_sense and code_sense != sense_options[0]:
        # Trust the concrete generated NL statement over broad family defaults
        # when they disagree; many OptMATH families contain max/min variants.
        if not (expected_sense and expected_sense == code_sense and expected_sense not in sense_options):
            issues.append(f"CONTRACT_OBJECTIVE_SENSE_MISMATCH expected={sense_options[0]} code={code_sense}")

    required_variable_families = [str(value).lower() for value in answer_contract.get("required_variable_families") or []]
    required_variable_families = [
        value for value in required_variable_families
        if _contract_item_applies(value, source_context)
    ]
    if any(_contract_variable_family_requires_binary_signal(value) for value in required_variable_families):
        if not _contains_any(code.lower(), ("grb.binary", "vtype=grb.binary", "vtype = grb.binary", "binary")):
            issues.append("CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL")

    enforce_required_constraints = family_contract.get("contract_coverage") != "low"
    for constraint in answer_contract.get("required_constraints") or []:
        if not enforce_required_constraints:
            continue
        constraint_name = str(constraint).lower()
        if not _contract_item_applies(constraint_name, source_context):
            continue
        if "nonnegative_integer" in constraint_name:
            if not _contains_any(code.lower(), ("grb.integer", "vtype=grb.integer", "vtype = grb.integer")):
                issues.append(f"CONTRACT_MISSING_REQUIRED_CONSTRAINT:{constraint_name}")
            continue
        signals = _constraint_signals(constraint_name)
        if signals and not _contains_any(combined, signals):
            issues.append(f"CONTRACT_MISSING_REQUIRED_CONSTRAINT:{constraint_name}")

    for objective_term in answer_contract.get("required_objective_terms") or []:
        if family_contract.get("contract_coverage") == "low":
            continue
        term_name = str(objective_term).lower()
        if not _contract_item_applies(term_name, source_context):
            continue
        signals = _objective_term_signals(term_name)
        if signals and not _contains_any(combined, signals):
            issues.append(f"CONTRACT_MISSING_REQUIRED_OBJECTIVE_TERM:{term_name}")

    for forbidden_change in answer_contract.get("forbidden_changes") or []:
        violated = _forbidden_change_violation(str(forbidden_change).lower(), combined, code)
        if violated:
            issues.append(f"CONTRACT_FORBIDDEN_CHANGE:{violated}")

    return _dedupe_keep_order(issues)


def _forbidden_change_violation(forbidden_change: str, combined: str, code: str) -> str | None:
    code_lower = code.lower()
    combined_lower = combined.lower()
    code_for_forbidden_state = re.sub(r"(?:without|no)[_ -]?backlog", "nobgok", code_lower)
    if "peaks and amino acids" in forbidden_change:
        nmr_misframing_terms = (
            "staff",
            "staffing",
            "job",
            "jobs",
            "worker",
            "workers",
            "technician",
            "technicians",
            "service job",
            "service jobs",
            "inspection job",
            "inspection jobs",
            "field-service",
            "dispatch manager",
            "provider",
            "providers",
            "service provider",
            "service providers",
            "request",
            "requests",
            "transport request",
            "transport requests",
            "asset",
            "assets",
            "patient transport",
            "industrial plant",
            "production scheduler",
            "production peak",
            "production peaks",
            "product slot",
            "product slots",
            "slot",
            "slots",
            "shift",
            "shifts",
            "shift position",
            "shift positions",
            "vehicle",
            "vehicles",
            "route",
            "routes",
            "gate assignment",
            "crew assignment",
        )
        if any(contains_unnegated_phrase(combined_lower, term) for term in nmr_misframing_terms):
            if any(contains_unnegated_phrase(combined_lower, term) for term in ("vehicle", "vehicles", "route", "routes")):
                return "vehicle_routing_structure_added"
            return "nmr_domain_misframed_as_staffing"
    if "per-unit production cost" in forbidden_change or "unit production cost" in forbidden_change:
        if re.search(r"\b(?:prod_cost|production_cost|unit_production_cost|material_cost)\b", code_lower):
            return "unit_production_cost_added"
    if "binary product-selection" in forbidden_change or "binary product selection" in forbidden_change:
        if _contains_any(code_lower, ("grb.binary", "vtype=grb.binary", "vtype = grb.binary")):
            return "binary_product_selection_added"
    if (
        "zero final inventory" not in forbidden_change
        and re.search(r"do not (?:add|introduce).*inventory|storage variables", forbidden_change)
    ):
        if _contains_any(code_lower, ("inventory", "stock", "storage")):
            return "inventory_or_backlog_added"
    if re.search(r"do not (?:add|introduce)[^.]*backlog|no backlog", forbidden_change):
        forbidden_state_patterns = (
            r"\b(?:backlog|backlogged|shortage|lost_sales|lostsales|unmet_demand|unmetdemand)\s*=\s*model\.addvars?\b",
            r"\b(?:b|b_t|backlog|backlogged|shortage|lost_sales|lostsales|unmet_demand|unmetdemand)\s*=\s*model\.addvars?\([^)]*name\s*=\s*['\"][^'\"]*(?:backlog|backlogged|shortage|lost[_ -]?sales|unmet[_ -]?demand)[^'\"]*['\"]",
            r"name\s*=\s*['\"][^'\"]*(?:backlog|backlogged|shortage|lost[_ -]?sales|unmet[_ -]?demand)[^'\"]*['\"]",
            r"\b(?:backlog_penalty|final_backlog|zero_final_backlog|backloggedamount|lostsales|unmetdemand)\b",
        )
        if any(re.search(pattern, code_for_forbidden_state) for pattern in forbidden_state_patterns):
            return "backlog_added"
    if "zero final inventory" in forbidden_change and (
        "do not add" in forbidden_change or "separate hard constraint" in forbidden_change
    ):
        if "zero_final_inventory" in code_lower or "finalinventory" in code_lower:
            return "zero_final_inventory_added"
        if re.search(r"(final|last).{0,80}(inventory|stock).{0,80}(==|<=)\s*0", code_lower, re.DOTALL):
            return "zero_final_inventory_added"
        if re.search(r"(inventory|stock).{0,80}(final|last).{0,80}(==|<=)\s*0", code_lower, re.DOTALL):
            return "zero_final_inventory_added"
    if "vehicles" in forbidden_change and "routes" in forbidden_change:
        if _contains_any(code_lower, ("vehicle", "subtour", "time_window", "time window")):
            return "vehicle_routing_structure_added"
    return None


def _constraint_signals(constraint_name: str) -> tuple[str, ...]:
    if "assignment_only_if_eligible" in constraint_name:
        return ("assignments", "eligibility", "eligible", "<=", "only if")
    if "participant_at_most_one_car" in constraint_name:
        return ("participant", "at most one", "assignments", "<=")
    if "car_at_most_one_participant" in constraint_name:
        return ("car", "at most one", "assignments", "<=")
    if "region_coverage_linking" in constraint_name:
        return ("covered", "coverage", "build", "tower", "<=")
    if "tower_budget_limit" in constraint_name:
        return ("budget", "setup cost", "build", "<=")
    if "customer_demand_exactly_satisfied" in constraint_name:
        return ("customer", "demand", "exact", "shippedamount", "==")
    if "ship_only_from_open_facility" in constraint_name:
        return ("facilityopen", "shippedamount", "open", "only if", "demand")
    if "open_facility_capacity_limit" in constraint_name:
        return ("facilityopen", "capacity", "shippedamount", "<=")
    if "package_demand_upper_bound" in constraint_name:
        return ("packagesales", "demand", "upper", "<=")
    if "resource_capacity_consumption_limit" in constraint_name:
        return ("resource", "capacity", "usage", "packagesales", "<=")
    if "nonnegative_integer_package_sales" in constraint_name:
        return ("packagesales", "integer", "nonnegative", "grb.integer")
    if "market_product_demand_exactly_satisfied" in constraint_name:
        return ("market", "product", "demand", "supply", "==")
    if "nonnegative_integer_supply" in constraint_name:
        return ("supply", "integer", "nonnegative", "grb.integer")
    if "each_amino_acid_at_most_one_peak" in constraint_name:
        return ("amino acid", "amino_acid", "at most one", "<=")
    if "each_peak_at_most_one_amino_acid" in constraint_name:
        return ("peak", "at most one", "amino", "<=")
    if "exact_total_assignment_count" in constraint_name:
        return ("exactly", "total assignment", "assignment_count", "required_assignment_count", "==")
    if "noe_compatibility_pair_constraints" in constraint_name:
        return ("noe", "compatibility", "compatible", "compatibility_matrix", "<=")
    if "origin_supply_exactly_shipped" in constraint_name:
        return ("origin", "supply", "transport", "exact", "==")
    if "destination_demand_exactly_received" in constraint_name:
        return ("destination", "demand", "transport", "exact", "==")
    if "nonnegative_continuous_shipments" in constraint_name:
        return ("continuous", "nonnegative", "shipment", "transport")
    if "aircraft_type_availability" in constraint_name:
        return ("aircraft type", "availability", "available", "allocation", "<=")
    if "route_capacity_meets_or_exceeds_demand" in constraint_name or "route_demand_requirement" in constraint_name:
        return ("route", "demand", "capacity", "allocation", ">=")
    if "one_precedence_order_for_each_aircraft_pair" in constraint_name:
        return ("precedence", "aircraftorder", "aircraft order", "exactly one", "pair")
    if "minimum_separation_time_between_ordered_landings" in constraint_name:
        return ("separation", "landing", "aircraftorder", "before", "after")
    if "aircraft_landing_time_window" in constraint_name:
        return ("earliest", "latest", "landing", "time window", "window")
    if "early_deviation_lower_bound" in constraint_name:
        return ("early", "target", "landing", "deviation")
    if "late_deviation_lower_bound" in constraint_name:
        return ("late", "target", "landing", "deviation")
    if "exactly_one_inbound_arc_per_city" in constraint_name:
        return ("inbound", "exactly one", "city", "route")
    if "exactly_one_outbound_arc_per_city" in constraint_name:
        return ("outbound", "exactly one", "city", "route")
    if "single_tour_mtz_subtour_elimination" in constraint_name:
        return ("mtz", "subtour", "u[", "order")
    if "source_sink_flow_balance" in constraint_name:
        return ("source", "sink", "outflow", "inflow", "balance")
    if "intermediate_node_flow_conservation" in constraint_name:
        return ("intermediate", "flow conservation", "outflow", "inflow", "0")
    if "each_demand_point_assigned_exactly_once" in constraint_name:
        return ("demand point", "exactly once", "assign", "assignment")
    if "assign_only_to_selected_location" in constraint_name:
        return ("assign", "selectedlocation", "selected location", "only if", "link")
    if "selected_location_capacity_with_existing_and_new_scooters" in constraint_name:
        return ("existing scooters", "new scooters", "capacity", "selectedlocation")
    if "maximum_selected_locations" in constraint_name:
        return ("max selected", "maximum selected", "selected locations", "limit")
    if "total_new_scooter_budget" in constraint_name:
        return ("new scooter", "budget", "newmax", "total")
    if "new_scooters_only_at_selected_locations" in constraint_name:
        return ("newscooters", "selectedlocation", "only if", "<=", "link")
    if "commodity_flow_conservation" in constraint_name:
        return ("flow conservation", "commodity", "origin", "destination", "delta")
    if "arc_capacity_linked_to_integer_facility_count" in constraint_name:
        return ("capacity", "facility", "integer", "y[", "demand")
    if "flow_fraction_bounds" in constraint_name:
        return ("flow fraction", "<= 1", "bounds", "x[")
    if "origin_product_supply" in constraint_name:
        return ("origin", "product", "supply", "transport")
    if "destination_product_demand" in constraint_name:
        return ("destination", "product", "demand", "transport")
    if "origin_destination_joint_capacity" in constraint_name:
        return ("shared lane capacity", "joint capacity", "sum_p", "products")
    if "origin_supply" in constraint_name:
        return ("origin", "supply", "shipping", "supply")
    if "destination_demand" in constraint_name:
        return ("destination", "demand", "shipping", "demand")
    if "lane_capacity" in constraint_name:
        return ("lane capacity", "limit", "shipping")
    if "city_product_flow_balance" in constraint_name:
        return ("city", "product", "flow balance", "supply")
    if "directed_link_joint_capacity" in constraint_name:
        return ("joint capacity", "directed link", "sum", "ship")
    if "directed_link_product_capacity" in constraint_name:
        return ("product-specific", "capacity", "directed link", "ship")
    if "arc_capacity_activation_linking" in constraint_name:
        return ("capacity", "activation", "binary", "x[", "y[")
    if "node_flow_balance" in constraint_name:
        return ("flow balance", "inflow", "outflow", "demand", "supply")
    if "exact_order_width_fulfillment" in constraint_name:
        return ("exact", "order width", "fulfillment", "cut", "pieces")
    if "pattern_width_feasibility" in constraint_name:
        return ("roll width", "pattern", "width", "feasible")
    if "select_exactly_p_nodes" in constraint_name:
        return ("exactly p", "select", "nodes", "== p")
    if "pair_selection_upper_bound_left_node" in constraint_name:
        return ("pairselection", "pair selection", "<= x", "left")
    if "pair_selection_upper_bound_right_node" in constraint_name:
        return ("pairselection", "pair selection", "<= x", "right")
    if "pair_selection_lower_bound_both_nodes" in constraint_name:
        return ("pairselection", "pair selection", "x_i + x_j - 1")
    if "min_distance_upper_bound_for" in constraint_name:
        return ("mindistance", "minimum distance", "pairwise distance", "big-m")
    if "exact_contract_fulfillment" in constraint_name:
        return ("contract fulfillment", "exact", "=", "contract size", "fulfill")
    if "minimum_contributors_per_contract" in constraint_name:
        return ("minimum contributors", "min contributors", "contributors", "incidence")
    if "delivery_activation_lower_bound" in constraint_name:
        return ("minimum delivery", "min delivery", ">=", "incidence", "generationincidence")
    if "delivery_activation_upper_bound" in constraint_name:
        return ("upper bound", "<=", "contract size", "incidence", "generationincidence")
    if "monthly_land_occupation_capacity" in constraint_name:
        return ("monthly land", "land occupation", "land capacity")
    if "monthly_labor_requirement_coverage" in constraint_name:
        return ("monthly labor", "labor requirement", "labor coverage", "temporary labor")
    if "monthly_water" in constraint_name:
        return ("monthly water", "water limit", "water requirement")
    if "annual_water" in constraint_name:
        return ("annual water", "annual_water", "total water")
    if "crop_yield" in constraint_name:
        return ("yield", "sales", "consumption", "crop yield")
    if "family_consumption_bundle" in constraint_name:
        return ("consumption bundle", "fractionconsumed", "bundle fraction")
    if "each_person_assigned_to_exactly_one_project" in constraint_name:
        return ("exactly one", "person", "project", "assign")
    if "each_project_receives_at_least_one_person" in constraint_name:
        return ("at least one", "project", "person", "staff")
    if "attained_skill_equals" in constraint_name:
        return ("attainedskill", "attained skill", "individual skill", "sum")
    if "skill_shortage_covers" in constraint_name:
        return ("skillshortage", "skill shortage", "requirement gap", "requiredskill")
    if "max_skill_shortage" in constraint_name:
        return ("maxskillshortage", "maximum skill shortage", "skillshortage")
    if "minimum_production_if_running" in constraint_name:
        return ("minimum production", "min production", "if running", "rundecision")
    if "maximum_production_if_running" in constraint_name:
        return ("maximum production", "max production", "if running", "rundecision")
    if "monthly_demand_satisfaction" in constraint_name:
        return ("monthly demand", "demand satisfaction", "production")
    if "served_zone_commodity_demand_exactly_shipped" in constraint_name:
        return ("shipped", "served", "demand", "flow")
    if "each_zone_assigned" in constraint_name or "zone_assigned" in constraint_name:
        return ("exactly one", "assigned", "served")
    if "distribution_center_minimum_throughput" in constraint_name:
        return ("minimum throughput", "minthroughput", "min_throughput")
    if "distribution_center_maximum_throughput" in constraint_name:
        return ("maximum throughput", "maxthroughput", "max_throughput", "capacity")
    if "initial_fleet_balance" in constraint_name:
        return ("initial", "init", "balance")
    if "time_expanded_fleet_flow_conservation" in constraint_name:
        return ("time-expanded", "time expanded", "flow conservation", "idle", "balance")
    if "fleet_availability" in constraint_name:
        return ("availability", "available", "fleet count")
    if "active_leg_passenger_demand_satisfaction" in constraint_name:
        return ("active leg", "passenger demand", "demand satisfaction")
    if "inactive_leg_route_restriction" in constraint_name:
        return ("inactive", "leg_is_active", "route restriction", "delta")
    if "full_investment_budget" in constraint_name or "weights_sum_to_one" in constraint_name:
        return ("sum", "weights", "= 1", "budget")
    if "minimum_expected_return" in constraint_name:
        return ("target return", "expected return", "minimum return")
    if "continuous_asset_weight_bounds" in constraint_name:
        return ("weight bounds", "lower", "upper", "lb", "ub")
    if "sales" in constraint_name:
        return ("sales upper", "sales limit", "market cap", "forecast cap", "demand cap")
    if "zero_final_inventory" in constraint_name:
        return ("final period", "final inventory", "ending inventory", "backlog", "zero", "= 0")
    if "final_inventory" in constraint_name or "inventory_target" in constraint_name:
        return ("final inventory", "target inventory", "end-horizon", "final stock")
    if "inventory_upper" in constraint_name or ("inventory" in constraint_name and "bound" in constraint_name):
        return ("inventory upper", "storage limit", "max inventory", "inventory limit")
    if "inventory_balance" in constraint_name:
        return ("inventory balance", "carry", "previous inventory", "ending inventory", "balance")
    if "material_balance" in constraint_name:
        return ("sum of all", "sum_i", "sum(", "total blend", "total weight", "proportion", "= 1", "== 1")
    if "demand" in constraint_name:
        return ("demand", "satisf", "serve", "requirement")
    if "capacity" in constraint_name:
        return ("capacity", "limit", "resource", "machine", "processing", "nonoverlap", "work center", "weight")
    if "link" in constraint_name or "open_to" in constraint_name or "activation" in constraint_name:
        return ("link", "activate", "open", "only if", "binary")
    if "balance" in constraint_name or "conservation" in constraint_name:
        return ("balance", "conservation", "flow")
    if "coverage" in constraint_name or "covered" in constraint_name:
        return ("cover", "covered", "coverage")
    if "visit" in constraint_name or "service" in constraint_name:
        return ("visit", "service", "served")
    if "machine" in constraint_name or "resource" in constraint_name or "processing" in constraint_name:
        return ("machine", "resource", "processing", "nonoverlap", "capacity", "assignment")
    if "time_window" in constraint_name:
        return ("time window", "window", "arrival")
    if "risk" in constraint_name:
        return ("risk", "variance", "volatility")
    if "budget" in constraint_name:
        return ("budget", "allocation", "weight")
    if "quality" in constraint_name or "ratio" in constraint_name:
        return ("quality", "ratio", "blend", "specification")
    if "nutrient" in constraint_name:
        return ("nutrient", "minimum", "maximum")
    return ()


def _objective_term_signals(term_name: str) -> tuple[str, ...]:
    if "one_unit_value_per_selected_assignment" in term_name:
        return ("assignments", "maximize", "sum", "eligible")
    if "region_population_times_covered_binary" in term_name:
        return ("population", "covered", "maximize")
    if "facility_fixed_opening_cost" in term_name:
        return ("fixed", "opening", "facilityopen", "facility")
    if "customer_facility_transport_cost_times_shipped_amount" in term_name:
        return ("transport", "cost", "shippedamount", "customer", "facility")
    if "package_revenue_times_package_sales" in term_name:
        return ("revenue", "packagesales", "maximize")
    if "net_profit_times_integer_supply" in term_name:
        return ("net profit", "unit_profit", "supply", "maximize")
    if "assignment_cost_times_binary_assignment" in term_name:
        return ("assignment cost", "cost matrix", "cost[p", "cost_", "minimize")
    if "lane_unit_cost_times_transport_quantity" in term_name:
        return ("lane", "cost", "transport", "shipment")
    if "aircraft_route_operating_cost" in term_name or "aircraft-route operating cost" in term_name:
        return ("aircraft", "route", "operating cost", "allocation")
    if "early_landing_penalty" in term_name:
        return ("early", "landing", "penalty")
    if "late_landing_penalty" in term_name:
        return ("late", "landing", "penalty")
    if "directed_arc_distance_times_route_binary" in term_name:
        return ("distance", "route", "arc", "travel")
    if "arc_cost_times_binary_path_arc" in term_name:
        return ("arc cost", "path", "binary", "arcs")
    if "demand_weighted_walking_distance" in term_name:
        return ("demand", "distance", "walking", "assign")
    if "station_opening_cost" in term_name:
        return ("opening cost", "selectedlocation", "station")
    if "new_scooter_deployment_cost" in term_name:
        return ("new scooter", "deployment cost", "newscooters")
    if "demand_weighted_commodity_arc_routing_cost" in term_name:
        return ("demand", "routing cost", "commodity", "x[")
    if "fixed_facility_installation_cost" in term_name:
        return ("fixed", "facility", "installation", "y[")
    if "origin_destination_product_unit_shipping_cost" in term_name or "unit_shipping_cost_times_transport_quantity" in term_name:
        return ("shipping cost", "transport", "product", "unit")
    if "lane_shipping_cost" in term_name or "lane_rate_times_shipping_quantity" in term_name:
        return ("shipping", "rate", "cost")
    if "product_link_shipping_cost" in term_name:
        return ("shipment cost", "product", "link", "ship")
    if "fixed_arc_activation_cost" in term_name:
        return ("fixed", "activation", "arc", "y[")
    if "unit_arc_flow_cost" in term_name:
        return ("unit", "flow cost", "arc", "x[")
    if "arc_shipping_cost" in term_name:
        return ("arc cost", "per-unit cost", "latency cost", "flow cost", "cost[", "cost * flow", "c_{ij}")
    if "total_raw_roll_count" in term_name:
        return ("raw roll", "roll count", "sum", "cut")
    if "continuous_minimum_selected_pair_distance" in term_name or "minimum_selected_pairwise_distance" in term_name:
        return ("mindistance", "minimum distance", "maximize", "pairwise")
    if "unit_production_cost_times_delivered_quantity" in term_name:
        return ("unit production cost", "delivered", "generation", "quantity")
    if "crop_sales_revenue" in term_name:
        return ("crop sales", "sales revenue", "price")
    if "labor_cost" in term_name:
        return ("labor cost", "wage", "family labor", "temporary labor", "permanent labor")
    if "water_cost" in term_name:
        return ("water cost", "price of water", "water price")
    if "maximum_skill_shortage_priority" in term_name:
        return ("maxskillshortage", "maximum skill shortage", "1000")
    if "total_skill_shortage_secondary" in term_name:
        return ("skillshortage", "total skill shortage", "sum")
    if "fixed_factory_month_run_cost" in term_name:
        return ("fixed run cost", "rundecision", "fixed")
    if "variable_unit_production_cost" in term_name:
        return ("unit production cost", "production", "variable")
    if "covariance" in term_name or "variance" in term_name:
        return ("covariance", "variance", "quadratic")
    if "fixed_distribution_center" in term_name or "fixed" in term_name:
        return ("fixed", "selected", "opening", "open")
    if "unit_distribution_center_throughput" in term_name or "throughput_cost" in term_name:
        return ("unit throughput", "throughput cost", "unit_throughput")
    if "shipping" in term_name:
        return ("shipping", "shipped", "variable cost")
    if "fleet_type_operating_cost" in term_name:
        return ("operating cost", "fleet", "cost")
    if "cost" in term_name:
        return ("cost",)
    if "revenue" in term_name or "profit" in term_name:
        return ("revenue", "profit")
    return ()


def _contract_item_applies(contract_item: str, source_context: str) -> bool:
    lowered = contract_item.lower()
    if "_if_" not in lowered and " if " not in lowered and not lowered.endswith("_if_present"):
        return True
    if "if_present" in lowered or "if_required" in lowered or "if_source" in lowered or "if_bin_packing" in lowered:
        return _source_context_has_requirement(lowered, source_context)
    if "if_cardinality" in lowered or "if_selection" in lowered or "if_setup" in lowered:
        return _source_context_has_requirement(lowered, source_context)
    return False


def _contract_variable_family_requires_binary_signal(variable_family: str) -> bool:
    """Return True only for variable families that explicitly require binaries.

    Some valid LP families use words like "assignment" while remaining
    continuous, e.g. optmath_netasgn's continuous assigned hours. Those should
    not be rejected merely because the family name contains assignment-like
    language.
    """
    lowered = variable_family.lower()
    if "binary_or_continuous" in lowered or lowered.startswith("continuous_") or "continuous" in lowered:
        return False
    return _contains_any(
        lowered,
        (
            "binary",
            "open_",
            "select",
            "selected",
            "activation",
            "route_arc",
        ),
    )


def _source_context_has_requirement(contract_item: str, source_context: str) -> bool:
    signals = _constraint_signals(contract_item)
    if "cardinality" in contract_item:
        signals += ("cardinality", "at most", "at least", "select", "choose", "binary")
    if "selection" in contract_item:
        signals += ("select", "choose", "selected", "binary", "yes/no")
    if "setup" in contract_item:
        signals += ("setup", "fixed charge", "startup", "activation")
    if "bin_packing" in contract_item:
        signals += ("bin", "container", "packing", "item assignment")
    if not signals:
        return False
    return _contains_any(source_context, tuple(_dedupe_keep_order(list(signals))))


def _source_context_text(row: dict[str, Any]) -> str:
    signature = row.get("canonical_math_signature") or {}
    structured = row.get("structured_problem_data") or {}
    pieces = [
        str(row.get("problem_statement") or ""),
        str(row.get("source_math_formula") or ""),
        str(row.get("math_formula") or ""),
        str(signature),
        str(structured),
        " ".join(str(value) for value in row.get("concept_tags") or []),
    ]
    return "\n".join(pieces).lower()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    existing = row.get("source_metadata")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    llm_metadata = row.get("llm_metadata") or {}
    return {
        key: llm_metadata.get(key)
        for key in (
            "scenario_id",
            "base_scenario_id",
            "scenario_variant_id",
            "scenario_source",
            "task_family",
            "business_trigger",
            "industry_lens_id",
            "narrative_angle_id",
            "organization_profile_id",
            "planning_horizon_id",
            "entity_naming_style_id",
        )
        if llm_metadata.get(key) is not None
    }


def _mock_forward_model(row: dict[str, Any]) -> dict[str, Any]:
    lp_text = str(row.get("source_lp_text") or "").strip()
    if lp_text:
        code = (
            "import pathlib\n"
            "import gurobipy as gp\n\n"
            f"lp_text = {lp_text!r}\n"
            "lp_path = pathlib.Path('embedded_model.lp')\n"
            "lp_path.write_text(lp_text, encoding='utf-8')\n"
            "model = gp.read(str(lp_path))\n"
            "model.Params.OutputFlag = 0\n"
            "model.optimize()\n"
            "print({'objective_value': model.ObjVal if model.SolCount > 0 else None})\n"
        )
        return {
            "modeling_explanation": "This deterministic smoke-test model solves the embedded LP instance generated by OptMATH.",
            "math_model": row.get("source_math_formula") or "The mathematical model is represented by the embedded LP formulation.",
            "gurobipy_code": code,
        }
    return {
        "modeling_explanation": "This mock response echoes a minimal Gurobi model for pipeline testing only.",
        "math_model": "max x subject to x <= 1, x >= 0",
        "gurobipy_code": (
            "import gurobipy as gp\n"
            "from gurobipy import GRB\n"
            "model = gp.Model('mock_forward_model')\n"
            "model.Params.OutputFlag = 0\n"
            "x = model.addVar(lb=0, name='x')\n"
            "model.addConstr(x <= 1)\n"
            "model.setObjective(x, GRB.MAXIMIZE)\n"
            "model.optimize()\n"
            "print({'objective_value': model.ObjVal})\n"
        ),
    }
