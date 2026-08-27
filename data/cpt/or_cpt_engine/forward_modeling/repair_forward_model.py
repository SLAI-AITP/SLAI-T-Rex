from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cpt_cleaner.utils.code_utils import extract_executable_code

from or_cpt_engine.contracts import resolve_family_contract
from or_cpt_engine.forward_modeling.code_quality import prepare_forward_code
from or_cpt_engine.forward_modeling.generate_forward_model import (
    _modeling_guardrails,
    _short_hash,
    _source_fact_digest,
    _source_metadata,
    _static_signature_issues,
)
from or_cpt_engine.llm.openai_compatible_client import (
    AsyncOpenAICompatibleClient,
    extract_json_object,
    is_retryable_llm_error,
)
from or_cpt_engine.schemas.common import ForwardModelingOutput, LLMStageConfig
from or_cpt_engine.utils.prompt_registry import PromptSpec


PROMPT_NAME = "forward_model_repair_prompt"
_MAX_TEXT_CHARS = 12000
_MAX_LOG_CHARS = 5000


def should_attempt_forward_repair(record: dict[str, Any], repair_config: dict[str, Any] | None) -> bool:
    config = repair_config or {}
    if config.get("enabled", True) is False:
        return False
    if not bool(record.get("repairable")):
        return False
    max_attempts = max(0, int(config.get("max_attempts", 1)))
    attempted = int((record.get("forward_repair") or {}).get("attempt", 0) or record.get("repair_attempt", 0) or 0)
    if attempted >= max_attempts:
        return False
    allowed_families = {str(value) for value in config.get("repairable_failure_families") or []}
    if allowed_families and str(record.get("failure_family") or "") not in allowed_families:
        return False
    return True


async def repair_forward_model_one(
    rejected_record: dict[str, Any],
    prompt_spec: PromptSpec,
    code_dir: Path,
    *,
    client: AsyncOpenAICompatibleClient | None,
    mock: bool,
    stage_config: LLMStageConfig | None = None,
    family_contracts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt one LLM repair for a repairable 08 forward-eval rejection."""
    original_fm_id = str(rejected_record.get("fm_id") or "unknown_fm")
    attempt = int((rejected_record.get("forward_repair") or {}).get("attempt", 0) or rejected_record.get("repair_attempt", 0) or 0) + 1
    repair_fm_id = f"fmrepair_{_short_hash(f'{original_fm_id}:{attempt}')}"
    try:
        family_contract = rejected_record.get("family_contract") or resolve_family_contract(rejected_record, family_contracts)
        source_metadata = _source_metadata(rejected_record)
        if mock:
            payload = _mock_repair_payload(rejected_record)
            llm_metadata = {"mock": True}
            raw_content: Any = payload
        else:
            if client is None:
                raise RuntimeError("async client is not initialized")
            prompt = _render_repair_prompt(prompt_spec.text, rejected_record, attempt=attempt, family_contract=family_contract)
            response = await client.chat(
                prompt,
                log_context={"instance_id": str(rejected_record.get("instance_id") or ""), "prompt_name": PROMPT_NAME},
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
            }
        )
        code = extract_executable_code(payload.get("gurobipy_code") or payload.get("code") or payload.get("python_code"))
        if not code:
            raise ValueError("missing executable gurobipy code")
        code_preparation = prepare_forward_code(code)
        if not code_preparation.passed:
            return {
                "kind": "rejected",
                "rejected": _repair_rejection_record(
                    rejected_record,
                    repair_fm_id=repair_fm_id,
                    attempt=attempt,
                    reason="STATIC_CODE_CHECK_FAILED:" + "; ".join(code_preparation.issues),
                    family_contract=family_contract,
                    llm_metadata=llm_metadata,
                    payload=payload,
                    code=code_preparation.code,
                    code_preparation=code_preparation.to_dict(),
                ),
            }
        code = code_preparation.code
        static_issues = _static_signature_issues(rejected_record, payload, code, family_contract=family_contract)
        if static_issues:
            return {
                "kind": "rejected",
                "rejected": _repair_rejection_record(
                    rejected_record,
                    repair_fm_id=repair_fm_id,
                    attempt=attempt,
                    reason="STATIC_SIGNATURE_MISMATCH:" + "; ".join(static_issues),
                    family_contract=family_contract,
                    llm_metadata=llm_metadata,
                    payload=payload,
                    code=code,
                    code_preparation=code_preparation.to_dict(),
                ),
            }

        code_dir.mkdir(parents=True, exist_ok=True)
        code_path = code_dir / f"{repair_fm_id}.py"
        code_path.write_text(code, encoding="utf-8")
        output = ForwardModelingOutput(
            fm_id=repair_fm_id,
            bt_id=rejected_record["bt_id"],
            instance_id=rejected_record["instance_id"],
            generator_id=rejected_record.get("generator_id", ""),
            problem_statement=rejected_record.get("problem_statement") or "",
            structured_problem_data=rejected_record.get("structured_problem_data") or {},
            source_lp_text=rejected_record.get("source_lp_text"),
            source_math_formula=rejected_record.get("source_math_formula"),
            source_compact_data=rejected_record.get("source_compact_data") or {},
            generated_answer={
                "modeling_explanation": payload.get("modeling_explanation"),
                "math_model": payload.get("math_model"),
                "gurobipy_code": code,
                "self_check": payload.get("self_check") or {},
            },
            code_preparation=code_preparation.to_dict(),
            code_path=str(code_path),
            reference_answer=rejected_record.get("reference_answer") or {},
            generator_profile_version=rejected_record.get("generator_profile_version"),
            difficulty_level=rejected_record.get("difficulty_level"),
            sub_family=rejected_record.get("sub_family"),
            canonical_math_signature=rejected_record.get("canonical_math_signature") or {},
            concept_tags=list(rejected_record.get("concept_tags") or []),
            variant_id=rejected_record.get("variant_id"),
            source_metadata=source_metadata,
            family_contract_version=family_contract.get("version"),
            family_contract_id=family_contract.get("contract_id"),
            family_contract=family_contract,
            llm_metadata=llm_metadata,
            forward_repair={
                "attempt": attempt,
                "original_fm_id": original_fm_id,
                "original_rejection_reason": rejected_record.get("rejection_reason"),
                "original_fine_rejection_reason": rejected_record.get("fine_rejection_reason"),
                "original_failure_family": rejected_record.get("failure_family"),
            },
        )
        raw_response = {
            "fm_id": repair_fm_id,
            "original_fm_id": original_fm_id,
            "bt_id": rejected_record.get("bt_id"),
            "instance_id": rejected_record.get("instance_id"),
            "raw_response": raw_content,
            "source_metadata": source_metadata,
            "family_contract_version": family_contract.get("version"),
            "family_contract_id": family_contract.get("contract_id"),
            "family_contract": family_contract,
            "llm_metadata": llm_metadata,
            "forward_repair": output.forward_repair,
        }
        return {"kind": "output", "output": output, "raw_response": raw_response}
    except Exception as exc:  # noqa: BLE001
        if is_retryable_llm_error(exc):
            return {"kind": "requeue"}
        return {
            "kind": "rejected",
            "rejected": {
                **_repair_base(rejected_record, repair_fm_id=repair_fm_id, attempt=attempt),
                "fm_status": "REJECTED",
                "rejection_stage": "forward_repair",
                "rejection_reason": f"{exc.__class__.__name__}: {exc}",
            },
        }


def render_forward_repair_report(
    *,
    attempted: int,
    generated: int,
    recovered: int,
    failed_generation: int,
    failed_eval: int,
    requeued: int = 0,
) -> str:
    return "\n".join(
        [
            "# Forward Repair Report",
            "",
            f"- Attempted repairs: {attempted}",
            f"- Repaired forward-model outputs generated: {generated}",
            f"- Recovered accepted pairs: {recovered}",
            f"- Repair generation/static failures: {failed_generation}",
            f"- Repair outputs still rejected by forward eval: {failed_eval}",
            f"- Re-queued transient repair LLM errors: {requeued}",
            "",
        ]
    )


def _render_repair_prompt(template: str, row: dict[str, Any], *, attempt: int, family_contract: dict[str, Any]) -> str:
    source_fact_digest = _source_fact_digest(row)
    payload = {
        "repair_attempt": attempt,
        "bt_id": row.get("bt_id"),
        "original_fm_id": row.get("fm_id"),
        "instance_id": row.get("instance_id"),
        "generator_id": row.get("generator_id"),
        "problem_statement": row.get("problem_statement"),
        "structured_problem_data": row.get("structured_problem_data") or {},
        "source_compact_data": _repair_prompt_source_compact_data(row, source_fact_digest=source_fact_digest),
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
        "reference_answer": row.get("reference_answer") or {},
        "previous_model": {
            "modeling_explanation": _truncate((row.get("generated_answer") or {}).get("modeling_explanation")),
            "math_model": _truncate((row.get("generated_answer") or {}).get("math_model")),
            "gurobipy_code": _truncate((row.get("generated_answer") or {}).get("gurobipy_code")),
            "self_check": (row.get("generated_answer") or {}).get("self_check") or {},
        },
        "validation_failure": {
            "rejection_reason": row.get("rejection_reason"),
            "fine_rejection_reason": row.get("fine_rejection_reason"),
            "failure_family": row.get("failure_family"),
            "diagnostic_evidence": row.get("diagnostic_evidence") or {},
            "objective_diagnostics": _objective_diagnostics(row, family_contract=family_contract),
            "solver_result": _compact_solver_result((row.get("forward_eval") or {}).get("solver_result") or {}),
            "correctness": (row.get("forward_eval") or {}).get("correctness") or {},
            "code_preparation": (row.get("forward_eval") or {}).get("code_preparation") or row.get("code_preparation") or {},
        },
    }
    return template.replace("{{REPAIR_JSON}}", json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _repair_prompt_source_compact_data(row: dict[str, Any], *, source_fact_digest: list[str]) -> dict[str, Any]:
    """Avoid duplicating large source tables in repair prompts when a digest exists.

    The original source_compact_data remains attached to artifacts and repaired
    outputs. This helper only shrinks the LLM repair prompt payload, where the
    source_fact_digest is the authoritative compact fact card.
    """
    source_compact = row.get("source_compact_data") or {}
    if not source_fact_digest or not isinstance(source_compact, dict) or not source_compact.get("available"):
        return source_compact if isinstance(source_compact, dict) else {}

    value = source_compact.get("value")
    value_keys = sorted(str(key) for key in value.keys()) if isinstance(value, dict) else []
    return {
        "available": True,
        "truncated_for_repair_prompt": True,
        "reason": (
            "source_fact_digest carries the authoritative compact facts for the repair prompt; "
            "the full source_compact_data is preserved in stage artifacts."
        ),
        "value_keys": value_keys,
    }


def _repair_rejection_record(
    original: dict[str, Any],
    *,
    repair_fm_id: str,
    attempt: int,
    reason: str,
    family_contract: dict[str, Any],
    llm_metadata: dict[str, Any],
    payload: dict[str, Any],
    code: str,
    code_preparation: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_repair_base(original, repair_fm_id=repair_fm_id, attempt=attempt),
        "source_metadata": _source_metadata(original),
        "source_lp_text": original.get("source_lp_text"),
        "source_math_formula": original.get("source_math_formula"),
        "source_compact_data": original.get("source_compact_data") or {},
        "structured_problem_data": original.get("structured_problem_data") or {},
        "reference_answer": original.get("reference_answer") or {},
        "generator_profile_version": original.get("generator_profile_version"),
        "difficulty_level": original.get("difficulty_level"),
        "sub_family": original.get("sub_family"),
        "canonical_math_signature": original.get("canonical_math_signature") or {},
        "concept_tags": list(original.get("concept_tags") or []),
        "variant_id": original.get("variant_id"),
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
        "code_preparation": code_preparation,
        "fm_status": "REJECTED",
        "rejection_stage": "forward_repair",
        "rejection_reason": reason,
    }


def _repair_base(original: dict[str, Any], *, repair_fm_id: str, attempt: int) -> dict[str, Any]:
    return {
        "fm_id": repair_fm_id,
        "original_fm_id": original.get("fm_id"),
        "bt_id": original.get("bt_id"),
        "instance_id": original.get("instance_id"),
        "generator_id": original.get("generator_id"),
        "repair_attempt": attempt,
        "original_rejection_stage": original.get("rejection_stage"),
        "original_rejection_reason": original.get("rejection_reason"),
        "original_fine_rejection_reason": original.get("fine_rejection_reason"),
        "original_failure_family": original.get("failure_family"),
    }


def _compact_solver_result(solver_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_status": solver_result.get("execution_status"),
        "status": solver_result.get("status"),
        "objective_value": solver_result.get("objective_value"),
        "runtime_sec": solver_result.get("runtime_sec"),
        "stdout": _truncate(solver_result.get("stdout"), _MAX_LOG_CHARS),
        "stderr": _truncate(solver_result.get("stderr"), _MAX_LOG_CHARS),
        "solution_summary": (solver_result.get("solution") or {}).get("solution_summary") or {},
    }


def _objective_diagnostics(row: dict[str, Any], *, family_contract: dict[str, Any]) -> dict[str, Any]:
    forward_eval = row.get("forward_eval") if isinstance(row.get("forward_eval"), dict) else {}
    solver_result = forward_eval.get("solver_result") if isinstance(forward_eval.get("solver_result"), dict) else {}
    correctness = forward_eval.get("correctness") if isinstance(forward_eval.get("correctness"), dict) else {}
    reference = row.get("reference_answer") if isinstance(row.get("reference_answer"), dict) else {}
    signature = row.get("canonical_math_signature") if isinstance(row.get("canonical_math_signature"), dict) else {}
    answer_contract = family_contract.get("answer_contract") if isinstance(family_contract.get("answer_contract"), dict) else {}
    sense_options = answer_contract.get("objective_sense_options") if isinstance(answer_contract, dict) else []
    return {
        "reference_objective_value": reference.get("objective_value"),
        "generated_objective_value": solver_result.get("objective_value"),
        "abs_error": correctness.get("abs_error"),
        "rel_error": correctness.get("rel_error"),
        "is_correct": correctness.get("is_correct"),
        "solver_status": solver_result.get("status"),
        "objective_sense": signature.get("objective_sense"),
        "family_objective_sense_options": list(sense_options or []),
    }


def _truncate(value: Any, limit: int = _MAX_TEXT_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _mock_repair_payload(row: dict[str, Any]) -> dict[str, Any]:
    previous = row.get("generated_answer") or {}
    return {
        "modeling_explanation": previous.get("modeling_explanation") or "Mock repair keeps the original forward model structure.",
        "math_model": previous.get("math_model") or "Mock repaired model.",
        "gurobipy_code": previous.get("gurobipy_code") or "",
        "self_check": {"repair_actions": ["mock_repair"], "uses_external_files": False},
    }
