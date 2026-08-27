from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from or_cpt_engine.llm.openai_compatible_client import AsyncOpenAICompatibleClient, extract_json_object
from or_cpt_engine.schemas.common import EngineConfig
from or_cpt_engine.utils.io import read_jsonl, write_jsonl, write_text
from or_cpt_engine.utils.prompt_registry import PromptRegistry, load_default_prompt_registry

OPTIMIZER_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompt_optimizer_prompt.md"

BACKTRANSLATION_FILES = (
    "05_backtranslation/backtranslation_rejected.jsonl",
    "06_nl_quality_filter/nl_rejected.jsonl",
)
FORWARD_MODELING_FILES = (
    "07_forward_modeling/forward_modeling_rejected.jsonl",
    "08_forward_eval/rejected_pairs.jsonl",
)


def analyze_run_for_prompt_optimization(
    run_dir: str | Path,
    *,
    sample_limit: int = 5,
) -> dict[str, Any]:
    base = Path(run_dir)
    bt_rows = _load_rows(base, BACKTRANSLATION_FILES)
    fm_rows = _load_rows(base, FORWARD_MODELING_FILES)
    accepted_rows = read_jsonl(base / "08_forward_eval" / "accepted_pairs.jsonl")

    bt_reason_counts = _reason_counts(bt_rows)
    fm_reason_counts = _reason_counts(fm_rows)
    target_prompt, routing_reason = _route_target_prompt(bt_rows, fm_rows, bt_reason_counts, fm_reason_counts)
    target_rows = fm_rows if target_prompt == "forward_modeling_prompt" else bt_rows
    triggered = target_prompt is not None

    return {
        "run_dir": str(base),
        "triggered": triggered,
        "target_prompt": target_prompt,
        "routing_reason": routing_reason,
        "accepted_pairs": len(accepted_rows),
        "backtranslation_issue_count": len(bt_rows),
        "forward_modeling_issue_count": len(fm_rows),
        "backtranslation_reason_counts": dict(bt_reason_counts),
        "forward_modeling_reason_counts": dict(fm_reason_counts),
        "sample_rejections": [_compact_rejection(row) for row in target_rows[:sample_limit]],
    }


def optimize_prompts_for_run(
    run_dir: str | Path,
    config: EngineConfig,
    *,
    auto_promote: bool = False,
    force: bool = False,
    mock: bool = False,
    sample_limit: int = 5,
    concurrency_per_endpoint: int | None = None,
    registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        optimize_prompts_for_run_async(
            run_dir,
            config,
            auto_promote=auto_promote,
            force=force,
            mock=mock,
            sample_limit=sample_limit,
            concurrency_per_endpoint=concurrency_per_endpoint,
            registry=registry,
        )
    )


async def optimize_prompts_for_run_async(
    run_dir: str | Path,
    config: EngineConfig,
    *,
    auto_promote: bool = False,
    force: bool = False,
    mock: bool = False,
    sample_limit: int = 5,
    concurrency_per_endpoint: int | None = None,
    registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    base = Path(run_dir)
    report_root = base / "reports" / "prompt_optimization"
    report_root.mkdir(parents=True, exist_ok=True)
    cycle_index = _next_cycle_index(report_root)
    cycle_id = f"cycle_{cycle_index:04d}"
    cycle_dir = report_root / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)

    prompt_registry = registry or load_default_prompt_registry()
    diagnostics = analyze_run_for_prompt_optimization(base, sample_limit=sample_limit)
    if not diagnostics["triggered"] and not force:
        result = {
            "cycle_id": cycle_id,
            "status": "skipped",
            "reason": "No prompt-owned failure pattern was found in this run.",
            "diagnostics": diagnostics,
        }
        _write_cycle_outputs(cycle_dir, result)
        return result

    target_prompt = diagnostics["target_prompt"] or "forward_modeling_prompt"
    current_prompt = prompt_registry.get(target_prompt)
    if mock:
        candidate = _mock_candidate(target_prompt, current_prompt.text, diagnostics)
    else:
        client = AsyncOpenAICompatibleClient(
            config.llm,
            config.llm.forward_modeling,
            concurrency_per_endpoint=concurrency_per_endpoint,
        )
        try:
            response = await client.chat(_render_optimizer_prompt(current_prompt.text, diagnostics))
            payload = extract_json_object(response["content"])
            candidate = {
                "target_prompt": str(payload.get("target_prompt") or target_prompt),
                "revised_prompt": str(payload.get("revised_prompt") or "").strip(),
                "change_summary": str(payload.get("change_summary") or "").strip(),
                "risk_notes": str(payload.get("risk_notes") or "").strip(),
                "llm_metadata": {
                    "request_id": response["request_id"],
                    "latency_sec": response["latency_sec"],
                    "model": response["model"],
                    "endpoint": response.get("endpoint"),
                    "attempt": response["attempt"],
                },
            }
        finally:
            await client.close()

    revised_prompt = str(candidate.get("revised_prompt") or "").strip()
    if not revised_prompt:
        result = {
            "cycle_id": cycle_id,
            "status": "failed",
            "reason": "Optimizer did not return a non-empty revised_prompt.",
            "target_prompt": target_prompt,
            "diagnostics": diagnostics,
            "candidate": candidate,
        }
        _write_cycle_outputs(cycle_dir, result)
        return result

    candidate_path = cycle_dir / f"{target_prompt}_candidate.md"
    candidate_path.write_text(revised_prompt, encoding="utf-8")
    result = {
        "cycle_id": cycle_id,
        "status": "candidate_generated",
        "target_prompt": target_prompt,
        "previous_version": current_prompt.version,
        "previous_hash": current_prompt.hash,
        "candidate_path": str(candidate_path),
        "auto_promote": auto_promote,
        "diagnostics": diagnostics,
        "candidate": candidate,
    }

    if auto_promote:
        previous_spec, new_spec = prompt_registry.promote(
            target_prompt,
            text=revised_prompt,
            notes=f"Auto-promoted by OR-CPT prompt optimizer {cycle_id}: {candidate.get('change_summary')}",
        )
        result.update(
            {
                "status": "promoted",
                "previous_version": previous_spec.version,
                "previous_hash": previous_spec.hash,
                "new_version": new_spec.version,
                "new_hash": new_spec.hash,
            }
        )

    _write_cycle_outputs(cycle_dir, result)
    return result


def _load_rows(run_dir: Path, relative_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        rows.extend(read_jsonl(run_dir / relative_path))
    return rows


def _reason_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("rejection_reason") or row.get("reason") or "UNKNOWN")
        counter[reason.split(":")[0]] += 1
    return counter


def _route_target_prompt(
    bt_rows: list[dict[str, Any]],
    fm_rows: list[dict[str, Any]],
    bt_reason_counts: Counter[str],
    fm_reason_counts: Counter[str],
) -> tuple[str | None, str]:
    if not bt_rows and not fm_rows:
        return None, "No rejected rows were found."

    if _has_reason(fm_reason_counts, ("OBJECTIVE_MISMATCH", "FORWARD_CODE_EXECUTION_FAILED", "FORWARD_SOLVER_NOT_OPTIMAL")):
        return "forward_modeling_prompt", "Forward eval failures indicate the model did not reconstruct the verified formulation."

    if _has_reason(fm_reason_counts, ("ValueError", "missing executable gurobipy code")) or len(fm_rows) >= len(bt_rows):
        return "forward_modeling_prompt", "Forward modeling/parsing failures dominate this run."

    return "backtranslation_prompt", "Backtranslation or NL-quality failures dominate this run."


def _has_reason(counter: Counter[str], needles: tuple[str, ...]) -> bool:
    lowered_keys = {key.lower() for key in counter}
    return any(needle.lower() in lowered_keys for needle in needles)


def _compact_rejection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": row.get("instance_id"),
        "generator_id": row.get("generator_id"),
        "rejection_stage": row.get("rejection_stage"),
        "rejection_reason": row.get("rejection_reason") or row.get("reason"),
        "problem_statement": _truncate(str(row.get("problem_statement") or ""), 900),
        "raw_response": _truncate(str(row.get("raw_response") or ""), 900),
    }


def _render_optimizer_prompt(current_prompt: str, diagnostics: dict[str, Any]) -> str:
    import json

    template = OPTIMIZER_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{{RUN_DIAGNOSTICS_JSON}}", json.dumps(diagnostics, ensure_ascii=False, sort_keys=True, indent=2))
        .replace("{{CURRENT_PROMPT}}", current_prompt)
    )


def _mock_candidate(target_prompt: str, current_prompt: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    addition = (
        "\n\nAdditional optimizer guidance:\n"
        "- Re-check every coefficient and bound against the structured source before finalizing.\n"
        "- If a field is ambiguous, preserve the exact source value rather than inventing a simplified version.\n"
    )
    return {
        "target_prompt": target_prompt,
        "revised_prompt": current_prompt.rstrip() + addition,
        "change_summary": f"Mock optimizer update based on {diagnostics.get('routing_reason')}",
        "risk_notes": "Mock candidate for testing only.",
        "llm_metadata": {"mock": True},
    }


def _write_cycle_outputs(cycle_dir: Path, result: dict[str, Any]) -> None:
    write_jsonl(cycle_dir / "cycle_result.jsonl", [result])
    write_text(cycle_dir / "cycle_report.md", _render_cycle_report(result))


def _render_cycle_report(result: dict[str, Any]) -> str:
    diagnostics = result.get("diagnostics") or {}
    lines = [
        f"# Prompt Optimization {result.get('cycle_id')}",
        "",
        f"- Status: {result.get('status')}",
        f"- Target prompt: {result.get('target_prompt') or diagnostics.get('target_prompt')}",
        f"- Auto promote: {result.get('auto_promote', False)}",
        f"- Routing reason: {diagnostics.get('routing_reason')}",
        f"- Accepted pairs: {diagnostics.get('accepted_pairs')}",
        f"- Backtranslation issues: {diagnostics.get('backtranslation_issue_count')}",
        f"- Forward modeling issues: {diagnostics.get('forward_modeling_issue_count')}",
        "",
        "## Reason Counts",
        "",
        "### Backtranslation / NL Filter",
        "",
        _counter_table(diagnostics.get("backtranslation_reason_counts") or {}),
        "",
        "### Forward Modeling / Forward Eval",
        "",
        _counter_table(diagnostics.get("forward_modeling_reason_counts") or {}),
        "",
    ]
    candidate = result.get("candidate") or {}
    if candidate:
        lines.extend(
            [
                "## Candidate Summary",
                "",
                str(candidate.get("change_summary") or ""),
                "",
                "## Risk Notes",
                "",
                str(candidate.get("risk_notes") or ""),
                "",
            ]
        )
    return "\n".join(lines)


def _counter_table(counter: dict[str, int]) -> str:
    if not counter:
        return "_No rows._"
    lines = ["| Reason | Count |", "| --- | ---: |"]
    for reason, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    return "\n".join(lines)


def _next_cycle_index(report_root: Path) -> int:
    existing = []
    for child in report_root.glob("cycle_*"):
        if child.is_dir():
            try:
                existing.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(existing, default=0) + 1


def _truncate(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "...[truncated]"
