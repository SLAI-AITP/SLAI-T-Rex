from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from or_cpt_engine.schemas.common import InstanceRecord, SolverValidationRecord
from or_cpt_engine.solver.gurobi_executor import execute_gurobi_code, execute_gurobi_lp
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text


def validate_solver(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: int = 60,
    concurrency: int = 1,
    threads_per_process: int = 1,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "solver_validated_instances.jsonl"
    rejected_path = output / "solver_rejected_instances.jsonl"
    accepted_count = 0
    rejected_count = 0
    worker_count = max(1, int(concurrency))
    solver_threads = max(1, int(threads_per_process))

    if worker_count <= 1:
        for row in iter_jsonl(input_path) or []:
            accepted, record = _validate_solver_row(row, timeout_seconds=timeout_seconds, threads_per_process=solver_threads)
            if accepted:
                append_jsonl(accepted_path, record)
                accepted_count += 1
            else:
                append_jsonl(rejected_path, record)
                rejected_count += 1
    else:
        # Keep only a bounded number of futures in memory; each worker starts an
        # isolated solver subprocess, while the main thread serializes JSONL writes.
        max_pending = max(worker_count * 2, worker_count)
        pending: set[Future] = set()
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="solver04") as executor:
            for row in iter_jsonl(input_path) or []:
                pending.add(executor.submit(_validate_solver_row, row, timeout_seconds=timeout_seconds, threads_per_process=solver_threads))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    accepted_delta, rejected_delta = _write_solver_results(done, accepted_path, rejected_path)
                    accepted_count += accepted_delta
                    rejected_count += rejected_delta
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                accepted_delta, rejected_delta = _write_solver_results(done, accepted_path, rejected_path)
                accepted_count += accepted_delta
                rejected_count += rejected_delta

    write_text(output / "solver_validation_report.md", render_solver_report(accepted_count, rejected_count, worker_count, solver_threads))
    return {"validated": accepted_count, "rejected": rejected_count}


def _validate_solver_row(row: dict[str, Any], *, timeout_seconds: int, threads_per_process: int = 1) -> tuple[bool, SolverValidationRecord]:
    instance = InstanceRecord.model_validate(row)
    solver_log_ctx = {"instance_id": instance.instance_id, "stage": "04"}
    if instance.gurobi_code:
        result = execute_gurobi_code(
            instance.gurobi_code,
            timeout_seconds=timeout_seconds,
            threads_per_process=threads_per_process,
            log_context=solver_log_ctx,
        )
        status = result.get("status")
        objective = result.get("objective_value")
    elif instance.lp_text:
        result = execute_gurobi_lp(
            instance.lp_text,
            timeout_seconds=timeout_seconds,
            threads_per_process=threads_per_process,
            log_context=solver_log_ctx,
        )
        status = result.get("status")
        objective = result.get("objective_value")
    elif instance.initial_objective_value is not None:
        result = {
            "status": instance.initial_solver_status or "SOURCE_REFERENCE",
            "objective_value": instance.initial_objective_value,
            "runtime_sec": None,
        }
        status = result["status"]
        objective = instance.initial_objective_value
    else:
        result = {"status": "NO_EXECUTABLE_MODEL", "objective_value": None}
        status = "NO_EXECUTABLE_MODEL"
        objective = None

    record = SolverValidationRecord(
        instance_id=instance.instance_id,
        generator_id=instance.generator_id,
        model_type=instance.model_type,
        difficulty_level=instance.difficulty_level,
        generator_profile_version=instance.generator_profile_version,
        sub_family=instance.sub_family,
        canonical_math_signature=instance.canonical_math_signature,
        concept_tags=instance.concept_tags,
        variant_id=instance.variant_id,
        generation_params=instance.generation_params,
        source_compact_data=instance.source_compact_data,
        lp_text=instance.lp_text,
        math_formula=instance.math_formula,
        gurobi_code=instance.gurobi_code,
        solver_validation=result,
        reference_answer={"objective_value": objective, "status": status},
    )
    accepted = status in {"OPTIMAL", "SOURCE_REFERENCE"} and objective is not None
    if not accepted:
        record.rejection_stage = "solver_validation"
        record.rejection_reason = "NON_OPTIMAL_STATUS"
    return accepted, record


def _write_solver_results(done: set[Future], accepted_path: Path, rejected_path: Path) -> tuple[int, int]:
    accepted_count = 0
    rejected_count = 0
    for future in done:
        accepted, record = future.result()
        if accepted:
            append_jsonl(accepted_path, record)
            accepted_count += 1
        else:
            append_jsonl(rejected_path, record)
            rejected_count += 1
    return accepted_count, rejected_count


def render_solver_report(accepted_count: int, rejected_count: int, concurrency: int = 1, threads_per_process: int = 1) -> str:
    return "\n".join(
        [
            "# Solver Validation Report",
            "",
            f"- Total input: {accepted_count + rejected_count}",
            f"- Solver validated: {accepted_count}",
            f"- Solver rejected: {rejected_count}",
            f"- Concurrency: {concurrency}",
            f"- Threads per solver process: {threads_per_process}",
            "",
        ]
    )
