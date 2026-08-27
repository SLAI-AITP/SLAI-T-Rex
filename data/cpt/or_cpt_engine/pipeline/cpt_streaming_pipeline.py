from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from or_cpt_engine.backtranslation.backtranslate import (
    _backtranslate_one,
    render_backtranslation_report,
    select_backtranslation_prompt_spec,
)
from or_cpt_engine.backtranslation.nl_quality_filter import filter_nl_candidate_row, render_nl_report
from or_cpt_engine.evaluation.forward_eval import (
    evaluate_forward_output_row,
    render_forward_eval_rejection_dashboard,
    render_forward_eval_report,
)
from or_cpt_engine.export.train_val_export import StreamingTrainOnlyExporter
from or_cpt_engine.forward_modeling.generate_forward_model import (
    PROMPT_NAME as FORWARD_MODELING_PROMPT_NAME,
    _forward_model_one,
    render_forward_modeling_report,
)
from or_cpt_engine.forward_modeling.repair_forward_model import (
    PROMPT_NAME as FORWARD_REPAIR_PROMPT_NAME,
    render_forward_repair_report,
    repair_forward_model_one,
    should_attempt_forward_repair,
)
from or_cpt_engine.llm.openai_compatible_client import AsyncOpenAICompatibleClient
from or_cpt_engine.rendering.cpt_renderer import render_cpt_document_row, render_cpt_report
from or_cpt_engine.schemas.common import EngineConfig
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text
from or_cpt_engine.utils.prompt_registry import load_default_prompt_registry


_SENTINEL = object()


async def run_cpt_streaming_pipeline_async(
    *,
    seed_input: str | Path,
    stage_dirs: dict[str, Path],
    config: EngineConfig,
    limit: int | None = None,
    mock_llm: bool = False,
    llm_concurrency: int | None = None,
    forward_eval_concurrency: int = 1,
    solver_threads_per_process: int = 1,
    queue_size: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, Any]]:
    active_logger = logger or logging.getLogger(__name__)
    for key in ("backtranslation", "nl_filter", "forward_modeling", "forward_eval", "rendering", "export"):
        stage_dirs[key].mkdir(parents=True, exist_ok=True)
    (stage_dirs["forward_modeling"] / "extracted_code").mkdir(parents=True, exist_ok=True)

    prompt_registry = load_default_prompt_registry()
    forward_modeling_prompt = prompt_registry.get(FORWARD_MODELING_PROMPT_NAME)
    forward_repair_prompt = prompt_registry.get(FORWARD_REPAIR_PROMPT_NAME)

    bt_client, fm_client, clients, bt_capacity, fm_capacity = _create_llm_clients(
        config,
        mock_llm=mock_llm,
        llm_concurrency=llm_concurrency,
        logger=active_logger,
    )
    total_llm_capacity = max(1, bt_capacity + (0 if bt_client is fm_client else fm_capacity))
    resolved_queue_size = max(1, int(queue_size or max(1024, total_llm_capacity * 8)))
    bt_worker_count = 1 if mock_llm else max(1, bt_capacity)
    fm_worker_count = 1 if mock_llm else max(1, fm_capacity)
    nl_worker_count = max(1, min(4, total_llm_capacity))
    eval_worker_count = max(1, int(forward_eval_concurrency))

    active_logger.info(
        "streaming_pipeline status=started queue_size=%s bt_workers=%s nl_workers=%s fm_workers=%s "
        "forward_eval_workers=%s solver_threads_per_process=%s shared_llm_pool=%s",
        resolved_queue_size,
        bt_worker_count,
        nl_worker_count,
        fm_worker_count,
        eval_worker_count,
        solver_threads_per_process,
        bt_client is not None and bt_client is fm_client,
    )

    bt_queue: asyncio.Queue = asyncio.Queue(maxsize=resolved_queue_size)
    nl_queue: asyncio.Queue = asyncio.Queue(maxsize=resolved_queue_size)
    fm_queue: asyncio.Queue = asyncio.Queue(maxsize=resolved_queue_size)
    eval_queue: asyncio.Queue = asyncio.Queue(maxsize=resolved_queue_size)
    render_queue: asyncio.Queue = asyncio.Queue(maxsize=resolved_queue_size)
    write_lock = asyncio.Lock()

    paths = _artifact_paths(stage_dirs)
    counters: dict[str, Counter[str]] = {
        "backtranslation": Counter(),
        "nl_filter": Counter(),
        "forward_modeling": Counter(),
        "forward_eval": Counter(),
        "forward_repair": Counter(),
        "rendering": Counter(),
        "export": Counter(),
    }
    nl_rejection_reasons: Counter[str] = Counter()
    eval_rejection_reasons: Counter[str] = Counter()
    render_rejection_reasons: Counter[str] = Counter()
    render_type_counts: Counter[str] = Counter()
    render_views_by_instance: dict[str, int] = {}
    render_index = 0

    bt_thresholds = config.quality_thresholds.backtranslation
    eval_thresholds = config.quality_thresholds.forward_modeling_eval
    repair_thresholds = config.quality_thresholds.forward_repair
    solve_timeout = int(config.generation_plan.filters.get("solve_timeout_sec", 60))
    max_views = max(1, int(config.rendering.max_views_per_seed_instance or 1))
    train_exporter = StreamingTrainOnlyExporter(stage_dirs["export"], exact_dedup=True)
    active_logger.info("stage=10_train_export status=started mode=train_only_streaming output=%s", train_exporter.train_path)

    async def backtranslation_worker(worker_id: int) -> None:
        while True:
            item = await bt_queue.get()
            try:
                if item is _SENTINEL:
                    return
                row, candidate_index, requeue_count = item
                active_logger.info(
                    "pipeline_progress stage=05_backtranslation status=started worker=%s instance_id=%s candidate_index=%s queue_depth=%s",
                    worker_id,
                    row.get("instance_id"),
                    candidate_index,
                    bt_queue.qsize(),
                )
                while True:
                    prompt_name, backtranslation_prompt = select_backtranslation_prompt_spec(prompt_registry, row)
                    result = await _backtranslate_one(
                        row,
                        candidate_index,
                        backtranslation_prompt,
                        config,
                        client=bt_client,
                        mock=mock_llm,
                        stage_config=config.llm.backtranslation,
                        prompt_name=prompt_name,
                    )
                    if result["kind"] != "requeue":
                        break
                    requeue_count += 1
                    counters["backtranslation"]["requeued"] += 1
                    active_logger.info(
                        "pipeline_progress stage=05_backtranslation status=requeued worker=%s instance_id=%s candidate_index=%s requeue_count=%s",
                        worker_id,
                        row.get("instance_id"),
                        candidate_index,
                        requeue_count,
                    )
                    await asyncio.sleep(1.0)
                if result["kind"] == "candidate":
                    candidate = _as_dict(result["candidate"])
                    async with write_lock:
                        append_jsonl(paths["bt_candidates"], result["candidate"])
                        append_jsonl(paths["bt_raw"], result["raw_response"])
                    counters["backtranslation"]["candidates"] += 1
                    await nl_queue.put(candidate)
                    active_logger.info(
                        "pipeline_progress stage=05_backtranslation status=completed worker=%s instance_id=%s bt_id=%s",
                        worker_id,
                        candidate.get("instance_id"),
                        candidate.get("bt_id"),
                    )
                else:
                    async with write_lock:
                        if result.get("raw_response") is not None:
                            append_jsonl(paths["bt_raw"], result["raw_response"])
                        append_jsonl(paths["bt_rejected"], result["rejected"])
                    counters["backtranslation"]["rejected"] += 1
                    active_logger.info(
                        "pipeline_progress stage=05_backtranslation status=rejected worker=%s instance_id=%s reason=%s",
                        worker_id,
                        result["rejected"].get("instance_id"),
                        result["rejected"].get("rejection_reason"),
                    )
            finally:
                bt_queue.task_done()

    async def nl_worker(worker_id: int) -> None:
        while True:
            row = await nl_queue.get()
            try:
                if row is _SENTINEL:
                    return
                accepted, output_row, reasons = filter_nl_candidate_row(
                    row,
                    min_chars=int(bt_thresholds.get("min_chars", 160)),
                    max_chars=int(bt_thresholds.get("max_chars", 4000)),
                    min_number_coverage=float(bt_thresholds.get("min_number_coverage", 0.4)),
                )
                if accepted:
                    async with write_lock:
                        append_jsonl(paths["nl_accepted"], output_row)
                    counters["nl_filter"]["accepted"] += 1
                    await fm_queue.put(output_row)
                    active_logger.info(
                        "pipeline_progress stage=06_nl_quality_filter status=accepted worker=%s bt_id=%s instance_id=%s",
                        worker_id,
                        output_row.get("bt_id"),
                        output_row.get("instance_id"),
                    )
                else:
                    reason_key = reasons[0].split(":")[0] if reasons else "UNKNOWN"
                    nl_rejection_reasons[reason_key] += 1
                    async with write_lock:
                        append_jsonl(paths["nl_rejected"], output_row)
                    counters["nl_filter"]["rejected"] += 1
                    active_logger.info(
                        "pipeline_progress stage=06_nl_quality_filter status=rejected worker=%s bt_id=%s reason=%s",
                        worker_id,
                        output_row.get("bt_id"),
                        output_row.get("rejection_reason"),
                    )
            finally:
                nl_queue.task_done()

    async def forward_modeling_worker(worker_id: int) -> None:
        while True:
            item = await fm_queue.get()
            try:
                if item is _SENTINEL:
                    return
                row = item
                active_logger.info(
                    "pipeline_progress stage=07_forward_modeling status=started worker=%s bt_id=%s instance_id=%s queue_depth=%s",
                    worker_id,
                    row.get("bt_id"),
                    row.get("instance_id"),
                    fm_queue.qsize(),
                )
                requeue_count = 0
                while True:
                    result = await _forward_model_one(
                        row,
                        forward_modeling_prompt,
                        stage_dirs["forward_modeling"] / "extracted_code",
                        client=fm_client,
                        mock=mock_llm,
                        stage_config=config.llm.forward_modeling,
                        family_contracts=config.family_contracts,
                        config=config,
                    )
                    if result["kind"] != "requeue":
                        break
                    requeue_count += 1
                    counters["forward_modeling"]["requeued"] += 1
                    active_logger.info(
                        "pipeline_progress stage=07_forward_modeling status=requeued worker=%s bt_id=%s instance_id=%s requeue_count=%s",
                        worker_id,
                        row.get("bt_id"),
                        row.get("instance_id"),
                        requeue_count,
                    )
                    await asyncio.sleep(1.0)
                if result["kind"] == "output":
                    output_row = _as_dict(result["output"])
                    async with write_lock:
                        append_jsonl(paths["fm_outputs"], result["output"])
                        append_jsonl(paths["fm_raw"], result["raw_response"])
                    counters["forward_modeling"]["outputs"] += 1
                    await eval_queue.put(output_row)
                    active_logger.info(
                        "pipeline_progress stage=07_forward_modeling status=completed worker=%s fm_id=%s instance_id=%s",
                        worker_id,
                        output_row.get("fm_id"),
                        output_row.get("instance_id"),
                    )
                else:
                    async with write_lock:
                        append_jsonl(paths["fm_rejected"], result["rejected"])
                    counters["forward_modeling"]["rejected"] += 1
                    active_logger.info(
                        "pipeline_progress stage=07_forward_modeling status=rejected worker=%s bt_id=%s reason=%s",
                        worker_id,
                        result["rejected"].get("bt_id"),
                        result["rejected"].get("rejection_reason"),
                    )
            finally:
                fm_queue.task_done()

    async def forward_eval_worker(worker_id: int) -> None:
        while True:
            item = await eval_queue.get()
            try:
                if item is _SENTINEL:
                    return
                row = item
                accepted, record = await asyncio.to_thread(
                    evaluate_forward_output_row,
                    row,
                    timeout_seconds=solve_timeout,
                    threads_per_process=max(1, int(solver_threads_per_process)),
                    abs_tolerance=float(eval_thresholds.get("objective_abs_tolerance", 1e-4)),
                    rel_tolerance=float(eval_thresholds.get("objective_rel_tolerance", 1e-4)),
                )
                if accepted:
                    accepted_row = _as_dict(record)
                    async with write_lock:
                        append_jsonl(paths["eval_accepted"], record)
                    counters["forward_eval"]["accepted"] += 1
                    await render_queue.put(accepted_row)
                    active_logger.info(
                        "pipeline_progress stage=08_forward_eval status=accepted worker=%s fm_id=%s instance_id=%s",
                        worker_id,
                        accepted_row.get("fm_id"),
                        accepted_row.get("instance_id"),
                    )
                else:
                    final_rejection = record
                    if isinstance(record, dict) and should_attempt_forward_repair(record, repair_thresholds):
                        counters["forward_repair"]["attempted"] += 1
                        active_logger.info(
                            "pipeline_progress stage=08_forward_repair status=started worker=%s fm_id=%s instance_id=%s reason=%s",
                            worker_id,
                            record.get("fm_id"),
                            record.get("instance_id"),
                            record.get("fine_rejection_reason") or record.get("rejection_reason"),
                        )
                        repair_requeue_count = 0
                        while True:
                            repair_result = await repair_forward_model_one(
                                record,
                                forward_repair_prompt,
                                stage_dirs["forward_modeling"] / "extracted_code",
                                client=fm_client,
                                mock=mock_llm,
                                stage_config=config.llm.forward_modeling,
                                family_contracts=config.family_contracts,
                            )
                            if repair_result["kind"] != "requeue":
                                break
                            repair_requeue_count += 1
                            counters["forward_repair"]["requeued"] += 1
                            active_logger.info(
                                "pipeline_progress stage=08_forward_repair status=requeued worker=%s fm_id=%s requeue_count=%s",
                                worker_id,
                                record.get("fm_id"),
                                repair_requeue_count,
                            )
                            await asyncio.sleep(1.0)
                        if repair_result["kind"] == "output":
                            repaired_output = _as_dict(repair_result["output"])
                            async with write_lock:
                                append_jsonl(paths["repair_outputs"], repair_result["output"])
                                append_jsonl(paths["repair_raw"], repair_result["raw_response"])
                            counters["forward_repair"]["generated"] += 1
                            repaired_accepted, repaired_record = await asyncio.to_thread(
                                evaluate_forward_output_row,
                                repaired_output,
                                timeout_seconds=solve_timeout,
                                threads_per_process=max(1, int(solver_threads_per_process)),
                                abs_tolerance=float(eval_thresholds.get("objective_abs_tolerance", 1e-4)),
                                rel_tolerance=float(eval_thresholds.get("objective_rel_tolerance", 1e-4)),
                            )
                            if repaired_accepted:
                                accepted_row = _as_dict(repaired_record)
                                async with write_lock:
                                    append_jsonl(paths["eval_repair_attempts"], _repair_attempt_summary(record, "recovered", repaired_output, accepted_row))
                                    append_jsonl(paths["eval_accepted"], repaired_record)
                                counters["forward_eval"]["accepted"] += 1
                                counters["forward_repair"]["recovered"] += 1
                                await render_queue.put(accepted_row)
                                active_logger.info(
                                    "pipeline_progress stage=08_forward_repair status=recovered worker=%s original_fm_id=%s repaired_fm_id=%s instance_id=%s",
                                    worker_id,
                                    record.get("fm_id"),
                                    accepted_row.get("fm_id"),
                                    accepted_row.get("instance_id"),
                                )
                                continue
                            final_rejection = _attach_original_rejection(_as_dict(repaired_record), record)
                            counters["forward_repair"]["failed_eval"] += 1
                            async with write_lock:
                                append_jsonl(paths["eval_repair_attempts"], _repair_attempt_summary(record, "failed_eval", repaired_output, final_rejection))
                            active_logger.info(
                                "pipeline_progress stage=08_forward_repair status=failed_eval worker=%s original_fm_id=%s repaired_fm_id=%s reason=%s",
                                worker_id,
                                record.get("fm_id"),
                                repaired_output.get("fm_id"),
                                final_rejection.get("rejection_reason"),
                            )
                        else:
                            repair_rejection = _as_dict(repair_result["rejected"])
                            counters["forward_repair"]["failed_generation"] += 1
                            final_rejection = _attach_repair_generation_failure(record, repair_rejection)
                            async with write_lock:
                                append_jsonl(paths["repair_rejected"], repair_rejection)
                                append_jsonl(paths["eval_repair_attempts"], _repair_attempt_summary(record, "failed_generation", repair_rejection, final_rejection))
                            active_logger.info(
                                "pipeline_progress stage=08_forward_repair status=failed_generation worker=%s original_fm_id=%s reason=%s",
                                worker_id,
                                record.get("fm_id"),
                                repair_rejection.get("rejection_reason"),
                            )

                    if isinstance(final_rejection, dict):
                        eval_rejection_reasons[str(final_rejection.get("fine_rejection_reason") or final_rejection.get("rejection_reason") or "unknown")] += 1
                    async with write_lock:
                        append_jsonl(paths["eval_rejected"], final_rejection)
                    counters["forward_eval"]["rejected"] += 1
                    active_logger.info(
                        "pipeline_progress stage=08_forward_eval status=rejected worker=%s fm_id=%s reason=%s",
                        worker_id,
                        final_rejection.get("fm_id") if isinstance(final_rejection, dict) else row.get("fm_id"),
                        final_rejection.get("rejection_reason") if isinstance(final_rejection, dict) else None,
                    )
            finally:
                eval_queue.task_done()

    async def rendering_worker() -> None:
        nonlocal render_index
        while True:
            item = await render_queue.get()
            try:
                if item is _SENTINEL:
                    return
                row = item
                accepted, record = render_cpt_document_row(
                    row,
                    config,
                    index=render_index,
                    views_by_instance=render_views_by_instance,
                    max_views=max_views,
                )
                render_index += 1
                if accepted:
                    doc = _as_dict(record)
                    async with write_lock:
                        append_jsonl(paths["render_docs"], record)
                        exported = train_exporter.append_document(doc)
                    counters["rendering"]["documents"] += 1
                    counters["export"]["train" if exported else "duplicates_removed"] += 1
                    render_type_counts[doc.get("doc_type", "unknown")] += 1
                    active_logger.info(
                        "pipeline_progress stage=09_cpt_rendering status=document doc_id=%s instance_id=%s doc_type=%s exported_to_train=%s",
                        doc.get("doc_id"),
                        doc.get("instance_id"),
                        doc.get("doc_type"),
                        exported,
                    )
                else:
                    if isinstance(record, dict):
                        for reason in _split_rejection_reason(record.get("rejection_reason")):
                            render_rejection_reasons[reason] += 1
                    async with write_lock:
                        append_jsonl(paths["render_rejected"], record)
                    counters["rendering"]["rejected"] += 1
                    active_logger.info(
                        "pipeline_progress stage=09_cpt_rendering status=rejected instance_id=%s reason=%s",
                        record.get("instance_id") if isinstance(record, dict) else None,
                        record.get("rejection_reason") if isinstance(record, dict) else None,
                    )
            finally:
                render_queue.task_done()

    started = time.perf_counter()
    workers = {
        "backtranslation": [asyncio.create_task(backtranslation_worker(index + 1)) for index in range(bt_worker_count)],
        "nl_filter": [asyncio.create_task(nl_worker(index + 1)) for index in range(nl_worker_count)],
        "forward_modeling": [asyncio.create_task(forward_modeling_worker(index + 1)) for index in range(fm_worker_count)],
        "forward_eval": [asyncio.create_task(forward_eval_worker(index + 1)) for index in range(eval_worker_count)],
        "rendering": [asyncio.create_task(rendering_worker())],
    }

    try:
        await _produce_backtranslation_work(seed_input, bt_queue, config, limit=limit, logger=active_logger)
        await _drain_and_stop(bt_queue, workers["backtranslation"])
        await _drain_and_stop(nl_queue, workers["nl_filter"])
        await _drain_and_stop(fm_queue, workers["forward_modeling"])
        await _drain_and_stop(eval_queue, workers["forward_eval"])
        await _drain_and_stop(render_queue, workers["rendering"])
    finally:
        for client in _unique_clients(clients):
            await client.close()

    write_text(
        stage_dirs["backtranslation"] / "backtranslation_report.md",
        render_backtranslation_report(
            counters["backtranslation"]["candidates"],
            counters["backtranslation"]["rejected"],
            counters["backtranslation"]["requeued"],
            paths["bt_candidates"],
        ),
    )
    write_text(
        stage_dirs["nl_filter"] / "nl_quality_report.md",
        render_nl_report(counters["nl_filter"]["accepted"], counters["nl_filter"]["rejected"], dict(nl_rejection_reasons)),
    )
    write_text(
        stage_dirs["forward_modeling"] / "forward_modeling_report.md",
        render_forward_modeling_report(
            counters["forward_modeling"]["outputs"],
            counters["forward_modeling"]["rejected"],
            counters["forward_modeling"]["requeued"],
        ),
    )
    write_text(
        stage_dirs["forward_eval"] / "forward_eval_report.md",
        render_forward_eval_report(
            counters["forward_eval"]["accepted"],
            counters["forward_eval"]["rejected"],
            dict(eval_rejection_reasons),
        ),
    )
    write_text(
        stage_dirs["forward_eval"] / "forward_eval_rejection_dashboard.md",
        render_forward_eval_rejection_dashboard(
            counters["forward_eval"]["accepted"],
            counters["forward_eval"]["rejected"],
            dict(eval_rejection_reasons),
        ),
    )
    write_text(
        stage_dirs["forward_eval"] / "forward_repair_report.md",
        render_forward_repair_report(
            attempted=counters["forward_repair"]["attempted"],
            generated=counters["forward_repair"]["generated"],
            recovered=counters["forward_repair"]["recovered"],
            failed_generation=counters["forward_repair"]["failed_generation"],
            failed_eval=counters["forward_repair"]["failed_eval"],
            requeued=counters["forward_repair"]["requeued"],
        ),
    )
    write_text(
        stage_dirs["rendering"] / "cpt_rendering_report.md",
        render_cpt_report(
            counters["rendering"]["documents"],
            counters["rendering"]["rejected"],
            dict(render_type_counts),
            dict(render_rejection_reasons),
        ),
    )

    export_result = train_exporter.finalize()
    active_logger.info(
        "stage=10_train_export status=completed train=%s duplicates_removed=%s",
        export_result.get("train"),
        export_result.get("duplicates_removed"),
    )

    stage_results = {
        "backtranslation": {
            "candidates": counters["backtranslation"]["candidates"],
            "rejected": counters["backtranslation"]["rejected"],
            "requeued": counters["backtranslation"]["requeued"],
        },
        "nl_filter": {
            "accepted": counters["nl_filter"]["accepted"],
            "rejected": counters["nl_filter"]["rejected"],
        },
        "forward_modeling": {
            "outputs": counters["forward_modeling"]["outputs"],
            "rejected": counters["forward_modeling"]["rejected"],
            "requeued": counters["forward_modeling"]["requeued"],
        },
        "forward_eval": {
            "accepted": counters["forward_eval"]["accepted"],
            "rejected": counters["forward_eval"]["rejected"],
            "fine_rejection_reasons": dict(eval_rejection_reasons),
        },
        "forward_repair": {
            "attempted": counters["forward_repair"]["attempted"],
            "generated": counters["forward_repair"]["generated"],
            "recovered": counters["forward_repair"]["recovered"],
            "failed_generation": counters["forward_repair"]["failed_generation"],
            "failed_eval": counters["forward_repair"]["failed_eval"],
            "requeued": counters["forward_repair"]["requeued"],
        },
        "rendering": {
            "documents": counters["rendering"]["documents"],
            "rejected": counters["rendering"]["rejected"],
        },
        "export": export_result,
    }
    elapsed = time.perf_counter() - started
    for stage_name, result in stage_results.items():
        active_logger.info("pipeline_summary stage=%s %s wall_seconds=%.3f", stage_name, _format_summary(result), elapsed)
    return stage_results


def run_cpt_streaming_pipeline(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return asyncio.run(run_cpt_streaming_pipeline_async(**kwargs))


async def _produce_backtranslation_work(
    seed_input: str | Path,
    queue: asyncio.Queue,
    config: EngineConfig,
    *,
    limit: int | None,
    logger: logging.Logger,
) -> None:
    consumed_seed_count = 0
    produced_candidate_count = 0
    for row in iter_jsonl(seed_input) or []:
        if limit is not None and consumed_seed_count >= limit:
            break
        consumed_seed_count += 1
        for candidate_index in range(1, config.llm.backtranslation.num_candidates + 1):
            await queue.put((row, candidate_index, 0))
            produced_candidate_count += 1
    logger.info(
        "pipeline_progress stage=05_backtranslation status=producer_completed seed_rows=%s candidate_work_items=%s",
        consumed_seed_count,
        produced_candidate_count,
    )


async def _drain_and_stop(queue: asyncio.Queue, workers: list[asyncio.Task]) -> None:
    await queue.join()
    for _ in workers:
        await queue.put(_SENTINEL)
    await asyncio.gather(*workers)


def _create_llm_clients(
    config: EngineConfig,
    *,
    mock_llm: bool,
    llm_concurrency: int | None,
    logger: logging.Logger,
) -> tuple[AsyncOpenAICompatibleClient | None, AsyncOpenAICompatibleClient | None, list[AsyncOpenAICompatibleClient], int, int]:
    if mock_llm:
        return None, None, [], 1, 1
    if config.llm.backtranslation.provider == config.llm.forward_modeling.provider:
        client = AsyncOpenAICompatibleClient(
            config.llm,
            config.llm.backtranslation,
            concurrency_per_endpoint=llm_concurrency,
        )
        capacity = client.total_concurrency_capacity()
        return client, client, [client], capacity, capacity
    logger.warning(
        "streaming_pipeline shared_llm_pool=disabled reason=different_providers backtranslation_provider=%s forward_modeling_provider=%s",
        config.llm.backtranslation.provider,
        config.llm.forward_modeling.provider,
    )
    bt_client = AsyncOpenAICompatibleClient(
        config.llm,
        config.llm.backtranslation,
        concurrency_per_endpoint=llm_concurrency,
    )
    fm_client = AsyncOpenAICompatibleClient(
        config.llm,
        config.llm.forward_modeling,
        concurrency_per_endpoint=llm_concurrency,
    )
    return bt_client, fm_client, [bt_client, fm_client], bt_client.total_concurrency_capacity(), fm_client.total_concurrency_capacity()


def _unique_clients(clients: list[AsyncOpenAICompatibleClient]) -> list[AsyncOpenAICompatibleClient]:
    seen: set[int] = set()
    unique: list[AsyncOpenAICompatibleClient] = []
    for client in clients:
        marker = id(client)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(client)
    return unique


def _artifact_paths(stage_dirs: dict[str, Path]) -> dict[str, Path]:
    return {
        "bt_candidates": stage_dirs["backtranslation"] / "backtranslation_candidates.jsonl",
        "bt_rejected": stage_dirs["backtranslation"] / "backtranslation_rejected.jsonl",
        "bt_raw": stage_dirs["backtranslation"] / "backtranslation_raw_responses.jsonl",
        "nl_accepted": stage_dirs["nl_filter"] / "nl_validated_candidates.jsonl",
        "nl_rejected": stage_dirs["nl_filter"] / "nl_rejected.jsonl",
        "fm_outputs": stage_dirs["forward_modeling"] / "forward_modeling_outputs.jsonl",
        "fm_rejected": stage_dirs["forward_modeling"] / "forward_modeling_rejected.jsonl",
        "fm_raw": stage_dirs["forward_modeling"] / "forward_modeling_raw_responses.jsonl",
        "repair_outputs": stage_dirs["forward_modeling"] / "forward_repair_outputs.jsonl",
        "repair_rejected": stage_dirs["forward_modeling"] / "forward_repair_rejected.jsonl",
        "repair_raw": stage_dirs["forward_modeling"] / "forward_repair_raw_responses.jsonl",
        "eval_accepted": stage_dirs["forward_eval"] / "accepted_pairs.jsonl",
        "eval_rejected": stage_dirs["forward_eval"] / "rejected_pairs.jsonl",
        "eval_repair_attempts": stage_dirs["forward_eval"] / "forward_repair_attempts.jsonl",
        "render_docs": stage_dirs["rendering"] / "cpt_documents.jsonl",
        "render_rejected": stage_dirs["rendering"] / "cpt_rendering_rejected.jsonl",
    }


def _as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    return dict(row)


def _format_summary(result: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(result.items()))


def _split_rejection_reason(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        reasons: list[str] = []
        for item in value:
            reasons.extend(_split_rejection_reason(item))
        return reasons or ["UNKNOWN"]
    text = str(value).strip()
    if not text:
        return ["UNKNOWN"]
    return [part.strip() for part in re.split(r";\s*", text) if part.strip()] or ["UNKNOWN"]


def _attach_original_rejection(repaired_record: dict[str, Any], original_record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(repaired_record)
    updated["original_forward_eval_rejection"] = _rejection_summary(original_record)
    updated.setdefault("forward_repair", {})
    if isinstance(updated["forward_repair"], dict):
        updated["forward_repair"].update(
            {
                "status": "failed_eval",
                "original_fm_id": original_record.get("fm_id"),
                "original_rejection_reason": original_record.get("rejection_reason"),
                "original_fine_rejection_reason": original_record.get("fine_rejection_reason"),
            }
        )
    return updated


def _attach_repair_generation_failure(original_record: dict[str, Any], repair_rejection: dict[str, Any]) -> dict[str, Any]:
    updated = dict(original_record)
    updated["forward_repair"] = {
        "status": "failed_generation",
        "attempt": repair_rejection.get("repair_attempt"),
        "repair_fm_id": repair_rejection.get("fm_id"),
        "repair_rejection_reason": repair_rejection.get("rejection_reason"),
    }
    return updated


def _repair_attempt_summary(
    original_record: dict[str, Any],
    status: str,
    repair_record: dict[str, Any],
    final_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "generator_id": original_record.get("generator_id"),
        "instance_id": original_record.get("instance_id"),
        "original_fm_id": original_record.get("fm_id"),
        "repair_fm_id": repair_record.get("fm_id"),
        "original_rejection_reason": original_record.get("rejection_reason"),
        "original_fine_rejection_reason": original_record.get("fine_rejection_reason"),
        "original_failure_family": original_record.get("failure_family"),
        "final_rejection_reason": None if status == "recovered" else final_record.get("rejection_reason"),
        "final_fine_rejection_reason": None if status == "recovered" else final_record.get("fine_rejection_reason"),
        "final_failure_family": None if status == "recovered" else final_record.get("failure_family"),
        "original": _rejection_summary(original_record),
        "repair": {
            "fm_id": repair_record.get("fm_id"),
            "rejection_reason": repair_record.get("rejection_reason"),
            "code_path": repair_record.get("code_path"),
        },
        "final": _rejection_summary(final_record) if status != "recovered" else {
            "pair_id": final_record.get("pair_id"),
            "fm_id": final_record.get("fm_id"),
            "instance_id": final_record.get("instance_id"),
            "generator_id": final_record.get("generator_id"),
        },
    }


def _rejection_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "fm_id": record.get("fm_id"),
        "bt_id": record.get("bt_id"),
        "instance_id": record.get("instance_id"),
        "generator_id": record.get("generator_id"),
        "rejection_stage": record.get("rejection_stage"),
        "rejection_reason": record.get("rejection_reason"),
        "fine_rejection_reason": record.get("fine_rejection_reason"),
        "failure_family": record.get("failure_family"),
        "repairable": record.get("repairable"),
    }
