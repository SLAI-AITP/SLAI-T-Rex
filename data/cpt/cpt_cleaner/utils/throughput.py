from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ThroughputBucket:
    api_attempts: int = 0
    successful_api_attempts: int = 0
    attempts_with_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ThroughputSummary:
    log_path: str
    wall_seconds: float
    npu_count: int
    overall: ThroughputBucket
    by_prompt: dict[str, ThroughputBucket] = field(default_factory=dict)
    by_run_context: dict[str, ThroughputBucket] = field(default_factory=dict)


def write_throughput_summary(
    report_dir: str | Path,
    log_path: str | Path,
    *,
    wall_seconds: float,
    npu_count: int,
) -> ThroughputSummary:
    summary = build_throughput_summary(log_path, wall_seconds=wall_seconds, npu_count=npu_count)
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    markdown_path = report_root / "throughput_summary.md"
    json_path = report_root / "throughput_summary.json"

    markdown_path.write_text(_markdown_summary(summary), encoding="utf-8")
    json_path.write_text(_json_summary(summary), encoding="utf-8")
    return summary


def build_throughput_summary(
    log_path: str | Path,
    *,
    wall_seconds: float,
    npu_count: int,
) -> ThroughputSummary:
    overall = ThroughputBucket()
    by_prompt: dict[str, ThroughputBucket] = defaultdict(ThroughputBucket)
    by_run_context: dict[str, ThroughputBucket] = defaultdict(ThroughputBucket)
    file_path = Path(log_path)

    if file_path.exists():
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if "model_api_call " not in line:
                continue
            prompt_name = _extract_field(line, "prompt_name") or "unknown"
            run_context = _extract_field(line, "run_context") or "unknown"
            status = _extract_field(line, "status") or "unknown"
            prompt_tokens = _extract_int_field(line, "prompt_tokens")
            completion_tokens = _extract_int_field(line, "completion_tokens")
            total_tokens = _extract_int_field(line, "total_tokens")

            _accumulate_bucket(
                overall,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            _accumulate_bucket(
                by_prompt[prompt_name],
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            _accumulate_bucket(
                by_run_context[run_context],
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    return ThroughputSummary(
        log_path=str(file_path.resolve()),
        wall_seconds=wall_seconds,
        npu_count=max(1, npu_count),
        overall=overall,
        by_prompt=dict(sorted(by_prompt.items())),
        by_run_context=dict(sorted(by_run_context.items())),
    )


def _accumulate_bucket(
    bucket: ThroughputBucket,
    *,
    status: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> None:
    bucket.api_attempts += 1
    if _is_success_status(status):
        bucket.successful_api_attempts += 1
    if total_tokens is None:
        return
    bucket.attempts_with_usage += 1
    bucket.prompt_tokens += prompt_tokens or 0
    bucket.completion_tokens += completion_tokens or 0
    bucket.total_tokens += total_tokens


def _extract_field(line: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    if not match:
        return None
    value = match.group(1)
    if value == "n/a":
        return None
    return value


def _is_success_status(status: str) -> bool:
    return status.lower() in {"success", "ok"}


def _extract_int_field(line: str, key: str) -> int | None:
    value = _extract_field(line, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _markdown_summary(summary: ThroughputSummary) -> str:
    lines = [
        "# Throughput Summary",
        "",
        "## Basic Info",
        "",
        f"- Log file: {summary.log_path}",
        f"- Wall time: {summary.wall_seconds:.3f}s",
        f"- NPU count: {summary.npu_count}",
        "",
        "## Overall",
        "",
        *_bucket_lines(summary.overall, summary.wall_seconds, summary.npu_count),
        "",
        "## By Prompt",
        "",
    ]
    if not summary.by_prompt:
        lines.append("- No model API calls were found in the log.")
    else:
        for prompt_name, bucket in summary.by_prompt.items():
            lines.append(f"### {prompt_name}")
            lines.append("")
            lines.extend(_bucket_lines(bucket, summary.wall_seconds, summary.npu_count))
            lines.append("")

    lines.extend(["## By Run Context", ""])
    if not summary.by_run_context:
        lines.append("- No model API calls were found in the log.")
    else:
        for run_context, bucket in summary.by_run_context.items():
            lines.append(f"### {run_context}")
            lines.append("")
            lines.extend(_bucket_lines(bucket, summary.wall_seconds, summary.npu_count))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_summary(summary: ThroughputSummary) -> str:
    payload = {
        "log_path": summary.log_path,
        "wall_seconds": summary.wall_seconds,
        "npu_count": summary.npu_count,
        "overall": _bucket_payload(summary.overall, summary.wall_seconds, summary.npu_count),
        "by_prompt": {
            prompt_name: _bucket_payload(bucket, summary.wall_seconds, summary.npu_count)
            for prompt_name, bucket in summary.by_prompt.items()
        },
        "by_run_context": {
            run_context: _bucket_payload(bucket, summary.wall_seconds, summary.npu_count)
            for run_context, bucket in summary.by_run_context.items()
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _bucket_lines(bucket: ThroughputBucket, wall_seconds: float, npu_count: int) -> list[str]:
    stats = _bucket_payload(bucket, wall_seconds, npu_count)
    lines = [
        f"- API attempts: {stats['api_attempts']}",
        f"- Successful API attempts: {stats['successful_api_attempts']}",
        f"- Attempts with usage: {stats['attempts_with_usage']}",
        f"- Prompt tokens: {stats['prompt_tokens']}",
        f"- Completion tokens: {stats['completion_tokens']}",
        f"- Total tokens: {stats['total_tokens']}",
    ]
    if stats["tokens_per_second"] is None:
        lines.append("- Tokens/sec: unavailable (token usage not returned by model service)")
        lines.append("- Btokens/day: unavailable")
        lines.append("- Btokens/day/kNPU: unavailable")
        lines.append("- Output Btokens/day: unavailable")
        lines.append("- Output Btokens/day/kNPU: unavailable")
    else:
        lines.append(f"- Tokens/sec: {stats['tokens_per_second']:.6f}")
        lines.append(f"- Btokens/day: {stats['btokens_per_day']:.6f}")
        lines.append(f"- Btokens/day/kNPU: {stats['btokens_per_day_per_knpu']:.6f}")
        lines.append(f"- Output tokens/sec: {stats['output_tokens_per_second']:.6f}")
        lines.append(f"- Output Btokens/day: {stats['output_btokens_per_day']:.6f}")
        lines.append(f"- Output Btokens/day/kNPU: {stats['output_btokens_per_day_per_knpu']:.6f}")
    return lines


def _bucket_payload(bucket: ThroughputBucket, wall_seconds: float, npu_count: int) -> dict[str, float | int | None]:
    tokens_per_second = None
    output_tokens_per_second = None
    btokens_per_day = None
    btokens_per_day_per_knpu = None
    output_btokens_per_day = None
    output_btokens_per_day_per_knpu = None

    if bucket.total_tokens > 0 and wall_seconds > 0:
        tokens_per_second = bucket.total_tokens / wall_seconds
        btokens_per_day = tokens_per_second * 86400 / 1_000_000_000
        btokens_per_day_per_knpu = btokens_per_day * 1000 / max(1, npu_count)

    if bucket.completion_tokens > 0 and wall_seconds > 0:
        output_tokens_per_second = bucket.completion_tokens / wall_seconds
        output_btokens_per_day = output_tokens_per_second * 86400 / 1_000_000_000
        output_btokens_per_day_per_knpu = output_btokens_per_day * 1000 / max(1, npu_count)

    return {
        **asdict(bucket),
        "tokens_per_second": tokens_per_second,
        "btokens_per_day": btokens_per_day,
        "btokens_per_day_per_knpu": btokens_per_day_per_knpu,
        "output_tokens_per_second": output_tokens_per_second,
        "output_btokens_per_day": output_btokens_per_day,
        "output_btokens_per_day_per_knpu": output_btokens_per_day_per_knpu,
    }
