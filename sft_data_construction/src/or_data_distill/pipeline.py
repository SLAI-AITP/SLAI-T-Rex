from __future__ import annotations

import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .io_utils import read_jsonl, sha256_json, timestamp, write_json, write_jsonl
from .ir_schema import infer_data_interface, normalize_mode, validate_ir
from .llm_client import ChatClient, LLMConfig, base_urls_from_dict, config_from_dict
from .quality_gate import validate_sft_record


OUTPUT_NAMES = (
    "requests.jsonl",
    "attempts.jsonl",
    "synthetic_ir.jsonl",
    "accepted_synthetic_pool.jsonl",
    "sft.jsonl",
    "rejected.jsonl",
    "surplus_sft.jsonl",
    "surplus_synthetic_pool.jsonl",
    "manifest.json",
)


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _prompt_messages(template: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You create high-quality optimization modeling training data."},
        {"role": "user", "content": template + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _problem_text(row: dict[str, Any]) -> str:
    if isinstance(row.get("problem"), str):
        return str(row["problem"])
    if isinstance(row.get("synthetic_problem_brief"), str):
        return str(row["synthetic_problem_brief"])
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return ""


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "mode": row.get("mode") or metadata.get("mode"),
        "domain": row.get("domain") or metadata.get("domain"),
        "structure": row.get("structure") or metadata.get("structure"),
        "difficulty": row.get("difficulty") or metadata.get("difficulty"),
        "data_interface": row.get("data_interface") or metadata.get("data_interface"),
        "answer_style": row.get("answer_style") or metadata.get("answer_style"),
    }


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("ir_id") or sha256_json(row)[:16])


def _score_parent(row: dict[str, Any], target_buckets: dict[str, str]) -> int:
    metadata = _row_metadata(row)
    score = 0
    for key in ("mode", "domain", "structure", "difficulty"):
        if metadata.get(key) and metadata.get(key) == target_buckets.get(key):
            score += 1
    return score


def _weighted_choice(rows: list[dict[str, Any]], weights: list[float], rng: random.Random) -> dict[str, Any]:
    total = sum(weights)
    if total <= 0:
        return rng.choice(rows)
    pick = rng.random() * total
    running = 0.0
    for row, weight in zip(rows, weights):
        running += weight
        if running >= pick:
            return row
    return rows[-1]


def _choose_parent(
    *,
    seeds: list[dict[str, Any]],
    synthetic_pool: list[dict[str, Any]],
    target_buckets: dict[str, str],
    parent_cfg: dict[str, Any],
    usage: dict[str, int],
    rng: random.Random,
) -> dict[str, Any]:
    mode = str(parent_cfg.get("parent_pool_mode") or "hybrid")
    synthetic_share = float(parent_cfg.get("synthetic_parent_share", 0.5))
    top_k = max(1, int(parent_cfg.get("parent_match_top_k", 8)))
    usage_penalty = max(0.0, float(parent_cfg.get("parent_usage_penalty", 0.25)))
    exact_probability = min(1.0, max(0.0, float(parent_cfg.get("parent_exact_match_probability", 0.5))))

    if mode in {"seed_only", "semantic_only"} or not synthetic_pool:
        source_rows = seeds
    elif mode == "synthetic_only":
        source_rows = synthetic_pool
    elif mode == "snowball":
        source_rows = seeds + synthetic_pool
    else:
        source_rows = synthetic_pool if rng.random() < synthetic_share else seeds
        if not source_rows:
            source_rows = seeds or synthetic_pool

    ranked = sorted(source_rows, key=lambda row: (_score_parent(row, target_buckets), rng.random()), reverse=True)
    candidates = ranked[: min(top_k, len(ranked))]
    if not candidates:
        raise ValueError("No parent examples available")

    exact = [row for row in candidates if _score_parent(row, target_buckets) >= 4]
    if exact and len(candidates) > len(exact) and rng.random() > exact_probability:
        candidates = [row for row in candidates if row not in exact]

    weights = [1.0 / (1.0 + usage.get(_row_id(row), 0) * usage_penalty) for row in candidates]
    selected = _weighted_choice(candidates, weights, rng)
    usage[_row_id(selected)] = usage.get(_row_id(selected), 0) + 1
    return selected


def _choose_target_buckets(config: dict[str, Any], rng: random.Random) -> dict[str, str]:
    modes = config.get("problem_modes") or ["DP", "DT", "DPS"]
    domains = config.get("domains") or ["generic"]
    structures = config.get("structures") or ["generic_lp_mip"]
    difficulties = config.get("difficulties") or ["medium"]
    return {
        "mode": str(rng.choice(modes)),
        "domain": str(rng.choice(domains)),
        "structure": str(rng.choice(structures)),
        "difficulty": str(rng.choice(difficulties)),
    }


def _render_data_files(data_files: Any, *, max_chars: int = 6000) -> str:
    if not isinstance(data_files, list) or not data_files:
        return ""
    blocks = ["\n\nThe following structured data files are provided:"]
    for item in data_files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("name") or "data.txt")
        kind = str(item.get("kind") or "").lower()
        content = item.get("content", "")
        if isinstance(content, (dict, list)):
            language = "json" if kind == "json" or path.endswith(".json") else ""
            rendered = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            language = "csv" if kind == "csv" or path.endswith(".csv") else ""
            rendered = str(content)
        if len(rendered) > max_chars:
            continue
        blocks.append(f"\n### {path}\n```{language}\n{rendered}\n```")
    return "\n".join(blocks) if len(blocks) > 1 else ""


def normalize_synthetic_ir(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    mode = normalize_mode(str(out.get("mode") or "DP"))
    out["mode"] = mode
    out["data_interface"] = infer_data_interface(mode, str(out.get("synthetic_problem_brief") or ""))
    if not out.get("domain"):
        out["domain"] = "generic"
    if not out.get("structure"):
        out["structure"] = "generic_lp_mip"
    if not out.get("difficulty"):
        out["difficulty"] = "medium"
    if not out.get("answer_style"):
        out["answer_style"] = "math_model"
    return out


def _chat_cache_key(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    return sha256_json(
        {
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "messages": messages,
        }
    )


def _chat(
    *,
    client_configs: list[LLMConfig],
    client_index: int,
    messages: list[dict[str, str]],
    stage: str,
    cache_dir: Path,
    cache_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = client_configs[client_index % len(client_configs)]
    request_hash = _chat_cache_key(config, messages)
    cache_path = cache_dir / f"{request_hash}.json"
    if cache_enabled and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload, {
            "id": request_hash[:16],
            "stage": stage,
            "request_hash": request_hash,
            "cached": True,
            "latency_seconds": 0.0,
            "usage": payload.get("usage"),
        }

    result = ChatClient(config).chat(messages)
    payload = {
        "content": result.get("content", ""),
        "raw": result.get("raw"),
        "usage": result.get("usage"),
    }
    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload, {
        "id": request_hash[:16],
        "stage": stage,
        "request_hash": request_hash,
        "cached": False,
        "latency_seconds": result.get("latency_seconds"),
        "usage": result.get("usage"),
        "client_index": client_index % len(client_configs),
    }


def _run_one_sample(
    *,
    attempt_id: str,
    round_id: int,
    idx: int,
    parent: dict[str, Any],
    target_buckets: dict[str, str],
    config: dict[str, Any],
    synthetic_prompt: str,
    problem_prompt: str,
    answer_prompt: str,
    client_configs: list[LLMConfig],
    cache_dir: Path,
    cache_enabled: bool,
    dry_run: bool,
) -> dict[str, Any]:
    base_payload = {
        "parent": parent,
        "requirements": {
            "problem_modes": config.get("problem_modes", ["DP", "DT", "DPS"]),
            "domains": config.get("domains", []),
            "structures": config.get("structures", []),
            "difficulties": config.get("difficulties", []),
            "target_buckets": target_buckets,
        },
    }
    messages = _prompt_messages(synthetic_prompt, base_payload)
    request_id = sha256_json({"stage": "synthetic_ir", "messages": messages})[:16]
    requests: list[dict[str, Any]] = [
        {
            "id": request_id,
            "stage": "synthetic_ir",
            "attempt_id": attempt_id,
            "round": round_id,
            "messages": messages,
        }
    ]
    if dry_run:
        return {
            "idx": idx,
            "attempt_id": attempt_id,
            "round": round_id,
            "requests": requests,
            "synthetic": None,
            "sft": None,
            "rejected": None,
        }

    client_index = idx
    try:
        syn_payload, syn_record = _chat(
            client_configs=client_configs,
            client_index=client_index,
            messages=messages,
            stage="synthetic_ir",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        requests[-1].update(syn_record)
        syn = normalize_synthetic_ir(extract_json_object(str(syn_payload.get("content") or "")))
        ir_issues = validate_ir(
            {
                **syn,
                "decision_variables": syn.get("decision_variables")
                if isinstance(syn.get("decision_variables"), list)
                else [syn.get("decision_variables")]
                if syn.get("decision_variables")
                else [],
                "constraints": syn.get("constraints")
                if isinstance(syn.get("constraints"), list)
                else [syn.get("constraints")]
                if syn.get("constraints")
                else [],
            }
        )
        if ir_issues:
            return {
                "idx": idx,
                "attempt_id": attempt_id,
                "round": round_id,
                "requests": requests,
                "synthetic": None,
                "sft": None,
                "rejected": {"stage": "synthetic_ir", "attempt_id": attempt_id, "issues": ir_issues, "row": syn},
            }
        syn["id"] = f"ir_{sha256_json(syn)[:16]}"

        problem_messages = _prompt_messages(problem_prompt, {"ir": syn})
        problem_payload, problem_record = _chat(
            client_configs=client_configs,
            client_index=client_index,
            messages=problem_messages,
            stage="problem",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        requests.append(
            {
                "id": problem_record["id"],
                "stage": "problem",
                "attempt_id": attempt_id,
                "round": round_id,
                "messages": problem_messages,
                **problem_record,
            }
        )
        problem = extract_json_object(str(problem_payload.get("content") or ""))

        answer_messages = _prompt_messages(answer_prompt, {"ir": syn, "problem": problem})
        answer_payload, answer_record = _chat(
            client_configs=client_configs,
            client_index=client_index,
            messages=answer_messages,
            stage="answer",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
        requests.append(
            {
                "id": answer_record["id"],
                "stage": "answer",
                "attempt_id": attempt_id,
                "round": round_id,
                "messages": answer_messages,
                **answer_record,
            }
        )
        answer = extract_json_object(str(answer_payload.get("content") or ""))

        user_text = str(problem.get("problem") or problem.get("text") or "")
        user_text += _render_data_files(problem.get("data_files"))
        assistant_text = str(answer.get("answer") or answer.get("text") or "")
        metadata = {
            "mode": syn.get("mode"),
            "domain": syn.get("domain"),
            "structure": syn.get("structure"),
            "difficulty": syn.get("difficulty"),
            "data_interface": syn.get("data_interface"),
            "answer_style": syn.get("answer_style"),
            "generation_stage": "silver",
            "quality_status": "accepted",
            "source_ir_id": syn.get("id"),
            "target_buckets": target_buckets,
        }
        row = {
            "id": f"sft_{sha256_json({'u': user_text, 'a': assistant_text, 'm': metadata})[:16]}",
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "metadata": metadata,
        }
        issues = validate_sft_record(row)
        if issues:
            row["metadata"]["quality_status"] = "rejected"
            return {
                "idx": idx,
                "attempt_id": attempt_id,
                "round": round_id,
                "requests": requests,
                "synthetic": syn,
                "sft": None,
                "rejected": {"stage": "sft_quality", "attempt_id": attempt_id, "row": row, "issues": issues},
            }
        return {
            "idx": idx,
            "attempt_id": attempt_id,
            "round": round_id,
            "requests": requests,
            "synthetic": syn,
            "sft": row,
            "rejected": None,
        }
    except Exception as exc:
        return {
            "idx": idx,
            "attempt_id": attempt_id,
            "round": round_id,
            "requests": requests,
            "synthetic": None,
            "sft": None,
            "rejected": {"stage": "pipeline", "attempt_id": attempt_id, "error": str(exc), "request_id": request_id},
        }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], append=True)


def _load_synthetic_pool(paths: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_value in _as_list(paths):
        rows.extend(read_jsonl(Path(path_value)))
    return rows


def _existing_attempt_summary(run_dir: Path) -> tuple[int, int]:
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    if not attempts:
        return 0, 0
    max_idx = max(int(row.get("idx") or 0) for row in attempts)
    max_round = max(int(row.get("round") or 0) for row in attempts)
    return max_idx + 1, max_round


def _clear_outputs(run_dir: Path) -> None:
    for name in OUTPUT_NAMES:
        path = run_dir / name
        if path.exists():
            path.unlink()


def _similarity_issue(row: dict[str, Any], token_index: list[set[str]], threshold: float) -> dict[str, Any] | None:
    if threshold <= 0:
        return None
    tokens = _text_tokens(str(row.get("messages", [{}])[0].get("content") or ""))
    if not tokens:
        return {"issue": "empty_problem_tokens", "similarity": 1.0}
    best = max((_jaccard(tokens, old) for old in token_index), default=0.0)
    if best > threshold:
        return {"issue": "problem_similarity_too_high", "similarity": round(best, 4), "threshold": threshold}
    return None


def _build_client_configs(llm_cfg: dict[str, Any]) -> list[LLMConfig]:
    return [config_from_dict(llm_cfg, base_url=base_url) for base_url in base_urls_from_dict(llm_cfg)]


def _compute_concurrency(run: dict[str, Any], llm_cfg: dict[str, Any], client_count: int) -> int:
    if run.get("concurrency") or run.get("max_workers"):
        return max(1, int(run.get("concurrency") or run.get("max_workers")))
    workers_per_api = int(llm_cfg.get("workers_per_api") or 0)
    if workers_per_api > 0:
        return max(1, workers_per_api * max(1, client_count))
    return 1


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def run_pipeline(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run = config.get("run", {})
    paths = config.get("paths", {})
    prompts = config.get("prompts", {})
    llm_cfg = config.get("llm", {})
    cache_cfg = config.get("cache", {})
    quality_cfg = config.get("quality", {})
    parent_cfg = config.get("parent_pool", {})

    run_id = str(run.get("run_id") or f"run_{timestamp()}")
    output_root = Path(run.get("output_root") or "runs")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(run.get("resume", False))
    force = bool(run.get("force", False))
    if force or not resume:
        _clear_outputs(run_dir)

    target_count = int(run.get("accepted_target_count") or run.get("target_count") or 10)
    max_rounds = max(1, int(run.get("max_rounds") or 1))
    generation_oversample = max(1.0, float(run.get("generation_oversample") or 1.0))
    max_attempts = int(run.get("max_attempts") or 0)
    rng = random.Random(int(run.get("seed") or 42))

    client_configs = _build_client_configs(llm_cfg)
    concurrency = _compute_concurrency(run, llm_cfg, len(client_configs))
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_dir = Path(cache_cfg.get("dir") or "cache/chat")
    similarity_threshold = float(
        quality_cfg.get("problem_similarity_threshold")
        or quality_cfg.get("max_problem_similarity")
        or 0.0
    )
    compare_to_seeds = bool(quality_cfg.get("compare_to_seeds", True))
    compare_to_run = bool(quality_cfg.get("compare_to_run", True))

    seed_path = Path(paths.get("seeds") or "examples/seeds/small_seed.jsonl")
    seeds = read_jsonl(seed_path)
    if not seeds:
        raise ValueError(f"No seeds found at {seed_path}")

    synthetic_pool = _load_synthetic_pool(paths.get("synthetic_pool"))
    if resume:
        synthetic_pool.extend(read_jsonl(run_dir / "accepted_synthetic_pool.jsonl"))

    existing_sft = read_jsonl(run_dir / "sft.jsonl") if resume else []
    existing_ids = {str(row.get("id")) for row in existing_sft if row.get("id")}
    next_idx, previous_round = _existing_attempt_summary(run_dir) if resume else (0, 0)

    token_index: list[set[str]] = []
    if compare_to_seeds:
        token_index.extend(_text_tokens(_problem_text(row)) for row in seeds if _problem_text(row))
    if compare_to_run:
        token_index.extend(
            _text_tokens(str(row.get("messages", [{}])[0].get("content") or ""))
            for row in existing_sft
            if isinstance(row.get("messages"), list)
        )

    synthetic_prompt = load_prompt(prompts.get("generate_synthetic_ir") or "prompts/generate_synthetic_ir.txt")
    problem_prompt = load_prompt(prompts.get("render_problem") or "prompts/render_problem.txt")
    answer_prompt = load_prompt(prompts.get("render_answer") or "prompts/render_answer.txt")

    accepted_count = len(existing_sft)
    attempts_submitted = next_idx
    rejected_count = len(read_jsonl(run_dir / "rejected.jsonl")) if resume else 0
    surplus_count = len(read_jsonl(run_dir / "surplus_sft.jsonl")) if resume else 0
    requests_count = len(read_jsonl(run_dir / "requests.jsonl")) if resume else 0
    synthetic_count = len(read_jsonl(run_dir / "synthetic_ir.jsonl")) if resume else 0
    usage: dict[str, int] = {}

    if dry_run:
        planned = target_count
        for idx in range(planned):
            target_buckets = _choose_target_buckets(config, rng)
            parent = _choose_parent(
                seeds=seeds,
                synthetic_pool=synthetic_pool,
                target_buckets=target_buckets,
                parent_cfg=parent_cfg,
                usage=usage,
                rng=rng,
            )
            payload = {
                "parent": parent,
                "requirements": {
                    "problem_modes": config.get("problem_modes", ["DP", "DT", "DPS"]),
                    "domains": config.get("domains", []),
                    "structures": config.get("structures", []),
                    "difficulties": config.get("difficulties", []),
                    "target_buckets": target_buckets,
                },
            }
            messages = _prompt_messages(synthetic_prompt, payload)
            _append_jsonl(
                run_dir / "requests.jsonl",
                {
                    "id": sha256_json({"stage": "synthetic_ir", "messages": messages})[:16],
                    "stage": "synthetic_ir",
                    "attempt_id": f"attempt_{idx:08d}",
                    "round": 1,
                    "messages": messages,
                },
            )
        requests_count = planned
    else:
        for round_id in range(previous_round + 1, previous_round + max_rounds + 1):
            if accepted_count >= target_count:
                break
            remaining = target_count - accepted_count
            planned = max(1, math.ceil(remaining * generation_oversample))
            if max_attempts > 0:
                remaining_attempt_budget = max_attempts - attempts_submitted
                if remaining_attempt_budget <= 0:
                    break
                planned = min(planned, remaining_attempt_budget)

            jobs: list[dict[str, Any]] = []
            for _ in range(planned):
                idx = next_idx
                next_idx += 1
                target_buckets = _choose_target_buckets(config, rng)
                parent = _choose_parent(
                    seeds=seeds,
                    synthetic_pool=synthetic_pool,
                    target_buckets=target_buckets,
                    parent_cfg=parent_cfg,
                    usage=usage,
                    rng=rng,
                )
                jobs.append(
                    {
                        "attempt_id": f"attempt_{idx:08d}",
                        "round_id": round_id,
                        "idx": idx,
                        "parent": parent,
                        "target_buckets": target_buckets,
                        "config": config,
                        "synthetic_prompt": synthetic_prompt,
                        "problem_prompt": problem_prompt,
                        "answer_prompt": answer_prompt,
                        "client_configs": client_configs,
                        "cache_dir": cache_dir,
                        "cache_enabled": cache_enabled,
                        "dry_run": False,
                    }
                )
            attempts_submitted += len(jobs)
            completed_this_round = 0
            print(
                f"[or-data-distill] round {round_id}: target_remaining={remaining} attempts={len(jobs)} concurrency={concurrency}",
                flush=True,
            )

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {executor.submit(_run_one_sample, **job): job["attempt_id"] for job in jobs}
                for future in as_completed(future_map):
                    result = future.result()
                    completed_this_round += 1
                    for request in result.get("requests") or []:
                        requests_count += 1
                        _append_jsonl(run_dir / "requests.jsonl", request)
                    synthetic = result.get("synthetic")
                    if synthetic:
                        synthetic_count += 1
                        _append_jsonl(run_dir / "synthetic_ir.jsonl", synthetic)

                    status = "rejected"
                    sft = result.get("sft")
                    rejected = result.get("rejected")
                    if sft:
                        similarity = _similarity_issue(sft, token_index, similarity_threshold)
                        if sft.get("id") in existing_ids:
                            similarity = {"issue": "duplicate_sft_id", "similarity": 1.0}
                        if similarity:
                            rejected = {"stage": "similarity", "attempt_id": result["attempt_id"], "row": sft, **similarity}
                        elif accepted_count < target_count:
                            status = "accepted"
                            accepted_count += 1
                            existing_ids.add(str(sft.get("id")))
                            token_index.append(_text_tokens(str(sft["messages"][0]["content"])))
                            _append_jsonl(run_dir / "sft.jsonl", sft)
                            if synthetic:
                                _append_jsonl(run_dir / "accepted_synthetic_pool.jsonl", synthetic)
                                synthetic_pool.append(synthetic)
                        else:
                            status = "surplus"
                            surplus_count += 1
                            _append_jsonl(run_dir / "surplus_sft.jsonl", sft)
                            if synthetic:
                                _append_jsonl(run_dir / "surplus_synthetic_pool.jsonl", synthetic)

                    if status == "rejected":
                        rejected_count += 1
                        _append_jsonl(run_dir / "rejected.jsonl", rejected or {"stage": "unknown", "attempt_id": result["attempt_id"]})

                    _append_jsonl(
                        run_dir / "attempts.jsonl",
                        {
                            "attempt_id": result["attempt_id"],
                            "idx": result["idx"],
                            "round": result["round"],
                            "status": status,
                            "accepted_count_after": accepted_count,
                        },
                    )

                    if (
                        completed_this_round == 1
                        or completed_this_round % max(1, min(20, len(jobs) // 10 or 1)) == 0
                        or completed_this_round == len(jobs)
                    ):
                        print(
                            f"[or-data-distill] round {round_id}: completed {completed_this_round}/{len(jobs)} "
                            f"accepted={accepted_count}/{target_count} rejected={rejected_count} surplus={surplus_count}",
                            flush=True,
                        )
            if accepted_count >= target_count:
                break

    manifest = {
        "run_id": run_id,
        "seed_path": str(seed_path),
        "dry_run": dry_run,
        "accepted_target_count": target_count,
        "accepted_count": accepted_count,
        "resume": resume,
        "generation_oversample": generation_oversample,
        "max_rounds": max_rounds,
        "concurrency": concurrency,
        "api_endpoints": len(client_configs),
        "cache_enabled": cache_enabled,
        "similarity_threshold": similarity_threshold,
        "requests": requests_count,
        "synthetic_ir_rows": synthetic_count,
        "sft_rows": accepted_count,
        "rejected_rows": rejected_count,
        "surplus_rows": surplus_count,
        "paths": {
            "run_dir": str(run_dir),
            "requests": str(run_dir / "requests.jsonl"),
            "attempts": str(run_dir / "attempts.jsonl"),
            "sft": str(run_dir / "sft.jsonl"),
            "accepted_synthetic_pool": str(run_dir / "accepted_synthetic_pool.jsonl"),
            "rejected": str(run_dir / "rejected.jsonl"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }
    _write_manifest(run_dir / "manifest.json", manifest)
    return manifest
