from __future__ import annotations

import hashlib
import json
import os
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


DEFAULT_SFT_FEWSHOT_SOURCE = None

TARGET_ID_PATTERN = re.compile(r"\[Target ID\]\s*([A-Za-z0-9_-]+)")
REFERENCE_DOMAIN_PATTERN = re.compile(r"^\s*#\s*Domain:\s*(.+?)\s*$", re.MULTILINE)
REFERENCE_PROBLEM_TYPE_PATTERN = re.compile(r"^\s*#\s*Problem type:\s*(.+?)\s*$", re.MULTILINE)
REFERENCE_VARIANT_PATTERN = re.compile(r"^\s*#\s*Variant description:\s*(.+?)\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def attempt_seed(base_seed: int, attempt_index: int) -> int:
    return int(base_seed) + int(attempt_index)


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _sample_retrieval_metadata(
    sample: Dict[str, Any],
    include_reference_annotations: bool = True,
) -> Dict[str, str]:
    reference_code = str(sample.get("reference_code") or "") if include_reference_annotations else ""
    domain = str(sample.get("domain") or _first_match(REFERENCE_DOMAIN_PATTERN, reference_code))
    problem_type = str(
        sample.get("problem_type") or _first_match(REFERENCE_PROBLEM_TYPE_PATTERN, reference_code)
    )
    variant = str(sample.get("variant") or _first_match(REFERENCE_VARIANT_PATTERN, reference_code))
    wp_type = str(sample.get("wp_type") or "")
    retrieval_text = "\n".join(
        value
        for value in [
            str(sample.get("problem") or ""),
            domain,
            problem_type,
            variant,
            wp_type,
        ]
        if value
    )
    return {
        "domain": domain,
        "problem_type": problem_type,
        "variant": variant,
        "wp_type": wp_type,
        "retrieval_text": retrieval_text,
    }


def _messages_by_role(record: Dict[str, Any], role: str) -> List[str]:
    return [
        str(message.get("content", ""))
        for message in record.get("messages", [])
        if isinstance(message, dict) and message.get("role") == role
    ]


def _record_target_id(record: Dict[str, Any]) -> Optional[str]:
    user_text = "\n".join(_messages_by_role(record, "user"))
    match = TARGET_ID_PATTERN.search(user_text)
    if match:
        return match.group(1)

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    answer_id = str(record.get("answer_id") or metadata.get("answer_id") or "")
    for target_id in (
        "nl4opt_solver",
        "optibench_solver",
        "bench4opt_feasible_solver",
        "bench4opt_orgeval",
    ):
        if target_id in answer_id:
            return target_id
    return None


def load_sft_fewshot_pool(
    target_id: str,
    source_path: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    if not source_path:
        raise FileNotFoundError("Pass --few_shot_source with a validated messages jsonl.")
    path = Path(source_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"few-shot source not found: {path}. Pass --few_shot_source with a validated messages jsonl."
        )

    pool: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if _record_target_id(record) != target_id:
                continue
            user_messages = _messages_by_role(record, "user")
            assistant_messages = _messages_by_role(record, "assistant")
            if not user_messages or not assistant_messages:
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}
            example_id = (
                str(record.get("answer_id") or metadata.get("answer_id") or "")
                or f"{path.name}:{line_number}"
            )
            source_key = "|".join(
                value
                for value in [
                    example_id,
                    str(record.get("problem_id") or lineage.get("problem_id") or ""),
                    str(lineage.get("source_ir_id") or ""),
                ]
                if value
            )
            pool.append(
                {
                    "example_id": example_id,
                    "source": str(path),
                    "source_key": source_key,
                    "user": user_messages[-1],
                    "assistant": assistant_messages[-1],
                }
            )
            if max_examples is not None and len(pool) >= max_examples:
                break
    return pool


def load_result_fewshot_pool(
    result_path: str | Path,
    samples: Iterable[Dict[str, Any]],
    current_prompt_renderer: Callable[[Dict[str, Any]], str],
    source_label: Optional[str] = None,
) -> List[Dict[str, str]]:
    path = Path(result_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"few-shot result source not found: {path}")

    sample_by_id: Dict[str, Dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        sample_id = sample.get("sample_id", sample.get("id", index))
        sample_by_id[str(sample_id)] = sample

    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list):
        raise ValueError(f"few-shot result source must be a JSON list: {path}")

    pool: List[Dict[str, str]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        reward = result.get("reward") if isinstance(result.get("reward"), dict) else {}
        verified_reward = (
            float(reward.get("code_reward", 0.0) or 0.0) >= 1.0
            and float(reward.get("wl_reward", 0.0) or 0.0) >= 1.0
        )
        if result.get("success") is not True and not verified_reward:
            continue
        completion = str(result.get("completion") or "").strip()
        if not completion:
            continue
        result_id = result.get("sample_id", result.get("id"))
        if result_id is None:
            continue
        sample = sample_by_id.get(str(result_id))
        if sample is None:
            continue
        example_id = str(result_id)
        retrieval_metadata = _sample_retrieval_metadata(sample)
        pool.append(
            {
                "example_id": f"{source_label or path.name}:{example_id}",
                "source": source_label or str(path),
                "source_key": example_id,
                "user": current_prompt_renderer(sample),
                "assistant": completion,
                **retrieval_metadata,
            }
        )
    return pool


def _candidate_fewshot_examples(
    pool: List[Dict[str, str]],
    current_id: Any,
) -> List[Dict[str, str]]:
    current_text = str(current_id)

    def is_current_problem(example: Dict[str, str]) -> bool:
        example_id = str(example.get("example_id", ""))
        source_key = str(example.get("source_key", ""))
        if example_id == current_text or source_key == current_text:
            return True
        return current_text in {part for part in source_key.split("|") if part}

    return [
        example
        for example in pool
        if current_text and not is_current_problem(example)
    ]


def _first_unique_source_keys(examples: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    seen: set[str] = set()
    for example in examples:
        source_key = str(example.get("source_key") or example.get("example_id") or "")
        if source_key in seen:
            continue
        seen.add(source_key)
        selected.append(example)
        if len(selected) >= k:
            break
    return selected


def select_fewshot_examples(
    pool: List[Dict[str, str]],
    current_id: Any,
    k: int,
    seed: int,
) -> List[Dict[str, str]]:
    if k <= 0:
        return []
    current_text = str(current_id)

    candidates = _candidate_fewshot_examples(pool, current_id)
    if len(candidates) < k:
        raise ValueError(
            f"Need {k} few-shot examples excluding current_id={current_text}, only {len(candidates)} available"
        )
    rng = random.Random(stable_int(f"{seed}:{current_text}"))
    rng.shuffle(candidates)
    selected = _first_unique_source_keys(candidates, k)
    if len(selected) < k:
        raise ValueError(
            f"Need {k} unique few-shot examples excluding current_id={current_text}, only {len(selected)} available"
        )
    return selected


def _bm25_scores(
    query_tokens: List[str],
    candidate_tokens: List[List[str]],
) -> List[float]:
    if not query_tokens:
        return [0.0 for _ in candidate_tokens]

    doc_count = max(1, len(candidate_tokens))
    document_frequency: Counter[str] = Counter()
    for tokens in candidate_tokens:
        document_frequency.update(set(tokens))

    average_length = sum(len(tokens) for tokens in candidate_tokens) / doc_count
    average_length = max(1.0, average_length)
    query_terms = Counter(query_tokens)
    k1 = 1.2
    b = 0.75
    scores: List[float] = []

    for tokens in candidate_tokens:
        term_counts = Counter(tokens)
        doc_length = max(1, len(tokens))
        score = 0.0
        for term, query_weight in query_terms.items():
            term_frequency = term_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            idf = math.log(1.0 + (doc_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * doc_length / average_length)
            score += query_weight * idf * (term_frequency * (k1 + 1.0) / denominator)
        scores.append(score)
    return scores


def select_similar_fewshot_examples(
    pool: List[Dict[str, str]],
    current_sample: Dict[str, Any],
    k: int,
    seed: int,
) -> List[Dict[str, str]]:
    if k <= 0:
        return []

    current_id = current_sample.get("id", current_sample.get("sample_id", ""))
    current_text = str(current_id)
    candidates = _candidate_fewshot_examples(pool, current_id)
    if len(candidates) < k:
        raise ValueError(
            f"Need {k} few-shot examples excluding current_id={current_text}, only {len(candidates)} available"
        )

    current_metadata = _sample_retrieval_metadata(
        current_sample,
        include_reference_annotations=False,
    )
    query_tokens = _tokenize(current_metadata["retrieval_text"])
    candidate_tokens = [
        _tokenize(str(example.get("retrieval_text") or example.get("user") or ""))
        for example in candidates
    ]
    lexical_scores = _bm25_scores(query_tokens, candidate_tokens)

    ranked_examples: List[Dict[str, str]] = []
    for example, lexical_score in zip(candidates, lexical_scores):
        score = lexical_score
        if current_metadata["domain"] and example.get("domain") == current_metadata["domain"]:
            score += 2.0
        if current_metadata["problem_type"] and example.get("problem_type") == current_metadata["problem_type"]:
            score += 1.0
        if current_metadata["wp_type"] and example.get("wp_type") == current_metadata["wp_type"]:
            score += 0.25

        ranked = dict(example)
        ranked["similarity_score"] = f"{score:.6f}"
        ranked["few_shot_strategy"] = "similar"
        ranked_examples.append(ranked)

    ranked_examples.sort(
        key=lambda example: (
            -float(example.get("similarity_score", "0") or 0.0),
            stable_int(f"{seed}:{current_text}:{example.get('example_id', '')}"),
        )
    )
    selected = _first_unique_source_keys(ranked_examples, k)
    if len(selected) < k:
        raise ValueError(
            f"Need {k} unique few-shot examples excluding current_id={current_text}, only {len(selected)} available"
        )
    return selected


def augment_chat_messages(
    messages: List[Dict[str, str]],
    examples: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    if not examples:
        return messages

    augmented: List[Dict[str, str]] = []
    first_index = 0
    if messages and messages[0].get("role") == "system":
        system_message = dict(messages[0])
        system_message["content"] = (
            system_message.get("content", "").rstrip()
            + "\n\nYou will see solved examples from the same target before the current question. "
            "Use them only as format and modeling examples. Do not copy example-specific data."
        )
        augmented.append(system_message)
        first_index = 1

    for index, example in enumerate(examples, 1):
        augmented.append(
            {
                "role": "user",
                "content": (
                    f"[Few-shot example {index}; not the current problem]\n"
                    f"{example['user']}"
                ),
            }
        )
        augmented.append({"role": "assistant", "content": example["assistant"]})

    for message in messages[first_index:]:
        current = dict(message)
        if current.get("role") == "user":
            current["content"] = "[Current problem to solve]\n" + current.get("content", "")
        augmented.append(current)
    return augmented


def augment_prompt_text(prompt: str, examples: List[Dict[str, str]]) -> str:
    if not examples:
        return prompt

    sections = [
        "The following are solved examples from the same target. They are not the current problem. "
        "Use them only as output-format and modeling examples. Do not copy example-specific data."
    ]
    for index, example in enumerate(examples, 1):
        sections.append(
            "\n".join(
                [
                    f"### Few-shot example {index} (not current problem)",
                    "[Example prompt]",
                    example["user"],
                    "[Example answer]",
                    example["assistant"],
                    f"### End few-shot example {index}",
                ]
            )
        )
    sections.append("### Current problem")
    sections.append(prompt)
    return "\n\n".join(sections)


def fewshot_metadata(examples: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "few_shot_count": len(examples),
        "few_shot_examples": [
            {
                "example_id": example.get("example_id"),
                "source": example.get("source"),
                "source_key": example.get("source_key"),
                "strategy": example.get("few_shot_strategy", "random"),
                "similarity_score": example.get("similarity_score"),
                "domain": example.get("domain"),
                "problem_type": example.get("problem_type"),
            }
            for example in examples
        ],
    }
