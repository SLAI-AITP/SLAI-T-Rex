from __future__ import annotations

import re
from typing import Any

from .ir_schema import ANSWER_STYLES, DATA_INTERFACES, DIFFICULTIES, DOMAINS, PROBLEM_MODES, STRUCTURES

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("placeholder", re.compile(r"\bplaceholder\b", re.I)),
    ("todo", re.compile(r"\bTODO\b|\bTBD\b", re.I)),
    ("na", re.compile(r"\bN/A\b", re.I)),
    ("locked", re.compile(r"\bLocked\b", re.I)),
    ("sc0", re.compile(r"\bSc0\b", re.I)),
    ("skip_artifact", re.compile(r"\)\s*Skip\b|\bSkip the rest\b", re.I)),
)


def text_issues(text: str) -> list[str]:
    issues = []
    if not text or not text.strip():
        return ["empty_text"]
    for name, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            issues.append(f"forbidden:{name}")
    return issues


def validate_sft_record(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return ["messages_must_be_user_assistant_pair"]
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        issues.append("invalid_roles")
    issues.extend(f"user:{issue}" for issue in text_issues(str(messages[0].get("content") or "")))
    issues.extend(f"assistant:{issue}" for issue in text_issues(str(messages[1].get("content") or "")))
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_metadata")
    else:
        for key in ("mode", "domain", "structure", "difficulty", "data_interface", "answer_style"):
            if not metadata.get(key):
                issues.append(f"missing_metadata:{key}")
        mode = metadata.get("mode")
        data_interface = metadata.get("data_interface")
        if mode and mode not in PROBLEM_MODES:
            issues.append("invalid_metadata:mode")
        if metadata.get("domain") and metadata.get("domain") not in DOMAINS:
            issues.append("invalid_metadata:domain")
        if metadata.get("structure") and metadata.get("structure") not in STRUCTURES:
            issues.append("invalid_metadata:structure")
        if metadata.get("difficulty") and metadata.get("difficulty") not in DIFFICULTIES:
            issues.append("invalid_metadata:difficulty")
        if data_interface and data_interface not in DATA_INTERFACES:
            issues.append("invalid_metadata:data_interface")
        if metadata.get("answer_style") and metadata.get("answer_style") not in ANSWER_STYLES:
            issues.append("invalid_metadata:answer_style")
        if mode == "DP" and data_interface != "inline_text":
            issues.append("mode_data_interface_mismatch:DP_requires_inline_text")
        if mode == "DT":
            if data_interface != "markdown_table":
                issues.append("mode_data_interface_mismatch:DT_requires_markdown_table")
            user_text = str(messages[0].get("content") or "") if isinstance(messages, list) else ""
            if "|" not in user_text or not re.search(r"\|[\\s:-]+\\|", user_text):
                issues.append("mode_content_mismatch:DT_requires_markdown_table")
        if mode == "DPS" and data_interface != "attached_files":
            issues.append("mode_data_interface_mismatch:DPS_requires_attached_files")
        if mode == "DPS":
            user_text = str(messages[0].get("content") or "") if isinstance(messages, list) else ""
            has_file_block = bool(
                "```" in user_text
                and (
                    "structured data files are provided" in user_text.lower()
                    or re.search(r"###\s+\S+\.(json|csv|txt)", user_text, flags=re.I)
                )
            )
            if not has_file_block:
                issues.append("mode_content_mismatch:DPS_requires_inline_file_block")
    return issues
