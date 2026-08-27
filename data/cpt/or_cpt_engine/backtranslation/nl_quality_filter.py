from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from or_cpt_engine.quality.semantic_coherence import semantic_coherence_issues
from or_cpt_engine.utils.io import append_jsonl, iter_jsonl, write_text


FORBIDDEN_PHRASES = (
    "optimal objective",
    "solver status",
    "gurobi",
    "python code",
    "chatgpt",
    "as an ai",
    "assistant",
    "user:",
)


def filter_nl_candidates(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    min_chars: int = 160,
    max_chars: int = 4000,
    min_number_coverage: float = 0.4,
) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "nl_validated_candidates.jsonl"
    rejected_path = output / "nl_rejected.jsonl"
    accepted_count = 0
    rejected_count = 0
    rejection_reasons: dict[str, int] = {}

    for row in (iter_jsonl(input_path) or []):
        accepted, output_row, reasons = filter_nl_candidate_row(
            row,
            min_chars=min_chars,
            max_chars=max_chars,
            min_number_coverage=min_number_coverage,
        )
        if not accepted:
            reason_key = reasons[0].split(":")[0] if reasons else "UNKNOWN"
            rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1
            append_jsonl(rejected_path, output_row)
            rejected_count += 1
        else:
            append_jsonl(accepted_path, output_row)
            accepted_count += 1

    write_text(output / "nl_quality_report.md", render_nl_report(accepted_count, rejected_count, rejection_reasons))
    return {"accepted": accepted_count, "rejected": rejected_count}


def filter_nl_candidate_row(
    row: dict[str, Any],
    *,
    min_chars: int = 160,
    max_chars: int = 4000,
    min_number_coverage: float = 0.4,
) -> tuple[bool, dict[str, Any], list[str]]:
    reasons = evaluate_nl_candidate(
        row,
        min_chars=min_chars,
        max_chars=max_chars,
        min_number_coverage=min_number_coverage,
    )
    if reasons:
        return (
            False,
            {
                **row,
                "nl_status": "REJECTED",
                "rejection_stage": "nl_quality_filter",
                "rejection_reason": "; ".join(reasons),
            },
            reasons,
        )
    return True, {**row, "nl_status": "PASS"}, []


def evaluate_nl_candidate(
    row: dict[str, Any],
    *,
    min_chars: int = 160,
    max_chars: int = 4000,
    min_number_coverage: float = 0.4,
) -> list[str]:
    text = str(row.get("problem_statement") or "")
    lowered = text.lower()
    reasons: list[str] = []
    if len(text) < min_chars:
        reasons.append("TOO_SHORT")
    if len(text) > max_chars:
        reasons.append("TOO_LONG")
    forbidden_hits = _forbidden_phrase_hits(lowered)
    if forbidden_hits:
        reasons.append("FORBIDDEN_PHRASE:" + ",".join(forbidden_hits[:5]))
    script_scan_text = "\n".join(
        [
            str(row.get("problem_background") or ""),
            text,
            json.dumps(row.get("structured_problem_data") or {}, ensure_ascii=False, sort_keys=True),
        ]
    )
    script_hits = _non_english_script_hits(row, script_scan_text)
    if script_hits:
        reasons.append("NON_ENGLISH_SCRIPT:" + ",".join(script_hits[:5]))
    semantic_scan_text = "\n".join([str(row.get("problem_background") or ""), text])
    semantic_hits = semantic_coherence_issues(str(row.get("generator_id") or ""), semantic_scan_text)
    if semantic_hits:
        reasons.append("SEMANTIC_COHERENCE:" + ",".join(semantic_hits[:5]))
    reasons.extend(_source_family_forbidden_reasons(row, lowered))
    reasons.extend(_business_context_reasons(row))
    coverage = _number_coverage(row)
    if coverage is not None and coverage < min_number_coverage:
        reasons.append(f"LOW_NUMBER_COVERAGE:{coverage:.2f}")
    reasons.extend(_source_compact_data_fidelity_reasons(row))
    return reasons


def _forbidden_phrase_hits(lowered_text: str) -> list[str]:
    hits = [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered_text]
    # A natural-language problem may describe the symbolic objective value,
    # e.g. "the total objective value is sum distance*z". Reject only solver
    # leakage such as "optimal objective value" or "solver objective value".
    if re.search(r"\b(?:optimal|solver|known|validated|reported)\s+objective\s+value\b", lowered_text):
        hits.append("objective value")
    return hits


def _non_english_script_hits(row: dict[str, Any], text: str) -> list[str]:
    language = str(row.get("language") or "en").lower()
    if language and not language.startswith("en"):
        return []
    hits: list[str] = []
    script_patterns = {
        "CJK": r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]",
        "HIRAGANA_KATAKANA": r"[\u3040-\u30FF]",
        "HANGUL": r"[\uAC00-\uD7AF]",
    }
    for name, pattern in script_patterns.items():
        if re.search(pattern, text):
            hits.append(name)
    return hits


def _source_family_forbidden_reasons(row: dict[str, Any], lowered_text: str) -> list[str]:
    if _is_lot_sizing_without_backlog(row):
        if _has_positive_backlog_or_shortage_misframing(lowered_text):
            return ["FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added"]
    if _is_capacitated_lot_sizing_without_expansion(row):
        if _has_positive_backlog_or_shortage_misframing(lowered_text):
            return ["FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added"]
        if _has_positive_capacity_expansion_misframing(lowered_text):
            return ["FORBIDDEN_SOURCE_MISFRAMING:capacity_expansion_added"]
        if _has_positive_zero_final_inventory_misframing(lowered_text):
            return ["FORBIDDEN_SOURCE_MISFRAMING:zero_final_inventory_added"]
    return []


def _is_lot_sizing_without_backlog(row: dict[str, Any]) -> bool:
    generator_id = str(row.get("generator_id") or "").lower()
    if "uncapacitatedlotsizingbacklogging" in generator_id:
        return False
    if "optmath_uncapacitatedlotsizing" in generator_id or "uncapacitatedlotsizing" in generator_id:
        return True
    source_compact = row.get("source_compact_data") or {}
    if not isinstance(source_compact, dict) or not source_compact.get("available") or source_compact.get("truncated"):
        return False
    source_value = source_compact.get("value")
    if not isinstance(source_value, dict):
        return False
    tables = source_value.get("compact_lotsizing_tables") if isinstance(source_value.get("compact_lotsizing_tables"), dict) else {}
    model_family = str(tables.get("model_family") or "").lower()
    return "without_backlog" in model_family or "without backlog" in model_family


def _is_capacitated_lot_sizing_without_expansion(row: dict[str, Any]) -> bool:
    generator_id = str(row.get("generator_id") or "").lower()
    if "optmath_clsp_expand_capacity" in generator_id or "clsp_expand_capacity" in generator_id:
        return True
    source_compact = row.get("source_compact_data") or {}
    if not isinstance(source_compact, dict) or not source_compact.get("available") or source_compact.get("truncated"):
        return False
    source_value = source_compact.get("value")
    if not isinstance(source_value, dict):
        return False
    tables = source_value.get("compact_lotsizing_tables") if isinstance(source_value.get("compact_lotsizing_tables"), dict) else {}
    model_family = str(tables.get("model_family") or "").lower()
    note = str(tables.get("source_model_note") or "").lower()
    return "capacitated" in model_family and (
        "without_backlog" in model_family
        or "without backlog" in model_family
        or "no capacity-expansion" in note
        or "no capacity expansion" in note
    )


def _has_positive_backlog_or_shortage_misframing(lowered_text: str) -> bool:
    positive_patterns = (
        r"\bbacklog\s+costs?\b",
        r"\bbacklog(?:ged)?\s+penalt(?:y|ies)\b",
        r"\bbackorders?\s+(?:are\s+)?allowed\b",
        r"\bbacklog(?:s|ged)?\s+(?:are\s+)?allowed\b",
        r"\bcan\s+backlog\b",
        r"\bmay\s+backlog\b",
        r"\ballow(?:s|ed|ing)?\s+backlog\b",
        r"\bending\s+backlog\b",
        r"\bfinal\s+backlog\b",
        r"\bbacklogged\s+amount\b",
        r"\bbackloggedamount\b",
        r"\bbacklogged\s+demand\b",
        r"\bbacklog\s+(?:state|variable|variables|carryover|carry-over|at\s+the\s+end)\b",
        r"\bcarry\s+forward\s+or\s+backlog\b",
        r"\bbacklog\s+if\s+demand\b",
        r"\blost[-\s]?sales\b",
        r"\bshortage\s+(?:cost|costs|penalt(?:y|ies)|variable|variables)\b",
        r"\bshortages?\s+(?:are\s+)?allowed\b",
        r"\bshortage\s+slack\b",
        r"\bunmet[-\s]?demand\b",
        r"\bunmet[-\s]?demand\s+slack\b",
    )
    for pattern in positive_patterns:
        for match in re.finditer(pattern, lowered_text):
            if not (_is_negated_context(lowered_text, match.start()) or _is_negated_list_context(lowered_text, match.start())):
                return True
    return False


def _has_positive_capacity_expansion_misframing(lowered_text: str) -> bool:
    positive_patterns = (
        r"\bcapacity[-\s]?expansion\b",
        r"\bexpand(?:ing)?\s+capacity\b",
        r"\bexpansion\s+(?:variable|variables|decision|decisions|cost|costs)\b",
        r"\bovertime\s+(?:purchase|capacity|hours|decision|decisions|cost|costs|variable|variables)\b",
        r"\bpurchase\s+(?:additional|extra|more)?\s*capacity\b",
        r"\bbuy\s+(?:additional|extra|more)?\s*capacity\b",
        r"\badd\s+(?:additional|extra|more)?\s*capacity\b",
    )
    for pattern in positive_patterns:
        for match in re.finditer(pattern, lowered_text):
            if not (_is_negated_context(lowered_text, match.start()) or _is_negated_list_context(lowered_text, match.start())):
                return True
    return False


def _has_positive_zero_final_inventory_misframing(lowered_text: str) -> bool:
    positive_patterns = (
        r"\bfinal\s+inventory\s+(?:must|should|has\s+to|is\s+required\s+to)\s+(?:be\s+)?(?:zero|0)\b",
        r"\bending\s+inventory\s+(?:must|should|has\s+to|is\s+required\s+to)\s+(?:be\s+)?(?:zero|0)\b",
        r"\bzero\s+final\s+inventory\b",
        r"\bzero\s+ending\s+inventory\b",
        r"\bending\s+no\s+inventory\b",
    )
    for pattern in positive_patterns:
        for match in re.finditer(pattern, lowered_text):
            if not (_is_negated_context(lowered_text, match.start()) or _is_negated_list_context(lowered_text, match.start())):
                return True
    return False


def _is_negated_context(lowered_text: str, start: int) -> bool:
    context = lowered_text[max(0, start - 36) : start]
    return bool(re.search(r"\b(?:no|not|without|never|neither|nor|forbid|forbidden|prohibit|prohibited)\b", context))


def _is_negated_list_context(lowered_text: str, start: int) -> bool:
    context = lowered_text[max(0, start - 120) : start]
    sentence_tail = re.split(r"[.;:!?]", context)[-1]
    negation_match = list(re.finditer(r"\b(?:no|without)\b", sentence_tail))
    if not negation_match:
        return False
    negated_segment = sentence_tail[negation_match[-1].start() :]
    if not re.search(r"[,/]|(?:\bor\b)|(?:\band\b)", negated_segment):
        return False
    positive_action_after_negation = re.search(
        r"\b(?:may|can|could|must|should|will)\s+(?:purchase|buy|add|expand)\b|\b(?:purchase|buy|add|expand)\b",
        negated_segment,
    )
    return positive_action_after_negation is None


def render_nl_report(accepted_count: int, rejected_count: int, rejection_reasons: dict[str, int]) -> str:
    lines = [
        "# NL Quality Report",
        "",
        f"- Accepted: {accepted_count}",
        f"- Rejected: {rejected_count}",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(rejection_reasons.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count} |")
    return "\n".join(lines) + "\n"


def _business_context_reasons(row: dict[str, Any]) -> list[str]:
    statement = str(row.get("problem_statement") or "")
    background = str(row.get("problem_background") or "")
    combined = f"{background}\n{statement}"
    words = set(re.findall(r"[A-Za-z][A-Za-z_-]{2,}", combined.lower()))
    metadata = row.get("llm_metadata") or {}
    structured = row.get("structured_problem_data") or {}

    organization_signal = bool(words & _organization_words()) or bool(metadata.get("organization_profile_id"))
    trigger_signal = bool(metadata.get("business_trigger")) or _contains_business_trigger(combined)
    decision_signal = bool(words & _decision_words())
    unit_or_entity_signal = (
        bool(words & _domain_words())
        or bool(structured.get("entities"))
        or bool(structured.get("units"))
        or _has_structured_entity_keys(structured)
    )

    if organization_signal and trigger_signal and decision_signal and unit_or_entity_signal:
        return []

    # Do not over-penalize strong scenario-guided rows that already contain a
    # concrete stakeholder and decision, even if the trigger wording is subtle.
    if metadata.get("scenario_id") and organization_signal and decision_signal and unit_or_entity_signal:
        return []

    reasons: list[str] = []
    if not organization_signal:
        reasons.append("MISSING_ORGANIZATION")
    if not trigger_signal:
        reasons.append("MISSING_BUSINESS_TRIGGER")
    if not decision_signal:
        reasons.append("MISSING_DECISION_CONTEXT")
    if not unit_or_entity_signal:
        reasons.append("GENERIC_MATH_ONLY")
    return reasons or ["GENERIC_MATH_ONLY"]


def _organization_words() -> set[str]:
    return {
        "agency",
        "airline",
        "bank",
        "brand",
        "campus",
        "carrier",
        "city",
        "clinic",
        "commission",
        "company",
        "coordinator",
        "department",
        "dispatcher",
        "distributor",
        "enterprise",
        "factory",
        "farm",
        "firm",
        "hospital",
        "laboratory",
        "manager",
        "manufacturer",
        "network",
        "office",
        "operator",
        "planner",
        "plant",
        "platform",
        "provider",
        "retailer",
        "school",
        "service",
        "team",
        "utility",
        "warehouse",
        "workshop",
    }


def _domain_words() -> set[str]:
    return {
        "asset",
        "assets",
        "budget",
        "car",
        "cars",
        "capacity",
        "containers",
        "cost",
        "costs",
        "customer",
        "customers",
        "demand",
        "district",
        "districts",
        "facility",
        "facilities",
        "inventory",
        "item",
        "items",
        "jobs",
        "kit",
        "kits",
        "machine",
        "machines",
        "neighborhood",
        "neighborhoods",
        "nutrient",
        "order",
        "orders",
        "parcel",
        "parcels",
        "participant",
        "participants",
        "plant",
        "production",
        "profit",
        "profits",
        "provider",
        "providers",
        "project",
        "projects",
        "request",
        "requests",
        "region",
        "regions",
        "resource",
        "resources",
        "route",
        "routes",
        "schedule",
        "shift",
        "shipment",
        "shipments",
        "site",
        "sites",
        "staff",
        "supplier",
        "suppliers",
        "task",
        "tasks",
        "truck",
        "trucks",
        "unit",
        "units",
        "vendor",
        "vendors",
        "vehicle",
        "vehicles",
        "warehouse",
        "warehouses",
        "worker",
        "workers",
        "zone",
        "zones",
        "category",
        "categories",
    }


def _has_structured_entity_keys(structured: Any) -> bool:
    if not isinstance(structured, dict):
        return False
    entity_keys = {
        "aircraft",
        "cars",
        "customers",
        "facilities",
        "foods",
        "ingredients",
        "items",
        "jobs",
        "machines",
        "nodes",
        "participants",
        "periods",
        "plants",
        "products",
        "projects",
        "routes",
        "servers",
        "sites",
        "tasks",
        "trucks",
        "vehicles",
        "warehouses",
        "workloads",
        "zones",
    }
    if any(key in structured and structured.get(key) not in (None, [], {}) for key in entity_keys):
        return True
    sets = structured.get("sets")
    if isinstance(sets, dict):
        return any(key in sets and sets.get(key) not in (None, [], {}) for key in entity_keys)
    return False


def _decision_words() -> set[str]:
    return {
        "allocate",
        "allocates",
        "allocated",
        "allocating",
        "allocation",
        "assemble",
        "assembles",
        "assembling",
        "assign",
        "assigned",
        "choose",
        "chooses",
        "commit",
        "commits",
        "committing",
        "decide",
        "decides",
        "deciding",
        "decision",
        "design",
        "designs",
        "designing",
        "determine",
        "determines",
        "formulate",
        "formulates",
        "formulating",
        "identify",
        "identifies",
        "identifying",
        "load",
        "loaded",
        "open",
        "opened",
        "plan",
        "plans",
        "produce",
        "produces",
        "route",
        "routes",
        "reschedule",
        "reschedules",
        "schedule",
        "schedules",
        "select",
        "selected",
        "sequence",
        "sequences",
        "sequencing",
        "serve",
        "served",
        "sell",
        "selling",
        "ship",
        "ships",
        "shipping",
    }


def _contains_business_trigger(text: str) -> bool:
    lowered = text.lower()
    trigger_terms = {
        "after",
        "before",
        "during",
        "weekly",
        "quarterly",
        "seasonal",
        "planning cycle",
        "budget",
        "demand surge",
        "forecast",
        "capacity shortage",
        "disruption",
        "service-level",
        "cutoff",
        "review",
        "rollout",
        "contract",
        "outage",
        "replenishment",
        "recovery",
    }
    return any(term in lowered for term in trigger_terms)


def _has_business_context(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    domain_words = {
        "company",
        "factory",
        "warehouse",
        "transport",
        "production",
        "planning",
        "resource",
        "budget",
        "demand",
        "schedule",
        "project",
        "facility",
        "inventory",
        "customer",
        "supplier",
    }
    return bool(set(words) & domain_words)


def _number_coverage(row: dict[str, Any]) -> float | None:
    text_numbers = set(_canonical_numbers(str(row.get("problem_statement") or "")))
    compact_numbers = _source_compact_required_numbers(row)
    if compact_numbers:
        return len(text_numbers & compact_numbers) / len(compact_numbers)
    source_text = str(row.get("source_math_formula") or "")
    source_numbers = set(_canonical_numbers(source_text))
    if _is_trivial_formula_number_set(source_numbers):
        lp_numbers = set(_canonical_numbers(str(row.get("source_lp_text") or "")))
        if lp_numbers:
            source_numbers = lp_numbers
    if _is_trivial_formula_number_set(source_numbers):
        return None
    if len(source_numbers) > 80:
        return None
    if not source_numbers:
        return None
    return len(text_numbers & source_numbers) / len(source_numbers)


def _is_trivial_formula_number_set(numbers: set[str]) -> bool:
    return not numbers or numbers <= {"-1", "0", "-0", "1"} or len(numbers) <= 1


def _source_compact_data_fidelity_reasons(row: dict[str, Any]) -> list[str]:
    """Reject rows that drop source-fact-card numbers for drift-prone generators."""
    if (row.get("llm_metadata") or {}).get("mock"):
        return []
    generator_id = str(row.get("generator_id") or "").lower()
    if not any(
        pattern in generator_id
        for pattern in (
            "optmath_clsp_expand_capacity",
            "optmath_uncapacitatedlotsizing",
            "optmath_uncapacitatedlotsizingbacklogging",
            "uncapacitatedlotsizingbacklogging",
            "optmath_net1",
            "net1",
            "optmath_netasgn",
            "netasgn",
            "optmath_structure_based_assignment",
            "structure_based_assignment",
            "optmath_steel4",
            "steel4",
            "optmath_steel3",
            "steel3",
            "optmath_singlelevelsmallbucket",
            "singlelevelsmallbucket",
            "optmath_aircraftlanding",
            "aircraftlanding",
            "optmath_electrical_power",
            "electrical_power",
            "optmath_marketshare",
            "marketshare",
        )
    ):
        return []
    source_compact = row.get("source_compact_data") or {}
    is_backtranslation_artifact = bool(row.get("bt_id") or row.get("bt_status"))
    if not isinstance(source_compact, dict) or not source_compact.get("available"):
        if is_backtranslation_artifact and _source_compact_required_for_generator(generator_id):
            return ["MISSING_SOURCE_COMPACT_DATA"]
        return []
    if source_compact.get("truncated"):
        if is_backtranslation_artifact and _source_compact_required_for_generator(generator_id):
            return ["TRUNCATED_SOURCE_COMPACT_DATA"]
        return []
    source_value = source_compact.get("value")
    required_numbers = _required_source_numbers(generator_id, source_value)
    if not required_numbers:
        return []
    candidate_text = "\n".join(
        [
            str(row.get("problem_statement") or ""),
            json.dumps(row.get("structured_problem_data") or {}, ensure_ascii=False, sort_keys=True),
        ]
    )
    candidate_numbers = set(_canonical_numbers(candidate_text))
    missing = sorted(required_numbers - candidate_numbers, key=_numeric_sort_key)
    if missing:
        preview = ",".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f",...(+{len(missing) - 12})"
        return [f"MISSING_SOURCE_COMPACT_NUMBERS:{preview}{suffix}"]
    return []


def _source_compact_required_numbers(row: dict[str, Any]) -> set[str]:
    generator_id = str(row.get("generator_id") or "").lower()
    source_compact = row.get("source_compact_data") or {}
    if not isinstance(source_compact, dict) or not source_compact.get("available") or source_compact.get("truncated"):
        return set()
    return _required_source_numbers(generator_id, source_compact.get("value"))


def _source_compact_required_for_generator(generator_id: str) -> bool:
    return any(
        pattern in generator_id
        for pattern in (
            "optmath_clsp_expand_capacity",
            "optmath_uncapacitatedlotsizing",
            "optmath_uncapacitatedlotsizingbacklogging",
            "uncapacitatedlotsizingbacklogging",
            "optmath_net1",
            "net1",
            "optmath_netasgn",
            "netasgn",
            "optmath_structure_based_assignment",
            "structure_based_assignment",
            "optmath_steel4",
            "steel4",
            "optmath_singlelevelsmallbucket",
            "singlelevelsmallbucket",
            "optmath_aircraftlanding",
            "aircraftlanding",
            "optmath_electrical_power",
            "electrical_power",
            "optmath_marketshare",
            "marketshare",
            "optmath_steel3",
            "steel3",
        )
    )


def _required_source_numbers(generator_id: str, source_value: Any) -> set[str]:
    if not isinstance(source_value, dict):
        return set()
    if "optmath_clsp_expand_capacity" in generator_id:
        table = _nested_get(source_value, ["compact_lotsizing_tables", "period_demand_table"], [])
        capacity_table = _nested_get(source_value, ["compact_lotsizing_tables", "period_capacity_table"], [])
        consumption_table = _nested_get(source_value, ["compact_lotsizing_tables", "capacity_consumption_per_product"], [])
        fields = (
            "period_demand",
            "cumulative_demand_through_period",
            "setup_cost",
            "unit_production_cost",
            "unit_holding_cost",
            "capacity_consumption_per_unit",
            "period_capacity",
            "setup_big_m_remaining_demand",
        )
        values = _numbers_from_rows(table, fields)
        values |= _numbers_from_rows(capacity_table, ("capacity",))
        values |= _numbers_from_rows(consumption_table, ("capacity_consumption_per_unit",))
        return values
    if "uncapacitatedlotsizingbacklogging" in generator_id:
        table = _nested_get(source_value, ["compact_lotsizing_tables", "period_cost_table"], [])
        values = _numbers_from_rows(
            table,
            (
                "demand",
                "fixed_ordering_cost",
                "unit_order_cost",
                "unit_holding_cost",
                "unit_backlog_penalty",
            ),
        )
        total_demand = _nested_get(source_value, ["compact_lotsizing_tables", "total_demand_big_m"], None)
        values |= _canonical_number_set([total_demand])
        return values
    if "uncapacitatedlotsizing" in generator_id:
        table = _nested_get(source_value, ["compact_lotsizing_tables", "period_cost_table"], [])
        values = _numbers_from_rows(
            table,
            (
                "demand",
                "fixed_ordering_cost",
                "unit_order_cost",
                "unit_holding_cost",
            ),
        )
        total_demand = _nested_get(source_value, ["compact_lotsizing_tables", "total_demand_big_m"], None)
        values |= _canonical_number_set([total_demand])
        return values
    if "net1" in generator_id:
        tables = source_value.get("compact_network_flow_tables") or source_value.get("compact_net1_tables") or {}
        node_table = tables.get("node_balance_table") if isinstance(tables, dict) else []
        arc_table = tables.get("directed_arc_table") if isinstance(tables, dict) else []
        values = _numbers_from_rows(node_table, ("supply", "demand"))
        values |= _numbers_from_rows(arc_table, ("unit_arc_flow_cost", "arc_capacity"))
        return {value for value in values if value != "0"}
    if "netasgn" in generator_id:
        tables = source_value.get("compact_project_assignment_tables") or {}
        person_table = tables.get("person_supply_table") if isinstance(tables, dict) else []
        project_table = tables.get("project_demand_table") if isinstance(tables, dict) else []
        values = _numbers_from_rows(person_table, ("available_hours",))
        values |= _numbers_from_rows(project_table, ("required_hours",))
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["cost_per_hour_matrix", "rows"], []))
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["max_contribution_hours_matrix", "rows"], []))
        values |= _canonical_number_set([tables.get("total_supply_hours"), tables.get("total_demand_hours")])
        return {value for value in values if value != "0"}
    if "structure_based_assignment" in generator_id:
        tables = source_value.get("compact_structure_assignment_tables") or {}
        values = _canonical_number_set([tables.get("distance_threshold"), tables.get("required_assignment_count")])
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["assignment_cost_matrix", "rows"], []))
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["acid_compatibility_matrix", "rows"], []))
        for peak, related_peak in tables.get("noe_relation_pairs") or []:
            values |= _canonical_number_set([peak, related_peak])
        return values
    if "steel4" in generator_id or "steel3" in generator_id:
        tables = source_value.get("compact_steel_product_mix_tables") or {}
        values = _numbers_from_table_rows(_nested_get(tables, ["product_table", "rows"], []))
        values |= _numbers_from_table_rows(_nested_get(tables, ["stage_capacity_table", "rows"], []))
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["processing_hours_per_ton_matrix", "rows"], []))
        time_capacity = tables.get("time_capacity") if isinstance(tables, dict) else {}
        if isinstance(time_capacity, dict):
            values |= _canonical_number_set([time_capacity.get("available_hours")])
        return values
    if "aircraftlanding" in generator_id:
        tables = source_value.get("compact_aircraft_landing_tables") or {}
        time_table = tables.get("aircraft_time_penalty_table") if isinstance(tables, dict) else []
        matrix = tables.get("ordered_pair_separation_matrix") if isinstance(tables, dict) else {}
        values = _numbers_from_rows(
            time_table,
            (
                "earliest_landing",
                "target_landing",
                "latest_landing",
                "early_penalty_per_minute",
                "late_penalty_per_minute",
            ),
        )
        if isinstance(matrix, dict):
            values |= _numbers_from_matrix_rows(matrix.get("rows") or [])
        return values
    if "electrical_power" in generator_id:
        tables = source_value.get("compact_electrical_power_tables") or {}
        generator_rows = tables.get("generator_type_table") if isinstance(tables, dict) else []
        demand_rows = tables.get("period_demand_table") if isinstance(tables, dict) else []
        values = _numbers_from_rows(
            generator_rows,
            (
                "base_operating_cost_per_period",
                "per_mw_generation_cost",
                "startup_cost",
                "minimum_output_mw",
                "maximum_output_mw",
                "generators_available",
                "generators_on_initially",
            ),
        )
        values |= _numbers_from_rows(demand_rows, ("demand_mw", "reserve_required_capacity_mw"))
        return values
    if "marketshare" in generator_id:
        tables = source_value.get("compact_marketshare_tables") or {}
        demand_rows = tables.get("demand_table") if isinstance(tables, dict) else []
        profit_rows = tables.get("profit_table") if isinstance(tables, dict) else []
        values = _numbers_from_rows(demand_rows, ("demand",))
        values |= _numbers_from_rows(profit_rows, ("unit_profit",))
        if isinstance(tables, dict):
            values |= _numbers_from_matrix_rows(_nested_get(tables, ["demand_matrix", "rows"], []))
            values |= _numbers_from_matrix_rows(
                _nested_get(tables, ["unit_profit_matrices_by_company", "rows"], [])
            )
        return values
    if "singlelevelsmallbucket" in generator_id:
        tables = source_value.get("compact_lotsizing_tables") or {}
        values = _canonical_number_set(list((tables.get("global_costs") or {}).values()))
        values |= _numbers_from_table_rows(_nested_get(tables, ["item_cost_table", "rows"], []))
        values |= _numbers_from_table_rows(_nested_get(tables, ["machine_table", "rows"], []))
        values |= _numbers_from_matrix_rows(_nested_get(tables, ["demand_matrix", "rows"], []))
        return values
    return set()


def _numbers_from_rows(rows: Any, fields: tuple[str, ...]) -> set[str]:
    if not isinstance(rows, list):
        return set()
    values: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field in fields:
            values.append(row.get(field))
    return _canonical_number_set(values)


def _numbers_from_table_rows(rows: Any) -> set[str]:
    if not isinstance(rows, list):
        return set()
    values: list[Any] = []
    for row in rows:
        if isinstance(row, list):
            values.extend(row)
        elif isinstance(row, dict):
            values.extend(row.values())
        else:
            values.append(row)
    return _canonical_number_set(values)


def _numbers_from_matrix_rows(rows: Any) -> set[str]:
    if not isinstance(rows, list):
        return set()
    values: list[Any] = []
    for row in rows:
        if isinstance(row, dict):
            values.append(row.get("peak"))
            values.append(row.get("product"))
            values.append(row.get("item"))
            values.append(row.get("amino_acid"))
            values.extend(row.get("values") or [])
        elif isinstance(row, list):
            values.extend(row)
    return _canonical_number_set(values)


def _canonical_number_set(values: list[Any]) -> set[str]:
    numbers: set[str] = set()
    for value in values:
        if value is None:
            continue
        numbers.update(_canonical_numbers(str(value)))
    return numbers


def _nested_get(value: Any, path: list[str], default: Any) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _numeric_sort_key(value: str) -> tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (0.0, value)


def _canonical_numbers(text: str) -> list[str]:
    values: list[str] = []
    for match in re.findall(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?", text):
        try:
            value = float(match)
        except ValueError:
            continue
        if abs(value) > 1e8:
            continue
        if value == 0:
            value = 0.0
        values.append(f"{value:.8g}")
    return values
