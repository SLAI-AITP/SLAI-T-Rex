from __future__ import annotations

import re


SEMANTIC_COHERENCE_RULES: dict[str, tuple[str, ...]] = {
    "optmath_aircraftassignment": (
        "aircraft landing sequencing",
        "tail rotation",
        "vehicle routing",
        "time windows",
    ),
    "optmath_aircraftlanding": (
        "fleet assignment",
        "tail rotation",
        "vehicle routing",
        "cargo allocation",
    ),
    "optmath_carselection": (
        "technician",
        "technicians",
        "job matching",
        "service job",
        "service jobs",
        "inspection job",
        "inspection jobs",
        "field-service workforce",
        "worker assignment",
        "workers to shifts",
        "staff to roles",
        "shift scheduling",
        "staffing",
    ),
    "optmath_clsp_expand_capacity": (
        "backlog",
        "backlogs",
        "backorder",
        "backorders",
        "lost sales",
        "unmet demand",
        "unmet-demand slack",
        "shortage slack",
        "shortage variable",
        "shortage cost",
        "shortage penalty",
        "capacity expansion variable",
    ),
    "optmath_net1": (
        "vehicle routing",
        "time window",
        "time windows",
        "subtour",
        "facility opening",
    ),
    "optmath_nltrans": (
        "vehicle routing",
        "time window",
        "time windows",
        "subtour",
        "facility opening",
    ),
    "optmath_singlelevelsmallbucket": (
        "unit production cost",
        "per-unit production cost",
        "production cost is",
        "production costs are",
        "cost per unit produced",
        "per unit produced",
        "each unit of production",
        "material cost",
        "sales revenue",
    ),
    "optmath_steel4": (
        "inventory carryover",
        "backlog",
        "backlogs",
        "storage variable",
        "time period inventory",
    ),
    "optmath_structure_based_assignment": (
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
        "vehicle routing",
        "gate assignment",
        "crew assignment",
    ),
    "optmath_the_shortest_path_problem": (
        "facility opening",
        "vehicle capacity",
        "time window",
        "time windows",
        "subtour",
    ),
    "optmath_tsp": (
        "vehicle capacity",
        "time window",
        "time windows",
        "multiple vehicles",
        "fleet sizing",
    ),
    "optmath_uncapacitatedlotsizing": (
        "backlog",
        "backlogs",
        "backorder",
        "backorders",
        "lost sales",
        "unmet demand",
        "unmet-demand slack",
        "shortage slack",
        "shortage variable",
        "shortage cost",
        "shortage penalty",
        "production capacity",
    ),
}


def semantic_coherence_issues(generator_id: str, text: str) -> list[str]:
    forbidden_terms = SEMANTIC_COHERENCE_RULES.get(str(generator_id or ""), ())
    if not forbidden_terms:
        return []
    lowered = str(text or "").lower()
    issues: list[str] = []
    for term in forbidden_terms:
        if contains_unnegated_phrase(lowered, term):
            issues.append(f"SEMANTIC_FORBIDDEN_TERM:{term}")
    return issues


def contains_unnegated_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = phrase.lower().strip()
    if not normalized_phrase:
        return False
    pattern = re.escape(normalized_phrase).replace(r"\ ", r"[\s\-]+")
    for match in re.finditer(rf"(?<![a-z0-9_]){pattern}(?![a-z0-9_])", text):
        prefix = text[max(0, match.start() - 90) : match.start()]
        suffix = text[match.end() : match.end() + 60]
        sentence_prefix = re.split(r"[.;:\n]", prefix)[-1]
        if re.search(
            r"\b(do not|does not|without|no|not|forbid|forbids|forbidden|avoid|avoids)\b[\w\s,\-()]{0,80}$",
            sentence_prefix,
        ):
            continue
        if re.match(
            r"[\w\s,\-()]{0,36}\b(is|are|was|were|must be)?\s*"
            r"(not allowed|not permitted|not present|not included|not used|prohibited|forbidden|excluded)\b",
            suffix,
        ):
            continue
        return True
    return False
