from __future__ import annotations

from typing import Any


DEFAULT_CONTRACT_ID = "default_linear_or_contract"


def resolve_family_contract(row: dict[str, Any], contract_config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the most specific modeling contract for a seed/candidate row."""
    config = contract_config or {}
    families = _as_dict(config.get("families"))
    generator_overrides = _as_dict(config.get("generator_overrides"))
    aliases = _as_dict(config.get("generator_aliases"))
    defaults = _as_dict(config.get("defaults"))

    generator_id = _generator_id(row)
    task_family = _task_family(row)
    sub_family = str(row.get("sub_family") or "").strip()
    alias_family = _find_alias_family(generator_id, aliases)

    candidates = [
        f"generator:{generator_id}",
        generator_id,
        f"{task_family}:{sub_family}" if task_family and sub_family else "",
        task_family,
        alias_family,
        DEFAULT_CONTRACT_ID,
    ]
    selected_id = ""
    selected_contract: dict[str, Any] = {}
    selected_source = "none"

    for candidate in candidates:
        if not candidate:
            continue
        if candidate in generator_overrides:
            selected_id = candidate
            selected_contract = _extract_contract(generator_overrides[candidate])
            selected_source = "generator_override"
            break
        if candidate in families:
            selected_id = candidate
            selected_contract = _extract_contract(families[candidate])
            selected_source = "family"
            break
        if candidate in defaults:
            selected_id = candidate
            selected_contract = _extract_contract(defaults[candidate])
            selected_source = "default"
            break

    if not selected_contract:
        selected_id = DEFAULT_CONTRACT_ID
        selected_contract = _extract_contract(defaults.get(DEFAULT_CONTRACT_ID, {}))
        selected_source = "default"

    tags = _contract_tags(selected_contract)
    coverage = "high" if selected_source in {"family", "generator_override"} else "low"
    return {
        "version": str(config.get("version") or "unknown"),
        "contract_id": selected_id or DEFAULT_CONTRACT_ID,
        "contract_source": selected_source,
        "contract_coverage": coverage,
        "task_family": task_family or alias_family or "unknown",
        "sub_family": sub_family or None,
        "generator_id": generator_id or None,
        "contract_tags": tags,
        "answer_contract": selected_contract,
    }


def render_family_contract_inventory(contract_config: dict[str, Any] | None) -> str:
    config = contract_config or {}
    lines = [
        "# Family Contract Inventory",
        "",
        f"- Version: {config.get('version', 'unknown')}",
        "",
        "| Scope | Contract ID | Model Type | Objective Sense | Required Constraints | Tags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scope, bucket in (("family", _as_dict(config.get("families"))), ("generator", _as_dict(config.get("generator_overrides"))), ("default", _as_dict(config.get("defaults")))):
        for contract_id, spec in sorted(bucket.items()):
            contract = _extract_contract(spec)
            lines.append(
                "| {scope} | `{contract_id}` | {model_type} | {objective} | {constraints} | {tags} |".format(
                    scope=scope,
                    contract_id=contract_id,
                    model_type=_cell(contract.get("model_type")),
                    objective=_cell(contract.get("objective_sense_options")),
                    constraints=_cell(contract.get("required_constraints")),
                    tags=_cell(_contract_tags(contract)),
                )
            )
    return "\n".join(lines) + "\n"


def _task_family(row: dict[str, Any]) -> str:
    source_metadata = row.get("source_metadata") or {}
    llm_metadata = row.get("llm_metadata") or {}
    for source in (row, source_metadata, llm_metadata):
        value = source.get("task_family") if isinstance(source, dict) else None
        if value:
            return str(value).strip()
    return ""


def _generator_id(row: dict[str, Any]) -> str:
    source_metadata = row.get("source_metadata") or {}
    llm_metadata = row.get("llm_metadata") or {}
    for source in (row, source_metadata, llm_metadata):
        value = source.get("generator_id") if isinstance(source, dict) else None
        if value:
            return str(value).strip()
    return ""


def _find_alias_family(generator_id: str, aliases: dict[str, Any]) -> str:
    lowered = generator_id.lower()
    for family, patterns in aliases.items():
        for pattern in _as_list(patterns):
            if str(pattern).lower() in lowered:
                return str(family)
    return ""


def _extract_contract(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    answer_contract = spec.get("answer_contract")
    if isinstance(answer_contract, dict):
        return dict(answer_contract)
    return dict(spec)


def _contract_tags(contract: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in (
        "model_type",
        "objective_sense_options",
        "required_variable_families",
        "required_constraints",
        "high_risk_failures",
    ):
        value = contract.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            tags.extend(str(item) for item in value)
        else:
            tags.append(str(value))
    return _dedupe_keep_order([tag.strip() for tag in tags if tag and str(tag).strip()])


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)
