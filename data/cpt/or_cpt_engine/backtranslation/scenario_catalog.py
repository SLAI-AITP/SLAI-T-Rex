from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from or_cpt_engine.backtranslation.scenario_variants import (
    DEFAULT_FAMILY_SCENARIO_LENSES,
    DEFAULT_VARIATION_AXES,
)

DEFAULT_SCENARIO_CATALOG_PATH = Path("engine_configs/scenario_catalog.yaml")

_BUILTIN_DEFAULT_CATALOG: dict[str, Any] = {
    "global": {
        "default_task_family": "general_operations_research",
        "common_scenarios": [
            {
                "scenario_id": "general_logistics_planning",
                "industry": "logistics",
                "background": "a planning team allocates limited resources across operational activities",
                "entities": ["activities", "resources", "capacity limits"],
                "units": ["units", "cost units"],
            },
            {
                "scenario_id": "general_budget_allocation",
                "industry": "budget allocation",
                "background": "an organization chooses among competing options under budget and policy constraints",
                "entities": ["options", "budgets", "benefits"],
                "units": ["dollars", "benefit points"],
            },
        ],
        "modeling_notes": [
            "Choose a realistic business setting that matches the source sets, coefficients, and constraints.",
            "Do not change the mathematical structure while adding business context.",
        ],
        "business_triggers": [
            "during a weekly planning cycle",
            "during a budget approval round",
            "after a demand forecast revision",
            "after a capacity shortage was identified",
            "during disruption recovery planning",
        ],
        "applicability": {
            "required_source_features": [
                "linear objective and constraints",
                "numeric coefficients preserved from the source instance",
            ],
            "incompatible_source_features": [
                "unstated constraints",
                "unprovided external files",
                "business rules not present in the source instance",
            ],
        },
    },
    "task_families": {},
    "generators": {},
}

DEFAULT_FAMILY_UNITS: dict[str, dict[str, str]] = {
    "assignment": {"decision": "assignments", "cost": "cost units", "objective": "total cost or total score"},
    "scheduling": {"decision": "scheduled jobs or time slots", "time": "hours or minutes", "objective": "total time, lateness, or cost"},
    "packing": {"decision": "selected items", "capacity": "kilograms, volume units, or budget units", "objective": "total value or total cost"},
    "cutting_stock": {"decision": "cutting patterns", "demand": "pieces", "length": "meters or material units", "objective": "waste or stock usage"},
    "blending": {"decision": "blend quantities", "quality": "quality units or percentages", "cost": "dollars", "objective": "total blend cost"},
    "diet": {"decision": "servings or ingredient quantities", "nutrient": "nutrient units", "cost": "dollars", "objective": "total cost"},
    "facility_location": {"decision": "opened facilities and served demand", "demand": "units", "capacity": "units", "cost": "dollars"},
    "covering": {"decision": "selected covering options", "coverage": "covered zones or requirements", "cost": "dollars"},
    "portfolio": {"decision": "investment or project allocation", "budget": "dollars", "risk": "risk units", "objective": "return or benefit"},
    "revenue_management": {"decision": "allocated capacity", "capacity": "seats, rooms, impressions, or units", "revenue": "dollars"},
    "marketing": {"decision": "campaign allocation", "budget": "dollars", "impact": "reach or lift units"},
    "contract_allocation": {"decision": "awarded volume", "capacity": "units", "cost": "dollars"},
    "supply_chain": {"decision": "production, shipment, or inventory quantities", "flow": "units", "cost": "dollars"},
    "network_design": {"decision": "activated arcs and routed flow", "flow": "units", "capacity": "units", "cost": "dollars"},
    "network_flow": {"decision": "arc flow", "flow": "units", "cost": "cost units"},
    "routing": {"decision": "route arcs or visits", "time": "minutes or hours", "distance": "kilometers or miles", "cost": "dollars"},
    "transportation": {"decision": "shipment quantity", "supply": "units", "demand": "units", "cost": "dollars per unit shipped"},
    "lot_sizing": {"decision": "production, order, inventory, or backlog quantity", "time": "periods", "cost": "dollars"},
    "production_planning": {"decision": "production quantity", "resource": "hours or resource units", "cost": "dollars"},
    "agriculture": {"decision": "acreage or resource allocation", "land": "acres", "resource": "hours, water units, or dollars"},
    "energy": {"decision": "generation or capacity quantity", "power": "MW or MWh", "cost": "dollars"},
    "workforce_deployment": {"decision": "personnel or team assignment", "staffing": "people or shifts", "cost": "cost units"},
    "dispersion": {"decision": "selected sites", "distance": "kilometers or miles", "objective": "separation or diversity score"},
    "general_linear_programming": {"decision": "activity level", "resource": "resource units", "cost": "cost units"},
}


def scenario_guidance_for_generator(
    generator_id: str | None,
    *,
    instance_id: str | None = None,
    candidate_index: int = 1,
    random_seed: int = 42,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    catalog = load_scenario_catalog(catalog_path)
    normalized_generator = (generator_id or "").lower()
    generator_config = _lookup_generator_config(catalog, normalized_generator)
    global_config = catalog.get("global") or {}
    task_family = (
        (generator_config or {}).get("task_family")
        or _infer_task_family(normalized_generator)
        or global_config.get("default_task_family")
        or "general_operations_research"
    )
    family_config = (catalog.get("task_families") or {}).get(task_family) or {}

    scenario_pool = _scenario_pool(
        generator_config=generator_config or {},
        family_config=family_config,
        global_config=global_config,
        task_family=task_family,
    )
    selected_scenario = _select_scenario(
        scenario_pool,
        generator_id=normalized_generator or "unknown_generator",
        instance_id=instance_id or "",
        candidate_index=candidate_index,
        random_seed=random_seed,
    )
    base_scenario = {
        key: value
        for key, value in selected_scenario.items()
        if not str(key).startswith("_")
    }
    base_scenario_id = str(base_scenario.get("scenario_id") or "default_general_scenario")
    base_scenario_source = selected_scenario.get("_scenario_source") or "global"
    scenario_variant = _select_scenario_variant(
        task_family=task_family,
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config or {},
        selected_scenario=selected_scenario,
        generator_id=normalized_generator or "unknown_generator",
        instance_id=instance_id or "",
        candidate_index=candidate_index,
        random_seed=random_seed,
    )
    selected_scenario = _apply_scenario_variant(base_scenario, scenario_variant)
    entity_suggestions = _merge_dicts(
        global_config.get("entity_suggestions") or {},
        family_config.get("entity_suggestions") or {},
        (generator_config or {}).get("entity_suggestions") or {},
        selected_scenario.get("entity_suggestions") or {},
    )
    business_triggers = _merge_lists(
        global_config.get("business_triggers"),
        family_config.get("business_triggers"),
        (generator_config or {}).get("business_triggers"),
        selected_scenario.get("business_triggers"),
        selected_scenario.get("business_trigger"),
        scenario_variant.get("business_triggers"),
        (scenario_variant.get("narrative_angle") or {}).get("business_triggers"),
    )
    selected_business_trigger = _select_text(
        business_triggers,
        generator_id=normalized_generator or "unknown_generator",
        instance_id=instance_id or "",
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="business_trigger",
    )
    return {
        "catalog_version": str(catalog.get("version") or "unknown"),
        "generator_id": generator_id,
        "task_family": task_family,
        "scenario_id": selected_scenario.get("scenario_id") or "default_general_scenario",
        "base_scenario_id": base_scenario_id,
        "scenario_variant_id": scenario_variant.get("scenario_variant_id"),
        "scenario_source": base_scenario_source,
        "scenario_variant": scenario_variant,
        "selected_scenario": {
            key: value
            for key, value in selected_scenario.items()
            if not str(key).startswith("_")
        },
        "selected_business_trigger": selected_business_trigger,
        "units": _merge_unit_specs(
            DEFAULT_FAMILY_UNITS.get(task_family),
            global_config.get("units"),
            family_config.get("units"),
            (generator_config or {}).get("units"),
            selected_scenario.get("units"),
        ),
        "entity_suggestions": entity_suggestions,
        "modeling_notes": _merge_lists(
            global_config.get("modeling_notes"),
            family_config.get("modeling_notes"),
            (generator_config or {}).get("modeling_notes"),
            selected_scenario.get("modeling_notes"),
        ),
        "required_parameter_presentation": _merge_lists(
            family_config.get("required_parameter_presentation"),
            (generator_config or {}).get("required_parameter_presentation"),
            selected_scenario.get("required_parameter_presentation"),
        ),
        "forbidden_misframings": _merge_lists(
            family_config.get("forbidden_misframings"),
            (generator_config or {}).get("forbidden_misframings"),
            selected_scenario.get("forbidden_misframings"),
        ),
        "applicability": _merge_applicability(
            global_config.get("applicability"),
            family_config.get("applicability"),
            (generator_config or {}).get("applicability"),
            selected_scenario.get("applicability"),
        ),
        "available_scenario_count": len(scenario_pool),
        "available_scenario_variant_count": _scenario_variant_count(
            scenario_pool=scenario_pool,
            task_family=task_family,
            global_config=global_config,
            family_config=family_config,
            generator_config=generator_config or {},
            selected_scenario=base_scenario,
        ),
        "available_business_trigger_count": len(business_triggers),
    }


@lru_cache(maxsize=8)
def load_scenario_catalog(catalog_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(catalog_path or DEFAULT_SCENARIO_CATALOG_PATH)
    if not path.exists():
        return _BUILTIN_DEFAULT_CATALOG
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else _BUILTIN_DEFAULT_CATALOG


def _lookup_generator_config(catalog: dict[str, Any], normalized_generator: str) -> dict[str, Any] | None:
    generators = catalog.get("generators") or {}
    if normalized_generator in generators:
        return generators[normalized_generator] or {}
    for key, config in generators.items():
        if str(key).lower() in normalized_generator:
            return config or {}
    return None


def _scenario_pool(
    *,
    generator_config: dict[str, Any],
    family_config: dict[str, Any],
    global_config: dict[str, Any],
    task_family: str,
) -> list[dict[str, Any]]:
    generator_pool = _with_source(generator_config.get("scenarios"), "generator_specific")
    if generator_pool:
        return generator_pool
    family_pool = _with_source(family_config.get("common_scenarios"), "task_family")
    if family_pool:
        return family_pool
    global_pool = _with_source(global_config.get("common_scenarios"), "global")
    if global_pool:
        return global_pool
    return _with_source(_BUILTIN_DEFAULT_CATALOG["global"]["common_scenarios"], "builtin")


def _with_source(rows: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            copied = dict(row)
            copied["_scenario_source"] = source
            result.append(copied)
    return result


def _select_scenario(
    scenario_pool: list[dict[str, Any]],
    *,
    generator_id: str,
    instance_id: str,
    candidate_index: int,
    random_seed: int,
) -> dict[str, Any]:
    if not scenario_pool:
        return {}
    digest = hashlib.sha1(f"{random_seed}:{generator_id}:{instance_id}:{candidate_index}".encode("utf-8")).hexdigest()
    index = int(digest[:12], 16) % len(scenario_pool)
    return scenario_pool[index]


def _select_text(
    values: list[Any],
    *,
    generator_id: str,
    instance_id: str,
    candidate_index: int,
    random_seed: int,
    salt: str,
) -> str | None:
    text_values = [str(value).strip() for value in values if str(value).strip()]
    if not text_values:
        return None
    digest = hashlib.sha1(
        f"{random_seed}:{salt}:{generator_id}:{instance_id}:{candidate_index}".encode("utf-8")
    ).hexdigest()
    index = int(digest[:12], 16) % len(text_values)
    return text_values[index]


def _select_scenario_variant(
    *,
    task_family: str,
    global_config: dict[str, Any],
    family_config: dict[str, Any],
    generator_config: dict[str, Any],
    selected_scenario: dict[str, Any],
    generator_id: str,
    instance_id: str,
    candidate_index: int,
    random_seed: int,
) -> dict[str, Any]:
    lens_pool = _scenario_lens_pool(
        task_family=task_family,
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config,
        selected_scenario=selected_scenario,
    )
    narrative_angles = _variation_axis_pool(
        "narrative_angles",
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config,
        selected_scenario=selected_scenario,
    )
    organization_profiles = _variation_axis_pool(
        "organization_profiles",
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config,
        selected_scenario=selected_scenario,
    )
    planning_horizons = _variation_axis_pool(
        "planning_horizons",
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config,
        selected_scenario=selected_scenario,
    )
    entity_naming_styles = _variation_axis_pool(
        "entity_naming_styles",
        global_config=global_config,
        family_config=family_config,
        generator_config=generator_config,
        selected_scenario=selected_scenario,
    )
    industry_lens = _select_record(
        lens_pool,
        generator_id=generator_id,
        instance_id=instance_id,
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="industry_lens",
    )
    narrative_angle = _select_record(
        narrative_angles,
        generator_id=generator_id,
        instance_id=instance_id,
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="narrative_angle",
    )
    organization_profile = _select_record(
        organization_profiles,
        generator_id=generator_id,
        instance_id=instance_id,
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="organization_profile",
    )
    planning_horizon = _select_record(
        planning_horizons,
        generator_id=generator_id,
        instance_id=instance_id,
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="planning_horizon",
    )
    entity_naming_style = _select_record(
        entity_naming_styles,
        generator_id=generator_id,
        instance_id=instance_id,
        candidate_index=candidate_index,
        random_seed=random_seed,
        salt="entity_naming_style",
    )
    scenario_variant_id = "__".join(
        [
            _record_id(industry_lens, "lens"),
            _record_id(narrative_angle, "angle"),
            _record_id(organization_profile, "org"),
            _record_id(planning_horizon, "horizon"),
        ]
    )
    return {
        "scenario_variant_id": scenario_variant_id,
        "industry_lens": industry_lens,
        "narrative_angle": narrative_angle,
        "organization_profile": organization_profile,
        "planning_horizon": planning_horizon,
        "entity_naming_style": entity_naming_style,
        "industry_lens_id": _record_id(industry_lens, "lens"),
        "narrative_angle_id": _record_id(narrative_angle, "angle"),
        "organization_profile_id": _record_id(organization_profile, "org"),
        "planning_horizon_id": _record_id(planning_horizon, "horizon"),
        "entity_naming_style_id": _record_id(entity_naming_style, "style"),
        "business_triggers": _merge_lists(
            industry_lens.get("business_triggers"),
            narrative_angle.get("business_triggers"),
        ),
        "available_lens_count": len(lens_pool),
        "available_narrative_angle_count": len(narrative_angles),
        "available_organization_profile_count": len(organization_profiles),
        "available_planning_horizon_count": len(planning_horizons),
        "available_entity_naming_style_count": len(entity_naming_styles),
    }


def _apply_scenario_variant(base_scenario: dict[str, Any], scenario_variant: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(base_scenario)
    base_scenario_id = str(base_scenario.get("scenario_id") or "default_general_scenario")
    variant_id = str(scenario_variant.get("scenario_variant_id") or "default_variant")
    industry_lens = scenario_variant.get("industry_lens") or {}
    narrative_angle = scenario_variant.get("narrative_angle") or {}
    organization_profile = scenario_variant.get("organization_profile") or {}
    planning_horizon = scenario_variant.get("planning_horizon") or {}
    entity_naming_style = scenario_variant.get("entity_naming_style") or {}
    primary_context = industry_lens.get("context") or base_scenario.get("background")
    background_parts = _merge_lists(
        primary_context,
        narrative_angle.get("description"),
        organization_profile.get("description"),
        planning_horizon.get("description"),
    )
    enriched.update(
        {
            "scenario_id": f"{base_scenario_id}__{variant_id}",
            "base_scenario_id": base_scenario_id,
            "scenario_variant_id": variant_id,
            "industry": industry_lens.get("industry") or base_scenario.get("industry"),
            "base_background": base_scenario.get("background"),
            "background": " ".join(str(part).strip() for part in background_parts if str(part).strip()),
            "entities": _merge_lists(industry_lens.get("entities") or base_scenario.get("entities")),
            "units": _merge_lists(industry_lens.get("units") or base_scenario.get("units")),
            "variation_instructions": [
                f"Base mathematical wrapper: {base_scenario_id}",
                f"Industry lens: {industry_lens.get('context')}",
                f"Narrative angle: {narrative_angle.get('description')}",
                f"Organization profile: {organization_profile.get('description')}",
                f"Planning horizon: {planning_horizon.get('description')}",
                f"Entity naming style: {entity_naming_style.get('description')}",
            ],
        }
    )
    return enriched


def _scenario_variant_count(
    *,
    scenario_pool: list[dict[str, Any]],
    task_family: str,
    global_config: dict[str, Any],
    family_config: dict[str, Any],
    generator_config: dict[str, Any],
    selected_scenario: dict[str, Any],
) -> int:
    lens_count = len(
        _scenario_lens_pool(
            task_family=task_family,
            global_config=global_config,
            family_config=family_config,
            generator_config=generator_config,
            selected_scenario=selected_scenario,
        )
    )
    angle_count = len(
        _variation_axis_pool(
            "narrative_angles",
            global_config=global_config,
            family_config=family_config,
            generator_config=generator_config,
            selected_scenario=selected_scenario,
        )
    )
    organization_count = len(
        _variation_axis_pool(
            "organization_profiles",
            global_config=global_config,
            family_config=family_config,
            generator_config=generator_config,
            selected_scenario=selected_scenario,
        )
    )
    return max(1, len(scenario_pool)) * max(1, lens_count) * max(1, angle_count) * max(1, organization_count)


def _scenario_lens_pool(
    *,
    task_family: str,
    global_config: dict[str, Any],
    family_config: dict[str, Any],
    generator_config: dict[str, Any],
    selected_scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    policy = generator_config.get("variant_policy") or {}
    if policy.get("use_only_generator_scenario_lenses") and generator_config.get("scenario_lenses"):
        return _merge_records(
            generator_config.get("scenario_lenses"),
            selected_scenario.get("scenario_lenses"),
            id_keys=("lens_id", "scenario_lens_id", "id"),
        )
    fallback = DEFAULT_FAMILY_SCENARIO_LENSES.get(task_family) or DEFAULT_FAMILY_SCENARIO_LENSES["general_linear_programming"]
    return _merge_records(
        fallback,
        (global_config.get("variation_axes") or {}).get("scenario_lenses"),
        family_config.get("scenario_lenses"),
        generator_config.get("scenario_lenses"),
        selected_scenario.get("scenario_lenses"),
        id_keys=("lens_id", "scenario_lens_id", "id"),
    )


def _variation_axis_pool(
    axis_name: str,
    *,
    global_config: dict[str, Any],
    family_config: dict[str, Any],
    generator_config: dict[str, Any],
    selected_scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    policy = generator_config.get("variant_policy") or {}
    only_axes = {
        str(value)
        for value in (
            policy.get("use_only_generator_variation_axes")
            or policy.get("use_only_generator_axes")
            or []
        )
    }
    if axis_name in only_axes and (generator_config.get("variation_axes") or {}).get(axis_name):
        return _merge_records(
            (generator_config.get("variation_axes") or {}).get(axis_name),
            (selected_scenario.get("variation_axes") or {}).get(axis_name),
            id_keys=("lens_id", "scenario_lens_id", "angle_id", "profile_id", "horizon_id", "style_id", "id"),
        )
    fallback = DEFAULT_VARIATION_AXES.get(axis_name) or []
    return _merge_records(
        fallback,
        (global_config.get("variation_axes") or {}).get(axis_name),
        (family_config.get("variation_axes") or {}).get(axis_name),
        (generator_config.get("variation_axes") or {}).get(axis_name),
        (selected_scenario.get("variation_axes") or {}).get(axis_name),
        id_keys=("lens_id", "angle_id", "profile_id", "horizon_id", "style_id", "id"),
    )


def _merge_records(*values: Any, id_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            rows = [value]
        elif isinstance(value, list):
            rows = value
        else:
            continue
        for row in rows:
            normalized = _normalize_record(row)
            if not normalized:
                continue
            marker = next((str(normalized.get(key)) for key in id_keys if normalized.get(key)), repr(normalized))
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(normalized)
    return merged


def _normalize_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"id": _slug(value), "description": value.strip()}
    return {}


def _select_record(
    values: list[dict[str, Any]],
    *,
    generator_id: str,
    instance_id: str,
    candidate_index: int,
    random_seed: int,
    salt: str,
) -> dict[str, Any]:
    if not values:
        return {}
    digest = hashlib.sha1(
        f"{random_seed}:{salt}:{generator_id}:{instance_id}:{candidate_index}".encode("utf-8")
    ).hexdigest()
    return values[int(digest[:12], 16) % len(values)]


def _record_id(record: dict[str, Any], prefix: str) -> str:
    for key in ("lens_id", "angle_id", "profile_id", "horizon_id", "style_id", "id"):
        value = record.get(key)
        if value:
            return _slug(str(value))
    text = str(record.get("description") or record.get("context") or record.get("industry") or prefix)
    return f"{prefix}_{_slug(text)[:40]}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def _infer_task_family(generator_id: str) -> str | None:
    rules = [
        ("aircraftassignment", "assignment"),
        ("carselection", "assignment"),
        ("structure_based_assignment", "assignment"),
        ("netasgn", "assignment"),
        ("team_formulation", "assignment"),
        ("aircraftlanding", "scheduling"),
        ("flowshop", "scheduling"),
        ("jopshop", "scheduling"),
        ("scheduling", "scheduling"),
        ("multi_factory_schedule", "scheduling"),
        ("binpacking", "packing"),
        ("knapsack", "packing"),
        ("cut", "cutting_stock"),
        ("blending", "blending"),
        ("diet", "diet"),
        ("dietu", "diet"),
        ("cflp", "facility_location"),
        ("facility_location", "facility_location"),
        ("scooter_location", "facility_location"),
        ("cell_tower", "facility_location"),
        ("p_dispersion", "facility_location"),
        ("setcover", "covering"),
        ("multisetcover", "covering"),
        ("portfolio", "portfolio"),
        ("revenue", "revenue_management"),
        ("marketshare", "marketing"),
        ("contractallocation", "contract_allocation"),
        ("supplychain", "supply_chain"),
        ("mcnd", "network_design"),
        ("netmcol", "network_design"),
        ("netthru", "network_design"),
        ("net1", "network_flow"),
        ("shortest_path", "network_flow"),
        ("tsp", "routing"),
        ("vrptw", "routing"),
        ("fleet_routing", "routing"),
        ("transp", "transportation"),
        ("nltrans", "transportation"),
        ("lotsizing", "lot_sizing"),
        ("clsp", "lot_sizing"),
        ("factory_planning", "production_planning"),
        ("prod", "production_planning"),
        ("steel", "production_planning"),
        ("farmplanning", "agriculture"),
        ("electrical_power", "energy"),
        ("military_personnel", "workforce_deployment"),
        ("maxisum", "dispersion"),
        ("multi", "general_linear_programming"),
    ]
    for needle, family in rules:
        if needle in generator_id:
            return family
    return None


def _merge_lists(*values: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str):
            rows = [value]
        elif isinstance(value, list):
            rows = value
        else:
            continue
        for item in rows:
            marker = repr(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
    return merged


def _merge_applicability(*values: Any) -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key in ("required_source_features", "compatible_source_features", "incompatible_source_features"):
            merged[key] = _merge_lists(merged.get(key), value.get(key))
    return {key: rows for key, rows in merged.items() if rows}


def _merge_unit_specs(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
        elif isinstance(value, list):
            merged.setdefault("generic", value)
        elif isinstance(value, str) and value.strip():
            merged.setdefault("generic", [value.strip()])
    return merged


def _merge_dicts(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged
