from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from or_cpt_engine.utils.io import read_jsonl, write_text


PROFILE_VERSION = "v1.1.0"


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

MAIN_GENERATORS = {
    "optmath_facility_location",
    "optmath_cflp",
    "optmath_supplychain",
    "optmath_transp",
    "optmath_netasgn",
    "optmath_netthru",
    "optmath_setcover",
    "optmath_multisetcover",
    "optmath_schedulingproblem",
    "optmath_jopshop",
    "optmath_flowshop_2",
    "optmath_factory_planning_problem",
    "optmath_uncapacitatedlotsizing",
    "optmath_uncapacitatedlotsizingbacklogging",
    "optmath_vrptw",
    "optmath_fleet_routing",
    "optmath_portfolio",
    "optmath_revenue_management",
}

CAUTIOUS_GENERATOR_NEEDLES = (
    "multi",
    "net1",
    "nltrans",
    "prod",
    "dietu",
    "singlelevelsmallbucket",
)

FAMILY_CONCEPTS: dict[str, list[str]] = {
    "assignment": ["binary_assignment", "eligibility_constraint", "capacity_assignment", "exclusivity"],
    "scheduling": ["time_indexed_decision", "sequencing", "capacity_constraint", "precedence_or_time_window"],
    "packing": ["binary_selection", "capacity_constraint", "value_or_cost_tradeoff"],
    "cutting_stock": ["pattern_selection", "demand_satisfaction", "waste_minimization"],
    "blending": ["continuous_blend", "quality_bounds", "composition_constraint", "least_cost_mix"],
    "diet": ["continuous_quantity", "nutrient_bounds", "ingredient_cost", "lower_upper_requirements"],
    "facility_location": ["binary_opening_decision", "customer_assignment", "fixed_cost", "capacity_constraint", "linking_constraint"],
    "covering": ["binary_selection", "coverage_matrix", "set_covering_requirement"],
    "portfolio": ["budget_constraint", "risk_constraint", "return_objective", "allocation_or_selection"],
    "revenue_management": ["capacity_allocation", "demand_upper_bound", "revenue_maximization"],
    "marketing": ["budget_allocation", "market_response", "segment_constraint"],
    "contract_allocation": ["supplier_capacity", "eligibility_constraint", "allocation_bounds"],
    "supply_chain": ["multi_echelon_flow", "capacity_constraint", "demand_satisfaction", "inventory_or_shipment_balance"],
    "network_design": ["arc_activation", "fixed_charge", "flow_balance", "capacity_constraint"],
    "network_flow": ["node_arc_flow", "flow_conservation", "arc_cost", "source_sink_or_supply_demand"],
    "routing": ["routing_binary_variable", "visit_once", "flow_conservation", "capacity_or_time_window"],
    "transportation": ["nonnegative_shipment", "supply_constraint", "demand_constraint", "cost_matrix"],
    "lot_sizing": ["inventory_balance", "setup_decision", "holding_cost", "time_indexed_decision"],
    "production_planning": ["product_mix", "resource_capacity", "profit_or_cost_objective", "demand_bound"],
    "agriculture": ["resource_allocation", "land_or_water_constraint", "yield_or_profit"],
    "energy": ["dispatch_balance", "generation_capacity", "energy_cost"],
    "workforce_deployment": ["deployment_assignment", "staffing_requirement", "readiness_or_skill_constraint"],
    "dispersion": ["pairwise_distance", "site_selection", "diversity_objective"],
    "general_linear_programming": ["linear_objective", "linear_capacity", "activity_level"],
}

FAMILY_SIGNATURES: dict[str, dict[str, Any]] = {
    "assignment": {
        "variable_types": ["binary"],
        "core_constraints": ["assignment_balance", "capacity_or_exclusivity_limit"],
        "required_tables": ["cost_or_score_matrix", "eligibility_or_capacity_data"],
        "forbidden_changes": ["do not add route continuity unless arcs and paths exist"],
    },
    "scheduling": {
        "variable_types": ["binary", "continuous_or_integer_time"],
        "core_constraints": ["resource_capacity", "sequencing_or_time_assignment"],
        "required_tables": ["processing_times", "resource_or_slot_data"],
        "forbidden_changes": ["do not rewrite as transportation or facility location"],
    },
    "packing": {
        "variable_types": ["binary"],
        "core_constraints": ["capacity_limit"],
        "required_tables": ["item_size_or_weight_vector", "item_value_or_cost_vector"],
        "forbidden_changes": ["do not introduce flow balance or routing constraints"],
    },
    "facility_location": {
        "variable_types": ["binary_opening", "continuous_or_binary_assignment"],
        "core_constraints": ["demand_satisfaction", "capacity_limit", "open_before_serve_linking"],
        "required_tables": ["fixed_cost_vector", "assignment_or_shipping_cost_matrix", "demand_vector"],
        "forbidden_changes": ["do not remove fixed opening costs", "do not turn site opening into pure transportation"],
    },
    "transportation": {
        "variable_types": ["continuous_nonnegative"],
        "objective_sense": "minimize",
        "core_constraints": ["supply_balance", "demand_balance"],
        "required_tables": ["supply_vector", "demand_vector", "lane_cost_matrix"],
        "forbidden_changes": ["do not add vehicles", "do not add time windows", "do not add subtour constraints"],
    },
    "routing": {
        "variable_types": ["binary_arc_or_visit", "continuous_time_or_load_optional"],
        "core_constraints": ["visit_once", "route_continuity", "capacity_or_time_window"],
        "required_tables": ["travel_cost_or_time_matrix", "customer_or_node_data"],
        "forbidden_changes": ["do not simplify to pure transportation flow"],
    },
    "lot_sizing": {
        "variable_types": ["continuous_or_integer_quantity", "binary_setup_optional"],
        "core_constraints": ["inventory_balance", "demand_satisfaction", "setup_or_capacity_limit"],
        "required_tables": ["period_demand", "setup_or_order_cost", "holding_or_backlog_cost"],
        "forbidden_changes": ["do not remove time periods or inventory carryover"],
    },
    "network_flow": {
        "variable_types": ["continuous_nonnegative"],
        "core_constraints": ["flow_conservation", "arc_capacity_optional"],
        "required_tables": ["node_set", "arc_set", "arc_cost_or_capacity"],
        "forbidden_changes": ["do not add facility fixed charges unless present"],
    },
    "network_design": {
        "variable_types": ["binary_activation", "continuous_flow"],
        "core_constraints": ["flow_balance", "activation_linking", "capacity_limit"],
        "required_tables": ["arc_set", "fixed_activation_cost", "flow_cost_or_capacity"],
        "forbidden_changes": ["do not remove activation or fixed-charge logic"],
    },
    "general_linear_programming": {
        "variable_types": ["continuous_or_integer_as_source"],
        "core_constraints": ["linear_capacity_or_requirement"],
        "required_tables": ["objective_coefficients", "constraint_coefficients", "bounds"],
        "forbidden_changes": ["do not introduce nonlinear or unstated business rules"],
    },
}


def load_generator_profiles(profiles_path: str | Path | None) -> dict[str, Any]:
    if profiles_path is None:
        return {"version": "none", "profiles": {}}
    path = Path(profiles_path)
    if not path.exists():
        return {"version": "missing", "profiles": {}}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {"version": "invalid", "profiles": {}}
    payload.setdefault("profiles", {})
    return payload


def profile_generators(
    registry_path: str | Path,
    profiles_output: str | Path,
    *,
    report_output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    profiles_path = Path(profiles_output)
    if profiles_path.exists() and not overwrite:
        raise FileExistsError(f"generator profile file already exists: {profiles_path}")
    registry_rows = read_jsonl(registry_path)
    profiles = {
        row["generator_id"]: build_default_profile(row)
        for row in sorted(registry_rows, key=lambda item: item.get("generator_id", ""))
        if row.get("status") == "REGISTERED"
    }
    payload = {
        "version": PROFILE_VERSION,
        "description": "Project-local generator profiles for OR-CPT production.",
        "defaults": {
            "difficulty_mix": {"level_1": 0.50, "level_2": 0.30, "level_3": 0.15, "level_4": 0.05},
            "expected_acceptance_rate": 0.35,
        },
        "profiles": profiles,
    }
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(
        yaml.dump(payload, Dumper=_NoAliasSafeDumper, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    if report_output is not None:
        write_text(report_output, render_profile_inventory(payload))
    return {"profiles": len(profiles)}


def build_default_profile(registry_row: dict[str, Any]) -> dict[str, Any]:
    generator_id = registry_row["generator_id"]
    family = infer_task_family(generator_id)
    sub_family = infer_sub_family(generator_id, family)
    priority = _priority_for_generator(generator_id)
    sampling_weight = _sampling_weight(priority)
    return {
        "enabled": True,
        "priority": priority,
        "recommended_weight": _recommended_weight(priority),
        "sampling_weight": sampling_weight,
        "task_family": family,
        "sub_family": sub_family,
        "difficulty_support": "basic",
        "default_variant_id": "base",
        "modeling_concepts": FAMILY_CONCEPTS.get(family, FAMILY_CONCEPTS["general_linear_programming"]),
        "canonical_math_signature": _canonical_signature_for_family(family, registry_row),
        "difficulty_config": _difficulty_axes_for_family(family),
        "difficulty_axes": _difficulty_axes_for_family(family),
        "semantic_scenarios": [],
        "formulation_variants": [
            {
                "variant_id": "base",
                "enabled": True,
                "notes": "Original OptMATH formulation imported into this project.",
            }
        ],
        "quality_risks": _quality_risks_for_family(family),
        "rationale_requirements": _rationale_requirements_for_family(family),
        "quality_controls": _quality_controls_for_family(family),
        "quality_controller": _quality_controller_for_family(generator_id, family),
        "max_views_per_seed_instance": 12,
        "expected_acceptance_rate": _expected_acceptance_rate(priority),
        "source_generator_name": registry_row.get("name"),
        "source": registry_row.get("source", "optmath"),
    }


def render_profile_inventory(profile_payload: dict[str, Any]) -> str:
    profiles = profile_payload.get("profiles") or {}
    lines = [
        "# Generator Profile Inventory",
        "",
        f"- Profile version: {profile_payload.get('version')}",
        f"- Total profiles: {len(profiles)}",
        "",
        "| generator_id | priority | weight | family | sub_family | difficulty | concepts |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for generator_id, profile in sorted(profiles.items()):
        lines.append(
            f"| {generator_id} | {profile.get('priority')} | {float(profile.get('recommended_weight') or 0):.4f} | "
            f"{profile.get('task_family')} | {profile.get('sub_family')} | {profile.get('difficulty_support')} | "
            f"{len(profile.get('modeling_concepts') or [])} |"
        )
    return "\n".join(lines) + "\n"


def select_difficulty_level(
    *,
    generator_id: str,
    seed: int,
    difficulty_mix: dict[str, float] | None,
    profile: dict[str, Any] | None,
) -> str:
    mix = dict(difficulty_mix or {})
    if not mix:
        mix = {"level_1": 1.0}
    if (profile or {}).get("difficulty_support") == "none":
        return "level_1"
    total = sum(max(0.0, float(value)) for value in mix.values())
    if total <= 0:
        return "level_1"
    digest = hashlib.sha1(f"{generator_id}:{seed}:difficulty".encode("utf-8")).hexdigest()
    point = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    cumulative = 0.0
    for level, weight in sorted(mix.items()):
        cumulative += max(0.0, float(weight)) / total
        if point <= cumulative:
            return str(level)
    return str(sorted(mix)[-1])


def build_difficulty_config(profile: dict[str, Any] | None, difficulty_level: str, seed: int) -> dict[str, Any]:
    profile = profile or {}
    parameters: dict[str, Any] = {}
    axes = profile.get("difficulty_config") or profile.get("difficulty_axes") or {}
    difficulty_support = profile.get("difficulty_support", "none")
    if difficulty_support == "native" and isinstance(axes, dict):
        for axis_name, axis_config in axes.items():
            if isinstance(axis_config, dict) and difficulty_level in axis_config:
                parameters[axis_name] = axis_config[difficulty_level]
    return {
        "difficulty_level": difficulty_level,
        "difficulty_support": difficulty_support,
        "variant_id": profile.get("default_variant_id") or "base",
        "parameters": parameters,
        "profile_axes": axes if difficulty_support != "native" else {},
        "seed": seed,
    }


def infer_task_family(generator_id: str) -> str:
    gid = generator_id.lower()
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
        ("staticlineplanning", "network_design"),
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
        if needle in gid:
            return family
    return "general_linear_programming"


def infer_sub_family(generator_id: str, family: str) -> str:
    gid = generator_id.lower()
    rules = [
        ("aircraftlanding", "aircraft_landing_scheduling"),
        ("flowshop", "flow_shop_scheduling"),
        ("jopshop", "job_shop_scheduling"),
        ("schedulingproblem", "generic_scheduling"),
        ("multi_factory_schedule", "multi_factory_scheduling"),
        ("vrptw", "vehicle_routing_with_time_windows"),
        ("tsp", "traveling_salesperson"),
        ("fleet_routing", "fleet_routing"),
        ("cflp", "capacitated_facility_location"),
        ("facility_location", "facility_location"),
        ("cell_tower", "coverage_facility_location"),
        ("scooter_location", "micromobility_facility_location"),
        ("setcover", "set_covering"),
        ("multisetcover", "multi_set_covering"),
        ("transp", "transportation_problem"),
        ("nltrans", "transportation_variant"),
        ("uncapacitatedlotsizingbacklogging", "uncapacitated_lot_sizing_with_backlogging"),
        ("uncapacitatedlotsizing", "uncapacitated_lot_sizing"),
        ("clsp", "capacitated_lot_sizing"),
        ("supplychain", "multi_echelon_supply_chain"),
        ("netmcol", "multi_commodity_network_design"),
        ("mcnd", "multi_commodity_network_design"),
        ("netthru", "network_throughput"),
        ("net1", "network_flow"),
        ("shortest_path", "shortest_path"),
        ("portfolio", "portfolio_optimization"),
        ("diet", "diet_problem"),
        ("blending", "blending_problem"),
        ("knapsack", "knapsack"),
        ("binpacking", "bin_packing"),
    ]
    for needle, sub_family in rules:
        if needle in gid:
            return sub_family
    return family


def _priority_for_generator(generator_id: str) -> str:
    if generator_id in MAIN_GENERATORS:
        return "high"
    if any(needle in generator_id for needle in CAUTIOUS_GENERATOR_NEEDLES):
        return "cautious"
    return "medium"


def _recommended_weight(priority: str) -> float:
    return {"high": 3.0, "medium": 1.0, "cautious": 0.35}.get(priority, 1.0)


def _sampling_weight(priority: str) -> dict[str, Any]:
    return {
        "cpt_weight": _recommended_weight(priority),
        "max_daily_samples": {"high": 5000, "medium": 2500, "cautious": 800}.get(priority, 1500),
        "min_quality_score": {"high": 0.75, "medium": 0.65, "cautious": 0.70}.get(priority, 0.65),
        "solver_timeout_seconds": {"high": 60, "medium": 45, "cautious": 30}.get(priority, 45),
    }


def _expected_acceptance_rate(priority: str) -> float:
    return {"high": 0.45, "medium": 0.35, "cautious": 0.20}.get(priority, 0.30)


def _difficulty_axes_for_family(family: str) -> dict[str, dict[str, Any]]:
    generic = {
        "size_scale": {
            "level_1": "small",
            "level_2": "medium",
            "level_3": "large",
            "level_4": "stress",
        },
        "constraint_tightness": {
            "level_1": "loose",
            "level_2": "moderate",
            "level_3": "tight",
            "level_4": "very_tight",
        },
    }
    family_specific = {
        "facility_location": {"facility_count": {"level_1": [3, 5], "level_2": [6, 12], "level_3": [13, 30], "level_4": [31, 80]}},
        "transportation": {"node_count": {"level_1": [3, 5], "level_2": [6, 12], "level_3": [13, 30], "level_4": [31, 80]}},
        "routing": {"customer_count": {"level_1": [5, 12], "level_2": [13, 30], "level_3": [31, 80], "level_4": [81, 200]}},
        "lot_sizing": {"period_count": {"level_1": [4, 8], "level_2": [9, 20], "level_3": [21, 52], "level_4": [53, 120]}},
    }
    return generic | family_specific.get(family, {})


def _canonical_signature_for_family(family: str, registry_row: dict[str, Any]) -> dict[str, Any]:
    signature = dict(FAMILY_SIGNATURES.get(family, FAMILY_SIGNATURES["general_linear_programming"]))
    model_type = registry_row.get("model_type")
    objective_sense = registry_row.get("optimization_sense")
    if model_type:
        signature["model_type"] = model_type
    if objective_sense and "objective_sense" not in signature:
        signature["objective_sense"] = objective_sense
    return signature


def _quality_risks_for_family(family: str) -> list[str]:
    common = ["LLM may omit a numeric coefficient", "Natural-language statement may become too generic"]
    risks = {
        "facility_location": ["LLM may omit fixed opening costs", "LLM may miss open-before-serve linking constraints"],
        "transportation": ["LLM may confuse transportation flow with vehicle routing"],
        "routing": ["LLM may omit visit-once or time-window constraints"],
        "lot_sizing": ["LLM may miss inventory balance across periods"],
        "network_design": ["LLM may omit fixed-charge activation decisions"],
    }
    return common + risks.get(family, [])


def _rationale_requirements_for_family(family: str) -> list[str]:
    requirements = {
        "facility_location": ["explain binary opening variables", "explain assignment or shipment variables", "explain capacity and linking constraints"],
        "transportation": ["explain source supply constraints", "explain destination demand constraints", "explain nonnegative lane shipment variables"],
        "routing": ["explain route continuity", "explain customer visit-once logic", "explain capacity or time-window constraints"],
        "lot_sizing": ["explain inventory balance", "explain setup or order decisions", "explain holding and backlog costs"],
    }
    return requirements.get(family, ["explain decision variables", "explain objective coefficients", "explain each major constraint family"])


def _quality_controls_for_family(family: str) -> dict[str, list[str]]:
    common_required = [
        "source_instance_parse_ok",
        "gurobi_code_or_lp_executes",
        "solver_status_recorded",
        "objective_recomputed",
        "constraint_residual_checked",
        "no_forbidden_misframing",
        "no_duplicate_exact_text",
    ]
    family_required = {
        "transportation": ["preserve_supply_vector", "preserve_demand_vector", "preserve_lane_cost_matrix"],
        "facility_location": ["preserve_fixed_costs", "preserve_capacity_constraints", "preserve_open_before_serve_linking"],
        "routing": ["preserve_visit_once_logic", "preserve_route_continuity", "preserve_time_window_or_capacity_if_present"],
        "lot_sizing": ["preserve_inventory_balance", "preserve_period_demand"],
    }
    return {
        "validation_required": common_required + family_required.get(family, []),
        "rejection_reasons": [
            "changed_objective_sense",
            "missing_core_constraint",
            "added_unstated_constraint",
            "wrong_variable_type",
            "objective_mismatch",
            "code_not_executable",
            "scenario_math_mismatch",
            "high_text_similarity",
        ],
    }


def _quality_controller_for_family(generator_id: str, family: str) -> dict[str, Any]:
    gid = generator_id.lower()
    if "staticlineplanning" in gid:
        return {
            "business_variable_patterns": [r"^x\["],
            "slack_variable_patterns": [r"^s\[", "reserve", "slack", "unmet", "short"],
            "activation_variable_patterns": [r"^x\["],
            "core_constraint_patterns": ["demand", "capacity", "fleet", "line"],
            "hard_reject_rules": [
                "reject_all_activation_zero",
                "reject_slack_dominates_business",
                "reject_no_business_nonzero",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"max_slack_solution_ratio": 0.25, "min_business_nonzero_count": 1},
        }
    if family == "production_planning":
        return {
            "business_variable_patterns": ["prod", "production", "make", "output", r"^x\["],
            "slack_variable_patterns": ["slack", "short", "unmet", "reserve"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["capacity", "resource", "time", "labor", "machine", "material"],
            "hard_reject_rules": [
                "reject_business_all_upper_bound",
                "reject_business_all_lower_bound",
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.08,
                "min_business_nonzero_count": 2,
            },
        }
    if family in {"facility_location", "network_design"}:
        return {
            "business_variable_patterns": ["assign", "serve", "ship", "flow", "cover", r"^x\["],
            "slack_variable_patterns": ["slack", "unmet", "short", "reserve"],
            "activation_variable_patterns": ["open", "activate", "build", "facility", "site", "selected", "location", r"^y\["],
            "core_constraint_patterns": ["capacity", "demand", "balance", "budget", "limit", "cover"],
            "hard_reject_rules": [
                "reject_all_activation_zero",
                "reject_all_activation_one",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.05,
            },
        }
    if family in {"transportation", "network_flow", "supply_chain"}:
        return {
            "business_variable_patterns": ["flow", "ship", "shipment", "arc", "lane", r"^x\["],
            "slack_variable_patterns": ["slack", "unmet", "short", "backlog"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["supply", "demand", "balance", "conservation", "capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_single_flow_variable_dominates",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "max_single_business_variable_share": 0.95,
                "max_slack_solution_ratio": 0.15,
                "min_core_binding_ratio": 0.05,
            },
        }
    if family in {"packing", "covering"}:
        return {
            "business_variable_patterns": ["item", "select", "set", "cover", r"^x\["],
            "slack_variable_patterns": ["slack", "uncovered"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["capacity", "cover", "demand", "weight", "assignment"],
            "hard_reject_rules": [
                "reject_empty_selection",
                "reject_full_selection",
                "reject_single_selected_item",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "min_business_nonzero_count": 2,
                "min_core_binding_ratio": 0.20,
            },
        }
    if family == "lot_sizing":
        return {
            "business_variable_patterns": ["prod", "order", "quantity", r"^x\["],
            "slack_variable_patterns": ["backlog", "unmet", "short", "slack"],
            "activation_variable_patterns": ["setup", r"^y\["],
            "core_constraint_patterns": ["inventory", "balance", "demand", "capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {"max_slack_solution_ratio": 0.25},
        }
    if family == "assignment":
        return {
            "business_variable_patterns": ["assign", "allocation", "team", "selected", r"^x\["],
            "slack_variable_patterns": ["shortage", "slack", "unmet", "gap"],
            "activation_variable_patterns": ["assign", "allocation", r"^x\["],
            "core_constraint_patterns": [
                "assignment",
                "staffing",
                "capacity",
                "skill",
                "balance",
                "exclusive",
                "availability",
                "demand",
                "allocation",
            ],
            "hard_reject_rules": [
                "reject_no_business_nonzero",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.35,
                "min_business_nonzero_count": 2,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "scheduling":
        return {
            "business_variable_patterns": [r"^s_", "start", "completion", "c_max", r"^x_", "assign", "landing"],
            "slack_variable_patterns": ["slack", "tard", "lateness", "short"],
            "activation_variable_patterns": [r"^x_", "assign", "sequence", "order"],
            "core_constraint_patterns": [
                "prec",
                "machine",
                "makespan",
                "capacity",
                "sequence",
                "landing",
                "runway",
                "time",
                "order",
                "separation",
                "earliest",
                "latest",
                "deviation",
            ],
            "hard_reject_rules": [
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.30,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "routing":
        return {
            "business_variable_patterns": ["route", "arc", "travel", "numplanes", "vehicle", r"^x\["],
            "slack_variable_patterns": ["slack", "unserved", "unmet", "late"],
            "activation_variable_patterns": ["route", "arc", "numplanes", "vehicle", r"^x\["],
            "core_constraint_patterns": ["flow", "balance", "demand", "visit", "degree", "capacity", "route", "time"],
            "hard_reject_rules": [
                "reject_no_business_nonzero",
                "reject_slack_dominates_business",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_constraints": 1,
                "min_core_binding_ratio": 0.03,
            },
        }
    if family == "diet":
        return {
            "business_variable_patterns": ["buy", "servings", "foods", "food", r"^x\["],
            "slack_variable_patterns": ["slack", "excess", "deficit"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["nutrient", "requirement", "volume", "min", "max"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "energy":
        return {
            "business_variable_patterns": ["numgenerators", "poweroutput", "generation", "output", r"^x\["],
            "slack_variable_patterns": ["slack", "unmet", "short", "reserve_gap"],
            "activation_variable_patterns": ["numgenerators", "startup", r"^x\["],
            "core_constraint_patterns": [
                "available",
                "demand",
                "mingeneration",
                "maxgeneration",
                "reserve",
                "startup",
                "capacity",
            ],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
                "reject_slack_dominates_business",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {
                "max_slack_solution_ratio": 0.20,
                "min_core_binding_ratio": 0.03,
                "min_business_nonzero_count": 2,
            },
        }
    if family == "workforce_deployment":
        return {
            "business_variable_patterns": ["soldiers", "deploy", "assignment", r"^x\["],
            "slack_variable_patterns": ["shortage", "slack", "unmet", "gap"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["totalsoldiers", "skillrequirement", "skillavailability", "staffing", "capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "cutting_stock":
        return {
            "business_variable_patterns": ["cut", "pattern"],
            "slack_variable_patterns": ["slack", "waste"],
            "activation_variable_patterns": ["cut", "pattern"],
            "core_constraint_patterns": ["fill", "order", "demand", "width"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "blending":
        return {
            "business_variable_patterns": ["alloy", "blend", "ingredient", r"^x\["],
            "slack_variable_patterns": ["slack", "deviation"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["element", "total", "blend", "composition"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    if family == "contract_allocation":
        return {
            "business_variable_patterns": ["generation", "delivery", "contract", "allocation", r"^x\["],
            "slack_variable_patterns": ["slack", "unmet", "short"],
            "activation_variable_patterns": ["incidence", "selected", r"^y\["],
            "core_constraint_patterns": ["capacity", "contract", "fulfillment", "contributor", "delivery", "activation"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_core_binding_ratio", "review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_core_binding_ratio": 0.03, "min_business_nonzero_count": 2},
        }
    if family == "marketing":
        return {
            "business_variable_patterns": ["supply", "market", "allocation", r"^x\["],
            "slack_variable_patterns": ["slack", "unmet"],
            "activation_variable_patterns": [],
            "core_constraint_patterns": ["demand", "market", "capacity"],
            "hard_reject_rules": [
                "reject_business_all_zero",
                "reject_no_core_binding_constraint",
            ],
            "review_rules": ["review_low_business_nonzero_count"],
            "family_specific_thresholds": {"min_business_nonzero_count": 2},
        }
    return {
        "business_variable_patterns": [],
        "slack_variable_patterns": ["slack", "unmet", "short", "reserve"],
        "activation_variable_patterns": [],
        "core_constraint_patterns": [],
        "hard_reject_rules": [],
        "review_rules": [],
        "family_specific_thresholds": {},
    }
