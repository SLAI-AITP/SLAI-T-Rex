from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
import yaml

import or_cpt_engine.backtranslation.backtranslate as backtranslate_module
from or_cpt_engine.backtranslation.backtranslate import (
    _MAX_BACKTRANSLATION_PROMPT_CHARS,
    _SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS,
    _TARGET_BACKTRANSLATION_PROMPT_CHARS,
    _combine_background_and_statement,
    _compact_source_data,
    _ensure_required_problem_statement_tables,
    _structured_problem_data_from_compact_source,
    _profile_normalized_row,
    _payload_mapping,
    _payload_text,
    _render_prompt as render_backtranslation_prompt,
    _target_backtranslation_prompt_chars,
    _backtranslate_one,
    backtranslate_instances,
    select_backtranslation_prompt_spec,
)
from or_cpt_engine.backtranslation.nl_quality_filter import _canonical_numbers, _number_coverage, evaluate_nl_candidate
from or_cpt_engine.backtranslation.scenario_catalog import scenario_guidance_for_generator
from or_cpt_engine.cli.main import (
    _audit_seed_input_for_cpt,
    _build_per_generator_count_from_plan,
    _record_production_plan_usage,
    _resolve_per_generator_counts,
    _write_cpt_run_manifest,
    _write_seed_manifest,
)
from or_cpt_engine.contracts import resolve_family_contract
from or_cpt_engine.evaluation.rejection_classifier import classify_forward_eval_rejection
from or_cpt_engine.export.train_val_export import export_train_val_test
from or_cpt_engine.evaluation.forward_eval import evaluate_forward_output_row
from or_cpt_engine.forward_modeling.code_quality import classify_execution_failure, prepare_forward_code
from or_cpt_engine.forward_modeling.generate_forward_model import (
    _forward_model_one,
    _forward_retry_issue_hints,
    _modeling_guardrails,
    _render_prompt as render_forward_prompt,
    _static_signature_issues,
)
from or_cpt_engine.forward_modeling.repair_forward_model import _render_repair_prompt
from or_cpt_engine.forward_modeling.generate_forward_model import _profile_normalized_row as normalize_forward_profile_row
from or_cpt_engine.generation.generator_profiles import profile_generators
from or_cpt_engine.generation.instance_generator import generate_instances, smoke_test_generators
from or_cpt_engine.generation.production_planner import plan_production
from or_cpt_engine.generation.quality_analyzer import analyze_generator_quality
from or_cpt_engine.generators.import_optmath import import_optmath_generators
from or_cpt_engine.llm.openai_compatible_client import (
    AsyncOpenAICompatibleClient,
    ResolvedLLMEndpoint,
    SimpleOpenAICompatibleClient,
    _AsyncEndpointSession,
)
from or_cpt_engine.prompt_optimizer.engine import analyze_run_for_prompt_optimization, optimize_prompts_for_run
from or_cpt_engine.quality.instance_quality import DEFAULT_THRESHOLDS, evaluate_instance_quality, validate_instance_quality
from or_cpt_engine.registry.scan_optmath_generators import scan_optmath_generators
from or_cpt_engine.reporting.run_quality_dashboard import analyze_run_quality
from or_cpt_engine.rendering.cpt_renderer import (
    _check_rendered_text,
    _choose_doc_type,
    _choose_style_context,
    _normalize_rendered_numeric_precision,
    render_cpt_document_row,
    render_cpt_documents,
)
from or_cpt_engine.schemas.common import (
    AcceptedPair,
    EngineConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMStageConfig,
    ObjectiveComparison,
    QualityThresholdsConfig,
    RenderingConfig,
)
from or_cpt_engine.solver.objective_comparator import compare_objective
from or_cpt_engine.solver.solver_validation import validate_solver
from or_cpt_engine.utils.io import read_jsonl, write_jsonl
from or_cpt_engine.utils.logging import configure_logging as configure_or_cpt_logging
from or_cpt_engine.utils.prompt_registry import PromptRegistry


def test_objective_comparator_accepts_abs_or_rel_tolerance() -> None:
    assert compare_objective(10.0, 10.000001).is_correct
    assert compare_objective(10.0, 10.1, abs_tolerance=1e-4, rel_tolerance=1e-4).is_correct is False
    assert compare_objective(None, 10.0).is_correct is False


def test_llm_client_loads_endpoint_pool(tmp_path: Path) -> None:
    pool_path = tmp_path / "model_api_pool.yaml"
    pool_path.write_text(
        "\n".join(
            [
                "endpoints:",
                "  - name: a",
                "    base_url: http://127.0.0.1:8000/v1",
                "    model_name: model-a",
                "    api_key: EMPTY",
                "  - name: b",
                "    base_url: 127.0.0.2:8000",
                "    model_name: model-b",
                "    api_key: EMPTY",
            ]
        ),
        encoding="utf-8",
    )
    llm_config = LLMConfig(
        providers={
            "default": LLMProviderConfig(
                endpoint_pool_path=str(pool_path),
                model="fallback",
                timeout_sec=10,
            )
        }
    )
    client = SimpleOpenAICompatibleClient(llm_config, LLMStageConfig(provider="default"))

    assert [endpoint.name for endpoint in client.endpoints] == ["a", "b"]
    assert client.endpoints[1].base_url == "http://127.0.0.2:8000/v1"


def test_or_cpt_llm_client_writes_structured_model_failure_log(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "run.log"
    failure_log_path = tmp_path / "logs" / "run_model_failures.jsonl"
    configure_or_cpt_logging("INFO", log_file=str(log_path), model_failure_log_file=str(failure_log_path))
    session = _AsyncEndpointSession(
        endpoint=ResolvedLLMEndpoint(
            name="dsv4_test",
            base_url="http://127.0.0.1:8005/v1",
            model="deepseek-v4-flash",
        ),
        client=httpx.AsyncClient(),
    )
    try:
        AsyncOpenAICompatibleClient._log_model_attempt(
            "req_timeout",
            0,
            session,
            "short prompt",
            "",
            180.0,
            "error",
            error="",
            log_context={"instance_id": "seed_1", "prompt_name": "forward_modeling_prompt"},
        )
    finally:
        asyncio.run(session.client.aclose())

    rows = read_jsonl(failure_log_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "req_timeout"
    assert row["attempt"] == 1
    assert row["instance_id"] == "seed_1"
    assert row["prompt_name"] == "forward_modeling_prompt"
    assert row["endpoint_name"] == "dsv4_test"
    assert row["error_type"] == "LIKELY_TIMEOUT"


def test_family_contract_resolution_uses_family_alias_and_default() -> None:
    config = {
        "version": "v_test",
        "generator_aliases": {"packing": ["knapsack"]},
        "families": {
            "packing": {
                "answer_contract": {
                    "model_type": "MILP",
                    "objective_sense_options": ["maximize"],
                    "required_variable_families": ["select_item_binary"],
                    "required_constraints": ["capacity_limit"],
                }
            }
        },
        "defaults": {"default_linear_or_contract": {"model_type": "LP_or_MILP"}},
    }

    contract = resolve_family_contract({"generator_id": "optmath_knapsack"}, config)
    fallback = resolve_family_contract({"generator_id": "unknown_generator"}, config)

    assert contract["contract_id"] == "packing"
    assert contract["contract_coverage"] == "high"
    assert contract["answer_contract"]["model_type"] == "MILP"
    assert fallback["contract_id"] == "default_linear_or_contract"
    assert fallback["contract_coverage"] == "low"


def test_family_contract_config_has_scheduling_generator_overrides() -> None:
    import yaml

    config = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))

    staff_contract = resolve_family_contract({"generator_id": "optmath_schedulingproblem"}, config)
    flowshop_contract = resolve_family_contract({"generator_id": "optmath_flowshop_2"}, config)
    fleet_contract = resolve_family_contract({"generator_id": "optmath_fleet_routing"}, config)
    facility_contract = resolve_family_contract({"generator_id": "optmath_facility_location"}, config)
    cflp_contract = resolve_family_contract({"generator_id": "optmath_cflp"}, config)
    factory_contract = resolve_family_contract({"generator_id": "optmath_factory_planning_problem"}, config)
    jobshop_contract = resolve_family_contract({"generator_id": "optmath_jopshop"}, config)
    supplychain_contract = resolve_family_contract({"generator_id": "optmath_supplychain"}, config)
    portfolio_contract = resolve_family_contract({"generator_id": "optmath_portfolio"}, config)
    contractallocation_contract = resolve_family_contract({"generator_id": "optmath_contractallocation"}, config)
    cut_contract = resolve_family_contract({"generator_id": "optmath_cut_edited"}, config)
    vrptw_contract = resolve_family_contract({"generator_id": "optmath_vrptw"}, config)
    pdispersion_contract = resolve_family_contract({"generator_id": "optmath_the_p_dispersion_model"}, config)
    aircraft_assignment_contract = resolve_family_contract({"generator_id": "optmath_aircraftassignment"}, config)
    aircraft_landing_contract = resolve_family_contract({"generator_id": "optmath_aircraftlanding"}, config)
    multisetcover_contract = resolve_family_contract({"generator_id": "optmath_multisetcover"}, config)
    staticline_contract = resolve_family_contract({"generator_id": "optmath_staticlineplanning"}, config)
    farmplanning_contract = resolve_family_contract({"generator_id": "optmath_farmplanning"}, config)
    carselection_contract = resolve_family_contract({"generator_id": "optmath_carselection"}, config)
    cell_tower_contract = resolve_family_contract({"generator_id": "optmath_cell_tower"}, config)
    revenue_contract = resolve_family_contract({"generator_id": "optmath_revenue"}, config)
    revenue_management_contract = resolve_family_contract({"generator_id": "optmath_revenue_management"}, config)
    marketshare_contract = resolve_family_contract({"generator_id": "optmath_marketshare"}, config)
    transp_contract = resolve_family_contract({"generator_id": "optmath_transp"}, config)
    nltrans_contract = resolve_family_contract({"generator_id": "optmath_nltrans"}, config)
    clsp_contract = resolve_family_contract({"generator_id": "optmath_clsp_expand_capacity"}, config)
    uls_contract = resolve_family_contract({"generator_id": "optmath_uncapacitatedlotsizing"}, config)
    netmcol_contract = resolve_family_contract({"generator_id": "optmath_netmcol"}, config)
    multi_contract = resolve_family_contract({"generator_id": "optmath_multi"}, config)
    mcnd_contract = resolve_family_contract({"generator_id": "optmath_mcnd"}, config)
    tsp_contract = resolve_family_contract({"generator_id": "optmath_tsp"}, config)
    shortest_path_contract = resolve_family_contract({"generator_id": "optmath_the_shortest_path_problem"}, config)
    scooter_contract = resolve_family_contract({"generator_id": "optmath_scooter_location"}, config)
    netasgn_contract = resolve_family_contract({"generator_id": "optmath_netasgn"}, config)
    net1_contract = resolve_family_contract({"generator_id": "optmath_net1"}, config)
    netthru_contract = resolve_family_contract({"generator_id": "optmath_netthru"}, config)
    steel4_contract = resolve_family_contract({"generator_id": "optmath_steel4"}, config)
    structure_contract = resolve_family_contract({"generator_id": "optmath_structure_based_assignment"}, config)
    smallbucket_contract = resolve_family_contract({"generator_id": "optmath_singlelevelsmallbucket"}, config)
    multi_factory_contract = resolve_family_contract({"generator_id": "optmath_the_multi_factory_schedule_problem"}, config)
    team_contract = resolve_family_contract({"generator_id": "optmath_team_formulation"}, config)

    assert staff_contract["contract_source"] == "generator_override"
    assert "coverage_equal_demand_with_shortage" in staff_contract["answer_contract"]["required_constraints"]
    assert flowshop_contract["contract_source"] == "generator_override"
    assert "one_job_per_sequence_position" in flowshop_contract["answer_contract"]["required_constraints"]
    assert "explicit_makespan_variable" in flowshop_contract["answer_contract"]["required_objective_terms"]
    assert "makespan_completion_bound" in flowshop_contract["answer_contract"]["required_constraints"]
    assert (
        "do not reinterpret final-machine processing-time coefficients as assignment costs, delay weights, or weighted completion costs"
        in flowshop_contract["answer_contract"]["forbidden_changes"]
    )
    assert fleet_contract["contract_source"] == "generator_override"
    assert fleet_contract["contract_id"] == "optmath_fleet_routing"
    assert "integer_flight_leg_count" in fleet_contract["answer_contract"]["required_variable_families"]
    assert "time_expanded_fleet_flow_conservation" in fleet_contract["answer_contract"]["required_constraints"]
    assert "do not require binary route-arc variables" in fleet_contract["answer_contract"]["forbidden_changes"]
    assert "route_arc_binary" not in fleet_contract["contract_tags"]
    assert facility_contract["contract_source"] == "generator_override"
    assert "served_zone_commodity_demand_exactly_shipped" in facility_contract["answer_contract"]["required_constraints"]
    assert "explicit_distribution_center_open_before_zone_assignment" in facility_contract["answer_contract"]["required_constraints"]
    assert "do not omit unit throughput costs" in facility_contract["answer_contract"]["forbidden_changes"]
    assert "do not omit explicit Served[d,z] <= Selected[d] linking" in facility_contract["answer_contract"]["forbidden_changes"]
    assert cflp_contract["contract_source"] == "generator_override"
    assert "customer_demand_exactly_satisfied" in cflp_contract["answer_contract"]["required_constraints"]
    assert "ship_only_from_open_facility" in cflp_contract["answer_contract"]["required_constraints"]
    assert "facility_fixed_opening_cost" in cflp_contract["answer_contract"]["required_objective_terms"]
    assert "do not remove fixed facility opening costs" in cflp_contract["answer_contract"]["forbidden_changes"]
    assert factory_contract["contract_source"] == "generator_override"
    assert "machine_hour_capacity_after_maintenance" in factory_contract["answer_contract"]["required_constraints"]
    assert "do not aggregate product-period sales limits into one demand number per product" in factory_contract["answer_contract"]["forbidden_changes"]
    assert "do not infer demand totals from the reference answer" in factory_contract["answer_contract"]["forbidden_changes"]
    assert jobshop_contract["contract_source"] == "generator_override"
    assert "same_machine_pairwise_nonoverlap" in jobshop_contract["answer_contract"]["required_constraints"]
    assert supplychain_contract["contract_source"] == "generator_override"
    assert supplychain_contract["contract_id"] == "optmath_supplychain"
    assert "arc_capacity_activation_linking" in supplychain_contract["answer_contract"]["required_constraints"]
    assert "fixed_arc_activation_cost" in supplychain_contract["answer_contract"]["required_objective_terms"]
    assert "do not omit fixed arc activation costs" in supplychain_contract["answer_contract"]["forbidden_changes"]
    assert "do not imply a complete directed graph when the source gives a sparse arc set" in supplychain_contract["answer_contract"]["forbidden_changes"]
    assert portfolio_contract["contract_source"] == "generator_override"
    assert "minimum_expected_return_requirement" in portfolio_contract["answer_contract"]["required_constraints"]
    assert "do not introduce binary asset selection or cardinality variables" in portfolio_contract["answer_contract"]["forbidden_changes"]
    assert contractallocation_contract["contract_source"] == "generator_override"
    assert "exact_contract_fulfillment" in contractallocation_contract["answer_contract"]["required_constraints"]
    assert "do not weaken exact fulfillment into at-least or optional fulfillment" in contractallocation_contract["answer_contract"]["forbidden_changes"]
    assert cut_contract["contract_source"] == "generator_override"
    assert cut_contract["contract_id"] == "optmath_cut_edited"
    assert "nonnegative_integer_pattern_count" in cut_contract["answer_contract"]["required_variable_families"]
    assert "exact_order_width_fulfillment" in cut_contract["answer_contract"]["required_constraints"]
    assert "total_raw_roll_count" in cut_contract["answer_contract"]["required_objective_terms"]
    assert vrptw_contract["contract_source"] == "generator_override"
    assert "single_route_leaves_the_depot_once" in vrptw_contract["answer_contract"]["required_constraints"]
    assert "time_propagation_with_service_times_on_used_arcs" in vrptw_contract["answer_contract"]["required_constraints"]
    assert (
        "do not introduce multiple vehicles, vehicle indices, fleet-sizing decisions, or customer-to-vehicle assignment variables"
        in vrptw_contract["answer_contract"]["forbidden_changes"]
    )
    assert (
        "do not describe the source problem as a fleet with multiple independent routes"
        in vrptw_contract["answer_contract"]["forbidden_changes"]
    )
    assert "fleet_or_multiple_routes_wording" in vrptw_contract["answer_contract"]["high_risk_failures"]
    assert pdispersion_contract["contract_source"] == "generator_override"
    assert "select_exactly_p_nodes" in pdispersion_contract["answer_contract"]["required_constraints"]
    assert "min_distance_upper_bound_for_every_selected_pair" in pdispersion_contract["answer_contract"]["required_constraints"]
    assert "do not add fixed opening costs" in pdispersion_contract["answer_contract"]["forbidden_changes"]
    assert aircraft_assignment_contract["contract_source"] == "generator_override"
    assert "route_demand_requirement" in aircraft_assignment_contract["answer_contract"]["required_constraints"]
    assert "do not convert integer fleet counts into binary one-aircraft-to-one-route matching" in aircraft_assignment_contract["answer_contract"]["forbidden_changes"]
    assert aircraft_landing_contract["contract_source"] == "generator_override"
    assert "minimum_separation_time_between_ordered_landings" in aircraft_landing_contract["answer_contract"]["required_constraints"]
    assert "do not convert landing sequencing into aircraft fleet assignment" in aircraft_landing_contract["answer_contract"]["forbidden_changes"]
    assert multisetcover_contract["contract_source"] == "generator_override"
    assert multisetcover_contract["contract_id"] == "optmath_multisetcover"
    assert "binary_set_selection" in multisetcover_contract["answer_contract"]["required_variable_families"]
    assert "element_multicover_requirement_lower_bound" in multisetcover_contract["answer_contract"]["required_constraints"]
    assert "selected_set_cost" in multisetcover_contract["answer_contract"]["required_objective_terms"]
    assert "do not replace >= required coverage lower bounds with exact equality" in multisetcover_contract["answer_contract"]["forbidden_changes"]
    assert staticline_contract["contract_source"] == "generator_override"
    assert staticline_contract["contract_id"] == "optmath_staticlineplanning"
    assert "binary_line_activation" in staticline_contract["answer_contract"]["required_variable_families"]
    assert "continuous_line_frequency" in staticline_contract["answer_contract"]["required_variable_families"]
    assert "od_demand_coverage_with_shortage" in staticline_contract["answer_contract"]["required_constraints"]
    assert "fleet_vehicle_hours_budget" in staticline_contract["answer_contract"]["required_constraints"]
    assert "transfer_required_node_min_two_selected_lines" in staticline_contract["answer_contract"]["required_constraints"]
    assert "do not invent z[i,j,l] line-continuity variables" in staticline_contract["answer_contract"]["forbidden_changes"]
    assert farmplanning_contract["contract_source"] == "generator_override"
    assert "crop_yield_balance" in farmplanning_contract["answer_contract"]["required_constraints"]
    assert "family_consumption_bundle_balance" in farmplanning_contract["answer_contract"]["required_constraints"]
    assert "do not introduce binary crop-selection variables or fixed planting setup costs" in farmplanning_contract["answer_contract"]["forbidden_changes"]
    assert revenue_management_contract["contract_source"] == "generator_override"
    assert "package_demand_upper_bound" in revenue_management_contract["answer_contract"]["required_constraints"]
    assert "resource_capacity_consumption_limit" in revenue_management_contract["answer_contract"]["required_constraints"]
    assert "package_revenue_times_package_sales" in revenue_management_contract["answer_contract"]["required_objective_terms"]
    assert "do not change revenue maximization into cost minimization" in revenue_management_contract["answer_contract"]["forbidden_changes"]
    assert marketshare_contract["contract_source"] == "generator_override"
    assert marketshare_contract["contract_id"] == "optmath_marketshare"
    assert "market_product_demand_exactly_satisfied" in marketshare_contract["answer_contract"]["required_constraints"]
    assert "resource_capacity" not in marketshare_contract["answer_contract"]["required_constraints"]
    assert "net_profit_times_integer_supply" in marketshare_contract["answer_contract"]["required_objective_terms"]
    assert carselection_contract["contract_source"] == "generator_override"
    assert "assignment_only_if_eligible" in carselection_contract["answer_contract"]["required_constraints"]
    assert "participant_at_most_one_car" in carselection_contract["answer_contract"]["required_constraints"]
    assert "do not require every participant or every car to be assigned" in carselection_contract["answer_contract"]["forbidden_changes"]
    assert cell_tower_contract["contract_source"] == "generator_override"
    assert "region_coverage_linking" in cell_tower_contract["answer_contract"]["required_constraints"]
    assert "tower_budget_limit" in cell_tower_contract["answer_contract"]["required_constraints"]
    assert "region_population_times_covered_binary" in cell_tower_contract["answer_contract"]["required_objective_terms"]
    assert revenue_contract["contract_source"] == "generator_override"
    assert "package_demand_upper_bound" in revenue_contract["answer_contract"]["required_constraints"]
    assert "resource_capacity_consumption_limit" in revenue_contract["answer_contract"]["required_constraints"]
    blending_contract = resolve_family_contract({"generator_id": "optmath_blending_problem"}, config)
    assert blending_contract["contract_source"] == "family"
    assert "quality_ratio_or_specification" in blending_contract["answer_contract"]["required_constraints"]
    assert "demand_or_output_requirement_if_present" not in blending_contract["answer_contract"]["required_constraints"]
    assert transp_contract["contract_source"] == "generator_override"
    assert "origin_supply_exactly_shipped" in transp_contract["answer_contract"]["required_constraints"]
    assert "destination_demand_exactly_received" in transp_contract["answer_contract"]["required_constraints"]
    assert "lane_unit_cost_times_transport_quantity" in transp_contract["answer_contract"]["required_objective_terms"]
    assert "do not add lane capacity, fixed-charge activation, facility opening, or binary lane decisions" in transp_contract["answer_contract"]["forbidden_changes"]
    assert nltrans_contract["contract_source"] == "generator_override"
    assert "origin_supply_balance" in nltrans_contract["answer_contract"]["required_constraints"]
    assert "destination_demand_balance" in nltrans_contract["answer_contract"]["required_constraints"]
    assert "lane_capacity_limit" in nltrans_contract["answer_contract"]["required_constraints"]
    assert clsp_contract["contract_source"] == "generator_override"
    assert "cumulative_inventory_balance_without_backlog" in clsp_contract["answer_contract"]["required_constraints"]
    assert "period_capacity_limit" in clsp_contract["answer_contract"]["required_constraints"]
    assert "backlog_penalty_cost" not in clsp_contract["answer_contract"]["required_objective_terms"]
    assert "do not introduce backlog variables, backlog penalties, lost sales, or unmet-demand slack" in clsp_contract["answer_contract"]["forbidden_changes"]
    assert uls_contract["contract_source"] == "generator_override"
    assert uls_contract["contract_id"] == "optmath_uncapacitatedlotsizing"
    assert "order_quantity_linked_to_binary_setup" in uls_contract["answer_contract"]["required_constraints"]
    assert "capacity_if_present" not in uls_contract["answer_contract"]["required_constraints"]
    assert "do not add production capacity constraints" in uls_contract["answer_contract"]["forbidden_changes"]
    assert netmcol_contract["contract_source"] == "generator_override"
    assert "city_product_flow_balance" in netmcol_contract["answer_contract"]["required_constraints"]
    assert "directed_link_joint_capacity" in netmcol_contract["answer_contract"]["required_constraints"]
    assert "do not add binary arc activation variables or fixed activation costs" in netmcol_contract["answer_contract"]["forbidden_changes"]
    assert multi_contract["contract_source"] == "generator_override"
    assert multi_contract["contract_id"] == "optmath_multi"
    assert "continuous_origin_destination_product_shipment" in multi_contract["answer_contract"]["required_variable_families"]
    assert "origin_product_supply_equality" in multi_contract["answer_contract"]["required_constraints"]
    assert "destination_product_demand_equality" in multi_contract["answer_contract"]["required_constraints"]
    assert "origin_destination_joint_capacity_across_products" in multi_contract["answer_contract"]["required_constraints"]
    assert "origin_destination_product_unit_shipping_cost" in multi_contract["answer_contract"]["required_objective_terms"]
    assert "do not collapse products into a single aggregate commodity" in multi_contract["answer_contract"]["forbidden_changes"]
    assert mcnd_contract["contract_source"] == "generator_override"
    assert mcnd_contract["contract_id"] == "optmath_mcnd"
    assert "nonnegative_integer_arc_facility_count" in mcnd_contract["answer_contract"]["required_variable_families"]
    assert "arc_capacity_linked_to_integer_facility_count" in mcnd_contract["answer_contract"]["required_constraints"]
    assert "demand_weighted_commodity_arc_routing_cost" in mcnd_contract["answer_contract"]["required_objective_terms"]
    assert "do not make y[i,j] binary; it is a nonnegative integer arc facility count" in mcnd_contract["answer_contract"]["forbidden_changes"]
    assert tsp_contract["contract_source"] == "generator_override"
    assert tsp_contract["contract_id"] == "optmath_tsp"
    assert "binary_directed_route_arc" in tsp_contract["answer_contract"]["required_variable_families"]
    assert "single_tour_mtz_subtour_elimination" in tsp_contract["answer_contract"]["required_constraints"]
    assert "do not add vehicle capacity, time windows, service times, delivery quantities, or multiple vehicles" in tsp_contract["answer_contract"]["forbidden_changes"]
    assert shortest_path_contract["contract_source"] == "generator_override"
    assert shortest_path_contract["contract_id"] == "optmath_the_shortest_path_problem"
    assert "binary_path_arc_selection" in shortest_path_contract["answer_contract"]["required_variable_families"]
    assert "source_sink_flow_balance" in shortest_path_contract["answer_contract"]["required_constraints"]
    assert "do not add arc capacity limits, fixed charges, vehicle routes, subtours, time windows, or facility-opening decisions" in shortest_path_contract["answer_contract"]["forbidden_changes"]
    assert scooter_contract["contract_source"] == "generator_override"
    assert scooter_contract["contract_id"] == "optmath_scooter_location"
    assert "nonnegative_integer_new_scooter_count" in scooter_contract["answer_contract"]["required_variable_families"]
    assert "total_new_scooter_budget" in scooter_contract["answer_contract"]["required_constraints"]
    assert "new_scooter_deployment_cost" in scooter_contract["answer_contract"]["required_objective_terms"]
    assert netasgn_contract["contract_source"] == "generator_override"
    assert netasgn_contract["contract_id"] == "optmath_netasgn"
    assert "continuous_assignment_hours" in netasgn_contract["answer_contract"]["required_variable_families"]
    assert "person_supply_hours_equality" in netasgn_contract["answer_contract"]["required_constraints"]
    assert "project_demand_hours_equality" in netasgn_contract["answer_contract"]["required_constraints"]
    assert "person_project_contribution_upper_bound" in netasgn_contract["answer_contract"]["required_constraints"]
    assert "cost_per_hour_times_assigned_hours" in netasgn_contract["answer_contract"]["required_objective_terms"]
    assert "do not relax person supply equality into a less-than-or-equal capacity constraint" in netasgn_contract["answer_contract"]["forbidden_changes"]
    assert "do not reveal or force internally generated feasible allocation certificates" in netasgn_contract["answer_contract"]["forbidden_changes"]
    assert "feasible_allocation_certificate_leaked" in netasgn_contract["answer_contract"]["high_risk_failures"]
    assert net1_contract["contract_source"] == "generator_override"
    assert "node_flow_balance" in net1_contract["answer_contract"]["required_constraints"]
    assert "arc_capacity_limit" in net1_contract["answer_contract"]["required_constraints"]
    assert "unit_arc_flow_cost" in net1_contract["answer_contract"]["required_objective_terms"]
    assert "arc_shipping_cost" not in net1_contract["answer_contract"]["required_objective_terms"]
    assert "do not add vehicles, routes, time windows, subtour constraints, or path-ordering variables" in net1_contract["answer_contract"]["forbidden_changes"]
    assert netthru_contract["contract_source"] == "generator_override"
    assert "node_flow_balance" in netthru_contract["answer_contract"]["required_constraints"]
    assert "directed_link_capacity" in netthru_contract["answer_contract"]["required_constraints"]
    assert "node_throughput_capacity" in netthru_contract["answer_contract"]["required_constraints"]
    assert "do not change the objective to maximum throughput" in netthru_contract["answer_contract"]["forbidden_changes"]
    assert "do not reveal or force internally generated feasible flow certificates" in netthru_contract["answer_contract"]["forbidden_changes"]
    assert "feasible_flow_certificate_leaked" in netthru_contract["answer_contract"]["high_risk_failures"]
    assert steel4_contract["contract_source"] == "generator_override"
    assert steel4_contract["contract_id"] == "optmath_steel4"
    assert "continuous_production_quantity" in steel4_contract["answer_contract"]["required_variable_families"]
    assert "stage_time_capacity_by_processing_hours_per_ton" in steel4_contract["answer_contract"]["required_constraints"]
    assert "profit_per_ton_times_production_tons" in steel4_contract["answer_contract"]["required_objective_terms"]
    assert "do not add inventory carryover, backlog, or storage variables" in steel4_contract["answer_contract"]["forbidden_changes"]
    assert structure_contract["contract_source"] == "generator_override"
    assert structure_contract["contract_id"] == "optmath_structure_based_assignment"
    assert "binary_peak_acid_assignment" in structure_contract["answer_contract"]["required_variable_families"]
    assert "exact_total_assignment_count" in structure_contract["answer_contract"]["required_constraints"]
    assert "noe_compatibility_pair_constraints" in structure_contract["answer_contract"]["required_constraints"]
    assert "do not reinterpret peaks and amino acids as staff, jobs, equipment, gates, vehicles, or routes" in structure_contract["answer_contract"]["forbidden_changes"]
    assert smallbucket_contract["contract_source"] == "generator_override"
    assert smallbucket_contract["contract_id"] == "optmath_singlelevelsmallbucket"
    assert "binary_startup_event" in smallbucket_contract["answer_contract"]["required_variable_families"]
    assert "startup_transition_from_previous_period" in smallbucket_contract["answer_contract"]["required_constraints"]
    assert "do not add a per-unit production cost, material cost, or revenue term" in smallbucket_contract["answer_contract"]["forbidden_changes"]
    assert multi_factory_contract["contract_source"] == "generator_override"
    assert multi_factory_contract["contract_id"] == "optmath_the_multi_factory_schedule_problem"
    assert "binary_factory_run_decision" in multi_factory_contract["answer_contract"]["required_variable_families"]
    assert "monthly_demand_satisfaction" in multi_factory_contract["answer_contract"]["required_constraints"]
    assert "fixed_factory_month_run_cost" in multi_factory_contract["answer_contract"]["required_objective_terms"]
    assert "do not rewrite as job-shop, flow-shop, aircraft landing, or machine sequencing" in multi_factory_contract["answer_contract"]["forbidden_changes"]
    assert team_contract["contract_source"] == "generator_override"
    assert team_contract["contract_id"] == "optmath_team_formulation"
    assert "binary_person_project_assignment" in team_contract["answer_contract"]["required_variable_families"]
    assert "attained_skill_equals_sum_of_assigned_individual_skills" in team_contract["answer_contract"]["required_constraints"]
    assert "maximum_skill_shortage_priority" in team_contract["answer_contract"]["required_objective_terms"]
    assert "do not replace shortage minimization with compatibility-score maximization" in team_contract["answer_contract"]["forbidden_changes"]
    ulsb_contract = resolve_family_contract({"generator_id": "optmath_uncapacitatedlotsizingbacklogging"}, config)
    assert ulsb_contract["contract_source"] == "generator_override"
    assert "net_inventory_balance_with_backlog_carryover" in ulsb_contract["answer_contract"]["required_constraints"]
    assert "do not add production capacity constraints" in ulsb_contract["answer_contract"]["forbidden_changes"]


def test_scenario_catalog_selects_deterministic_generator_specific_scenarios() -> None:
    guidance_a = scenario_guidance_for_generator(
        "optmath_transp",
        instance_id="inst_optmath_transp_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_b = scenario_guidance_for_generator(
        "optmath_transp",
        instance_id="inst_optmath_transp_000001",
        candidate_index=1,
        random_seed=42,
    )

    assert guidance_a == guidance_b
    assert guidance_a["task_family"] == "transportation"
    assert guidance_a["scenario_source"] == "generator_specific"
    assert guidance_a["scenario_id"].startswith("transp_")
    assert guidance_a["available_scenario_count"] >= 10
    assert guidance_a["available_scenario_variant_count"] >= 50
    assert guidance_a["scenario_variant"]["scenario_variant_id"]
    assert "complete lane cost matrix" in guidance_a["applicability"]["required_source_features"]
    assert "vehicles or vehicle routes" in guidance_a["applicability"]["incompatible_source_features"]
    assert guidance_a["selected_business_trigger"]
    assert "required_source_features" in guidance_a["applicability"]
    assert "incompatible_source_features" in guidance_a["applicability"]
    assert "vehicle routes" in guidance_a["applicability"]["incompatible_source_features"]
    assert guidance_a["units"]["cost"]
    assert guidance_a["units"]["supply"]
    assert guidance_a["units"]["demand"]


def test_scenario_catalog_expands_generator_specific_scenarios_with_variants() -> None:
    guidance = scenario_guidance_for_generator(
        "optmath_cflp",
        instance_id="inst_optmath_cflp_000001",
        candidate_index=1,
        random_seed=42,
    )
    scenario_ids = {
        scenario_guidance_for_generator(
            "optmath_cflp",
            instance_id=f"inst_optmath_cflp_{idx:06d}",
            candidate_index=(idx % 2) + 1,
            random_seed=42,
        )["scenario_id"]
        for idx in range(20)
    }

    assert guidance["available_scenario_variant_count"] >= 50
    assert guidance["available_scenario_count"] >= 10
    assert guidance["base_scenario_id"].startswith("cflp_")
    assert guidance["scenario_variant_id"]
    assert guidance["scenario_id"].startswith(guidance["base_scenario_id"])
    assert guidance["scenario_variant"]["industry_lens_id"]
    assert guidance["scenario_variant"]["narrative_angle_id"]
    assert guidance["scenario_variant"]["organization_profile_id"]
    assert "binary facility opening variable" in guidance["applicability"]["required_source_features"]
    assert "pure transportation without fixed opening decisions" in guidance["applicability"]["incompatible_source_features"]
    assert len(scenario_ids) >= 8


def test_high_failure_generators_have_deeper_business_scenario_coverage() -> None:
    for generator_id in (
        "optmath_setcover",
        "optmath_multisetcover",
        "optmath_schedulingproblem",
        "optmath_flowshop_2",
        "optmath_fleet_routing",
        "optmath_facility_location",
        "optmath_jopshop",
        "optmath_supplychain",
        "optmath_uncapacitatedlotsizingbacklogging",
        "optmath_diet_problem",
        "optmath_the_military_personnel_deployment_problem",
        "optmath_structure_based_assignment",
        "optmath_factory_planning_problem",
        "optmath_portfolio",
        "optmath_contractallocation",
        "optmath_carselection",
        "optmath_cell_tower",
        "optmath_cut_edited",
        "optmath_vrptw",
        "optmath_the_p_dispersion_model",
        "optmath_aircraftassignment",
        "optmath_aircraftlanding",
        "optmath_farmplanning",
        "optmath_scooter_location",
        "optmath_mcnd",
        "optmath_nltrans",
        "optmath_netasgn",
        "optmath_netmcol",
        "optmath_net1",
        "optmath_netthru",
        "optmath_the_shortest_path_problem",
        "optmath_team_formulation",
        "optmath_the_multi_factory_schedule_problem",
        "optmath_tsp",
        "optmath_steel4",
        "optmath_revenue",
    ):
        guidance = scenario_guidance_for_generator(
            generator_id,
            instance_id=f"inst_{generator_id}_000001",
            candidate_index=1,
            random_seed=42,
        )
        assert guidance["available_scenario_count"] >= 10
        assert guidance["available_scenario_variant_count"] >= 50
        assert guidance["selected_business_trigger"]


def test_revenue_management_scenario_guidance_preserves_static_capacity_allocation() -> None:
    guidance = scenario_guidance_for_generator(
        "optmath_revenue_management",
        instance_id="inst_optmath_revenue_management_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert guidance["task_family"] == "revenue_management"
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "package-resource usage matrix" in guidance["applicability"]["required_source_features"]
    assert "dynamic pricing periods" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as dynamic pricing." in guidance["forbidden_misframings"]
    assert "resource capacity" in guidance_text


def test_remaining_refined_generator_profiles_and_scenarios_preserve_source_structure() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))["profiles"]

    car_profile = profiles["optmath_carselection"]
    car_guidance = scenario_guidance_for_generator(
        "optmath_carselection",
        instance_id="inst_optmath_carselection_000001",
        candidate_index=1,
        random_seed=42,
    )
    assert car_profile["canonical_math_signature"]["objective_sense"] == "maximize"
    assert "compact_car_selection_tables" in car_profile["canonical_math_signature"]["required_tables"]
    assert "assignment_only_if_eligible" in car_profile["canonical_math_signature"]["core_constraints"]
    assert car_guidance["available_scenario_count"] >= 10
    assert car_guidance["available_scenario_variant_count"] >= 50
    assert "binary participant-car eligibility matrix" in car_guidance["applicability"]["required_source_features"]
    assert "assignment costs or scores" in car_guidance["applicability"]["incompatible_source_features"]
    car_guidance_text = json.dumps(
        {
            "selected_scenario": car_guidance["selected_scenario"],
            "scenario_variant": car_guidance["scenario_variant"],
        },
        ensure_ascii=False,
    ).lower()
    for forbidden in ("technician", "technicians", "job matching", "worker assignment", "shift scheduling", "staffing"):
        assert forbidden not in car_guidance_text
    assert car_guidance["scenario_variant"]["industry_lens_id"] in {
        "participant_vehicle_access",
        "corporate_pool_vehicle_assignment",
        "campus_motor_pool_assignment",
        "healthcare_transport_asset_access",
        "emergency_response_vehicle_access",
        "demo_fleet_vehicle_assignment",
    }

    structure_profile = profiles["optmath_structure_based_assignment"]
    structure_guidance = scenario_guidance_for_generator(
        "optmath_structure_based_assignment",
        instance_id="inst_optmath_structure_based_assignment_000001",
        candidate_index=1,
        random_seed=42,
    )
    assert structure_profile["sub_family"] == "nmr_peak_amino_acid_structure_assignment"
    assert "amino_acid_distance_or_compatibility_matrix" in structure_profile["canonical_math_signature"]["required_tables"]
    assert structure_guidance["available_scenario_count"] >= 10
    assert structure_guidance["available_scenario_variant_count"] >= 50
    structure_guidance_text = json.dumps(
        {
            "selected_scenario": structure_guidance["selected_scenario"],
            "scenario_variant": structure_guidance["scenario_variant"],
        },
        ensure_ascii=False,
    ).lower()
    for forbidden in (
        "technician",
        "technicians",
        "staffing",
        "worker",
        "workers",
        "job matching",
        "equipment",
        "gate assignment",
        "crew assignment",
        "vehicle routing",
    ):
        assert forbidden not in structure_guidance_text
    assert structure_guidance["scenario_variant"]["industry_lens_id"] in {
        "nmr_peak_annotation",
        "protein_resonance_mapping",
        "molecular_spectroscopy_curation",
        "structural_proteomics_validation",
        "drug_discovery_nmr_annotation",
        "biophysics_distance_evidence",
    }

    cell_profile = profiles["optmath_cell_tower"]
    cell_guidance = scenario_guidance_for_generator(
        "optmath_cell_tower",
        instance_id="inst_optmath_cell_tower_000001",
        candidate_index=1,
        random_seed=42,
    )
    assert cell_profile["canonical_math_signature"]["objective_sense"] == "maximize"
    assert "compact_cell_tower_tables" in cell_profile["canonical_math_signature"]["required_tables"]
    assert "region_coverage_linking" in cell_profile["canonical_math_signature"]["core_constraints"]
    assert "continuous_nonnegative_shipment" not in cell_profile["canonical_math_signature"]["variable_types"]
    assert cell_guidance["available_scenario_count"] >= 10
    assert cell_guidance["available_scenario_variant_count"] >= 50
    assert "tower-building budget" in cell_guidance["applicability"]["required_source_features"]
    assert "continuous shipments" in cell_guidance["applicability"]["incompatible_source_features"]

    revenue_profile = profiles["optmath_revenue"]
    revenue_guidance = scenario_guidance_for_generator(
        "optmath_revenue",
        instance_id="inst_optmath_revenue_000001",
        candidate_index=1,
        random_seed=42,
    )
    assert revenue_profile["canonical_math_signature"]["objective_sense"] == "maximize"
    assert "compact_revenue_management_tables" in revenue_profile["canonical_math_signature"]["required_tables"]
    assert "package_demand_upper_bound" in revenue_profile["canonical_math_signature"]["core_constraints"]
    assert revenue_guidance["available_scenario_count"] >= 10
    assert revenue_guidance["available_scenario_variant_count"] >= 50
    assert "package-resource usage matrix" in revenue_guidance["applicability"]["required_source_features"]
    assert "dynamic pricing" in revenue_guidance["applicability"]["incompatible_source_features"]

    marketshare_profile = profiles["optmath_marketshare"]
    marketshare_guidance = scenario_guidance_for_generator(
        "optmath_marketshare",
        instance_id="inst_optmath_marketshare_000001",
        candidate_index=1,
        random_seed=42,
    )
    assert marketshare_profile["canonical_math_signature"]["objective_sense"] == "maximize"
    assert "nonnegative_integer_company_market_product_supply" in marketshare_profile["canonical_math_signature"]["variable_types"]
    assert "market_product_demand_exactly_satisfied" in marketshare_profile["canonical_math_signature"]["core_constraints"]
    assert "compact_marketshare_tables" in marketshare_profile["canonical_math_signature"]["required_tables"]
    assert not any("budget_allocation" == concept for concept in marketshare_profile["modeling_concepts"])
    assert any("no resource capacity" in item for item in marketshare_profile["rationale_requirements"])
    assert marketshare_guidance["available_scenario_count"] >= 10
    assert marketshare_guidance["available_scenario_variant_count"] >= 50


def test_facility_location_profile_and_scenario_guidance_preserve_multicommodity_dc_selection() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_facility_location"]
    guidance = scenario_guidance_for_generator(
        "optmath_facility_location",
        instance_id="inst_optmath_facility_location_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "facility_location"
    assert profile["sub_family"] == "multi_commodity_distribution_center_selection"
    assert "open_before_zone_assignment_linking" in profile["modeling_concepts"]
    assert (
        "explicit_distribution_center_open_before_zone_assignment"
        in profile["canonical_math_signature"]["core_constraints"]
    )
    assert "fixed_unit_and_throughput_bounds_by_distribution_center" in profile["canonical_math_signature"]["required_tables"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list variable shipping costs indexed by commodity, plant, distribution center, and customer zone" in guidance[
        "required_parameter_presentation"
    ]
    assert "state a zone can be assigned to a distribution center only if that center is selected/opened" in guidance[
        "required_parameter_presentation"
    ]
    assert any("vehicle routes" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not introduce vehicles" in item.lower() for item in guidance["forbidden_misframings"])
    assert "served[d,z] <= selected[d]" in guidance_text
    assert "four-index variable shipping cost table" in guidance_text
    assert guidance["scenario_variant"]["industry_lens_id"] in {
        "multicommodity_dc_network_redesign",
        "healthcare_supply_hub_assignment",
        "relief_warehouse_zone_assignment",
        "pharma_compliant_distribution_centers",
        "retail_fulfillment_center_selection",
    }
    for candidate_index in range(1, 25):
        variant = scenario_guidance_for_generator(
            "optmath_facility_location",
            instance_id=f"inst_optmath_facility_location_{candidate_index:06d}",
            candidate_index=candidate_index,
            random_seed=42,
        )["scenario_variant"]
        assert variant["industry_lens_id"] in {
            "multicommodity_dc_network_redesign",
            "healthcare_supply_hub_assignment",
            "relief_warehouse_zone_assignment",
            "pharma_compliant_distribution_centers",
            "retail_fulfillment_center_selection",
        }


def test_fleet_routing_profile_and_scenario_guidance_preserve_time_expanded_fleet_flow() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_fleet_routing"]
    guidance = scenario_guidance_for_generator(
        "optmath_fleet_routing",
        instance_id="inst_optmath_fleet_routing_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "network_flow"
    assert profile["sub_family"] == "time_expanded_fleet_flow"
    assert "integer_fleet_assignment" in profile["modeling_concepts"]
    assert "inactive_leg_route_restriction" in profile["modeling_concepts"]
    assert "time_expanded_fleet_flow_conservation" in profile["canonical_math_signature"]["core_constraints"]
    assert "leg_is_active_indicator" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_visit_once_logic" not in profile["quality_controls"]["validation_required"]
    assert "preserve_time_expanded_fleet_balance" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state variables are nonnegative integer fleet counts, idle fleet counts, and initial fleet placement counts" in guidance[
        "required_parameter_presentation"
    ]
    assert any("binary route arc variables" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not introduce binary route-arc" in item.lower() for item in guidance["forbidden_misframings"])
    assert guidance["scenario_variant"]["industry_lens_id"] in {
        "airline_scheduled_leg_assignment",
        "ferry_scheduled_sailing_balance",
        "intercity_bus_departure_block",
        "cargo_air_leg_capacity",
        "medical_transfer_leg_assignment",
    }
    for candidate_index in range(1, 25):
        sampled_guidance = scenario_guidance_for_generator(
            "optmath_fleet_routing",
            instance_id=f"inst_optmath_fleet_routing_{candidate_index:06d}",
            candidate_index=candidate_index,
            random_seed=42,
        )
        variant_text = json.dumps(sampled_guidance["scenario_variant"], ensure_ascii=False).lower()
        assert sampled_guidance["scenario_variant"]["industry_lens_id"] in {
            "airline_scheduled_leg_assignment",
            "ferry_scheduled_sailing_balance",
            "intercity_bus_departure_block",
            "cargo_air_leg_capacity",
            "medical_transfer_leg_assignment",
        }
        assert "inspection_or_audit_tour" not in variant_text
        assert "visit" not in variant_text
    assert "time-expanded fleet conservation" in guidance_text
    assert "leg_is_active" in guidance_text


def test_portfolio_profile_and_scenario_guidance_preserve_continuous_qp() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_portfolio"]
    guidance = scenario_guidance_for_generator(
        "optmath_portfolio",
        instance_id="inst_optmath_portfolio_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["sub_family"] == "continuous_mean_variance_portfolio"
    assert "continuous_asset_weight" in profile["canonical_math_signature"]["variable_types"]
    assert "covariance_matrix" in profile["canonical_math_signature"]["required_tables"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state expected return is a constraint, not the maximization objective" in guidance[
        "required_parameter_presentation"
    ]
    assert any("binary asset-selection" in item.lower() for item in guidance["forbidden_misframings"])
    assert "covariance" in guidance_text
    assert guidance["scenario_variant"]["industry_lens_id"] in {
        "investment_committee_asset_mix",
        "pension_liability_reserve_mix",
        "treasury_liquidity_allocation",
        "endowment_risk_budget_allocation",
        "insurance_surplus_variance_control",
    }
def test_supplychain_profile_and_scenario_guidance_preserve_sparse_fixed_charge_network() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_supplychain"]
    guidance = scenario_guidance_for_generator(
        "optmath_supplychain",
        instance_id="inst_optmath_supplychain_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "supply_chain"
    assert profile["sub_family"] == "fixed_charge_capacitated_network_flow"
    assert "sparse_declared_arc_set" in profile["modeling_concepts"]
    assert "compact_supplychain_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "arc_table_with_capacity_fixed_cost_and_unit_cost" in profile["canonical_math_signature"]["required_tables"]
    assert "node_balance_rhs_demand_minus_supply" in profile["canonical_math_signature"]["required_tables"]
    assert "do not reveal or force internally generated feasible flow certificates" in profile[
        "canonical_math_signature"
    ]["forbidden_changes"]
    assert "preserve_declared_sparse_arc_set" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list only declared directed arcs; do not describe the network as complete" in guidance[
        "required_parameter_presentation"
    ]
    assert "state node balance as inflow minus outflow equals demand minus supply" in guidance[
        "required_parameter_presentation"
    ]
    assert any("node facility opening variables" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not imply the network is a complete directed graph" in item.lower() for item in guidance["forbidden_misframings"])
    assert "sparse declared directed arc set" in guidance_text
    assert "inflow minus outflow equals demand minus supply" in guidance_text


def test_steel4_profile_and_scenario_guidance_preserve_stage_capacity_product_mix() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_steel4"]
    guidance = scenario_guidance_for_generator(
        "optmath_steel4",
        instance_id="inst_optmath_steel4_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "production_planning"
    assert profile["sub_family"] == "continuous_stage_capacity_product_mix"
    assert "processing_hours_per_ton" in profile["modeling_concepts"]
    assert "stage_time_capacity_by_processing_hours_per_ton" in profile["canonical_math_signature"]["core_constraints"]
    assert "capacity_envelope_by_stage" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_stage_capacity_constraints" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state production variables are continuous tons, not binary product selection" in guidance[
        "required_parameter_presentation"
    ]
    assert "state the model has no inventory, backlog, setup, sequencing, or time-period carryover" in guidance[
        "required_parameter_presentation"
    ]
    assert any("time periods" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not describe stages as time periods" in item.lower() for item in guidance["forbidden_misframings"])
    assert "processing hours per ton" in guidance_text
    assert "minimum commitment" in guidance_text
    assert "maximum market" in guidance_text


def test_multi_factory_schedule_profile_and_scenario_guidance_preserve_fixed_charge_production() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_the_multi_factory_schedule_problem"]
    guidance = scenario_guidance_for_generator(
        "optmath_the_multi_factory_schedule_problem",
        instance_id="inst_optmath_the_multi_factory_schedule_problem_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "production_planning"
    assert profile["sub_family"] == "fixed_charge_multi_factory_production"
    assert "binary_factory_run_decision" in profile["modeling_concepts"]
    assert "monthly_demand_satisfaction" in profile["canonical_math_signature"]["core_constraints"]
    assert "factory_fixed_run_costs" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_fixed_charge_run_costs" in profile["quality_controls"]["validation_required"]
    assert "compact_multi_factory_schedule_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "sequencing" not in profile["modeling_concepts"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "show each factory's fixed run cost per month" in guidance["required_parameter_presentation"]
    assert "show monthly aggregate demand" in guidance["required_parameter_presentation"]
    assert any("job sequencing" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not describe this as job-shop" in item.lower() for item in guidance["forbidden_misframings"])
    assert "fixed run cost" in guidance_text
    assert "monthly demand" in guidance_text


def test_multi_factory_schedule_generator_records_fixed_charge_production_tables() -> None:
    module_path = Path(
        "or_cpt_engine/generators/optmath_seed/The_multi-factory_schedule_problem/The_multi-factory_schedule_problem.py"
    )
    spec = importlib.util.spec_from_file_location("multi_factory_schedule_generator", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = module.Generator(seed=7)
    model = generator.generate_instance()
    model.Params.OutputFlag = 0
    model.optimize()
    params = generator.parameters

    assert model.Status == 2
    assert params["factories"]
    assert params["months"]
    assert params["factory_table"]
    assert params["month_demand_table"]
    assert params["compact_multi_factory_schedule_tables"]
    assert params["structured_problem_data"]
    assert "fixed_run_cost" in params["objective_terms"]
    assert "minimum_production_if_running" in params["required_constraints"]
    compact_source = _compact_source_data({"generation_params": params})
    assert compact_source["available"] is True
    assert "compact_multi_factory_schedule_tables" in compact_source["value"]
    assert "monthly_demand_satisfaction" in params["required_constraints"]
    assert "for every factory, list fixed run cost per month" in params["required_parameter_presentation"]
    assert "This is a fixed-charge multi-factory monthly production planning model." in params[
        "business_interpretation_guardrails"
    ]
    assert all(row["requires_multiple_factories"] for row in params["month_demand_table"].values())
    for factory, row in params["factory_table"].items():
        assert row["minimum_production_if_running"] <= row["maximum_production_if_running"]
        assert factory in params["fixed_costs"]


def test_team_formulation_profile_and_scenario_guidance_preserve_skill_shortage_model() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_team_formulation"]
    guidance = scenario_guidance_for_generator(
        "optmath_team_formulation",
        instance_id="inst_optmath_team_formulation_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "workforce_planning"
    assert profile["sub_family"] == "team_skill_shortage_assignment"
    assert "skill_shortage_variable" in profile["modeling_concepts"]
    assert "attained_skill_equals_sum_of_assigned_individual_skills" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_team_formulation_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "compatibility-score maximization" in " ".join(profile["canonical_math_signature"]["forbidden_changes"])
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list individual skill levels for every person-skill pair" in guidance["required_parameter_presentation"]
    assert "compatibility-score maximization" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not omit AttainedSkill, SkillShortage, or MaxSkillShortage." in guidance["forbidden_misframings"]
    assert "skill shortage" in guidance_text


def test_flowshop_profile_and_scenario_guidance_preserve_common_permutation_makespan() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_flowshop_2"]
    guidance = scenario_guidance_for_generator(
        "optmath_flowshop_2",
        instance_id="inst_optmath_flowshop_2_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["sub_family"] == "permutation_flow_shop_makespan"
    assert "common_sequence_across_all_machines" in profile["modeling_concepts"]
    assert "explicit_makespan_variable" in profile["modeling_concepts"]
    assert "explicit_makespan_variable" in profile["canonical_math_signature"]["objective_terms"]
    assert "makespan_completion_bound" in profile["canonical_math_signature"]["core_constraints"]
    assert "processing_time_table_by_job_machine" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_common_permutation_sequence" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state every machine uses the same sequence of shared positions" in guidance[
        "required_parameter_presentation"
    ]
    assert "state the objective minimizes an explicit Makespan variable" in guidance[
        "required_parameter_presentation"
    ]
    assert any("explicit Makespan variable" in item for item in guidance["applicability"]["required_source_features"])
    assert any("machine-specific job sequences" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("weighted completion costs" in item for item in guidance["applicability"]["incompatible_source_features"])
    assert any("do not describe each machine as choosing its own sequence" in item.lower() for item in guidance["forbidden_misframings"])
    assert "common permutation sequence" in guidance_text
    assert "explicit makespan" in guidance_text


def test_flowshop_generator_uses_explicit_makespan_variable() -> None:
    module_path = Path("or_cpt_engine/generators/optmath_seed/Flowshop_2/Flowshop_2_parsed.py")
    spec = importlib.util.spec_from_file_location("flowshop_generator", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    generator = module.Generator(seed=7)
    model = generator.generate_instance()
    model.update()

    assert any(variable.VarName == "Makespan" for variable in model.getVars())
    assert "Makespan" in generator.parameters["decision_variables"]
    assert "makespan_completion_bound" in generator.parameters["required_constraints"]


def test_netasgn_profile_and_scenario_guidance_preserve_continuous_balanced_assignment() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_netasgn"]
    guidance = scenario_guidance_for_generator(
        "optmath_netasgn",
        instance_id="inst_optmath_netasgn_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["sub_family"] == "continuous_resource_assignment"
    assert "continuous_assignment_hours" in profile["modeling_concepts"]
    assert "balanced_supply_demand_hours" in profile["modeling_concepts"]
    assert "cost_per_hour_times_assigned_hours" in profile["canonical_math_signature"]["objective_terms"]
    assert "complete_max_contribution_hours_matrix" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_complete_pairwise_upper_bounds" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state x[i,j] is continuous nonnegative assigned hours" in guidance["required_parameter_presentation"]
    assert "state total available hours equals total required hours" in guidance["required_parameter_presentation"]
    assert "binary one-to-one assignment" in guidance["applicability"]["incompatible_source_features"]
    assert any("do not convert the model into binary one-to-one matching" in item.lower() for item in guidance["forbidden_misframings"])
    assert any("feasible allocation certificate" in item.lower() for item in guidance["forbidden_misframings"])
    assert any("feasible allocation certificates are generator-only" in item.lower() for item in guidance["modeling_notes"])
    assert "balanced transportation-style continuous allocation" in guidance_text
    assert "complete maximum contribution matrix" in guidance_text


def test_multisetcover_profile_and_scenario_guidance_preserve_complete_multicover_matrix() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_multisetcover"]
    guidance = scenario_guidance_for_generator(
        "optmath_multisetcover",
        instance_id="inst_optmath_multisetcover_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["sub_family"] == "set_multicover"
    assert "binary_set_selection" in profile["modeling_concepts"]
    assert "set_multicover_requirement" in profile["modeling_concepts"]
    assert "selected_set_cost" in profile["canonical_math_signature"]["objective_terms"]
    assert "compact_multisetcover_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "set_to_element_coverage_membership" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_complete_coverage_matrix" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list the complete coverage membership for every candidate set" in guidance[
        "required_parameter_presentation"
    ]
    assert "state this is multicover, so requirements may be greater than one" in guidance[
        "required_parameter_presentation"
    ]
    assert "exact-cover equality requirements" in guidance["applicability"]["incompatible_source_features"]
    assert any("do not downgrade multicover" in item.lower() for item in guidance["forbidden_misframings"])
    assert "complete coverage matrix" in guidance_text
    assert "coverage requirements as lower bounds" in guidance_text


def test_cutting_stock_profile_and_scenario_guidance_preserve_integer_pattern_counts() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_cut_edited"]
    guidance = scenario_guidance_for_generator(
        "optmath_cut_edited",
        instance_id="inst_optmath_cut_edited_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["sub_family"] == "integer_cutting_stock_pattern_count"
    assert "nonnegative_integer_pattern_count" in profile["modeling_concepts"]
    assert "exact_order_width_fulfillment" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_cutting_stock_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "do not make pattern-count variables binary" in profile["canonical_math_signature"]["forbidden_changes"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list every cutting pattern and piece counts by width" in guidance["required_parameter_presentation"]
    assert "binary set selection" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not make Cut[j] binary." in guidance["forbidden_misframings"]
    assert "pattern-count" in guidance_text or "cut[j]" in guidance_text


def test_staticlineplanning_profile_and_scenario_guidance_preserve_line_frequency_model() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_staticlineplanning"]
    guidance = scenario_guidance_for_generator(
        "optmath_staticlineplanning",
        instance_id="inst_optmath_staticlineplanning_000001",
        candidate_index=1,
        random_seed=42,
    )
    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()

    assert profile["task_family"] == "network_design"
    assert profile["sub_family"] == "static_line_frequency_planning"
    assert "binary_line_activation" in profile["modeling_concepts"]
    assert "continuous_line_frequency" in profile["modeling_concepts"]
    assert "od_service_coverage_matrix" in profile["modeling_concepts"]
    assert "fixed_line_activation_cost" in profile["canonical_math_signature"]["objective_terms"]
    assert "od_demand_coverage_with_shortage" in profile["canonical_math_signature"]["core_constraints"]
    assert "candidate_line_cost_capacity_frequency_table" in profile["canonical_math_signature"]["required_tables"]
    assert "preserve_service_coverage_matrix" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert any(
        "list every candidate line with fixed cost" in item
        for item in guidance["required_parameter_presentation"]
    )
    assert any(
        "list the candidate lines that can serve each od pair" in item.lower()
        for item in guidance["required_parameter_presentation"]
    )
    assert any(
        "vehicle route sequences" in item
        for item in guidance["applicability"]["incompatible_source_features"]
    )
    assert any("do not invent z[i,j,l]" in item.lower() for item in guidance["forbidden_misframings"])
    assert "continuous service frequency" in guidance_text
    assert "fleet budget" in guidance_text


def test_backtranslation_profile_normalization_refreshes_stale_fleet_routing_signature() -> None:
    from or_cpt_engine.schemas.common import EngineConfig, PathsConfig

    stale_row = {
        "instance_id": "inst_optmath_fleet_routing_legacy",
        "generator_id": "optmath_fleet_routing",
        "generator_profile_version": "legacy",
        "sub_family": "fleet_routing",
        "concept_tags": ["routing_binary_variable", "visit_once", "capacity_or_time_window"],
        "canonical_math_signature": {
            "variable_types": ["binary_arc_or_visit"],
            "core_constraints": ["visit_once", "route_continuity", "capacity_or_time_window"],
        },
    }
    config = EngineConfig(paths=PathsConfig(generator_profiles_path="engine_configs/generator_profiles.yaml"))

    normalized = _profile_normalized_row(stale_row, config)

    assert normalized["generator_profile_version"] == "v1.1.0"
    assert normalized["sub_family"] == "time_expanded_fleet_flow"
    assert "integer_fleet_assignment" in normalized["concept_tags"]
    assert "visit_once" not in normalized["concept_tags"]
    assert "nonnegative_integer_flight_leg_count" in normalized["canonical_math_signature"]["variable_types"]
    assert "time_expanded_fleet_flow_conservation" in normalized["canonical_math_signature"]["core_constraints"]
    assert normalized["llm_metadata"]["profile_normalized"] is True
    assert normalized["llm_metadata"]["profile_normalized_from"] == "legacy"


def test_factory_planning_scenario_guidance_preserves_multi_period_inventory_structure() -> None:
    guidance = scenario_guidance_for_generator(
        "optmath_factory_planning_problem",
        instance_id="inst_optmath_factory_planning_problem_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "show available machine hours by period and machine" in guidance["required_parameter_presentation"]
    assert "show sales upper bounds by period and product" in guidance["required_parameter_presentation"]
    assert "use compact_factory_planning_tables when available" in guidance["required_parameter_presentation"]
    assert "do not say sales are unbounded." in [value.lower() for value in guidance["forbidden_misframings"]]
    assert "do not infer demand totals from the reference answer." in [
        value.lower() for value in guidance["forbidden_misframings"]
    ]
    assert "product-period production, sales, and ending inventory variables" in guidance["applicability"]["required_source_features"]
    assert "public program activity mix" not in guidance_text
    assert guidance["scenario_variant"]["industry_lens_id"] in {
        "factory_monthly_production_inventory",
        "packaging_line_production_inventory",
        "specialty_materials_production_plan",
        "electronics_assembly_capacity_plan",
        "industrial_spares_sop_plan",
        "medical_device_batch_plan",
    }


def test_forward_modeling_normalizes_stale_factory_profile_before_contract_resolution() -> None:
    import yaml

    config = EngineConfig(
        family_contracts=yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    )
    stale_row = {
        "bt_id": "bt_factory_stale",
        "instance_id": "inst_optmath_factory_planning_problem_000001",
        "generator_id": "optmath_factory_planning_problem",
        "generator_profile_version": "legacy",
        "sub_family": "production_planning",
        "concept_tags": ["product_mix", "resource_capacity", "profit_or_cost_objective", "demand_bound"],
        "canonical_math_signature": {
            "variable_types": ["continuous_or_integer_as_source"],
            "core_constraints": ["linear_capacity_or_requirement"],
        },
    }

    normalized = normalize_forward_profile_row(stale_row, config)
    contract = resolve_family_contract(normalized, config.family_contracts)

    assert normalized["generator_profile_version"] == "v1.1.0"
    assert normalized["sub_family"] == "multi_period_factory_production_inventory"
    assert "multi_period_production" in normalized["concept_tags"]
    assert "product_mix" not in normalized["concept_tags"]
    assert "machine_hour_capacity_after_maintenance" in normalized["canonical_math_signature"]["core_constraints"]
    assert "sales_upper_bound_by_period_product" in normalized["canonical_math_signature"]["core_constraints"]
    assert contract["contract_id"] == "optmath_factory_planning_problem"
    assert contract["contract_source"] == "generator_override"


def test_vrptw_scenario_guidance_preserves_single_vehicle_cvrptw_structure() -> None:
    guidance = scenario_guidance_for_generator(
        "optmath_vrptw",
        instance_id="inst_optmath_vrptw_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state that exactly one vehicle is available" in guidance["required_parameter_presentation"]
    assert "state there is exactly one depot-to-depot route, not a fleet of routes" in guidance[
        "required_parameter_presentation"
    ]
    assert "list customer demand, service time, and time window for every customer" in guidance["required_parameter_presentation"]
    assert "exactly one vehicle" in guidance["applicability"]["required_source_features"]
    assert "binary directed arc routing variables" in guidance["applicability"]["required_source_features"]
    assert any("fleet" in item.lower() and "all routes" in item.lower() for item in guidance["forbidden_misframings"])
    assert any("internally generated feasible route" in item.lower() for item in guidance["forbidden_misframings"])
    assert "multiple vehicle types or fleet-sizing decisions" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not introduce multiple vehicles." in guidance["forbidden_misframings"]
    assert "single-vehicle" in guidance_text or "one vehicle" in guidance_text


def test_p_dispersion_profile_and_scenario_guidance_do_not_use_facility_location_contract() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_the_p_dispersion_model"]
    guidance = scenario_guidance_for_generator(
        "optmath_the_p_dispersion_model",
        instance_id="inst_optmath_the_p_dispersion_model_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "dispersion"
    assert profile["sub_family"] == "p_dispersion_min_distance"
    assert "fixed_cost" not in profile["modeling_concepts"]
    assert "binary_node_selection" in profile["modeling_concepts"]
    assert "select_exactly_p_nodes" in profile["canonical_math_signature"]["core_constraints"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state the exact number p of nodes to select" in guidance["required_parameter_presentation"]
    assert "fixed opening costs" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not add fixed opening costs." in guidance["forbidden_misframings"]
    assert "customer demand satisfaction" in guidance["applicability"]["incompatible_source_features"]
    assert "max-min" in guidance_text or "minimum distance" in guidance_text


def test_aircraft_scenario_guidance_preserves_assignment_vs_landing_distinction() -> None:
    assignment_guidance = scenario_guidance_for_generator(
        "optmath_aircraftassignment",
        instance_id="inst_optmath_aircraftassignment_000001",
        candidate_index=1,
        random_seed=42,
    )
    landing_guidance = scenario_guidance_for_generator(
        "optmath_aircraftlanding",
        instance_id="inst_optmath_aircraftlanding_000001",
        candidate_index=1,
        random_seed=42,
    )

    assignment_text = json.dumps(assignment_guidance, ensure_ascii=False).lower()
    landing_text = json.dumps(landing_guidance, ensure_ascii=False).lower()
    assert assignment_guidance["available_scenario_count"] >= 10
    assert landing_guidance["available_scenario_count"] >= 10
    assert "nonnegative integer aircraft-route allocation variables" in assignment_guidance["applicability"]["required_source_features"]
    assert "fleet assignment or tail routing" in landing_guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as aircraft routing or tail rotation." in assignment_guidance["forbidden_misframings"]
    assert "Do not describe this as aircraft fleet assignment." in landing_guidance["forbidden_misframings"]
    assert "capacity contribution" in assignment_text
    assert "separation" in landing_text
    assert "early and late" in landing_text


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_aircraft_assignment_generator_tightens_core_constraints() -> None:
    from or_cpt_engine.generators.optmath_seed.AircraftAssignment.AircraftAssignment import (
        Generator as AircraftAssignment,
    )

    for seed in range(8):
        generator = AircraftAssignment(seed=seed)
        model = generator.generate_instance()
        model.optimize()
        core_binding = sum(
            1
            for constraint in model.getConstrs()
            if constraint.ConstrName.startswith(("Availability_", "Demand_"))
            and abs(float(constraint.Slack)) <= 1e-6
        )

        assert model.Status == 2
        assert core_binding >= 1
        assert generator.parameters["compact_aircraft_assignment_tables"]
        assert generator.parameters["postprocess_iterations"] >= 1


def test_farmplanning_profile_and_scenario_guidance_preserve_continuous_resource_lp() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_farmplanning"]
    guidance = scenario_guidance_for_generator(
        "optmath_farmplanning",
        instance_id="inst_optmath_farmplanning_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["sub_family"] == "continuous_farm_resource_planning"
    assert "continuous_crop_area" in profile["modeling_concepts"]
    assert "family_consumption_bundle_fraction_sum" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_farm_planning_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "binary crop-selection" in " ".join(profile["canonical_math_signature"]["forbidden_changes"])
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list crops, months, and family consumption bundles" in guidance["required_parameter_presentation"]
    assert "binary crop-selection variables" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not omit family consumption bundle fractions." in guidance["forbidden_misframings"]
    assert "yield split" in guidance_text


def test_nltrans_profile_and_scenario_guidance_preserve_balanced_transportation_lp() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_nltrans"]
    guidance = scenario_guidance_for_generator(
        "optmath_nltrans",
        instance_id="inst_optmath_nltrans_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["sub_family"] == "capacitated_balanced_transportation"
    assert "exact_supply_balance" in profile["modeling_concepts"]
    assert "lane_capacity_limit" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_transportation_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "lane_capacity_matrix" in profile["canonical_math_signature"]["required_tables"]
    assert "do not reveal or force internally generated feasible shipment certificates" in profile[
        "canonical_math_signature"
    ]["forbidden_changes"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state total supply equals total demand" in guidance["required_parameter_presentation"]
    assert "vehicles or vehicle routes" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as vehicle routing." in guidance["forbidden_misframings"]
    assert "lane capacity" in guidance_text
    assert "total supply equals total demand" in guidance_text


def test_netmcol_profile_and_scenario_guidance_preserve_continuous_multicommodity_flow() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_netmcol"]
    guidance = scenario_guidance_for_generator(
        "optmath_netmcol",
        instance_id="inst_optmath_netmcol_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "network_flow"
    assert profile["sub_family"] == "continuous_multi_commodity_flow"
    assert "fixed_charge" not in profile["modeling_concepts"]
    assert "continuous_multi_commodity_flow" in profile["modeling_concepts"]
    assert "directed_link_joint_capacity" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_netmcol_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "do not reveal or force internally generated feasible flow certificates" in profile[
        "canonical_math_signature"
    ]["forbidden_changes"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list cities, products, and directed links" in guidance["required_parameter_presentation"]
    assert "binary arc activation" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as fixed-charge network design." in guidance["forbidden_misframings"]
    assert "joint capacity" in guidance_text
    assert "product-specific" in guidance_text


def test_multi_profile_and_scenario_guidance_preserve_balanced_multicommodity_transport() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_multi"]
    guidance = scenario_guidance_for_generator(
        "optmath_multi",
        instance_id="inst_optmath_multi_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "transportation"
    assert profile["sub_family"] == "balanced_multi_commodity_transportation"
    assert "continuous_multicommodity_shipment" in profile["modeling_concepts"]
    assert "line_activation" not in profile["modeling_concepts"]
    assert "origin_product_supply_equality" in profile["canonical_math_signature"]["core_constraints"]
    assert "destination_product_demand_equality" in profile["canonical_math_signature"]["core_constraints"]
    assert "origin_destination_joint_capacity_across_products" in profile["canonical_math_signature"]["core_constraints"]
    assert "compact_multi_commodity_transportation_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "origin_destination_product_shipping_cost_table" in profile["canonical_math_signature"]["required_tables"]
    assert "do not reveal or force internally generated feasible flow certificates" in profile[
        "canonical_math_signature"
    ]["forbidden_changes"]
    assert "preserve_product_dimension" in profile["quality_controls"]["validation_required"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "show supply for every origin-product pair" in guidance["required_parameter_presentation"]
    assert "show unit shipping cost for every origin-destination-product triple" in guidance[
        "required_parameter_presentation"
    ]
    assert "binary lane activation" in guidance["applicability"]["incompatible_source_features"]
    assert any("do not collapse all products" in item.lower() for item in guidance["forbidden_misframings"])
    assert "shared lane capacity" in guidance_text
    assert "origin" in guidance_text and "destination" in guidance_text


def test_mcnd_profile_and_scenario_guidance_preserve_integer_facility_counts() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_mcnd"]
    guidance = scenario_guidance_for_generator(
        "optmath_mcnd",
        instance_id="inst_optmath_mcnd_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "network_design"
    assert profile["sub_family"] == "multi_commodity_network_design"
    assert "integer_arc_facility_count" in profile["modeling_concepts"]
    assert "continuous_commodity_flow_fraction" in profile["modeling_concepts"]
    assert "nonnegative_integer_arc_facility_count" in profile["canonical_math_signature"]["variable_types"]
    assert "arc_capacity_linked_to_integer_facility_count" in profile["canonical_math_signature"]["core_constraints"]
    assert "demand_weighted_commodity_arc_routing_cost" in profile["canonical_math_signature"]["objective_terms"]
    assert "compact_mcnd_tables" in profile["canonical_math_signature"]["required_tables"]
    assert "do not make y[i,j] binary; it is a nonnegative integer facility count" in profile[
        "canonical_math_signature"
    ]["forbidden_changes"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state y[i,j] is a nonnegative integer number of installed capacity facilities, not binary" in guidance[
        "required_parameter_presentation"
    ]
    assert "binary arc activation only" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not make y[i,j] binary." in guidance["forbidden_misframings"]
    assert "demand-weighted flow" in guidance_text
    assert "nonnegative integer" in guidance_text


def test_path_and_micromobility_profiles_preserve_source_specific_contracts() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))

    tsp_profile = profiles["profiles"]["optmath_tsp"]
    tsp_guidance = scenario_guidance_for_generator(
        "optmath_tsp",
        instance_id="inst_optmath_tsp_000001",
        candidate_index=1,
        random_seed=42,
    )
    tsp_text = json.dumps(tsp_guidance, ensure_ascii=False).lower()
    assert tsp_profile["sub_family"] == "traveling_salesperson"
    assert "mtz_subtour_elimination" in tsp_profile["modeling_concepts"]
    assert "compact_tsp_tables" in tsp_profile["canonical_math_signature"]["required_tables"]
    assert "single_tour_mtz_subtour_elimination" in tsp_profile["canonical_math_signature"]["core_constraints"]
    assert "capacity_or_time_window" not in tsp_profile["canonical_math_signature"]["core_constraints"]
    assert tsp_guidance["available_scenario_count"] >= 10
    assert "state there are no capacity, time-window, service-time, or multi-vehicle constraints" in tsp_guidance[
        "required_parameter_presentation"
    ]
    assert "vehicle capacity" in tsp_guidance["applicability"]["incompatible_source_features"]
    assert "mtz" in tsp_text

    shortest_profile = profiles["profiles"]["optmath_the_shortest_path_problem"]
    shortest_guidance = scenario_guidance_for_generator(
        "optmath_the_shortest_path_problem",
        instance_id="inst_optmath_the_shortest_path_problem_000001",
        candidate_index=1,
        random_seed=42,
    )
    shortest_text = json.dumps(shortest_guidance, ensure_ascii=False).lower()
    assert shortest_profile["sub_family"] == "shortest_path"
    assert "binary_directed_arc_selection" in shortest_profile["modeling_concepts"]
    assert "compact_shortest_path_tables" in shortest_profile["canonical_math_signature"]["required_tables"]
    assert "source_sink_flow_balance" in shortest_profile["canonical_math_signature"]["core_constraints"]
    assert "arc_capacity_optional" not in shortest_profile["canonical_math_signature"]["core_constraints"]
    assert shortest_guidance["available_scenario_count"] >= 10
    assert "state there are no arc capacities, fixed charges, vehicle routes, time windows, or facility openings" in shortest_guidance[
        "required_parameter_presentation"
    ]
    assert "tsp closed tour" in [
        item.lower()
        for item in shortest_guidance["applicability"]["incompatible_source_features"]
    ]
    assert "source-to-sink shortest path" in shortest_text

    scooter_profile = profiles["profiles"]["optmath_scooter_location"]
    scooter_guidance = scenario_guidance_for_generator(
        "optmath_scooter_location",
        instance_id="inst_optmath_scooter_location_000001",
        candidate_index=1,
        random_seed=42,
    )
    scooter_text = json.dumps(scooter_guidance, ensure_ascii=False).lower()
    assert scooter_profile["sub_family"] == "micromobility_facility_location"
    assert "integer_new_scooter_deployment" in scooter_profile["modeling_concepts"]
    assert "compact_scooter_location_tables" in scooter_profile["canonical_math_signature"]["required_tables"]
    assert "new_scooters_only_at_selected_locations" in scooter_profile["canonical_math_signature"]["core_constraints"]
    assert scooter_guidance["available_scenario_count"] >= 10
    assert "state new scooters can be deployed only at selected stations" in scooter_guidance[
        "required_parameter_presentation"
    ]
    assert "traveling-salesperson tour" in scooter_guidance["applicability"]["incompatible_source_features"]
    assert "new scooters" in scooter_text


def test_net1_profile_and_scenario_guidance_preserve_capacitated_min_cost_flow() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_net1"]
    guidance = scenario_guidance_for_generator(
        "optmath_net1",
        instance_id="inst_optmath_net1_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "network_flow"
    assert profile["sub_family"] == "capacitated_min_cost_network_flow"
    assert "continuous_arc_flow" in profile["modeling_concepts"]
    assert "arc_capacity_limit" in profile["canonical_math_signature"]["core_constraints"]
    assert "arc_capacity_vector" in profile["canonical_math_signature"]["required_tables"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list all nodes and directed arcs" in guidance["required_parameter_presentation"]
    assert "vehicle routes" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as vehicle routing." in guidance["forbidden_misframings"]
    assert "capacity" in guidance_text
    assert "supply plus inbound flow equals demand plus outbound flow" in guidance_text


def test_netthru_profile_and_scenario_guidance_preserve_node_throughput_flow() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_netthru"]
    guidance = scenario_guidance_for_generator(
        "optmath_netthru",
        instance_id="inst_optmath_netthru_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["task_family"] == "network_flow"
    assert profile["sub_family"] == "node_throughput_capacitated_min_cost_flow"
    assert "node_throughput_capacity" in profile["modeling_concepts"]
    assert "directed_link_capacity" in profile["canonical_math_signature"]["core_constraints"]
    assert "node_throughput_capacity" in profile["canonical_math_signature"]["required_tables"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "list nodes and sparse directed links only" in guidance["required_parameter_presentation"]
    assert "binary link activation" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as maximum throughput." in guidance["forbidden_misframings"]
    assert any("feasible flow certificate" in item.lower() for item in guidance["forbidden_misframings"])
    assert any("feasible flow certificates are generator-only" in item.lower() for item in guidance["modeling_notes"])
    assert "throughput capacity" in guidance_text
    assert "sparse directed" in guidance_text


def test_schedulingproblem_profile_and_scenario_guidance_preserve_staff_coverage_assignment() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_schedulingproblem"]
    guidance = scenario_guidance_for_generator(
        "optmath_schedulingproblem",
        instance_id="inst_optmath_schedulingproblem_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["sub_family"] == "staff_shift_coverage_assignment"
    assert "shortage_penalty_slack" in profile["modeling_concepts"]
    assert "coverage_equal_demand_with_shortage" in profile["canonical_math_signature"]["core_constraints"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert (
        "list only positive site-shift-skill demand and state omitted combinations have zero demand"
        in guidance["required_parameter_presentation"]
    )
    assert "machine sequencing" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not describe this as machine sequencing." in guidance["forbidden_misframings"]
    assert any("internal demand-generation staff sample" in item for item in guidance["forbidden_misframings"])
    assert "unfilled-position" in guidance_text
    assert "positive demand" in guidance_text


def test_schedulingproblem_generator_does_not_expose_internal_staff_sample() -> None:
    module_path = Path("or_cpt_engine/generators/optmath_seed/SchedulingProblem/SchedulingProblem_parsed.py")
    spec = importlib.util.spec_from_file_location("scheduling_generator", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    generator = module.Generator(seed=11)
    generator.generate_instance()
    payload = json.dumps(generator.parameters, ensure_ascii=False)

    assert "latent_feasible_roster" not in payload
    assert "latent" not in payload.lower()
    assert "demand_generation_summary" in generator.parameters


def test_uls_profile_and_scenario_guidance_preserve_no_backlog_model() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_uncapacitatedlotsizing"]
    guidance = scenario_guidance_for_generator(
        "optmath_uncapacitatedlotsizing",
        instance_id="inst_optmath_uncapacitatedlotsizing_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["sub_family"] == "uncapacitated_lot_sizing"
    assert "inventory_balance_without_backlog" in profile["canonical_math_signature"]["core_constraints"]
    assert "zero_final_inventory" in profile["canonical_math_signature"]["core_constraints"]
    assert "backlog_state" not in profile["modeling_concepts"]
    assert "explain OrderedAmount, EndingInventory, BackloggedAmount" not in json.dumps(profile, ensure_ascii=False)
    assert any("no BackloggedAmount" in item for item in profile["rationale_requirements"])
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "backlog variables" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not introduce backlog" in guidance["forbidden_misframings"][0]
    assert "no backlog variable" in guidance_text
    assert "zero final backlog" in guidance_text
    assert "inventory balance without backlog" in guidance_text


def test_ulsb_profile_and_scenario_guidance_preserve_uncapacitated_backlog_model() -> None:
    import yaml

    profiles = yaml.safe_load(Path("engine_configs/generator_profiles.yaml").read_text(encoding="utf-8"))
    profile = profiles["profiles"]["optmath_uncapacitatedlotsizingbacklogging"]
    guidance = scenario_guidance_for_generator(
        "optmath_uncapacitatedlotsizingbacklogging",
        instance_id="inst_optmath_uncapacitatedlotsizingbacklogging_000001",
        candidate_index=1,
        random_seed=42,
    )

    guidance_text = json.dumps(guidance, ensure_ascii=False).lower()
    assert profile["sub_family"] == "uncapacitated_lot_sizing_with_backlogging"
    assert "backlog_state" in profile["modeling_concepts"]
    assert "setup_or_capacity_limit" not in profile["canonical_math_signature"]["core_constraints"]
    assert "net_inventory_balance_with_backlog_carryover" in profile["canonical_math_signature"]["core_constraints"]
    assert "total_demand_big_m" in profile["canonical_math_signature"]["required_tables"]
    assert guidance["available_scenario_count"] >= 10
    assert guidance["available_scenario_variant_count"] >= 50
    assert "state net inventory is ending inventory minus backlog" in guidance["required_parameter_presentation"]
    assert "production capacity constraints" in guidance["applicability"]["incompatible_source_features"]
    assert "Do not add production capacity." in guidance["forbidden_misframings"]
    assert "backlog carryover" in guidance_text
    assert "total-demand" in guidance_text
    assert "capacity_if_present" in guidance_text


def test_backtranslation_records_scenario_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "instance_id": "inst_optmath_knapsack_000001",
                "generator_id": "optmath_knapsack",
                "reference_answer": {"objective_value": 10.0, "status": "OPTIMAL"},
                "lp_text": "Maximize\n obj: x\nSubject To\n c0: x <= 1\nBounds\n x >= 0\nEnd",
            }
        ],
    )
    output_dir = tmp_path / "backtranslation"

    result = backtranslate_instances(input_path, output_dir, EngineConfig(), mock=True)

    assert result["candidates"] == 2
    rows = read_jsonl(output_dir / "backtranslation_candidates.jsonl")
    assert rows[0]["llm_metadata"]["scenario_id"].startswith("knapsack_")
    assert rows[0]["llm_metadata"]["base_scenario_id"].startswith("knapsack_")
    assert rows[0]["llm_metadata"]["scenario_variant_id"]
    assert rows[0]["llm_metadata"]["industry_lens_id"]
    assert rows[0]["llm_metadata"]["narrative_angle_id"]
    assert rows[0]["llm_metadata"]["task_family"] == "packing"
    assert rows[0]["llm_metadata"]["business_trigger"]
    report = (output_dir / "backtranslation_report.md").read_text(encoding="utf-8")
    assert "## Scenarios" in report
    assert "## Business Triggers" in report
    assert "knapsack_" in report


def test_backtranslation_prompt_compacts_large_lp_artifacts() -> None:
    large_lp = "Minimize\n" + "\n".join(f" c{i} x{i}" for i in range(15000)) + "\nEnd"
    scenario_guidance = scenario_guidance_for_generator(
        "optmath_aircraftassignment",
        instance_id="inst_optmath_aircraftassignment_000012",
        candidate_index=1,
        random_seed=42,
    )

    prompt = render_backtranslation_prompt(
        "Verified instance:\n{{INSTANCE_JSON}}",
        {
            "instance_id": "inst_optmath_aircraftassignment_000012",
            "generator_id": "optmath_aircraftassignment",
            "model_type": "MIP",
            "num_variables": 870,
            "num_constraints": 59,
            "math_formula": "minimize assignment cost subject to availability and route demand constraints",
            "lp_text": large_lp,
            "generation_params": {"aircraft": [f"aircraft_{idx}" for idx in range(30)]},
        },
        1,
        scenario_guidance,
    )

    assert len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS
    assert len(large_lp) > 100_000
    assert large_lp not in prompt
    assert '"truncated": true' in prompt
    assert '"original_chars"' in prompt


def test_backtranslation_prompt_uses_ultra_compact_payload_for_large_signatures() -> None:
    scenario_guidance = scenario_guidance_for_generator(
        "optmath_contractallocation",
        instance_id="inst_optmath_contractallocation_000012",
        candidate_index=1,
        random_seed=42,
    )
    prompt = render_backtranslation_prompt(
        "Verified instance:\n{{INSTANCE_JSON}}",
        {
            "instance_id": "inst_optmath_contractallocation_000012",
            "generator_id": "optmath_contractallocation",
            "model_type": "MIP",
            "optimization_sense": "minimize",
            "num_variables": 1800,
            "num_constraints": 900,
            "canonical_math_signature": {
                "variable_types": [f"integer_delivery_quantity_{idx}" for idx in range(300)],
                "objective_terms": [f"unit_production_cost_times_delivered_quantity_{idx}" for idx in range(300)],
                "core_constraints": [f"exact_contract_fulfillment_{idx}" for idx in range(300)],
                "forbidden_changes": [f"do_not_weaken_contract_fulfillment_{idx}" for idx in range(300)],
            },
            "generation_params": {
                "contract_sizes": [100 + idx for idx in range(120)],
                "producer_capacity": [50 + idx for idx in range(120)],
            },
        },
        1,
        scenario_guidance,
    )

    assert len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS
    assert "ultra_compact_payload" in prompt
    assert "integer_delivery_quantity_299" not in prompt


def test_backtranslation_prompt_last_resort_prevents_length_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_guidance = scenario_guidance_for_generator(
        "optmath_contractallocation",
        instance_id="inst_optmath_contractallocation_last_resort",
        candidate_index=0,
        random_seed=99,
    )
    monkeypatch.setattr(
        backtranslate_module,
        "_EMERGENCY_BACKTRANSLATION_TEMPLATE",
        "X" * (_MAX_BACKTRANSLATION_PROMPT_CHARS + 100) + "{{INSTANCE_JSON}}",
    )
    huge_template = "Y" * (_MAX_BACKTRANSLATION_PROMPT_CHARS + 100) + "{{INSTANCE_JSON}}"

    prompt = render_backtranslation_prompt(
        huge_template,
        {
            "instance_id": "inst_optmath_contractallocation_last_resort",
            "generator_id": "optmath_contractallocation",
            "model_type": "MIP",
            "optimization_sense": "minimize",
            "num_variables": 1800,
            "num_constraints": 900,
            "canonical_math_signature": {
                "variable_types": [f"integer_delivery_quantity_{idx}" for idx in range(500)],
                "objective_terms": [f"cost_term_{idx}" for idx in range(500)],
                "core_constraints": [f"constraint_{idx}" for idx in range(500)],
                "forbidden_changes": [f"forbidden_{idx}" for idx in range(500)],
            },
            "generation_params": {
                "contract_sizes": [100 + idx for idx in range(500)],
                "producer_capacity": [50 + idx for idx in range(500)],
            },
        },
        0,
        scenario_guidance,
    )

    assert len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS
    assert "last_resort_compact_template" in prompt
    assert "integer_delivery_quantity_499" not in prompt


def test_backtranslation_prompt_last_resort_truncates_explicit_source_compact_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_guidance = scenario_guidance_for_generator(
        "optmath_facility_location",
        instance_id="inst_optmath_facility_location_last_resort",
        candidate_index=1,
        random_seed=99,
    )
    monkeypatch.setattr(
        backtranslate_module,
        "_EMERGENCY_BACKTRANSLATION_TEMPLATE",
        "X" * (_MAX_BACKTRANSLATION_PROMPT_CHARS + 100) + "{{INSTANCE_JSON}}",
    )
    huge_template = "Y" * (_MAX_BACKTRANSLATION_PROMPT_CHARS + 100) + "{{INSTANCE_JSON}}"

    prompt = render_backtranslation_prompt(
        huge_template,
        {
            "instance_id": "inst_optmath_facility_location_last_resort",
            "generator_id": "optmath_facility_location",
            "model_type": "MIP",
            "optimization_sense": "minimize",
            "num_variables": 3000,
            "num_constraints": 1200,
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_facility_location_tables": {
                        "customer_table": [
                            {"customer": f"customer_{idx}", "demand": idx + 1, "service_cost": 100 + idx}
                            for idx in range(800)
                        ],
                        "facility_table": [
                            {"facility": f"facility_{idx}", "capacity": 500 + idx, "fixed_cost": 1000 + idx}
                            for idx in range(400)
                        ],
                    }
                },
            },
        },
        1,
        scenario_guidance,
    )

    assert len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS
    assert "last_resort_compact_template" in prompt
    assert '"truncated": true' in prompt
    assert "customer_799" not in prompt
    assert "facility_399" not in prompt


def test_backtranslation_prompt_target_preserves_source_fact_generators() -> None:
    assert _target_backtranslation_prompt_chars({"generator_id": "optmath_steel4"}) == _SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS
    assert _target_backtranslation_prompt_chars({"generator_id": "optmath_aircraftlanding"}) == _SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS
    assert _target_backtranslation_prompt_chars({"generator_id": "optmath_jopshop"}) == _TARGET_BACKTRANSLATION_PROMPT_CHARS
    assert _TARGET_BACKTRANSLATION_PROMPT_CHARS < _SOURCE_FACT_TARGET_BACKTRANSLATION_PROMPT_CHARS


def test_backtranslation_keeps_broad_compact_prompt_router_disabled() -> None:
    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_supplychain"},
    )
    facility_prompt_name, _ = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_facility_location"},
    )
    fleet_prompt_name, _ = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_fleet_routing"},
    )
    portfolio_prompt_name, _ = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_portfolio"},
    )
    aircraft_assignment_prompt_name, aircraft_assignment_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_aircraftassignment"},
    )
    aircraft_landing_prompt_name, aircraft_landing_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_aircraftlanding"},
    )
    transp_prompt_name, transp_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_transp"},
    )
    cflp_prompt_name, cflp_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_cflp"},
    )
    revenue_management_prompt_name, revenue_management_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_revenue_management"},
    )
    revenue_prompt_name, revenue_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_revenue"},
    )
    carselection_prompt_name, carselection_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_carselection"},
    )
    cell_tower_prompt_name, cell_tower_prompt = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_cell_tower"},
    )
    source_fact_names = [
        select_backtranslation_prompt_spec(registry, {"generator_id": generator_id})[0]
        for generator_id in (
            "optmath_clsp_expand_capacity",
            "optmath_uncapacitatedlotsizing",
            "optmath_uncapacitatedlotsizingbacklogging",
            "optmath_net1",
            "optmath_structure_based_assignment",
            "optmath_steel3",
            "optmath_steel4",
            "optmath_singlelevelsmallbucket",
        )
    ]
    generic_batch_names = [
        select_backtranslation_prompt_spec(registry, {"generator_id": generator_id})[0]
        for generator_id in (
            "optmath_contractallocation",
            "optmath_cut_edited",
            "optmath_farmplanning",
            "optmath_mcnd",
            "optmath_multisetcover",
            "optmath_multi",
            "optmath_netmcol",
            "optmath_nltrans",
            "optmath_scooter_location",
            "optmath_team_formulation",
            "optmath_the_multi_factory_schedule_problem",
            "optmath_the_p_dispersion_model",
            "optmath_the_shortest_path_problem",
            "optmath_tsp",
        )
    ]
    regular_name, regular_spec = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_knapsack"},
    )

    assert prompt_name == "backtranslation_prompt"
    assert prompt_spec.path.endswith("backtranslation_prompt.md")
    assert facility_prompt_name == "backtranslation_prompt"
    assert fleet_prompt_name == "backtranslation_prompt"
    assert portfolio_prompt_name == "backtranslation_prompt"
    assert aircraft_assignment_prompt_name == "backtranslation_aircraft_prompt"
    assert aircraft_landing_prompt_name == "backtranslation_aircraft_prompt"
    assert aircraft_assignment_prompt.path.endswith("backtranslation_aircraft_prompt.md")
    assert aircraft_landing_prompt.path.endswith("backtranslation_aircraft_prompt.md")
    assert transp_prompt_name == "backtranslation_transportation_prompt"
    assert transp_prompt.path.endswith("backtranslation_transportation_prompt.md")
    assert cflp_prompt_name == "backtranslation_cflp_prompt"
    assert cflp_prompt.path.endswith("backtranslation_cflp_prompt.md")
    assert revenue_management_prompt_name == "backtranslation_revenue_management_prompt"
    assert revenue_management_prompt.path.endswith("backtranslation_revenue_management_prompt.md")
    assert revenue_prompt_name == "backtranslation_revenue_management_prompt"
    assert revenue_prompt.path.endswith("backtranslation_revenue_management_prompt.md")
    assert carselection_prompt_name == "backtranslation_car_selection_prompt"
    assert carselection_prompt.path.endswith("backtranslation_car_selection_prompt.md")
    assert carselection_prompt.version == "v1.0.1"
    assert "Forbidden terms include `technician`, `technicians`" in carselection_prompt.text
    assert "eligible participant-to-vehicle access matching" in carselection_prompt.text
    assert cell_tower_prompt_name == "backtranslation_cell_tower_prompt"
    assert cell_tower_prompt.path.endswith("backtranslation_cell_tower_prompt.md")
    assert source_fact_names == ["backtranslation_source_fact_prompt"] * 8
    assert registry.get("backtranslation_source_fact_prompt").version == "v1.0.5"
    assert "binary NMR peak-to-amino-acid residue assignment" in registry.get("backtranslation_source_fact_prompt").text
    assert "Do not reframe this as staffing, technician-job matching" in registry.get("backtranslation_source_fact_prompt").text
    assert generic_batch_names == ["backtranslation_prompt"] * 14
    assert regular_name == "backtranslation_prompt"
    assert regular_spec.path.endswith("backtranslation_prompt.md")


def test_backtranslation_uses_emergency_template_when_compact_template_still_too_long() -> None:
    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_facility_location"},
    )
    long_phrase = "very_long_signature_component_" * 40
    row = {
        "instance_id": "inst_optmath_facility_location_oversized",
        "generator_id": "optmath_facility_location",
        "difficulty_level": "level_1",
        "sub_family": long_phrase,
        "concept_tags": [long_phrase for _ in range(20)],
        "canonical_math_signature": {
            "variable_types": [long_phrase for _ in range(20)],
            "objective_sense": "minimize",
            "objective_terms": [long_phrase for _ in range(20)],
            "core_constraints": [long_phrase for _ in range(20)],
            "required_tables": [long_phrase for _ in range(20)],
            "forbidden_changes": [long_phrase for _ in range(20)],
        },
        "generation_params": {
            "compact_facility_location_tables": {
                "plants": list(range(30)),
                "centers": list(range(30)),
                "zones": list(range(30)),
                "cost_preview": [[i, j, (i + j) % 7] for i in range(30) for j in range(30)],
            }
        },
    }
    scenario_guidance = {
        "generator_id": "optmath_facility_location",
        "task_family": "facility_location",
        "scenario_id": "fixture",
        "selected_business_trigger": long_phrase,
        "selected_scenario": {
            "industry": long_phrase,
            "background": long_phrase,
            "entities": [long_phrase for _ in range(10)],
        },
        "required_parameter_presentation": [long_phrase for _ in range(20)],
        "forbidden_misframings": [long_phrase for _ in range(20)],
        "applicability": {"incompatible_source_features": [long_phrase for _ in range(20)]},
    }

    prompt = render_backtranslation_prompt(prompt_spec.text, row, 1, scenario_guidance)

    assert prompt_name == "backtranslation_prompt"
    assert len(prompt) <= _MAX_BACKTRANSLATION_PROMPT_CHARS
    assert len(prompt) <= _TARGET_BACKTRANSLATION_PROMPT_CHARS
    assert "emergency_compact_template" in prompt
    assert "Emergency compact seed" in prompt


def test_backtranslation_accepts_common_payload_aliases() -> None:
    class AliasPayloadClient:
        async def chat(self, prompt: str, **kwargs):  # noqa: ANN001
            return {
                "content": json.dumps(
                    {
                        "business_context": "A clinic planning team is revising its weekly staffing plan.",
                        "question": "Choose staffing levels for 3 clinics while staying within 12 available shifts.",
                        "data": {"clinics": 3, "available_shifts": 12},
                    }
                ),
                "request_id": "req_alias",
                "latency_sec": 0.1,
                "model": "mock-model",
                "endpoint": {"name": "mock-endpoint"},
                "attempt": 1,
            }

    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(registry, {"generator_id": "optmath_knapsack"})

    result = asyncio.run(
        _backtranslate_one(
                {
                    "instance_id": "inst_optmath_knapsack_alias_payload",
                    "generator_id": "optmath_knapsack",
                    "generation_params": {
                        "weights": [1, 2, 3],
                        "profits": [4, 5, 6],
                        "capacity": 4,
                        "compact_cell_tower_tables": {
                            "budget": 12,
                            "tower_table": [{"tower": "tower_0", "cost": 4}],
                        },
                    },
                    "canonical_math_signature": {"objective_sense": "maximize"},
                },
            1,
            prompt_spec,
            EngineConfig(),
            client=AliasPayloadClient(),
            mock=False,
            prompt_name=prompt_name,
        )
    )

    assert result["kind"] == "candidate"
    assert "clinic planning team" in result["candidate"].problem_statement
    assert "3 clinics" in result["candidate"].problem_statement
    assert result["candidate"].structured_problem_data["available_shifts"] == 12
    dumped = result["candidate"].model_dump(mode="json")
    assert dumped["source_compact_data"]["available"] is True
    assert dumped["source_compact_data"]["value"]["compact_cell_tower_tables"]["budget"] == 12


def test_backtranslation_compact_source_prefers_seed_source_compact_data() -> None:
    compact = _compact_source_data(
        {
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_lotsizing_tables": {
                        "model_family": "uncapacitated_lot_sizing_without_backlog",
                        "periods": [1, 2],
                    }
                },
            },
            "generation_params": {},
        }
    )

    assert compact["available"] is True
    assert compact["value"]["compact_lotsizing_tables"]["model_family"] == "uncapacitated_lot_sizing_without_backlog"


def test_backtranslation_rejected_payload_keeps_raw_response_for_diagnostics() -> None:
    class MissingStatementClient:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, prompt: str, **kwargs):  # noqa: ANN001
            self.calls += 1
            return {
                "content": json.dumps({"structured_problem_data": {"items": 3}}),
                "request_id": f"req_missing_statement_{self.calls}",
                "latency_sec": 0.1,
                "model": "mock-model",
                "endpoint": {"name": "mock-endpoint"},
                "attempt": 1,
            }

    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(registry, {"generator_id": "optmath_knapsack"})
    client = MissingStatementClient()

    result = asyncio.run(
        _backtranslate_one(
            {
                "instance_id": "inst_optmath_knapsack_missing_statement",
                "generator_id": "optmath_knapsack",
                "generation_params": {"weights": [1, 2, 3], "profits": [4, 5, 6], "capacity": 4},
            },
            1,
            prompt_spec,
            EngineConfig(),
            client=client,
            mock=False,
            prompt_name=prompt_name,
        )
    )

    assert result["kind"] == "rejected"
    assert client.calls == 2
    assert "missing problem_statement" in result["rejected"]["rejection_reason"]
    assert result["raw_response"]["raw_response"] == '{"structured_problem_data": {"items": 3}}'
    assert result["rejected"]["llm_metadata"]["quality_attempt"] == 2
    assert result["rejected"]["llm_metadata"]["quality_retries"] == 1
    assert result["rejected"]["llm_metadata"]["quality_retry_events"][0]["request_id"] == "req_missing_statement_1"


def test_backtranslation_retries_missing_problem_statement_once() -> None:
    class MissingThenValidStatementClient:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        async def chat(self, prompt: str, **kwargs):  # noqa: ANN001
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                payload = {"structured_problem_data": {"items": 3}}
            else:
                payload = {
                    "problem_background": "A warehouse planner is preparing a same-day loading plan.",
                    "problem_statement": "Choose which 3 packages to load while respecting a 4 kg truck capacity.",
                    "structured_problem_data": {"items": 3, "capacity": 4},
                }
            return {
                "content": json.dumps(payload),
                "request_id": f"req_bt_retry_{self.calls}",
                "latency_sec": 0.1,
                "model": "mock-model",
                "endpoint": {"name": "mock-endpoint"},
                "attempt": 1,
            }

    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(registry, {"generator_id": "optmath_knapsack"})
    client = MissingThenValidStatementClient()

    result = asyncio.run(
        _backtranslate_one(
            {
                "instance_id": "inst_optmath_knapsack_bt_retry",
                "generator_id": "optmath_knapsack",
                "generation_params": {"weights": [1, 2, 3], "profits": [4, 5, 6], "capacity": 4},
            },
            1,
            prompt_spec,
            EngineConfig(),
            client=client,
            mock=False,
            prompt_name=prompt_name,
        )
    )

    assert result["kind"] == "candidate"
    assert client.calls == 2
    assert "QUALITY RETRY DIRECTIVE" not in client.prompts[0]
    assert "QUALITY RETRY DIRECTIVE" in client.prompts[1]
    assert "missing problem_statement" in client.prompts[1]
    assert "non-empty string fields `problem_background` and `problem_statement`" in client.prompts[1]
    dumped = result["candidate"].model_dump(mode="json")
    assert "warehouse planner" in dumped["problem_statement"]
    assert dumped["llm_metadata"]["quality_attempt"] == 2
    assert dumped["llm_metadata"]["quality_retries"] == 1
    assert "missing problem_statement" in dumped["llm_metadata"]["quality_retry_reasons"][0]
    assert dumped["llm_metadata"]["quality_retry_events"][0]["request_id"] == "req_bt_retry_1"
    assert dumped["llm_metadata"]["quality_retry_events"][0]["endpoint"]["name"] == "mock-endpoint"


def test_aircraft_landing_backtranslation_appends_required_numeric_tables() -> None:
    class SparseAircraftLandingClient:
        async def chat(self, prompt: str, **kwargs):  # noqa: ANN001
            return {
                "content": json.dumps(
                    {
                        "problem_background": "An airport recovery team is resequencing arrivals after a storm.",
                        "problem_statement": (
                            "The team must schedule landing times for aircraft while respecting landing windows, "
                            "early and late penalties, and ordered-pair separation times."
                        ),
                        "structured_problem_data": {"sets": {"aircraft": ["aircraft_0", "aircraft_1"]}},
                    }
                ),
                "request_id": "req_aircraft_landing_sparse",
                "latency_sec": 0.1,
                "model": "mock-model",
                "endpoint": {"name": "mock-endpoint"},
                "attempt": 1,
            }

    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_name, prompt_spec = select_backtranslation_prompt_spec(
        registry,
        {"generator_id": "optmath_aircraftlanding"},
    )
    row = {
        "instance_id": "inst_optmath_aircraftlanding_sparse",
        "generator_id": "optmath_aircraftlanding",
        "generation_params": {
            "compact_aircraft_landing_tables": {
                "aircraft_time_penalty_table": [
                    {
                        "aircraft": "aircraft_0",
                        "earliest_landing": 43,
                        "target_landing": 63,
                        "latest_landing": 83,
                        "early_penalty_per_minute": 93,
                        "late_penalty_per_minute": 94,
                    },
                    {
                        "aircraft": "aircraft_1",
                        "earliest_landing": 45,
                        "target_landing": 65,
                        "latest_landing": 85,
                        "early_penalty_per_minute": 28,
                        "late_penalty_per_minute": 34,
                    },
                ],
                "ordered_pair_separation_matrix": {
                    "columns": ["aircraft_0", "aircraft_1"],
                    "rows": [
                        {"aircraft_before": "aircraft_0", "values": [None, 4]},
                        {"aircraft_before": "aircraft_1", "values": [2, None]},
                    ],
                },
            }
        },
    }

    result = asyncio.run(
        _backtranslate_one(
            row,
            1,
            prompt_spec,
            EngineConfig(),
            client=SparseAircraftLandingClient(),
            mock=False,
            prompt_name=prompt_name,
        )
    )

    assert result["kind"] == "candidate"
    statement = result["candidate"].problem_statement
    assert "Aircraft landing data:" in statement
    assert "aircraft_0: earliest=43, target=63, latest=83, early_penalty=93, late_penalty=94" in statement
    assert "aircraft_1: earliest=45, target=65, latest=85, early_penalty=28, late_penalty=34" in statement
    assert "columns: aircraft_0, aircraft_1" in statement
    assert "aircraft_0: -, 4" in statement
    assert "aircraft_1: 2, -" in statement


def test_clsp_backtranslation_compacts_overlong_or_wrong_backlog_statement() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "capacitated_lot_sizing_without_backlog",
                "products": [0, 1],
                "periods": [0, 1],
                "period_demand_table": [
                    {
                        "product": 0,
                        "period": 0,
                        "period_demand": 70,
                        "cumulative_demand_through_period": 70,
                        "setup_cost": 1000,
                        "unit_production_cost": 40,
                        "unit_holding_cost": 4,
                        "capacity_consumption_per_unit": 1.5,
                        "period_capacity": 800,
                        "setup_big_m_remaining_demand": 150,
                    },
                    {
                        "product": 0,
                        "period": 1,
                        "period_demand": 80,
                        "cumulative_demand_through_period": 150,
                        "setup_cost": 1100,
                        "unit_production_cost": 41,
                        "unit_holding_cost": 5,
                        "capacity_consumption_per_unit": 1.5,
                        "period_capacity": 800,
                        "setup_big_m_remaining_demand": 80,
                    },
                    {
                        "product": 1,
                        "period": 0,
                        "period_demand": 60,
                        "cumulative_demand_through_period": 60,
                        "setup_cost": 1200,
                        "unit_production_cost": 42,
                        "unit_holding_cost": 4,
                        "capacity_consumption_per_unit": 1.8,
                        "period_capacity": 800,
                        "setup_big_m_remaining_demand": 130,
                    },
                    {
                        "product": 1,
                        "period": 1,
                        "period_demand": 70,
                        "cumulative_demand_through_period": 130,
                        "setup_cost": 1300,
                        "unit_production_cost": 43,
                        "unit_holding_cost": 5,
                        "capacity_consumption_per_unit": 1.8,
                        "period_capacity": 800,
                        "setup_big_m_remaining_demand": 70,
                    },
                ],
                "period_capacity_table": [{"period": 0, "capacity": 800}, {"period": 1, "capacity": 800}],
                "capacity_consumption_per_product": [
                    {"product": 0, "capacity_consumption_per_unit": 1.5},
                    {"product": 1, "capacity_consumption_per_unit": 1.8},
                ],
            }
        },
    }
    bad_statement = (
        "A planning team has backlog allowed and backlog penalty costs. "
        "The plan may expand capacity through overtime purchases. "
        + "This verbose sentence repeats operational context. " * 120
    )

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_clsp_expand_capacity"},
        bad_statement,
        source_compact_data=source_compact_data,
    )

    assert len(statement) < 4000
    assert "no backlog variables" in statement
    assert "no capacity-expansion decision" in statement
    assert "backlog allowed" not in statement
    assert "overtime purchases" not in statement
    assert "use period_demand in each one-period inventory balance" in statement
    assert "cumulative_demand_through_t only in a cumulative balance expression" in statement
    assert "p=0, t=0, period_demand=70, cumulative_demand_through_t=70, setup=1000, prod=40, hold=4, M=150" in statement
    assert "p=1, t=1, period_demand=70, cumulative_demand_through_t=130, setup=1300, prod=43, hold=5, M=70" in statement
    assert "p=0: 1.5" in statement
    assert "t=1: 800" in statement


def test_ulsb_backtranslation_rewrites_to_exact_backlog_fact_card() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "uncapacitated_lot_sizing_with_backlogging",
                "periods": ["period_1", "period_2"],
                "period_cost_table": [
                    {
                        "period": "period_1",
                        "demand": 17,
                        "fixed_ordering_cost": 197,
                        "unit_order_cost": 5,
                        "unit_holding_cost": 1,
                        "unit_backlog_penalty": 14,
                    },
                    {
                        "period": "period_2",
                        "demand": 39,
                        "fixed_ordering_cost": 132,
                        "unit_order_cost": 3,
                        "unit_holding_cost": 3,
                        "unit_backlog_penalty": 5,
                    },
                ],
                "total_demand_big_m": 56,
                "initial_inventory": 0,
                "initial_backlog": 0,
                "required_final_inventory": 0,
                "required_final_backlog": 0,
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_uncapacitatedlotsizingbacklogging"},
        "A planner has approximate demand and may use shortage slack. Some rows are omitted.",
        source_compact_data=source_compact_data,
    )

    assert "with carried backlog" in statement
    assert "period=period_1, demand=17, fixed=197, unit_order=5, holding=1, backlog_penalty=14" in statement
    assert "period=period_2, demand=39, fixed=132, unit_order=3, holding=3, backlog_penalty=5" in statement
    assert "Use total-demand big-M 56" in statement
    assert "Initial inventory is 0 and initial backlog is 0" in statement
    assert "Final inventory in the last period must be 0 and final backlog must be 0" in statement
    assert "net inventory is EndingInventory[t] - BackloggedAmount[t]" in statement
    assert "independent shortage slack" in statement
    assert "approximate demand" not in statement
    assert "omitted" not in statement


def test_uls_backtranslation_rewrites_to_no_backlog_fact_card() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "uncapacitated_lot_sizing_without_backlog",
                "periods": ["period_1", "period_2"],
                "period_cost_table": [
                    {
                        "period": "period_1",
                        "demand": 10,
                        "fixed_ordering_cost": 27,
                        "unit_order_cost": 5,
                        "unit_holding_cost": 1,
                    },
                    {
                        "period": "period_2",
                        "demand": 7,
                        "fixed_ordering_cost": 40,
                        "unit_order_cost": 5,
                        "unit_holding_cost": 1,
                    },
                ],
                "total_demand_big_m": 17,
                "initial_inventory": 0,
                "required_final_inventory": 0,
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_uncapacitatedlotsizing"},
        "A planner can backlog unmet demand and pay backlog penalties.",
        source_compact_data=source_compact_data,
    )

    assert "without backlog" in statement
    assert "There are no backlog variables" in statement
    assert "period=period_1, demand=10, fixed=27, unit_order=5, holding=1" in statement
    assert "Use total-demand big-M 17" in statement
    assert "backlog penalties" in statement
    assert "pay backlog penalties" not in statement


def test_legacy_generation_params_rebuild_ulsb_compact_fact_card() -> None:
    row = {
        "generator_id": "optmath_uncapacitatedlotsizingbacklogging",
        "generation_params": {
            "periods": ["period_1", "period_2", "period_3", "period_4", "period_5"],
            "period_cost_table": [
                {"period": "period_1", "demand": 17, "fixed_ordering_cost": 197, "unit_order_cost": 5, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
                {"period": "period_2", "demand": 39, "fixed_ordering_cost": 132, "unit_order_cost": 3, "unit_holding_cost": 3, "unit_backlog_penalty": 5},
                {"period": "period_3", "demand": 66, "fixed_ordering_cost": 50, "unit_order_cost": 10, "unit_holding_cost": 1, "unit_backlog_penalty": 6},
                {"period": "period_4", "demand": 11, "fixed_ordering_cost": 103, "unit_order_cost": 2, "unit_holding_cost": 3, "unit_backlog_penalty": 13},
                {"period": "period_5", "demand": 36, "fixed_ordering_cost": 182, "unit_order_cost": 7, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
            ],
            "initial_inventory": 0,
            "initial_backlog": 0,
            "required_final_inventory": 0,
            "required_final_backlog": 0,
            "big_m_order_upper_bound": 169,
        },
    }

    source_compact_data = _compact_source_data(row)
    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_lotsizing_tables"]
    assert tables["model_family"] == "uncapacitated_lot_sizing_with_backlogging"
    assert tables["total_demand_big_m"] == 169

    statement = _ensure_required_problem_statement_tables(
        row,
        "Old drift statement says period_5 demand is 28 and big-M is 161.",
        source_compact_data=source_compact_data,
    )

    assert "period=period_5, demand=36" in statement
    assert "Use total-demand big-M 169" in statement
    assert "161" not in statement


def test_legacy_ulsb_compact_fact_card_prefers_lp_over_drifted_generation_params() -> None:
    lp_text = r"""
\ Model UncapacitatedLotSizingBacklogging
Minimize
  5 OrderedAmount[period_1] + 3 OrderedAmount[period_2]
   + 10 OrderedAmount[period_3] + 2 OrderedAmount[period_4]
   + 7 OrderedAmount[period_5] + EndingInventory[period_1]
   + 3 EndingInventory[period_2] + EndingInventory[period_3]
   + 3 EndingInventory[period_4] + EndingInventory[period_5]
   + 197 OrderIsPlaced[period_1] + 132 OrderIsPlaced[period_2]
   + 50 OrderIsPlaced[period_3] + 103 OrderIsPlaced[period_4]
   + 182 OrderIsPlaced[period_5] + 14 BackloggedAmount[period_1]
   + 5 BackloggedAmount[period_2] + 6 BackloggedAmount[period_3]
   + 13 BackloggedAmount[period_4] + 14 BackloggedAmount[period_5]
Subject To
 FlowBalance_period_1: - OrderedAmount[period_1]
   + EndingInventory[period_1] - BackloggedAmount[period_1] = -17
 FlowBalance_period_2: - OrderedAmount[period_2]
   - EndingInventory[period_1] + EndingInventory[period_2]
   + BackloggedAmount[period_1] - BackloggedAmount[period_2] = -39
 FlowBalance_period_3: - OrderedAmount[period_3]
   - EndingInventory[period_2] + EndingInventory[period_3]
   + BackloggedAmount[period_2] - BackloggedAmount[period_3] = -66
 FlowBalance_period_4: - OrderedAmount[period_4]
   - EndingInventory[period_3] + EndingInventory[period_4]
   + BackloggedAmount[period_3] - BackloggedAmount[period_4] = -11
 FlowBalance_period_5: - OrderedAmount[period_5]
   - EndingInventory[period_4] + EndingInventory[period_5]
   + BackloggedAmount[period_4] - BackloggedAmount[period_5] = -36
 OrderedUpperBound_period_1: OrderedAmount[period_1] - 169 OrderIsPlaced[period_1] <= 0
 OrderedUpperBound_period_2: OrderedAmount[period_2] - 169 OrderIsPlaced[period_2] <= 0
 OrderedUpperBound_period_3: OrderedAmount[period_3] - 169 OrderIsPlaced[period_3] <= 0
 OrderedUpperBound_period_4: OrderedAmount[period_4] - 169 OrderIsPlaced[period_4] <= 0
 OrderedUpperBound_period_5: OrderedAmount[period_5] - 169 OrderIsPlaced[period_5] <= 0
 EndingInventoryZero: EndingInventory[period_5] = 0
 EndingBacklogZero: BackloggedAmount[period_5] = 0
Bounds
Binaries
 OrderIsPlaced[period_1] OrderIsPlaced[period_2] OrderIsPlaced[period_3]
 OrderIsPlaced[period_4] OrderIsPlaced[period_5]
End
"""
    row = {
        "generator_id": "optmath_uncapacitatedlotsizingbacklogging",
        "generation_params": {
            "periods": ["period_1", "period_2", "period_3", "period_4", "period_5"],
            "period_cost_table": [
                {"period": "period_1", "demand": 17, "fixed_ordering_cost": 197, "unit_order_cost": 5, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
                {"period": "period_2", "demand": 39, "fixed_ordering_cost": 132, "unit_order_cost": 3, "unit_holding_cost": 3, "unit_backlog_penalty": 5},
                {"period": "period_3", "demand": 66, "fixed_ordering_cost": 50, "unit_order_cost": 10, "unit_holding_cost": 1, "unit_backlog_penalty": 6},
                {"period": "period_4", "demand": 11, "fixed_ordering_cost": 103, "unit_order_cost": 2, "unit_holding_cost": 3, "unit_backlog_penalty": 13},
                {"period": "period_5", "demand": 28, "fixed_ordering_cost": 182, "unit_order_cost": 7, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
            ],
            "big_m_order_upper_bound": 161,
        },
        "source_lp_text": lp_text,
    }

    source_compact_data = _compact_source_data(row)
    tables = source_compact_data["value"]["compact_lotsizing_tables"]

    assert source_compact_data["available"] is True
    assert tables["source_model_note"].startswith("Legacy LP-derived")
    assert tables["total_demand_big_m"] == 169
    assert tables["period_cost_table"][-1]["demand"] == 36

    statement = _ensure_required_problem_statement_tables(
        row,
        "Old drift statement says period_5 demand is 28 and big-M is 161.",
        source_compact_data=source_compact_data,
    )

    assert "period=period_5, demand=36" in statement
    assert "Use total-demand big-M 169" in statement
    assert "161" not in statement


def test_legacy_generation_params_rebuild_steel4_and_net1_compact_fact_cards() -> None:
    steel4_source = _compact_source_data(
        {
            "generator_id": "optmath_steel4",
            "generation_params": {
                "products": ["product_0", "product_1"],
                "stages": ["stage_0", "stage_1"],
                "profit": {"product_0": 296, "product_1": 463},
                "commit": {"product_0": 43, "product_1": 18},
                "market": {"product_0": 122, "product_1": 128},
                "available": {"stage_0": 61.13, "stage_1": 104.81},
                "processing_hours_per_unit": {
                    "product_0|stage_0": 0.1,
                    "product_0|stage_1": 0.25,
                    "product_1|stage_0": 0.33,
                    "product_1|stage_1": 0.14,
                },
            },
        }
    )
    assert steel4_source["available"] is True
    steel4_statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_steel4"},
        "Old drift statement copies market bounds into stage capacities.",
        source_compact_data=steel4_source,
    )
    assert "product=product_0, profit=296, min_commit=43, max_market=122" in steel4_statement
    assert "stage=stage_1, available_hours=104.81" in steel4_statement
    assert "do not copy product market bounds into stage capacities" in steel4_statement

    net1_source = _compact_source_data(
        {
            "generator_id": "optmath_net1",
            "generation_params": {
                "cities": ["city_0", "city_1", "city_2"],
                "links": [["city_0", "city_1"], ["city_1", "city_2"]],
                "supply": {"city_0": 10, "city_1": 0, "city_2": 0},
                "demand": {"city_0": 0, "city_1": 0, "city_2": 10},
                "net_balance": {"city_0": 10, "city_1": 0, "city_2": -10},
                "shipping_cost": {"city_0|city_1": 3, "city_1|city_2": 4},
                "capacity": {"city_0|city_1": 10, "city_1|city_2": 10},
            },
        }
    )
    assert net1_source["available"] is True
    net1_tables = net1_source["value"]["compact_network_flow_tables"]
    assert net1_tables["total_supply"] == 10
    assert net1_tables["total_demand"] == 10
    net1_statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_net1"},
        "Old drift statement says demand is negative signed supply.",
        source_compact_data=net1_source,
    )
    assert "node=city_2, supply=0, demand=10, net=-10" in net1_statement
    assert "from=city_1, to=city_2, cost=4, capacity=10" in net1_statement
    assert "supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node]" in net1_statement

    netasgn_source = _compact_source_data(
        {
            "generator_id": "optmath_netasgn",
            "generation_params": {
                "people": ["person_0", "person_1"],
                "projects": ["project_0", "project_1"],
                "supply_hours": {"person_0": 8, "person_1": 6},
                "demand_hours": {"project_0": 5, "project_1": 9},
                "cost_per_hour": {
                    "person_0": {"project_0": 32, "project_1": 11},
                    "person_1": {"project_0": 17, "project_1": 23},
                },
                "max_contribution_hours": {
                    "person_0": {"project_0": 5, "project_1": 8},
                    "person_1": {"project_0": 6, "project_1": 6},
                },
                "total_supply_hours": 14,
                "total_demand_hours": 14,
            },
        }
    )
    assert netasgn_source["available"] is True
    netasgn_tables = netasgn_source["value"]["compact_project_assignment_tables"]
    assert netasgn_tables["total_supply_hours"] == 14
    assert netasgn_tables["cost_per_hour_matrix"]["rows"][0]["values"] == [32, 11]
    netasgn_statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_netasgn"},
        "Old drift statement says Staff A has enquiry hours and a broken }u upper bound.",
        source_compact_data=netasgn_source,
    )
    assert "person=person_0, available_hours=8" in netasgn_statement
    assert "project=project_1, required_hours=9" in netasgn_statement
    assert "Cost per assigned hour matrix columns: project_0, project_1" in netasgn_statement
    assert "Maximum contribution hours matrix columns: project_0, project_1" in netasgn_statement
    assert "enquiry" not in netasgn_statement
    assert "}u" not in netasgn_statement


def test_legacy_structure_assignment_compact_fact_card_from_lp() -> None:
    lp_text = r"""
\ Model SBA_Problem
Minimize
 0.5 x[0,0] + x[0,1] + 0.25 x[1,0]
Subject To
 amino_acid_assignment[0]: x[0,0] + x[1,0] <= 1
 amino_acid_assignment[1]: x[0,1] + x[1,1] <= 1
 peak_assignment[0]: x[0,0] + x[0,1] <= 1
 peak_assignment[1]: x[1,0] + x[1,1] <= 1
 total_assignments: x[0,0] + x[0,1] + x[1,0] + x[1,1] = 1
 NOE_0_1_0_1: x[0,0] + x[1,1] <= 1
 NOE_0_1_1_0: x[0,1] + x[1,0] <= 2
Bounds
Binaries
 x[0,0] x[0,1] x[1,0] x[1,1]
End
"""

    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_structure_based_assignment",
            "generation_params": {},
            "lp_text": lp_text,
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_structure_assignment_tables"]
    assert tables["required_assignment_count"] == 1
    assert tables["assignment_cost_matrix"]["rows"] == [
        {"peak": 0, "values": [0.5, 1.0]},
        {"peak": 1, "values": [0.25, 0.0]},
    ]
    assert tables["acid_compatibility_matrix"]["rows"] == [
        {"amino_acid": 0, "values": [1, 0]},
        {"amino_acid": 1, "values": [1, 1]},
    ]

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_structure_based_assignment"},
        "Old statement assumes missing compatibility values.",
        source_compact_data=source_compact_data,
    )
    assert "exactly 1 peak-amino-acid assignments must be selected" in statement
    assert "peak=1: 0.25, 0.0" in statement
    assert "Do not assume missing compatibility values" in statement


def test_legacy_marketshare_compact_fact_card_from_lp() -> None:
    lp_text = r"""
\ Model MarketSharing
Maximize
 4 supply[0,0,0] + 6 supply[1,0,0] + 3 supply[0,0,1] + 5 supply[1,0,1]
Subject To
 demand_0_0: supply[0,0,0] + supply[1,0,0] = 12
 demand_0_1: supply[0,0,1] + supply[1,0,1] = 8
Bounds
Generals
 supply[0,0,0] supply[1,0,0] supply[0,0,1] supply[1,0,1]
End
"""

    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_marketshare",
            "generation_params": {},
            "lp_text": lp_text,
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_marketshare_tables"]
    assert tables["sets"] == {"companies": [0, 1], "markets": [0], "products": [0, 1]}
    assert tables["demand_table"] == [
        {"market": 0, "product": 0, "demand": 12},
        {"market": 0, "product": 1, "demand": 8},
    ]
    assert {"company": 1, "market": 0, "values": [6, 5]} in tables["unit_profit_matrices_by_company"]["rows"]
    assert "no resource capacity" in tables["source_contract_note"]

    prompt = render_forward_prompt(
        "{{PROBLEM_JSON}}",
        {
            "generator_id": "optmath_marketshare",
            "problem_statement": "Legacy statement may mention campaign budget by mistake.",
            "source_compact_data": source_compact_data,
            "concept_tags": ["integer_supply_allocation"],
        },
    )
    digest = "\n".join(json.loads(prompt)["source_fact_digest"])
    assert "marketshare: nonnegative integer Supply[i,j,k]" in digest
    assert "market=0: 12, 8" in digest
    assert "company=1,market=0: 6, 5" in digest


def test_legacy_uls_compact_fact_card_from_lp() -> None:
    lp_text = r"""
\ Model ULS
Minimize
 OrderedAmount[period_1] + 4 OrderedAmount[period_2]
 + 3 EndingInventory[period_1] + 2 EndingInventory[period_2]
 + 25 OrderIsPlaced[period_1] + 15 OrderIsPlaced[period_2]
Subject To
 FlowBalance_period_1: OrderedAmount[period_1] - EndingInventory[period_1] = 10
 FlowBalance_period_2: EndingInventory[period_1] + OrderedAmount[period_2] - EndingInventory[period_2] = 14
 OrderedUpperBound_period_1: OrderedAmount[period_1] - 24 OrderIsPlaced[period_1] <= 0
 OrderedUpperBound_period_2: OrderedAmount[period_2] - 24 OrderIsPlaced[period_2] <= 0
 EndingInventory: EndingInventory[period_2] = 0
Bounds
Binaries
 OrderIsPlaced[period_1] OrderIsPlaced[period_2]
End
"""

    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_uncapacitatedlotsizing",
            "generation_params": {},
            "lp_text": lp_text,
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_lotsizing_tables"]
    assert tables["model_family"] == "uncapacitated_lot_sizing_without_backlog"
    assert tables["total_demand_big_m"] == 24
    assert tables["period_cost_table"] == [
        {
            "period": "period_1",
            "demand": 10,
            "fixed_ordering_cost": 25,
            "unit_order_cost": 1,
            "unit_holding_cost": 3,
        },
        {
            "period": "period_2",
            "demand": 14,
            "fixed_ordering_cost": 15,
            "unit_order_cost": 4,
            "unit_holding_cost": 2,
        },
    ]
    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_uncapacitatedlotsizing"},
        "Old statement accidentally adds backlogs.",
        source_compact_data=source_compact_data,
    )
    assert "There are no backlog variables" in statement
    assert "period=period_2, demand=14, fixed=15, unit_order=4, holding=2" in statement


def test_legacy_clsp_compact_fact_card_from_lp() -> None:
    lp_text = r"""
\ Model CLSP
Minimize
 47 Production[0,0] + 47 Production[0,1] + 31 Production[1,0] + 31 Production[1,1]
 + 390 Setup[0,0] + 390 Setup[0,1] + 250 Setup[1,0] + 250 Setup[1,1]
 + 4 Inventory_0_0 + 4 Inventory_0_1 + 3 Inventory_1_0 + 3 Inventory_1_1
Subject To
 InventoryBalance_0_0: - Production[0,0] + Inventory_0_0 = -64
 InventoryBalance_0_1: - Production[0,0] - Production[0,1] + Inventory_0_1 = -116
 InventoryBalance_1_0: - Production[1,0] + Inventory_1_0 = -20
 InventoryBalance_1_1: - Production[1,0] - Production[1,1] + Inventory_1_1 = -50
 Capacity_0: 1.72 Production[0,0] + 1.64 Production[1,0] <= 800
 Capacity_1: 1.72 Production[0,1] + 1.64 Production[1,1] <= 800
 Setup_0_0: Production[0,0] - 116 Setup[0,0] <= 0
 Setup_0_1: Production[0,1] - 52 Setup[0,1] <= 0
 Setup_1_0: Production[1,0] - 50 Setup[1,0] <= 0
 Setup_1_1: Production[1,1] - 30 Setup[1,1] <= 0
Bounds
Binaries
 Setup[0,0] Setup[0,1] Setup[1,0] Setup[1,1]
End
"""

    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "generation_params": {},
            "lp_text": lp_text,
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_lotsizing_tables"]
    assert tables["model_family"] == "capacitated_lot_sizing_without_backlog"
    assert tables["period_demand_table"][1] == {
        "product": 0,
        "period": 1,
        "period_demand": 52,
        "cumulative_demand_through_period": 116,
        "setup_cost": 390,
        "unit_production_cost": 47,
        "unit_holding_cost": 4,
        "setup_big_m_remaining_demand": 52,
    }
    assert tables["capacity_consumption_per_product"] == [
        {"product": 0, "capacity_consumption_per_unit": 1.72},
        {"product": 1, "capacity_consumption_per_unit": 1.64},
    ]
    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_clsp_expand_capacity"},
        "Old statement describes capacity expansion.",
        source_compact_data=source_compact_data,
    )
    assert "fixed period capacities" in statement
    assert "p=1, t=1, period_demand=30, cumulative_demand_through_t=50" in statement
    assert "no capacity-expansion decision" in statement


def test_legacy_steel3_compact_fact_card_from_generation_params() -> None:
    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_steel3",
            "generation_params": {
                "products": ["product_0", "product_1", "product_2"],
                "production_rates": {"product_0": 9, "product_1": 6, "product_2": 1},
                "processing_hours_per_unit": {"product_0": 0.11, "product_1": 0.17, "product_2": 1.0},
                "profits": {"product_0": 25, "product_1": 19, "product_2": 42},
                "min_sold": {"product_0": 31, "product_1": 18, "product_2": 45},
                "max_sold": {"product_0": 93, "product_1": 52, "product_2": 61},
                "available_hours": 65.2,
            },
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_steel_product_mix_tables"]
    assert tables["model_family"] == "single_stage_continuous_product_mix"
    assert tables["time_capacity"]["available_hours"] == 65.2
    assert tables["product_table"]["rows"][0] == ["product_0", 25, 9, 0.11, 31, 93]


def test_legacy_electrical_power_compact_fact_card_from_generation_params() -> None:
    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_electrical_power",
            "generation_params": {
                "generator_types": ["type_0", "type_1"],
                "time_periods": ["period_0", "period_1"],
                "demand": {"period_0": 218, "period_1": 116},
                "min_output": {"type_0": 19, "type_1": 30},
                "max_output": {"type_0": 171, "type_1": 141},
                "base_cost": {"type_0": 76, "type_1": 97},
                "per_mw_cost": {"type_0": 1, "type_1": 1},
                "startup_cost": {"type_0": 494, "type_1": 475},
                "generators_available": {"type_0": 2, "type_1": 2},
                "on_start": {"type_0": 2, "type_1": 0},
                "reserve_margin": 1.15,
            },
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_electrical_power_tables"]
    assert tables["model_family"] == "unit_commitment_electrical_power"
    assert tables["generator_type_table"][0]["maximum_output_mw"] == 171
    assert tables["period_demand_table"][0] == {
        "period": "period_0",
        "demand_mw": 218,
        "reserve_required_capacity_mw": 250.7,
    }


def test_legacy_smallbucket_compact_fact_card_from_lp() -> None:
    lp_text = r"""
\ Model DLSP
Minimize
 150 Production[0,0,0] + 150 Production[0,0,1] + 150 Production[0,1,0] + 150 Production[0,1,1]
 + 150 Production[1,0,0] + 150 Production[1,0,1] + 150 Production[1,1,0] + 150 Production[1,1,1]
 + 120 Startup[0,0,0] + 120 Startup[0,0,1] + 120 Startup[0,1,0] + 120 Startup[0,1,1]
 + 120 Startup[1,0,0] + 120 Startup[1,0,1] + 120 Startup[1,1,0] + 120 Startup[1,1,1]
 + 2 Stock[0,0] + 2 Stock[0,1] + 3 Stock[1,0] + 3 Stock[1,1]
 + 9 Backlog[0,0] + 9 Backlog[0,1] + 11 Backlog[1,0] + 11 Backlog[1,1]
Subject To
 R0: Amount[0,0,0] + Amount[0,1,0] - Stock[0,0] + Backlog[0,0] = 20
 R1: Amount[0,0,1] + Amount[0,1,1] + Stock[0,0] - Stock[0,1] - Backlog[0,0] + Backlog[0,1] = 25
 R2: Amount[1,0,0] + Amount[1,1,0] - Stock[1,0] + Backlog[1,0] = 30
 R3: Amount[1,0,1] + Amount[1,1,1] + Stock[1,0] - Stock[1,1] - Backlog[1,0] + Backlog[1,1] = 35
 R4: - 80 Production[0,0,0] + 10 Startup[0,0,0] + Amount[0,0,0] <= 0
 R5: - 80 Production[0,0,1] + 10 Startup[0,0,1] + Amount[0,0,1] <= 0
 R6: - 90 Production[0,1,0] + 12 Startup[0,1,0] + Amount[0,1,0] <= 0
 R7: - 90 Production[0,1,1] + 12 Startup[0,1,1] + Amount[0,1,1] <= 0
 R8: - 80 Production[1,0,0] + 10 Startup[1,0,0] + Amount[1,0,0] <= 0
 R9: - 80 Production[1,0,1] + 10 Startup[1,0,1] + Amount[1,0,1] <= 0
 R10: - 90 Production[1,1,0] + 12 Startup[1,1,0] + Amount[1,1,0] <= 0
 R11: - 90 Production[1,1,1] + 12 Startup[1,1,1] + Amount[1,1,1] <= 0
Bounds
Binaries
 Production[0,0,0] Startup[0,0,0]
End
"""

    source_compact_data = _compact_source_data(
        {
            "generator_id": "optmath_singlelevelsmallbucket",
            "generation_params": {},
            "lp_text": lp_text,
        }
    )

    assert source_compact_data["available"] is True
    tables = source_compact_data["value"]["compact_lotsizing_tables"]
    assert tables["global_costs"]["setup_cost_per_active_item_machine_period"] == 150
    assert tables["global_costs"]["startup_cost_per_start_event"] == 120
    assert tables["item_cost_table"]["rows"] == [[0, 2, 9], [1, 3, 11]]
    assert tables["machine_table"]["rows"] == [[0, 80, 10], [1, 90, 12]]
    assert tables["demand_matrix"]["rows"] == [
        {"item": 0, "values": [20, 25]},
        {"item": 1, "values": [30, 35]},
    ]
    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_singlelevelsmallbucket"},
        "Old statement adds production cost per unit and placeholder demand.",
        source_compact_data=source_compact_data,
    )
    assert "There is no per-unit production cost" in statement
    assert "item=1: 30, 35" in statement
    assert "machine=1, capacity=90, startup_time=12" in statement


def test_lotsizing_structured_data_rebuilt_from_compact_source() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "uncapacitated_lot_sizing_with_backlogging",
                "periods": ["period_1"],
                "period_cost_table": [
                    {
                        "period": "period_1",
                        "demand": 17,
                        "fixed_ordering_cost": 197,
                        "unit_order_cost": 5,
                        "unit_holding_cost": 1,
                        "unit_backlog_penalty": 14,
                    },
                ],
                "total_demand_big_m": 17,
                "initial_inventory": 0,
                "initial_backlog": 0,
                "required_final_inventory": 0,
                "required_final_backlog": 0,
            }
        },
    }

    structured = _structured_problem_data_from_compact_source(
        {"generator_id": "optmath_uncapacitatedlotsizingbacklogging"},
        source_compact_data=source_compact_data,
        fallback={"entities": ["generic shortage slack"]},
    )

    assert structured["entities"] == ["planning periods", "orders", "ending inventory", "carried backlog"]
    assert structured["sets"]["periods"] == ["period_1"]
    assert structured["parameters"]["total_demand_big_m"] == 17
    assert "BackloggedAmount[t]" in structured["variables"]
    assert "independent shortage slack" in structured["forbidden_structures"]


def test_net1_backtranslation_rewrites_signed_net_supply_statement() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_network_flow_tables": {
                "model_family": "capacitated_min_cost_network_flow",
                "cities": ["city_0", "city_1", "city_2"],
                "node_balance_table": [
                    {"city": "city_0", "supply": 10, "demand": 0, "net_supply_minus_demand": 10},
                    {"city": "city_1", "supply": 0, "demand": 0, "net_supply_minus_demand": 0},
                    {"city": "city_2", "supply": 0, "demand": 10, "net_supply_minus_demand": -10},
                ],
                "directed_arc_table": [
                    {"from": "city_0", "to": "city_1", "unit_arc_flow_cost": 3, "arc_capacity": 10},
                    {"from": "city_1", "to": "city_2", "unit_arc_flow_cost": 4, "arc_capacity": 10},
                ],
                "total_supply": 10,
                "total_demand": 10,
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_net1"},
        "The network has signed net supply values: city_2 has demand -10 and optional unmet demand slack may be used.",
        source_compact_data=source_compact_data,
    )

    assert "Total supply is 10 and total demand is 10" in statement
    assert "node=city_0, supply=10, demand=0, net=10" in statement
    assert "node=city_2, supply=0, demand=10, net=-10" in statement
    assert "from=city_0, to=city_1, cost=3, capacity=10" in statement
    assert "supply[node] + inbound_flow[node] = demand[node] + outbound_flow[node]" in statement
    assert "optional unmet demand slack may be used" not in statement
    assert "demand -10" not in statement


def test_net1_structured_data_rebuilt_from_compact_source() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_network_flow_tables": {
                "model_family": "capacitated_min_cost_network_flow",
                "cities": ["city_0", "city_1"],
                "node_balance_table": [
                    {"city": "city_0", "supply": 5, "demand": 0, "net_supply_minus_demand": 5},
                    {"city": "city_1", "supply": 0, "demand": 5, "net_supply_minus_demand": -5},
                ],
                "directed_arc_table": [
                    {"from": "city_0", "to": "city_1", "unit_arc_flow_cost": 7, "arc_capacity": 5},
                ],
                "total_supply": 5,
                "total_demand": 5,
            }
        },
    }

    structured = _structured_problem_data_from_compact_source(
        {"generator_id": "optmath_net1"},
        source_compact_data=source_compact_data,
        fallback={"parameters": {"supply_demand": {"city_1": -5}}},
    )

    assert structured["entities"] == ["network nodes", "declared directed arcs"]
    assert structured["sets"]["nodes"] == ["city_0", "city_1"]
    assert structured["sets"]["directed_arcs"] == [["city_0", "city_1"]]
    assert structured["parameters"]["total_supply"] == 5
    assert structured["parameters"]["total_demand"] == 5
    assert "Ship[i,j]" in structured["variables"]
    assert "supply_demand" not in json.dumps(structured)


def test_smallbucket_backtranslation_compacts_wrong_unit_production_cost_statement() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "sets": {"items": [0, 1], "machines": [0], "periods": [0, 1]},
                "global_costs": {
                    "setup_cost_per_active_item_machine_period": 180.5,
                    "startup_cost_per_start_event": 120.25,
                    "omitted_objective_terms": ["per-unit production cost"],
                    "source_objective_note": "There is no per-unit production cost term in this source model.",
                },
                "item_cost_table": {
                    "columns": ["item", "holding_cost", "backlog_cost"],
                    "rows": [[0, 1.5, 9.0], [1, 2.0, 10.0]],
                },
                "machine_table": {
                    "columns": ["machine", "capacity", "startup_time"],
                    "rows": [[0, 100, 12]],
                },
                "demand_matrix": {
                    "columns": [0, 1],
                    "rows": [
                        {"item": 0, "values": [20, 25]},
                        {"item": 1, "values": [30, 35]},
                    ],
                },
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_singlelevelsmallbucket"},
        (
            "A factory must plan production. The objective includes a production cost per unit of 180.5, "
            "startup cost, inventory cost, and backlog cost. Some demand values are truncated."
        ),
        source_compact_data=source_compact_data,
    )

    assert "There is no per-unit production cost" in statement
    assert "setup_cost_per_active_item_machine_period=180.5" in statement
    assert "startup_cost_per_start_event=120.25" in statement
    assert "item=0, holding=1.5, backlog=9.0" in statement
    assert "machine=0, capacity=100, startup_time=12" in statement
    assert "item=1: 30, 35" in statement
    assert "truncated" not in statement.lower()


def test_structure_assignment_backtranslation_rewrites_partial_provider_request_statement() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_structure_assignment_tables": {
                "sets": {"peaks": [0, 1], "amino_acids": [0, 1, 2]},
                "required_assignment_count": 2,
                "distance_threshold": 5.0,
                "assignment_cost_matrix": {
                    "columns": [0, 1, 2],
                    "rows": [
                        {"peak": 0, "values": [0.12, 0.34, 0.56]},
                        {"peak": 1, "values": [0.22, 0.11, 0.44]},
                    ],
                },
                "acid_compatibility_matrix": {
                    "columns": [0, 1, 2],
                    "rows": [
                        {"amino_acid": 0, "values": [1, 0, 1]},
                        {"amino_acid": 1, "values": [0, 1, 1]},
                        {"amino_acid": 2, "values": [1, 1, 1]},
                    ],
                },
                "noe_relation_pairs": [[0, 1]],
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_structure_based_assignment"},
        (
            "A marketplace assigns providers to requests. The compatibility limit table is partially "
            "available; missing entries are assumed. b[(0, perm?)] = 2."
        ),
        source_compact_data=source_compact_data,
    )

    lowered = statement.lower()
    assert "nmr spectroscopy" in lowered
    assert "provider" not in lowered
    assert "request" not in lowered
    assert "perm?" not in statement
    assert "partially available" not in lowered
    assert "assumed" not in lowered
    assert "exactly 2 peak-amino-acid assignments" in statement
    assert "Assignment cost matrix c[p,a] columns are amino acids: 0, 1, 2" in statement
    assert "- peak=0: 0.12, 0.34, 0.56" in statement
    assert "NOE-related peak pairs: (0,1)" in statement
    assert "- amino_acid=1: 0, 1, 1" in statement
    assert "x[p,a] + x[q,b] <= compatibility[a,b] + 1" in statement


def test_steel4_backtranslation_rewrites_market_bound_stage_capacity_drift() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_steel_product_mix_tables": {
                "sets": {
                    "products": ["product_0", "product_1"],
                    "production_stages": ["stage_0", "stage_1"],
                },
                "product_table": {
                    "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
                    "rows": [
                        ["product_0", 296, 37.66, 100.2],
                        ["product_1", 463, 12.36, 55.19],
                    ],
                },
                "stage_capacity_table": {
                    "columns": ["stage", "available_hours"],
                    "rows": [["stage_0", 61.13], ["stage_1", 104.81]],
                },
                "processing_hours_per_ton_matrix": {
                    "columns": ["stage_0", "stage_1"],
                    "rows": [
                        {"product": "product_0", "values": [0.1, 0.25]},
                        {"product": "product_1", "values": [0.33, 0.14]},
                    ],
                },
            }
        },
    }

    statement = _ensure_required_problem_statement_tables(
        {"generator_id": "optmath_steel4"},
        (
            "A product mix plan has product_0 minimum commitment 苹果 50 tons and maximum market 110 tons. "
            "Stage 0 has available hours 100.2 and Stage 1 has available hours 55.19."
        ),
        source_compact_data=source_compact_data,
    )

    assert "苹果" not in statement
    assert "product=product_0, profit=296, min_commit=37.66, max_market=100.2" in statement
    assert "product=product_1, profit=463, min_commit=12.36, max_market=55.19" in statement
    assert "stage=stage_0, available_hours=61.13" in statement
    assert "stage=stage_1, available_hours=104.81" in statement
    assert "- product=product_0: 0.1, 0.25" in statement
    assert "do not copy product market bounds into stage capacities" in statement
    assert "available hours 100.2" not in statement


def test_steel4_structured_data_rebuilt_from_compact_source() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_steel_product_mix_tables": {
                "sets": {
                    "products": ["product_0"],
                    "production_stages": ["stage_0"],
                },
                "product_table": {"columns": [], "rows": []},
                "stage_capacity_table": {"columns": [], "rows": []},
                "processing_hours_per_ton_matrix": {"columns": [], "rows": []},
            }
        },
    }

    structured = _structured_problem_data_from_compact_source(
        {"generator_id": "optmath_steel4"},
        source_compact_data=source_compact_data,
        fallback={"parameters": {"stage_available_hours": {"stage_0": 100.2}}},
    )

    assert structured["entities"] == ["steel products", "shared production stages"]
    assert structured["sets"]["P"] == ["product_0"]
    assert structured["sets"]["S"] == ["stage_0"]
    assert "Production[p]" in structured["variables"]
    assert "stage_available_hours" not in json.dumps(structured)


def test_structure_assignment_structured_data_rebuilt_from_compact_source() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_structure_assignment_tables": {
                "sets": {"peaks": [0, 1], "amino_acids": [0, 1, 2]},
                "required_assignment_count": 2,
                "distance_threshold": 5.0,
                "assignment_cost_matrix": {"columns": [0, 1, 2], "rows": []},
                "acid_compatibility_matrix": {"columns": [0, 1, 2], "rows": []},
                "noe_relation_pairs": [[0, 1]],
            }
        },
    }

    structured = _structured_problem_data_from_compact_source(
        {"generator_id": "optmath_structure_based_assignment"},
        source_compact_data=source_compact_data,
        fallback={"entities": ["providers", "requests"]},
    )

    assert structured["entities"] == ["NMR spectral peaks", "protein amino acids"]
    assert structured["sets"]["P"] == [0, 1]
    assert structured["sets"]["A"] == [0, 1, 2]
    assert structured["parameters"]["required_assignment_count"] == 2
    assert "x[p,a]" in structured["variables"]
    assert "providers" not in json.dumps(structured).lower()


def test_backtranslation_mock_records_generic_prompt_after_compact_router_disabled(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "instance_id": "inst_optmath_supplychain_000001",
                "generator_id": "optmath_supplychain",
                "optimization_sense": "minimize",
                "reference_answer": {"objective_value": 10.0, "status": "OPTIMAL"},
            }
        ],
    )
    output_dir = tmp_path / "backtranslation"

    backtranslate_instances(input_path, output_dir, EngineConfig(), mock=True, limit=1)

    row = read_jsonl(output_dir / "backtranslation_candidates.jsonl")[0]
    assert row["llm_metadata"]["prompt_name"] == "backtranslation_prompt"
    report = (output_dir / "backtranslation_report.md").read_text(encoding="utf-8")
    assert "backtranslation_prompt" in report


def test_backtranslation_keeps_background_in_downstream_problem_statement() -> None:
    combined = _combine_background_and_statement(
        "A hospital planning office is revising weekly clinic staffing after a demand surge.",
        "There are 3 clinics and 4 service zones. The office must decide which clinics to open.",
    )

    assert combined.startswith("A hospital planning office")
    assert "There are 3 clinics" in combined


def test_backtranslation_payload_helpers_read_controlled_nested_problem_objects() -> None:
    payload = {
        "problem": {
            "background": "A depot planner is revising replenishment after a supplier delay.",
            "statement": "Choose order quantities over three periods while preserving inventory balance.",
            "data": {"periods": [1, 2, 3], "demand": [5, 8, 4]},
        }
    }

    assert _payload_text(payload, "problem_background", "background") == (
        "A depot planner is revising replenishment after a supplier delay."
    )
    assert _payload_text(payload, "problem_statement", "statement") == (
        "Choose order quantities over three periods while preserving inventory balance."
    )
    assert _payload_mapping(payload, "structured_problem_data", "data") == {
        "periods": [1, 2, 3],
        "demand": [5, 8, 4],
    }

    top_level_payload = {
        "problem_statement": "Use the explicit top-level statement.",
        "problem": {"problem_statement": "Do not prefer the nested statement."},
    }
    assert _payload_text(top_level_payload, "problem_statement") == "Use the explicit top-level statement."


def test_nl_quality_filter_uses_structured_business_context_signals() -> None:
    accepted_reasons = evaluate_nl_candidate(
        {
            "problem_background": "A hospital planning office is revising weekly clinic staffing after a demand surge.",
            "problem_statement": (
                "A hospital planning office must decide which clinic teams to assign to service zones. "
                "Each zone has demand, each team has capacity, and the objective is to minimize operating cost."
            ),
            "structured_problem_data": {"entities": ["clinic teams", "service zones"], "units": {"cost": "dollars"}},
            "llm_metadata": {
                "scenario_id": "clinic_assignment",
                "business_trigger": "after a demand surge",
                "organization_profile_id": "hospital_or_clinic_network",
            },
        },
        min_number_coverage=0.0,
    )
    rejected_reasons = evaluate_nl_candidate(
        {
            "problem_statement": "There are sets I and J with coefficients c_ij. Choose variables x_ij to optimize the objective.",
        },
        min_number_coverage=0.0,
    )

    assert accepted_reasons == []
    assert "GENERIC_MATH_ONLY" in rejected_reasons
    assert "MISSING_ORGANIZATION" in rejected_reasons


def test_nl_quality_filter_rejects_uls_backlog_misframing() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "uncapacitated_lot_sizing_without_backlog",
                "period_cost_table": [
                    {"period": "period_1", "demand": 5, "fixed_ordering_cost": 20, "unit_order_cost": 2, "unit_holding_cost": 1},
                    {"period": "period_2", "demand": 7, "fixed_ordering_cost": 30, "unit_order_cost": 3, "unit_holding_cost": 2},
                ],
                "total_demand_big_m": 12,
            }
        },
    }
    bad_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_uncapacitatedlotsizing",
            "problem_background": "A warehouse planner is revising replenishment after period demand forecasts changed.",
            "problem_statement": (
                "A warehouse planner must decide order quantities over two periods. Backorders are allowed, "
                "ending backlog must be zero, and backlog penalties are included with ordering and holding costs. "
                "The period demands are 5 and 7 units."
            ),
            "structured_problem_data": {"entities": ["periods", "orders", "inventory"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )
    good_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_uncapacitatedlotsizing",
            "problem_background": "A warehouse planner is revising replenishment after period demand forecasts changed.",
            "problem_statement": (
                "A warehouse planner must decide order quantities over two periods. No backlog is allowed, "
                "there are no backlog variables or backlog penalties, initial inventory is zero, and final inventory is zero. "
                "The period demands are 5 and 7 units, and total-demand big-M is 12."
            ),
            "structured_problem_data": {"entities": ["periods", "orders", "inventory"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )

    assert "FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added" in bad_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added" not in good_reasons


def test_nl_quality_filter_rejects_clsp_backlog_and_capacity_expansion_misframing() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "capacitated_lot_sizing_without_backlog",
                "source_model_note": "fixed period capacities, no backlog, no capacity-expansion variables",
                "period_demand_table": [
                    {"product": "product_1", "period": "period_1", "demand": 5, "cumulative_demand": 5},
                    {"product": "product_1", "period": "period_2", "demand": 7, "cumulative_demand": 12},
                ],
                "period_capacity_table": [
                    {"period": "period_1", "capacity": 10},
                    {"period": "period_2", "capacity": 10},
                ],
            }
        },
    }
    bad_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "problem_background": "A factory is revising a two-period production release plan.",
            "problem_statement": (
                "A factory must choose product lots and setup decisions over two periods. "
                "Unmet demand may be carried as backlog with backlog costs, and managers may purchase "
                "capacity expansion if the fixed line capacity is insufficient. Final inventory must be zero. "
                "The period demands are 5 and 7 units, and the fixed capacities are 10 and 10 hours."
            ),
            "structured_problem_data": {"entities": ["products", "periods", "production lots"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )
    capacity_bad_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "problem_background": "A factory is revising a two-period production release plan.",
            "problem_statement": (
                "A factory must choose product lots and setup decisions over two periods. "
                "Demand must be fully satisfied with inventory carryover, but managers may purchase capacity expansion "
                "when the fixed line capacity is insufficient. The period demands are 5 and 7 units, "
                "and the fixed capacities are 10 and 10 hours."
            ),
            "structured_problem_data": {"entities": ["products", "periods", "production lots"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )
    zero_final_bad_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "problem_background": "A factory is revising a two-period production release plan.",
            "problem_statement": (
                "A factory must choose product lots and setup decisions over two periods under fixed period capacities. "
                "Demand must be fully satisfied with inventory carryover, and final inventory must be zero. "
                "The period demands are 5 and 7 units, and the fixed capacities are 10 and 10 hours."
            ),
            "structured_problem_data": {"entities": ["products", "periods", "production lots"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )
    good_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "problem_background": "A factory is revising a two-period production release plan.",
            "problem_statement": (
                "A factory must choose product lots and setup decisions over two periods under fixed period capacities. "
                "No backlog, lost sales, unmet-demand slack, or capacity-expansion variables are allowed; inventory carries "
                "production across periods and is not forced to zero by a separate final-inventory constraint. "
                "The period demands are 5 and 7 units, and the fixed capacities are 10 and 10 hours."
            ),
            "structured_problem_data": {"entities": ["products", "periods", "production lots"], "units": {"quantity": "units"}},
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )

    assert "FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added" in bad_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:capacity_expansion_added" in capacity_bad_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:zero_final_inventory_added" in zero_final_bad_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:backlog_or_shortage_added" not in good_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:capacity_expansion_added" not in good_reasons
    assert "FORBIDDEN_SOURCE_MISFRAMING:zero_final_inventory_added" not in good_reasons


def test_nl_semantic_coherence_ignores_structured_forbidden_structure_notes() -> None:
    reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_clsp_expand_capacity",
            "problem_background": "A factory is revising a two-period production release plan.",
            "problem_statement": (
                "A factory must choose product lots and setup decisions over two periods under fixed period capacities. "
                "No backlog, no lost sales, no unmet-demand slack, and no capacity-expansion variables are allowed. "
                "The period demands are 5 and 7 units, and the fixed capacities are 10 and 10 hours."
            ),
            "structured_problem_data": {
                "forbidden_structures": [
                    "backlog variables",
                    "lost sales",
                    "capacity expansion variables",
                ]
            },
        },
        min_number_coverage=0.0,
    )
    net1_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_net1",
            "problem_background": "A logistics planner is revising a directed network dispatch plan.",
            "problem_statement": (
                "Ship flow on declared arcs only. Total supply is 45 and total demand is 45. "
                "Do not add binary activation, vehicles, routes, time windows, unmet-demand slack, or arcs not listed."
            ),
            "structured_problem_data": {
                "forbidden_structures": [
                    "vehicle routing",
                    "time windows",
                    "facility opening",
                ]
            },
        },
        min_number_coverage=0.0,
    )

    assert not any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)
    assert not any(reason.startswith("SEMANTIC_COHERENCE:") for reason in net1_reasons)


def test_nl_number_coverage_uses_lp_when_formula_only_has_binary_domain_numbers() -> None:
    coverage = _number_coverage(
        {
            "problem_statement": (
                "Assign 11 workloads WK-0 through WK-10 to servers with capacity 74. "
                "The workload consumptions are 30, 53, 35, 23, 16, 31, 58, 12, 54, 11, and 20."
            ),
            "source_math_formula": "x_i_j in {0,1}",
            "source_lp_text": "capacity 74 weights 30 53 35 23 16 31 58 12 54 11 20",
        }
    )

    assert coverage == 1.0
    assert "10" in _canonical_numbers("WK-10")
    assert "-10" not in _canonical_numbers("WK-10")


def test_nl_number_coverage_prefers_source_compact_data() -> None:
    row = {
        "generator_id": "optmath_aircraftlanding",
        "problem_statement": (
            "An airport sequencing team must schedule aircraft a0 and a1 after a weather disruption. "
            "Aircraft a0 has earliest 10, target 20, latest 40, early penalty 3, and late penalty 7. "
            "Aircraft a1 has earliest 12, target 25, latest 45, early penalty 4, and late penalty 8. "
            "If a0 lands before a1, separation is 6 minutes; if a1 lands before a0, separation is 5 minutes. "
            "The planner decides landing times and order variables to minimize early and late penalty cost."
        ),
        "source_math_formula": "999 888 777 666 555",
        "source_compact_data": {
            "available": True,
            "value": {
                "compact_aircraft_landing_tables": {
                    "aircraft_time_penalty_table": [
                        {
                            "aircraft": "a0",
                            "earliest_landing": 10,
                            "target_landing": 20,
                            "latest_landing": 40,
                            "early_penalty_per_minute": 3,
                            "late_penalty_per_minute": 7,
                        },
                        {
                            "aircraft": "a1",
                            "earliest_landing": 12,
                            "target_landing": 25,
                            "latest_landing": 45,
                            "early_penalty_per_minute": 4,
                            "late_penalty_per_minute": 8,
                        },
                    ],
                    "ordered_pair_separation_matrix": {
                        "columns": ["a0", "a1"],
                        "rows": [
                            {"aircraft_before": "a0", "values": [None, 6]},
                            {"aircraft_before": "a1", "values": [5, None]},
                        ],
                    },
                }
            },
        },
    }

    assert _number_coverage(row) == 1.0


def test_nl_number_coverage_uses_steel3_compact_product_mix_numbers() -> None:
    row = {
        "generator_id": "optmath_steel3",
        "problem_statement": (
            "A steel mill must decide tons of Product A, Product B, and Product C after a capacity shortage. "
            "The shared casting machine has 22.1 available hours. "
            "Product A has profit 71, production rate 8 tons per hour, processing time 0.125 hours per ton, "
            "minimum commitment 29 tons, and maximum market 89 tons. "
            "Product B has profit 64, production rate 6 tons per hour, processing time 0.1667 hours per ton, "
            "minimum commitment 44 tons, and maximum market 55 tons. "
            "Product C has profit 70, production rate 7 tons per hour, processing time 0.1429 hours per ton, "
            "minimum commitment 31 tons, and maximum market 60 tons."
        ),
        "source_math_formula": "999 888 777 666",
        "source_compact_data": {
            "available": True,
            "value": {
                "compact_steel_product_mix_tables": {
                    "product_table": {
                        "rows": [
                            ["Product A", 71, 8, 0.125, 29, 89],
                            ["Product B", 64, 6, 0.1667, 44, 55],
                            ["Product C", 70, 7, 0.1429, 31, 60],
                        ],
                    },
                    "time_capacity": {"available_hours": 22.1},
                }
            },
        },
    }

    assert _number_coverage(row) == 1.0


def test_electrical_power_compact_rewrite_and_number_coverage() -> None:
    compact = {
        "available": True,
        "value": {
            "compact_electrical_power_tables": {
                "sets": {
                    "generator_types": ["type_0", "type_1"],
                    "time_periods": ["period_0", "period_1"],
                },
                "generator_type_table": [
                    {
                        "generator_type": "type_0",
                        "base_operating_cost_per_period": 80,
                        "per_mw_generation_cost": 3,
                        "startup_cost": 200,
                        "minimum_output_mw": 20,
                        "maximum_output_mw": 100,
                        "generators_available": 2,
                        "generators_on_initially": 1,
                    },
                    {
                        "generator_type": "type_1",
                        "base_operating_cost_per_period": 60,
                        "per_mw_generation_cost": 5,
                        "startup_cost": 240,
                        "minimum_output_mw": 10,
                        "maximum_output_mw": 90,
                        "generators_available": 3,
                        "generators_on_initially": 0,
                    },
                ],
                "period_demand_table": [
                    {"period": "period_0", "demand_mw": 120, "reserve_required_capacity_mw": 138},
                    {"period": "period_1", "demand_mw": 150, "reserve_required_capacity_mw": 172.5},
                ],
            }
        },
    }
    row = {"generator_id": "optmath_electrical_power"}

    statement = _ensure_required_problem_statement_tables(
        row,
        "A long generated statement with missing tables.",
        source_compact_data=compact,
    )
    structured = _structured_problem_data_from_compact_source(
        row,
        source_compact_data=compact,
        fallback={"old": "freeform"},
    )
    coverage = _number_coverage(
        {
            **row,
            "problem_statement": statement,
            "source_math_formula": "999 888 777",
            "source_compact_data": compact,
        }
    )

    assert "Generator type data rows use" in statement
    assert "Period demand rows use" in statement
    assert len(statement) < 2500
    assert structured["variables"]["NumGenerators[t,p]"].startswith("nonnegative integer")
    assert structured["constraints"]["reserve"].startswith("sum_t maximum_output")
    assert coverage == 1.0


def test_marketshare_compact_rewrite_and_number_coverage() -> None:
    compact = {
        "available": True,
        "value": {
            "compact_marketshare_tables": {
                "sets": {
                    "companies": [0, 1],
                    "markets": [0, 1],
                    "products": [0, 1],
                },
                "demand_table": [
                    {"market": 0, "product": 0, "demand": 10},
                    {"market": 0, "product": 1, "demand": 12},
                    {"market": 1, "product": 0, "demand": 14},
                    {"market": 1, "product": 1, "demand": 16},
                ],
                "profit_table": [
                    {"company": 0, "market": 0, "product": 0, "unit_cost": 2, "unit_revenue": 8, "unit_profit": 6},
                    {"company": 0, "market": 0, "product": 1, "unit_cost": 3, "unit_revenue": 10, "unit_profit": 7},
                    {"company": 0, "market": 1, "product": 0, "unit_cost": 4, "unit_revenue": 12, "unit_profit": 8},
                    {"company": 0, "market": 1, "product": 1, "unit_cost": 5, "unit_revenue": 14, "unit_profit": 9},
                    {"company": 1, "market": 0, "product": 0, "unit_cost": 1, "unit_revenue": 10, "unit_profit": 9},
                    {"company": 1, "market": 0, "product": 1, "unit_cost": 2, "unit_revenue": 11, "unit_profit": 9},
                    {"company": 1, "market": 1, "product": 0, "unit_cost": 3, "unit_revenue": 12, "unit_profit": 9},
                    {"company": 1, "market": 1, "product": 1, "unit_cost": 4, "unit_revenue": 13, "unit_profit": 9},
                ],
            }
        },
    }
    row = {"generator_id": "optmath_marketshare"}

    statement = _ensure_required_problem_statement_tables(
        row,
        "A long generated statement with optional market coverage and missing demand values.",
        source_compact_data=compact,
    )
    structured = _structured_problem_data_from_compact_source(
        row,
        source_compact_data=compact,
        fallback={"old": "freeform"},
    )
    coverage = _number_coverage(
        {
            **row,
            "problem_statement": statement,
            "source_math_formula": "999 888 777",
            "source_compact_data": compact,
        }
    )

    assert "Demand table by market" in statement
    assert "Unit-profit table by company and market" in statement
    assert "optional market-coverage constraints" in statement
    assert len(statement) < 2500
    assert structured["variables"]["Supply[i,j,k]"].startswith("nonnegative integer")
    assert structured["constraints"]["market_product_demand"].startswith("sum_i Supply")
    assert coverage == 1.0


def test_marketshare_compact_rewrite_supports_profit_matrix_format() -> None:
    compact = {
        "available": True,
        "value": {
            "compact_marketshare_tables": {
                "model_family": "integer_market_share_allocation",
                "sets": {
                    "companies": [0, 1],
                    "markets": [0, 1],
                    "products": [0, 1],
                },
                "demand_matrix": {
                    "columns": [0, 1],
                    "rows": [
                        {"market": 0, "values": [10, 12]},
                        {"market": 1, "values": [14, 16]},
                    ],
                },
                "unit_profit_matrices_by_company": {
                    "columns": [0, 1],
                    "rows": [
                        {"company": 0, "market": 0, "values": [6, 7]},
                        {"company": 0, "market": 1, "values": [8, 9]},
                        {"company": 1, "market": 0, "values": [9, 10]},
                        {"company": 1, "market": 1, "values": [11, 12]},
                    ],
                },
            }
        },
    }
    row = {"generator_id": "optmath_marketshare"}

    statement = _ensure_required_problem_statement_tables(
        row,
        "A generated statement incorrectly says the allocation variables are continuous.",
        source_compact_data=compact,
    )
    structured = _structured_problem_data_from_compact_source(
        row,
        source_compact_data=compact,
        fallback={"old": "freeform"},
    )
    coverage = _number_coverage(
        {
            **row,
            "problem_statement": statement,
            "source_math_formula": "999 888 777",
            "source_compact_data": compact,
        }
    )

    assert "Decision variable Supply[i,j,k] is the nonnegative integer quantity" in statement
    assert "market=0: 0=10, 1=12" in statement
    assert "company=1, market=1: 0=11, 1=12" in statement
    assert structured["variables"]["Supply[i,j,k]"].startswith("nonnegative integer")
    assert structured["parameters"]["unit_profit_table"].endswith("unit_profit_matrices_by_company")
    assert coverage == 1.0


def test_nl_filter_allows_symbolic_objective_value_but_rejects_solver_leakage() -> None:
    symbolic_reasons = evaluate_nl_candidate(
        {
            "problem_background": "A regional utility operator is preparing a deployment plan during a sustainability review.",
            "problem_statement": (
                "A regional utility operator must select monitoring sites. "
                "The total objective value is the sum of pairwise distance times z[i,j]. "
                "The planner decides binary site selections to maximize spatial spread."
            ),
            "structured_problem_data": {"sites": ["node_0", "node_1"], "units": {"distance": "km"}},
            "llm_metadata": {
                "scenario_id": "sensor_dispersion",
                "business_trigger": "during a sustainability review",
                "organization_profile_id": "regulated_utility_operator",
            },
        },
        min_number_coverage=0.0,
    )
    leakage_reasons = evaluate_nl_candidate(
        {
            "problem_statement": (
                "A planner chooses sites. The optimal objective value reported by the solver is 10."
            ),
        },
        min_number_coverage=0.0,
    )

    assert not any(reason.startswith("FORBIDDEN_PHRASE") for reason in symbolic_reasons)
    assert any(reason.startswith("FORBIDDEN_PHRASE:optimal objective,objective value") for reason in leakage_reasons)


def test_nl_business_context_accepts_vehicle_participant_entities() -> None:
    reasons = evaluate_nl_candidate(
        {
            "problem_background": "A field-service platform is finalizing its weekly roster after a schedule reset.",
            "problem_statement": (
                "A field-service platform must assign participants to available cars. "
                "The planner decides which eligible participant-car pairs to use so that each participant "
                "and each car is used at most once while maximizing assigned service coverage."
            ),
            "structured_problem_data": {
                "participants": ["participant_0", "participant_1"],
                "cars": ["car_0", "car_1"],
                "eligibility": {"participant_0": ["car_0"], "participant_1": ["car_1"]},
            },
            "llm_metadata": {
                "scenario_id": "car_assignment",
                "business_trigger": "after a schedule reset",
                "organization_profile_id": "field_service_platform",
            },
        },
        min_number_coverage=0.0,
    )

    assert "GENERIC_MATH_ONLY" not in reasons
    assert reasons == []


def test_nl_business_context_accepts_sequence_and_reschedule_decisions() -> None:
    reasons = evaluate_nl_candidate(
        {
            "problem_background": "An airport recovery unit is rebuilding arrivals after a weather disruption.",
            "problem_statement": (
                "The arrival sequencing controller must reschedule landing times for aircraft_0 and aircraft_1. "
                "The plan sequences the aircraft on one runway while respecting landing windows, separation times, "
                "and early or late penalty costs."
            ),
            "structured_problem_data": {
                "aircraft": ["aircraft_0", "aircraft_1"],
                "units": {"time": "minutes", "penalty": "dollars per minute"},
            },
            "llm_metadata": {
                "scenario_id": "aircraft_landing",
                "business_trigger": "after a weather disruption",
                "organization_profile_id": "airport_recovery_unit",
            },
        },
        min_number_coverage=0.0,
    )

    assert "MISSING_DECISION_CONTEXT" not in reasons
    assert reasons == []


def test_nl_business_context_accepts_common_or_decision_verbs_and_nested_sets() -> None:
    provider_reasons = evaluate_nl_candidate(
        {
            "problem_background": "A public digital marketplace agency is finalizing quarterly service commitments.",
            "problem_statement": (
                "The agency must match service providers to service request categories. "
                "The planner decides which eligible provider-request assignments to use so each provider "
                "and each request category is matched at most once."
            ),
            "structured_problem_data": {
                "sets": {
                    "participants": ["participant_0", "participant_1"],
                    "cars": ["car_0", "car_1"],
                },
                "eligibility": {"participant_0": ["car_0"], "participant_1": ["car_1"]},
            },
            "llm_metadata": {
                "scenario_id": "provider_request_matching",
                "business_trigger": "before quarterly commitments",
                "organization_profile_id": "public_marketplace_agency",
            },
        },
        min_number_coverage=0.0,
    )
    formulation_reasons = evaluate_nl_candidate(
        {
            "problem_background": "A healthcare catering team is responding after a supplier disruption.",
            "problem_statement": (
                "The team must design and formulate a meal kit, identify ingredient quantities, "
                "and assemble a least-cost blend that satisfies nutrient bounds. The operating decision is the "
                "serving amount for each available ingredient, with costs, nutrient minimums, nutrient maximums, "
                "and ingredient availability limits all stated by the planning office."
            ),
            "structured_problem_data": {
                "sets": {"foods": ["food_0", "food_1"], "nutrients": ["nutrient_0"]},
                "units": {"cost": "dollars"},
            },
            "llm_metadata": {
                "scenario_id": "diet_formulation",
                "business_trigger": "after a supplier disruption",
                "organization_profile_id": "healthcare_catering_team",
            },
        },
        min_number_coverage=0.0,
    )

    assert provider_reasons == []
    assert formulation_reasons == []


def test_nl_quality_filter_rejects_missing_source_compact_numbers() -> None:
    reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_uncapacitatedlotsizingbacklogging",
            "problem_background": "A regional maintenance operator is recovering after an outage.",
            "problem_statement": (
                "A regional maintenance operator must decide order quantities over 5 periods. "
                "Period demands are 17, 39, 66, 11, and 5 units, and the total-demand big-M is 138. "
                "The planner minimizes ordering, holding, and backlog costs."
            ),
            "structured_problem_data": {
                "entities": ["periods", "orders", "inventory", "backlog"],
                "units": {"cost": "dollars", "quantity": "units"},
                "period_data": [
                    {"period": 1, "demand": 17},
                    {"period": 2, "demand": 39},
                    {"period": 3, "demand": 66},
                    {"period": 4, "demand": 11},
                    {"period": 5, "demand": 5},
                ],
            },
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_lotsizing_tables": {
                        "period_cost_table": [
                            {"period": "period_1", "demand": 17, "fixed_ordering_cost": 197, "unit_order_cost": 5, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
                            {"period": "period_2", "demand": 39, "fixed_ordering_cost": 132, "unit_order_cost": 3, "unit_holding_cost": 3, "unit_backlog_penalty": 5},
                            {"period": "period_3", "demand": 66, "fixed_ordering_cost": 50, "unit_order_cost": 10, "unit_holding_cost": 1, "unit_backlog_penalty": 6},
                            {"period": "period_4", "demand": 11, "fixed_ordering_cost": 103, "unit_order_cost": 2, "unit_holding_cost": 3, "unit_backlog_penalty": 13},
                            {"period": "period_5", "demand": 36, "fixed_ordering_cost": 182, "unit_order_cost": 7, "unit_holding_cost": 1, "unit_backlog_penalty": 14},
                        ],
                        "total_demand_big_m": 169,
                    }
                },
            },
            "llm_metadata": {
                "scenario_id": "lot_sizing_backlogging_service_parts",
                "business_trigger": "after an outage",
                "organization_profile_id": "regional_operator",
            },
        },
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("MISSING_SOURCE_COMPACT_NUMBERS") for reason in reasons)
    assert any("36" in reason and "169" in reason for reason in reasons)


def test_nl_quality_filter_rejects_steel4_source_number_drift() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_steel_product_mix_tables": {
                "sets": {
                    "products": ["product_0", "product_1"],
                    "production_stages": ["stage_0", "stage_1"],
                },
                "product_table": {
                    "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
                    "rows": [
                        ["product_0", 296, 43, 122],
                        ["product_1", 463, 18, 128],
                    ],
                },
                "stage_capacity_table": {
                    "columns": ["stage", "available_hours"],
                    "rows": [["stage_0", 61.13], ["stage_1", 104.81]],
                },
                "processing_hours_per_ton_matrix": {
                    "columns": ["stage_0", "stage_1"],
                    "rows": [
                        {"product": "product_0", "values": [0.10, 0.25]},
                        {"product": "product_1", "values": [0.33, 0.14]},
                    ],
                },
            }
        },
    }
    drift_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_steel4",
            "problem_background": "A steel mill is resetting product commitments after a capacity review.",
            "problem_statement": (
                "A steel production planner must choose continuous tons of product_0 and product_1. "
                "The objective is to maximize profit. Product_0 has profit 296, minimum commitment 37.66, "
                "and maximum market 100.20. Product_1 has profit 463, minimum commitment 12.36, and maximum market 55.19. "
                "Stage 0 has 61.13 hours and uses 0.10 and 0.33 hours per ton. "
                "Stage 1 has 104.81 hours and uses 0.25 and 0.14 hours per ton."
            ),
            "structured_problem_data": {
                "products": ["product_0", "product_1"],
                "stages": ["stage_0", "stage_1"],
                "profit_per_ton": {"product_0": 296, "product_1": 463},
                "minimum_commitment_tons": {"product_0": 37.66, "product_1": 12.36},
                "maximum_market_tons": {"product_0": 100.20, "product_1": 55.19},
                "available_hours": {"stage_0": 61.13, "stage_1": 104.81},
            },
            "source_compact_data": source_compact_data,
            "llm_metadata": {
                "scenario_id": "steel4_product_mix",
                "business_trigger": "after a capacity review",
                "organization_profile_id": "steel_mill",
            },
        },
        min_number_coverage=0.0,
    )
    faithful_reasons = evaluate_nl_candidate(
        {
            "generator_id": "optmath_steel4",
            "problem_background": "A steel mill is resetting product commitments after a capacity review.",
            "problem_statement": (
                "A steel production planner must choose continuous tons of product_0 and product_1. "
                "The objective is to maximize profit. Product_0 has profit 296, minimum commitment 43, "
                "and maximum market 122. Product_1 has profit 463, minimum commitment 18, and maximum market 128. "
                "Stage 0 has 61.13 hours and uses 0.10 and 0.33 hours per ton. "
                "Stage 1 has 104.81 hours and uses 0.25 and 0.14 hours per ton."
            ),
            "structured_problem_data": {
                "products": ["product_0", "product_1"],
                "stages": ["stage_0", "stage_1"],
                "profit_per_ton": {"product_0": 296, "product_1": 463},
                "minimum_commitment_tons": {"product_0": 43, "product_1": 18},
                "maximum_market_tons": {"product_0": 122, "product_1": 128},
                "available_hours": {"stage_0": 61.13, "stage_1": 104.81},
            },
            "source_compact_data": source_compact_data,
            "llm_metadata": {
                "scenario_id": "steel4_product_mix",
                "business_trigger": "after a capacity review",
                "organization_profile_id": "steel_mill",
            },
        },
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("MISSING_SOURCE_COMPACT_NUMBERS") for reason in drift_reasons)
    assert any("43" in reason and "122" in reason for reason in drift_reasons)
    assert not any(reason.startswith("MISSING_SOURCE_COMPACT_NUMBERS") for reason in faithful_reasons)


def test_nl_quality_filter_rejects_high_risk_backtranslation_without_source_compact_data() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_missing_source_compact",
            "bt_status": "PASS",
            "generator_id": "optmath_steel4",
            "language": "en",
            "problem_background": "A steel mill is resetting product commitments after a capacity review.",
            "problem_statement": (
                "A steel production planner must choose continuous tons of product_0 and product_1. "
                "The objective is to maximize profit while respecting product commitments and stage capacities."
            ),
            "structured_problem_data": {
                "products": ["product_0", "product_1"],
                "stages": ["stage_0", "stage_1"],
            },
            "llm_metadata": {
                "scenario_id": "steel4_product_mix",
                "business_trigger": "after a capacity review",
                "organization_profile_id": "steel_mill",
            },
        },
        min_number_coverage=0.0,
    )

    assert "MISSING_SOURCE_COMPACT_DATA" in reasons


def test_nl_quality_filter_rejects_netasgn_source_number_drift() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_project_assignment_tables": {
                "model_family": "continuous_resource_assignment_hours",
                "sets": {
                    "people": ["person_0", "person_1"],
                    "projects": ["project_0", "project_1"],
                },
                "person_supply_table": [
                    {"person": "person_0", "available_hours": 8},
                    {"person": "person_1", "available_hours": 6},
                ],
                "project_demand_table": [
                    {"project": "project_0", "required_hours": 5},
                    {"project": "project_1", "required_hours": 9},
                ],
                "cost_per_hour_matrix": {
                    "columns": ["project_0", "project_1"],
                    "rows": [
                        {"person": "person_0", "values": [32, 11]},
                        {"person": "person_1", "values": [17, 23]},
                    ],
                },
                "max_contribution_hours_matrix": {
                    "columns": ["project_0", "project_1"],
                    "rows": [
                        {"person": "person_0", "values": [5, 8]},
                        {"person": "person_1", "values": [6, 6]},
                    ],
                },
                "total_supply_hours": 14,
                "total_demand_hours": 14,
            }
        },
    }
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_netasgn_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_netasgn",
            "language": "en",
            "problem_background": "A utility service planner is reallocating project hours after a capacity review.",
            "problem_statement": (
                "The planner assigns continuous hours from two people to two projects. "
                "Person 0 has 8 available hours, while Person 1 has enquiry hours. "
                "Project 0 requires 5 hours and Project 1 requires 9 hours. "
                "The cost matrix rows are [32, 11] and [17, 23]. "
                "The contribution limit matrix rows are [5, 8] and [6, }u]."
            ),
            "structured_problem_data": {
                "people": ["person_0", "person_1"],
                "projects": ["project_0", "project_1"],
                "supply_hours": {"person_0": 8, "person_1": "enquiry"},
                "demand_hours": {"project_0": 5, "project_1": 9},
            },
            "source_compact_data": source_compact_data,
        },
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("MISSING_SOURCE_COMPACT_NUMBERS") for reason in reasons)


def test_nl_quality_filter_does_not_require_source_compact_for_low_risk_generator() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_blending_without_source_compact",
            "bt_status": "PASS",
            "generator_id": "optmath_blending_problem",
            "language": "en",
            "problem_background": "A refinery planning team is revising a blend after a demand forecast update.",
            "problem_statement": (
                "A refinery planning team must choose continuous stream quantities to meet a product blend at minimum cost. "
                "The statement lists stream costs, quality bounds, and the total blend amount."
            ),
            "structured_problem_data": {
                "streams": ["stream_0", "stream_1"],
                "units": {"blend": "tons", "cost": "dollars"},
            },
            "llm_metadata": {
                "scenario_id": "blend_planning",
                "business_trigger": "after a forecast update",
                "organization_profile_id": "refinery_operator",
            },
        },
        min_number_coverage=0.0,
    )

    assert "MISSING_SOURCE_COMPACT_DATA" not in reasons


def test_nl_quality_filter_rejects_structure_assignment_business_drift() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_structure_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_structure_based_assignment",
            "language": "en",
            "problem_background": (
                "A regional dispatch manager is preparing field-service schedules after a demand spike."
            ),
            "problem_statement": (
                "The planner assigns technicians to service jobs and inspection jobs. "
                "Vehicle routing is outside the cost calculation, but each worker-job match has a compatibility score. "
                "The goal is to choose assignments that maximize the total service coverage."
            ),
            "structured_problem_data": {
                "providers": ["technician_0", "technician_1"],
                "requests": ["service_job_0", "inspection_job_1"],
                "vehicle_routing": False,
            },
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_structure_assignment_tables": {
                        "sets": {"amino_acids": ["acid_0"], "peaks": ["peak_0"]},
                        "required_assignment_count": 1,
                        "assignment_cost_matrix": {
                            "columns": ["peak_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [0.41]}],
                        },
                        "peak_pair_compatibility_matrix": {
                            "columns": ["peak_0"],
                            "rows": [{"peak": "peak_0", "values": [0]}],
                        },
                        "amino_acid_noe_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [0]}],
                        },
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)
    assert any("technician" in reason or "service job" in reason for reason in reasons)


def test_nl_quality_filter_rejects_structure_assignment_provider_request_drift() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_structure_provider_request_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_structure_based_assignment",
            "language": "en",
            "problem_background": "A digital marketplace is finalizing a same-day allocation plan.",
            "problem_statement": (
                "The platform matches service providers to incoming requests. "
                "Each provider can serve at most one request, each request can be assigned at most once, "
                "and the model minimizes match cost while selecting exactly 1 pairing."
            ),
            "structured_problem_data": {"providers": ["provider_0"], "requests": ["request_0"]},
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_structure_assignment_tables": {
                        "sets": {"amino_acids": ["acid_0"], "peaks": ["peak_0"]},
                        "required_assignment_count": 1,
                        "assignment_cost_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"peak": "peak_0", "values": [0.41]}],
                        },
                        "acid_compatibility_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [1]}],
                        },
                        "noe_relation_pairs": [],
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)
    assert any("service provider" in reason or "request" in reason for reason in reasons)


def test_nl_quality_filter_rejects_structure_assignment_production_slot_drift() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_structure_production_slot_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_structure_based_assignment",
            "language": "en",
            "problem_background": "An industrial plant is preparing a quarterly service commitment.",
            "problem_statement": (
                "A production scheduler assigns production peaks to amino-acid product slots. "
                "Each production peak can use at most one product slot, and exactly 1 assignment is selected."
            ),
            "structured_problem_data": {"production_peaks": ["peak_0"], "product_slots": ["slot_0"]},
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_structure_assignment_tables": {
                        "sets": {"amino_acids": ["acid_0"], "peaks": ["peak_0"]},
                        "required_assignment_count": 1,
                        "assignment_cost_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"peak": "peak_0", "values": [0.41]}],
                        },
                        "acid_compatibility_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [1]}],
                        },
                        "noe_relation_pairs": [],
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)
    assert any("product slot" in reason or "production peak" in reason for reason in reasons)


def test_nl_quality_filter_allows_negated_structure_assignment_forbidden_terms() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_structure_negated",
            "bt_status": "PASS",
            "generator_id": "optmath_structure_based_assignment",
            "language": "en",
            "problem_background": "A protein analysis lab is reconciling NMR evidence after a calibration review.",
            "problem_statement": (
                "The task is an NMR peak-to-amino-acid assignment problem, not technician scheduling, "
                "not service job matching, and not vehicle routing. "
                "The planner chooses binary assignments between peak_0 and acid_0 using cost 0.41, "
                "requires exactly 1 assignment, and uses the stated NOE and peak-pair compatibility data."
            ),
            "structured_problem_data": {
                "amino_acids": ["acid_0"],
                "peaks": ["peak_0"],
                "assignment_cost": {"acid_0": {"peak_0": 0.41}},
                "required_assignment_count": 1,
            },
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_structure_assignment_tables": {
                        "sets": {"amino_acids": ["acid_0"], "peaks": ["peak_0"]},
                        "required_assignment_count": 1,
                        "assignment_cost_matrix": {
                            "columns": ["peak_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [0.41]}],
                        },
                        "peak_pair_compatibility_matrix": {
                            "columns": ["peak_0"],
                            "rows": [{"peak": "peak_0", "values": [0]}],
                        },
                        "amino_acid_noe_matrix": {
                            "columns": ["acid_0"],
                            "rows": [{"amino_acid": "acid_0", "values": [0]}],
                        },
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert not any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)


def test_nl_quality_filter_rejects_smallbucket_unit_production_cost_drift() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_smallbucket_unit_cost_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_singlelevelsmallbucket",
            "language": "en",
            "problem_background": "A regional production team is rebuilding a weekly plan after a capacity outage.",
            "problem_statement": (
                "The planner chooses production amounts, machine setup states, startup events, inventory, "
                "and backlog over several periods. Each unit of production on any machine costs 223.50 dollars, "
                "and the objective includes production costs, startup costs, holding costs, and backlog penalties."
            ),
            "structured_problem_data": {
                "items": ["item_0", "item_1"],
                "machines": ["machine_0"],
                "costs": {"unit_production_cost": 223.50, "startup_cost": 113.88},
            },
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_lotsizing_tables": {
                        "model_family": "single_level_small_bucket_without_unit_production_cost",
                        "global_costs": {
                            "setup_cost_per_active_item_machine_period": 15,
                            "startup_cost_per_start_event": 113.88,
                        },
                        "item_cost_table": {
                            "columns": ["item", "holding_cost", "backlog_cost"],
                            "rows": [["item_0", 2.27, 9.62], ["item_1", 2.86, 10.97]],
                        },
                        "machine_table": {
                            "columns": ["machine", "capacity", "startup_time"],
                            "rows": [["machine_0", 100, 1]],
                        },
                        "demand_matrix": {
                            "columns": ["period_0", "period_1"],
                            "rows": [
                                {"item": "item_0", "values": [10, 12]},
                                {"item": "item_1", "values": [8, 9]},
                            ],
                        },
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)
    assert any("each unit of production" in reason or "production cost" in reason for reason in reasons)


def test_nl_quality_filter_allows_negated_smallbucket_unit_cost_terms() -> None:
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_smallbucket_negated_unit_cost",
            "bt_status": "PASS",
            "generator_id": "optmath_singlelevelsmallbucket",
            "language": "en",
            "problem_background": "A regional production team is rebuilding a weekly plan after a capacity outage.",
            "problem_statement": (
                "The planner chooses production amounts, machine setup states, startup events, inventory, "
                "and backlog over several periods. There is no unit production cost, no material cost, "
                "and no sales revenue term; the objective uses only setup, startup, holding, and backlog costs."
            ),
            "structured_problem_data": {
                "items": ["item_0", "item_1"],
                "machines": ["machine_0"],
                "costs": {"startup_cost": 113.88, "holding_cost": [2.27, 2.86], "backlog_cost": [9.62, 10.97]},
            },
            "source_compact_data": {
                "available": True,
                "truncated": False,
                "value": {
                    "compact_lotsizing_tables": {
                        "model_family": "single_level_small_bucket_without_unit_production_cost",
                        "global_costs": {
                            "setup_cost_per_active_item_machine_period": 15,
                            "startup_cost_per_start_event": 113.88,
                        },
                        "item_cost_table": {
                            "columns": ["item", "holding_cost", "backlog_cost"],
                            "rows": [["item_0", 2.27, 9.62], ["item_1", 2.86, 10.97]],
                        },
                        "machine_table": {
                            "columns": ["machine", "capacity", "startup_time"],
                            "rows": [["machine_0", 100, 1]],
                        },
                        "demand_matrix": {
                            "columns": ["period_0", "period_1"],
                            "rows": [
                                {"item": "item_0", "values": [10, 12]},
                                {"item": "item_1", "values": [8, 9]},
                            ],
                        },
                    }
                },
            },
        },
        min_chars=0,
        min_number_coverage=0.0,
    )

    assert not any(reason.startswith("SEMANTIC_COHERENCE:") for reason in reasons)


def test_nl_quality_filter_rejects_uls_without_backlog_source_number_drift() -> None:
    source_compact_data = {
        "available": True,
        "truncated": False,
        "value": {
            "compact_lotsizing_tables": {
                "model_family": "uncapacitated_lot_sizing_without_backlog",
                "period_cost_table": [
                    {"period": "period_1", "demand": 3, "fixed_ordering_cost": 37, "unit_order_cost": 2, "unit_holding_cost": 3},
                    {"period": "period_2", "demand": 5, "fixed_ordering_cost": 12, "unit_order_cost": 2, "unit_holding_cost": 2},
                ],
                "total_demand_big_m": 8,
            }
        },
    }
    reasons = evaluate_nl_candidate(
        {
            "bt_id": "bt_uls_drift",
            "bt_status": "PASS",
            "generator_id": "optmath_uncapacitatedlotsizing",
            "language": "en",
            "problem_background": "A warehouse planner is revising replenishment after period demand forecasts changed.",
            "problem_statement": (
                "A warehouse planner must decide order quantities over two periods with no backlog. "
                "Period demands are 3 and 4 units, fixed ordering costs are 37 and 12, unit order costs are 2 and 2, "
                "holding costs are 3 and 2, and total-demand big-M is 7."
            ),
            "structured_problem_data": {
                "periods": ["period_1", "period_2"],
                "demands": [3, 4],
                "big_m": 7,
            },
            "source_compact_data": source_compact_data,
            "llm_metadata": {
                "scenario_id": "lot_sizing_without_backlog",
                "business_trigger": "after forecast changes",
                "organization_profile_id": "warehouse_operator",
            },
        },
        min_number_coverage=0.0,
    )

    assert any(reason.startswith("MISSING_SOURCE_COMPACT_NUMBERS") for reason in reasons)
    assert any("5" in reason and "8" in reason for reason in reasons)


def test_nl_quality_filter_rejects_cjk_contamination_in_english_problem() -> None:
    english_reasons = evaluate_nl_candidate(
        {
            "language": "en",
            "problem_background": "A steel mill is resetting product commitments after a capacity review.",
            "problem_statement": (
                "A steel production planner must choose continuous tons of product_0 and product_1. "
                "Product_0 has minimum commitment 苹果 43 tons and product_1 has 18 tons."
            ),
            "structured_problem_data": {
                "products": ["product_0", "product_1"],
                "units": {"production": "tons"},
            },
            "llm_metadata": {
                "scenario_id": "steel4_product_mix",
                "business_trigger": "after a capacity review",
                "organization_profile_id": "steel_mill",
            },
        },
        min_number_coverage=0.0,
    )
    non_english_reasons = evaluate_nl_candidate(
        {
            "language": "zh",
            "problem_background": "一家钢厂在产能复盘后重设产品承诺。",
            "problem_statement": (
                "一家钢厂需要决定 product_0 和 product_1 的连续生产吨数，product_0 的最低承诺为 43 吨。"
            ),
            "structured_problem_data": {
                "products": ["product_0", "product_1"],
                "units": {"production": "tons"},
            },
        },
        min_number_coverage=0.0,
    )

    assert "NON_ENGLISH_SCRIPT:CJK" in english_reasons
    assert "NON_ENGLISH_SCRIPT:CJK" not in non_english_reasons


def test_streaming_cpt_pipeline_runs_with_mock_llm_and_mock_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_cpt_engine.pipeline import cpt_streaming_pipeline

    seed_input = tmp_path / "quality_validated_instances.jsonl"
    write_jsonl(
        seed_input,
        [
            {
                "instance_id": "inst_streaming_fixture_000001",
                "generator_id": "optmath_knapsack",
                "reference_answer": {"objective_value": 1.0, "status": "OPTIMAL"},
                "lp_text": "Maximize\n obj: x\nSubject To\n c0: x <= 1\nBounds\n x >= 0\nEnd",
                "generation_params": {
                    "compact_cell_tower_tables": {
                        "budget": 1,
                        "tower_table": [{"tower": "tower_0", "cost": 1}],
                    },
                },
            }
        ],
    )
    run_dir = tmp_path / "run"
    stage_dirs = _make_stage_dirs(run_dir)

    def fake_evaluate_forward_output_row(row, **_kwargs):
        assert row["source_compact_data"]["available"] is True
        assert row["source_compact_data"]["value"]["compact_cell_tower_tables"]["budget"] == 1
        return True, AcceptedPair(
            pair_id=f"pair_{row['fm_id']}",
            fm_id=row["fm_id"],
            bt_id=row["bt_id"],
            instance_id=row["instance_id"],
            generator_id=row.get("generator_id", ""),
            problem_statement=row.get("problem_statement") or "",
            reference=row.get("reference_answer") or {},
            generated={
                "solver_result": {
                    "execution_status": "SUCCESS",
                    "status": "OPTIMAL",
                    "objective_value": 1.0,
                    "solution": {
                        "objective_value": 1.0,
                        "feasibility": {
                            "constraint_violation": 0.0,
                            "bound_violation": 0.0,
                            "integer_violation": 0.0,
                            "max_violation": 0.0,
                        },
                    },
                }
            },
            source_compact_data=row.get("source_compact_data") or {},
            correctness=ObjectiveComparison(is_correct=True),
            modeling_answer=row.get("generated_answer") or {},
        )

    monkeypatch.setattr(cpt_streaming_pipeline, "evaluate_forward_output_row", fake_evaluate_forward_output_row)

    result = cpt_streaming_pipeline.run_cpt_streaming_pipeline(
        seed_input=seed_input,
        stage_dirs=stage_dirs,
        config=EngineConfig(quality_thresholds=QualityThresholdsConfig(backtranslation={"min_number_coverage": 0.0})),
        limit=1,
        mock_llm=True,
        llm_concurrency=1,
        forward_eval_concurrency=2,
        solver_threads_per_process=1,
        queue_size=1,
    )

    assert result["backtranslation"]["candidates"] == 2
    assert result["nl_filter"]["accepted"] == 2
    assert result["forward_modeling"]["outputs"] == 2
    assert result["forward_eval"]["accepted"] == 2
    assert result["rendering"]["documents"] == 2
    assert (stage_dirs["export"] / "train.jsonl").exists()
    assert (stage_dirs["forward_eval"] / "forward_eval_rejection_dashboard.md").exists()
    assert read_jsonl(stage_dirs["export"] / "train.jsonl")
    assert read_jsonl(stage_dirs["backtranslation"] / "backtranslation_candidates.jsonl")[0]["source_compact_data"][
        "available"
    ] is True
    assert read_jsonl(stage_dirs["forward_modeling"] / "forward_modeling_outputs.jsonl")[0]["source_compact_data"][
        "available"
    ] is True
    assert read_jsonl(stage_dirs["forward_eval"] / "accepted_pairs.jsonl")[0]["source_compact_data"]["available"] is True


def test_streaming_cpt_pipeline_repairs_forward_eval_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from or_cpt_engine.pipeline import cpt_streaming_pipeline

    seed_input = tmp_path / "quality_validated_instances.jsonl"
    write_jsonl(
        seed_input,
        [
            {
                "instance_id": "inst_streaming_repair_000001",
                "generator_id": "optmath_knapsack",
                "reference_answer": {"objective_value": 1.0, "status": "OPTIMAL"},
                "lp_text": "Maximize\n obj: x\nSubject To\n c0: x <= 1\nBounds\n x >= 0\nEnd",
            }
        ],
    )
    run_dir = tmp_path / "run"
    stage_dirs = _make_stage_dirs(run_dir)

    def fake_evaluate_forward_output_row(row, **_kwargs):
        if str(row["fm_id"]).startswith("fmrepair_"):
            return True, AcceptedPair(
                pair_id=f"pair_{row['fm_id']}",
                fm_id=row["fm_id"],
                bt_id=row["bt_id"],
                instance_id=row["instance_id"],
                generator_id=row.get("generator_id", ""),
                problem_statement=row.get("problem_statement") or "",
                reference=row.get("reference_answer") or {},
                generated={"solver_result": {"execution_status": "SUCCESS", "status": "OPTIMAL", "objective_value": 1.0}},
                correctness=ObjectiveComparison(is_correct=True),
                modeling_answer=row.get("generated_answer") or {},
                forward_repair=row.get("forward_repair") or {},
            )
        return False, {
            **row,
            "rejection_stage": "forward_eval",
            "rejection_reason": "OBJECTIVE_MISMATCH",
            "fine_rejection_reason": "objective_mismatch_general",
            "failure_family": "objective_mismatch",
            "repairable": True,
            "forward_eval": {
                "solver_result": {"execution_status": "SUCCESS", "status": "OPTIMAL", "objective_value": 2.0},
                "correctness": {"is_correct": False, "abs_error": 1.0, "rel_error": 1.0},
            },
        }

    monkeypatch.setattr(cpt_streaming_pipeline, "evaluate_forward_output_row", fake_evaluate_forward_output_row)

    result = cpt_streaming_pipeline.run_cpt_streaming_pipeline(
        seed_input=seed_input,
        stage_dirs=stage_dirs,
        config=EngineConfig(quality_thresholds=QualityThresholdsConfig(backtranslation={"min_number_coverage": 0.0})),
        limit=1,
        mock_llm=True,
        llm_concurrency=1,
        forward_eval_concurrency=1,
        solver_threads_per_process=1,
        queue_size=1,
    )

    assert result["forward_eval"]["accepted"] == 2
    assert result["forward_eval"]["rejected"] == 0
    assert result["forward_repair"]["attempted"] == 2
    assert result["forward_repair"]["recovered"] == 2
    assert read_jsonl(stage_dirs["forward_modeling"] / "forward_repair_outputs.jsonl")
    repair_attempts = read_jsonl(stage_dirs["forward_eval"] / "forward_repair_attempts.jsonl")
    assert repair_attempts
    assert repair_attempts[0]["generator_id"] == "optmath_knapsack"
    assert repair_attempts[0]["original_failure_family"] == "objective_mismatch"
    assert repair_attempts[0]["original_fine_rejection_reason"] == "objective_mismatch_general"
    assert repair_attempts[0]["final_rejection_reason"] is None
    assert (stage_dirs["forward_eval"] / "forward_repair_report.md").exists()


def test_scan_smoke_and_generate_with_fixture_generator(tmp_path: Path) -> None:
    generators_dir = _make_fixture_generators(tmp_path)
    registry_dir = tmp_path / "registry"
    result = scan_optmath_generators(generators_dir, registry_dir)
    assert result == {"scanned": 1, "registered": 1, "invalid": 0}

    smoke_dir = tmp_path / "smoke"
    smoke = smoke_test_generators(registry_dir / "generator_registry.jsonl", smoke_dir)
    assert smoke["passed"] == 1

    generation_dir = tmp_path / "instances"
    generated = generate_instances(
        registry_dir / "generator_registry.jsonl",
        smoke_dir / "adapter_smoke_report.jsonl",
        generation_dir,
        run_id="test_run",
        num_instances_per_generator=2,
    )
    assert generated["generated"] == 2
    rows = read_jsonl(generation_dir / "instances_generated.jsonl")
    assert rows[0]["initial_objective_value"] == 1.0


def test_import_optmath_generators_writes_provenance(tmp_path: Path) -> None:
    source = _make_fixture_generators(tmp_path)
    output = tmp_path / "internal"

    dry_run = import_optmath_generators(source, output, dry_run=True)
    assert dry_run["generators"] == 1
    assert output.exists() is False

    result = import_optmath_generators(source, output)

    assert result["copied_files"] == 2
    manifest = read_jsonl(output / "provenance_manifest.jsonl")
    assert manifest[0]["generator_id"] == "optmath_fixture_lp"
    assert manifest[0]["local_modified"] is False
    assert (output / "fixture_lp" / "generator.py").exists()


def test_profile_generation_and_difficulty_metadata(tmp_path: Path) -> None:
    generators_dir = _make_fixture_generators(tmp_path)
    registry_dir = tmp_path / "registry"
    scan_optmath_generators(generators_dir, registry_dir)
    profiles_path = tmp_path / "profiles.yaml"
    profile_generators(
        registry_dir / "generator_registry.jsonl",
        profiles_path,
        report_output=tmp_path / "profile_inventory.md",
    )
    smoke_dir = tmp_path / "smoke"
    smoke_test_generators(registry_dir / "generator_registry.jsonl", smoke_dir)
    generation_dir = tmp_path / "instances"

    generated = generate_instances(
        registry_dir / "generator_registry.jsonl",
        smoke_dir / "adapter_smoke_report.jsonl",
        generation_dir,
        run_id="test_run",
        num_instances_per_generator=4,
        profiles_path=profiles_path,
        difficulty_mix={"level_1": 0.25, "level_2": 0.75},
    )

    assert generated["generated"] == 4
    rows = read_jsonl(generation_dir / "instances_generated.jsonl")
    assert rows[0]["generator_profile_version"] == "v1.1.0"
    assert rows[0]["concept_tags"]
    assert rows[0]["variant_id"] == "base"
    assert rows[0]["canonical_math_signature"]
    assert rows[0]["sub_family"]
    assert {row["difficulty_level"] for row in rows} <= {"level_1", "level_2"}


def test_generator_quality_analysis_and_production_plan(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        "\n".join(
            [
                "version: v1.0.0",
                "profiles:",
                "  fixture:",
                "    enabled: true",
                "    priority: high",
                "    recommended_weight: 3.0",
                "    task_family: general_linear_programming",
                "    expected_acceptance_rate: 0.5",
                "    modeling_concepts: [linear_objective]",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "03_instance_generation" / "instances_generated.jsonl", [{"generator_id": "fixture"}])
    write_jsonl(run_dir / "04_solver_validation" / "solver_validated_instances.jsonl", [{"generator_id": "fixture"}])
    write_jsonl(run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl", [{"generator_id": "fixture"}])
    write_jsonl(run_dir / "06_nl_quality_filter" / "nl_validated_candidates.jsonl", [{"generator_id": "fixture"}])
    write_jsonl(run_dir / "08_forward_eval" / "accepted_pairs.jsonl", [{"generator_id": "fixture", "concept_tags": ["linear_objective"]}])
    write_jsonl(run_dir / "09_cpt_rendering" / "cpt_documents.jsonl", [{"generator_id": "fixture", "metadata": {"concept_tags": ["linear_objective"]}}])
    output_dir = tmp_path / "quality"

    result = analyze_generator_quality(run_dir, output_dir, profiles_path=profiles_path)

    assert result["generators"] == 1
    quality_rows = read_jsonl(output_dir / "generator_quality_report.jsonl")
    assert quality_rows[0]["generator_id"] == "fixture"
    assert quality_rows[0]["quality_pass_rate"] == 1.0
    assert quality_rows[0]["recommended_action"] in {"increase_weight", "keep"}

    plan_result = plan_production(
        target_accepted=10,
        profiles_path=profiles_path,
        output_dir=tmp_path / "plan",
        quality_report_path=output_dir / "generator_quality_report.jsonl",
    )
    assert plan_result["planned_attempts"] >= 10


def test_run_quality_dashboard_reports_stage_funnel_and_rejections(tmp_path: Path) -> None:
    run_dir = tmp_path / "cpt_run"
    valid_text = "\n\n".join(
        [
            "# Optimization Modeling Report",
            "## Problem\nA technician job matching team chooses a feasible plan.",
            "## Model Formulation\nA compact linear optimization model is used.",
            "## Implementation\n```python\nimport gurobipy as gp\nmodel = gp.Model()\nmodel.optimize()\n```",
            "## Optimal Solution\nThe optimal objective value reported by the solver is `10`.",
            "## Feasibility Check\nThe solution is feasible.",
            "## Validation\nThe generated model solved with status `OPTIMAL`.",
        ]
    )
    write_jsonl(
        run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl",
        [
            {"instance_id": "seed_a", "generator_id": "gen_a"},
            {"instance_id": "seed_b", "generator_id": "gen_b"},
        ],
    )
    write_jsonl(
        run_dir / "05_backtranslation" / "backtranslation_candidates.jsonl",
        [
            {
                "bt_id": "bt_a",
                "instance_id": "seed_a",
                "generator_id": "gen_a",
                "problem_statement": "A warehouse plans capacity.",
                "llm_metadata": {
                    "quality_retries": 1,
                    "quality_retry_reasons": ["ValueError: missing problem_statement"],
                },
            },
            {"bt_id": "bt_b", "instance_id": "seed_b", "generator_id": "gen_b", "problem_statement": "An abstract set problem."},
        ],
    )
    write_jsonl(
        run_dir / "06_nl_quality_filter" / "nl_validated_candidates.jsonl",
        [{"bt_id": "bt_a", "instance_id": "seed_a", "generator_id": "gen_a"}],
    )
    write_jsonl(
        run_dir / "06_nl_quality_filter" / "nl_rejected.jsonl",
        [{"bt_id": "bt_b", "instance_id": "seed_b", "generator_id": "gen_b", "rejection_reason": "MISSING_BUSINESS_CONTEXT"}],
    )
    write_jsonl(
        run_dir / "07_forward_modeling" / "forward_modeling_outputs.jsonl",
        [{"fm_id": "fm_a", "bt_id": "bt_a", "instance_id": "seed_a", "generator_id": "gen_a"}],
    )
    write_jsonl(
        run_dir / "07_forward_modeling" / "forward_modeling_rejected.jsonl",
        [
            {
                "fm_id": "fm_bad_static",
                "bt_id": "bt_b",
                "instance_id": "seed_b",
                "generator_id": "optmath_marketshare",
                "rejection_reason": "STATIC_SIGNATURE_MISMATCH:CONTRACT_MISSING_REQUIRED_CONSTRAINT:nonnegative_integer_supply",
                "generated_answer": {
                    "modeling_explanation": "Use continuous supply quantities to meet market-product demand.",
                    "math_model": "maximize net profit subject to demand equality",
                    "gurobipy_code": "\n".join(
                        [
                            "import gurobipy as gp",
                            "from gurobipy import GRB",
                            "model = gp.Model()",
                            "companies = [0, 1]",
                            "markets = [0]",
                            "products = [0]",
                            "supply = model.addVars(companies, markets, products, lb=0, vtype=GRB.CONTINUOUS, name='supply')",
                            "model.addConstr(gp.quicksum(supply[i,0,0] for i in companies) == 5)",
                            "model.setObjective(gp.quicksum(supply[i,0,0] for i in companies), GRB.MAXIMIZE)",
                        ]
                    ),
                },
                "llm_metadata": {
                    "quality_retries": 1,
                    "quality_retry_events": [
                        {"reason": "STATIC_SIGNATURE_MISMATCH:CONTRACT_MISSING_REQUIRED_CONSTRAINT:nonnegative_integer_supply", "request_id": "req_static_1"}
                    ],
                },
            }
        ],
    )
    write_jsonl(
        run_dir / "08_forward_eval" / "accepted_pairs.jsonl",
        [{"pair_id": "pair_a", "fm_id": "fm_a", "bt_id": "bt_a", "instance_id": "seed_a", "generator_id": "gen_a"}],
    )
    write_jsonl(
        run_dir / "08_forward_eval" / "rejected_pairs.jsonl",
        [{"fm_id": "fm_bad", "instance_id": "seed_b", "generator_id": "gen_b", "rejection_reason": "OBJECTIVE_MISMATCH"}],
    )
    write_jsonl(
        run_dir / "08_forward_eval" / "forward_repair_attempts.jsonl",
        [
            {
                "status": "failed_eval",
                "original": {
                    "fm_id": "fm_bad",
                    "instance_id": "seed_b",
                    "generator_id": "gen_b",
                    "failure_family": "objective_mismatch",
                    "fine_rejection_reason": "objective_mismatch_general",
                },
                "repair": {"fm_id": "fmrepair_bad"},
                "final": {
                    "fm_id": "fmrepair_bad",
                    "instance_id": "seed_b",
                    "generator_id": "gen_b",
                    "failure_family": "solver_status_failure",
                    "fine_rejection_reason": "solver_infeasible",
                },
            },
            {
                "status": "recovered",
                "generator_id": "gen_a",
                "original_failure_family": "objective_mismatch",
                "final_failure_family": None,
                "original": {"fm_id": "fm_a", "instance_id": "seed_a"},
                "repair": {"fm_id": "fmrepair_a"},
            },
        ],
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "fixture.log").write_text(
        "2026-06-08 12:00:00,000 INFO [or_cpt_engine.model_call] model_api_call "
        "request_id=req_timeout attempt=1 model=deepseek-v4-flash endpoint_name=dsv4_a "
        "endpoint_base_url=http://127.0.0.1:8005/v1 latency_seconds=180.000 status=error "
        "instance_id=seed_b prompt_name=forward_modeling_prompt prompt_chars=10000 completion_chars=0 "
        "prompt_tokens=n/a completion_tokens=n/a total_tokens=n/a error=Request timed out\n"
        "2026-06-08 12:00:00,001 INFO [or_cpt_engine.model_call] model_api_call "
        "request_id=req_ok_a attempt=1 model=deepseek-v4-flash endpoint_name=dsv4_a "
        "endpoint_base_url=http://127.0.0.1:8005/v1 latency_seconds=20.000 status=ok http_status=200 "
        "instance_id=seed_a prompt_name=backtranslation_compact_prompt prompt_chars=1000 completion_chars=100 "
        "prompt_tokens=250 completion_tokens=25 total_tokens=275\n"
        "2026-06-08 12:00:00,002 INFO [or_cpt_engine.model_call] model_api_call "
        "request_id=req_ok_b attempt=1 model=deepseek-v4-flash endpoint_name=dsv4_b "
        "endpoint_base_url=http://127.0.0.2:8005/v1 latency_seconds=10.000 status=ok http_status=200 "
        "instance_id=seed_a prompt_name=backtranslation_compact_prompt prompt_chars=1000 completion_chars=100 "
        "prompt_tokens=250 completion_tokens=25 total_tokens=275\n"
        "2026-06-08 12:00:00,001 WARNING [or_cpt_engine.llm.openai_compatible_client] "
        "or_cpt_model_endpoint_health endpoint_name=dsv4_a "
        "endpoint_base_url=http://127.0.0.1:8005/v1 status=quarantined cooldown_seconds=60 reason=Request timed out\n",
        encoding="utf-8",
    )
    write_jsonl(
        logs_dir / "fixture_model_failures.jsonl",
        [
            {
                "request_id": "req_timeout",
                "attempt": 1,
                "instance_id": "seed_b",
                "prompt_name": "forward_modeling_prompt",
                "endpoint_name": "dsv4_a",
                "http_status": 502,
                "error_message": "duplicate structured row for the model_api_call log event",
            },
            {
                "request_id": "req_json",
                "attempt": 1,
                "instance_id": "seed_a",
                "prompt_name": "backtranslation_compact_prompt",
                "endpoint_name": "dsv4_b",
                "http_status": 502,
                "error_message": "bad gateway",
            }
        ],
    )
    write_jsonl(
        run_dir / "09_cpt_rendering" / "cpt_documents.jsonl",
        [
            {
                "doc_id": "doc_a",
                "pair_id": "pair_a",
                "instance_id": "seed_a",
                "generator_id": "gen_a",
                "doc_type": "final_model_report",
                "text": valid_text,
                "token_count": 120,
                "metadata": {"task_family": "transportation", "sub_family": "network_flow", "difficulty_level": "level_1"},
            }
        ],
    )
    write_jsonl(
        run_dir / "10_train_val_export" / "train.jsonl",
        [
            {
                "id": "doc_a",
                "source_id": "seed_a",
                "text": valid_text,
                "metadata": {
                    "generator_id": "optmath_carselection",
                    "doc_type": "final_model_report",
                    "token_count": 120,
                    "task_family": "transportation",
                    "sub_family": "network_flow",
                    "difficulty_level": "level_1",
                    "scenario_id": "scenario_a",
                    "business_trigger": "weekly planning cycle",
                },
            },
            {
                "id": "doc_b",
                "source_id": "seed_b",
                "text": "\n\n".join(
                    [
                        "# Optimization Modeling Report",
                        "## Problem\nA steel mill chooses product tons. There is no inventory, backlog, setup, sequencing, or time-period carryover.",
                        "## Model Formulation\nA product-mix LP uses processing hours per ton and stage capacity.",
                        "## Implementation\n```python\nimport gurobipy as gp\nmodel = gp.Model()\nmodel.optimize()\n```",
                        "## Optimal Solution\nThe optimal objective value reported by the solver is `10`.",
                        "## Feasibility Check\nThe solution is feasible.",
                        "## Validation\nThe previous model failed because validation showed an objective mismatch.",
                    ]
                ),
                "metadata": {
                    "generator_id": "optmath_steel4",
                    "doc_type": "final_model_report",
                    "token_count": 130,
                    "task_family": "production_planning",
                    "sub_family": "steel_product_mix",
                    "difficulty_level": "level_1",
                    "scenario_id": "steel4_product_mix",
                    "business_trigger": "monthly product-mix review",
                },
            }
        ],
    )

    result = analyze_run_quality(run_dir)

    assert result["stage_counts"]["seed_count"] == 2
    assert result["stage_counts"]["train"] == 2
    assert result["rejection_rows"] == 3
    assert (run_dir / "reports" / "run_quality_dashboard.md").exists()
    summary = json.loads((run_dir / "reports" / "run_quality_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_retries"]["total_rows_with_retries"] == 2
    assert summary["quality_retries"]["by_stage"]["05_backtranslation"]["candidate_rows_with_retries"] == 1
    assert summary["quality_retries"]["by_stage"]["07_forward_modeling"]["rejected_rows_with_retries"] == 1
    assert summary["repair_attempts"]["total_attempts"] == 2
    assert summary["repair_attempts"]["recovered"] == 1
    assert summary["repair_attempts"]["failed_eval"] == 1
    assert summary["repair_attempts"]["top_generators"][0]["generator_id"] in {"gen_a", "gen_b"}
    assert summary["model_failures"]["total_failures"] == 2
    assert summary["model_failures"]["by_prompt"]["forward_modeling_prompt"] == 1
    assert summary["model_failures"]["by_prompt"]["backtranslation_compact_prompt"] == 1
    assert summary["model_failures"]["by_error_type"]["TIMEOUT"] == 1
    assert summary["model_failures"]["by_error_type"]["HTTP_5XX"] == 1
    assert summary["model_failures"]["top_failures"][0]["generator_id"] in {"gen_a", "gen_b"}
    assert summary["throughput"]["endpoint_health_events"] == 1
    assert summary["throughput"]["endpoint_failure_count"] == 1
    assert summary["forward_static_replay"]["total_forward_rejected_rows"] == 1
    assert summary["forward_static_replay"]["rows_with_current_static_issues"] == 1
    assert summary["forward_static_replay"]["top_current_static_issues"][0]["generator_id"] == "optmath_marketshare"
    assert summary["data_quality"]["forbidden_phrase_count"] >= 3
    assert summary["data_quality"]["forbidden_phrase_hits"]["previous model"] == 1
    assert summary["data_quality"]["forbidden_phrase_hits"]["failed because"] == 1
    assert summary["data_quality"]["forbidden_phrase_hits"]["objective mismatch"] == 1
    endpoint_rows = {row["endpoint_name"]: row for row in summary["endpoint_reliability"]["rows"]}
    assert endpoint_rows["dsv4_a"]["api_attempts"] == 2
    assert endpoint_rows["dsv4_a"]["failed_api_attempts"] == 1
    assert endpoint_rows["dsv4_a"]["failure_rate"] == 0.5
    assert endpoint_rows["dsv4_b"]["api_attempts"] == 1
    assert endpoint_rows["dsv4_b"]["failed_api_attempts"] == 0
    assert summary["data_quality"]["semantic_coherence_row_count"] == 1
    assert summary["data_quality"]["semantic_coherence_issue_count"] >= 1
    assert "Quality Retries" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Forward Repair Attempts" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Endpoint Reliability" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Forward Static Replay" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Model Failures" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Top Forbidden Phrase Hits" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "previous model" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "Semantic Coherence Warnings" in (run_dir / "reports" / "run_quality_dashboard.md").read_text(encoding="utf-8")
    assert "quality_retries" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "forward_repair" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "endpoint_reliability" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "forward_static_replay" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "model_failures" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "forbidden_phrase|previous model" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "semantic_coherence_issue_count" in (run_dir / "reports" / "run_quality_dashboard.csv").read_text(encoding="utf-8")
    assert "MISSING_BUSINESS_CONTEXT" in (run_dir / "reports" / "rejection_reason_by_generator.csv").read_text(encoding="utf-8")
    assert "gen_a" in (run_dir / "reports" / "generator_stage_funnel.csv").read_text(encoding="utf-8")
    assert "transportation" in (run_dir / "reports" / "train_distribution.md").read_text(encoding="utf-8")


def test_forward_modeling_guardrails_and_static_signature_check() -> None:
    row = {
        "generator_id": "optmath_facility_location",
        "problem_statement": "The objective is to minimize total fixed opening and service cost.",
        "concept_tags": ["fixed_cost", "linking_constraint"],
        "canonical_math_signature": {
            "required_tables": ["fixed_cost_vector", "demand_vector"],
            "core_constraints": ["demand_satisfaction", "capacity_limit", "open_before_serve_linking"],
            "forbidden_changes": ["do not remove fixed opening costs"],
        },
    }
    bad_payload = {
        "modeling_explanation": "Use assignment quantities to serve demand.",
        "math_model": "Minimize shipping cost subject to demand constraints.",
    }
    bad_code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVar(lb=0)",
            "model.setObjective(x, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )
    good_payload = {
        "modeling_explanation": "Use binary open variables, fixed opening costs, demand satisfaction, capacity, and linking constraints.",
        "math_model": "Minimize fixed opening cost plus service cost with open-before-serve linking and capacity limits.",
    }
    good_code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "open_0 = model.addVar(vtype=GRB.BINARY, name='open_0')",
            "assign_0 = model.addVar(lb=0, name='assign_0')",
            "fixed_cost = 10",
            "demand = 5",
            "capacity = 7",
            "model.addConstr(assign_0 == demand)",
            "model.addConstr(assign_0 <= capacity * open_0)",
            "model.setObjective(fixed_cost * open_0 + assign_0, GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    guardrails = _modeling_guardrails(row)
    bad_issues = _static_signature_issues(row, bad_payload, bad_code)
    good_issues = _static_signature_issues(row, good_payload, good_code)

    assert any("fixed opening" in guardrail.lower() for guardrail in guardrails)
    assert any(issue.startswith("OBJECTIVE_SENSE_MISMATCH") for issue in bad_issues)
    assert "MISSING_FIXED_COST_SIGNAL" in bad_issues
    assert good_issues == []


def test_forward_modeling_guardrails_cover_high_risk_generators() -> None:
    cases = [
        (
            {
                "generator_id": "optmath_uncapacitatedlotsizing",
                "sub_family": "uncapacitated_lot_sizing",
                "concept_tags": ["inventory_balance"],
            },
            ["without backlog", "do not add backloggedamount", "production capacity"],
        ),
        (
            {
                "generator_id": "optmath_uncapacitatedlotsizingbacklogging",
                "sub_family": "uncapacitated_lot_sizing_with_backlogging",
                "concept_tags": ["inventory_balance", "backlog_state"],
            },
            ["carried backlog state", "backloggedamount", "not one-period lost demand"],
        ),
        (
            {
                "generator_id": "optmath_clsp_expand_capacity",
                "sub_family": "capacitated_lot_sizing",
                "concept_tags": ["inventory_balance", "period_capacity"],
            },
            ["capacitated lot sizing without backlog", "capacity-expansion", "zero-final-inventory"],
        ),
        (
            {
                "generator_id": "optmath_singlelevelsmallbucket",
                "sub_family": "single_level_small_bucket_lot_sizing_with_backlog",
                "concept_tags": ["dynamic_lot_sizing"],
            },
            ["per-unit production cost", "startup transition", "one item per machine-period"],
        ),
        (
            {
                "generator_id": "optmath_structure_based_assignment",
                "sub_family": "nmr_peak_amino_acid_structure_assignment",
                "concept_tags": ["binary_peak_acid_assignment"],
            },
            ["nmr peak-to-amino-acid", "not as staff/job/equipment/vehicle", "noe compatibility"],
        ),
    ]

    for row, expected_phrases in cases:
        guardrails = "\n".join(_modeling_guardrails(row)).lower()
        for phrase in expected_phrases:
            assert phrase in guardrails


def test_forward_prompt_payload_includes_generator_specific_guardrails() -> None:
    row = {
        "bt_id": "bt_guardrail",
        "instance_id": "inst_guardrail",
        "generator_id": "optmath_singlelevelsmallbucket",
        "problem_statement": "Minimize setup, startup, holding, and backlog costs.",
        "sub_family": "single_level_small_bucket_lot_sizing_with_backlog",
        "concept_tags": ["dynamic_lot_sizing"],
        "source_compact_data": {"available": True, "value": {"compact_lotsizing_tables": {}}},
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)

    assert '"modeling_guardrails"' in prompt
    assert "Do not add a per-unit production cost" in prompt
    assert "Preserve machine-specific capacity" in prompt


def test_forward_modeling_static_signature_uses_family_contract() -> None:
    row = {
        "generator_id": "optmath_knapsack",
        "problem_statement": "The goal is to maximize total value.",
        "concept_tags": [],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Choose fractional quantities subject to a resource limit.",
        "math_model": "Maximize value subject to total weight.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVar(lb=0, name='x')",
            "model.addConstr(x <= 1)",
            "model.setObjective(x, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )
    family_contract = {
        "answer_contract": {
            "objective_sense_options": ["maximize"],
            "required_variable_families": ["select_item_binary"],
            "required_constraints": ["capacity_limit"],
        }
    }

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL" in issues


def test_netasgn_static_signature_allows_continuous_assignment_hours() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_netasgn"}, contracts)
    row = {
        "generator_id": "optmath_netasgn",
        "problem_statement": "Assign continuous work hours from people to projects at minimum cost.",
        "concept_tags": ["continuous_assignment", "network_flow"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use continuous assignment hours. Person supply hours are met exactly, project demand hours are met exactly, "
            "and each person-project contribution is capped by an upper bound."
        ),
        "math_model": (
            "Minimize cost per hour times assigned hours subject to person supply equality, project demand equality, "
            "and contribution upper bounds."
        ),
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "assigned_hours = model.addVars(people, projects, lb=0.0, vtype=GRB.CONTINUOUS, name='AssignedHours')",
            "for p in people:",
            "    model.addConstr(gp.quicksum(assigned_hours[p, j] for j in projects) == supply_hours[p], name='person_supply_hours_equality')",
            "for j in projects:",
            "    model.addConstr(gp.quicksum(assigned_hours[p, j] for p in people) == demand_hours[j], name='project_demand_hours_equality')",
            "for p in people:",
            "    for j in projects:",
            "        model.addConstr(assigned_hours[p, j] <= contribution_upper_bound[p, j], name='person_project_contribution_upper_bound')",
            "model.setObjective(gp.quicksum(cost_per_hour[p, j] * assigned_hours[p, j] for p in people for j in projects), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL" not in issues


def test_structure_assignment_static_signature_accepts_nmr_peak_acid_model() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_structure_based_assignment"}, contracts)
    row = {
        "generator_id": "optmath_structure_based_assignment",
        "problem_statement": "Assign NMR peaks to amino acids with NOE compatibility constraints.",
        "sub_family": "nmr_peak_amino_acid_structure_assignment",
        "concept_tags": ["binary_peak_acid_assignment"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use binary peak-amino-acid assignment variables. Each amino acid receives at most one peak, "
            "each peak is assigned to at most one amino acid, exactly N assignments are selected, and "
            "NOE-related peak pairs must use compatible amino-acid pairs."
        ),
        "math_model": (
            "Minimize assignment cost over binary x[p,a] subject to amino acid at-most-one, peak at-most-one, "
            "exact total assignment count, and NOE compatibility pair constraints."
        ),
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "peaks = [0, 1]",
            "amino_acids = [0, 1]",
            "assignment_cost = {(0,0): 0.1, (0,1): 0.2, (1,0): 0.3, (1,1): 0.4}",
            "compatibility = {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 1}",
            "noe_pairs = [(0, 1)]",
            "required_assignment_count = 1",
            "x = model.addVars(peaks, amino_acids, vtype=GRB.BINARY, name='peak_amino_acid_assignment')",
            "for a in amino_acids:",
            "    model.addConstr(gp.quicksum(x[p, a] for p in peaks) <= 1, name=f'each_amino_acid_at_most_one_peak_{a}')",
            "for p in peaks:",
            "    model.addConstr(gp.quicksum(x[p, a] for a in amino_acids) <= 1, name=f'each_peak_at_most_one_amino_acid_{p}')",
            "model.addConstr(gp.quicksum(x[p, a] for p in peaks for a in amino_acids) == required_assignment_count, name='exact_total_assignment_count')",
            "for p, q in noe_pairs:",
            "    for a in amino_acids:",
            "        for b in amino_acids:",
            "            model.addConstr(x[p, a] + x[q, b] <= compatibility[a, b] + 1, name=f'noe_compatibility_pair_constraints_{p}_{q}_{a}_{b}')",
            "model.setObjective(gp.quicksum(assignment_cost[p, a] * x[p, a] for p in peaks for a in amino_acids), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert issues == []


def test_structure_assignment_static_signature_rejects_staff_job_misframing() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_structure_based_assignment"}, contracts)
    row = {
        "generator_id": "optmath_structure_based_assignment",
        "problem_statement": "Assign NMR peaks to amino acids with NOE compatibility constraints.",
        "sub_family": "nmr_peak_amino_acid_structure_assignment",
        "concept_tags": ["binary_peak_acid_assignment"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "This is a staff-to-service-job assignment model. Each technician handles at most one job, "
            "each job gets at most one technician, and a dispatch manager chooses exactly N jobs."
        ),
        "math_model": "Minimize technician service cost subject to job matching compatibility constraints.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "technicians = [0, 1]",
            "service_jobs = [0, 1]",
            "cost = {(0,0): 1, (0,1): 2, (1,0): 3, (1,1): 4}",
            "x = model.addVars(technicians, service_jobs, vtype=GRB.BINARY, name='technician_job_assignment')",
            "for t in technicians:",
            "    model.addConstr(gp.quicksum(x[t, j] for j in service_jobs) <= 1, name='worker_capacity')",
            "for j in service_jobs:",
            "    model.addConstr(gp.quicksum(x[t, j] for t in technicians) <= 1, name='service_job_exclusivity')",
            "model.addConstr(gp.quicksum(x[t, j] for t in technicians for j in service_jobs) == 1, name='exact_dispatch_count')",
            "model.setObjective(gp.quicksum(cost[t, j] * x[t, j] for t in technicians for j in service_jobs), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_FORBIDDEN_CHANGE:nmr_domain_misframed_as_staffing" in issues


def test_forward_prompt_includes_authoritative_source_fact_digest_for_steel4() -> None:
    row = {
        "bt_id": "bt_steel4",
        "instance_id": "inst_optmath_steel4_digest",
        "generator_id": "optmath_steel4",
        "problem_statement": "Build a profitable product mix. Some wording is intentionally vague.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_steel_product_mix_tables": {
                    "sets": {"products": ["product_0", "product_1"], "production_stages": ["stage_0"]},
                    "product_table": {
                        "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
                        "rows": [["product_0", 10, 1, 4], ["product_1", 20, 2, 5]],
                    },
                    "stage_capacity_table": {
                        "columns": ["stage", "available_hours"],
                        "rows": [["stage_0", 3.5]],
                    },
                    "processing_hours_per_ton_matrix": {
                        "columns": ["stage_0"],
                        "rows": [
                            {"product": "product_0", "values": [0.5]},
                            {"product": "product_1", "values": [0.75]},
                        ],
                    },
                }
            },
        },
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])

    assert "AUTHORITATIVE_SOURCE_FACTS" in digest
    assert "products product,profit,min_commit,max_market: product_0,10,1,4" in digest
    assert "stage_available_hours stage,available: stage_0,3.5" in digest
    assert "processing product_1: 0.75" in digest
    assert "do not use product market bounds as stage capacities" in digest
    assert payload["source_compact_data"]["truncated_for_forward_prompt"] is True
    assert payload["source_compact_data"]["value_keys"] == ["compact_steel_product_mix_tables"]
    assert "compact_steel_product_mix_tables" not in payload["source_compact_data"]


def test_forward_prompt_includes_authoritative_source_fact_digest_for_netasgn() -> None:
    row = {
        "bt_id": "bt_netasgn",
        "instance_id": "inst_optmath_netasgn_digest",
        "generator_id": "optmath_netasgn",
        "problem_statement": "Assign staff hours. Some wording is intentionally vague.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_project_assignment_tables": {
                    "model_family": "continuous_resource_assignment_hours",
                    "sets": {
                        "people": ["person_0", "person_1"],
                        "projects": ["project_0", "project_1"],
                    },
                    "person_supply_table": [
                        {"person": "person_0", "available_hours": 8},
                        {"person": "person_1", "available_hours": 6},
                    ],
                    "project_demand_table": [
                        {"project": "project_0", "required_hours": 5},
                        {"project": "project_1", "required_hours": 9},
                    ],
                    "cost_per_hour_matrix": {
                        "columns": ["project_0", "project_1"],
                        "rows": [
                            {"person": "person_0", "values": [32, 11]},
                            {"person": "person_1", "values": [17, 23]},
                        ],
                    },
                    "max_contribution_hours_matrix": {
                        "columns": ["project_0", "project_1"],
                        "rows": [
                            {"person": "person_0", "values": [5, 8]},
                            {"person": "person_1", "values": [6, 6]},
                        ],
                    },
                    "total_supply_hours": 14,
                    "total_demand_hours": 14,
                }
            },
        },
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])

    assert "netasgn_project_assignment: continuous Assign[person,project] hours" in digest
    assert "netasgn_totals: total_supply_hours=14, total_demand_hours=14" in digest
    assert "person_0=8" in digest
    assert "project_1=9" in digest
    assert "netasgn_cost person=person_0: 32, 11" in digest
    assert "netasgn_limit person=person_1: 6, 6" in digest


def test_forward_prompt_includes_authoritative_source_fact_digest_for_marketshare() -> None:
    row = {
        "bt_id": "bt_marketshare",
        "instance_id": "inst_optmath_marketshare_digest",
        "generator_id": "optmath_marketshare",
        "problem_statement": "Allocate regional product commitments without inventing budgets.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_marketshare_tables": {
                    "sets": {"companies": [0, 1], "markets": [0], "products": [0, 1]},
                    "demand_table": [
                        {"market": 0, "product": 0, "demand": 12},
                        {"market": 0, "product": 1, "demand": 8},
                    ],
                    "profit_table": [
                        {"company": 0, "market": 0, "product": 0, "unit_cost": 2, "unit_revenue": 7, "unit_profit": 5},
                        {"company": 1, "market": 0, "product": 0, "unit_cost": 3, "unit_revenue": 9, "unit_profit": 6},
                    ],
                }
            },
        },
        "concept_tags": ["integer_supply_allocation"],
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])
    guardrails = "\n".join(payload["modeling_guardrails"])

    assert "AUTHORITATIVE_SOURCE_FACTS" in digest
    assert "marketshare: nonnegative integer Supply[i,j,k]" in digest
    assert "vtype=GRB.INTEGER" in digest
    assert "sum_i Supply[i,j,k] == demand[j,k]" in digest
    assert "market=0,product=0,demand=12" in digest
    assert "unit_profit=6" in digest
    assert "no resource capacity, budget" in digest
    assert "not revenue-management capacity allocation" in guardrails
    assert "vtype=GRB.INTEGER" in guardrails
    assert "do not use continuous supply variables" in guardrails
    assert payload["source_compact_data"]["truncated_for_forward_prompt"] is True
    assert payload["source_compact_data"]["value_keys"] == ["compact_marketshare_tables"]
    assert "compact_marketshare_tables" not in payload["source_compact_data"]


def test_forward_prompt_includes_no_backlog_uls_source_fact_skeleton() -> None:
    row = {
        "bt_id": "bt_uls",
        "instance_id": "inst_optmath_uls_digest",
        "generator_id": "optmath_uncapacitatedlotsizing",
        "problem_statement": "Plan orders over periods; do not add backlog.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_lotsizing_tables": {
                    "model_family": "uncapacitated_lot_sizing_without_backlog",
                    "initial_inventory": 0,
                    "required_final_inventory": 0,
                    "total_demand_big_m": 24,
                    "period_cost_table": [
                        {
                            "period": "period_1",
                            "demand": 10,
                            "fixed_ordering_cost": 25,
                            "unit_order_cost": 1,
                            "unit_holding_cost": 3,
                        },
                        {
                            "period": "period_2",
                            "demand": 14,
                            "fixed_ordering_cost": 15,
                            "unit_order_cost": 4,
                            "unit_holding_cost": 2,
                        },
                    ],
                }
            },
        },
        "concept_tags": ["inventory_balance_without_backlog"],
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])
    guardrails = "\n".join(payload["modeling_guardrails"])

    assert "uls: uncapacitated lot sizing without backlog" in digest
    assert "use ONLY OrderedAmount[t], EndingInventory[t], and binary OrderIsPlaced[t]" in digest
    assert "uls_forbidden_variables" in digest
    assert "BackloggedAmount[t]" in digest
    assert "zero_final_backlog" in digest
    assert "uls_no_backlog_formula" in digest
    assert "EndingInventory[prev] + OrderedAmount[t] == demand[t] + EndingInventory[t]" in digest
    assert "period=period_2,demand=14,fixed=15,unit_order=4,holding=2" in digest
    assert "Do not add BackloggedAmount" in guardrails
    assert payload["source_compact_data"]["truncated_for_forward_prompt"] is True


def test_forward_prompt_clsp_distinguishes_period_and_cumulative_demand() -> None:
    row = {
        "bt_id": "bt_clsp",
        "instance_id": "inst_optmath_clsp_digest",
        "generator_id": "optmath_clsp_expand_capacity",
        "problem_statement": "Plan production with fixed period capacities and no backlog.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_lotsizing_tables": {
                    "model_family": "capacitated_lot_sizing_without_backlog",
                    "products": [0],
                    "periods": [0, 1],
                    "period_demand_table": [
                        {
                            "product": 0,
                            "period": 0,
                            "period_demand": 64,
                            "cumulative_demand_through_period": 64,
                            "setup_cost": 390,
                            "unit_production_cost": 47,
                            "unit_holding_cost": 4,
                            "setup_big_m_remaining_demand": 116,
                        },
                        {
                            "product": 0,
                            "period": 1,
                            "period_demand": 52,
                            "cumulative_demand_through_period": 116,
                            "setup_cost": 390,
                            "unit_production_cost": 47,
                            "unit_holding_cost": 4,
                            "setup_big_m_remaining_demand": 52,
                        },
                    ],
                    "period_capacity_table": [{"period": 0, "capacity": 800}, {"period": 1, "capacity": 800}],
                    "capacity_consumption_per_product": [{"product": 0, "capacity_consumption_per_unit": 1.72}],
                }
            },
        },
        "concept_tags": ["capacitated_lot_sizing"],
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])

    assert "clsp_demand_rule" in digest
    assert "period_demand is the only demand value for recursive inventory balance" in digest
    assert "clsp_balance_options" in digest
    assert "recursive form EndingInventory[p,prev] + Production[p,t] == period_demand[p,t] + EndingInventory[p,t]" in digest
    assert "cumulative form EndingInventory[p,t] == sum_{tau<=t} Production[p,tau] - cumulative_demand_through_period[p,t]" in digest
    assert "never put cumulative_demand_through_period as the RHS demand in a one-period recursive balance" in digest
    assert "p=0,t=1,period_demand=52,cumulative_demand_through_period=116" in digest


def test_forward_prompt_includes_authoritative_source_fact_digest_for_aircraft_landing() -> None:
    row = {
        "bt_id": "bt_aircraft_landing",
        "instance_id": "inst_optmath_aircraftlanding_digest",
        "generator_id": "optmath_aircraftlanding",
        "problem_statement": "Schedule arrivals without turning the task into route allocation.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_aircraft_landing_tables": {
                    "aircraft_time_penalty_table": [
                        {
                            "aircraft": "aircraft_0",
                            "earliest_landing": 43,
                            "target_landing": 63,
                            "latest_landing": 83,
                            "early_penalty_per_minute": 93,
                            "late_penalty_per_minute": 94,
                        },
                        {
                            "aircraft": "aircraft_1",
                            "earliest_landing": 45,
                            "target_landing": 65,
                            "latest_landing": 85,
                            "early_penalty_per_minute": 28,
                            "late_penalty_per_minute": 34,
                        },
                    ],
                    "ordered_pair_separation_matrix": {
                        "columns": ["aircraft_0", "aircraft_1"],
                        "rows": [
                            {"aircraft_before": "aircraft_0", "values": [None, 4]},
                            {"aircraft_before": "aircraft_1", "values": [2, None]},
                        ],
                    },
                }
            },
        },
    }

    prompt = render_forward_prompt("{{PROBLEM_JSON}}", row)
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])

    assert "AUTHORITATIVE_SOURCE_FACTS" in digest
    assert "aircraft_landing: static runway landing-time sequencing" in digest
    assert "aircraft=aircraft_0,earliest=43,target=63,latest=83,early_penalty=93,late_penalty=94" in digest
    assert "aircraft_landing_separation_columns: aircraft_0, aircraft_1" in digest
    assert "aircraft_landing_separation before=aircraft_1: 2, -" in digest
    assert "do not convert to aircraft type assignment" in digest
    assert payload["source_compact_data"]["truncated_for_forward_prompt"] is True
    assert payload["source_compact_data"]["value_keys"] == ["compact_aircraft_landing_tables"]
    assert "compact_aircraft_landing_tables" not in payload["source_compact_data"]


def test_forward_repair_prompt_includes_authoritative_source_fact_digest_for_steel4() -> None:
    row = {
        "bt_id": "bt_steel4",
        "fm_id": "fm_failed_steel4",
        "instance_id": "inst_optmath_steel4_repair_digest",
        "generator_id": "optmath_steel4",
        "problem_statement": "Repair a steel product mix model with mismatched objective.",
        "source_compact_data": {
            "available": True,
            "truncated": False,
            "value": {
                "compact_steel_product_mix_tables": {
                    "product_table": {
                        "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
                        "rows": [["product_0", 10, 1, 4]],
                    },
                    "stage_capacity_table": {
                        "columns": ["stage", "available_hours"],
                        "rows": [["stage_0", 3.5]],
                    },
                    "processing_hours_per_ton_matrix": {
                        "columns": ["stage_0"],
                        "rows": [{"product": "product_0", "values": [0.5]}],
                    },
                }
            },
        },
        "generated_answer": {
            "modeling_explanation": "Previous model confused market bounds with stage capacity.",
            "math_model": "bad model",
            "gurobipy_code": "bad code",
        },
        "reference_answer": {"objective_value": 42.0, "status": "OPTIMAL"},
        "forward_eval": {
            "solver_result": {"status": "OPTIMAL", "objective_value": 37.5},
            "correctness": {"is_correct": False, "abs_error": 4.5, "rel_error": 0.107142857},
        },
        "canonical_math_signature": {"objective_sense": "maximize"},
        "rejection_reason": "OBJECTIVE_MISMATCH",
        "fine_rejection_reason": "objective_mismatch_general",
        "failure_family": "objective_mismatch",
        "repairable": True,
    }

    prompt = _render_repair_prompt(
        "{{REPAIR_JSON}}",
        row,
        attempt=1,
        family_contract={"answer_contract": {"objective_sense_options": ["maximize"]}},
    )
    payload = json.loads(prompt)
    digest = "\n".join(payload["source_fact_digest"])

    assert "AUTHORITATIVE_SOURCE_FACTS" in digest
    assert "stage_available_hours stage,available: stage_0,3.5" in digest
    assert "do not use product market bounds as stage capacities" in digest
    assert payload["source_compact_data"]["truncated_for_repair_prompt"] is True
    assert payload["source_compact_data"]["value_keys"] == ["compact_steel_product_mix_tables"]
    assert "compact_steel_product_mix_tables" not in payload["source_compact_data"]
    diagnostics = payload["validation_failure"]["objective_diagnostics"]
    assert diagnostics["reference_objective_value"] == 42.0
    assert diagnostics["generated_objective_value"] == 37.5
    assert diagnostics["abs_error"] == 4.5
    assert diagnostics["objective_sense"] == "maximize"
    assert diagnostics["family_objective_sense_options"] == ["maximize"]


def test_marketshare_static_signature_accepts_exact_integer_supply_allocation() -> None:
    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_marketshare"}, contracts)
    row = {
        "generator_id": "optmath_marketshare",
        "problem_statement": "Assign integer supply quantities from companies to market-product demand commitments.",
        "concept_tags": ["integer_supply_allocation"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use nonnegative integer supply[i,j,k]. For each market-product pair, total supply over companies "
            "equals the listed demand exactly. Maximize total net profit."
        ),
        "math_model": "maximize sum unit_profit[i,j,k] supply[i,j,k] subject to market-product demand equalities.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "companies = [0, 1]",
            "markets = [0]",
            "products = [0, 1]",
            "unit_profit = {(0,0,0): 5, (1,0,0): 6, (0,0,1): 4, (1,0,1): 3}",
            "demand = {(0,0): 12, (0,1): 8}",
            "supply = model.addVars(companies, markets, products, lb=0, vtype=GRB.INTEGER, name='supply')",
            "model.setObjective(gp.quicksum(unit_profit[i,j,k] * supply[i,j,k] for i in companies for j in markets for k in products), GRB.MAXIMIZE)",
            "for j in markets:",
            "    for k in products:",
            "        model.addConstr(gp.quicksum(supply[i,j,k] for i in companies) == demand[j,k], name=f'market_product_demand_{j}_{k}')",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert issues == []


def test_marketshare_static_signature_rejects_continuous_supply_allocation() -> None:
    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_marketshare"}, contracts)
    row = {
        "generator_id": "optmath_marketshare",
        "problem_statement": "Assign integer supply quantities from companies to market-product demand commitments.",
        "concept_tags": ["integer_supply_allocation"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use continuous supply quantities to meet market-product demand.",
        "math_model": "maximize net profit subject to market-product demand equality.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "companies = [0, 1]",
            "markets = [0]",
            "products = [0]",
            "supply = model.addVars(companies, markets, products, lb=0, vtype=GRB.CONTINUOUS, name='supply')",
            "model.addConstr(gp.quicksum(supply[i,0,0] for i in companies) == 5)",
            "model.setObjective(gp.quicksum(supply[i,0,0] for i in companies), GRB.MAXIMIZE)",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:nonnegative_integer_supply" in issues


def test_net1_static_signature_accepts_latency_cost_as_arc_shipping_cost() -> None:
    row = {
        "generator_id": "optmath_net1",
        "problem_statement": "The objective is to minimize total network latency cost.",
        "concept_tags": ["network_flow"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use continuous nonnegative flow on each directed arc. The objective minimizes total latency cost, "
            "computed as per-unit cost on each arc times the flow on that arc."
        ),
        "math_model": "min sum_{(i,j)} c_{ij} x_{ij} subject to node flow balance and arc capacity.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "arcs = [('a', 'b')]",
            "cost = {('a', 'b'): 3}",
            "flow = model.addVars(arcs, lb=0, name='flow')",
            "model.setObjective(gp.quicksum(cost[a] * flow[a] for a in arcs), GRB.MINIMIZE)",
            "model.addConstr(flow['a', 'b'] <= 10)",
            "model.optimize()",
        ]
    )
    family_contract = {
        "answer_contract": {
            "objective_sense_options": ["minimize"],
            "required_variable_families": ["continuous_arc_flow"],
            "required_constraints": [],
            "required_objective_terms": ["arc_shipping_cost"],
        }
    }

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_OBJECTIVE_TERM:arc_shipping_cost" not in issues
    assert issues == []


def test_blending_static_signature_accepts_total_blend_sum_as_material_balance() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_blending_problem"}, contracts)
    row = {
        "generator_id": "optmath_blending_problem",
        "problem_statement": "Choose ingredient proportions to minimize total blend cost subject to quality specifications.",
        "concept_tags": ["quality_ratio"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use continuous nonnegative ingredient proportions. The sum of all ingredient proportions must equal 1, "
            "and each quality ratio must remain within its lower and upper specification bounds."
        ),
        "math_model": (
            "Minimize ingredient cost subject to total blend sum = 1 and quality ratio constraints."
        ),
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVars(ingredients, lb=0.0, name='x')",
            "model.addConstr(gp.quicksum(x[i] for i in ingredients) == 1, name='total_blend_sum')",
            "for q in qualities:",
            "    model.addConstr(gp.quicksum(pct[q][i] * x[i] for i in ingredients) >= min_q[q])",
            "    model.addConstr(gp.quicksum(pct[q][i] * x[i] for i in ingredients) <= max_q[q])",
            "model.setObjective(gp.quicksum(cost[i] * x[i] for i in ingredients), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:material_balance" not in issues


def test_ulsb_static_signature_accepts_final_period_zero_inventory_and_backlog() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_uncapacitatedlotsizingbacklogging"}, contracts)
    row = {
        "generator_id": "optmath_uncapacitatedlotsizingbacklogging",
        "problem_statement": "Minimize lot-sizing cost with ending inventory and backlog both zero in the final period.",
        "concept_tags": ["inventory_backlog_balance"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use order quantities, ending inventory, backlog, and binary setups. Net inventory balance carries over by period. "
            "In the final period, ending inventory and backlog must both be zero."
        ),
        "math_model": (
            "Order quantity is linked to setup by a big-M bound, net inventory balance is enforced, and final inventory/backlog are zero."
        ),
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVars(T, lb=0, name='order_quantity')",
            "I = model.addVars(T, lb=0, name='ending_inventory')",
            "B = model.addVars(T, lb=0, name='backlog')",
            "y = model.addVars(T, vtype=GRB.BINARY, name='order_setup')",
            "for t in T:",
            "    model.addConstr(I[t] - B[t] == prev_net[t] + x[t] - demand[t], name='net_inventory_balance')",
            "    model.addConstr(x[t] <= M[t] * y[t], name='setup_link')",
            "model.addConstr(I[T[-1]] == 0, name='final_ending_inventory_zero')",
            "model.addConstr(B[T[-1]] == 0, name='final_backlog_zero')",
            "model.setObjective(gp.quicksum(f[t]*y[t] + c[t]*x[t] + h[t]*I[t] + p[t]*B[t] for t in T), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:zero_final_inventory" not in issues
    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:zero_final_backlog" not in issues


def test_factory_contract_static_signature_rejects_missing_capacity_sales_and_final_inventory() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_factory_planning_problem"}, contracts)
    row = {
        "generator_id": "optmath_factory_planning_problem",
        "problem_statement": (
            "Maximize sales revenue minus inventory holding cost for products over periods with "
            "machine-hour capacity, product-period sales limits, inventory limits, and final inventory targets."
        ),
        "concept_tags": ["multi_period_production", "inventory_balance", "machine_hour_capacity"],
        "canonical_math_signature": {
            "core_constraints": [
                "initial_inventory_balance",
                "period_to_period_inventory_balance",
                "machine_hour_capacity_after_maintenance",
                "sales_upper_bound_by_period_product",
                "inventory_upper_bound_by_period_product",
                "required_final_inventory_target",
            ]
        },
    }
    payload = {
        "modeling_explanation": "Use production, sales, and inventory with period balance only.",
        "math_model": "Production plus previous inventory equals sales plus ending inventory.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "Production = model.addVar(lb=0, name='Production_0_0')",
            "Sales = model.addVar(lb=0, name='Sales_0_0')",
            "Inventory = model.addVar(lb=0, name='Inventory_0_0')",
            "model.addConstr(Production == Sales + Inventory, name='balance')",
            "model.setObjective(100 * Sales - 20 * Inventory, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:machine_hour_capacity_after_maintenance" in issues
    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:sales_upper_bound_by_period_product" in issues
    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:inventory_upper_bound_by_period_product" in issues
    assert "CONTRACT_MISSING_REQUIRED_CONSTRAINT:required_final_inventory_target" in issues


def test_p_dispersion_static_signature_does_not_require_fixed_cost_signal() -> None:
    row = {
        "generator_id": "optmath_the_p_dispersion_model",
        "problem_statement": "Select exactly p candidate sites to maximize the minimum pairwise distance.",
        "concept_tags": ["binary_node_selection", "pair_selection_linking", "max_min_dispersion"],
        "canonical_math_signature": {
            "objective_sense": "maximize",
            "core_constraints": ["select_exactly_p_nodes", "pair_selection_linking"],
            "forbidden_changes": ["do not add fixed opening costs"],
        },
    }
    payload = {
        "modeling_explanation": "Use binary node selection x_i, pair-selection z_ij, and a continuous MinDistance variable.",
        "math_model": "Maximize MinDistance subject to selecting exactly p nodes and linking z_ij to selected node pairs.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVar(vtype=GRB.BINARY, name='Nodes_0')",
            "z = model.addVar(vtype=GRB.BINARY, name='PairSelection_0_1')",
            "D = model.addVar(lb=0, name='MinDistance')",
            "p = 1",
            "distance = 10",
            "M = 101",
            "model.addConstr(x == p, name='SelectPFacilities')",
            "model.addConstr(z <= x, name='Z_leq_Xleft')",
            "model.addConstr(D <= distance + M * (1 - z), name='MinDistanceBound')",
            "model.setObjective(D, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )
    family_contract = {
        "answer_contract": {
            "objective_sense_options": ["maximize"],
            "required_variable_families": [
                "binary_node_selection",
                "binary_pair_selection",
                "continuous_minimum_distance",
            ],
            "required_constraints": [
                "select_exactly_p_nodes",
                "pair_selection_upper_bound_left_node",
                "min_distance_upper_bound_for_every_selected_pair",
            ],
            "forbidden_changes": ["do not add fixed opening costs"],
        }
    }

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "MISSING_FIXED_COST_SIGNAL" not in issues


def test_forward_modeling_contract_gate_skips_conditional_and_default_requirements() -> None:
    row = {
        "generator_id": "optmath_portfolio",
        "problem_statement": "Allocate a budget across assets to maximize expected return subject to a risk limit.",
        "concept_tags": ["risk_constraint"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use continuous allocation weights with budget and risk constraints.",
        "math_model": "Maximize return subject to budget, risk, and allocation bounds.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "x = model.addVar(lb=0, name='x')",
            "model.addConstr(x <= 1, name='budget')",
            "model.addConstr(x <= 0.5, name='risk')",
            "model.setObjective(x, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )
    portfolio_contract = {
        "contract_coverage": "high",
        "answer_contract": {
            "objective_sense_options": ["maximize", "minimize"],
            "required_variable_families": ["allocation_weight", "selected_asset_binary_if_cardinality_present"],
            "required_constraints": ["budget_or_total_weight", "risk_bound_if_present", "allocation_bounds"],
        },
    }
    default_contract = {
        "contract_coverage": "low",
        "contract_id": "default_linear_or_contract",
        "answer_contract": {
            "objective_sense_options": ["maximize", "minimize"],
            "required_constraints": ["all capacity, demand, assignment, balance, and bound constraints stated in the problem"],
        },
    }

    portfolio_issues = _static_signature_issues(row, payload, code, family_contract=portfolio_contract)
    default_issues = _static_signature_issues(row, payload, code, family_contract=default_contract)

    assert "CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL" not in portfolio_issues
    assert not any(issue.startswith("CONTRACT_MISSING_REQUIRED_CONSTRAINT") for issue in default_issues)


def test_forward_modeling_static_signature_checks_facility_objective_terms() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_facility_location"}, contracts)
    row = {
        "generator_id": "optmath_facility_location",
        "problem_statement": "The goal is to minimize shipping, fixed distribution-center, and unit throughput costs.",
        "sub_family": "multi_commodity_distribution_center_selection",
        "concept_tags": ["binary_distribution_center_selection", "binary_zone_assignment"],
        "canonical_math_signature": {
            "core_constraints": [
                "served_zone_commodity_demand_exactly_shipped",
                "explicit_distribution_center_open_before_zone_assignment",
            ],
        },
    }
    payload = {
        "modeling_explanation": (
            "Use Selected[d], Served[d,z], and Shipped[c,p,d,z] with exact served demand, "
            "open-before-assignment linking, and min/max throughput bounds."
        ),
        "math_model": "Minimize shipping cost plus fixed selected center cost.",
    }
    code_missing_unit = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "Selected = model.addVar(vtype=GRB.BINARY, name='Selected_dc0')",
            "Served = model.addVar(vtype=GRB.BINARY, name='Served_dc0_zone0')",
            "Shipped = model.addVar(lb=0, name='Shipped_c0_p0_dc0_zone0')",
            "model.addConstr(Shipped == 10 * Served)",
            "model.addConstr(Served <= Selected)",
            "model.addConstr(10 * Served >= 0, name='minimum throughput')",
            "model.addConstr(10 * Served <= 100, name='maximum throughput')",
            "model.setObjective(5 * Shipped + 100 * Selected, GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )
    code_with_unit = code_missing_unit.replace(
        "5 * Shipped + 100 * Selected",
        "5 * Shipped + 100 * Selected + 2 * 10 * Served",
    )
    payload_with_unit = {
        **payload,
        "math_model": "Minimize shipping cost plus fixed selected center cost plus unit throughput cost.",
    }

    missing_issues = _static_signature_issues(row, payload, code_missing_unit, family_contract=family_contract)
    good_issues = _static_signature_issues(row, payload_with_unit, code_with_unit, family_contract=family_contract)

    assert "CONTRACT_MISSING_REQUIRED_OBJECTIVE_TERM:unit_distribution_center_throughput_cost" in missing_issues
    assert "CONTRACT_MISSING_REQUIRED_OBJECTIVE_TERM:unit_distribution_center_throughput_cost" not in good_issues


def test_forward_modeling_static_signature_accepts_continuous_portfolio_qp() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_portfolio"}, contracts)
    row = {
        "generator_id": "optmath_portfolio",
        "problem_statement": "Minimize portfolio variance while meeting a minimum expected return.",
        "sub_family": "continuous_mean_variance_portfolio",
        "concept_tags": ["continuous_asset_weight", "covariance_matrix"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use continuous weights, a full-investment budget, a target return, and covariance variance.",
        "math_model": "Minimize covariance-based variance subject to sum weights = 1, expected return >= target return, and weight bounds.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "w0 = model.addVar(lb=0, ub=0.3, name='Weights_asset0')",
            "w1 = model.addVar(lb=0, ub=0.3, name='Weights_asset1')",
            "model.addConstr(w0 + w1 == 1, name='Budget')",
            "model.addConstr(0.1 * w0 + 0.12 * w1 >= 0.11, name='TargetReturn')",
            "model.setObjective(0.2 * w0 * w0 + 0.01 * w0 * w1 + 0.18 * w1 * w1, GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL" not in issues
    assert not any(issue.startswith("CONTRACT_MISSING_REQUIRED_CONSTRAINT") for issue in issues)
    assert not any(issue.startswith("CONTRACT_MISSING_REQUIRED_OBJECTIVE_TERM") for issue in issues)


def test_forward_modeling_static_signature_accepts_time_expanded_fleet_flow() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_fleet_routing"}, contracts)
    row = {
        "generator_id": "optmath_fleet_routing",
        "problem_statement": "Minimize operating cost for a time-expanded fleet-flow assignment.",
        "sub_family": "time_expanded_fleet_flow",
        "concept_tags": ["integer_fleet_assignment", "time_expanded_flow_conservation"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": (
            "Use integer NumPlanes on active legs, idle fleet variables, initial placement, "
            "fleet availability, active-leg passenger demand satisfaction, and inactive leg route restrictions."
        ),
        "math_model": "Time-expanded flow conservation balances idle fleet counts over periods.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "NumPlanes = model.addVar(vtype=GRB.INTEGER, lb=0, name='NumPlanes_plane0_location0_1_location1_2')",
            "NumIdlePlanes = model.addVar(vtype=GRB.INTEGER, lb=0, name='NumIdlePlanes_plane0_location0_1')",
            "NumIdlePlanesInit = model.addVar(vtype=GRB.INTEGER, lb=0, name='NumIdlePlanesInit_plane0_location0')",
            "available = 2",
            "capacity = 100",
            "leg_is_active = 1",
            "model.addConstr(NumIdlePlanesInit == NumIdlePlanes + NumPlanes, name='initial fleet balance')",
            "model.addConstr(NumIdlePlanes <= available, name='fleet availability')",
            "model.addConstr(capacity * NumPlanes >= 50, name='active leg passenger demand satisfaction')",
            "model.addConstr(NumPlanes <= available * leg_is_active, name='inactive leg route restriction')",
            "model.setObjective(7 * NumPlanes, GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_MISSING_BINARY_VARIABLE_SIGNAL" not in issues
    assert not any("visit" in issue.lower() for issue in issues)
    assert not any(issue.startswith("CONTRACT_MISSING_REQUIRED_CONSTRAINT") for issue in issues)


def test_forward_code_preparation_repairs_safe_gurobi_case_errors() -> None:
    code = "\n".join(
        [
            "import gurobipy as gp",
            "model = gp.Model('case_repair')",
            "x = model.addVar(lb=0, name='x')",
            "model.addConstr(x <= 1)",
            "model.setObjective(x, GRB.MAXIMIZE)",
            "model.optimize()",
            "if model.status == GRB.OPTIMAL:",
            "    print(model.objVal)",
        ]
    )

    prepared = prepare_forward_code(code)

    assert prepared.passed
    assert "ADD_GRB_IMPORT" in prepared.repairs_applied
    assert "NORMALIZE_MODEL_STATUS_CASE" in prepared.repairs_applied
    assert "NORMALIZE_MODEL_OBJVAL_CASE" in prepared.repairs_applied
    assert "from gurobipy import GRB" in prepared.code
    assert "model.Status" in prepared.code
    assert "model.ObjVal" in prepared.code


def test_forward_code_preparation_repairs_accidental_top_level_indent() -> None:
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model('indent_repair')",
            "x = model.addVar(lb=0, name='x')",
            " y = model.addVar(lb=0, name='y')",
            "model.addConstr(x + y <= 1)",
            "model.setObjective(x + y, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )

    prepared = prepare_forward_code(code)

    assert prepared.passed
    assert "STRIP_UNEXPECTED_TOP_LEVEL_INDENT" in prepared.repairs_applied
    assert "\n y = model.addVar" not in prepared.code
    assert "\ny = model.addVar" in prepared.code


def test_forward_code_preparation_keeps_valid_block_indentation() -> None:
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model('block_indent')",
            "x = model.addVars([0, 1], lb=0, name='x')",
            "for i in [0, 1]:",
            "    model.addConstr(x[i] <= 1)",
            "model.setObjective(gp.quicksum(x[i] for i in [0, 1]), GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )

    prepared = prepare_forward_code(code)

    assert prepared.passed
    assert "STRIP_UNEXPECTED_TOP_LEVEL_INDENT" not in prepared.repairs_applied
    assert "    model.addConstr" in prepared.code


def test_forward_code_preparation_rejects_external_files_and_placeholders() -> None:
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "# TODO replace with real file",
            "data = open('data.csv', 'r').read()",
            "model = gp.Model('bad_external')",
            "x = model.addVar(lb=0)",
            "model.optimize()",
        ]
    )

    prepared = prepare_forward_code(code)

    assert not prepared.passed
    assert "MISSING_SET_OBJECTIVE_CALL" in prepared.issues
    assert "PLACEHOLDER_TODO" in prepared.issues
    assert "PLACEHOLDER_FILE_PATH" in prepared.issues
    assert "EXTERNAL_FILE_OPEN_READ" in prepared.issues


def test_forward_code_preparation_rejects_perm_question_placeholder() -> None:
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model('bad_perm')",
            "x = model.addVar(vtype=GRB.BINARY, name='x')",
            "b = perm?",
            "model.setObjective(x, GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    prepared = prepare_forward_code(code)

    assert not prepared.passed
    assert "PLACEHOLDER_PERM_QUESTION" in prepared.issues


def test_static_signature_rejects_smallbucket_unit_production_cost_added() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_singlelevelsmallbucket"}, contracts)
    row = {
        "generator_id": "optmath_singlelevelsmallbucket",
        "problem_statement": "Small-bucket lot sizing with setup, startup, holding, and backlog costs only.",
        "concept_tags": ["dynamic_lot_sizing"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use production amounts, production state, startup, stock, and backlog.",
        "math_model": "Minimize setup, startup, holding, and backlog costs.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "Amount = model.addVars(items, machines, periods, lb=0, name='Amount')",
            "Production = model.addVars(items, machines, periods, vtype=GRB.BINARY, name='Production')",
            "Startup = model.addVars(items, machines, periods, vtype=GRB.BINARY, name='Startup')",
            "Stock = model.addVars(items, periods, lb=0, name='Stock')",
            "Backlog = model.addVars(items, periods, lb=0, name='Backlog')",
            "prod_cost = 12",
            "model.setObjective(gp.quicksum(prod_cost * Amount[i,m,t] for i in items for m in machines for t in periods), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert "CONTRACT_FORBIDDEN_CHANGE:unit_production_cost_added" in issues


def test_static_signature_distinguishes_clsp_inventory_from_forbidden_backlog() -> None:
    import yaml

    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_clsp_expand_capacity"}, contracts)
    row = {
        "generator_id": "optmath_clsp_expand_capacity",
        "problem_statement": "Capacitated lot sizing without backlog.",
        "concept_tags": ["inventory_balance"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use production, setup, inventory balance, and period capacity; no backlog.",
        "math_model": "Inventory balance and capacity constraints.",
    }
    code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "Production = model.addVars(periods, lb=0, name='Production')",
            "Setup = model.addVars(periods, vtype=GRB.BINARY, name='Setup')",
            "Inventory = model.addVars(periods, lb=0, name='Inventory')",
            "for t in periods:",
            "    model.addConstr(Inventory[t] == Production[t] - demand[t], name='cumulative_inventory_balance_without_backlog')",
            "    model.addConstr(Production[t] <= big_m[t] * Setup[t], name='production_quantity_linked_to_binary_setup_by_remaining_demand_big_m')",
            "    model.addConstr(Production[t] <= period_capacity[t], name='period_capacity_limit')",
            "model.setObjective(gp.quicksum(setup_cost[t] * Setup[t] + unit_cost[t] * Production[t] + holding_cost[t] * Inventory[t] for t in periods), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )
    bad_code = code + "\nBacklog = model.addVars(periods, lb=0, name='Backlog')\nmodel.addConstr(Inventory[periods[-1]] == 0, name='zero_final_inventory')"

    good_issues = _static_signature_issues(row, payload, code, family_contract=family_contract)
    bad_issues = _static_signature_issues(row, payload, bad_code, family_contract=family_contract)

    assert "CONTRACT_FORBIDDEN_CHANGE:backlog_added" not in good_issues
    assert "CONTRACT_FORBIDDEN_CHANGE:zero_final_inventory_added" not in good_issues
    assert "CONTRACT_FORBIDDEN_CHANGE:backlog_added" in bad_issues
    assert "CONTRACT_FORBIDDEN_CHANGE:zero_final_inventory_added" in bad_issues


def test_uls_static_signature_rejects_hidden_backlog_or_shortage_state() -> None:
    contracts = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_uncapacitatedlotsizing"}, contracts)
    row = {
        "generator_id": "optmath_uncapacitatedlotsizing",
        "problem_statement": "Uncapacitated lot sizing without backlog.",
        "concept_tags": ["inventory_balance_without_backlog"],
        "canonical_math_signature": {},
    }
    payload = {
        "modeling_explanation": "Use orders, ending inventory, and setup. Do not allow backlog.",
        "math_model": "Inventory recursion without backlog.",
    }
    good_code = "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model()",
            "periods = [1, 2]",
            "demand = {1: 10, 2: 14}",
            "fixed_cost = {1: 25, 2: 15}",
            "unit_cost = {1: 1, 2: 4}",
            "holding_cost = {1: 3, 2: 2}",
            "M = 24",
            "OrderedAmount = model.addVars(periods, lb=0, name='OrderedAmount')",
            "EndingInventory = model.addVars(periods, lb=0, name='EndingInventory')",
            "OrderIsPlaced = model.addVars(periods, vtype=GRB.BINARY, name='OrderIsPlaced')",
            "model.addConstr(OrderedAmount[1] == demand[1] + EndingInventory[1], name='inventory_balance_1')",
            "model.addConstr(EndingInventory[1] + OrderedAmount[2] == demand[2] + EndingInventory[2], name='inventory_balance_2')",
            "for t in periods:",
            "    model.addConstr(OrderedAmount[t] <= M * OrderIsPlaced[t], name=f'order_quantity_linked_to_binary_setup_{t}')",
            "model.addConstr(EndingInventory[2] == 0, name='zero_final_inventory')",
            "model.setObjective(gp.quicksum(fixed_cost[t] * OrderIsPlaced[t] + unit_cost[t] * OrderedAmount[t] + holding_cost[t] * EndingInventory[t] for t in periods), GRB.MINIMIZE)",
            "model.optimize()",
        ]
    )
    hidden_backlog_code = good_code + "\nB = model.addVars(periods, lb=0, name='BacklogState')"
    shortage_code = good_code + "\nShortage = model.addVars(periods, lb=0, name='period_shortage')"

    assert "CONTRACT_FORBIDDEN_CHANGE:backlog_added" not in _static_signature_issues(
        row, payload, good_code, family_contract=family_contract
    )
    assert "CONTRACT_FORBIDDEN_CHANGE:backlog_added" in _static_signature_issues(
        row, payload, hidden_backlog_code, family_contract=family_contract
    )
    assert "CONTRACT_FORBIDDEN_CHANGE:backlog_added" in _static_signature_issues(
        row, payload, shortage_code, family_contract=family_contract
    )


def test_forward_eval_rejects_static_code_failure_before_solver(tmp_path: Path) -> None:
    code_path = tmp_path / "bad_forward.py"
    code_path.write_text(
        "\n".join(
            [
                "import gurobipy as gp",
                "from gurobipy import GRB",
                "model = gp.Model('missing_objective')",
                "x = model.addVar(lb=0)",
                "model.optimize()",
            ]
        ),
        encoding="utf-8",
    )
    row = {
        "fm_id": "fm_static_bad",
        "bt_id": "bt_static_bad",
        "instance_id": "inst_static_bad",
        "generator_id": "fixture",
        "problem_statement": "Maximize value.",
        "code_path": str(code_path),
        "reference_answer": {"objective_value": 1.0},
        "source_compact_data": {"available": True, "value": {"capacity": 1}},
    }

    accepted, record = evaluate_forward_output_row(row)

    assert accepted is False
    assert record["rejection_reason"].startswith("FORWARD_CODE_STATIC_CHECK_FAILED:")
    assert "MISSING_SET_OBJECTIVE_CALL" in record["rejection_reason"]
    assert record["fine_rejection_reason"] == "missing_objective"
    assert record["failure_family"] == "static_code_gate"
    assert record["source_compact_data"]["value"]["capacity"] == 1


def test_forward_eval_accepted_pair_preserves_source_compact_data(tmp_path: Path) -> None:
    code_path = tmp_path / "good_forward.py"
    code_path.write_text(
        "\n".join(
            [
                "import gurobipy as gp",
                "from gurobipy import GRB",
                "model = gp.Model('good_forward')",
                "model.Params.OutputFlag = 0",
                "x = model.addVar(lb=0, name='x')",
                "model.addConstr(x <= 1)",
                "model.setObjective(x, GRB.MAXIMIZE)",
                "model.optimize()",
            ]
        ),
        encoding="utf-8",
    )
    row = {
        "fm_id": "fm_compact_good",
        "bt_id": "bt_compact_good",
        "instance_id": "inst_compact_good",
        "generator_id": "fixture",
        "problem_statement": "Maximize value.",
        "code_path": str(code_path),
        "reference_answer": {"objective_value": 1.0},
        "source_compact_data": {"available": True, "value": {"capacity": 1}},
    }

    accepted, record = evaluate_forward_output_row(row)

    assert accepted is True
    assert record.source_compact_data["value"]["capacity"] == 1


def test_marketshare_contract_accepts_demand_only_source_model() -> None:
    config = yaml.safe_load(Path("engine_configs/family_contracts.yaml").read_text(encoding="utf-8"))
    family_contract = resolve_family_contract({"generator_id": "optmath_marketshare"}, config)
    row = {
        "generator_id": "optmath_marketshare",
        "problem_statement": "Maximize net profit while meeting every market-product demand exactly.",
        "canonical_math_signature": {
            "core_constraints": ["market_product_demand_exactly_satisfied"],
            "forbidden_changes": ["do not introduce resource capacity constraints absent from the source"],
        },
    }
    payload = {
        "modeling_explanation": (
            "Integer Supply[i,j,k] quantities maximize profit. For each market-product pair, "
            "sum_i Supply[i,j,k] == demand[j,k]. The source has no resource capacity constraint."
        ),
        "math_model": "max sum profit[i,j,k] * Supply[i,j,k]; sum_i Supply[i,j,k] == demand[j,k]",
    }
    code = """
import gurobipy as gp
from gurobipy import GRB
model = gp.Model("marketshare")
I = [0, 1]
J = [0]
K = [0]
demand = {(0, 0): 5}
profit = {(0, 0, 0): 3, (1, 0, 0): 4}
supply = model.addVars(I, J, K, vtype=GRB.INTEGER, lb=0, name="Supply")
model.addConstr(gp.quicksum(supply[i, 0, 0] for i in I) == demand[0, 0], name="demand_0_0")
model.setObjective(gp.quicksum(profit[i, 0, 0] * supply[i, 0, 0] for i in I), GRB.MAXIMIZE)
model.optimize()
"""

    issues = _static_signature_issues(row, payload, code, family_contract=family_contract)

    assert issues == []


def test_forward_modeling_output_preserves_source_compact_data(tmp_path: Path) -> None:
    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_spec = registry.get("forward_modeling_prompt")

    result = asyncio.run(
        _forward_model_one(
            {
                "bt_id": "bt_compact_forward",
                "instance_id": "inst_compact_forward",
                "generator_id": "fixture",
                "problem_statement": "Maximize x subject to x <= 1.",
                "reference_answer": {"objective_value": 1.0},
                "source_compact_data": {"available": True, "value": {"capacity": 1}},
            },
            prompt_spec,
            tmp_path,
            client=None,
            mock=True,
            family_contracts={},
        )
    )

    assert result["kind"] == "output"
    dumped = result["output"].model_dump(mode="json")
    assert dumped["source_compact_data"]["value"]["capacity"] == 1


def test_forward_modeling_retries_placeholder_code_once(tmp_path: Path) -> None:
    class PlaceholderThenValidClient:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        async def chat(self, prompt: str, **kwargs):  # noqa: ANN001
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                payload = {
                    "modeling_explanation": "First attempt accidentally leaves placeholder text.",
                    "math_model": "Maximize x subject to x <= 1.",
                    "gurobipy_code": "\n".join(
                        [
                            "import gurobipy as gp",
                            "model = gp.Model('placeholder_attempt')",
                            "# placeholder: replace with real variables and constraints",
                            "model.optimize()",
                        ]
                    ),
                    "self_check": {},
                }
            else:
                payload = {
                    "modeling_explanation": "Second attempt provides complete executable code.",
                    "math_model": "Maximize x subject to x <= 1.",
                    "gurobipy_code": "\n".join(
                        [
                            "import gurobipy as gp",
                            "model = gp.Model('retry_ok')",
                            "x = model.addVar(lb=0, name='x')",
                            "model.addConstr(x <= 1, name='capacity')",
                            "model.setObjective(x, gp.GRB.MAXIMIZE)",
                            "model.optimize()",
                        ]
                    ),
                    "self_check": {},
                }
            return {
                "content": json.dumps(payload),
                "request_id": f"req_{self.calls}",
                "latency_sec": 0.1,
                "model": "mock-model",
                "endpoint": {"name": "mock-endpoint"},
                "attempt": 1,
            }

    registry = PromptRegistry(Path("or_cpt_engine/prompts/prompt_versions.yaml"))
    prompt_spec = registry.get("forward_modeling_prompt")
    client = PlaceholderThenValidClient()

    result = asyncio.run(
        _forward_model_one(
            {
                "bt_id": "bt_forward_retry",
                "instance_id": "inst_forward_retry",
                "generator_id": "fixture",
                "problem_statement": "Maximize x subject to x <= 1.",
                "reference_answer": {"objective_value": 1.0},
            },
            prompt_spec,
            tmp_path,
            client=client,
            mock=False,
            family_contracts={},
        )
    )

    assert result["kind"] == "output"
    assert client.calls == 2
    assert "QUALITY RETRY DIRECTIVE" not in client.prompts[0]
    assert "QUALITY RETRY DIRECTIVE" in client.prompts[1]
    assert "STATIC_CODE_CHECK_FAILED" in client.prompts[1]
    assert "PLACEHOLDER_TEXT" in client.prompts[1]
    assert "Replace placeholder/TODO code" in client.prompts[1]
    dumped = result["output"].model_dump(mode="json")
    assert dumped["llm_metadata"]["quality_attempt"] == 2
    assert dumped["llm_metadata"]["quality_max_attempts"] == 2
    assert dumped["llm_metadata"]["quality_retries"] == 1
    assert len(dumped["llm_metadata"]["quality_retry_reasons"]) == 1
    assert "STATIC_CODE_CHECK_FAILED" in dumped["llm_metadata"]["quality_retry_reasons"][0]
    assert "PLACEHOLDER_TEXT" in dumped["llm_metadata"]["quality_retry_reasons"][0]
    assert dumped["llm_metadata"]["quality_retry_events"][0]["quality_attempt"] == 1
    assert dumped["llm_metadata"]["quality_retry_events"][0]["request_id"] == "req_1"
    assert dumped["llm_metadata"]["quality_retry_events"][0]["endpoint"]["name"] == "mock-endpoint"
    assert "PLACEHOLDER_TEXT" in dumped["llm_metadata"]["quality_retry_events"][0]["reason"]
    assert "placeholder" not in dumped["generated_answer"]["gurobipy_code"].lower()


def test_forward_quality_retry_hints_cover_current_static_replay_failures() -> None:
    hints = _forward_retry_issue_hints(
        [
            "STATIC_SIGNATURE_MISMATCH:CONTRACT_FORBIDDEN_CHANGE:backlog_added",
            "STATIC_SIGNATURE_MISMATCH:CONTRACT_MISSING_REQUIRED_CONSTRAINT:nonnegative_integer_supply",
            "STATIC_SIGNATURE_MISMATCH:CONTRACT_FORBIDDEN_CHANGE:unit_production_cost_added",
            "STATIC_SIGNATURE_MISMATCH:CONTRACT_FORBIDDEN_CHANGE:vehicle_routing_structure_added",
        ]
    )
    joined = "\n".join(hints)

    assert "Do not add backlog" in joined
    assert "GRB.INTEGER" in joined
    assert "Do not invent unit production costs" in joined
    assert "Do not transform assignment or flow tasks into vehicle routing" in joined


def test_forward_execution_failure_classifier_uses_traceback_markers() -> None:
    assert classify_execution_failure({"execution_status": "SUCCESS", "status": "OPTIMAL"}) == ""
    assert classify_execution_failure({"status": "EXECUTION_ERROR", "stderr": "Traceback\nNameError: name 'x' is not defined"}) == "NAME_ERROR"
    assert classify_execution_failure({"status": "TIMEOUT", "stderr": ""}) == "TIMEOUT"
    assert (
        classify_execution_failure(
            {
                "status": "EXECUTION_ERROR",
                "stderr": "gurobipy._exception.GurobiError: Model too large for size-limited license",
            }
        )
        == "GUROBI_SIZE_LIMIT"
    )


def test_forward_eval_rejection_classifier_splits_common_failure_modes() -> None:
    key_error = classify_forward_eval_rejection(
        row={"generator_id": "fixture"},
        rejection_reason="FORWARD_CODE_EXECUTION_FAILED:EXECUTION_ERROR:KEY_ERROR",
        solver_result={"execution_status": "FAILED", "status": "EXECUTION_ERROR", "stderr": "KeyError: ('a', 'b')"},
    )
    infeasible = classify_forward_eval_rejection(
        row={"generator_id": "fixture"},
        rejection_reason="FORWARD_SOLVER_NOT_OPTIMAL:INFEASIBLE",
        solver_result={"execution_status": "SUCCESS", "status": "INFEASIBLE"},
    )

    assert key_error["fine_rejection_reason"] == "execution_key_error_or_invalid_schema"
    assert key_error["failure_family"] == "execution_failure"
    assert infeasible["fine_rejection_reason"] == "solver_infeasible"
    assert infeasible["failure_family"] == "solver_status_failure"


def test_forward_eval_rejection_classifier_identifies_gurobi_size_limit() -> None:
    classified = classify_forward_eval_rejection(
        row={"generator_id": "optmath_singlelevelsmallbucket"},
        rejection_reason="FORWARD_CODE_EXECUTION_FAILED:EXECUTION_ERROR:EXECUTION_ERROR",
        solver_result={
            "execution_status": "FAILED",
            "status": "EXECUTION_ERROR",
            "stderr": "gurobipy._exception.GurobiError: Model too large for size-limited license",
        },
    )

    assert classified["fine_rejection_reason"] == "gurobi_size_limited_license"
    assert classified["failure_family"] == "execution_failure"
    assert classified["repairable"] is False
    assert classified["diagnostic_evidence"]["failure_class"] == "GUROBI_SIZE_LIMIT"


def test_forward_eval_rejection_classifier_normalizes_gurobi_numeric_status() -> None:
    classified = classify_forward_eval_rejection(
        row={"generator_id": "fixture"},
        rejection_reason="FORWARD_CODE_EXECUTION_FAILED:3:UNCLASSIFIED_EXECUTION_ERROR",
        solver_result={"execution_status": "FAILED", "status": "3"},
    )

    assert classified["rejection_reason_base"] == "FORWARD_SOLVER_NOT_OPTIMAL:INFEASIBLE"
    assert classified["fine_rejection_reason"] == "solver_infeasible"
    assert classified["failure_family"] == "solver_status_failure"


def test_rendering_numeric_normalization_ignores_code_blocks_by_default() -> None:
    text = "\n".join(
        [
            "# Optimization Modeling Report",
            "",
            "## Problem",
            "The cost is 52.69059351204961 in the displayed summary.",
            "```python",
            "value = 52.69059351204961",
            "```",
        ]
    )

    normalized, metadata = _normalize_rendered_numeric_precision(
        text,
        {"enabled": True, "max_decimal_places": 2, "apply_to_code_blocks": False},
    )
    quality = _check_rendered_text(
        normalized
        + "\n## Model Formulation\nx\n## Implementation\n```python\nmodel.optimize()\n```\n## Optimal Solution\nx\n## Feasibility Check\nx\n## Validation\nx",
        "final_model_report",
        numeric_policy={"enabled": True, "max_decimal_places": 2, "apply_to_code_blocks": False},
    )

    assert "52.69 in the displayed summary" in normalized
    assert "value = 52.69059351204961" in normalized
    assert metadata["normalized_count"] == 1
    assert not any(issue.startswith("EXCESSIVE_DECIMAL_PRECISION") for issue in quality["issues"])


def test_seed_only_quality_analysis_drives_production_plan(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        "\n".join(
            [
                "version: v1.0.0",
                "profiles:",
                "  healthy_seed:",
                "    enabled: true",
                "    priority: high",
                "    recommended_weight: 3.0",
                "    task_family: routing",
                "    expected_acceptance_rate: 0.2",
                "    modeling_concepts: [routing_binary_variable]",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "seed_run"
    write_jsonl(
        run_dir / "03_instance_generation" / "instances_generated.jsonl",
        [{"generator_id": "healthy_seed", "concept_tags": ["routing_binary_variable"]} for _ in range(10)],
    )
    write_jsonl(
        run_dir / "04_solver_validation" / "solver_validated_instances.jsonl",
        [{"generator_id": "healthy_seed"} for _ in range(9)],
    )
    write_jsonl(
        run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl",
        [{"generator_id": "healthy_seed"} for _ in range(9)],
    )

    output_dir = tmp_path / "seed_quality"
    analyze_generator_quality(run_dir, output_dir, profiles_path=profiles_path)
    quality_row = read_jsonl(output_dir / "generator_quality_report.jsonl")[0]

    assert quality_row["stage_scope"] == "seed_only"
    assert quality_row["seed_quality_score"] > 0.85
    assert quality_row["recommended_action"] in {"increase_weight", "keep"}
    assert quality_row["production_tier"] in {"A_core", "B_improve"}

    plan_result = plan_production(
        target_accepted=9,
        profiles_path=profiles_path,
        output_dir=tmp_path / "seed_plan",
        quality_report_path=output_dir / "generator_quality_report.jsonl",
    )
    assert plan_result["planned_attempts"] == 10
    plan_payload = json.loads((tmp_path / "seed_plan" / "production_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["generators"][0]["production_tier"] in {"A_core", "B_improve"}


def test_production_plan_holds_low_quality_generators(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        "\n".join(
            [
                "version: v1.0.0",
                "profiles:",
                "  good_gen:",
                "    enabled: true",
                "    priority: high",
                "    recommended_weight: 1.0",
                "  bad_gen:",
                "    enabled: true",
                "    priority: high",
                "    recommended_weight: 10.0",
            ]
        ),
        encoding="utf-8",
    )
    quality_path = tmp_path / "quality.jsonl"
    write_jsonl(
        quality_path,
        [
            {
                "generator_id": "good_gen",
                "production_tier": "A_core",
                "recommended_action": "increase_weight",
                "forward_objective_match_rate": 0.8,
                "generator_score": 0.9,
            },
            {
                "generator_id": "bad_gen",
                "production_tier": "D_hold",
                "recommended_action": "optimize_generator_source",
                "forward_objective_match_rate": 0.1,
                "generator_score": 0.1,
                "quality_risk_flags": ["LOW_FORWARD_OBJECTIVE_MATCH_RATE"],
            },
        ],
    )

    plan_production(
        target_accepted=20,
        profiles_path=profiles_path,
        output_dir=tmp_path / "plan",
        quality_report_path=quality_path,
    )
    rows = {
        row["generator_id"]: row
        for row in json.loads((tmp_path / "plan" / "production_plan.json").read_text(encoding="utf-8"))["generators"]
    }

    assert rows["good_gen"]["planned_attempts"] > 0
    assert rows["bad_gen"]["production_tier"] == "D_hold"
    assert rows["bad_gen"]["planned_attempts"] == 0
    assert "LOW_FORWARD_OBJECTIVE_MATCH_RATE" in rows["bad_gen"]["quality_risk_flags"]


def test_run_full_counts_can_follow_scaled_production_plan(tmp_path: Path) -> None:
    smoke_report = tmp_path / "adapter_smoke_report.jsonl"
    write_jsonl(
        smoke_report,
        [
            {"generator_id": "gen_a", "load_status": "PASS", "generate_status": "PASS"},
            {"generator_id": "gen_b", "load_status": "PASS", "generate_status": "PASS"},
            {"generator_id": "gen_c", "load_status": "PASS", "generate_status": "PASS"},
            {"generator_id": "gen_failed", "load_status": "PASS", "generate_status": "FAIL"},
        ],
    )
    production_plan = tmp_path / "production_plan.json"
    production_plan.write_text(
        json.dumps(
            {
                "generators": [
                    {"generator_id": "gen_a", "planned_attempts": 100},
                    {"generator_id": "gen_b", "planned_attempts": 50},
                    {"generator_id": "gen_c", "planned_attempts": 25},
                    {"generator_id": "gen_failed", "planned_attempts": 1000},
                    {"generator_id": "gen_not_in_smoke", "planned_attempts": 1000},
                ]
            }
        ),
        encoding="utf-8",
    )

    full_counts = _build_per_generator_count_from_plan(
        smoke_report_path=smoke_report,
        selected_generator_ids=None,
        production_plan_path=production_plan,
        num_instances=None,
    )
    scaled_counts = _build_per_generator_count_from_plan(
        smoke_report_path=smoke_report,
        selected_generator_ids=None,
        production_plan_path=production_plan,
        num_instances=7,
    )
    selected_counts = _build_per_generator_count_from_plan(
        smoke_report_path=smoke_report,
        selected_generator_ids={"gen_b", "gen_c"},
        production_plan_path=production_plan,
        num_instances=6,
    )

    assert full_counts == {"gen_a": 100, "gen_b": 50, "gen_c": 25}
    assert scaled_counts == {"gen_a": 4, "gen_b": 2, "gen_c": 1}
    assert selected_counts == {"gen_b": 4, "gen_c": 2}


def test_solver_validation_can_run_with_concurrency_without_gurobi(tmp_path: Path) -> None:
    input_path = tmp_path / "instances.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "instance_id": f"inst_{idx}",
                "run_id": "test_run",
                "generator_id": "fixture",
                "seed": idx,
                "difficulty_level": "level_1",
                "initial_solver_status": "OPTIMAL",
                "initial_objective_value": float(idx),
            }
            for idx in range(10)
        ],
    )

    result = validate_solver(input_path, tmp_path / "solver", timeout_seconds=5, concurrency=4, threads_per_process=1)

    assert result == {"validated": 10, "rejected": 0}
    assert len(read_jsonl(tmp_path / "solver" / "solver_validated_instances.jsonl")) == 10
    report = (tmp_path / "solver" / "solver_validation_report.md").read_text(encoding="utf-8")
    assert "- Concurrency: 4" in report
    assert "- Threads per solver process: 1" in report


def test_production_plan_usage_is_written_for_scaled_seed_factory(tmp_path: Path) -> None:
    smoke_report = tmp_path / "adapter_smoke_report.jsonl"
    write_jsonl(
        smoke_report,
        [
            {"generator_id": "gen_a", "load_status": "PASS", "generate_status": "PASS"},
            {"generator_id": "gen_b", "load_status": "PASS", "generate_status": "PASS"},
            {"generator_id": "gen_failed", "load_status": "PASS", "generate_status": "FAIL"},
        ],
    )
    production_plan = tmp_path / "production_plan.json"
    production_plan.write_text(
        json.dumps(
            {
                "generators": [
                    {"generator_id": "gen_a", "planned_attempts": 80},
                    {"generator_id": "gen_b", "planned_attempts": 20},
                    {"generator_id": "gen_failed", "planned_attempts": 100},
                    {"generator_id": "gen_missing", "planned_attempts": 100},
                ]
            }
        ),
        encoding="utf-8",
    )
    production_plan.with_suffix(".md").write_text("# Plan\n", encoding="utf-8")

    per_gen_count, usage = _resolve_per_generator_counts(
        smoke_report_path=smoke_report,
        selected_generator_ids=None,
        production_plan_path=production_plan,
        num_instances=5,
        yaml_per_gen=1,
    )
    usage_path = _record_production_plan_usage(tmp_path / "run", production_plan, usage)

    assert per_gen_count == {"gen_a": 4, "gen_b": 1}
    assert usage is not None
    assert usage["planned_attempts_original"] == 300
    assert usage["planned_attempts_used"] == 5
    assert usage["skipped_generator_ids"] == ["gen_failed", "gen_missing"]
    assert usage_path is not None
    assert usage_path.exists()
    assert (tmp_path / "run" / "reports" / "production_plan_used" / "production_plan.json").exists()
    assert (tmp_path / "run" / "reports" / "production_plan_used" / "production_plan.md").exists()


def test_seed_and_cpt_manifests_record_lineage(tmp_path: Path) -> None:
    run_dir = tmp_path / "seed_run"
    write_jsonl(run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl", [{"instance_id": "seed_1"}, {"instance_id": "seed_2"}])
    write_jsonl(run_dir / "04b_instance_quality" / "quality_review_instances.jsonl", [{"instance_id": "seed_review"}])
    write_jsonl(run_dir / "04b_instance_quality" / "quality_rejected_instances.jsonl", [])

    seed_manifest = _write_seed_manifest(
        run_dir=run_dir,
        run_id="seed_run",
        config=EngineConfig(),
        stage_results={"instance_quality": {"accepted": 2, "review": 1, "rejected": 0}},
        per_generator_counts={"gen_a": 2},
        production_plan_usage=None,
        wall_seconds=12.34,
    )
    seed_payload = json.loads(seed_manifest.read_text(encoding="utf-8"))

    assert seed_payload["seed_count"] == 2
    assert seed_payload["review_count"] == 1
    assert seed_payload["seed_output"].endswith("quality_validated_instances.jsonl")

    cpt_run_dir = tmp_path / "cpt_run"
    write_jsonl(cpt_run_dir / "10_train_val_export" / "train.jsonl", [{"id": "a"}])

    cpt_manifest = _write_cpt_run_manifest(
        run_dir=cpt_run_dir,
        run_id="cpt_run",
        seed_input=run_dir / "04b_instance_quality" / "quality_validated_instances.jsonl",
        seed_manifest=seed_manifest,
        stage_results={"export": {"train": 1, "duplicates_removed": 0, "mode": "train_only_streaming"}},
        wall_seconds=56.78,
    )
    cpt_payload = json.loads(cpt_manifest.read_text(encoding="utf-8"))

    assert cpt_payload["seed_input_count"] == 2
    assert cpt_payload["export_mode"] == "train_only_streaming"
    assert cpt_payload["train_count"] == 1
    assert "val_count" not in cpt_payload
    assert "test_count" not in cpt_payload
    assert cpt_payload["seed_manifest"] == str(seed_manifest)


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_generate_instances_extracts_gurobi_model_from_generator_class(tmp_path: Path) -> None:
    generators_dir = _make_gurobi_model_generator(tmp_path)
    registry_dir = tmp_path / "registry"
    scan_optmath_generators(generators_dir, registry_dir)
    smoke_dir = tmp_path / "smoke"
    smoke_test_generators(registry_dir / "generator_registry.jsonl", smoke_dir)
    generation_dir = tmp_path / "instances"

    result = generate_instances(
        registry_dir / "generator_registry.jsonl",
        smoke_dir / "adapter_smoke_report.jsonl",
        generation_dir,
        run_id="test_run",
        num_instances_per_generator=1,
    )

    assert result["generated"] == 1
    row = read_jsonl(generation_dir / "instances_generated.jsonl")[0]
    assert row["lp_text"]
    assert row["initial_solver_status"] == "OPTIMAL"
    assert row["initial_objective_value"] == 1.0
    assert row["source_compact_data"]["available"] is True
    assert row["source_compact_data"]["value"]["compact_cell_tower_tables"]["budget"] == 1

    solver_dir = tmp_path / "solver"
    validate_solver(generation_dir / "instances_generated.jsonl", solver_dir, timeout_seconds=30)
    solver_row = read_jsonl(solver_dir / "solver_validated_instances.jsonl")[0]
    assert solver_row["source_compact_data"]["available"] is True
    assert solver_row["source_compact_data"]["value"]["compact_cell_tower_tables"]["budget"] == 1


def test_seed_input_preflight_reports_missing_high_risk_compact_facts(tmp_path: Path) -> None:
    seed_input = tmp_path / "quality_validated_instances.jsonl"
    write_jsonl(
        seed_input,
        [
            {
                "instance_id": "inst_clsp_old",
                "generator_id": "optmath_clsp_expand_capacity",
                "generation_params": {"difficulty_config": {"seed": 1}},
            },
            {
                "instance_id": "inst_uls_new",
                "generator_id": "optmath_uncapacitatedlotsizing",
                "source_compact_data": {
                    "available": True,
                    "value": {"compact_lotsizing_tables": {"model_family": "uncapacitated_lot_sizing_without_backlog"}},
                },
            },
            {
                "instance_id": "inst_marketshare_fallback",
                "generator_id": "optmath_marketshare",
                "generation_params": {"compact_marketshare_tables": {"demand_table": []}},
            },
            {
                "instance_id": "inst_steel4_legacy",
                "generator_id": "optmath_steel4",
                "generation_params": {
                    "products": ["product_0"],
                    "stages": ["stage_0"],
                    "profit": {"product_0": 296},
                    "commit": {"product_0": 43},
                    "market": {"product_0": 122},
                    "available": {"stage_0": 61.13},
                    "processing_hours_per_unit": {"product_0|stage_0": 0.1},
                },
            },
        ],
    )

    result = _audit_seed_input_for_cpt(seed_input, tmp_path / "cpt_run")

    assert result["seed_rows"] == 4
    assert result["explicit_source_compact"] == 1
    assert result["fallback_generation_param_compact"] == 1
    assert result["legacy_reconstructable_compact"] == 1
    assert result["high_risk_missing_compact"] == 1
    assert result["missing_compact_by_generator"] == {"optmath_clsp_expand_capacity": 1}
    assert result["legacy_reconstructable_by_generator"] == {"optmath_steel4": 1}
    report = (tmp_path / "cpt_run" / "reports" / "seed_input_preflight.md").read_text(encoding="utf-8")
    assert "High-risk rows missing compact facts" in report
    assert "Legacy reconstructable compact facts" in report
    assert "optmath_clsp_expand_capacity" in report


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_solver_validation_on_fixture_generator(tmp_path: Path) -> None:
    instances = [
        {
            "instance_id": "inst_1",
            "run_id": "test_run",
            "generator_id": "fixture",
            "seed": 1,
            "difficulty_level": "level_1",
            "gurobi_code": _tiny_gurobi_code(),
        }
    ]
    input_path = tmp_path / "instances.jsonl"
    write_jsonl(input_path, instances)
    output_dir = tmp_path / "solver"

    result = validate_solver(input_path, output_dir, timeout_seconds=30)

    assert result["validated"] == 1
    validated = read_jsonl(output_dir / "solver_validated_instances.jsonl")
    assert validated[0]["reference_answer"]["objective_value"] == 1.0
    solution = validated[0]["solver_validation"]["solution"]
    assert solution["nonzero_variable_values"]["x"] == 1.0
    assert solution["objective_recomputed"] == 1.0
    assert solution["variable_diagnostics"][0]["name"] == "x"
    assert solution["variable_diagnostics"][0]["value"] == 1.0
    assert solution["constraint_diagnostics"][0]["is_binding"] is True
    assert solution["solution_summary"]["binding_constraint_count"] == 1


def test_instance_quality_gate_rejects_all_upper_bound_solution(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    write_jsonl(
        input_path,
        [
            _quality_fixture_row(
                instance_id="inst_pass",
                lp_text="pass lp",
                variable_values=[("x", 0.6, 0.0, 1.0), ("y", 0.3, 0.0, 1.0)],
                all_vars_at_ub=False,
                binding_count=1,
            ),
            _quality_fixture_row(
                instance_id="inst_reject",
                lp_text="reject lp",
                variable_values=[("x", 1.0, 0.0, 1.0), ("y", 1.0, 0.0, 1.0)],
                all_vars_at_ub=True,
                binding_count=1,
            ),
        ],
    )

    result = validate_instance_quality(input_path, tmp_path / "quality")

    assert result["accepted"] == 1
    assert result["rejected"] == 1
    accepted = read_jsonl(tmp_path / "quality" / "quality_validated_instances.jsonl")
    rejected = read_jsonl(tmp_path / "quality" / "quality_rejected_instances.jsonl")
    assert accepted[0]["instance_quality"]["status"] == "PASS"
    assert "ALL_NON_FIXED_VARIABLES_AT_UPPER_BOUND" in rejected[0]["instance_quality"]["rejection_reasons"]


def test_instance_quality_gate_rejects_seed_long_decimals(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    row = _quality_fixture_row(
        instance_id="inst_long_decimal",
        lp_text="minimize x",
        variable_values=[("x", 0.6, 0.0, 1.0), ("y", 0.3, 0.0, 1.0)],
        all_vars_at_ub=False,
        binding_count=1,
    )
    row["generation_params"] = {"compact_fixture_tables": {"business_cost": 1.234567}}
    write_jsonl(
        input_path,
        [row],
    )

    result = validate_instance_quality(input_path, tmp_path / "quality")

    assert result["rejected"] == 1
    rejected = read_jsonl(tmp_path / "quality" / "quality_rejected_instances.jsonl")
    quality = rejected[0]["instance_quality"]
    assert quality["seed_long_decimal_count"] == 1
    assert "EXCESSIVE_SEED_DECIMAL_PRECISION" in quality["rejection_reasons"]


def test_instance_quality_decimal_gate_ignores_non_compact_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    row = _quality_fixture_row(
        instance_id="inst_metadata_decimal",
        lp_text="minimize 1.234567 x",
        variable_values=[("x", 0.6, 0.0, 1.0), ("y", 0.3, 0.0, 1.0)],
        all_vars_at_ub=False,
        binding_count=1,
    )
    row["generation_params"] = {
        "coverage_matrix_summary": {"density": 0.4481},
        "compact_fixture_tables": {"business_cost": 12.3, "business_limit": 4},
    }
    write_jsonl(input_path, [row])

    result = validate_instance_quality(input_path, tmp_path / "quality")

    assert result["accepted"] == 1
    accepted = read_jsonl(tmp_path / "quality" / "quality_validated_instances.jsonl")
    quality = accepted[0]["instance_quality"]
    assert quality["seed_long_decimal_count"] == 0
    assert "EXCESSIVE_SEED_DECIMAL_PRECISION" not in quality["rejection_reasons"]


def test_instance_quality_controller_rejects_static_line_reserve_only(tmp_path: Path) -> None:
    input_path = tmp_path / "solver_validated.jsonl"
    row = _quality_fixture_row(
        instance_id="inst_static_line",
        lp_text="static line lp",
        variable_values=[("x[0]", 0.0, 0.0, 1.0), ("s[0]", 10.0, 0.0, 100.0)],
        all_vars_at_ub=False,
        binding_count=1,
    )
    row["generator_id"] = "optmath_staticlineplanning"
    write_jsonl(input_path, [row])

    result = validate_instance_quality(input_path, tmp_path / "quality")

    assert result["rejected"] == 1
    rejected = read_jsonl(tmp_path / "quality" / "quality_rejected_instances.jsonl")
    reasons = rejected[0]["instance_quality"]["rejection_reasons"]
    assert "ACTIVATION_VARIABLES_ALL_ZERO" in reasons
    assert "SLACK_OR_RESERVE_DOMINATES_SOLUTION" in reasons


def test_instance_quality_lot_sizing_treats_stock_and_backlog_as_business_state() -> None:
    row = _quality_fixture_row(
        instance_id="inst_smallbucket",
        lp_text="small bucket lot sizing lp",
        variable_values=[
            ("Amount[0,0,0]", 20.0, 0.0, 100.0),
            ("Production[0,0,0]", 1.0, 0.0, 1.0),
            ("Startup[0,0,0]", 1.0, 0.0, 1.0),
            ("Stock[0,0]", 40.0, 0.0, 100.0),
            ("Backlog[0,0]", 15.0, 0.0, 100.0),
        ],
        all_vars_at_ub=False,
        binding_count=2,
    )
    row["generator_id"] = "optmath_singlelevelsmallbucket"
    row["task_family"] = "lot_sizing"

    quality = evaluate_instance_quality(
        row,
        thresholds=dict(DEFAULT_THRESHOLDS),
        profile={"task_family": "lot_sizing"},
    )

    assert quality["status"] == "PASS"
    assert "SLACK_OR_RESERVE_DOMINATES_SOLUTION" not in quality["rejection_reasons"]
    assert quality["generator_quality_metrics"]["slack_variable_count"] == 0
    assert quality["generator_quality_metrics"]["business_nonzero_count"] == 4
    assert quality["generator_quality_metrics"]["activation_variable_count"] == 2


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_target_generators_avoid_known_degenerate_patterns() -> None:
    from or_cpt_engine.generators.optmath_seed.StaticLinePlanning.StaticLinePlanning import Generator as StaticLinePlanning
    from or_cpt_engine.generators.optmath_seed.steel3.steel3_parsed import Generator as Steel3
    from or_cpt_engine.generators.optmath_seed.steel4.steel4_parsed import Generator as Steel4

    for generator_cls in (Steel3, Steel4):
        model = generator_cls(seed=0).generate_instance()
        model.optimize()
        assert model.Status == 2
        upper_bound_constraints = [
            constraint
            for constraint in model.getConstrs()
            if constraint.ConstrName.startswith(("MaxSold_", "Market_"))
        ]
        capacity_constraints = [
            constraint
            for constraint in model.getConstrs()
            if constraint.ConstrName.startswith(("TimeCapacity", "Time_"))
        ]
        assert upper_bound_constraints
        assert capacity_constraints
        assert sum(abs(constraint.Slack) <= 1e-6 for constraint in upper_bound_constraints) < len(upper_bound_constraints)
        assert any(abs(constraint.Slack) <= 1e-6 for constraint in capacity_constraints)

    steel3 = Steel3(seed=0)
    steel3_model = steel3.generate_instance()
    steel3_model.optimize()
    assert steel3_model.Status == 2
    assert steel3.parameters["compact_steel_product_mix_tables"]["product_table"]["rows"]
    assert steel3.parameters["compact_steel_product_mix_tables"]["time_capacity"]["available_hours"] == steel3.parameters[
        "available_hours"
    ]
    assert steel3.parameters["structured_problem_data"]["parameters"]["product_table"]
    compact_steel3_source = _compact_source_data({"generation_params": steel3.parameters})
    assert compact_steel3_source["available"] is True
    assert compact_steel3_source["truncated"] is False
    assert "compact_steel_product_mix_tables" in compact_steel3_source["value"]
    assert "single_stage_continuous_product_mix" in json.dumps(compact_steel3_source["value"])
    assert "state there are no inventory" in " ".join(steel3.parameters["required_parameter_presentation"]).lower()
    assert all(variable.VType == "C" for variable in steel3_model.getVars())

    steel4 = Steel4(seed=0)
    steel4_model = steel4.generate_instance()
    steel4_model.optimize()
    assert steel4_model.Status == 2
    assert 3 <= len(steel4.parameters["products"]) <= 5
    assert 3 <= len(steel4.parameters["stages"]) <= 6
    assert steel4.parameters["product_table"]
    assert steel4.parameters["stage_capacity_table"]
    assert steel4.parameters["compact_steel_product_mix_tables"]["product_table"]["rows"]
    assert steel4.parameters["compact_steel_product_mix_tables"]["stage_capacity_table"]["rows"]
    assert steel4.parameters["compact_steel_product_mix_tables"]["processing_hours_per_ton_matrix"]["rows"]
    compact_steel4_source = _compact_source_data({"generation_params": steel4.parameters})
    assert compact_steel4_source["available"] is True
    assert compact_steel4_source["truncated"] is False
    assert "compact_steel_product_mix_tables" in compact_steel4_source["value"]
    assert steel4.parameters["capacity_envelope_by_stage"]
    assert steel4.parameters["planned_bottleneck_candidate_stages"]
    assert steel4.parameters["structured_problem_data"]
    assert "State production variables are continuous tons, not binary product selection." in steel4.parameters[
        "required_parameter_presentation"
    ]
    assert "Do not confuse stage available hours with product maximum market tons." in steel4.parameters[
        "required_parameter_presentation"
    ]
    assert any("Do not reinterpret stages as time periods" in item for item in steel4.parameters["business_interpretation_guardrails"])
    assert all(variable.VType == "C" for variable in steel4_model.getVars())
    assert any(
        constraint.ConstrName.startswith("Time_") and abs(constraint.Slack) <= 1e-6
        for constraint in steel4_model.getConstrs()
    )
    market_constraints = [
        constraint for constraint in steel4_model.getConstrs() if constraint.ConstrName.startswith("Market_")
    ]
    assert sum(abs(constraint.Slack) <= 1e-6 for constraint in market_constraints) < len(market_constraints)
    production = {variable.VarName.split("[", 1)[1].rstrip("]"): variable.X for variable in steel4_model.getVars()}
    for product, value in production.items():
        assert value + 1e-6 >= steel4.parameters["commit"][product]
        assert value <= steel4.parameters["market"][product] + 1e-6
    for stage in steel4.parameters["stages"]:
        used_hours = sum(
            steel4.parameters["processing_hours_per_unit"][f"{product}|{stage}"] * production[product]
            for product in steel4.parameters["products"]
        )
        assert used_hours <= steel4.parameters["available"][stage] + 1e-6

    model = StaticLinePlanning(seed=0).generate_instance()
    model.optimize()
    assert model.Status == 2
    reserve_sum = sum(variable.X for variable in model.getVars() if variable.VarName.startswith("s["))
    active_lines = sum(variable.X for variable in model.getVars() if variable.VarName.startswith("x["))
    demand_constraints = [
        constraint
        for constraint in model.getConstrs()
        if constraint.ConstrName.startswith("Demand_")
    ]
    assert reserve_sum <= 1e-6
    assert active_lines >= 1
    assert any(abs(constraint.Slack) <= 1e-6 for constraint in demand_constraints)
    assert not any(variable.VarName.startswith("y[") for variable in model.getVars())
    assert StaticLinePlanning(seed=0).problem_type == "static_line_planning"

    static_line = StaticLinePlanning(seed=0)
    static_line_model = static_line.generate_instance()
    static_line_model.optimize()
    assert static_line_model.Status == 2
    assert static_line.parameters["line_table"]
    assert static_line.parameters["od_table"]
    assert static_line.parameters["od_service_table"]
    assert static_line.parameters["line_pass_table"]
    assert static_line.parameters["node_transfer_table"]
    assert static_line.parameters["structured_problem_data"]
    assert "State transfer requirements are fixed parameters r[n], not decision variables." in static_line.parameters[
        "required_parameter_presentation"
    ]
    assert any("Do not invent z[i,j,l]" in item for item in static_line.parameters["business_interpretation_guardrails"])
    assert all(len(row["serving_lines"]) >= 1 for row in static_line.parameters["od_service_table"])
    for row in static_line.parameters["node_transfer_table"]:
        if row["transfer_required"]:
            assert len(row["lines_passing"]) >= 2


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_batch2_facility_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.CFLP.CFLP_parsed import Generator as CFLP
    from or_cpt_engine.generators.optmath_seed.facility_location.facility_location_parsed import (
        Generator as FacilityLocation,
    )
    from or_cpt_engine.generators.optmath_seed.The_p_dispersion_model.The_p_dispersion_model import Generator as PDispersion
    from or_cpt_engine.generators.optmath_seed.scooter_location.scooter_location_parsed import Generator as ScooterLocation

    facility = FacilityLocation(seed=0)
    facility_model = facility.generate_instance()
    facility_model.optimize()
    assert facility_model.Status == 2
    assert facility_model.NumVars <= 200
    assert facility_model.NumConstrs <= 120
    assert facility.parameters["variable_shipping_cost"]
    assert facility.parameters["compact_facility_location_tables"]
    assert facility.parameters["compact_facility_location_tables"]["shipping_cost_table"]
    assert facility.parameters["shipping_cost_table_shape"]["entries"] <= 144
    assert facility.parameters["structured_problem_data"]
    assert facility.parameters["throughput_bounds_by_dc"]
    assert len(json.dumps(facility.parameters, ensure_ascii=False)) < 18000
    assert facility.parameters["fixed_throughput_cost"]
    assert facility.parameters["unit_throughput_cost"]
    assert "served_zone_commodity_demand_exactly_shipped" in facility.parameters["required_constraints"]
    assert "explicit_distribution_center_open_before_zone_assignment" in facility.parameters["required_constraints"]
    assert "Do not omit fixed distribution-center selection costs or unit throughput costs." in facility.parameters["business_interpretation_guardrails"]
    assert (
        "Do not omit the explicit Served[d,z] <= Selected[d] open-before-assignment link."
        in facility.parameters["business_interpretation_guardrails"]
    )
    assert any(variable.VarName.startswith("Selected") and variable.X > 0.5 for variable in facility_model.getVars())
    assert any(variable.VarName.startswith("Served") and variable.X > 0.5 for variable in facility_model.getVars())
    assert any(variable.VarName.startswith("Shipped") and variable.X > 1e-6 for variable in facility_model.getVars())
    assert any(constraint.ConstrName.startswith("Demand_") and abs(constraint.Slack) <= 1e-6 for constraint in facility_model.getConstrs())
    assert any(constraint.ConstrName.startswith("MaxThroughput_") for constraint in facility_model.getConstrs())
    compact_source = _compact_source_data({"generation_params": facility.parameters})
    assert compact_source["available"] is True
    assert "compact_facility_location_tables" in compact_source["value"]
    assert any(constraint.ConstrName.startswith("OpenBeforeServe_") for constraint in facility_model.getConstrs())

    cflp = CFLP(seed=0)
    cflp_model = cflp.generate_instance()
    cflp_model.optimize()
    assert cflp_model.Status == 2
    assert cflp.parameters["compact_cflp_tables"]["facility_table"]
    assert cflp.parameters["compact_cflp_tables"]["customer_demand_table"]
    assert cflp.parameters["compact_cflp_tables"]["transport_cost_matrix"]["rows"]
    compact_cflp_source = _compact_source_data({"generation_params": cflp.parameters})
    assert compact_cflp_source["available"] is True
    assert "compact_cflp_tables" in compact_cflp_source["value"]
    assert "customer_demand_exactly_satisfied" in cflp.parameters["required_constraints"]
    assert "ship_only_from_open_facility" in cflp.parameters["required_constraints"]
    assert "open_facility_capacity_limit" in cflp.parameters["required_constraints"]
    assert any(variable.VarName.startswith("FacilityOpen") and variable.VType == "B" for variable in cflp_model.getVars())
    assert any(variable.VarName.startswith("ShippedAmount") and variable.X > 1e-6 for variable in cflp_model.getVars())
    assert any(constraint.ConstrName.startswith("Valid") for constraint in cflp_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Capacity") for constraint in cflp_model.getConstrs())
    for customer in cflp.parameters["customers"]:
        shipped = sum(
            cflp_model.getVarByName(f"ShippedAmount[{customer},{facility_id}]").X
            for facility_id in cflp.parameters["facilities"]
        )
        assert abs(shipped - cflp.parameters["demands"][customer]) <= 1e-6

    scooter = ScooterLocation(seed=0)
    scooter_model = scooter.generate_instance()
    scooter_model.optimize()
    assert scooter_model.Status == 2
    assignment_count = sum(variable.X for variable in scooter_model.getVars() if variable.VarName.startswith("Assign"))
    selected_count = sum(variable.X for variable in scooter_model.getVars() if variable.VarName.startswith("SelectedLocation"))
    capacity_binding = [
        constraint
        for constraint in scooter_model.getConstrs()
        if constraint.ConstrName.startswith("LocationCapacity") and abs(constraint.Slack) <= 1e-6
    ]
    assert assignment_count == scooter.n_demand_points
    assert 1 <= selected_count <= scooter.max_selected_locations
    assert capacity_binding
    assert scooter.parameters["compact_scooter_location_tables"]
    assert scooter.parameters["compact_scooter_location_tables"]["demand_table"]["rows"]
    assert scooter.parameters["compact_scooter_location_tables"]["candidate_location_table"]["rows"]
    assert scooter.parameters["compact_scooter_location_tables"]["distance_table"]["rows"]
    compact_scooter_source = _compact_source_data({"generation_params": scooter.parameters})
    assert compact_scooter_source["available"] is True
    assert "compact_scooter_location_tables" in compact_scooter_source["value"]
    assert "new_scooters_only_at_selected_locations" in scooter.parameters["required_constraints"]
    assert any(variable.VarName.startswith("NewScooters") and variable.VType != "B" for variable in scooter_model.getVars())

    p_dispersion = PDispersion(seed=0)
    dispersion_model = p_dispersion.generate_instance()
    dispersion_model.optimize()
    assert dispersion_model.Status == 2
    selected_nodes = sum(variable.X for variable in dispersion_model.getVars() if variable.VarName.startswith("Nodes"))
    min_distance_binding = [
        constraint
        for constraint in dispersion_model.getConstrs()
        if constraint.ConstrName.startswith("MinDistance") and abs(constraint.Slack) <= 1e-6
    ]
    assert selected_nodes == p_dispersion.p
    assert p_dispersion.M <= p_dispersion.distance_range[1] + 1
    assert min_distance_binding
    assert p_dispersion.parameters["distances"]
    assert p_dispersion.parameters["candidate_pairs"]
    assert p_dispersion.parameters["compact_p_dispersion_tables"]
    compact_dispersion_source = _compact_source_data({"generation_params": p_dispersion.parameters})
    assert compact_dispersion_source["available"] is True
    assert "compact_p_dispersion_tables" in compact_dispersion_source["value"]
    assert "select_exactly_p_nodes" in p_dispersion.parameters["required_constraints"]
    assert "min_distance_upper_bound_for_every_selected_pair" in p_dispersion.parameters["required_constraints"]
    assert "state the exact number p of nodes to select" in p_dispersion.parameters["required_parameter_presentation"]
    assert "This is a p-dispersion model, not a capacitated facility-location model." in p_dispersion.parameters["business_interpretation_guardrails"]
    assert p_dispersion.parameters["structured_problem_data"]["parameters"]["p"] == p_dispersion.p
    assert "maximize the minimum pairwise distance" in p_dispersion.parameters["structured_problem_data"]["objective"]


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_batch3_flow_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.MCND.MCND import Generator as MCND
    from or_cpt_engine.generators.optmath_seed.SupplyChain.SupplyChain import Generator as SupplyChain
    from or_cpt_engine.generators.optmath_seed.The_shortest_path_problem.The_shortest_path_problem import (
        Generator as ShortestPath,
    )
    from or_cpt_engine.generators.optmath_seed.net1.net1_parsed import Generator as Net1
    from or_cpt_engine.generators.optmath_seed.multi.multi_parsed import Generator as Multi
    from or_cpt_engine.generators.optmath_seed.netmcol.netmcol_parsed import Generator as NetMCol
    from or_cpt_engine.generators.optmath_seed.nltrans.nltrans_parsed import Generator as NLTrans
    from or_cpt_engine.generators.optmath_seed.netthru.netthru_parsed import Generator as NetThru
    from or_cpt_engine.generators.optmath_seed.transp.transp_parsed import Generator as Transp

    mcnd = MCND(seed=0)
    mcnd_model = mcnd.generate_instance()
    mcnd_model.optimize()
    assert mcnd_model.Status == 2
    assert mcnd.parameters["compact_mcnd_tables"]
    assert mcnd.parameters["compact_mcnd_tables"]["commodity_table"]["rows"]
    assert mcnd.parameters["compact_mcnd_tables"]["arc_table"]["rows"]
    assert mcnd.parameters["compact_mcnd_tables"]["commodity_arc_cost_table"]["rows"]
    compact_mcnd_source = _compact_source_data({"generation_params": mcnd.parameters})
    assert compact_mcnd_source["available"] is True
    assert "compact_mcnd_tables" in compact_mcnd_source["value"]
    assert "nonnegative integer number of facilities" in mcnd.parameters["decision_variables"]["y[i,j]"]
    assert any(variable.VarName.startswith("y[") and variable.X > 1e-6 for variable in mcnd_model.getVars())
    assert any(variable.VarName.startswith("x[") and variable.X > 1e-6 for variable in mcnd_model.getVars())
    assert all(variable.VType != "B" for variable in mcnd_model.getVars() if variable.VarName.startswith("y["))
    assert any(variable.VType == "I" for variable in mcnd_model.getVars() if variable.VarName.startswith("y["))
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in mcnd_model.getConstrs())
    assert any(constraint.ConstrName.startswith("ArcCapacity") for constraint in mcnd_model.getConstrs())

    supply_chain = SupplyChain(seed=0)
    supply_chain_model = supply_chain.generate_instance()
    supply_chain_model.optimize()
    assert supply_chain_model.Status == 2
    assert supply_chain_model.NumVars < 80
    assert supply_chain_model.NumConstrs < 50
    assert supply_chain.parameters["arcs"]
    assert supply_chain.parameters["network_summary"]["sparse_declared_arc_set"] is True
    assert supply_chain.parameters["network_summary"]["arc_count"] == len(supply_chain.parameters["arcs"])
    assert supply_chain.parameters["network_summary"]["arc_count"] < supply_chain.parameters["n_nodes"] * (supply_chain.parameters["n_nodes"] - 1)
    assert supply_chain.parameters["arc_table"]
    assert supply_chain.parameters["node_roles"]
    assert supply_chain.parameters["node_balance_rhs"]
    assert "latent_feasible_flow" not in supply_chain.parameters
    assert supply_chain.parameters["network_summary"]["latent_feasible_arc_count_not_exposed"] > 0
    assert supply_chain.parameters["compact_supplychain_tables"]
    compact_supplychain_source = _compact_source_data({"generation_params": supply_chain.parameters})
    assert compact_supplychain_source["available"] is True
    assert "compact_supplychain_tables" in compact_supplychain_source["value"]
    assert supply_chain.parameters["structured_problem_data"]
    assert supply_chain.parameters["arc_capacities"]
    assert supply_chain.parameters["fixed_arc_activation_costs"]
    assert supply_chain.parameters["unit_arc_flow_costs"]
    assert supply_chain.parameters["total_supply"] == supply_chain.parameters["total_demand"]
    assert "arc_capacity_activation_linking" in supply_chain.parameters["required_constraints"]
    assert "List only the declared directed arcs; do not imply a complete graph." in supply_chain.parameters["required_parameter_presentation"]
    assert "This is a fixed-charge capacitated directed network-flow model." in supply_chain.parameters["business_interpretation_guardrails"]
    assert "Preserve the flow-balance sign convention: inflow minus outflow equals demand minus supply." in supply_chain.parameters["business_interpretation_guardrails"]
    assert any(variable.VarName.startswith("x[") and variable.X > 1e-6 for variable in supply_chain_model.getVars())
    assert any(variable.VarName.startswith("y[") and variable.X > 0.5 for variable in supply_chain_model.getVars())
    assert any(constraint.ConstrName.startswith("ArcCapacity") for constraint in supply_chain_model.getConstrs())
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in supply_chain_model.getConstrs())
    declared_arcs = {
        tuple(arc.split("|"))
        for arc in supply_chain.parameters["arcs"]
    }
    assert all(supply_chain_model.getVarByName(f"x[{i},{j}]") is not None for i, j in declared_arcs)
    for i, j in declared_arcs:
        flow = supply_chain_model.getVarByName(f"x[{i},{j}]").X
        active = supply_chain_model.getVarByName(f"y[{i},{j}]").X
        capacity = supply_chain.parameters["arc_capacities"][f"{i}|{j}"]
        assert flow <= capacity * active + 1e-6
    for node in supply_chain.parameters["nodes"]:
        inflow = sum(
            supply_chain_model.getVarByName(f"x[{i},{j}]").X
            for i, j in declared_arcs
            if j == node
        )
        outflow = sum(
            supply_chain_model.getVarByName(f"x[{i},{j}]").X
            for i, j in declared_arcs
            if i == node
        )
        assert abs(inflow - outflow - supply_chain.parameters["node_balance_rhs"][node]) <= 1e-6

    netthru = NetThru(seed=0)
    netthru_model = netthru.generate_instance()
    netthru_model.optimize()
    assert netthru_model.Status == 2
    assert netthru_model.NumVars < 40
    assert len(str(netthru.parameters)) < 8000
    assert all(variable.VType != "B" for variable in netthru_model.getVars())
    assert netthru.parameters["total_supply"] == netthru.parameters["total_demand"]
    assert netthru.parameters["supplies"]
    assert netthru.parameters["demands"]
    assert netthru.parameters["costs"]
    assert netthru.parameters["link_capacities"]
    assert netthru.parameters["city_capacities"]
    assert "latent_feasible_flow" not in netthru.parameters
    assert "latent_node_throughput" not in netthru.parameters
    assert "feasibility_certificate" not in netthru.parameters["structured_problem_data"]
    assert "no flow certificate should be included" in netthru.parameters["structured_problem_data"]["generation_note"]
    assert "node_flow_balance_supply_plus_inflow_equals_demand_plus_outflow" in netthru.parameters["required_constraints"]
    assert "node_throughput_capacity" in netthru.parameters["required_constraints"]
    netthru_guardrails = " ".join(netthru.parameters["business_interpretation_guardrails"])
    assert "binary link activation" in netthru_guardrails
    assert "fixed-charge" in netthru_guardrails
    assert any(variable.VarName.startswith("Shipping") and variable.X > 1e-6 for variable in netthru_model.getVars())
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in netthru_model.getConstrs())
    assert any(constraint.ConstrName.startswith("LinkCapacity") for constraint in netthru_model.getConstrs())
    assert any(constraint.ConstrName.startswith("CityCapacity") for constraint in netthru_model.getConstrs())
    for city in netthru.parameters["cities"]:
        inbound = sum(
            netthru_model.getVarByName(f"Shipping[{origin},{city}]").X
            for origin, dest in (tuple(link) for link in netthru.parameters["links"])
            if dest == city
        )
        outbound = sum(
            netthru_model.getVarByName(f"Shipping[{city},{dest}]").X
            for origin, dest in (tuple(link) for link in netthru.parameters["links"])
            if origin == city
        )
        residual = netthru.parameters["supplies"][city] + inbound - netthru.parameters["demands"][city] - outbound
        assert abs(residual) <= 1e-6
        assert netthru.parameters["supplies"][city] + inbound <= netthru.parameters["city_capacities"][city] + 1e-6
    for origin, dest in (tuple(link) for link in netthru.parameters["links"]):
        flow = netthru_model.getVarByName(f"Shipping[{origin},{dest}]").X
        assert flow <= netthru.parameters["link_capacities"][f"{origin}|{dest}"] + 1e-6

    nltrans = NLTrans(seed=0)
    nltrans_model = nltrans.generate_instance()
    nltrans_model.optimize()
    assert nltrans_model.Status == 2
    assert nltrans.parameters["total_supply"] == nltrans.parameters["total_demand"]
    assert all(value > 0 for value in nltrans.parameters["supply"].values())
    assert all(value > 0 for value in nltrans.parameters["demand"].values())
    assert nltrans.parameters["rates"]
    assert nltrans.parameters["lane_limits"]
    assert "latent_feasible_shipment" not in nltrans.parameters
    assert "feasibility_certificate" not in nltrans.parameters["structured_problem_data"]
    assert nltrans.parameters["compact_transportation_tables"]
    compact_nltrans_source = _compact_source_data({"generation_params": nltrans.parameters})
    assert compact_nltrans_source["available"] is True
    assert "compact_transportation_tables" in compact_nltrans_source["value"]
    assert "origin_supply_exactly_shipped" in nltrans.parameters["required_constraints"]
    assert "destination_demand_exactly_received" in nltrans.parameters["required_constraints"]
    assert "lane_capacity_limit" in nltrans.parameters["required_constraints"]
    assert "This is a capacitated balanced transportation LP, not a vehicle-routing problem." in nltrans.parameters["business_interpretation_guardrails"]
    assert all(variable.VType != "B" for variable in nltrans_model.getVars())
    assert any(variable.VarName.startswith("Shipping") and variable.X > 1e-6 for variable in nltrans_model.getVars())
    assert any(constraint.ConstrName.startswith("SupplyConstraint") for constraint in nltrans_model.getConstrs())
    assert any(constraint.ConstrName.startswith("DemandConstraint") for constraint in nltrans_model.getConstrs())
    assert any(constraint.ConstrName.startswith("LimitConstraint") for constraint in nltrans_model.getConstrs())
    for origin in nltrans.parameters["origins"]:
        shipped = sum(
            nltrans_model.getVarByName(f"Shipping[{origin},{destination}]").X
            for destination in nltrans.parameters["destinations"]
        )
        assert abs(shipped - nltrans.parameters["supply"][origin]) <= 1e-6
    for destination in nltrans.parameters["destinations"]:
        received = sum(
            nltrans_model.getVarByName(f"Shipping[{origin},{destination}]").X
            for origin in nltrans.parameters["origins"]
        )
        assert abs(received - nltrans.parameters["demand"][destination]) <= 1e-6

    transp = Transp(seed=0)
    transp_model = transp.generate_instance()
    transp_model.optimize()
    assert transp_model.Status == 2
    assert transp.parameters["total_supply"] == transp.parameters["total_demand"]
    assert transp.parameters["compact_transportation_tables"]["origin_supply_table"]
    assert transp.parameters["compact_transportation_tables"]["destination_demand_table"]
    assert transp.parameters["compact_transportation_tables"]["lane_cost_matrix"]["rows"]
    compact_transp_source = _compact_source_data({"generation_params": transp.parameters})
    assert compact_transp_source["available"] is True
    assert "compact_transportation_tables" in compact_transp_source["value"]
    assert "latent_feasible_shipment" not in transp.parameters
    assert "origin_supply_exactly_shipped" in transp.parameters["required_constraints"]
    assert "destination_demand_exactly_received" in transp.parameters["required_constraints"]
    transp_guardrails = " ".join(transp.parameters["business_interpretation_guardrails"])
    assert "vehicle-routing" in transp_guardrails
    assert "lane capacities" in transp_guardrails
    assert all(variable.VType != "B" for variable in transp_model.getVars())
    assert any(variable.VarName.startswith("Transport") and variable.X > 1e-6 for variable in transp_model.getVars())
    assert any(constraint.ConstrName.startswith("Supply") for constraint in transp_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Demand") for constraint in transp_model.getConstrs())
    for origin in transp.parameters["origins"]:
        shipped = sum(
            transp_model.getVarByName(f"Transport[{origin},{destination}]").X
            for destination in transp.parameters["destinations"]
        )
        assert abs(shipped - transp.parameters["supply"][origin]) <= 1e-6
    for destination in transp.parameters["destinations"]:
        received = sum(
            transp_model.getVarByName(f"Transport[{origin},{destination}]").X
            for origin in transp.parameters["origins"]
        )
        assert abs(received - transp.parameters["demand"][destination]) <= 1e-6

    netmcol = NetMCol(seed=0)
    netmcol_model = netmcol.generate_instance()
    netmcol_model.optimize()
    assert netmcol_model.Status == 2
    assert netmcol_model.NumVars < 120
    assert len(json.dumps(netmcol.parameters, ensure_ascii=False)) < 20000
    assert all(variable.VType != "B" for variable in netmcol_model.getVars())
    assert "latent_feasible_flow" not in netmcol.parameters
    assert "feasibility_certificate" not in netmcol.parameters["structured_problem_data"]
    assert netmcol.parameters["compact_netmcol_tables"]
    compact_netmcol_source = _compact_source_data({"generation_params": netmcol.parameters})
    assert compact_netmcol_source["available"] is True
    assert "compact_netmcol_tables" in compact_netmcol_source["value"]
    assert "city_product_flow_balance" in netmcol.parameters["required_constraints"]
    assert "directed_link_joint_capacity" in netmcol.parameters["required_constraints"]
    assert "directed_link_product_capacity" in netmcol.parameters["required_constraints"]
    assert "This is a continuous multi-commodity network flow LP, not a fixed-charge network design model." in netmcol.parameters["business_interpretation_guardrails"]
    for product in netmcol.parameters["products"]:
        assert netmcol.parameters["total_supply_by_product"][product] == netmcol.parameters["total_demand_by_product"][product]
    assert any(variable.VarName.startswith("Ship") and variable.X > 1e-6 for variable in netmcol_model.getVars())
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in netmcol_model.getConstrs())
    assert any(constraint.ConstrName.startswith("JointCapacity") for constraint in netmcol_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Capacity") for constraint in netmcol_model.getConstrs())
    for city in netmcol.parameters["cities"]:
        for product in netmcol.parameters["products"]:
            inbound = sum(
                netmcol_model.getVarByName(f"Ship[{origin},{city},{product}]").X
                for origin, dest in (tuple(link) for link in netmcol.parameters["links"])
                if dest == city
            )
            outbound = sum(
                netmcol_model.getVarByName(f"Ship[{city},{dest},{product}]").X
                for origin, dest in (tuple(link) for link in netmcol.parameters["links"])
                if origin == city
            )
            assert abs(netmcol.parameters["supply"][city][product] + inbound - netmcol.parameters["demand"][city][product] - outbound) <= 1e-6

    multi = Multi(seed=0)
    multi_model = multi.generate_instance()
    multi_model.optimize()
    assert multi_model.Status == 2
    assert multi_model.NumVars <= 105
    assert len(json.dumps(multi.parameters, ensure_ascii=False)) < 20000
    assert all(variable.VType != "B" for variable in multi_model.getVars())
    assert "latent_feasible_flow" not in multi.parameters
    assert "feasibility_certificate" not in multi.parameters["structured_problem_data"]
    assert multi.parameters["compact_multi_commodity_transportation_tables"]
    compact_multi_source = _compact_source_data({"generation_params": multi.parameters})
    assert compact_multi_source["available"] is True
    assert "compact_multi_commodity_transportation_tables" in compact_multi_source["value"]
    assert multi.parameters["shipping_cost"]
    assert multi.parameters["lane_limits"]
    assert "origin_product_supply_exactly_shipped" in multi.parameters["required_constraints"]
    assert "destination_product_demand_exactly_received" in multi.parameters["required_constraints"]
    assert "origin_destination_joint_capacity_across_products" in multi.parameters["required_constraints"]
    assert "state lane capacity is shared across all products on the same origin-destination lane" in multi.parameters[
        "required_parameter_presentation"
    ]
    assert "This is a continuous multi-commodity transportation LP." in multi.parameters["business_interpretation_guardrails"]
    for product in multi.parameters["products"]:
        assert multi.parameters["total_supply_by_product"][product] == multi.parameters["total_demand_by_product"][product]
    assert any(variable.VarName.startswith("Transport") and variable.X > 1e-6 for variable in multi_model.getVars())
    assert any(constraint.ConstrName.startswith("Supply") for constraint in multi_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Demand") for constraint in multi_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Limit") for constraint in multi_model.getConstrs())
    for origin in multi.parameters["origins"]:
        for product in multi.parameters["products"]:
            shipped = sum(
                multi_model.getVarByName(f"Transport[{origin},{destination},{product}]").X
                for destination in multi.parameters["destinations"]
            )
            assert abs(shipped - multi.parameters["supply"][origin][product]) <= 1e-6
    for destination in multi.parameters["destinations"]:
        for product in multi.parameters["products"]:
            received = sum(
                multi_model.getVarByName(f"Transport[{origin},{destination},{product}]").X
                for origin in multi.parameters["origins"]
            )
            assert abs(received - multi.parameters["demand"][destination][product]) <= 1e-6
    for origin in multi.parameters["origins"]:
        for destination in multi.parameters["destinations"]:
            lane_total = sum(
                multi_model.getVarByName(f"Transport[{origin},{destination},{product}]").X
                for product in multi.parameters["products"]
            )
            assert lane_total <= multi.parameters["lane_limits"][origin][destination] + 1e-6

    net1 = Net1(seed=0)
    net1_model = net1.generate_instance()
    net1_model.optimize()
    assert net1_model.Status == 2
    assert net1_model.NumVars < 80
    assert len(str(net1.parameters)) < 8000
    assert all(variable.VType != "B" for variable in net1_model.getVars())
    assert net1.parameters["latent_feasible_flow"]
    assert net1.parameters["total_supply"] == net1.parameters["total_demand"]
    assert net1.parameters["compact_network_flow_tables"]["node_balance_table"]
    assert net1.parameters["compact_network_flow_tables"]["directed_arc_table"]
    compact_net1_source = _compact_source_data({"generation_params": net1.parameters})
    assert compact_net1_source["available"] is True
    assert "compact_network_flow_tables" in compact_net1_source["value"]
    assert "node_flow_balance_supply_plus_inflow_equals_demand_plus_outflow" in net1.parameters["required_constraints"]
    assert "directed_arc_capacity_limit" in net1.parameters["required_constraints"]
    assert "This is a continuous capacitated min-cost network flow LP, not facility location or vehicle routing." in net1.parameters["business_interpretation_guardrails"]
    assert any(variable.VarName.startswith("Ship") and variable.X > 1e-6 for variable in net1_model.getVars())
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in net1_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Capacity") for constraint in net1_model.getConstrs())
    for city in net1.parameters["cities"]:
        inbound = sum(
            net1_model.getVarByName(f"Ship[{origin},{city}]").X
            for origin, dest in (tuple(link) for link in net1.parameters["links"])
            if dest == city
        )
        outbound = sum(
            net1_model.getVarByName(f"Ship[{city},{dest}]").X
            for origin, dest in (tuple(link) for link in net1.parameters["links"])
            if origin == city
        )
        assert abs(net1.parameters["supply"][city] + inbound - net1.parameters["demand"][city] - outbound) <= 1e-6
    for origin, dest in (tuple(link) for link in net1.parameters["links"]):
        flow = net1_model.getVarByName(f"Ship[{origin},{dest}]").X
        assert flow <= net1.parameters["capacity"][f"{origin}|{dest}"] + 1e-6

    for seed in range(10):
        shortest_path = ShortestPath(seed=seed)
        shortest_path_model = shortest_path.generate_instance()
        shortest_path_model.optimize()
        assert shortest_path_model.Status == 2
        arc_variables = [variable for variable in shortest_path_model.getVars() if variable.VarName.startswith("Arcs")]
        selected_arcs = [variable for variable in arc_variables if variable.X > 0.5]
        assert len(arc_variables) > len(selected_arcs)
        assert len(selected_arcs) >= 2
        assert shortest_path.parameters["compact_shortest_path_tables"]
        compact_shortest_path_source = _compact_source_data({"generation_params": shortest_path.parameters})
        assert compact_shortest_path_source["available"] is True
        assert "compact_shortest_path_tables" in compact_shortest_path_source["value"]
        assert "source_sink_flow_balance" in shortest_path.parameters["required_constraints"]
        assert "Do not expand the graph to a complete directed network." in shortest_path.parameters[
            "business_interpretation_guardrails"
        ]


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_batch4_selection_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.Binpacking_Problem.Binpacking import Generator as BinPacking
    from or_cpt_engine.generators.optmath_seed.KnapSack.Knapsack import Generator as Knapsack
    from or_cpt_engine.generators.optmath_seed.MultiSetCover.SetMultiCover import Generator as MultiSetCover
    from or_cpt_engine.generators.optmath_seed.Portfolio.Portfolio import Generator as Portfolio
    from or_cpt_engine.generators.optmath_seed.SetCover.SetCover import Generator as SetCover
    from or_cpt_engine.generators.optmath_seed.The_maxisum_model.The_maxisum_model import Generator as Maxisum

    for seed in range(5):
        knapsack = Knapsack(seed=seed)
        knapsack_model = knapsack.generate_instance()
        knapsack_model.optimize()
        capacity = next(constraint for constraint in knapsack_model.getConstrs() if constraint.ConstrName == "WeightCapacity")
        selected_items = sum(variable.X > 0.5 for variable in knapsack_model.getVars())
        assert knapsack_model.Status == 2
        assert abs(capacity.Slack) <= 1e-6
        assert 2 <= selected_items <= knapsack.n_items - 2

    binpacking = BinPacking(seed=0)
    binpacking_model = binpacking.generate_instance()
    binpacking_model.optimize()
    used_bins = sum(variable.X > 0.5 for variable in binpacking_model.getVars() if variable.VarName.startswith("y["))
    assert binpacking_model.Status == 2
    assert 2 <= used_bins <= binpacking.n_items - 1
    assert any(constraint.ConstrName.startswith("UsedBinOrder") for constraint in binpacking_model.getConstrs())

    for generator_cls in (SetCover, MultiSetCover):
        for seed in range(5):
            generator = generator_cls(seed=seed)
            model = generator.generate_instance()
            model.optimize()
            selected_sets = sum(variable.X > 0.5 for variable in model.getVars() if variable.VarName.startswith("Selected"))
            assert model.Status == 2
            assert 2 <= selected_sets <= generator.n_sets - 1
            if generator_cls is MultiSetCover:
                assert generator.parameters["set_table"]
                assert generator.parameters["element_coverage_table"]
                assert generator.parameters["coverage_matrix_summary"]
                assert generator.parameters["compact_multisetcover_tables"]
                compact_multisetcover_source = _compact_source_data({"generation_params": generator.parameters})
                assert compact_multisetcover_source["available"] is True
                assert "compact_multisetcover_tables" in compact_multisetcover_source["value"]
                assert generator.parameters["structured_problem_data"]
                assert len(json.dumps(generator.parameters, ensure_ascii=False)) < 8000
                assert "element_multicover_requirement" in generator.parameters["required_constraints"]
                assert "state requirements are lower bounds, not exact coverage counts" in generator.parameters[
                    "required_parameter_presentation"
                ]
                assert "Do not replace >= required coverage with exact equality." in generator.parameters[
                    "business_interpretation_guardrails"
                ]
                for element, row in generator.parameters["element_coverage_table"].items():
                    assert row["required_coverage_count"] < row["available_covering_set_count"], element
                assert generator.parameters["coverage_matrix_summary"][
                    "all_elements_have_requirement_below_available_count"
                ] is True
                assert generator.parameters["coverage_matrix_summary"]["density"] == round(
                    generator.parameters["coverage_matrix_summary"]["density"],
                    2,
                )

    portfolio = Portfolio(seed=0)
    portfolio_model = portfolio.generate_instance()
    portfolio_model.optimize()
    assert portfolio_model.Status == 2
    assert portfolio_model.NumVars <= 20
    assert portfolio_model.getVarByName("PortfolioVariance") is None
    assert all(variable.VType != "B" for variable in portfolio_model.getVars())
    assert 6 <= portfolio.n_assets <= 10
    assert len(portfolio.parameters["covariance"]) == portfolio.n_assets * portfolio.n_assets
    assert portfolio.parameters["compact_portfolio_tables"]["covariance_matrix"]
    compact_portfolio_source = _compact_source_data({"generation_params": portfolio.parameters})
    assert compact_portfolio_source["available"] is True
    assert "compact_portfolio_tables" in compact_portfolio_source["value"]
    assert "portfolio_variance_quadratic_form" in portfolio.parameters["objective_terms"]
    assert "positive_semidefinite_covariance_risk_objective" in portfolio.parameters["required_constraints"]
    assert "This is a continuous mean-variance allocation model, not a binary asset-selection model." in portfolio.parameters["business_interpretation_guardrails"]
    assert any(constraint.ConstrName == "Budget" for constraint in portfolio_model.getConstrs())
    assert any(constraint.ConstrName == "TargetReturn" for constraint in portfolio_model.getConstrs())

    for seed in range(5):
        maxisum = Maxisum(seed=seed)
        maxisum_model = maxisum.generate_instance()
        maxisum_model.optimize()
        selected_nodes = sum(variable.X > 0.5 for variable in maxisum_model.getVars() if variable.VarName.startswith("x["))
        active_pair_vars = sum(variable.X > 0.5 for variable in maxisum_model.getVars() if variable.VarName.startswith("z["))
        assert maxisum_model.Status == 2
        assert selected_nodes == maxisum.p_facilities
        assert maxisum.p_facilities <= maxisum.n_nodes - 2
        assert active_pair_vars < sum(1 for variable in maxisum_model.getVars() if variable.VarName.startswith("z["))


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_marketshare_generator_uses_compact_profit_matrices() -> None:
    from or_cpt_engine.generators.optmath_seed.MarketShare.MarketShare import Generator as MarketShare

    generator = MarketShare(seed=0)
    model = generator.generate_instance()
    model.optimize()
    assert model.Status == 2

    compact_tables = generator.parameters["compact_marketshare_tables"]
    assert compact_tables["demand_matrix"]["rows"]
    assert compact_tables["unit_profit_matrices_by_company"]["rows"]
    assert "profit_table" not in compact_tables
    assert compact_tables["profit_table_format"].startswith("see unit_profit_matrices_by_company")
    assert generator.parameters["structured_problem_data"]["parameters"]["company_market_product_profit"].endswith(
        "unit_profit_matrices_by_company"
    )

    compact_source = _compact_source_data({"generator_id": "optmath_marketshare", "generation_params": generator.parameters})
    assert compact_source["available"] is True
    assert compact_source["truncated"] is False
    prompt = render_forward_prompt(
        "{{PROBLEM_JSON}}",
        {
            "generator_id": "optmath_marketshare",
            "source_compact_data": compact_source,
            "concept_tags": ["integer_supply_allocation"],
        },
    )
    digest = "\n".join(json.loads(prompt)["source_fact_digest"])
    assert "marketshare_unit_profit_matrix" in digest
    assert "truncated_for_forward_prompt" in prompt


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_batch5_lotsizing_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.CLSP_expand_capacity.CLSP_expand_capacity_parsed import (
        Generator as CLSPExpandCapacity,
    )
    from or_cpt_engine.generators.optmath_seed.Factory_Planning_Problem.Factory_Planning_Problem import (
        Generator as FactoryPlanning,
    )
    from or_cpt_engine.generators.optmath_seed.SingleLevelSmallBucket.SingleLevelSmallBucket_parsed import (
        Generator as SingleLevelSmallBucket,
    )
    from or_cpt_engine.generators.optmath_seed.UncapacitatedLotSizing.UncapacitatedLotSizing_parsed import (
        Generator as UncapacitatedLotSizing,
    )
    from or_cpt_engine.generators.optmath_seed.UncapacitatedLotSizingBacklogging.UncapacitatedLotSizingBacklogging_parsed import (
        Generator as UncapacitatedLotSizingBacklogging,
    )
    from or_cpt_engine.generators.optmath_seed.prod.prod_parsed import Generator as ProductMix

    for seed in range(10):
        product_mix = ProductMix(seed=seed)
        product_mix_model = product_mix.generate_instance()
        product_mix_model.optimize()
        resource = next(constraint for constraint in product_mix_model.getConstrs() if constraint.ConstrName == "ResourceConstraint")
        active_products = sum(variable.X > 1e-6 for variable in product_mix_model.getVars() if variable.VarName.startswith("X["))
        assert product_mix_model.Status == 2
        assert active_products >= 2
        assert abs(resource.Slack) <= 1e-6
        assert product_mix.parameters["target_active_item_count"] >= 2

    factory = FactoryPlanning(seed=0)
    factory_model = factory.generate_instance()
    factory_model.optimize()
    assert factory_model.Status == 2
    assert any(variable.VarName.startswith("Production") and variable.X > 1e-6 for variable in factory_model.getVars())
    assert any(constraint.ConstrName.startswith("Balance") for constraint in factory_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Capacity") for constraint in factory_model.getConstrs())
    assert "[Previous mathematical formulation here]" not in factory.mathematical_formulation
    assert factory.parameters["products"]
    assert factory.parameters["machines"]
    assert factory.parameters["profits"]
    assert factory.parameters["machine_time"]
    assert factory.parameters["sales_limits"]
    assert factory.parameters["available_machine_hours"]
    assert factory.parameters["compact_factory_planning_tables"]["product_table"]
    assert factory.parameters["compact_factory_planning_tables"]["machine_product_processing_hours"]
    assert factory.parameters["compact_factory_planning_tables"]["period_machine_available_hours"]
    assert factory.parameters["compact_factory_planning_tables"]["period_product_sales_upper_bounds"]
    assert factory.parameters["target_inventory"]
    assert factory.parameters["structured_problem_data"]["parameters"]["sales_upper_bounds"]
    assert factory.parameters["structured_problem_data"]["parameters"]["machine_time_hours_per_unit"]
    assert factory.parameters["structured_problem_data"]["parameters"]["available_machine_hours_after_maintenance"]
    assert factory.parameters["decision_variables"]["Production[t,p]"].startswith("continuous units")
    assert "list sales upper bound U[t,p] for every period-product pair" in factory.parameters["required_parameter_presentation"]
    assert "prefer compact_factory_planning_tables when writing the natural-language data tables" in factory.parameters["required_parameter_presentation"]
    assert "machine_hour_capacity_after_maintenance" in factory.parameters["required_constraints"]
    assert "sales_upper_bound_by_period_product" in factory.parameters["required_constraints"]
    assert "required_final_inventory_target" in factory.parameters["required_constraints"]
    assert "Do not aggregate sales limits into one total demand number per product." in factory.parameters["business_interpretation_guardrails"]
    assert (
        "Do not infer demand totals from the reference answer; use the listed product-period sales upper bounds."
        in factory.parameters["business_interpretation_guardrails"]
    )
    assert "Do not replace machine-hour capacity with budgets, workforce headcount, service coverage, or site capacity." in factory.parameters["business_interpretation_guardrails"]
    assert any(
        abs(constraint.Slack) <= 1e-6
        for constraint in factory_model.getConstrs()
        if constraint.ConstrName.startswith("Capacity")
    )
    for product, target in factory.parameters["target_inventory"].items():
        final_inventory = factory_model.getVarByName(f"Inventory[{factory.n_periods - 1},{product}]")
        assert abs(final_inventory.X - target) <= 1e-6

    clsp = CLSPExpandCapacity(seed=0)
    clsp_model = clsp.generate_instance()
    clsp_model.optimize()
    assert clsp_model.Status == 2
    assert any(variable.VarName.startswith("Production") and variable.X > 1e-6 for variable in clsp_model.getVars())
    assert any(variable.VarName.startswith("Setup") and variable.X > 0.5 for variable in clsp_model.getVars())
    assert any(constraint.ConstrName.startswith("InventoryBalance") for constraint in clsp_model.getConstrs())
    assert clsp.parameters["compact_lotsizing_tables"]["period_demand_table"]
    assert clsp.parameters["compact_lotsizing_tables"]["source_model_note"].endswith(
        "There are no backlog variables, no backlog penalties, and no capacity-expansion decision variables."
    )
    compact_clsp_source = _compact_source_data({"generation_params": clsp.parameters})
    assert compact_clsp_source["available"] is True
    assert "compact_lotsizing_tables" in compact_clsp_source["value"]
    assert "backlog_penalty_cost" not in clsp.parameters["objective_terms"]

    for generator_cls in (UncapacitatedLotSizing, UncapacitatedLotSizingBacklogging):
        generator = generator_cls(seed=0)
        model = generator.generate_instance()
        model.optimize()
        assert model.Status == 2
        assert any(variable.VarName.startswith("OrderedAmount") and variable.X > 1e-6 for variable in model.getVars())
        assert any(variable.VarName.startswith("OrderIsPlaced") and variable.X > 0.5 for variable in model.getVars())
        assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in model.getConstrs())

    uls = UncapacitatedLotSizing(seed=0)
    uls_model = uls.generate_instance()
    uls_model.optimize()
    assert uls_model.Status == 2
    assert uls.parameters["compact_lotsizing_tables"]["model_family"] == "uncapacitated_lot_sizing_without_backlog"
    assert uls.parameters["compact_lotsizing_tables"]["period_cost_table"] == uls.parameters["period_cost_table"]
    assert uls.parameters["compact_lotsizing_tables"]["total_demand_big_m"] == sum(uls.parameters["demands"].values())
    compact_uls_source = _compact_source_data({"generation_params": uls.parameters})
    assert compact_uls_source["available"] is True
    assert "compact_lotsizing_tables" in compact_uls_source["value"]
    assert "inventory_balance_without_backlog" in uls.parameters["required_constraints"]
    assert "BackloggedAmount[t]" not in uls.parameters["decision_variables"]
    assert any("without backlog" in item for item in uls.parameters["business_interpretation_guardrails"])
    assert not any(variable.VarName.startswith("BackloggedAmount") for variable in uls_model.getVars())

    backlog = UncapacitatedLotSizingBacklogging(seed=0)
    backlog_model = backlog.generate_instance()
    backlog_model.optimize()
    assert backlog_model.Status == 2
    first_period = backlog.parameters["periods"][0]
    x_first = backlog_model.getVarByName(f"OrderedAmount[{first_period}]")
    i_first = backlog_model.getVarByName(f"EndingInventory[{first_period}]")
    b_first = backlog_model.getVarByName(f"BackloggedAmount[{first_period}]")
    assert abs(x_first.X - (i_first.X + backlog.parameters["demands"][first_period] - b_first.X)) <= 1e-6
    assert backlog.parameters["initial_inventory"] == 0
    assert backlog.parameters["initial_backlog"] == 0
    assert backlog.parameters["required_final_inventory"] == 0
    assert backlog.parameters["required_final_backlog"] == 0
    assert backlog.parameters["period_cost_table"]
    assert backlog.parameters["compact_lotsizing_tables"]["period_cost_table"] == backlog.parameters["period_cost_table"]
    assert backlog.parameters["compact_lotsizing_tables"]["total_demand_big_m"] == sum(backlog.parameters["demands"].values())
    compact_backlog_source = _compact_source_data({"generation_params": backlog.parameters})
    assert compact_backlog_source["available"] is True
    assert "compact_lotsizing_tables" in compact_backlog_source["value"]
    assert backlog.parameters["balance_transition_by_period"]
    assert backlog.parameters["setup_linking_by_period"]
    assert backlog.parameters["structured_problem_data"]
    assert backlog.parameters["decision_variables"]["BackloggedAmount[t]"].startswith("continuous")
    assert "state net inventory is ending inventory minus backlog" in backlog.parameters["required_parameter_presentation"]
    assert "net_inventory_balance_with_backlog_carryover" in backlog.parameters["required_constraints"]
    assert "This is uncapacitated lot sizing with backlogging; there is no production capacity constraint." in backlog.parameters["business_interpretation_guardrails"]
    assert backlog_model.getConstrByName("StartingInventoryZero") is None
    assert backlog_model.getConstrByName("StartingBacklogZero") is None

    small_bucket = SingleLevelSmallBucket(seed=0)
    small_bucket_model = small_bucket.generate_instance()
    small_bucket_model.optimize()
    assert small_bucket_model.Status == 2
    assert small_bucket.parameters["compact_lotsizing_tables"]["demand_matrix"]["rows"]
    assert "unit_production_cost" not in small_bucket.parameters["compact_lotsizing_tables"]["global_costs"]
    assert "per-unit production cost" in small_bucket.parameters["compact_lotsizing_tables"]["global_costs"]["omitted_objective_terms"]
    compact_small_bucket_source = _compact_source_data({"generation_params": small_bucket.parameters})
    assert compact_small_bucket_source["available"] is True
    assert compact_small_bucket_source["truncated"] is False
    assert "compact_lotsizing_tables" in compact_small_bucket_source["value"]
    assert "production_capacity_with_startup_time" in small_bucket.parameters["required_constraints"]
    assert any(constraint.ConstrName.startswith("FlowBalance") for constraint in small_bucket_model.getConstrs())
    assert any(constraint.ConstrName.startswith("Capacity") for constraint in small_bucket_model.getConstrs())
    assert any(constraint.ConstrName.startswith("OneItemPerMachinePeriod") for constraint in small_bucket_model.getConstrs())


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_batch6_routing_scheduling_assignment_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.AircraftAssignment.AircraftAssignment import (
        Generator as AircraftAssignment,
    )
    from or_cpt_engine.generators.optmath_seed.AircraftLanding.AircraftLanding import Generator as AircraftLanding
    from or_cpt_engine.generators.optmath_seed.Flowshop_2.Flowshop_2_parsed import Generator as FlowShop
    from or_cpt_engine.generators.optmath_seed.JopShop.generator import Generator as JobShop
    from or_cpt_engine.generators.optmath_seed.SchedulingProblem.SchedulingProblem_parsed import (
        Generator as StaffScheduling,
    )
    from or_cpt_engine.generators.optmath_seed.Structure_based_assignment.Structure_based_assignment import (
        Generator as StructureBasedAssignment,
    )
    from or_cpt_engine.generators.optmath_seed.TSP.TSP import Generator as TSP
    from or_cpt_engine.generators.optmath_seed.VRPTW.VRPTW_parsed import Generator as VRPTW
    from or_cpt_engine.generators.optmath_seed.fleet_routing.fleet_routing_parsed import Generator as FleetRouting
    from or_cpt_engine.generators.optmath_seed.netasgn.netasgn_parsed import Generator as NetAssignment
    from or_cpt_engine.generators.optmath_seed.team_formulation.team_formulation_parsed import (
        Generator as TeamFormulation,
    )

    for seed in range(3):
        tsp = TSP(seed=seed)
        tsp_model = tsp.generate_instance()
        tsp_model.optimize()
        assert tsp_model.Status == 2
        assert tsp_model.NumVars < 90
        assert tsp.parameters["compact_tsp_tables"]
        assert tsp.parameters["compact_tsp_tables"]["distance_table"]["rows"]
        compact_tsp_source = _compact_source_data({"generation_params": tsp.parameters})
        assert compact_tsp_source["available"] is True
        assert "compact_tsp_tables" in compact_tsp_source["value"]
        assert "single_tour_mtz_subtour_elimination" in tsp.parameters["required_constraints"]
        assert "Do not add vehicle capacity, time windows, service times, multiple vehicles, or delivery quantities." in tsp.parameters[
            "business_interpretation_guardrails"
        ]
        assert any(variable.VarName.startswith("route") and variable.VType == "B" for variable in tsp_model.getVars())
        assert any(variable.VarName.startswith("u[") for variable in tsp_model.getVars())
        assert any(constraint.ConstrName.startswith("MTZSubtour") for constraint in tsp_model.getConstrs())

        aircraft_assignment = AircraftAssignment(seed=seed)
        aircraft_assignment_model = aircraft_assignment.generate_instance()
        aircraft_assignment_model.optimize()
        assert aircraft_assignment_model.Status == 2
        assert aircraft_assignment_model.NumVars <= 60
        assert len(str(aircraft_assignment.parameters)) < 8000
        assert all(variable.VType != "B" for variable in aircraft_assignment_model.getVars())
        assert any(variable.VarName.startswith("Allocation") and variable.X > 1e-6 for variable in aircraft_assignment_model.getVars())
        assert aircraft_assignment.parameters["compact_aircraft_assignment_tables"]["aircraft_type_table"]
        assert aircraft_assignment.parameters["compact_aircraft_assignment_tables"]["route_demand_table"]
        assert aircraft_assignment.parameters["compact_aircraft_assignment_tables"]["capacity_contribution_matrix"]["rows"]
        assert aircraft_assignment.parameters["compact_aircraft_assignment_tables"]["operating_cost_matrix"]["rows"]
        compact_aircraft_assignment_source = _compact_source_data({"generation_params": aircraft_assignment.parameters})
        assert compact_aircraft_assignment_source["available"] is True
        assert "compact_aircraft_assignment_tables" in compact_aircraft_assignment_source["value"]
        assert "hidden_feasible_allocation" not in aircraft_assignment.parameters
        assert "hidden_feasible_route_capacity" not in aircraft_assignment.parameters
        assert "route_capacity_meets_or_exceeds_demand" in aircraft_assignment.parameters["required_constraints"]
        assert "This is an integer fleet-to-route allocation model, not a binary one-aircraft-to-one-route assignment model." in aircraft_assignment.parameters["business_interpretation_guardrails"]
        for route in aircraft_assignment.routes:
            provided_capacity = sum(
                aircraft_assignment_model.getVarByName(f"Allocation[{aircraft},{route}]").X
                * aircraft_assignment.capabilities[aircraft, route]
                for aircraft in aircraft_assignment.aircraft
            )
            assert provided_capacity + 1e-6 >= aircraft_assignment.demand[route]
        assert any(
            abs(constraint.Slack) <= 1e-6
            for constraint in aircraft_assignment_model.getConstrs()
            if constraint.ConstrName.startswith("Demand_")
        )
        for aircraft in aircraft_assignment.aircraft:
            used_count = sum(
                aircraft_assignment_model.getVarByName(f"Allocation[{aircraft},{route}]").X
                for route in aircraft_assignment.routes
            )
            assert used_count <= aircraft_assignment.availability[aircraft] + 1e-6

        aircraft_landing = AircraftLanding(seed=seed)
        aircraft_landing_model = aircraft_landing.generate_instance()
        aircraft_landing_model.optimize()
        assert aircraft_landing_model.Status == 2
        assert aircraft_landing_model.NumVars < 160
        assert len(str(aircraft_landing.parameters)) < 8000
        assert aircraft_landing.parameters["compact_aircraft_landing_tables"]["aircraft_time_penalty_table"]
        assert aircraft_landing.parameters["compact_aircraft_landing_tables"]["ordered_pair_separation_matrix"]["rows"]
        compact_aircraft_landing_source = _compact_source_data({"generation_params": aircraft_landing.parameters})
        assert compact_aircraft_landing_source["available"] is True
        assert "compact_aircraft_landing_tables" in compact_aircraft_landing_source["value"]
        assert "big_m" not in aircraft_landing.parameters
        assert "minimum_separation_time_between_ordered_landings" in aircraft_landing.parameters["required_constraints"]
        assert "This is a static aircraft landing sequencing model, not fleet assignment or aircraft routing." in aircraft_landing.parameters["business_interpretation_guardrails"]
        assert any(variable.VarName.startswith("AircraftOrder") and variable.VType == "B" for variable in aircraft_landing_model.getVars())
        assert any(variable.VarName.startswith("Landing") for variable in aircraft_landing_model.getVars())
        assert any(constraint.ConstrName.startswith("OrderPair") for constraint in aircraft_landing_model.getConstrs())
        assert any(constraint.ConstrName.startswith("Separation") for constraint in aircraft_landing_model.getConstrs())
        for aircraft in aircraft_landing.aircraft:
            landing_time = aircraft_landing_model.getVarByName(f"Landing[{aircraft}]").X
            assert aircraft_landing.earliest_landing[aircraft] - 1e-6 <= landing_time <= aircraft_landing.latest_landing[aircraft] + 1e-6

        assignment = StructureBasedAssignment(seed=seed)
        assignment_model = assignment.generate_instance()
        assignment_model.optimize()
        assert assignment_model.Status == 2
        assert assignment_model.NumVars < 500
        assert assignment_model.NumConstrs < 1000
        assert any(constraint.ConstrName.startswith("NOE_") for constraint in assignment_model.getConstrs())
        assert assignment.parameters["compact_structure_assignment_tables"]["assignment_cost_matrix"]["rows"]
        assert assignment.parameters["compact_structure_assignment_tables"]["acid_compatibility_matrix"]["rows"]
        assert assignment.parameters["compact_structure_assignment_tables"]["noe_relation_pairs"]
        assert assignment.parameters["compact_structure_assignment_tables"]["required_assignment_count"] == assignment.n_assignments
        assert assignment.parameters["compact_structure_assignment_tables"]["distance_threshold"] == assignment.nth
        compact_structure_source = _compact_source_data({"generation_params": assignment.parameters})
        assert compact_structure_source["available"] is True
        assert compact_structure_source["truncated"] is False
        assert "compact_structure_assignment_tables" in compact_structure_source["value"]
        assert "exact_total_assignment_count" in assignment.parameters["required_constraints"]

        fleet = FleetRouting(seed=seed)
        fleet_model = fleet.generate_instance()
        fleet_model.optimize()
        assert fleet_model.Status == 2
        assert fleet_model.NumVars < 1000
        assert fleet_model.NumConstrs < 1000
        assert fleet.parameters["active_flight_legs"]
        assert fleet.parameters["active_leg_table"]
        assert fleet.parameters["compact_fleet_flow_tables"]["feasible_leg_table"]
        assert fleet.parameters["passenger_demand"]
        assert set(fleet.parameters["passenger_demand"]) == set(fleet.parameters["active_flight_legs"])
        assert fleet.parameters["capacity_by_fleet_type"]
        assert fleet.parameters["available_fleet_by_type"]
        assert fleet.parameters["fleet_type_table"]
        assert fleet.parameters["time_expanded_network_summary"]["active_leg_count"] <= 2
        assert fleet.parameters["structured_problem_data"]
        assert "time_expanded_fleet_flow_conservation" in fleet.parameters["required_constraints"]
        fleet_required_presentation = [item.lower() for item in fleet.parameters["required_parameter_presentation"]]
        assert "state all decision variables are nonnegative integer counts, not binary visit variables." in fleet_required_presentation
        assert "Do not require binary route-arc variables; the decision is the integer number of fleet units assigned to each active leg." in fleet.parameters["business_interpretation_guardrails"]
        assert "Do not omit idle fleet variables or initial fleet placement variables; they are required for time-expanded conservation." in fleet.parameters["business_interpretation_guardrails"]
        compact_fleet_source = _compact_source_data({"generation_params": fleet.parameters})
        assert compact_fleet_source["available"] is True
        assert "compact_fleet_flow_tables" in compact_fleet_source["value"]
        assert all(variable.VType != "B" for variable in fleet_model.getVars())
        assert any(variable.VarName.startswith("NumPlanes") and variable.X > 1e-6 for variable in fleet_model.getVars())
        assert any(variable.VarName.startswith("NumIdlePlanes") for variable in fleet_model.getVars())
        assert any(variable.VarName.startswith("NumIdlePlanesInit") for variable in fleet_model.getVars())
        assert any(constraint.ConstrName.startswith("DemandSatisfaction") for constraint in fleet_model.getConstrs())
        assert any(constraint.ConstrName.startswith("RouteRestriction") for constraint in fleet_model.getConstrs())

        staff = StaffScheduling(seed=seed)
        staff_model = staff.generate_instance()
        staff_model.optimize()
        assert staff_model.Status == 2
        assert staff.parameters["total_staff_demand"] > 0
        assert staff.parameters["demand"]
        assert staff.parameters["positive_demand"]
        assert staff.parameters["employee_has_skill"]
        assert staff.parameters["employee_availability"]
        assert staff.parameters["preference_cost"]
        assert staff.parameters["structured_problem_data"]
        assert len(str(staff.parameters)) < 8000
        assert "coverage_with_integer_shortage_slack" in staff.parameters["required_constraints"]
        assert "list only positive site-shift-skill demand and state omitted combinations have zero demand" in staff.parameters["required_parameter_presentation"]
        assert "This is a staff coverage assignment model, not a machine sequencing model." in staff.parameters["business_interpretation_guardrails"]
        assert any(variable.VarName.startswith("Assignment") and variable.X > 0.5 for variable in staff_model.getVars())
        assert any(variable.VarName.startswith("Unfulfilled") for variable in staff_model.getVars())
        assert any(constraint.ConstrName.startswith("Demand_") for constraint in staff_model.getConstrs())
        assert any(constraint.ConstrName.startswith("ShiftAvail_") for constraint in staff_model.getConstrs())
        assert any(constraint.ConstrName.startswith("SkillReq_") for constraint in staff_model.getConstrs())
        for employee in staff.parameters["employees"]:
            assigned = sum(
                variable.X
                for variable in staff_model.getVars()
                if variable.VarName.startswith("Assignment[") and f",{employee}," in variable.VarName
            )
            assert assigned <= 1 + 1e-6

        flow_shop = FlowShop(seed=seed)
        flow_shop_model = flow_shop.generate_instance()
        flow_shop_model.optimize()
        assert flow_shop_model.Status == 2
        assert flow_shop.parameters["processing_times"]
        assert flow_shop.parameters["processing_time_table_by_job"]
        assert flow_shop.parameters["processing_time_table_by_machine"]
        assert flow_shop.parameters["structured_problem_data"]
        assert "lp_objective_note" in flow_shop.parameters["objective_terms"]
        assert "MakespanCompletion" in flow_shop.parameters["objective_terms"]["lp_objective_note"]
        assert "one_job_per_sequence_position" in flow_shop.parameters["required_constraints"]
        assert "same_sequence_used_on_every_machine" in flow_shop.parameters["required_constraints"]
        assert "state every machine uses the same job sequence" in flow_shop.parameters["required_parameter_presentation"]
        assert "This is a permutation flow shop: all machines use the same job sequence." in flow_shop.parameters["business_interpretation_guardrails"]
        assert "Do not allow each machine to choose an independent job sequence." in flow_shop.parameters["business_interpretation_guardrails"]
        assert any(variable.VarName.startswith("JobSchedule") and variable.X > 0.5 for variable in flow_shop_model.getVars())
        assert any(variable.VarName.startswith("StartTime") for variable in flow_shop_model.getVars())
        assert any(constraint.ConstrName.startswith("OneJobPerSchedule_") for constraint in flow_shop_model.getConstrs())
        assert any(constraint.ConstrName.startswith("MachinePrecedence_") for constraint in flow_shop_model.getConstrs())
        assert any(constraint.ConstrName.startswith("JobPrecedence_") for constraint in flow_shop_model.getConstrs())

        vrptw = VRPTW(seed=seed)
        vrptw_model = vrptw.generate_instance()
        vrptw_model.optimize()
        assert vrptw_model.Status == 2
        assert "latent_feasible_route" not in vrptw.parameters
        assert "latent_arrival_times" not in vrptw.parameters
        assert "latent_cumulative_loads" not in vrptw.parameters
        assert vrptw.parameters["distance_matrix"]
        assert vrptw.parameters["time_windows"]
        assert vrptw.parameters["demands"]
        assert vrptw.parameters["vehicle_count"] == 1
        assert "single vehicle starts at the depot" in vrptw.parameters["route_policy"]
        assert vrptw.parameters["vehicle_capacity"] >= sum(vrptw.parameters["demands"].values())
        assert "customer_time_windows" in vrptw.parameters["required_constraints"]
        assert "single_route_leaves_and_returns_to_depot" in vrptw.parameters["required_constraints"]
        assert "time_propagation_on_used_arcs" in vrptw.parameters["required_constraints"]
        assert "load_propagation_on_used_arcs" in vrptw.parameters["required_constraints"]
        assert "state there is exactly one vehicle" in vrptw.parameters["required_parameter_presentation"]
        assert "This is a single-vehicle CVRPTW seed, not a fleet-sizing or multi-vehicle VRPTW seed." in vrptw.parameters["business_interpretation_guardrails"]
        assert "Do not reveal or constrain the model to any internally used feasible route certificate." in vrptw.parameters["business_interpretation_guardrails"]
        assert vrptw.parameters["structured_problem_data"]["parameters"]["vehicle_count"] == 1
        assert "feasibility_certificate" not in vrptw.parameters["structured_problem_data"]
        assert "no route certificate should be included" in vrptw.parameters["structured_problem_data"]["generation_note"]
        active_arcs = [variable for variable in vrptw_model.getVars() if variable.VarName.startswith("ArcVisit") and variable.X > 0.5]
        active_arc_names = [variable.VarName for variable in active_arcs]
        assert len(active_arcs) == len(vrptw.parameters["customers"]) + 1
        for customer in vrptw.parameters["customers"]:
            incoming = [name for name in active_arc_names if name.endswith(f",{customer}]")]
            outgoing = [name for name in active_arc_names if name.startswith(f"ArcVisit[{customer},")]
            assert len(incoming) == 1
            assert len(outgoing) == 1
            service_start = vrptw_model.getVarByName(f"ServiceStartTime[{customer}]")
            cumulative_load = vrptw_model.getVarByName(f"CumulativeLoad[{customer}]")
            lower, upper = vrptw.parameters["time_windows"][customer]
            assert lower - 1e-6 <= service_start.X <= upper + 1e-6
            assert vrptw.parameters["demands"][customer] - 1e-6 <= cumulative_load.X <= vrptw.parameters["vehicle_capacity"] + 1e-6
        assert any(constraint.ConstrName == "DepotDeparture" for constraint in vrptw_model.getConstrs())
        assert any(constraint.ConstrName == "DepotReturn" for constraint in vrptw_model.getConstrs())

        net_assignment = NetAssignment(seed=seed)
        net_assignment_model = net_assignment.generate_instance()
        net_assignment_model.optimize()
        assert net_assignment_model.Status == 2
        assert net_assignment.parameters["total_supply_hours"] == net_assignment.parameters["total_demand_hours"]
        assert net_assignment.parameters["supply_hours"]
        assert net_assignment.parameters["demand_hours"]
        assert net_assignment.parameters["cost_per_hour"]
        assert net_assignment.parameters["max_contribution_hours"]
        assert "latent_feasible_allocation" not in net_assignment.parameters
        assert "feasible allocation" not in str(net_assignment.parameters["person_table"]).lower()
        assert "feasible allocation" not in str(net_assignment.parameters["project_table"]).lower()
        assert "no allocation certificate should be included" in net_assignment.parameters["structured_problem_data"]["generation_note"]
        assert net_assignment.parameters["person_table"]
        assert net_assignment.parameters["project_table"]
        assert net_assignment.parameters["person_project_pair_table"]
        assert net_assignment.parameters["compact_project_assignment_tables"]["person_supply_table"]
        assert net_assignment.parameters["compact_project_assignment_tables"]["project_demand_table"]
        assert net_assignment.parameters["compact_project_assignment_tables"]["cost_per_hour_matrix"]["rows"]
        assert net_assignment.parameters["compact_project_assignment_tables"]["max_contribution_hours_matrix"]["rows"]
        compact_netasgn_source = _compact_source_data({"generation_params": net_assignment.parameters})
        assert compact_netasgn_source["available"] is True
        assert "compact_project_assignment_tables" in compact_netasgn_source["value"]
        assert net_assignment.parameters["balance_summary"]["balanced"] is True
        assert net_assignment.parameters["structured_problem_data"]
        assert net_assignment.parameters["decision_variables"]["Assign[i,j]"].startswith("continuous")
        assert "nonnegative_continuous_assignment_hours" in net_assignment.parameters["required_constraints"]
        assert "state total available hours equals total required hours" in net_assignment.parameters["required_parameter_presentation"]
        assert "Do not model this as binary one-to-one matching." in net_assignment.parameters["business_interpretation_guardrails"]
        assert "Do not relax supply equality to a less-than-or-equal capacity constraint." in net_assignment.parameters["business_interpretation_guardrails"]
        assert "Do not reveal or constrain the model to any internally used feasible allocation certificate." in net_assignment.parameters["business_interpretation_guardrails"]
        assert any(variable.VarName.startswith("Assign") and variable.X > 1e-6 for variable in net_assignment_model.getVars())
        assert any(constraint.ConstrName.startswith("SupplyConstraint") for constraint in net_assignment_model.getConstrs())
        assert any(constraint.ConstrName.startswith("DemandConstraint") for constraint in net_assignment_model.getConstrs())
        for person in net_assignment.parameters["people"]:
            assigned = sum(
                net_assignment_model.getVarByName(f"Assign[{person},{project}]").X
                for project in net_assignment.parameters["projects"]
            )
            assert abs(assigned - net_assignment.parameters["supply_hours"][person]) <= 1e-6
        for project in net_assignment.parameters["projects"]:
            received = sum(
                net_assignment_model.getVarByName(f"Assign[{person},{project}]").X
                for person in net_assignment.parameters["people"]
            )
            assert abs(received - net_assignment.parameters["demand_hours"][project]) <= 1e-6
        for person in net_assignment.parameters["people"]:
            for project in net_assignment.parameters["projects"]:
                value = net_assignment_model.getVarByName(f"Assign[{person},{project}]").X
                assert value <= net_assignment.parameters["max_contribution_hours"][person][project] + 1e-6

        team = TeamFormulation(seed=seed)
        team_model = team.generate_instance()
        team_model.optimize()
        assert team_model.Status == 2
        assert team_model.ObjVal > 0
        assert team.parameters["compact_team_formulation_tables"]
        assert team.parameters["individual_skill"]
        assert team.parameters["required_skill"]
        assert "maximum_skill_shortage_priority" in team.parameters["objective_terms"]
        assert "attained_skill_equals_sum_of_assigned_individual_skills" in team.parameters["required_constraints"]
        compact_team_source = _compact_source_data({"generation_params": team.parameters})
        assert compact_team_source["available"] is True
        assert "compact_team_formulation_tables" in compact_team_source["value"]
        assert any(variable.VarName.startswith("Assign") and variable.X > 0.5 for variable in team_model.getVars())
        assert any(variable.VarName.startswith("SkillShortage") for variable in team_model.getVars())
        assert any(variable.VarName.startswith("MaxSkillShortage") for variable in team_model.getVars())
        assert any(constraint.ConstrName.startswith("SkillBalance") for constraint in team_model.getConstrs())
        assert any(constraint.ConstrName.startswith("SkillShortageDef") for constraint in team_model.getConstrs())
        assert any(constraint.ConstrName.startswith("ProjectStaffing") for constraint in team_model.getConstrs())

        job_shop = JobShop(seed=seed)
        job_shop_model = job_shop.generate_instance()
        job_shop_model.optimize()
        assert job_shop_model.Status == 2
        assert job_shop_model.NumVars < 500
        assert job_shop_model.NumConstrs < 1000
        assert job_shop.parameters["operations_by_job"]
        assert job_shop.parameters["machine_route"]
        assert job_shop.parameters["processing_times"]
        assert "same_machine_pairwise_nonoverlap_with_binary_ordering" in job_shop.parameters["required_constraints"]
        assert "This is a job shop: each job has its own ordered machine route." in job_shop.parameters["business_interpretation_guardrails"]
        assert any(variable.VarName.startswith("X_") for variable in job_shop_model.getVars())
        assert any(constraint.ConstrName.startswith("machine_") for constraint in job_shop_model.getConstrs())


@pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")
def test_phase4_remaining_generators_keep_meaningful_structure() -> None:
    from or_cpt_engine.generators.optmath_seed.Blending_problem.Blending_Problem import Generator as Blending
    from or_cpt_engine.generators.optmath_seed.CarSelection.CarSelection_parsed import Generator as CarSelection
    from or_cpt_engine.generators.optmath_seed.ContractAllocation.ContractAllocation_parsed import (
        Generator as ContractAllocation,
    )
    from or_cpt_engine.generators.optmath_seed.MarketShare.MarketShare import Generator as MarketShare
    from or_cpt_engine.generators.optmath_seed.Revenue_management.Revenue_management import (
        Generator as RevenueManagement,
    )
    from or_cpt_engine.generators.optmath_seed.Diet_Problem.Diet_Problem import Generator as DietProblem
    from or_cpt_engine.generators.optmath_seed.cut_edited.cut_parsed import Generator as CuttingStock
    from or_cpt_engine.generators.optmath_seed.cell_tower.cell_tower_parsed import Generator as CellTower
    from or_cpt_engine.generators.optmath_seed.diet.diet_parsed import Generator as Diet
    from or_cpt_engine.generators.optmath_seed.dietu.dietu_parsed import Generator as DietU
    from or_cpt_engine.generators.optmath_seed.electrical_power.electrical_power_parsed import (
        Generator as ElectricalPower,
    )
    from or_cpt_engine.generators.optmath_seed.farmplanning.farmplanning_parsed import Generator as FarmPlanning
    from or_cpt_engine.generators.optmath_seed.revenue.revenue_parsed import Generator as Revenue

    for seed in range(3):
        blending = Blending(seed=seed)
        blending_model = blending.generate_instance()
        blending_model.optimize()
        assert blending_model.Status == 2
        assert sum(variable.X > 1e-6 for variable in blending_model.getVars()) >= 2
        assert any(constraint.ConstrName.startswith("Element_") for constraint in blending_model.getConstrs())

        contract = ContractAllocation(seed=seed)
        contract_model = contract.generate_instance()
        contract_model.optimize()
        assert contract_model.Status == 2
        assert contract.parameters["producer_capacities"]
        assert contract.parameters["contract_sizes"]
        assert contract.parameters["production_costs"]
        assert contract.parameters["compact_contract_allocation_tables"]
        assert "exact_contract_fulfillment" in contract.parameters["required_constraints"]
        assert "This is a contract quantity allocation model, not a facility-location or routing model." in contract.parameters["business_interpretation_guardrails"]
        compact_contract_source = _compact_source_data({"generation_params": contract.parameters})
        assert compact_contract_source["available"] is True
        assert "compact_contract_allocation_tables" in compact_contract_source["value"]
        assert any(variable.VarName.startswith("Generation[") and variable.X > 1e-6 for variable in contract_model.getVars())
        assert any(variable.VarName.startswith("GenerationIncidence[") and variable.VType == "B" for variable in contract_model.getVars())
        assert any(constraint.ConstrName.startswith("DeliveryActivation") for constraint in contract_model.getConstrs())
        for contract_id, contract_size in contract.parameters["contract_sizes"].items():
            fulfilled = sum(
                contract_model.getVarByName(f"Generation[{producer_id},{contract_id}]").X
                for producer_id in contract.parameters["producers"]
            )
            assert abs(fulfilled - contract_size) <= 1e-6

        market = MarketShare(seed=seed)
        market_model = market.generate_instance()
        market_model.optimize()
        assert market_model.Status == 2
        assert market_model.NumVars < 500
        assert market_model.NumConstrs < 1000
        assert market.parameters["compact_marketshare_tables"]["demand_table"]
        assert market.parameters["compact_marketshare_tables"]["unit_profit_matrices_by_company"]["rows"]
        assert "profit_table" not in market.parameters["compact_marketshare_tables"]
        compact_market_source = _compact_source_data({"generation_params": market.parameters})
        assert compact_market_source["available"] is True
        assert compact_market_source["truncated"] is False
        assert "compact_marketshare_tables" in compact_market_source["value"]
        assert "market_product_demand_exactly_satisfied" in market.parameters["required_constraints"]
        assert "resource-capacity" in " ".join(market.parameters["business_interpretation_guardrails"]).lower()
        assert all(variable.VType == "I" for variable in market_model.getVars())
        assert any(constraint.ConstrName.startswith("demand_") for constraint in market_model.getConstrs())

        electrical_power = ElectricalPower(seed=seed)
        electrical_model = electrical_power.generate_instance()
        electrical_model.optimize()
        assert electrical_model.Status == 2
        assert electrical_power.parameters["compact_electrical_power_tables"]["generator_type_table"]
        assert electrical_power.parameters["compact_electrical_power_tables"]["period_demand_table"]
        compact_power_source = _compact_source_data({"generation_params": electrical_power.parameters})
        assert compact_power_source["available"] is True
        assert "compact_electrical_power_tables" in compact_power_source["value"]
        assert any(variable.VarName.startswith("NumGenerators[") and variable.VType == "I" for variable in electrical_model.getVars())
        assert any(variable.VarName.startswith("PowerOutput[") and variable.VType == "C" for variable in electrical_model.getVars())
        assert any(constraint.ConstrName.startswith("Reserve_") for constraint in electrical_model.getConstrs())

        car_selection = CarSelection(seed=seed)
        car_model = car_selection.generate_instance()
        car_model.optimize()
        assert car_model.Status == 2
        assert car_selection.parameters["compact_car_selection_tables"]["eligibility_matrix"]["rows"]
        assert car_selection.parameters["compact_car_selection_tables"]["eligibility_by_participant"]
        compact_car_source = _compact_source_data({"generation_params": car_selection.parameters})
        assert compact_car_source["available"] is True
        assert "compact_car_selection_tables" in compact_car_source["value"]
        assert "assignment_only_if_eligible" in car_selection.parameters["required_constraints"]
        assert all(variable.VType == "B" for variable in car_model.getVars())
        assert any(constraint.ConstrName.startswith("Preference_") for constraint in car_model.getConstrs())
        assert any(constraint.ConstrName.startswith("OneCarPerParticipant_") for constraint in car_model.getConstrs())
        assert any(constraint.ConstrName.startswith("OneParticipantPerCar_") for constraint in car_model.getConstrs())

        cell_tower = CellTower(seed=seed)
        cell_model = cell_tower.generate_instance()
        cell_model.optimize()
        assert cell_model.Status == 2
        assert cell_tower.parameters["compact_cell_tower_tables"]["tower_table"]
        assert cell_tower.parameters["compact_cell_tower_tables"]["region_table"]
        assert cell_tower.parameters["compact_cell_tower_tables"]["coverage_matrix"]["rows"]
        compact_cell_source = _compact_source_data({"generation_params": cell_tower.parameters})
        assert compact_cell_source["available"] is True
        assert "compact_cell_tower_tables" in compact_cell_source["value"]
        assert "region_coverage_linking" in cell_tower.parameters["required_constraints"]
        assert "tower_budget_limit" in cell_tower.parameters["required_constraints"]
        assert any(variable.VarName.startswith("Build") and variable.VType == "B" for variable in cell_model.getVars())
        assert any(variable.VarName.startswith("Covered") and variable.VType == "B" for variable in cell_model.getVars())
        assert any(constraint.ConstrName.startswith("Coverage_") for constraint in cell_model.getConstrs())
        assert cell_tower.parameters["budget"] < sum(cell_tower.parameters["tower_costs"].values())

        revenue_management = RevenueManagement(seed=seed)
        revenue_model = revenue_management.generate_instance()
        revenue_model.optimize()
        assert revenue_model.Status == 2
        assert revenue_management.parameters["compact_revenue_management_tables"]["resource_capacity_table"]
        assert revenue_management.parameters["compact_revenue_management_tables"]["package_table"]
        assert revenue_management.parameters["compact_revenue_management_tables"]["package_resource_usage_matrix"]["rows"]
        compact_revenue_source = _compact_source_data({"generation_params": revenue_management.parameters})
        assert compact_revenue_source["available"] is True
        assert "compact_revenue_management_tables" in compact_revenue_source["value"]
        assert "package_demand_upper_bound" in revenue_management.parameters["required_constraints"]
        assert "resource_capacity_consumption_limit" in revenue_management.parameters["required_constraints"]
        assert "package_revenue_times_package_sales" in revenue_management.parameters["objective_terms"]
        assert "[Previous mathematical formulation here]" not in revenue_management.mathematical_formulation
        assert any(variable.VarName.startswith("PackageSales") and variable.VType != "B" for variable in revenue_model.getVars())
        assert any(constraint.ConstrName.startswith("DemandLimit") for constraint in revenue_model.getConstrs())
        assert any(constraint.ConstrName.startswith("CapacityLimit") for constraint in revenue_model.getConstrs())
        for package in revenue_management.parameters["packages"]:
            sales = revenue_model.getVarByName(f"PackageSales[{package}]").X
            assert sales <= revenue_management.parameters["demands"][package] + 1e-6
        for resource in revenue_management.parameters["resources"]:
            used = sum(
                revenue_management.resource_usage[package, resource]
                * revenue_model.getVarByName(f"PackageSales[{package}]").X
                for package in revenue_management.parameters["packages"]
            )
            assert used <= revenue_management.parameters["capacities"][resource] + 1e-6

        revenue = Revenue(seed=seed)
        revenue_model = revenue.generate_instance()
        revenue_model.optimize()
        assert revenue_model.Status == 2
        assert revenue.parameters["compact_revenue_tables"]["resource_capacity_table"]
        assert revenue.parameters["compact_revenue_management_tables"]["package_resource_usage_matrix"]["rows"]
        compact_revenue_source = _compact_source_data({"generation_params": revenue.parameters})
        assert compact_revenue_source["available"] is True
        assert "compact_revenue_management_tables" in compact_revenue_source["value"]
        assert "package_demand_upper_bound" in revenue.parameters["required_constraints"]
        assert "resource_capacity_consumption_limit" in revenue.parameters["required_constraints"]
        assert any(variable.VarName.startswith("Sell") and variable.VType != "B" for variable in revenue_model.getVars())
        assert any(constraint.ConstrName.startswith("Demand_") for constraint in revenue_model.getConstrs())
        assert any(constraint.ConstrName.startswith("Capacity_") for constraint in revenue_model.getConstrs())
        assert any(abs(constraint.Slack) <= 1e-6 for constraint in revenue_model.getConstrs() if constraint.ConstrName.startswith("Capacity_"))

        cutting = CuttingStock(seed=seed)
        cutting_model = cutting.generate_instance()
        cutting_model.optimize()
        assert cutting_model.Status == 2
        assert sum(variable.X > 0.5 for variable in cutting_model.getVars() if variable.VarName.startswith("Cut")) >= 2
        assert all(variable.VType != "B" for variable in cutting_model.getVars() if variable.VarName.startswith("Cut"))
        assert cutting.parameters["compact_cutting_stock_tables"]
        assert "exact_order_width_fulfillment" in cutting.parameters["required_constraints"]
        assert "state Cut[j] is a nonnegative integer count of stock rolls using pattern j" in cutting.parameters[
            "required_parameter_presentation"
        ]
        compact_cutting_source = _compact_source_data({"generation_params": cutting.parameters})
        assert compact_cutting_source["available"] is True
        assert "compact_cutting_stock_tables" in compact_cutting_source["value"]
        assert any(constraint.ConstrName.startswith("Fill_") for constraint in cutting_model.getConstrs())

        for diet_cls in (Diet, DietProblem, DietU):
            diet = diet_cls(seed=seed)
            diet_model = diet.generate_instance()
            diet_model.optimize()
            assert diet_model.Status == 2
            assert sum(variable.X > 1e-6 for variable in diet_model.getVars()) >= 2
            assert any(
                "nutrient" in constraint.ConstrName.lower()
                or "requirement" in constraint.ConstrName.lower()
                for constraint in diet_model.getConstrs()
            )

        farm = FarmPlanning(seed=seed)
        farm_model = farm.generate_instance()
        farm_model.optimize()
        assert farm_model.Status == 2
        assert farm_model.NumVars < 80
        assert len(str(farm.parameters)) < 10000
        assert farm.parameters["compact_farm_planning_tables"]
        compact_farm_source = _compact_source_data({"generation_params": farm.parameters})
        assert compact_farm_source["available"] is True
        assert "compact_farm_planning_tables" in compact_farm_source["value"]
        assert all(variable.VType != "B" for variable in farm_model.getVars())
        assert any(variable.VarName.startswith("AmountPlanted") and variable.X > 1e-6 for variable in farm_model.getVars())
        assert any(variable.VarName.startswith("Sales") and variable.X > 1e-6 for variable in farm_model.getVars())
        assert "crop_yield_split_between_family_consumption_and_sales" in farm.parameters["required_constraints"]
        assert "family_consumption_bundle_fraction_sum" in farm.parameters["required_constraints"]
        assert "This is a continuous farm planning LP, not a binary crop-selection or fixed-charge model." in farm.parameters["business_interpretation_guardrails"]
        assert any(constraint.ConstrName.startswith("MonthlyLandCapacity") for constraint in farm_model.getConstrs())
        assert any(constraint.ConstrName.startswith("MonthlyLaborCoverage") for constraint in farm_model.getConstrs())
        assert any(constraint.ConstrName.startswith("MonthlyWaterLimit") for constraint in farm_model.getConstrs())
        assert any(constraint.ConstrName == "AnnualWaterAvailability" for constraint in farm_model.getConstrs())
        assert any(constraint.ConstrName.startswith("CropYieldSplit") for constraint in farm_model.getConstrs())
        assert abs(sum(farm_model.getVarByName(f"FractionConsumed[{bundle}]").X for bundle in farm.parameters["consumption_bundles"]) - 1.0) <= 1e-6
        for crop in farm.parameters["crops"]:
            production = farm.parameters["yield_per_area"][crop] * farm_model.getVarByName(f"AmountPlanted[{crop}]").X
            consumed = sum(
                farm.parameters["amount_in_bundle"][f"{crop}|{bundle}"]
                * farm_model.getVarByName(f"FractionConsumed[{bundle}]").X
                for bundle in farm.parameters["consumption_bundles"]
            )
            sold = farm_model.getVarByName(f"Sales[{crop}]").X
            assert abs(production - consumed - sold) <= 1e-5


def test_render_and_export_documents(tmp_path: Path) -> None:
    accepted_pairs = [
        {
            "pair_id": "pair_1",
            "fm_id": "fm_1",
            "bt_id": "bt_1",
            "instance_id": "inst_1",
            "generator_id": "fixture",
            "difficulty_level": "level_1",
            "sub_family": "tiny_fixture",
            "canonical_math_signature": {"variables": ["x"], "objective": "maximize x"},
            "problem_statement": "A production planning team allocates capacity across products while meeting demand and budget constraints.",
            "reference": {"objective_value": 1.0},
            "source_metadata": {
                "task_family": "production_planning",
                "scenario_id": "scenario_fixture",
                "scenario_variant_id": "variant_fixture",
                "business_trigger": "weekly planning cycle",
            },
            "generated": {
                "solver_result": {
                    "status": "OPTIMAL",
                    "objective_value": 1.0,
                    "solution": {
                        "objective_value": 1.0,
                        "objective_recomputed": 1.0,
                        "nonzero_variable_values": {"x": 1.0},
                        "nonzero_variable_count": 1,
                        "variable_count": 1,
                        "feasibility": {
                            "constraint_violation": 0.0,
                            "bound_violation": 0.0,
                            "integer_violation": 0.0,
                            "max_violation": 0.0,
                            "is_feasible_within_tolerance": True,
                        },
                    },
                }
            },
            "correctness": {"is_correct": True, "abs_error": 0.0, "rel_error": 0.0},
            "modeling_answer": {
                "modeling_explanation": "Use one nonnegative decision variable and a capacity constraint.",
                "math_model": "max x subject to x <= 1, x >= 0",
                "gurobipy_code": _tiny_gurobi_code(),
            },
        }
    ]
    pairs_path = tmp_path / "accepted_pairs.jsonl"
    write_jsonl(pairs_path, accepted_pairs)

    render_dir = tmp_path / "render"
    render_result = render_cpt_documents(pairs_path, render_dir, EngineConfig())
    assert render_result["documents"] == 1
    rendered = read_jsonl(render_dir / "cpt_documents.jsonl")[0]
    assert "## Optimal Solution" in rendered["text"]
    assert "| `x` | 1 |" in rendered["text"]
    assert "## Feasibility Check" in rendered["text"]
    assert "The validated optimization model and solution are suitable for downstream CPT training data." not in rendered["text"]
    assert "downstream CPT training data" not in rendered["text"]
    assert "## Final Answer" not in rendered["text"]
    assert rendered["difficulty_level"] == "level_1"
    assert rendered["metadata"]["sub_family"] == "tiny_fixture"
    assert rendered["metadata"]["seed_instance_id"] == "inst_1"
    assert rendered["metadata"]["solver_objective_value"] == 1.0
    assert rendered["metadata"]["forward_eval_status"] == "ACCEPTED"

    export_dir = tmp_path / "export"
    export_result = export_train_val_test(render_dir / "cpt_documents.jsonl", export_dir, EngineConfig())
    assert export_result["mode"] == "train_only_streaming"
    assert export_result["train"] == 1
    assert export_result["duplicates_removed"] == 0
    assert not (export_dir / "val.jsonl").exists()
    assert not (export_dir / "test.jsonl").exists()
    assert (export_dir / "train_manifest.json").exists()
    assert not (export_dir / "train_all.jsonl").exists()
    assert not (export_dir / "train_code_heavy.jsonl").exists()
    assert not (export_dir / "train_validation_heavy.jsonl").exists()
    assert not (export_dir / "train_high_value_or.jsonl").exists()
    assert not (export_dir / "train_reasoning_heavy.jsonl").exists()
    assert not (export_dir / "train_balanced_by_generator.jsonl").exists()
    exported = read_jsonl(export_dir / "train.jsonl")
    assert exported[0]["source_id"] == "inst_1"
    assert exported[0]["metadata"]["seed_instance_id"] == "inst_1"
    assert exported[0]["metadata"]["solver_objective_value"] == 1.0
    assert exported[0]["metadata"]["forward_eval_status"] == "ACCEPTED"
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_mode"] == "train_only_streaming"
    assert "view_counts" not in manifest
    assert "view_files" not in manifest
    assert "sub_family_counts" in manifest
    assert manifest["metadata_coverage"]["seed_instance_id"]["missing"] == 0
    assert manifest["metadata_coverage"]["solver_objective_value"]["missing"] == 0
    assert manifest["data_quality_counts"] == {}


def test_render_solver_validation_note_doc_type(tmp_path: Path) -> None:
    accepted_pairs = [
        {
            "pair_id": "pair_1",
            "fm_id": "fm_1",
            "bt_id": "bt_1",
            "instance_id": "inst_1",
            "generator_id": "fixture",
            "difficulty_level": "level_1",
            "canonical_math_signature": {"variables": ["x"], "objective": "maximize x"},
            "problem_statement": "A planner chooses a production quantity subject to a simple upper bound.",
            "reference": {"objective_value": 1.0},
            "generated": {
                "solver_result": {
                    "status": "OPTIMAL",
                    "objective_value": 1.0,
                    "solution": {
                        "objective_value": 1.0,
                        "objective_recomputed": 1.0,
                        "nonzero_variable_values": {"x": 1.0},
                        "feasibility": {
                            "constraint_violation": 0.0,
                            "bound_violation": 0.0,
                            "integer_violation": 0.0,
                            "max_violation": 0.0,
                            "is_feasible_within_tolerance": True,
                        },
                    },
                }
            },
            "correctness": {"is_correct": True, "abs_error": 0.0, "rel_error": 0.0},
            "modeling_answer": {
                "modeling_explanation": "Use one variable and one upper-bound constraint.",
                "math_model": "max x subject to x <= 1",
                "gurobipy_code": _tiny_gurobi_code(),
            },
        }
    ]
    pairs_path = tmp_path / "accepted_pairs.jsonl"
    write_jsonl(pairs_path, accepted_pairs)
    config = EngineConfig(rendering=RenderingConfig(doc_type_policy={"solver_validation_note": {"ratio": 1.0}}))

    result = render_cpt_documents(pairs_path, tmp_path / "render", config)

    assert result["documents"] == 1
    rendered = read_jsonl(tmp_path / "render" / "cpt_documents.jsonl")[0]
    assert rendered["doc_type"] == "solver_validation_note"
    assert "# Solver Validation Note" in rendered["text"]
    assert "downstream training" not in rendered["text"].lower()
    assert "cpt corpus" not in rendered["text"].lower()
    assert "sample is exported" not in rendered["text"].lower()


def test_render_report_summarizes_rejection_reasons(tmp_path: Path) -> None:
    accepted_pairs = [
        {
            "pair_id": "pair_bad",
            "fm_id": "fm_bad",
            "bt_id": "bt_bad",
            "instance_id": "inst_bad",
            "generator_id": "optmath_carselection",
            "difficulty_level": "level_1",
            "canonical_math_signature": {"variables": ["x"], "objective": "maximize assignments"},
            "problem_statement": "A technician job matching team assigns technicians to inspection jobs.",
            "reference": {"objective_value": 1.0},
            "source_metadata": {"task_family": "assignment", "scenario_id": "bad_scenario"},
            "generated": {
                "solver_result": {
                    "status": "OPTIMAL",
                    "objective_value": 1.0,
                    "solution": {
                        "objective_value": 1.0,
                        "objective_recomputed": 1.0,
                        "nonzero_variable_values": {"x": 1.0},
                        "feasibility": {
                            "constraint_violation": 0.0,
                            "bound_violation": 0.0,
                            "integer_violation": 0.0,
                            "max_violation": 0.0,
                            "is_feasible_within_tolerance": True,
                        },
                    },
                }
            },
            "correctness": {"is_correct": True, "abs_error": 0.0, "rel_error": 0.0},
            "modeling_answer": {
                "modeling_explanation": "Use binary assignment variables.",
                "math_model": "max x subject to x <= 1",
                "gurobipy_code": _tiny_gurobi_code(),
            },
        }
    ]
    pairs_path = tmp_path / "accepted_pairs.jsonl"
    write_jsonl(pairs_path, accepted_pairs)

    result = render_cpt_documents(pairs_path, tmp_path / "render", EngineConfig())

    assert result["documents"] == 0
    assert result["rejected"] == 1
    report = (tmp_path / "render" / "cpt_rendering_report.md").read_text(encoding="utf-8")
    assert "## Rejection Reasons" in report
    assert "SEMANTIC_COHERENCE:SEMANTIC_FORBIDDEN_TERM:technician" in report


def test_doc_type_policy_is_interleaved_for_small_runs() -> None:
    config = RenderingConfig(
        doc_type_policy={
            "final_model_report": {"ratio": 0.5},
            "modeling_rationale": {"ratio": 0.5},
            "solver_validation_note": {"ratio": 0},
        }
    )

    first_ten = [_choose_doc_type(index, config) for index in range(10)]

    assert set(first_ten) == {"final_model_report", "modeling_rationale"}
    assert abs(first_ten.count("final_model_report") - first_ten.count("modeling_rationale")) <= 1


def test_rendering_style_policy_is_recorded_and_affects_section_order(tmp_path: Path) -> None:
    accepted_pairs = []
    for index in range(2):
        accepted_pairs.append(
            {
                "pair_id": f"pair_{index}",
                "fm_id": f"fm_{index}",
                "bt_id": f"bt_{index}",
                "instance_id": f"inst_{index}",
                "generator_id": "fixture",
                "difficulty_level": "level_1",
                "problem_statement": "A planner chooses a production quantity subject to a simple upper bound.",
                "reference": {"objective_value": 1.0},
                "generated": {
                    "solver_result": {
                        "status": "OPTIMAL",
                        "objective_value": 1.0,
                        "solution": {
                            "objective_value": 1.0,
                            "objective_recomputed": 1.0,
                            "nonzero_variable_values": {"x": 1.0},
                            "feasibility": {
                                "constraint_violation": 0.0,
                                "bound_violation": 0.0,
                                "integer_violation": 0.0,
                                "max_violation": 0.0,
                                "is_feasible_within_tolerance": True,
                            },
                        },
                    }
                },
                "correctness": {"is_correct": True, "abs_error": 0.0, "rel_error": 0.0},
                "modeling_answer": {
                    "modeling_explanation": "Use one variable and one upper-bound constraint.",
                    "math_model": "max x subject to x <= 1",
                    "gurobipy_code": _tiny_gurobi_code(),
                },
            }
        )
    pairs_path = tmp_path / "accepted_pairs.jsonl"
    write_jsonl(pairs_path, accepted_pairs)
    config = EngineConfig(
        rendering=RenderingConfig(
            doc_type_policy={"final_model_report": {"ratio": 1.0}},
            style_policy={
                "verbosity_levels": ["concise"],
                "writing_styles": ["operations analyst memo"],
                "audiences": ["training data curator"],
                "section_order_variants": ["implementation_first"],
            },
        )
    )

    result = render_cpt_documents(pairs_path, tmp_path / "render", config)
    rendered = read_jsonl(tmp_path / "render" / "cpt_documents.jsonl")
    style_context = _choose_style_context(0, config.rendering)

    assert result["documents"] == 2
    assert style_context["section_order_variant"] == "implementation_first"
    assert rendered[0]["metadata"]["verbosity_level"] == "concise"
    assert rendered[0]["metadata"]["writing_style"] == "operations analyst memo"
    assert rendered[0]["metadata"]["audience"] == "training data curator"
    assert rendered[0]["metadata"]["section_order_variant"] == "implementation_first"
    assert rendered[0]["text"].find("## Implementation") < rendered[0]["text"].find("## Model Formulation")


def test_renderer_strips_forward_repair_audit_language_before_training_text() -> None:
    row = {
        "pair_id": "pair_repair_audit",
        "fm_id": "fm_repair_audit",
        "bt_id": "bt_repair_audit",
        "instance_id": "inst_repair_audit",
        "generator_id": "fixture",
        "difficulty_level": "level_1",
        "problem_statement": "A planner chooses a production quantity subject to a simple upper bound.",
        "reference": {"objective_value": 1.0},
        "generated": {
            "solver_result": {
                "status": "OPTIMAL",
                "objective_value": 1.0,
                "solution": {
                    "objective_value": 1.0,
                    "objective_recomputed": 1.0,
                    "nonzero_variable_values": {"x": 1.0},
                    "feasibility": {
                        "constraint_violation": 0.0,
                        "bound_violation": 0.0,
                        "integer_violation": 0.0,
                        "max_violation": 0.0,
                        "is_feasible_within_tolerance": True,
                    },
                },
            }
        },
        "correctness": {"is_correct": True, "abs_error": 0.0, "rel_error": 0.0},
        "modeling_answer": {
            "modeling_explanation": (
                "The previous model failed because it changed the source balance. "
                "The repaired model adds a correction after validation showed an objective mismatch. "
                "The formulation introduces one decision variable for the controllable quantity, uses the stated "
                "capacity limit as the only resource restriction, and keeps the business objective aligned with "
                "the original planning question."
            ),
            "math_model": (
                "The previous formulation used the expected objective from the repair trace. "
                "Maximize x subject to x <= 1 and x >= 0, where x represents the selected production quantity "
                "and the single upper-bound constraint represents available capacity."
            ),
            "gurobipy_code": _tiny_gurobi_code(),
        },
    }
    config = EngineConfig(rendering=RenderingConfig(doc_type_policy={"final_model_report": {"ratio": 1.0}}))

    accepted, rendered = render_cpt_document_row(row, config, index=0, views_by_instance={})

    assert accepted is True
    text = rendered.text.lower()
    assert "previous model" not in text
    assert "failed because" not in text
    assert "validation showed" not in text
    assert "objective mismatch" not in text
    assert "repaired model" not in text
    assert "expected objective" not in text
    assert "one decision variable for the controllable quantity" in text
    assert "maximize x subject to x <= 1" in text


def test_render_quality_rejects_system_audit_phrases() -> None:
    quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A planner solves a small optimization problem.",
                "## Model Formulation",
                "max x subject to x <= 1",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The validated optimization model and solution are suitable for downstream CPT training data.",
            ]
        ),
        "final_model_report",
    )

    assert quality["status"] == "FAIL"
    assert any(issue.startswith("FORBIDDEN_PHRASE:downstream cpt training data") for issue in quality["issues"])

    repair_audit_quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A planner solves a small optimization problem.",
                "## Model Formulation",
                "max x subject to x <= 1",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The previous model failed because validation showed an objective mismatch with the reference objective.",
            ]
        ),
        "final_model_report",
    )

    assert repair_audit_quality["status"] == "FAIL"
    assert any(issue.startswith("FORBIDDEN_PHRASE:previous model") for issue in repair_audit_quality["issues"])
    assert any(issue.startswith("FORBIDDEN_PHRASE:failed because") for issue in repair_audit_quality["issues"])
    assert any(issue.startswith("FORBIDDEN_PHRASE:validation showed") for issue in repair_audit_quality["issues"])
    assert any(issue.startswith("FORBIDDEN_PHRASE:objective mismatch") for issue in repair_audit_quality["issues"])
    assert any(issue.startswith("FORBIDDEN_PHRASE:reference objective") for issue in repair_audit_quality["issues"])


def test_render_quality_rejects_excessive_decimal_precision() -> None:
    quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A planner uses coefficient 52.69059351204961 in a business table.",
                "## Model Formulation",
                "max x subject to x <= 1",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
    )

    assert quality["status"] == "FAIL"
    assert any(issue.startswith("EXCESSIVE_DECIMAL_PRECISION") for issue in quality["issues"])


def test_render_quality_rejects_semantic_coherence_drift() -> None:
    quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A technician job matching team assigns technicians to inspection jobs.",
                "## Model Formulation",
                "max x subject to x <= 1",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
        generator_id="optmath_carselection",
    )

    assert quality["status"] == "FAIL"
    assert any(issue.startswith("SEMANTIC_COHERENCE:SEMANTIC_FORBIDDEN_TERM:technician") for issue in quality["issues"])

    service_job_quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A field service planner assigns approved participants to inspection jobs based on eligibility.",
                "## Model Formulation",
                "max x subject to x <= 1",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
        generator_id="optmath_carselection",
    )

    assert service_job_quality["status"] == "FAIL"
    assert any(
        issue.startswith("SEMANTIC_COHERENCE:SEMANTIC_FORBIDDEN_TERM:inspection job")
        for issue in service_job_quality["issues"]
    )

    clsp_quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A factory plan allows unmet demand to be carried as backlog with shortage penalty costs.",
                "## Model Formulation",
                "minimize setup, production, holding, and shortage cost",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
        generator_id="optmath_clsp_expand_capacity",
    )

    assert clsp_quality["status"] == "FAIL"
    assert any(
        issue.startswith("SEMANTIC_COHERENCE:SEMANTIC_FORBIDDEN_TERM:unmet demand")
        for issue in clsp_quality["issues"]
    )
    assert any(
        issue.startswith("SEMANTIC_COHERENCE:SEMANTIC_FORBIDDEN_TERM:shortage penalty")
        for issue in clsp_quality["issues"]
    )


def test_render_quality_allows_negated_forbidden_semantic_terms() -> None:
    quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                "A steel mill chooses product tons. There is no inventory, backlog, setup, sequencing, or time-period carryover.",
                "## Model Formulation",
                "max x subject to stage capacity constraints",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
        generator_id="optmath_steel4",
    )

    assert quality["status"] == "PASS"

    clsp_quality = _check_rendered_text(
        "\n\n".join(
            [
                "# Optimization Modeling Report",
                "## Problem",
                (
                    "After a short-term capacity shortage was identified, a factory chooses production lots. "
                    "No backlog, lost sales, unmet-demand slack, or shortage penalty variables are allowed."
                ),
                "## Model Formulation",
                "minimize setup, production, and holding cost under fixed capacity",
                "## Implementation",
                "```python\nprint('model')\n```",
                "## Optimal Solution",
                "The optimal objective value reported by the solver is `1`.",
                "## Feasibility Check",
                "The solution is feasible.",
                "## Validation",
                "The generated model solved with status `OPTIMAL`.",
            ]
        ),
        "final_model_report",
        generator_id="optmath_clsp_expand_capacity",
    )

    assert clsp_quality["status"] == "PASS"


def test_or_cpt_prompt_optimizer_routes_forward_eval_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rejected_path = run_dir / "08_forward_eval" / "rejected_pairs.jsonl"
    write_jsonl(
        rejected_path,
        [
            {
                "instance_id": "inst_1",
                "generator_id": "fixture",
                "rejection_stage": "forward_eval",
                "rejection_reason": "OBJECTIVE_MISMATCH",
                "problem_statement": "Choose production quantities to maximize profit.",
            }
        ],
    )

    diagnostics = analyze_run_for_prompt_optimization(run_dir)

    assert diagnostics["triggered"] is True
    assert diagnostics["target_prompt"] == "forward_modeling_prompt"


def test_or_cpt_prompt_optimizer_can_promote_with_registry(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "backtranslation_prompt.md").write_text("Original backtranslation prompt.", encoding="utf-8")
    (prompts_dir / "forward_modeling_prompt.md").write_text("Original forward prompt.", encoding="utf-8")
    versions_path = prompts_dir / "prompt_versions.yaml"
    versions_path.write_text(
        "\n".join(
            [
                "backtranslation_prompt:",
                "  version: v1.0.0",
                "  path: backtranslation_prompt.md",
                "forward_modeling_prompt:",
                "  version: v1.0.0",
                "  path: forward_modeling_prompt.md",
            ]
        ),
        encoding="utf-8",
    )
    registry = PromptRegistry(versions_path)
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "08_forward_eval" / "rejected_pairs.jsonl",
        [{"instance_id": "inst_1", "rejection_stage": "forward_eval", "rejection_reason": "OBJECTIVE_MISMATCH"}],
    )

    result = optimize_prompts_for_run(
        run_dir,
        EngineConfig(),
        auto_promote=True,
        mock=True,
        registry=registry,
    )

    assert result["status"] == "promoted"
    assert registry.get("forward_modeling_prompt").version == "v1.0.1"
    assert "Additional optimizer guidance" in (prompts_dir / "forward_modeling_prompt.md").read_text(encoding="utf-8")


def _make_fixture_generators(tmp_path: Path) -> Path:
    generator_dir = tmp_path / "generators" / "fixture_lp"
    generator_dir.mkdir(parents=True)
    (generator_dir / "metadata.json").write_text('{"model_type": "LP", "domain": ["fixture"]}', encoding="utf-8")
    (generator_dir / "generator.py").write_text(
        "\n".join(
            [
                "def generate(seed=None, difficulty_config=None):",
                "    return {",
                "        'model_type': 'LP',",
                "        'optimization_sense': 'maximize',",
                "        'num_variables': 1,",
                "        'num_constraints': 1,",
                "        'math_formula': 'max x subject to x <= 1, x >= 0',",
                "        'gurobi_code': " + repr(_tiny_gurobi_code()) + ",",
                "        'solver_status': 'OPTIMAL',",
                "        'objective_value': 1.0,",
                "    }",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path / "generators"


def _make_gurobi_model_generator(tmp_path: Path) -> Path:
    generator_dir = tmp_path / "generators" / "model_fixture"
    generator_dir.mkdir(parents=True)
    (generator_dir / "metadata.json").write_text(
        '{"model_type": "LP", "math_formula": "max x subject to x <= 1, x >= 0"}',
        encoding="utf-8",
    )
    (generator_dir / "generator.py").write_text(
        "\n".join(
            [
                "import gurobipy as gp",
                "from gurobipy import GRB",
                "",
                "class Generator:",
                "    def __init__(self, seed=None, parameters=None):",
                "        self.seed = seed",
                "        self.mathematical_formulation = 'max x subject to x <= 1, x >= 0'",
                "        self.parameters = {}",
                "",
                "    def generate_instance(self):",
                "        model = gp.Model('model_fixture')",
                "        model.Params.OutputFlag = 0",
                "        x = model.addVar(lb=0, name='x')",
                "        model.addConstr(x <= 1)",
                "        model.setObjective(x, GRB.MAXIMIZE)",
                "        self.parameters = {",
                "            'compact_cell_tower_tables': {",
                "                'budget': 1,",
                "                'tower_table': [{'tower': 'tower_0', 'cost': 1}],",
                "            },",
                "        }",
                "        return model",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path / "generators"


def _quality_fixture_row(
    *,
    instance_id: str,
    lp_text: str,
    variable_values: list[tuple[str, float, float, float]],
    all_vars_at_ub: bool,
    binding_count: int,
) -> dict:
    variable_diagnostics = [
        {
            "name": name,
            "value": value,
            "lb": lb,
            "ub": ub,
            "vtype": "C",
            "obj": 1.0,
            "at_lb": abs(value - lb) <= 1e-6,
            "at_ub": abs(value - ub) <= 1e-6,
            "is_nonzero": abs(value) > 1e-9,
            "is_fixed": False,
        }
        for name, value, lb, ub in variable_values
    ]
    nonzero_values = {name: value for name, value, _, _ in variable_values if abs(value) > 1e-9}
    return {
        "instance_id": instance_id,
        "generator_id": "fixture",
        "model_type": "LP",
        "difficulty_level": "level_1",
        "lp_text": lp_text,
        "reference_answer": {"objective_value": 1.0, "status": "OPTIMAL"},
        "solver_validation": {
            "status": "OPTIMAL",
            "objective_value": 1.0,
            "solution": {
                "objective_value": 1.0,
                "objective_recomputed": 1.0,
                "variable_count": len(variable_values),
                "nonzero_variable_count": len(nonzero_values),
                "nonzero_variable_values": nonzero_values,
                "variable_diagnostics": variable_diagnostics,
                "constraint_diagnostics": [
                    {"name": f"c{i}", "sense": "<", "rhs": 1.0, "slack": 0.0 if i < binding_count else 1.0, "is_binding": i < binding_count}
                    for i in range(max(1, binding_count))
                ],
                "feasibility": {"max_violation": 0.0, "is_feasible_within_tolerance": True},
                "solution_summary": {
                    "non_fixed_variable_count": len(variable_values),
                    "non_fixed_nonzero_variable_count": len(nonzero_values),
                    "all_vars_at_lb": False,
                    "all_vars_at_ub": all_vars_at_ub,
                    "all_non_fixed_vars_zero": False,
                    "binding_constraint_count": binding_count,
                    "constraint_count": max(1, binding_count),
                },
            },
        },
    }


def _make_stage_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        "backtranslation": run_dir / "05_backtranslation",
        "nl_filter": run_dir / "06_nl_quality_filter",
        "forward_modeling": run_dir / "07_forward_modeling",
        "forward_eval": run_dir / "08_forward_eval",
        "rendering": run_dir / "09_cpt_rendering",
        "export": run_dir / "10_train_val_export",
    }


def _tiny_gurobi_code() -> str:
    return "\n".join(
        [
            "import gurobipy as gp",
            "from gurobipy import GRB",
            "model = gp.Model('tiny_lp')",
            "model.Params.OutputFlag = 0",
            "x = model.addVar(lb=0, name='x')",
            "model.addConstr(x <= 1)",
            "model.setObjective(x, GRB.MAXIMIZE)",
            "model.optimize()",
        ]
    )
