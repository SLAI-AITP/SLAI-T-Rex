from __future__ import annotations

from typing import Any


DEFAULT_VARIATION_AXES: dict[str, list[dict[str, Any]]] = {
    "narrative_angles": [
        {
            "angle_id": "weekly_operating_plan",
            "description": "frame the instance as a recurring weekly operating plan with explicit service or cost targets",
            "business_triggers": ["during a weekly planning cycle", "before the next operating plan is locked"],
        },
        {
            "angle_id": "peak_demand_surge",
            "description": "frame the instance as a response to a peak-season or promotion-driven demand surge",
            "business_triggers": ["after demand rose above the baseline forecast", "during peak-season capacity planning"],
        },
        {
            "angle_id": "disruption_recovery",
            "description": "frame the instance as recovery planning after a supplier, facility, weather, or capacity disruption",
            "business_triggers": ["during disruption recovery planning", "after a temporary capacity disruption"],
        },
        {
            "angle_id": "budget_committee_review",
            "description": "frame the instance as a decision prepared for a budget or investment committee",
            "business_triggers": ["during a budget approval round", "before the finance committee approves spending"],
        },
        {
            "angle_id": "service_level_protection",
            "description": "frame the instance as protecting service levels while respecting scarce resources",
            "business_triggers": ["after service-level risk was flagged", "before committing customer service targets"],
        },
        {
            "angle_id": "compliance_or_policy_audit",
            "description": "frame the instance as meeting policy, coverage, nutrition, risk, or regulatory thresholds",
            "business_triggers": ["during a compliance review", "after policy thresholds were updated"],
        },
        {
            "angle_id": "resilience_stress_test",
            "description": "frame the instance as a stress test of a network, staffing plan, portfolio, or supply plan",
            "business_triggers": ["during a resilience stress test", "after planners requested a contingency case"],
        },
        {
            "angle_id": "capacity_rationing",
            "description": "frame the instance as allocating scarce capacity among competing requests",
            "business_triggers": ["after a capacity shortage was identified", "before scarce capacity is rationed"],
        },
        {
            "angle_id": "sustainability_or_waste_review",
            "description": "frame the instance as reducing waste, emissions, excess inventory, or avoidable operating cost",
            "business_triggers": ["during a sustainability review", "after waste reduction targets were refreshed"],
        },
        {
            "angle_id": "expansion_or_network_redesign",
            "description": "frame the instance as revising capacity, process, siting, routing, or network design only when those structures exist in the source",
            "business_triggers": ["during an operating redesign proposal", "before expansion funding is committed"],
        },
        {
            "angle_id": "contract_or_vendor_reset",
            "description": "frame the instance as resetting supplier, vendor, carrier, or service commitments",
            "business_triggers": ["after vendor capacity information changed", "before contract award recommendations are finalized"],
        },
        {
            "angle_id": "urgent_dispatch_window",
            "description": "frame the instance as a time-sensitive dispatch, schedule, loading, or routing decision",
            "business_triggers": ["before a dispatch cutoff", "after urgent orders changed the priority list"],
        },
    ],
    "organization_profiles": [
        {"profile_id": "regional_operator", "description": "a regional operator coordinating several sites or service areas"},
        {"profile_id": "multi_site_enterprise", "description": "a multi-site enterprise balancing resources across locations"},
        {"profile_id": "public_agency", "description": "a public agency accountable for coverage, fairness, and budget control"},
        {"profile_id": "third_party_service_provider", "description": "a third-party service provider planning against client commitments"},
        {"profile_id": "hospital_or_clinic_network", "description": "a healthcare network coordinating clinical or logistics capacity"},
        {"profile_id": "industrial_plant", "description": "an industrial plant planning production, material, or maintenance resources"},
        {"profile_id": "digital_platform", "description": "a digital platform matching supply, demand, and infrastructure capacity"},
        {"profile_id": "emergency_response_cell", "description": "an emergency response cell preparing a constrained allocation plan"},
    ],
    "planning_horizons": [
        {"horizon_id": "same_day_dispatch", "description": "a near-term operating decision before execution"},
        {"horizon_id": "daily_shift_plan", "description": "a short-horizon operating plan for the next service or production window"},
        {"horizon_id": "weekly_plan", "description": "a weekly replenishment, staffing, or production plan"},
        {"horizon_id": "monthly_capacity_plan", "description": "a monthly capacity and budget plan"},
        {"horizon_id": "quarterly_commitment", "description": "a quarterly commitment or contract planning cycle"},
        {"horizon_id": "seasonal_peak_window", "description": "a seasonal peak-demand window"},
        {"horizon_id": "incident_recovery_window", "description": "a short recovery window after an operational incident"},
        {"horizon_id": "strategic_rollout", "description": "a strategic rollout or network-design planning horizon"},
    ],
    "entity_naming_styles": [
        {"style_id": "indexed_business_labels", "description": "use indexed business labels such as site_0, customer_0, job_0, or product_0"},
        {"style_id": "role_plus_index", "description": "use role-based names such as Supplier 1, Clinic 2, Route 3, or Machine 4"},
        {"style_id": "regional_letters", "description": "use region-style labels such as North, South, East, West, and Central when the source size allows it"},
        {"style_id": "operational_codes", "description": "use realistic operating codes such as DC-1, SKU-2, Lane-3, or Crew-4"},
        {"style_id": "plain_tables", "description": "use compact tables and neutral row labels to keep every coefficient readable"},
        {"style_id": "departmental_names", "description": "use department or program labels when the source represents activities or projects"},
    ],
}


DEFAULT_FAMILY_SCENARIO_LENSES: dict[str, list[dict[str, Any]]] = {
    "assignment": [
        {"lens_id": "healthcare_staff_to_shifts", "industry": "health care and social assistance", "context": "assign qualified staff to shifts, clinics, rooms, or patient-service tasks", "entities": ["staff", "shifts", "service tasks", "eligibility scores"]},
        {"lens_id": "field_service_job_matching", "industry": "professional and technical services", "context": "match technicians, inspectors, or consultants to jobs based on skill and capacity", "entities": ["technicians", "jobs", "skills", "assignment limits"]},
        {"lens_id": "mobility_asset_matching", "industry": "transportation and warehousing", "context": "assign vehicles, drivers, aircraft, or equipment to requests", "entities": ["assets", "requests", "compatibility", "availability"]},
        {"lens_id": "digital_marketplace_matching", "industry": "information services", "context": "match providers, resources, or service slots to platform requests", "entities": ["providers", "requests", "capacity", "match scores"]},
    ],
    "scheduling": [
        {"lens_id": "factory_work_center_schedule", "industry": "manufacturing", "context": "schedule jobs across machines, lines, or work centers", "entities": ["jobs", "machines", "processing times", "completion targets"]},
        {"lens_id": "airport_or_transport_recovery", "industry": "transportation and warehousing", "context": "recover a time-sensitive transport schedule after delays", "entities": ["flights", "slots", "separation buffers", "delay costs"]},
        {"lens_id": "laboratory_batch_processing", "industry": "professional and technical services", "context": "sequence laboratory, testing, or quality-control batches through stations", "entities": ["batches", "stations", "durations", "due windows"]},
        {"lens_id": "maintenance_window_planning", "industry": "utilities and industrial services", "context": "schedule repair or maintenance work into constrained windows", "entities": ["work orders", "crews", "assets", "time windows"]},
    ],
    "packing": [
        {"lens_id": "parcel_or_truck_loading", "industry": "transportation and warehousing", "context": "select or pack freight into trucks, pallets, or containers", "entities": ["items", "loads", "capacities", "values"]},
        {"lens_id": "cloud_resource_placement", "industry": "information services", "context": "place virtual workloads onto limited compute or storage capacity", "entities": ["workloads", "servers", "resource usage", "capacity"]},
        {"lens_id": "capital_project_selection", "industry": "management of companies", "context": "select projects under a capital or operating budget", "entities": ["projects", "costs", "benefits", "budget"]},
        {"lens_id": "emergency_supply_kit_selection", "industry": "public administration", "context": "choose emergency kits or assets under transport capacity", "entities": ["kits", "weights", "priority values", "capacity"]},
    ],
    "cutting_stock": [
        {"lens_id": "metal_roll_cutting", "industry": "manufacturing", "context": "cut metal coils, bars, or sheets into customer order sizes", "entities": ["stock material", "order sizes", "patterns", "waste"]},
        {"lens_id": "textile_panel_cutting", "industry": "manufacturing", "context": "cut fabric rolls into garment or upholstery panels", "entities": ["rolls", "panel sizes", "demand", "leftover material"]},
        {"lens_id": "lumber_order_cutting", "industry": "construction", "context": "cut lumber, boards, or beams for construction orders", "entities": ["boards", "lengths", "orders", "waste"]},
        {"lens_id": "packaging_sheet_cutting", "industry": "wholesale and packaging services", "context": "cut sheets into packaging blanks for client orders", "entities": ["sheets", "blank sizes", "patterns", "demand"]},
    ],
    "blending": [
        {"lens_id": "refinery_fuel_blend", "industry": "manufacturing", "context": "blend feedstocks into a fuel product while meeting quality limits", "entities": ["feedstocks", "quality attributes", "product limits", "costs"]},
        {"lens_id": "food_recipe_blend", "industry": "accommodation and food services", "context": "blend ingredients for a recipe or food formulation", "entities": ["ingredients", "attributes", "recipe bounds", "costs"]},
        {"lens_id": "chemical_batch_mix", "industry": "manufacturing", "context": "mix chemical inputs into a batch with required composition", "entities": ["inputs", "composition values", "batch requirements", "costs"]},
        {"lens_id": "recycled_material_mix", "industry": "waste management and remediation services", "context": "combine recycled material streams under quality thresholds", "entities": ["streams", "quality measures", "blend limits", "processing costs"]},
    ],
    "diet": [
        {"lens_id": "hospital_nutrition_menu", "industry": "health care and social assistance", "context": "design a meal plan that satisfies nutrient bounds", "entities": ["foods", "nutrients", "nutrient contents", "costs"]},
        {"lens_id": "school_cafeteria_menu", "industry": "educational services", "context": "plan school meals under nutrition and budget requirements", "entities": ["menu items", "nutrients", "requirements", "costs"]},
        {"lens_id": "animal_feed_formula", "industry": "agriculture, forestry, fishing and hunting", "context": "blend feed ingredients to satisfy livestock nutrition requirements", "entities": ["feed ingredients", "nutrients", "bounds", "costs"]},
        {"lens_id": "meal_kit_nutrition_design", "industry": "retail trade", "context": "design a meal-kit formulation with nutritional targets", "entities": ["ingredients", "nutrients", "serving sizes", "costs"]},
    ],
    "facility_location": [
        {"lens_id": "retail_distribution_sites", "industry": "retail trade", "context": "open facilities and assign demand zones to serve customers", "entities": ["candidate facilities", "demand zones", "fixed costs", "service costs"]},
        {"lens_id": "healthcare_access_sites", "industry": "health care and social assistance", "context": "choose clinics, supply sites, or care hubs to serve communities", "entities": ["sites", "communities", "capacities", "service costs"]},
        {"lens_id": "telecom_or_mobility_coverage", "industry": "information and telecommunications", "context": "place towers, hubs, or stations to cover demand areas", "entities": ["candidate sites", "zones", "coverage", "budget"]},
        {"lens_id": "public_safety_infrastructure", "industry": "public administration", "context": "select public service sites under coverage and budget limits", "entities": ["stations", "districts", "coverage", "costs"]},
    ],
    "covering": [
        {"lens_id": "emergency_service_coverage", "industry": "public administration", "context": "select options so all districts or risks receive required coverage", "entities": ["options", "districts", "coverage matrix", "costs"]},
        {"lens_id": "cybersecurity_control_coverage", "industry": "professional and technical services", "context": "select controls so every risk category is covered", "entities": ["controls", "risk categories", "coverage", "costs"]},
        {"lens_id": "software_test_coverage", "industry": "information services", "context": "select tests or checks so all requirements are covered", "entities": ["tests", "requirements", "coverage matrix", "effort"]},
        {"lens_id": "quality_audit_coverage", "industry": "manufacturing", "context": "select inspections to cover product families or risk classes", "entities": ["inspections", "risk classes", "coverage", "costs"]},
    ],
    "portfolio": [
        {"lens_id": "investment_asset_portfolio", "industry": "finance and insurance", "context": "allocate capital across assets under return and risk limits", "entities": ["assets", "returns", "risks", "budget"]},
        {"lens_id": "rd_project_portfolio", "industry": "professional and technical services", "context": "select research projects under cost and risk constraints", "entities": ["projects", "costs", "benefits", "risk scores"]},
        {"lens_id": "public_grant_portfolio", "industry": "public administration", "context": "choose grant-funded initiatives under policy and budget limits", "entities": ["initiatives", "budgets", "impact scores", "policy limits"]},
        {"lens_id": "energy_upgrade_portfolio", "industry": "utilities", "context": "select infrastructure upgrades under budget and reliability targets", "entities": ["upgrades", "costs", "reliability benefits", "risk"]},
    ],
    "revenue_management": [
        {"lens_id": "airline_fare_capacity", "industry": "transportation and warehousing", "context": "allocate limited capacity among fare classes or booking products", "entities": ["fare classes", "capacity", "demand limits", "revenues"]},
        {"lens_id": "hotel_room_inventory", "industry": "accommodation and food services", "context": "allocate room inventory across customer segments", "entities": ["segments", "room nights", "demand caps", "prices"]},
        {"lens_id": "media_ad_inventory", "industry": "information services", "context": "allocate limited ad inventory among contract products", "entities": ["ad products", "impressions", "demand limits", "prices"]},
        {"lens_id": "subscription_capacity_sales", "industry": "information services", "context": "allocate subscription or service capacity among product tiers", "entities": ["tiers", "capacity", "demand", "revenue"]},
    ],
    "marketing": [
        {"lens_id": "consumer_promotion_mix", "industry": "retail trade", "context": "allocate promotions across markets or customer segments", "entities": ["markets", "promotions", "lift", "budget"]},
        {"lens_id": "regional_sales_strategy", "industry": "wholesale trade", "context": "choose regional initiatives to improve market share", "entities": ["regions", "initiatives", "share lift", "resources"]},
        {"lens_id": "digital_campaign_allocation", "industry": "information services", "context": "allocate digital campaign effort across channels", "entities": ["channels", "segments", "reach", "spend limits"]},
        {"lens_id": "public_outreach_campaign", "industry": "public administration", "context": "select outreach actions under coverage and budget limits", "entities": ["actions", "communities", "impact", "budget"]},
    ],
    "contract_allocation": [
        {"lens_id": "supplier_award_allocation", "industry": "manufacturing procurement", "context": "allocate volumes across qualified suppliers", "entities": ["suppliers", "lots", "capacities", "bid prices"]},
        {"lens_id": "public_service_contracts", "industry": "public administration", "context": "award regional service contracts while respecting policy rules", "entities": ["vendors", "regions", "eligibility", "costs"]},
        {"lens_id": "carrier_lane_commitments", "industry": "transportation and warehousing", "context": "allocate shipping lanes among carriers with capacity commitments", "entities": ["carriers", "lanes", "capacities", "rates"]},
        {"lens_id": "facility_services_procurement", "industry": "administrative and support services", "context": "allocate facility-service packages across contractors", "entities": ["contractors", "service lots", "limits", "prices"]},
    ],
    "supply_chain": [
        {"lens_id": "multi_echelon_distribution", "industry": "manufacturing and distribution", "context": "coordinate suppliers, plants, warehouses, and customer zones", "entities": ["suppliers", "warehouses", "customers", "flows"]},
        {"lens_id": "resilient_sourcing_plan", "industry": "wholesale trade", "context": "allocate demand after supplier availability changed", "entities": ["suppliers", "hubs", "demand zones", "capacities"]},
        {"lens_id": "healthcare_supply_network", "industry": "health care and social assistance", "context": "move medical supplies through a constrained care network", "entities": ["depots", "clinics", "supplies", "demand"]},
        {"lens_id": "grocery_replenishment_network", "industry": "retail trade", "context": "plan replenishment flows across stores and warehouses", "entities": ["stores", "warehouses", "inventory", "shipments"]},
    ],
    "network_design": [
        {"lens_id": "freight_lane_activation", "industry": "transportation and warehousing", "context": "activate lanes or hubs and route commodity flows", "entities": ["nodes", "arcs", "commodities", "fixed costs"]},
        {"lens_id": "telecom_capacity_network", "industry": "information and telecommunications", "context": "activate links and route traffic classes", "entities": ["nodes", "links", "traffic classes", "capacities"]},
        {"lens_id": "water_or_utility_network", "industry": "utilities", "context": "choose network links and route utility flow", "entities": ["junctions", "links", "flows", "capacities"]},
        {"lens_id": "transit_line_portfolio", "industry": "public transit", "context": "select transit lines or corridors under service targets", "entities": ["lines", "corridors", "demand", "costs"]},
    ],
    "network_flow": [
        {"lens_id": "distribution_arc_flow", "industry": "transportation and warehousing", "context": "route goods through a network of arcs", "entities": ["nodes", "arcs", "supplies", "costs"]},
        {"lens_id": "data_traffic_routing", "industry": "information services", "context": "route traffic through network links", "entities": ["routers", "links", "traffic", "latencies"]},
        {"lens_id": "water_or_energy_flow", "industry": "utilities", "context": "route flow through utility infrastructure", "entities": ["junctions", "pipes or lines", "capacity", "costs"]},
        {"lens_id": "relief_corridor_flow", "industry": "public administration", "context": "move relief supplies through available corridors", "entities": ["depots", "corridors", "shelters", "transport costs"]},
    ],
    "routing": [
        {"lens_id": "single_route_last_mile_tour", "industry": "transportation and warehousing", "context": "plan a depot-to-depot route through required customer stops under capacity or time rules", "entities": ["route asset", "customers", "depot", "travel times"]},
        {"lens_id": "medical_courier_tour", "industry": "health care and social assistance", "context": "plan a courier tour among clinics, labs, or pharmacies", "entities": ["courier", "clinics", "time windows", "samples"]},
        {"lens_id": "field_service_tour", "industry": "professional and technical services", "context": "plan a service tour through appointment jobs", "entities": ["technician", "jobs", "travel costs", "time windows"]},
        {"lens_id": "inspection_or_audit_tour", "industry": "administrative and support services", "context": "plan an inspection tour across required sites", "entities": ["sites", "distances", "visit requirements", "route"]},
    ],
    "transportation": [
        {"lens_id": "warehouse_to_store_replenishment", "industry": "retail trade", "context": "ship goods from warehouses to stores at minimum cost", "entities": ["warehouses", "stores", "supply", "demand"]},
        {"lens_id": "relief_depot_to_shelter", "industry": "public administration", "context": "allocate relief supplies from depots to shelters or clinics", "entities": ["depots", "shelters", "available kits", "requests"]},
        {"lens_id": "plant_to_region_distribution", "industry": "manufacturing", "context": "ship production from plants to regional demand centers", "entities": ["plants", "regions", "capacities", "lane costs"]},
        {"lens_id": "food_bank_allocation", "industry": "social assistance", "context": "distribute food supplies from warehouses to community sites", "entities": ["food banks", "community sites", "supply", "demand"]},
    ],
    "lot_sizing": [
        {"lens_id": "retail_replenishment_cycle", "industry": "retail trade", "context": "plan orders, inventory, and backlog across periods", "entities": ["periods", "demand", "orders", "inventory"]},
        {"lens_id": "factory_lot_release", "industry": "manufacturing", "context": "schedule production lots while balancing setup and holding costs", "entities": ["products", "periods", "setup costs", "holding costs"]},
        {"lens_id": "service_parts_planning", "industry": "maintenance services", "context": "order spare parts over time under demand and backlog penalties", "entities": ["parts", "periods", "demand", "backlog"]},
        {"lens_id": "packaging_line_capacity", "industry": "manufacturing", "context": "plan production and possible capacity expansion over periods", "entities": ["lines", "periods", "capacity", "inventory"]},
    ],
    "production_planning": [
        {"lens_id": "factory_product_mix", "industry": "manufacturing", "context": "choose product quantities under machine and labor limits", "entities": ["products", "resources", "capacities", "margins"]},
        {"lens_id": "steel_or_materials_plan", "industry": "manufacturing", "context": "plan output across material grades and process resources", "entities": ["grades", "processes", "capacities", "profits"]},
        {"lens_id": "order_fulfillment_mix", "industry": "wholesale trade", "context": "choose a production or fulfillment mix under limited resources", "entities": ["orders", "products", "resources", "costs"]},
        {"lens_id": "multi_period_factory_plan", "industry": "manufacturing", "context": "plan production, sales, and inventory over time under machine-hour limits", "entities": ["products", "periods", "machines", "sales limits", "inventory targets"]},
    ],
    "agriculture": [
        {"lens_id": "crop_acreage_plan", "industry": "agriculture, forestry, fishing and hunting", "context": "allocate acreage and resources across crops", "entities": ["crops", "land", "water", "labor"]},
        {"lens_id": "livestock_feed_resource_plan", "industry": "agriculture, forestry, fishing and hunting", "context": "balance feed, land, and labor resources", "entities": ["feed options", "nutrients", "animals", "resources"]},
        {"lens_id": "greenhouse_production_mix", "industry": "agriculture", "context": "plan greenhouse crop quantities under space and labor limits", "entities": ["crops", "greenhouse space", "labor", "profit"]},
        {"lens_id": "irrigation_limited_farm_plan", "industry": "agriculture", "context": "allocate scarce water and land across production activities", "entities": ["fields", "water", "activities", "returns"]},
    ],
    "energy": [
        {"lens_id": "generation_dispatch", "industry": "utilities", "context": "dispatch generators to meet demand at minimum cost", "entities": ["generators", "demand", "capacity", "costs"]},
        {"lens_id": "microgrid_resource_schedule", "industry": "utilities", "context": "schedule local generation resources in a microgrid", "entities": ["resources", "periods", "demand", "capacity"]},
        {"lens_id": "renewable_capacity_allocation", "industry": "utilities", "context": "allocate renewable and backup capacity under budget or reliability limits", "entities": ["resources", "capacities", "costs", "requirements"]},
        {"lens_id": "industrial_energy_procurement", "industry": "manufacturing", "context": "choose energy procurement quantities under demand and contract limits", "entities": ["sources", "contracts", "demand", "prices"]},
    ],
    "workforce_deployment": [
        {"lens_id": "mission_staff_deployment", "industry": "public administration", "context": "deploy personnel groups to missions under readiness limits", "entities": ["personnel groups", "missions", "readiness", "capacity"]},
        {"lens_id": "training_assignment_plan", "industry": "educational services", "context": "assign personnel to training exercises or certification programs", "entities": ["participants", "exercises", "skills", "limits"]},
        {"lens_id": "public_safety_staffing", "industry": "public administration", "context": "deploy public safety teams across districts", "entities": ["teams", "districts", "coverage", "availability"]},
        {"lens_id": "healthcare_surge_staffing", "industry": "health care and social assistance", "context": "deploy clinical staff during a surge period", "entities": ["staff groups", "units", "demand", "availability"]},
    ],
    "dispersion": [
        {"lens_id": "sensor_site_diversity", "industry": "utilities and monitoring", "context": "select sensor sites to maximize spatial separation or coverage diversity", "entities": ["sites", "distances", "selected sensors", "coverage"]},
        {"lens_id": "emergency_cache_spacing", "industry": "public administration", "context": "place emergency caches far apart to improve resilience", "entities": ["candidate sites", "distances", "caches", "selection count"]},
        {"lens_id": "retail_site_separation", "industry": "retail trade", "context": "select retail sites with sufficient market separation", "entities": ["candidate sites", "distances", "stores", "selection count"]},
        {"lens_id": "habitat_or_conservation_sites", "industry": "environmental services", "context": "select conservation sites to maximize ecological spread", "entities": ["sites", "distances", "selected areas", "budget"]},
    ],
    "general_linear_programming": [
        {"lens_id": "enterprise_activity_mix", "industry": "management of companies", "context": "choose activity levels under multiple resource limits", "entities": ["activities", "resources", "coefficients", "bounds"]},
        {"lens_id": "service_capacity_mix", "industry": "administrative and support services", "context": "allocate service capacity among activity categories", "entities": ["services", "capacities", "requirements", "costs"]},
        {"lens_id": "public_program_mix", "industry": "public administration", "context": "allocate program levels under budget and policy constraints", "entities": ["programs", "budgets", "requirements", "benefits"]},
        {"lens_id": "industrial_resource_mix", "industry": "manufacturing", "context": "balance production activities with limited resource availability", "entities": ["activities", "resource limits", "returns", "costs"]},
    ],
}
