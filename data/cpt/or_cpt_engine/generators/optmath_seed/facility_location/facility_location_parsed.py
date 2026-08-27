import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded


def _sample_count(value):
    if isinstance(value, int):
        return value
    return random.randint(int(value[0]), int(value[1]))


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Facility Location optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_locations: Number of locations
                - n_commodities: Number of commodities
                - n_product_plants: Number of product plants
                - n_distribution_centers: Number of distribution centers
                - n_customer_zones: Number of customer zones
                - supply_range: Tuple of (min, max) for supply values
                - demand_range: Tuple of (min, max) for demand values
                - max_throughput_range: Tuple of (min, max) for maximum throughput values
                - min_throughput_range: Tuple of (min, max) for minimum throughput values
                - unit_throughput_cost_range: Tuple of (min, max) for unit throughput costs
                - fixed_throughput_cost_range: Tuple of (min, max) for fixed throughput costs
                - variable_cost_range: Tuple of (min, max) for variable costs
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "facility_location"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are:
        - Locations: \( \mathcal{L} \)
        - Commodities: \( \mathcal{C} \)
        - Product Plants: \( \mathcal{P} \subseteq \mathcal{L} \)
        - Distribution Centers: \( \mathcal{D} \subseteq \mathcal{L} \)
        - Customer Zones: \( \mathcal{Z} \subseteq \mathcal{L} \)

        Decision variables:
        - Shipped[c,p,d,z] >= 0: amount of commodity c shipped from plant p
          through distribution center d to customer zone z.
        - Selected[d] in {0,1}: whether distribution center d is opened.
        - Served[d,z] in {0,1}: whether customer zone z is assigned to center d.

        Minimize:
            variable shipping cost
            + fixed distribution-center selection cost
            + unit throughput cost on assigned zone demand.

        Subject to:
        - Supply capacity:
            sum_{d,z} Shipped[c,p,d,z] <= Supply[c,p].
        - Exact served demand:
            sum_p Shipped[c,p,d,z] = Demand[c,z] * Served[d,z].
        - Zone assignment:
            sum_d Served[d,z] = 1.
        - Throughput/opening link:
            MinThroughput[d] * Selected[d]
              <= sum_{c,z} Demand[c,z] * Served[d,z]
              <= MaxThroughput[d] * Selected[d].
        - Explicit open-before-assignment link:
            Served[d,z] <= Selected[d].
        - Facility-limit policy:
            not all candidate distribution centers may open in the generated
            small instances.
        """
        default_parameters = {
            "n_locations": (8, 9),
            "n_commodities": (2, 2),
            "n_product_plants": (2, 2),
            "n_distribution_centers": (3, 3),
            "n_customer_zones": (3, 3),
            "supply_range": (80, 320),
            "demand_range": (20, 80),
            "max_throughput_range": (350, 900),
            "min_throughput_range": (20, 160),
            "unit_throughput_cost_range": (1, 10),
            "fixed_throughput_cost_range": (600, 3000),
            "variable_cost_range": (1, 20)
        }
        parameters = {**default_parameters, **dict(parameters or {})}
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
        
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Facility Location problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (locations, commodities, product plants, distribution centers, customer zones, etc.)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the facility location problem
        """
        
        # Randomly select number of locations, commodities, product plants, distribution centers, and customer zones
        self.n_locations = _sample_count(self.n_locations)
        self.n_commodities = _sample_count(self.n_commodities)
        self.n_product_plants = _sample_count(self.n_product_plants)
        self.n_distribution_centers = _sample_count(self.n_distribution_centers)
        self.n_customer_zones = _sample_count(self.n_customer_zones)

        # Generate sets
        locations = [f"location_{i}" for i in range(self.n_locations)]
        commodities = [f"commodity_{i}" for i in range(self.n_commodities)]
        product_plants = [f"plant_{i}" for i in range(self.n_product_plants)]
        distribution_centers = [f"dc_{i}" for i in range(self.n_distribution_centers)]
        customer_zones = [f"zone_{i}" for i in range(self.n_customer_zones)]

        demand = {(c, z): random.randint(*self.demand_range) for c in commodities for z in customer_zones}
        supply = {}
        for c in commodities:
            required = sum(demand[c, z] for z in customer_zones)
            total_supply = int(required * uniform_rounded(1.10, 1.45)) + self.n_product_plants
            remaining_supply = total_supply
            for index, p in enumerate(product_plants):
                if index == len(product_plants) - 1:
                    supply[c, p] = remaining_supply
                else:
                    remaining_plants = len(product_plants) - index - 1
                    lower = 1
                    upper = max(lower, remaining_supply - remaining_plants)
                    share = random.randint(lower, upper)
                    supply[c, p] = share
                    remaining_supply -= share
        zone_throughput = {z: sum(demand[c, z] for c in commodities) for z in customer_zones}
        total_throughput = sum(zone_throughput.values())
        average_zone_throughput = max(1, total_throughput / max(1, len(customer_zones)))
        smallest_zone_throughput = max(1, min(zone_throughput.values()))
        max_throughput = {}
        min_throughput = {}
        hub = random.choice(distribution_centers)
        for d in distribution_centers:
            min_throughput[d] = max(0, int(smallest_zone_throughput * uniform_rounded(0.05, 0.25)))
            if d == hub:
                max_throughput[d] = int(total_throughput * uniform_rounded(1.00, 1.20))
            else:
                lower = max(max(zone_throughput.values()), int(total_throughput * 0.25))
                upper = max(lower, int(total_throughput * uniform_rounded(0.45, 0.85)))
                max_throughput[d] = random.randint(lower, upper)
        unit_throughput_cost = {d: random.randint(*self.unit_throughput_cost_range) for d in distribution_centers}
        fixed_throughput_cost = {d: random.randint(*self.fixed_throughput_cost_range) for d in distribution_centers}
        fixed_throughput_cost[hub] = max(self.fixed_throughput_cost_range[0], int(fixed_throughput_cost[hub] * 0.75))
        variable_cost = {(c, p, d, z): random.randint(*self.variable_cost_range) for c in commodities for p in product_plants for d in distribution_centers for z in customer_zones}
        cost_values = list(variable_cost.values())
        total_supply_by_commodity = {
            c: sum(supply[c, p] for p in product_plants) for c in commodities
        }
        total_demand_by_commodity = {
            c: sum(demand[c, z] for z in customer_zones) for c in commodities
        }
        self.parameters.update(
            {
                "commodities": commodities,
                "product_plants": product_plants,
                "distribution_centers": distribution_centers,
                "customer_zones": customer_zones,
                "supply": {f"{c}|{p}": supply[c, p] for c in commodities for p in product_plants},
                "demand": {f"{c}|{z}": demand[c, z] for c in commodities for z in customer_zones},
                "zone_throughput": zone_throughput,
                "total_throughput": total_throughput,
                "max_throughput": max_throughput,
                "min_throughput": min_throughput,
                "unit_throughput_cost": unit_throughput_cost,
                "fixed_throughput_cost": fixed_throughput_cost,
                "variable_shipping_cost": {
                    f"{c}|{p}|{d}|{z}": variable_cost[c, p, d, z]
                    for c in commodities
                    for p in product_plants
                    for d in distribution_centers
                    for z in customer_zones
                },
                "shipping_cost_table_shape": {
                    "commodities": len(commodities),
                    "product_plants": len(product_plants),
                    "distribution_centers": len(distribution_centers),
                    "customer_zones": len(customer_zones),
                    "entries": len(variable_cost),
                },
                "variable_shipping_cost_summary": {
                    "min": min(cost_values),
                    "max": max(cost_values),
                    "average": round(sum(cost_values) / len(cost_values), 2),
                },
                "throughput_bounds_by_dc": {
                    d: {
                        "min_throughput": min_throughput[d],
                        "max_throughput": max_throughput[d],
                        "fixed_cost": fixed_throughput_cost[d],
                        "unit_throughput_cost": unit_throughput_cost[d],
                    }
                    for d in distribution_centers
                },
                "total_supply_by_commodity": total_supply_by_commodity,
                "total_demand_by_commodity": total_demand_by_commodity,
                "supply_slack_by_commodity": {
                    c: total_supply_by_commodity[c] - total_demand_by_commodity[c]
                    for c in commodities
                },
                "hub_distribution_center": hub,
                "decision_variables": [
                    "Shipped[c,p,d,z] continuous nonnegative amount of commodity c shipped from plant p through distribution center d to zone z",
                    "Selected[d] binary indicator that distribution center d is open",
                    "Served[d,z] binary indicator that customer zone z is assigned to distribution center d",
                ],
                "objective_terms": [
                    "plant_to_distribution_center_to_zone_variable_shipping_cost",
                    "fixed_distribution_center_selection_cost",
                    "unit_distribution_center_throughput_cost_on_assigned_zone_demand",
                ],
                "required_constraints": [
                    "plant_commodity_supply_capacity",
                    "served_zone_commodity_demand_exactly_shipped",
                    "distribution_center_minimum_throughput_if_selected",
                    "distribution_center_maximum_throughput_if_selected",
                    "each_customer_zone_assigned_to_exactly_one_distribution_center",
                    "explicit_distribution_center_open_before_zone_assignment",
                    "facility_limit_prevents_all_open_degenerate_solution",
                ],
                "compact_facility_location_tables": {
                    "sets": {
                        "commodities": commodities,
                        "product_plants": product_plants,
                        "distribution_centers": distribution_centers,
                        "customer_zones": customer_zones,
                    },
                    "supply_table": {
                        "columns": ["commodity", "plant", "supply"],
                        "rows": [
                            [c, p, supply[c, p]]
                            for c in commodities
                            for p in product_plants
                        ],
                    },
                    "demand_table": {
                        "columns": ["commodity", "zone", "demand"],
                        "rows": [
                            [c, z, demand[c, z]]
                            for c in commodities
                            for z in customer_zones
                        ],
                    },
                    "distribution_center_table": {
                        "columns": [
                            "distribution_center",
                            "fixed_selection_cost",
                            "unit_throughput_cost",
                            "minimum_throughput_if_selected",
                            "maximum_throughput_if_selected",
                        ],
                        "rows": [
                            [
                                d,
                                fixed_throughput_cost[d],
                                unit_throughput_cost[d],
                                min_throughput[d],
                                max_throughput[d],
                            ]
                            for d in distribution_centers
                        ],
                    },
                    "shipping_cost_table": {
                        "columns": ["commodity", "plant", "distribution_center", "zone", "unit_shipping_cost"],
                        "rows": [
                            [c, p, d, z, variable_cost[c, p, d, z]]
                            for c in commodities
                            for p in product_plants
                            for d in distribution_centers
                            for z in customer_zones
                        ],
                    },
                    "required_decision_layers": [
                        "Selected[d] binary distribution-center open decision",
                        "Served[d,z] binary zone-to-center assignment decision",
                        "Shipped[c,p,d,z] continuous plant-center-zone commodity shipment",
                    ],
                },
                "structured_problem_data": {
                    "sets": {
                        "commodities": "C",
                        "product_plants": "P",
                        "distribution_centers": "D",
                        "customer_zones": "Z",
                    },
                    "parameters": [
                        "Supply[c,p]",
                        "Demand[c,z]",
                        "VariableShippingCost[c,p,d,z]",
                        "FixedThroughputCost[d]",
                        "UnitThroughputCost[d]",
                        "MinThroughput[d]",
                        "MaxThroughput[d]",
                    ],
                    "constraints": [
                        "sum_{d,z} Shipped[c,p,d,z] <= Supply[c,p]",
                        "sum_p Shipped[c,p,d,z] = Demand[c,z] * Served[d,z]",
                        "sum_d Served[d,z] = 1",
                        "Served[d,z] <= Selected[d]",
                        "MinThroughput[d] * Selected[d] <= assigned throughput at d <= MaxThroughput[d] * Selected[d]",
                        "sum_d Selected[d] <= |D|-1",
                    ],
                },
                "required_parameter_presentation": [
                    "List commodities, product plants, distribution centers, and customer zones.",
                    "List commodity-plant supply and commodity-zone demand tables.",
                    "List each distribution center's fixed cost, unit throughput cost, minimum throughput, and maximum throughput.",
                    "List the variable shipping cost table indexed by commodity, plant, distribution center, and customer zone.",
                    "State each zone must be assigned to exactly one open distribution center.",
                    "State this is not a vehicle-routing or visit-sequencing problem.",
                ],
                "business_interpretation_guardrails": [
                    "This is a multi-commodity distribution-center selection and zone-assignment model.",
                    "Do not collapse the plant-distribution-center-zone shipment variable into a simple one-index assignment.",
                    "Do not omit fixed distribution-center selection costs or unit throughput costs.",
                    "Do not omit the explicit Served[d,z] <= Selected[d] open-before-assignment link.",
                    "Do not add vehicle routes, subtour constraints, or customer visit sequencing.",
                ],
                "generation_notes": [
                    "Default sizes are intentionally compact so the complete four-index shipping-cost table can be stated in natural language.",
                    "The hub distribution center can serve all zones, preserving feasibility even when not all centers may open.",
                    "FacilityLimit prevents the all-open degenerate solution in small generated instances.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("FacilityLocation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        shipped = model.addVars(commodities, product_plants, distribution_centers, customer_zones, vtype=GRB.CONTINUOUS, name="Shipped")
        selected = model.addVars(distribution_centers, vtype=GRB.BINARY, name="Selected")
        served = model.addVars(distribution_centers, customer_zones, vtype=GRB.BINARY, name="Served")

        # Set objective: minimize total cost
        model.setObjective(
            gp.quicksum(variable_cost[c, p, d, z] * shipped[c, p, d, z] for c in commodities for p in product_plants for d in distribution_centers for z in customer_zones) +
            gp.quicksum(fixed_throughput_cost[d] * selected[d] + unit_throughput_cost[d] * gp.quicksum(demand[c, z] * served[d, z] for c in commodities for z in customer_zones) for d in distribution_centers),
            GRB.MINIMIZE
        )

        # Add constraints
        # Supply constraint
        for c in commodities:
            for p in product_plants:
                model.addConstr(
                    gp.quicksum(shipped[c, p, d, z] for d in distribution_centers for z in customer_zones) <= supply[c, p],
                    name=f"Supply_{c}_{p}"
                )

        # Exact served-demand shipment constraint. If zone z is assigned to
        # distribution center d, every commodity demand for that zone must be
        # shipped through d; otherwise no shipment should use that d,z pair.
        for c in commodities:
            for d in distribution_centers:
                for z in customer_zones:
                    model.addConstr(
                        gp.quicksum(shipped[c, p, d, z] for p in product_plants) == demand[c, z] * served[d, z],
                        name=f"Demand_{c}_{d}_{z}"
                    )

        # Minimum throughput constraint
        for d in distribution_centers:
            model.addConstr(
                gp.quicksum(demand[c, z] * served[d, z] for c in commodities for z in customer_zones) >= selected[d] * min_throughput[d],
                name=f"MinThroughput_{d}"
            )

        # Maximum throughput constraint
        for d in distribution_centers:
            model.addConstr(
                gp.quicksum(demand[c, z] * served[d, z] for c in commodities for z in customer_zones) <= selected[d] * max_throughput[d],
                name=f"MaxThroughput_{d}"
            )

        # Explicit open-before-assignment linking helps downstream model
        # reconstruction avoid treating Served[d,z] as independent of Selected[d].
        for d in distribution_centers:
            for z in customer_zones:
                model.addConstr(
                    served[d, z] <= selected[d],
                    name=f"OpenBeforeServe_{d}_{z}",
                )

        # Allocation constraint
        for z in customer_zones:
            model.addConstr(
                gp.quicksum(served[d, z] for d in distribution_centers) == 1,
                name=f"Allocation_{z}"
            )

        # The generated hub can serve all zones if needed, so this operating
        # policy preserves feasibility while preventing all distribution centers
        # from opening in small instances.
        if len(distribution_centers) > 1:
            model.addConstr(
                gp.quicksum(selected[d] for d in distribution_centers) <= len(distribution_centers) - 1,
                name="FacilityLimit",
            )

        return model


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("facility_location.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
