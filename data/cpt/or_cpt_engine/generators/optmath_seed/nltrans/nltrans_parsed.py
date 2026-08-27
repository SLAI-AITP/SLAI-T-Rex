import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Transportation optimization problem.
        Parameters:
            parameters (dict): Dictionary containing:
                - n_origins: Number of origins
                - n_destinations: Number of destinations
                - supply_range: Tuple of (min, max) for supply amounts of origins
                - demand_range: Tuple of (min, max) for demand amounts of destinations
                - rate_range: Tuple of (min, max) for shipment rates (costs)
                - limit_range: Tuple of (min, max) for shipment limits
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "transportation"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Given:
        - Origins `i ∈ Origins` with supply capacities `Supply_i`
        - Destinations `j ∈ Destinations` with demand requirements `Demand_j`
        - Shipping rates (cost per unit) `Rate_{i,j}` between origins `i` and destinations `j`
        - Limit constraints `Limit_{i,j}` for the amount shipped
        
        Variables:
        - `Shipping_{i,j}`: Amount to be shipped from origin `i` to destination `j`
        
        Objective:
        Minimize the total transportation cost:
        $$\min \sum_{i \in Origins} \sum_{j \in Destinations} Rate_{i,j} \cdot Shipping_{i,j}$$

        Subject to:
        - Supply constraint: $ \sum_{j \in Destinations} Shipping_{i,j} = Supply_{i} $
        - Demand constraint: $ \sum_{i \in Origins} Shipping_{i,j} = Demand_{j} $
        - Shipping limits: $ Shipping_{i,j} \leq Limit_{i,j} $
        """
        default_parameters = {
            "n_origins": (3, 5),
            "n_destinations": (3, 5),
            "supply_range": (20, 50),
            "demand_range": (20, 50),
            "rate_range": (1, 10),
            "limit_slack_range": (5, 20),
        }
        # Use default parameters if none are provided or empty dict is provided
        if parameters is None or not parameters:
            parameters = default_parameters
        else:
            parameters = {**default_parameters, **dict(parameters)}
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Transportation problem instance and create its corresponding Gurobi model.
        This method does two things:
        1. Generates random problem data (origins, destinations, supplies, demands, rates, and limits)
        2. Creates and returns a configured Gurobi model ready to solve

        Returns:
            gp.Model: Configured Gurobi model for the transportation problem
        """
        # Randomly select number of origins and destinations
        self.n_origins = _sample_int(self.n_origins)
        self.n_destinations = _sample_int(self.n_destinations)

        # Generate origins and destinations
        origins = [f"origin_{i}" for i in range(self.n_origins)]
        destinations = [f"destination_{j}" for j in range(self.n_destinations)]

        # Generate a hidden feasible transportation plan first, then derive
        # balanced positive supply/demand and lane upper bounds from it.
        latent_shipment = {(origin, destination): 0 for origin in origins for destination in destinations}
        for origin in origins:
            destination = random.choice(destinations)
            latent_shipment[origin, destination] += random.randint(*self.supply_range)
        for destination in destinations:
            if sum(latent_shipment[origin, destination] for origin in origins) == 0:
                origin = random.choice(origins)
                latent_shipment[origin, destination] += random.randint(*self.demand_range)
        extra_lanes = max(len(origins), len(destinations))
        for _ in range(extra_lanes):
            origin = random.choice(origins)
            destination = random.choice(destinations)
            latent_shipment[origin, destination] += random.randint(1, max(2, self.supply_range[1] // 3))

        supply = {
            origin: sum(latent_shipment[origin, destination] for destination in destinations)
            for origin in origins
        }
        demand = {
            destination: sum(latent_shipment[origin, destination] for origin in origins)
            for destination in destinations
        }

        # Generate shipment rates and lane capacities.
        rates = {(origin, destination): random.randint(*self.rate_range) for origin in origins for destination in destinations}
        limits = {
            (origin, destination): latent_shipment[origin, destination] + random.randint(*self.limit_slack_range)
            for origin in origins
            for destination in destinations
        }
        self.parameters.update(
            {
                "origins": origins,
                "destinations": destinations,
                "supply": supply,
                "demand": demand,
                "rates": {f"{origin}|{destination}": rates[origin, destination] for origin in origins for destination in destinations},
                "lane_limits": {f"{origin}|{destination}": limits[origin, destination] for origin in origins for destination in destinations},
                "compact_transportation_tables": {
                    "sets": {
                        "origins": origins,
                        "destinations": destinations,
                    },
                    "origin_table": {
                        "columns": ["origin", "supply"],
                        "rows": [[origin, supply[origin]] for origin in origins],
                    },
                    "destination_table": {
                        "columns": ["destination", "demand"],
                        "rows": [[destination, demand[destination]] for destination in destinations],
                    },
                    "lane_table": {
                        "columns": ["origin", "destination", "unit_shipping_rate", "lane_capacity"],
                        "rows": [
                            [origin, destination, rates[origin, destination], limits[origin, destination]]
                            for origin in origins
                            for destination in destinations
                        ],
                    },
                    "required_decision_layers": [
                        "Shipping[i,j] continuous nonnegative shipment from origin i to destination j",
                    ],
                },
                "total_supply": sum(supply.values()),
                "total_demand": sum(demand.values()),
                "decision_variables": {
                    "Shipping[i,j]": "continuous nonnegative units shipped from origin i to destination j",
                },
                "objective_terms": [
                    "lane_rate_times_shipping_quantity",
                ],
                "required_constraints": [
                    "origin_supply_exactly_shipped",
                    "destination_demand_exactly_received",
                    "lane_capacity_limit",
                    "nonnegative_shipping",
                ],
                "required_parameter_presentation": [
                    "list origins and positive supply for every origin",
                    "list destinations and positive demand for every destination",
                    "list per-unit shipping rate for every origin-destination lane",
                    "list lane capacity limit for every origin-destination lane",
                    "state total supply equals total demand",
                    "state every origin's full supply is shipped exactly",
                    "state every destination's demand is received exactly",
                    "state there are no vehicles, routes, time windows, or subtour constraints",
                ],
                "structured_problem_data": {
                    "sets": {
                        "origins": origins,
                        "destinations": destinations,
                    },
                    "parameters": {
                        "supply": supply,
                        "demand": demand,
                        "total_supply": sum(supply.values()),
                        "total_demand": sum(demand.values()),
                        "rate_table": "see top-level rates[origin|destination]",
                        "lane_limit_table": "see top-level lane_limits[origin|destination]",
                    },
                    "constraints": {
                        "supply": "for every origin, the sum shipped to all destinations equals its supply",
                        "demand": "for every destination, the sum received from all origins equals its demand",
                        "lane_capacity": "each lane shipment is at most its lane capacity limit",
                    },
                    "objective": "minimize total per-unit shipping cost",
                    "generation_note": "An internal feasible shipment was used only to derive balanced supply, demand, and lane limits; it is not exposed as a required solution.",
                },
                "business_interpretation_guardrails": [
                    "This is a capacitated balanced transportation LP, not a vehicle-routing problem.",
                    "Do not add vehicles, tours, subtour elimination, time windows, or route sequencing.",
                    "Do not weaken origin supply equality into an optional unused-supply capacity model.",
                    "Do not weaken destination demand equality into a lower-bound or unmet-demand model.",
                    "Do not omit lane capacity limits.",
                    "Do not reveal or constrain the model to any internally generated feasible shipment certificate.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("Transportation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create decision variables (continuous shipping amounts)
        shipping = model.addVars(origins, destinations, lb=0, vtype=GRB.CONTINUOUS, name="Shipping")

        # Set objective: minimize total transportation cost
        model.setObjective(
            gp.quicksum(rates[origin, destination] * shipping[origin, destination] for origin in origins for destination in destinations),
            GRB.MINIMIZE
        )

        # Add supply constraints
        model.addConstrs(
            (gp.quicksum(shipping[origin, destination] for destination in destinations) == supply[origin] for origin in origins),
            name="SupplyConstraint"
        )

        # Add demand constraints
        model.addConstrs(
            (gp.quicksum(shipping[origin, destination] for origin in origins) == demand[destination] for destination in destinations),
            name="DemandConstraint"
        )

        # Add limit constraints
        model.addConstrs(
            (shipping[origin, destination] <= limits[origin, destination] for origin in origins for destination in destinations),
            name="LimitConstraint"
        )

        return model


def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


if __name__ == '__main__':
    import time

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        model.write("transportation.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
