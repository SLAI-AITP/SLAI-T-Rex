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
                - supply_range: Tuple of (min, max) for origin supplies
                - demand_range: Tuple of (min, max) for destination demands
                - cost_range: Tuple of (min, max) for transportation costs
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "transportation"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are m origins and n destinations. Each origin i (where i = 1, 2, ..., m) has a supply S_i, and each destination j (where j = 1, 2, ..., n) has a demand D_j. The cost of transporting one unit from origin i to destination j is C_{i,j}.

        Decision Variables:
        - x_{i,j}: Amount of goods transported from origin i to destination j.

        Objective:
        Minimize the total transportation cost:
        $$
        \text{Minimize } \sum_{i=1}^m \sum_{j=1}^n C_{i,j} \cdot x_{i,j}
        $$

        Constraints:
        1. Supply constraints: Total amount shipped from each origin must not exceed its supply.
        $$
        \sum_{j=1}^n x_{i,j} = S_i \quad \forall i = 1, 2, \ldots, m
        $$

        2. Demand constraints: Total amount shipped to each destination must meet its demand.
        $$
        \sum_{i=1}^m x_{i,j} = D_j \quad \forall j = 1, 2, \ldots, n
        $$

        3. Non-negativity constraints:
        $$
        x_{i,j} \geq 0 \quad \forall i = 1, 2, \ldots, m, \forall j = 1, 2, \ldots, n
        $$
        """
        default_parameters = {
            "n_origins": (3, 5),
            "n_destinations": (3, 5),
            "supply_range": (10, 100),
            "demand_range": (10, 100),
            "cost_range": (1, 10)
        }
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
        1. Generates random problem data (origins, destinations, supplies, demands, costs)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the transportation problem
        """
        
        # Randomly select number of origins and destinations
        self.n_origins = _sample_int(self.n_origins)
        self.n_destinations = _sample_int(self.n_destinations)
        
        # Generate origins and destinations
        self.origins = [f"origin_{i}" for i in range(self.n_origins)]
        self.destinations = [f"destination_{j}" for j in range(self.n_destinations)]
        
        # Build a hidden positive shipment table first, then derive balanced
        # supplies and demands. The hidden table is only a generation device and
        # is intentionally not exposed in LLM-facing parameters.
        latent_shipments = {
            (origin, destination): random.randint(*self.supply_range)
            for origin in self.origins
            for destination in self.destinations
        }
        self.supplies = {
            origin: sum(latent_shipments[origin, destination] for destination in self.destinations)
            for origin in self.origins
        }
        self.demands = {
            destination: sum(latent_shipments[origin, destination] for origin in self.origins)
            for destination in self.destinations
        }
        self.costs = {
            (origin, destination): random.randint(*self.cost_range)
            for origin in self.origins
            for destination in self.destinations
        }
        self.total_supply = sum(self.supplies.values())
        self.total_demand = sum(self.demands.values())
        assert self.total_supply == self.total_demand, "Total supply and demand must be equal"

        compact_transportation_tables = {
            "sets": {
                "origins": self.origins,
                "destinations": self.destinations,
            },
            "origin_supply_table": [
                {"origin": origin, "supply": self.supplies[origin]}
                for origin in self.origins
            ],
            "destination_demand_table": [
                {"destination": destination, "demand": self.demands[destination]}
                for destination in self.destinations
            ],
            "lane_cost_matrix": {
                "columns": self.destinations,
                "rows": [
                    {
                        "origin": origin,
                        "values": [self.costs[origin, destination] for destination in self.destinations],
                    }
                    for origin in self.origins
                ],
            },
            "decision_variable": "Transport[i,j] is the continuous nonnegative quantity shipped from origin i to destination j",
            "objective": "minimize sum of lane_cost[i,j] * Transport[i,j]",
            "constraints": [
                "for each origin i, sum_j Transport[i,j] equals supply[i]",
                "for each destination j, sum_i Transport[i,j] equals demand[j]",
            ],
            "source_contract_note": (
                "This is a balanced transportation LP on a complete origin-destination lane table. "
                "It has no vehicles, routes, time windows, lane capacities, fixed charges, or binary decisions."
            ),
        }
        self.parameters.update(
            {
                "origins": self.origins,
                "destinations": self.destinations,
                "supply": self.supplies,
                "demand": self.demands,
                "total_supply": self.total_supply,
                "total_demand": self.total_demand,
                "compact_transportation_tables": compact_transportation_tables,
                "generation_note": (
                    "Balanced supply and demand were derived from an internal positive shipment table, "
                    "but the internal shipment table is not part of the business problem and should not be presented."
                ),
                "decision_variables": {
                    "Transport[i,j]": "continuous nonnegative shipment quantity from origin i to destination j",
                },
                "objective_terms": [
                    "lane_unit_cost_times_transport_quantity",
                ],
                "required_constraints": [
                    "origin_supply_exactly_shipped",
                    "destination_demand_exactly_received",
                    "nonnegative_continuous_shipments",
                ],
                "required_parameter_presentation": [
                    "list origins and supply for every origin",
                    "list destinations and demand for every destination",
                    "list complete origin-destination lane cost matrix",
                    "state total supply equals total demand",
                    "state shipment variables are continuous and nonnegative",
                    "state every origin ships exactly its supply and every destination receives exactly its demand",
                    "prefer compact_transportation_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "origins": self.origins,
                        "destinations": self.destinations,
                    },
                    "parameters": {
                        "origin_supply": self.supplies,
                        "destination_demand": self.demands,
                        "total_supply": self.total_supply,
                        "total_demand": self.total_demand,
                        "lane_cost_matrix": "see compact_transportation_tables.lane_cost_matrix",
                    },
                    "constraints": {
                        "origin_supply": "sum of Transport[i,j] over destinations equals supply[i]",
                        "destination_demand": "sum of Transport[i,j] over origins equals demand[j]",
                    },
                    "objective": "minimize total lane shipping cost",
                    "generation_note": "no internal feasible shipment certificate should be included in the business statement",
                },
                "business_interpretation_guardrails": [
                    "This is a balanced transportation LP, not a vehicle-routing problem.",
                    "Do not add vehicles, depot tours, subtour elimination, service times, or time windows.",
                    "Do not add lane capacities, fixed charges, binary lane activation, or facility-opening decisions.",
                    "Do not relax exact supply and demand equalities into optional unused supply or unmet demand.",
                    "Do not expose or rely on the internally generated feasible shipment table.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("Transportation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (x[i,j] = amount transported from i to j)
        x = model.addVars(self.origins, self.destinations, vtype=GRB.CONTINUOUS, name="Transport")

        # Set objective: minimize total transportation cost
        model.setObjective(
            gp.quicksum(self.costs[i, j] * x[i, j] for i in self.origins for j in self.destinations),
            GRB.MINIMIZE
        )

        # Add supply constraints: total amount shipped from each origin must equal its supply
        for i in self.origins:
            model.addConstr(
                gp.quicksum(x[i, j] for j in self.destinations) == self.supplies[i],
                name=f"Supply_{i}"
            )

        # Add demand constraints: total amount shipped to each destination must equal its demand
        for j in self.destinations:
            model.addConstr(
                gp.quicksum(x[i, j] for i in self.origins) == self.demands[j],
                name=f"Demand_{j}"
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
