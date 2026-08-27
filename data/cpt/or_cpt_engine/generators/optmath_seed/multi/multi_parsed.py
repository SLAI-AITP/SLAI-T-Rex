import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Multi-Commodity Transportation optimization problem.
        Parameters:
            parameters (dict): Dictionary containing:
                - n_origins: Range for the number of origins (min, max)
                - n_destinations: Range for the number of destinations (min, max)
                - n_products: Range for the number of products (min, max)
                - cost_range: Tuple of (min, max) for shipping costs
                - supply_range: Tuple of (min, max) for supply quantities
                - demand_range: Tuple of (min, max) for demand quantities
                - limit_range: Tuple of (min, max) for shipment capacity
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "multi_commodity_transportation"
        self.mathematical_formulation = r"""
        ### Mathematical Model\n
        A Multi-Commodity Transportation optimization problem involves:\n
        - Sets:
            - Origins: Set of origins (i)
            - Destinations: Set of destinations (j)
            - Products: Set of products (p)
        - Parameters:
            - **Supply** (Supply_{i,p}): Amounts each origin i can supply of product p
            - **Demand** (Demand_{j,p}): Amounts each destination j requires of product p
            - **ShippingCost** (ShippingCost_{i,j,p}): Cost to ship unit of product p from origin i to destination j
            - **Limit** (Limit_{i,j}): Max total units shipped from origin i to destination j
        - Decision variable:
            - **Transport** (Transport_{i,j,p}): Number of units of product p shipped from origin i to destination j
        - Objective:
            - Minimize total cost of shipping, subject to supply, demand, and shipment capacity constraints.
        """
        default_parameters = {
            "n_origins": (3, 4),
            "n_destinations": (4, 5),
            "n_products": (2, 2),
            "cost_range": (1, 10),
            "demand_range": (20, 70),
            "limit_slack_range": (5, 30),
            "inactive_lane_limit_range": (0, 20)
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
        Generate a Multi-Commodity Transportation problem instance and create its corresponding Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the Multi-Commodity Transportation Problem.
        """
        # Randomly determine number of origins, destinations, and products
        if isinstance(self.n_origins, (tuple, list)):
            self.n_origins = random.randint(*self.n_origins)
        if isinstance(self.n_destinations, (tuple, list)):
            self.n_destinations = random.randint(*self.n_destinations)
        if isinstance(self.n_products, (tuple, list)):
            self.n_products = random.randint(*self.n_products)

        # Generate sets
        origins = [f"origin_{i}" for i in range(self.n_origins)]
        destinations = [f"destination_{j}" for j in range(self.n_destinations)]
        products = [f"product_{p}" for p in range(self.n_products)]

        shipping_cost = {(i, j, p): random.randint(*self.cost_range) for i in origins for j in destinations for p in products}

        # Generate a latent feasible multi-commodity shipment plan first, then
        # derive supply, demand, and joint lane limits from it. This avoids the
        # old repair-by-truncation pattern, which could create zero or ambiguous
        # product rows that were hard for later LLM stages to reconstruct.
        latent_flow = {(i, j, p): 0 for i in origins for j in destinations for p in products}
        demand = {}
        for p in products:
            for j in destinations:
                remaining = random.randint(*self.demand_range)
                demand[j, p] = remaining
                candidate_origins = random.sample(origins, k=random.randint(1, min(3, len(origins))))
                for origin_index, i in enumerate(candidate_origins):
                    if origin_index == len(candidate_origins) - 1:
                        amount = remaining
                    else:
                        amount = random.randint(0, remaining)
                    latent_flow[i, j, p] += amount
                    remaining -= amount
        supply = {
            (i, p): sum(latent_flow[i, j, p] for j in destinations)
            for i in origins
            for p in products
        }
        limit = {}
        for i in origins:
            for j in destinations:
                latent_lane_total = sum(latent_flow[i, j, p] for p in products)
                if latent_lane_total:
                    limit[i, j] = latent_lane_total + random.randint(*self.limit_slack_range)
                else:
                    limit[i, j] = random.randint(*self.inactive_lane_limit_range)

        total_supply_by_product = {p: sum(supply[i, p] for i in origins) for p in products}
        total_demand_by_product = {p: sum(demand[j, p] for j in destinations) for p in products}
        active_lane_count = sum(1 for (i, j), cap in limit.items() if cap > 0)
        positive_latent_lane_count = sum(
            1 for i in origins for j in destinations if sum(latent_flow[i, j, p] for p in products) > 0
        )

        self.parameters.update(
            {
                "n_origins": self.n_origins,
                "n_destinations": self.n_destinations,
                "n_products": self.n_products,
                "origins": origins,
                "destinations": destinations,
                "products": products,
                "supply": {
                    i: {p: supply[i, p] for p in products}
                    for i in origins
                },
                "demand": {
                    j: {p: demand[j, p] for p in products}
                    for j in destinations
                },
                "lane_limits": {
                    i: {j: limit[i, j] for j in destinations}
                    for i in origins
                },
                "shipping_cost": {
                    i: {
                        j: {p: shipping_cost[i, j, p] for p in products}
                        for j in destinations
                    }
                    for i in origins
                },
                "compact_multi_commodity_transportation_tables": {
                    "sets": {
                        "origins": origins,
                        "destinations": destinations,
                        "products": products,
                    },
                    "origin_product_supply_table": {
                        "columns": ["origin", "product", "supply"],
                        "rows": [
                            [origin, product, supply[origin, product]]
                            for origin in origins
                            for product in products
                        ],
                    },
                    "destination_product_demand_table": {
                        "columns": ["destination", "product", "demand"],
                        "rows": [
                            [destination, product, demand[destination, product]]
                            for destination in destinations
                            for product in products
                        ],
                    },
                    "lane_capacity_table": {
                        "columns": ["origin", "destination", "shared_lane_capacity"],
                        "rows": [
                            [origin, destination, limit[origin, destination]]
                            for origin in origins
                            for destination in destinations
                        ],
                    },
                    "shipping_cost_table": {
                        "columns": ["origin", "destination", "product", "unit_shipping_cost"],
                        "rows": [
                            [origin, destination, product, shipping_cost[origin, destination, product]]
                            for origin in origins
                            for destination in destinations
                            for product in products
                        ],
                    },
                    "required_decision_layers": [
                        "Transport[i,j,p] continuous nonnegative units of product p shipped from origin i to destination j",
                    ],
                },
                "total_supply_by_product": total_supply_by_product,
                "total_demand_by_product": total_demand_by_product,
                "capacity_summary": {
                    "positive_latent_lane_count": positive_latent_lane_count,
                    "active_lane_count": active_lane_count,
                    "complete_lane_count": len(origins) * len(destinations),
                    "lane_capacity_is_shared_across_products": True,
                },
                "decision_variables": {
                    "Transport[i,j,p]": "continuous nonnegative units of product p shipped from origin i to destination j",
                },
                "objective_terms": {
                    "shipping_cost": "sum_i,j,p shipping_cost[i,j,p] * Transport[i,j,p]",
                    "sense": "minimize",
                },
                "required_constraints": [
                    "origin_product_supply_exactly_shipped",
                    "destination_product_demand_exactly_received",
                    "origin_destination_joint_capacity_across_products",
                    "nonnegative_transport_quantities",
                ],
                "required_parameter_presentation": [
                    "list origins, destinations, and products",
                    "show supply for every origin-product pair",
                    "show demand for every destination-product pair",
                    "show shared lane capacity for every origin-destination pair",
                    "show unit shipping cost for every origin-destination-product triple",
                    "state supply equals demand separately for every product",
                    "state lane capacity is shared across all products on the same origin-destination lane",
                ],
                "structured_problem_data": {
                    "sets": {
                        "origins": origins,
                        "destinations": destinations,
                        "products": products,
                    },
                    "parameters": {
                        "supply_by_origin_product": "see top-level supply table",
                        "demand_by_destination_product": "see top-level demand table",
                        "shared_lane_capacity_by_origin_destination": "see top-level lane_limits table",
                        "unit_shipping_cost_by_origin_destination_product": "see top-level shipping_cost table",
                    },
                    "constraints": {
                        "supply": "sum_j Transport[i,j,p] = supply[i,p] for every origin and product",
                        "demand": "sum_i Transport[i,j,p] = demand[j,p] for every destination and product",
                        "joint_lane_capacity": "sum_p Transport[i,j,p] <= lane_limit[i,j] for every origin-destination lane",
                    },
                    "objective": "minimize total unit shipping cost across all transported product units",
                    "generation_note": "An internal feasible multi-commodity shipment was used only to derive balanced supply, demand, and shared lane capacities; it is not exposed as a required solution.",
                },
                "business_interpretation_guardrails": [
                    "This is a continuous multi-commodity transportation LP.",
                    "It has no binary lane activation, no fixed costs, and no vehicle routes.",
                    "Do not collapse products into one aggregate commodity.",
                    "Do not turn supply or demand equalities into optional at-most or at-least constraints.",
                    "Lane capacity is shared across products on each origin-destination pair.",
                    "Do not add time periods, inventory carryover, backlog, service frequency, reserve penalties, or fleet capacity logic.",
                    "Do not reveal or constrain the model to any internally generated feasible flow certificate.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("Multi-Commodity Transportation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create variables
        transport = model.addVars(
            origins, destinations, products, vtype=GRB.CONTINUOUS, name="Transport"
        )

        # Set objective: minimize total shipping cost
        model.setObjective(
            gp.quicksum(
                shipping_cost[i, j, p] * transport[i, j, p]
                for i in origins for j in destinations for p in products
            ),
            GRB.MINIMIZE
        )

        # Add supply constraints
        for i in origins:
            for p in products:
                model.addConstr(
                    gp.quicksum(transport[i, j, p] for j in destinations) == supply[i, p],
                    name=f"Supply_{i}_{p}"
                )

        # Add demand constraints
        for j in destinations:
            for p in products:
                model.addConstr(
                    gp.quicksum(transport[i, j, p] for i in origins) == demand[j, p],
                    name=f"Demand_{j}_{p}"
                )

        # Add shipment limit constraints
        for i in origins:
            for j in destinations:
                model.addConstr(
                    gp.quicksum(transport[i, j, p] for p in products) <= limit[i, j],
                    name=f"Limit_{i}_{j}"
                )

        return model


if __name__ == "__main__":
    import time

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        model.write("multi_commodity_transportation.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
