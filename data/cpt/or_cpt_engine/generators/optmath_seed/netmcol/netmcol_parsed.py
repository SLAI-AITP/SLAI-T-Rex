import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Multi-Commodity Network Flow optimization problem.

        Parameters:
            parameters (dict): Problem parameters.
            seed (int, optional): Random seed for reproducibility.
        """
        self.problem_type = "multi_commodity_network_flow"
        default_parameters = {
            "n_cities": (4, 6),                # Cities range (minimum, maximum)
            "n_products": (2, 3),              # Products range (minimum, maximum)
            "shipment_range": (10, 40),        # Hidden feasible shipment quantities
            "shipment_cost_range": (1, 10),    # Shipment cost range
            "capacity_slack_range": (0, 8),    # Lane capacity slack over feasible flow
        }

        if parameters is None or not parameters:
            parameters = default_parameters
        else:
            parameters = {**default_parameters, **dict(parameters)}
        self.parameters = dict(parameters)
        for key, value in self.parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Multi-Commodity Network Flow problem instance and return the Gurobi model.
        """
        # Randomly select number of cities and products
        self.n_cities = _sample_int(self.n_cities)
        self.n_products = _sample_int(self.n_products)

        # Generate cities, products, and links
        cities = [f"city_{i}" for i in range(self.n_cities)]
        products = [f"product_{p}" for p in range(self.n_products)]
        links = [(cities[idx], cities[idx + 1]) for idx in range(len(cities) - 1)]
        links += [(cities[idx], cities[idx + 2]) for idx in range(len(cities) - 2)]
        links += [(cities[idx + 1], cities[idx]) for idx in range(len(cities) - 1) if random.random() < 0.35]
        links = list(dict.fromkeys(links))

        latent_flow = {(i, j, p): 0 for (i, j) in links for p in products}
        supply = {city: {product: 0 for product in products} for city in cities}
        demand = {city: {product: 0 for product in products} for city in cities}
        for product in products:
            source_index = random.randint(0, max(0, len(cities) // 2 - 1))
            sink_index = random.randint(max(1, len(cities) // 2), len(cities) - 1)
            quantity = random.randint(*self.shipment_range)
            supply[cities[source_index]][product] += quantity
            demand[cities[sink_index]][product] += quantity
            for idx in range(source_index, sink_index):
                latent_flow[cities[idx], cities[idx + 1], product] += quantity

            if len(cities) >= 4 and random.random() < 0.7:
                source_index = random.randint(0, len(cities) - 3)
                sink_index = random.randint(source_index + 2, len(cities) - 1)
                quantity = random.randint(max(1, self.shipment_range[0] // 2), max(2, self.shipment_range[1] // 2))
                supply[cities[source_index]][product] += quantity
                demand[cities[sink_index]][product] += quantity
                for idx in range(source_index, sink_index):
                    latent_flow[cities[idx], cities[idx + 1], product] += quantity

        shipment_cost = {(i, j): {p: random.randint(*self.shipment_cost_range) for p in products} for (i, j) in links}
        capacity = {
            (i, j): {
                p: latent_flow[i, j, p] + random.randint(*self.capacity_slack_range)
                for p in products
            }
            for (i, j) in links
        }
        joint_capacity = {
            (i, j): sum(latent_flow[i, j, p] for p in products) + random.randint(*self.capacity_slack_range)
            for (i, j) in links
        }
        self.parameters.update(
            {
                "cities": cities,
                "products": products,
                "links": [list(link) for link in links],
                "supply": {city: dict(values) for city, values in supply.items()},
                "demand": {city: dict(values) for city, values in demand.items()},
                "shipment_cost": {f"{i}|{j}|{p}": shipment_cost[i, j][p] for (i, j) in links for p in products},
                "product_capacity": {f"{i}|{j}|{p}": capacity[i, j][p] for (i, j) in links for p in products},
                "joint_capacity": {f"{i}|{j}": joint_capacity[i, j] for (i, j) in links},
                "compact_netmcol_tables": {
                    "sets": {
                        "cities": cities,
                        "products": products,
                        "directed_links": [list(link) for link in links],
                    },
                    "city_product_balance_table": {
                        "columns": ["city", "product", "supply", "demand", "net_supply_minus_demand"],
                        "rows": [
                            [city, product, supply[city][product], demand[city][product], supply[city][product] - demand[city][product]]
                            for city in cities
                            for product in products
                        ],
                    },
                    "link_capacity_table": {
                        "columns": ["origin", "destination", "joint_capacity"],
                        "rows": [[i, j, joint_capacity[i, j]] for i, j in links],
                    },
                    "product_link_table": {
                        "columns": ["origin", "destination", "product", "unit_shipping_cost", "product_capacity"],
                        "rows": [
                            [i, j, product, shipment_cost[i, j][product], capacity[i, j][product]]
                            for i, j in links
                            for product in products
                        ],
                    },
                    "required_decision_layers": [
                        "Ship[i,j,p] continuous nonnegative flow of product p on directed link i->j",
                    ],
                },
                "total_supply_by_product": {p: sum(supply[city][p] for city in cities) for p in products},
                "total_demand_by_product": {p: sum(demand[city][p] for city in cities) for p in products},
                "decision_variables": {
                    "Ship[i,j,p]": "continuous nonnegative flow of product p on directed link i->j",
                },
                "objective_terms": [
                    "shipment_cost_times_product_flow",
                ],
                "required_constraints": [
                    "city_product_flow_balance",
                    "directed_link_joint_capacity",
                    "directed_link_product_capacity",
                    "nonnegative_product_flow",
                ],
                "required_parameter_presentation": [
                    "list cities, products, and directed links",
                    "list supply and demand by city and product",
                    "state total supply equals total demand for every product",
                    "list per-unit shipment cost for every directed link and product",
                    "list product-specific capacity for every directed link and product",
                    "list joint capacity for every directed link across all products",
                    "state this is a continuous multi-commodity flow LP with no binary activation variables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "cities": cities,
                        "products": products,
                        "directed_links": [list(link) for link in links],
                    },
                    "parameters": {
                        "total_supply_by_product": {p: sum(supply[city][p] for city in cities) for p in products},
                        "total_demand_by_product": {p: sum(demand[city][p] for city in cities) for p in products},
                        "shipment_cost_table": "see top-level shipment_cost[i|j|p]",
                        "product_capacity_table": "see top-level product_capacity[i|j|p]",
                        "joint_capacity_table": "see top-level joint_capacity[i|j]",
                    },
                    "constraints": {
                        "flow_balance": "for each city and product, supply plus inbound flow equals demand plus outbound flow",
                        "joint_capacity": "sum of all product flows on a link is at most joint_capacity[i,j]",
                        "product_capacity": "flow of each product on each link is at most product_capacity[i,j,p]",
                    },
                    "objective": "minimize total product-specific shipment cost",
                    "generation_note": "An internal feasible product flow was used only to derive balanced city-product supply/demand and capacities; it is not exposed as a required solution.",
                },
                "business_interpretation_guardrails": [
                    "This is a continuous multi-commodity network flow LP, not a fixed-charge network design model.",
                    "Do not introduce binary arc activation variables or fixed activation costs.",
                    "Do not omit city-product flow balance constraints.",
                    "Do not omit joint link capacity or product-specific link capacity constraints.",
                    "Do not convert this into vehicle routing, path selection, or single-commodity transportation.",
                    "Do not reveal or constrain the model to any internally generated feasible flow certificate.",
                ],
            }
        )

        # Create the Gurobi model
        model = gp.Model("MultiCommodityNetworkFlow")
        model.Params.OutputFlag = 0

        # Create decision variables
        ship = model.addVars(links, products, vtype=GRB.CONTINUOUS, name="Ship")

        # Set the objective: minimize total shipment cost
        model.setObjective(
            gp.quicksum(shipment_cost[(i, j)][p] * ship[i, j, p] for (i, j) in links for p in products),
            GRB.MINIMIZE
        )

        # Add flow balance constraints
        for k in cities:
            for p in products:
                model.addConstr(
                    supply[k][p] + gp.quicksum(ship[i, k, p] for (i, k2) in links if k2 == k) ==
                    demand[k][p] + gp.quicksum(ship[k, j, p] for (k2, j) in links if k2 == k),
                    name=f"FlowBalance_{k}_{p}"
                )

        # Add joint capacity constraints
        for (i, j) in links:
            model.addConstr(
                gp.quicksum(ship[i, j, p] for p in products) <= joint_capacity[(i, j)],
                name=f"JointCapacity_{i}_{j}"
            )

        # Add product-specific capacity constraints
        for (i, j) in links:
            for p in products:
                model.addConstr(
                    ship[i, j, p] <= capacity[(i, j)][p],
                    name=f"Capacity_{i}_{j}_{p}"
                )
        return model

def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)

if __name__ == '__main__':
    import time

    def test_generator():
        generator = Generator()  # Fix the random seed for reproducibility
        model = generator.generate_instance()

        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
