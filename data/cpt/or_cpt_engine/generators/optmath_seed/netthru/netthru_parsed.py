import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize a node-throughput capacitated minimum-cost flow problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_cities: Number of network nodes
                - shipment_range: Tuple of (min, max) for latent source-to-sink shipment quantities
                - cost_range: Tuple of (min, max) for link costs
                - city_capacity_slack_range: Tuple of (min, max) for node throughput slack
                - link_capacity_slack_range: Tuple of (min, max) for link capacity slack
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "network_flow"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Capacitated Minimum-Cost Network Flow

        Sets:
        - \(N\): cities or network nodes
        - \(A\): directed links

        Parameters:
        - \(s_k\): exogenous supply available at node \(k\)
        - \(d_k\): exogenous demand required at node \(k\)
        - \(c_{i,j}\): per-unit shipping cost on directed link \((i,j)\)
        - \(u_{i,j}\): capacity of directed link \((i,j)\)
        - \(U_k\): maximum node throughput at node \(k\)

        Decision variable:
        - \(x_{i,j} \ge 0\): shipment on directed link \((i,j)\)

        Objective:
        \[
        \min \sum_{(i,j)\in A} c_{i,j}x_{i,j}
        \]

        Constraints:
        \[
        s_k + \sum_{(i,k)\in A}x_{i,k}
        =
        d_k + \sum_{(k,j)\in A}x_{k,j}
        \quad \forall k\in N
        \]
        \[
        0 \le x_{i,j} \le u_{i,j} \quad \forall (i,j)\in A
        \]
        \[
        s_k + \sum_{(i,k)\in A}x_{i,k} \le U_k \quad \forall k\in N
        \]

        This is a continuous minimum-cost flow model. It does not contain
        binary lane activation or fixed-charge decisions.
        """
        default_parameters = {
            "n_cities": (5, 8),
            "shipment_range": (8, 45),
            "cost_range": (1, 10),
            "city_capacity_slack_range": (0, 12),
            "link_capacity_slack_range": (0, 12),
            "reverse_arc_probability": 0.25,
        }
        # Use default parameters if none are provided or if an empty dict is given
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
        Generate a sparse feasible network-flow instance and its Gurobi model.
        
        Returns:
            gp.Model: Configured Gurobi model for the network flow problem
        """
        # Randomly select number of cities
        self.n_cities = _sample_int(self.n_cities)
        
        # Build a sparse directed network. Forward chain arcs provide a guaranteed
        # feasible backbone; skip and reverse arcs create meaningful alternatives.
        cities = [f"city_{i}" for i in range(self.n_cities)]
        links = [(cities[idx], cities[idx + 1]) for idx in range(self.n_cities - 1)]
        links += [(cities[idx], cities[idx + 2]) for idx in range(self.n_cities - 2)]
        links += [
            (cities[idx + 1], cities[idx])
            for idx in range(self.n_cities - 1)
            if random.random() < float(self.reverse_arc_probability)
        ]
        links = list(dict.fromkeys(links))

        latent_flow = {link: 0 for link in links}
        supplies = {city: 0 for city in cities}
        demands = {city: 0 for city in cities}
        shipment_count = random.randint(2, min(4, self.n_cities - 1))
        shipment_min, shipment_max = _as_range(self.shipment_range)
        for _ in range(shipment_count):
            source_idx = random.randint(0, self.n_cities - 2)
            sink_idx = random.randint(source_idx + 1, self.n_cities - 1)
            quantity = random.randint(shipment_min, shipment_max)
            supplies[cities[source_idx]] += quantity
            demands[cities[sink_idx]] += quantity
            for idx in range(source_idx, sink_idx):
                latent_flow[cities[idx], cities[idx + 1]] += quantity

        costs = {link: random.randint(*self.cost_range) for link in links}
        link_slack_min, link_slack_max = _as_range(self.link_capacity_slack_range)
        node_slack_min, node_slack_max = _as_range(self.city_capacity_slack_range)
        link_capacities = {
            link: max(1, latent_flow[link] + random.randint(link_slack_min, link_slack_max))
            for link in links
        }
        latent_inflow = {
            city: sum(value for (origin, dest), value in latent_flow.items() if dest == city)
            for city in cities
        }
        city_capacities = {}
        for city in cities:
            latent_throughput = supplies[city] + latent_inflow[city]
            city_capacities[city] = max(1, latent_throughput + random.randint(node_slack_min, node_slack_max))

        self.parameters.update(
            {
                "n_cities": self.n_cities,
                "cities": cities,
                "links": [[i, j] for i, j in links],
                "supplies": dict(supplies),
                "demands": dict(demands),
                "net_balance": {city: supplies[city] - demands[city] for city in cities},
                "city_capacities": dict(city_capacities),
                "costs": {f"{i}|{j}": costs[i, j] for i, j in links},
                "link_capacities": {f"{i}|{j}": link_capacities[i, j] for i, j in links},
                "total_supply": sum(supplies.values()),
                "total_demand": sum(demands.values()),
                "decision_variables": {
                    "Shipping[i,j]": "continuous nonnegative shipment amount on listed directed link i->j",
                },
                "objective_terms": [
                    "directed_link_unit_cost_times_shipment",
                ],
                "required_constraints": [
                    "node_flow_balance_supply_plus_inflow_equals_demand_plus_outflow",
                    "directed_link_capacity_upper_bounds",
                    "node_throughput_capacity",
                    "nonnegative_continuous_shipments",
                ],
                "required_parameter_presentation": [
                    "list nodes and directed links",
                    "list supply and demand for every node",
                    "state total supply equals total demand",
                    "list per-unit cost and hard capacity for every directed link",
                    "list node throughput capacity for every node",
                    "state node throughput is supply plus inbound flow",
                    "state all shipment variables are continuous and nonnegative",
                ],
                "structured_problem_data": {
                    "sets": {
                        "nodes": cities,
                        "directed_links": [[i, j] for i, j in links],
                    },
                    "parameters": {
                        "supply": dict(supplies),
                        "demand": dict(demands),
                        "node_throughput_capacity": dict(city_capacities),
                        "link_cost": "see top-level costs[i|j]",
                        "link_capacity": "see top-level link_capacities[i|j]",
                    },
                    "constraints": {
                        "node_flow_balance": "supply plus inbound shipment equals demand plus outbound shipment at every node",
                        "link_capacity": "shipment on each directed link is at most its capacity",
                        "node_throughput_capacity": "supply plus inbound shipment at each node is at most the node throughput capacity",
                    },
                    "objective": "minimize total directed-link shipping cost",
                    "generation_note": "The instance was internally generated from a feasible flow, but no flow certificate should be included in the natural-language problem.",
                },
                "business_interpretation_guardrails": [
                    "This is a continuous node-throughput capacitated minimum-cost flow LP.",
                    "Do not introduce binary link activation variables or fixed-charge opening costs.",
                    "Do not reinterpret this as vehicle routing, route sequencing, or maximum-flow path selection.",
                    "The objective is minimum shipping cost, not maximum throughput.",
                    "Preserve node balance, directed link capacity, and node throughput capacity constraints.",
                    "Do not reveal or constrain the model to any internally used feasible flow certificate.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("NetworkFlow")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (Shipping[i,j] = amount shipped from i to j)
        shipping = model.addVars(links, vtype=GRB.CONTINUOUS, name="Shipping")
        
        # Set objective: minimize total cost of shipping
        model.setObjective(
            gp.quicksum(costs[link] * shipping[link] for link in links),
            GRB.MINIMIZE
        )
        
        # Add flow balance constraints for each city
        for city in cities:
            inflow = gp.quicksum(shipping[(i, city)] for (i, j) in links if j == city)
            outflow = gp.quicksum(shipping[(city, j)] for (i, j) in links if i == city)
            model.addConstr(
                supplies[city] + inflow == demands[city] + outflow,
                name=f"FlowBalance_{city}"
            )
        
        # Add link capacity constraints
        for link in links:
            model.addConstr(
                shipping[link] <= link_capacities[link],
                name=f"LinkCapacity_{link}"
            )
        
        # Add city capacity constraints
        for city in cities:
            inflow = gp.quicksum(shipping[(i, city)] for (i, j) in links if j == city)
            model.addConstr(
                supplies[city] + inflow <= city_capacities[city],
                name=f"CityCapacity_{city}"
            )

        return model


def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


def _as_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("network_flow.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
