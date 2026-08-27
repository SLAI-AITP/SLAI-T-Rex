import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize a capacitated minimum-cost network-flow optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_cities: Number of cities
                - shipment_range: Tuple of (min, max) for source-to-sink shipment quantities
                - shipping_cost_range: Tuple of (min, max) for shipping costs
                - capacity_slack_range: Tuple of (min, max) capacity slack added to a feasible flow
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "network_flow"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are `n` cities and `m` links between them. Each city `i` has:
        - **Supply**: Supply_i (amount of goods available)
        - **Demand**: Demand_i (amount of goods required)
        Each link `(i, j)` has:
        - **Shipping Cost**: ShippingCost_{i,j} (cost to ship one unit of goods)
        - **Capacity**: Capacity_{i,j} (maximum units that can be shipped)
        
        Decision Variable:
        - Ship_{i,j}: Number of units shipped from city `i` to city `j`
        
        Objective:
        Minimize the total shipping cost:
        $$
        \text{Minimize} \quad \sum_{(i,j) \in \text{Links}} \text{ShippingCost}_{i,j} \cdot \text{Ship}_{i,j}
        $$

        Constraints:
        1. Flow Balance:
        For each city `k`:
        $$
        \text{Supply}_k + \sum_{(i,k) \in \text{Links}} \text{Ship}_{i,k} = \text{Demand}_k + \sum_{(k,j) \in \text{Links}} \text{Ship}_{k,j}
        $$

        2. Capacity:
        For each link `(i,j)`:
        $$
        \text{Ship}_{i,j} \leq \text{Capacity}_{i,j}
        $$

        3. Non-Negativity:
        $$
        \text{Ship}_{i,j} \geq 0 \quad \forall (i,j) \in \text{Links}
        $$
        """
        default_parameters = {
            "n_cities": (5, 7),
            "shipment_range": (10, 60),
            "shipping_cost_range": (1, 10),
            "capacity_slack_range": (0, 15),
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
        Generate a Network Flow problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (cities, supply, demand, shipping costs, capacities)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the network flow problem
        """
        
        # Randomly select number of cities
        self.n_cities = _sample_int(self.n_cities)
        
        # Generate cities and a sparse directed network with a guaranteed
        # feasible source-to-sink backbone.
        cities = [f"city_{i}" for i in range(self.n_cities)]
        links = [(cities[idx], cities[idx + 1]) for idx in range(self.n_cities - 1)]
        links += [(cities[idx], cities[idx + 2]) for idx in range(self.n_cities - 2)]
        links += [(cities[idx + 1], cities[idx]) for idx in range(self.n_cities - 1) if random.random() < 0.30]
        links = list(dict.fromkeys(links))

        latent_flow = {link: 0 for link in links}
        supply = {city: 0 for city in cities}
        demand = {city: 0 for city in cities}
        shipment_count = random.randint(2, min(4, self.n_cities - 1))
        for _ in range(shipment_count):
            source_idx = random.randint(0, self.n_cities - 2)
            sink_idx = random.randint(source_idx + 1, self.n_cities - 1)
            quantity = random.randint(*self.shipment_range)
            supply[cities[source_idx]] += quantity
            demand[cities[sink_idx]] += quantity
            for idx in range(source_idx, sink_idx):
                latent_flow[cities[idx], cities[idx + 1]] += quantity

        shipping_cost = {link: random.randint(*self.shipping_cost_range) for link in links}
        capacity = {
            link: max(1, latent_flow[link] + random.randint(*self.capacity_slack_range))
            for link in links
        }
        node_balance_table = [
            {
                "city": city,
                "supply": supply[city],
                "demand": demand[city],
                "net_supply_minus_demand": supply[city] - demand[city],
            }
            for city in cities
        ]
        arc_table = [
            {
                "from": i,
                "to": j,
                "unit_arc_flow_cost": shipping_cost[i, j],
                "arc_capacity": capacity[i, j],
            }
            for i, j in links
        ]
        self.parameters.update(
            {
                "cities": cities,
                "links": [list(link) for link in links],
                "supply": supply,
                "demand": demand,
                "net_balance": {city: supply[city] - demand[city] for city in cities},
                "shipping_cost": {f"{i}|{j}": shipping_cost[i, j] for (i, j) in links},
                "capacity": {f"{i}|{j}": capacity[i, j] for (i, j) in links},
                "compact_network_flow_tables": {
                    "model_family": "capacitated_min_cost_network_flow",
                    "cities": cities,
                    "directed_arc_table": arc_table,
                    "node_balance_table": node_balance_table,
                    "total_supply": sum(supply.values()),
                    "total_demand": sum(demand.values()),
                    "flow_balance_convention": (
                        "For every city: supply[city] + inbound_flow[city] = demand[city] + outbound_flow[city]. "
                        "Equivalently, outbound_flow[city] - inbound_flow[city] = supply[city] - demand[city]."
                    ),
                    "source_model_note": (
                        "Use separate supply and demand values from node_balance_table. "
                        "Do not collapse multiple demand nodes into one aggregate demand node."
                    ),
                },
                "compact_net1_tables": {
                    "model_family": "capacitated_min_cost_network_flow",
                    "cities": cities,
                    "directed_arc_table": arc_table,
                    "node_balance_table": node_balance_table,
                    "total_supply": sum(supply.values()),
                    "total_demand": sum(demand.values()),
                    "flow_balance_convention": (
                        "supply + inbound flow equals demand + outbound flow at every city"
                    ),
                },
                "latent_feasible_flow": {
                    f"{i}|{j}": value
                    for (i, j), value in latent_flow.items()
                    if value > 0
                },
                "total_supply": sum(supply.values()),
                "total_demand": sum(demand.values()),
                "decision_variables": {
                    "Ship[i,j]": "continuous nonnegative units shipped on directed link i->j",
                },
                "objective_terms": [
                    "arc_shipping_cost_times_flow",
                ],
                "required_constraints": [
                    "node_flow_balance_supply_plus_inflow_equals_demand_plus_outflow",
                    "directed_arc_capacity_limit",
                    "nonnegative_arc_flow",
                ],
                "required_parameter_presentation": [
                    "list cities and directed arcs",
                    "list supply and demand at every city",
                    "state total supply equals total demand",
                    "list per-unit shipping cost for every directed arc",
                    "list capacity for every directed arc",
                    "state this is a continuous min-cost network flow LP with no binary activation or routing variables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "cities": cities,
                        "directed_arcs": [list(link) for link in links],
                    },
                    "parameters": {
                        "supply": supply,
                        "demand": demand,
                        "net_balance": {city: supply[city] - demand[city] for city in cities},
                        "total_supply": sum(supply.values()),
                        "total_demand": sum(demand.values()),
                        "shipping_cost_table": "see top-level shipping_cost[i|j]",
                        "capacity_table": "see top-level capacity[i|j]",
                    },
                    "constraints": {
                        "flow_balance": "for each city, supply plus inbound flow equals demand plus outbound flow",
                        "arc_capacity": "flow on each directed arc is at most its capacity",
                    },
                    "objective": "minimize total arc shipping cost",
                    "feasibility_certificate": {
                        "latent_feasible_flow": {
                            f"{i}|{j}": value
                            for (i, j), value in latent_flow.items()
                            if value > 0
                        },
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a continuous capacitated min-cost network flow LP, not facility location or vehicle routing.",
                    "Do not introduce binary arc activation, fixed opening costs, vehicles, time windows, or subtour constraints.",
                    "Do not omit node flow balance constraints.",
                    "Do not omit directed arc capacity constraints.",
                    "Do not weaken total supply/demand balance with unmet-demand or disposal slack.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("NetworkFlow")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables (Ship[i,j] = units shipped from city i to city j)
        ship = model.addVars(links, vtype=GRB.CONTINUOUS, name="Ship")

        # Set objective: minimize total shipping cost
        model.setObjective(
            gp.quicksum(shipping_cost[link] * ship[link] for link in links),
            GRB.MINIMIZE
        )

        # Add flow balance constraints for each city
        for city in cities:
            inflow = gp.quicksum(ship[(i, city)] for (i, j) in links if j == city)
            outflow = gp.quicksum(ship[(city, j)] for (i, j) in links if i == city)
            model.addConstr(
                supply[city] + inflow == demand[city] + outflow,
                name=f"FlowBalance_{city}"
            )

        # Add capacity constraints for each link
        for link in links:
            model.addConstr(
                ship[link] <= capacity[link],
                name=f"Capacity_{link}"
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
        
        model.write("network_flow.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
