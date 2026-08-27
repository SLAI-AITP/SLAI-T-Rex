import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import time


def _sample_count(value):
    if isinstance(value, int):
        return value
    return random.randint(int(value[0]), int(value[1]))


def _positive_split(total: int, count: int, *, min_value: int = 1) -> list[int]:
    if count <= 1:
        return [total]
    remaining = total - min_value * count
    if remaining < 0:
        raise ValueError("total is too small for a positive split")
    cuts = sorted(random.sample(range(remaining + count - 1), count - 1))
    pieces = []
    previous = -1
    for cut in cuts + [remaining + count - 1]:
        pieces.append(cut - previous)
        previous = cut
    return [piece + min_value - 1 for piece in pieces]


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Supply Chain optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_nodes: Number of nodes
            - cap_range: Tuple of (min, max) for arc capacities
            - fixed_cost_range: Tuple of (min, max) for fixed costs
            - unit_cost_range: Tuple of (min, max) for unit transportation costs
            - total_supply: Total supply in the system
            - n_suppliers: Number of supplier nodes
            - n_customers: Number of customer nodes
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "supply_chain"
        self.description = "This problem aims to design a distribution network for a supply chain. We want to select the suppliers and warehouses locations, and decide on the amount of product transported between these locations to satisfy the customers demand. The problem elements are a) a set of nodes, ie, plants, warehouses/DCs, customers, and b) a set of links/arcs, listing pairs of nodes corresponding to allowed shipment options. Each node has a supply or demand amount and a fixed cost of including the node in the network. Each link or arc has cost/unit flow and capacity. We want to determine the facilities to use and the flow of each arc to minimize the flow cost and the fixed cost of nodes."
        self.mathematical_formulation = r"""
        \begin{align*}
        \min & \sum_{i \in N}\sum_{j \in N} f_{i,j}y_{i,j} + \sum_{i \in N}\sum_{j \in N} c_{i,j}x_{i,j} \\
        \text{s.t.} & x_{i,j} \leq cap_{i,j}y_{i,j}, \quad \forall i,j \in N \\
        & \sum_{j \in N} x_{j,i} - \sum_{j \in N} x_{i,j} = d_i - s_i, \quad \forall i \in N \\
        & y_{i,j} \in \{0,1\}, x_{i,j} \geq 0 \quad \forall i,j \in N
        \end{align*}
        """

        default_parameters = {
            "n_nodes": (6, 8),
            "cap_range": (60, 180),
            "fixed_cost_range": (300, 1800),
            "unit_cost_range": (4, 25),
            "total_supply": 300,
            "n_suppliers": (2, 3),
            "n_customers": (2, 3),
            "arc_density": 0.35,
            "max_extra_arcs": 8,
            "direct_arc_slack_range": (20, 80),
        }

        parameters = {**default_parameters, **dict(parameters or {})}
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)

        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        """
        Generate a Supply Chain problem instance and create its corresponding Gurobi model.
        Returns:
        gp.Model: Configured Gurobi model for the supply chain problem
        """
        
        # Randomly select number of nodes
        self.n_nodes = _sample_count(self.n_nodes)
        self.n_suppliers = min(
            _sample_count(self.n_suppliers),
            max(1, self.n_nodes // 2),
        )
        self.n_customers = min(
            _sample_count(self.n_customers),
            max(1, self.n_nodes - self.n_suppliers),
        )
        
        
        # Generate nodes
        nodes = [f"node_{i}" for i in range(self.n_nodes)]
        shuffled_nodes = list(nodes)
        random.shuffle(shuffled_nodes)

        # Generate supplies and demands
        supplies = {node: 0 for node in nodes}
        demands = {node: 0 for node in nodes}

        # Assign supplies to supplier nodes
        supplier_nodes = shuffled_nodes[: self.n_suppliers]
        for node, supply in zip(supplier_nodes, _positive_split(self.total_supply, len(supplier_nodes), min_value=20)):
            supplies[node] = supply

        # Assign demands to customer nodes
        customer_nodes = shuffled_nodes[self.n_suppliers:self.n_suppliers + self.n_customers]
        for node, demand in zip(customer_nodes, _positive_split(self.total_supply, len(customer_nodes), min_value=20)):
            demands[node] = demand

        transshipment_nodes = [node for node in nodes if node not in supplier_nodes and node not in customer_nodes]

        # Build a sparse declared arc set with a known feasible direct-flow certificate.
        latent_flow: dict[tuple[str, str], int] = {}
        remaining_supply = dict(supplies)
        remaining_demand = dict(demands)
        suppliers_order = list(supplier_nodes)
        customers_order = list(customer_nodes)
        random.shuffle(suppliers_order)
        random.shuffle(customers_order)
        customer_index = 0
        for supplier in suppliers_order:
            while remaining_supply[supplier] > 0:
                customer = customers_order[customer_index % len(customers_order)]
                amount = min(remaining_supply[supplier], remaining_demand[customer])
                if amount > 0:
                    latent_flow[supplier, customer] = latent_flow.get((supplier, customer), 0) + amount
                    remaining_supply[supplier] -= amount
                    remaining_demand[customer] -= amount
                if remaining_demand[customer] <= 0:
                    customer_index += 1
                if customer_index >= len(customers_order) and remaining_supply[supplier] > 0:
                    open_customers = [node for node, value in remaining_demand.items() if value > 0]
                    if not open_customers:
                        break
                    customer_index = customers_order.index(open_customers[0])

        arcs = set(latent_flow)
        candidate_extra_arcs: list[tuple[str, str]] = []
        for supplier in supplier_nodes:
            for transshipment in transshipment_nodes:
                candidate_extra_arcs.append((supplier, transshipment))
            for customer in customer_nodes:
                candidate_extra_arcs.append((supplier, customer))
        for transshipment in transshipment_nodes:
            for customer in customer_nodes:
                candidate_extra_arcs.append((transshipment, customer))
            for other in transshipment_nodes:
                if transshipment != other:
                    candidate_extra_arcs.append((transshipment, other))
        candidate_extra_arcs = [arc for arc in dict.fromkeys(candidate_extra_arcs) if arc not in arcs]
        random.shuffle(candidate_extra_arcs)
        density_target = int(len(candidate_extra_arcs) * float(self.arc_density))
        extra_count = min(len(candidate_extra_arcs), max(0, int(self.max_extra_arcs)), max(1, density_target))
        arcs.update(candidate_extra_arcs[:extra_count])
        arcs = sorted(arcs)

        cap = {}
        for i, j in arcs:
            if (i, j) in latent_flow:
                slack = random.randint(*self.direct_arc_slack_range)
                cap[i, j] = latent_flow[i, j] + slack
            else:
                cap[i, j] = random.randint(*self.cap_range)
        fixed_costs = {
            (i, j): random.randint(*self.fixed_cost_range)
            for i, j in arcs
        }
        unit_costs = {
            (i, j): random.randint(*self.unit_cost_range)
            for i, j in arcs
        }
        self.parameters.update(
            {
                "nodes": nodes,
                "node_roles": {
                    node: (
                        "supplier"
                        if node in supplier_nodes
                        else "customer"
                        if node in customer_nodes
                        else "transshipment"
                    )
                    for node in nodes
                },
                "arcs": [f"{i}|{j}" for i, j in arcs],
                "supplier_nodes": supplier_nodes,
                "customer_nodes": customer_nodes,
                "transshipment_nodes": transshipment_nodes,
                "supplies": supplies,
                "demands": demands,
                "node_balance_rhs": {
                    node: demands[node] - supplies[node]
                    for node in nodes
                },
                "arc_table": {
                    f"{i}|{j}": {
                        "origin": i,
                        "destination": j,
                        "capacity": cap[i, j],
                        "fixed_activation_cost": fixed_costs[i, j],
                        "unit_flow_cost": unit_costs[i, j],
                    }
                    for i, j in arcs
                },
                "compact_supplychain_tables": {
                    "sets": {
                        "nodes": nodes,
                        "declared_directed_arcs": [list(arc) for arc in arcs],
                    },
                    "node_table": {
                        "columns": ["node", "role", "supply", "demand", "balance_rhs_demand_minus_supply"],
                        "rows": [
                            [
                                node,
                                (
                                    "supplier"
                                    if node in supplier_nodes
                                    else "customer"
                                    if node in customer_nodes
                                    else "transshipment"
                                ),
                                supplies[node],
                                demands[node],
                                demands[node] - supplies[node],
                            ]
                            for node in nodes
                        ],
                    },
                    "arc_table": {
                        "columns": [
                            "origin",
                            "destination",
                            "capacity",
                            "fixed_activation_cost",
                            "unit_flow_cost",
                        ],
                        "rows": [
                            [i, j, cap[i, j], fixed_costs[i, j], unit_costs[i, j]]
                            for i, j in arcs
                        ],
                    },
                    "required_decision_layers": [
                        "y[i,j] binary activation for declared directed arc (i,j)",
                        "x[i,j] continuous nonnegative flow on declared directed arc (i,j)",
                    ],
                },
                "arc_capacities": {f"{i}|{j}": cap[i, j] for i, j in arcs},
                "fixed_arc_activation_costs": {f"{i}|{j}": fixed_costs[i, j] for i, j in arcs},
                "unit_arc_flow_costs": {f"{i}|{j}": unit_costs[i, j] for i, j in arcs},
                "network_summary": {
                    "node_count": len(nodes),
                    "arc_count": len(arcs),
                    "supplier_count": len(supplier_nodes),
                    "customer_count": len(customer_nodes),
                    "transshipment_count": len(transshipment_nodes),
                    "total_supply_equals_total_demand": True,
                    "sparse_declared_arc_set": True,
                    "latent_feasible_arc_count_not_exposed": sum(1 for value in latent_flow.values() if value > 0),
                },
                "total_supply": self.total_supply,
                "total_demand": self.total_supply,
                "n_nodes": self.n_nodes,
                "n_suppliers": self.n_suppliers,
                "n_customers": self.n_customers,
                "decision_variables": [
                    "y[i,j] binary variable indicating whether declared directed arc (i,j) is activated",
                    "x[i,j] nonnegative continuous flow on declared directed arc (i,j)",
                ],
                "objective_terms": [
                    "fixed_arc_activation_cost",
                    "unit_arc_flow_cost",
                ],
                "required_constraints": [
                    "arc_capacity_activation_linking",
                    "node_flow_balance_inflow_minus_outflow_equals_demand_minus_supply",
                    "binary_arc_activation_variables",
                    "nonnegative_continuous_arc_flow_variables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "nodes": "N",
                        "declared_directed_arcs": "A",
                        "suppliers": "S",
                        "customers": "C",
                        "transshipment_nodes": "T",
                    },
                    "parameters": [
                        "supply[i]",
                        "demand[i]",
                        "capacity[i,j] for (i,j) in A",
                        "fixed_activation_cost[i,j] for (i,j) in A",
                        "unit_flow_cost[i,j] for (i,j) in A",
                    ],
                    "constraints": [
                        "x[i,j] <= capacity[i,j] * y[i,j] for every declared directed arc",
                        "sum_{j:(j,i) in A} x[j,i] - sum_{j:(i,j) in A} x[i,j] = demand[i] - supply[i] for every node",
                    ],
                    "objective": "minimize fixed arc activation costs plus unit flow costs over declared directed arcs",
                },
                "required_parameter_presentation": [
                    "List all nodes and each node's role: supplier, customer, or transshipment.",
                    "List supply and demand for every node, including zeros.",
                    "List only the declared directed arcs; do not imply a complete graph.",
                    "For every declared arc, list capacity, fixed activation cost, and unit flow cost.",
                    "State flow balance uses inflow minus outflow equals demand minus supply.",
                    "State y[i,j] is binary arc activation and x[i,j] is nonnegative continuous arc flow.",
                    "State this is not node facility opening and has no inventory or time periods.",
                ],
                "business_interpretation_guardrails": [
                    "This is a fixed-charge capacitated directed network-flow model.",
                    "Do not replace arc activation costs with node facility opening costs.",
                    "Do not add inventory, time periods, or warehouse storage constraints.",
                    "Use only the listed directed arcs and their capacity, fixed cost, and unit cost tables.",
                    "Do not create arcs outside the declared sparse directed arc set.",
                    "Preserve the flow-balance sign convention: inflow minus outflow equals demand minus supply.",
                ],
                "generation_notes": [
                    "An internal latent feasible direct flow was used only to derive balanced supplies, demands, and capacities; it is not exposed and must not appear in the problem statement.",
                    "The sparse arc set reduces prompt length while preserving fixed-charge network design structure.",
                    "Supplier nodes have net negative balance, customer nodes have net positive balance, and transshipment nodes have zero balance.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("SupplyChain")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create variables
        y = model.addVars(
            arcs, vtype=GRB.BINARY, name="y"
        )
        x = model.addVars(
            arcs,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="x",
        )

        # Set objective
        model.setObjective(
            gp.quicksum(
                fixed_costs[i, j] * y[i, j] + unit_costs[i, j] * x[i, j]
                for i, j in arcs
            ),
            GRB.MINIMIZE,
        )

        # Add capacity constraints
        for i, j in arcs:
            model.addConstr(x[i, j] <= cap[i, j] * y[i, j], name=f"ArcCapacity_{i}_{j}")

        # Add flow conservation constraints
        incoming_arcs = {
            node: [(u, v) for u, v in arcs if v == node]
            for node in nodes
        }
        outgoing_arcs = {
            node: [(u, v) for u, v in arcs if u == node]
            for node in nodes
        }
        for i in nodes:
            model.addConstr(
                gp.quicksum(x[j, i] for j, _ in incoming_arcs[i])
                - gp.quicksum(x[i, j] for _, j in outgoing_arcs[i])
                == demands[i] - supplies[i],
                name=f"FlowBalance_{i}",
            )

        return model


if __name__ == "__main__":

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("SupplyChain.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")

        else:
            print("No optimal solution found")

    test_generator()
