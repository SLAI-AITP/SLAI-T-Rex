import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize the Shortest Path Problem.

        Parameters:
            parameters (dict): Dictionary containing:
                - n_nodes: Number of nodes in the graph
                - arc_cost_range: Tuple of (min, max) for arc costs
                - start_node: Starting node (default is 0)
                - end_node: Ending node (default is n_nodes - 1)
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "shortest_path"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there is a graph with nodes N and arcs E. Each arc (i,j) has a cost a_{i,j}.
        The goal is to find the path from a start node to an end node with the minimum total cost.

        Variables:
        - x_{i,j}: Binary variable indicating whether arc (i,j) is in the path.

        Objective:
        Minimize the total cost of the path:
        $$
        \text{Minimize} \quad \sum_{(i,j) \in E} a_{i,j} \cdot x_{i,j}
        $$

        Constraints:
        - Flow balance at each node:
        $$
        \sum_{j \in N \mid (i,j) \in E} x_{i,j} - \sum_{k \in N \mid (k,i) \in E} x_{k,i} = b_i \quad \forall i \in N
        $$
        where:
        - b_i = 1 if i is the start node,
        - b_i = -1 if i is the end node,
        - b_i = 0 otherwise.
        """
        default_parameters = {
            "n_nodes": (5, 8),
            "arc_cost_range": (1, 10),
            "start_node": 0,
            "end_node": None
        }
        # Use default parameters if none are provided or if an empty dict is given
        if parameters is None or not parameters:
            parameters = default_parameters
        parameters = dict(parameters)
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
        
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Shortest Path Problem instance and create its corresponding Gurobi model.

        This method does two things:
        1. Generates random problem data (nodes, arcs, costs)
        2. Creates and returns a configured Gurobi model ready to solve

        Returns:
            gp.Model: Configured Gurobi model for the shortest path problem
        """
        # Randomly select number of nodes
        self.n_nodes = random.randint(*self.n_nodes)
        
        if self.end_node is None:
            self.end_node = self.n_nodes - 1

        # Generate nodes
        nodes = [f"node_{i}" for i in range(self.n_nodes)]
        
        # Generate a directed acyclic graph with a guaranteed chain from source
        # to sink. Avoid the direct start-to-end shortcut so the optimal path has
        # a nontrivial multi-arc structure.
        arcs_set = {(nodes[index], nodes[index + 1]) for index in range(self.n_nodes - 1)}
        chain_arcs = set(arcs_set)
        start_node = nodes[self.start_node]
        end_node = nodes[self.end_node]
        optional_arcs = []
        for left_index, i in enumerate(nodes):
            for right_index, j in enumerate(nodes):
                if right_index <= left_index:
                    continue
                if i == start_node and j == end_node:
                    continue
                if (i, j) in chain_arcs:
                    continue
                optional_arcs.append((i, j))
                if random.random() < 0.45:
                    arcs_set.add((i, j))
        # Ensure every instance contains unused routing alternatives. Otherwise
        # small graphs can collapse to the guaranteed chain and all binary arc
        # variables become active, which is mathematically valid but too trivial
        # for CPT seed data.
        random.shuffle(optional_arcs)
        min_optional_arcs = min(len(optional_arcs), max(2, self.n_nodes // 2))
        for arc in optional_arcs[:min_optional_arcs]:
            arcs_set.add(arc)
        arcs = sorted(arcs_set)
        
        # Generate random arc costs
        arc_costs = {(i, j): random.randint(*self.arc_cost_range) for (i, j) in arcs}
        
        self.parameters.update(
            {
                "nodes": nodes,
                "arcs": arcs,
                "arc_costs": {f"{i}|{j}": arc_costs[i, j] for i, j in arcs},
                "start_node": start_node,
                "end_node": end_node,
                "node_balance_rhs": {
                    node: 1 if node == start_node else -1 if node == end_node else 0
                    for node in nodes
                },
                "compact_shortest_path_tables": {
                    "sets": {
                        "nodes": nodes,
                        "source": start_node,
                        "sink": end_node,
                        "directed_arcs": [[i, j] for i, j in arcs],
                    },
                    "node_balance_table": {
                        "columns": ["node", "balance_rhs_outflow_minus_inflow"],
                        "rows": [
                            [
                                node,
                                1 if node == start_node else -1 if node == end_node else 0,
                            ]
                            for node in nodes
                        ],
                    },
                    "arc_cost_table": {
                        "columns": ["from_node", "to_node", "arc_cost"],
                        "rows": [[i, j, arc_costs[i, j]] for i, j in arcs],
                    },
                    "required_decision_layers": [
                        "Arcs[i,j] binary, 1 if directed arc (i,j) is selected in the source-to-sink path",
                    ],
                },
                "decision_variables": {
                    "Arcs[i,j]": "binary directed-arc selection variable for the source-to-sink path",
                },
                "objective_terms": [
                    "arc_cost_times_binary_path_arc",
                ],
                "required_constraints": [
                    "source_sink_flow_balance",
                    "intermediate_node_flow_conservation",
                ],
                "required_parameter_presentation": [
                    "list all nodes and identify the source and sink",
                    "list only the declared directed arcs with their arc costs",
                    "state Arcs[i,j] is binary and can only use declared directed arcs",
                    "state source balance is outflow minus inflow equals 1",
                    "state sink balance is outflow minus inflow equals -1",
                    "state every intermediate node has outflow minus inflow equals 0",
                    "state there are no arc capacity limits, vehicle routes, time windows, or facility fixed charges",
                ],
                "structured_problem_data": {
                    "sets": {
                        "nodes": nodes,
                        "source": start_node,
                        "sink": end_node,
                        "directed_arcs": [[i, j] for i, j in arcs],
                    },
                    "parameters": {
                        "arc_cost_table": "see compact_shortest_path_tables.arc_cost_table",
                        "node_balance_table": "see compact_shortest_path_tables.node_balance_table",
                    },
                    "objective": "minimize total selected directed-arc cost from source to sink",
                    "constraints": {
                        "flow_balance": "outflow minus inflow is 1 at source, -1 at sink, and 0 at intermediate nodes",
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a binary shortest-path model on a declared directed acyclic graph.",
                    "Do not add arc capacities, vehicle capacities, time windows, subtour constraints, fixed charges, or facility-opening decisions.",
                    "Do not expand the graph to a complete directed network.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("ShortestPath")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create binary decision variables (x[i,j] = 1 if arc (i,j) is in the path)
        x = model.addVars(arcs, vtype=GRB.BINARY, name="Arcs")

        # Set objective: minimize total cost of the path
        model.setObjective(
            gp.quicksum(arc_costs[i, j] * x[i, j] for (i, j) in arcs),
            GRB.MINIMIZE
        )

        # Add flow balance constraints
        for i in nodes:
            if i == start_node:
                b_i = 1
            elif i == end_node:
                b_i = -1
            else:
                b_i = 0
            model.addConstr(
                gp.quicksum(x[i, j] for j in nodes if (i, j) in arcs) -
                gp.quicksum(x[k, i] for k in nodes if (k, i) in arcs) == b_i,
                name=f"FlowBalance_{i}"
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
        
        model.write("shortest_path.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
