import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize MCND (Multicommodity Capacitated Network Design) problem.
        
        Parameters:
        parameters (dict): Dictionary containing:
            - n_nodes: Number of nodes
            - n_commodities: Number of commodities
            - density: Network density (0-1)
            - capacity_range: (min, max) for u_{ij}
            - fixed_cost_range: (min, max) for f_{ij}
            - variable_cost_range: (min, max) for c_{ij}^k
            - demand_range: (min, max) for d^k
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "mcnd"
        self.mathematical_formulation = r"""### Mathematical Formulation

#### Parameters
- u_{ij}: Capacity provided by installing one facility on arc (i,j)
- f_{ij}: Cost of installing one facility on arc (i,j)
- c_{ij}^k: Routing cost per unit flow of commodity k on arc (i,j)

#### Decision Variables
- x_{ij}^k: Flow variable representing fraction of commodity k on arc (i,j)
- y_{ij}: Integer variable for number of facilities to install on arc (i,j)

#### Objective Function
Minimize total cost:
\min \sum_{k\in K}\sum_{(i,j)\in A} d^kc_{ij}^k x_{ij}^k + \sum_{(i,j)\in A} f_{ij}y_{ij}

#### Constraints
1. Flow Conservation:
\sum_{j\in N_i^+} x_{ij}^k - \sum_{j\in N_i^-} x_{ji}^k = \delta_i^k \quad \forall i \in N, k \in K

2. Capacity Constraints:
\sum_{k\in K} d^k x_{ij}^k \leq u_{ij}y_{ij} \quad \forall (i,j) \in A

3. Flow Bounds:
0 \leq x_{ij}^k \leq 1 \quad \forall (i,j) \in A, k \in K

4. Design Variables:
y_{ij} \geq 0, \text{ integer} \quad \forall (i,j) \in A

where:
- N_i^+ = {j \in N|(i,j) \in A}
- N_i^- = {j \in N|(j,i) \in A}
- \delta_i^k = 1 if i = O(k)
- \delta_i^k = -1 if i = D(k)
- \delta_i^k = 0 otherwise

This formulation represents a mixed-integer programming (MIP) model that can be applied to various applications in transportation, logistics, and telecommunications networks."""

        # Default parameters
        default_parameters = {
            "n_nodes": (5, 7),
            "n_commodities": (2, 4),
            "density": 0.35,
            "capacity_range": (20, 120),    # u_{ij}
            "fixed_cost_range": (100, 1200),  # f_{ij}
            "variable_cost_range": (1, 10),    # c_{ij}^k
            "demand_range": (10, 20)           # d^k
        }

        parameters = {**default_parameters, **dict(parameters or {})}
        
        # Set parameters as attributes
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)

        # Set random seed
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def get_Ni_plus(self, node, arcs):
        """Return N_i^+ = {j ∈ N|(i,j) ∈ A}"""
        return [j for i, j in arcs if i == node]

    def get_Ni_minus(self, node, arcs):
        """Return N_i^- = {j ∈ N|(j,i) ∈ A}"""
        return [i for i, j in arcs if j == node]

    def get_delta(self, node, origin, destination):
        """Return δ_i^k based on node's relation to O(k) and D(k)"""
        if node == origin:
            return 1
        elif node == destination:
            return -1
        return 0

    def generate_instance(self):
        """
        Generate an MCND instance according to the mathematical formulation.
        
        Returns:
        gp.Model: Configured Gurobi model for the MCND problem
        dict: Problem instance data
        """
        
        # Randomly select number of nodes and commodities
        self.n_nodes = random.randint(*self.n_nodes)
        self.n_commodities = random.randint(*self.n_commodities)
        
        # Generate network structure
        N = list(range(self.n_nodes))        # Set of nodes
        K = list(range(self.n_commodities))  # Set of commodities
        
        # O(k), D(k): Origin and destination for each commodity
        OD = {}
        for k in K:
            origin = random.choice(N)
            destination = random.choice([n for n in N if n != origin])
            OD[k] = (origin, destination)

        # Generate arcs with given density, then add each OD direct arc so every
        # commodity has at least one feasible route.
        A_set = {
            (i, j) for i in N for j in N
            if i != j and random.random() < self.density
        }
        for origin, destination in OD.values():
            A_set.add((origin, destination))
        A = sorted(A_set)

        # Generate parameters
        # d^k: Demand for each commodity
        d = {k: random.randint(*self.demand_range) for k in K}

        # u_{ij}: Capacity for each arc. OD arcs get enough capacity for direct
        # routing, while other arcs stay randomly capacitated.
        direct_arc_demand = {arc: 0 for arc in A}
        for k, arc in OD.items():
            direct_arc_demand[arc] = direct_arc_demand.get(arc, 0) + d[k]
        u = {
            (i, j): max(random.randint(*self.capacity_range), direct_arc_demand.get((i, j), 0))
            for i, j in A
        }
        
        # f_{ij}: Fixed cost for each arc
        f = {(i,j): random.randint(*self.fixed_cost_range) for i,j in A}
        
        # c_{ij}^k: Variable cost for each commodity on each arc
        c = {(k,i,j): random.randint(*self.variable_cost_range) 
             for k in K for i,j in A}
        self.parameters.update(
            {
                "nodes": [f"node_{node}" for node in N],
                "commodities": [f"commodity_{commodity}" for commodity in K],
                "arcs": [f"node_{i}|node_{j}" for i, j in A],
                "commodity_od": {
                    f"commodity_{commodity}": {
                        "origin": f"node_{origin}",
                        "destination": f"node_{destination}",
                    }
                    for commodity, (origin, destination) in OD.items()
                },
                "N": N,
                "K": K,
                "A": A,
                "OD": OD,
                "demand": d,
                "capacity": {f"{i}|{j}": u[i, j] for i, j in A},
                "fixed_arc_facility_cost": {f"{i}|{j}": f[i, j] for i, j in A},
                "commodity_arc_variable_cost": {f"{k}|{i}|{j}": c[k, i, j] for k in K for i, j in A},
                "compact_mcnd_tables": {
                    "sets": {
                        "nodes": [f"node_{node}" for node in N],
                        "commodities": [f"commodity_{commodity}" for commodity in K],
                        "directed_arcs": [[f"node_{i}", f"node_{j}"] for i, j in A],
                    },
                    "commodity_table": {
                        "columns": ["commodity", "origin", "destination", "demand"],
                        "rows": [
                            [f"commodity_{k}", f"node_{OD[k][0]}", f"node_{OD[k][1]}", d[k]]
                            for k in K
                        ],
                    },
                    "arc_table": {
                        "columns": ["origin", "destination", "facility_capacity", "fixed_facility_cost"],
                        "rows": [
                            [f"node_{i}", f"node_{j}", u[i, j], f[i, j]]
                            for i, j in A
                        ],
                    },
                    "commodity_arc_cost_table": {
                        "columns": ["commodity", "origin", "destination", "unit_routing_cost"],
                        "rows": [
                            [f"commodity_{k}", f"node_{i}", f"node_{j}", c[k, i, j]]
                            for k in K
                            for i, j in A
                        ],
                    },
                    "required_decision_layers": [
                        "x[k,i,j] continuous fraction of commodity k routed on directed arc (i,j)",
                        "y[i,j] nonnegative integer number of capacity facilities installed on directed arc (i,j)",
                    ],
                },
                "decision_variables": {
                    "x[k,i,j]": "continuous fraction of commodity k routed on directed arc (i,j)",
                    "y[i,j]": "nonnegative integer number of facilities installed on directed arc (i,j)",
                },
                "objective_terms": [
                    "demand_weighted_commodity_arc_routing_cost",
                    "fixed_facility_installation_cost",
                ],
                "required_constraints": [
                    "commodity_flow_conservation",
                    "arc_capacity_linked_to_integer_facility_count",
                    "flow_fraction_bounds_zero_to_one",
                    "nonnegative_integer_arc_facility_count",
                ],
                "required_parameter_presentation": [
                    "list nodes, commodities, and declared directed arcs",
                    "for every commodity, list origin, destination, and demand",
                    "for every directed arc, list facility capacity and fixed facility installation cost",
                    "for every commodity-arc pair, list unit routing cost",
                    "state x[k,i,j] is a continuous fraction of commodity k routed on arc i->j",
                    "state y[i,j] is a nonnegative integer number of installed capacity facilities, not binary",
                    "state flow conservation sends one unit of each commodity from its origin to its destination",
                    "state arc capacity is sum_k demand[k] * x[k,i,j] <= capacity[i,j] * y[i,j]",
                ],
                "structured_problem_data": {
                    "sets": {
                        "nodes": [f"node_{node}" for node in N],
                        "commodities": [f"commodity_{commodity}" for commodity in K],
                        "directed_arcs": [[f"node_{i}", f"node_{j}"] for i, j in A],
                    },
                    "parameters": {
                        "commodity_table": "see compact_mcnd_tables.commodity_table",
                        "arc_table": "see compact_mcnd_tables.arc_table",
                        "commodity_arc_cost_table": "see compact_mcnd_tables.commodity_arc_cost_table",
                    },
                    "constraints": {
                        "flow_conservation": "for each commodity and node, outgoing fraction minus incoming fraction equals 1 at the origin, -1 at the destination, and 0 otherwise",
                        "capacity": "for each arc, demand-weighted commodity flow is at most facility_capacity times integer facility count",
                    },
                    "objective": "minimize demand-weighted commodity routing cost plus fixed facility installation cost",
                },
                "business_interpretation_guardrails": [
                    "This is multicommodity capacitated network design with nonnegative integer arc facility counts.",
                    "Do not make y[i,j] binary; it can install multiple facilities on an arc.",
                    "Do not omit commodity-specific origin, destination, demand, or routing costs.",
                    "Do not collapse commodities into one aggregate commodity.",
                    "Do not add vehicle routes, time windows, inventory, node facility openings, or customer assignment variables.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("MCND")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create variables
        # x_{ij}^k: Flow variables
        x = model.addVars(K, A, vtype=GRB.CONTINUOUS, name="x")
        
        # y_{ij}: Design variables
        y = model.addVars(A, vtype=GRB.INTEGER, name="y")

        # Set objective function
        obj = (gp.quicksum(d[k] * c[k,i,j] * x[k,i,j] for k in K for i,j in A) +
               gp.quicksum(f[i,j] * y[i,j] for i,j in A))
        model.setObjective(obj, GRB.MINIMIZE)

        # Add constraints
        # 1. Flow conservation constraints
        for k in K:
            for i in N:
                origin, destination = OD[k]
                delta = self.get_delta(i, origin, destination)
                
                model.addConstr(
                    gp.quicksum(x[k,i,j] for j in self.get_Ni_plus(i, A)) -
                    gp.quicksum(x[k,j,i] for j in self.get_Ni_minus(i, A)) == delta,
                    name=f"FlowBalance_k{k}_n{i}"
                )

        # 2. Capacity constraints
        for i,j in A:
            model.addConstr(
                gp.quicksum(d[k] * x[k,i,j] for k in K) <= u[i,j] * y[i,j],
                name=f"ArcCapacity_{i}_{j}"
            )

        # 3. Flow bounds
        for k in K:
            for i,j in A:
                model.addConstr(x[k,i,j] >= 0, name=f"FlowLB_k{k}_{i}_{j}")
                model.addConstr(x[k,i,j] <= 1, name=f"FlowUB_k{k}_{i}_{j}")

        # 4. Design variables bounds
        for i,j in A:
            model.addConstr(y[i,j] >= 0, name=f"DesignNonnegative_{i}_{j}")

        # Store instance data
        instance_data = {
            "N": N,
            "A": A,
            "K": K,
            "u": u,
            "f": f,
            "c": c,
            "d": d,
            "OD": OD
        }

        return model

if __name__ == '__main__':
    def test_generator():
        # Create generator
        generator = Generator()
        
        # Generate instance
        model = generator.generate_instance()
        
        model.write("mcnd.lp")
        
        # Solve model
        import time
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

       
        print(f"Solve Time: {solve_time:.2f} seconds")
        
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
        
        print(model.NumVars, model.NumConstrs)

    test_generator()
