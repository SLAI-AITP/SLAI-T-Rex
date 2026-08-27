import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize DLSP optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_items: Number of items (products)
                - n_machines: Number of machines
                - n_periods: Number of time periods
                - setup_cost_range: Tuple of (min, max) for setup costs
                - startup_cost_range: Tuple of (min, max) for startup costs
                - holding_cost_range: Tuple of (min, max) for holding costs
                - backlog_cost_range: Tuple of (min, max) for backlog costs
                - startup_time_range: Tuple of (min, max) for startup times
                - capacity_range: Tuple of (min, max) for machine capacities
                - demand_range: Tuple of (min, max) for product demands
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "DLSP"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Sets: Items (I), Machines (M), Time Periods (T)
        Variables: y_{imt} (production), z_{imt} (startup), x_{imt} (quantity), s_{it} (inventory), b_{it} (backlog)
        
        $$
        \begin{aligned}
        &\text{Minimize} && \sum_{i,m,t} (f y_{imt} + g z_{imt}) + \sum_{i,t} (h_i s_{it} + π_i b_{it}) \\
        &\text{Subject to:} && s_{i,t-1} - b_{i,t-1} + \sum_m x_{imt} = d_{it} + s_{it} - b_{it} && \forall i,t \\
        & && x_{imt} + ST_m z_{imt} \leq C_m y_{imt} && \forall i,m,t \\
        & && \sum_i y_{imt} \leq 1 && \forall m,t \\
        & && z_{imt} \geq y_{imt} - y_{i,m,t-1} && \forall i,m,t>1 \\
        & && z_{im1} = y_{im1} && \forall i,m
        \end{aligned}
        $$
        """
        default_parameters = {
            "n_items": (3, 5),
            "n_machines": (2, 3),
            "n_periods": (5, 8),
            "setup_cost_range": (150, 250),
            "startup_cost_range": (100, 200),
            "holding_cost_range": (1, 3),
            "backlog_cost_range": (8, 12),
            "startup_time_range": (10, 20),
            "capacity_range": (80, 120),
            "demand_range": (20, 50)
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
        Generate a DLSP instance and create its corresponding Gurobi model.
        """
        # Generate problem dimensions
        n_items = random.randint(*self.n_items)
        n_machines = random.randint(*self.n_machines)
        n_periods = random.randint(*self.n_periods)

        # Generate sets
        items = list(range(n_items))
        machines = list(range(n_machines))
        periods = list(range(n_periods))

        # Generate parameters
        setup_cost = uniform_rounded(*self.setup_cost_range)
        startup_cost = uniform_rounded(*self.startup_cost_range)
        holding_costs = {i: uniform_rounded(*self.holding_cost_range) for i in items}
        backlog_costs = {i: uniform_rounded(*self.backlog_cost_range) for i in items}
        startup_times = {m: uniform_rounded(*self.startup_time_range) for m in machines}
        capacities = {m: uniform_rounded(*self.capacity_range) for m in machines}
        demands = {(i,t): uniform_rounded(*self.demand_range) for i in items for t in periods}
        self._record_instance_metadata(
            items,
            machines,
            periods,
            setup_cost,
            startup_cost,
            holding_costs,
            backlog_costs,
            startup_times,
            capacities,
            demands,
        )

        # Create Gurobi model
        model = gp.Model("DLSP")
        model.Params.OutputFlag = 0

        # Create variables
        y = model.addVars(items, machines, periods, vtype=GRB.BINARY, name="Production")
        z = model.addVars(items, machines, periods, vtype=GRB.BINARY, name="Startup")
        x = model.addVars(items, machines, periods, name="Amount")
        s = model.addVars(items, periods, name="Stock")
        b = model.addVars(items, periods, name="Backlog")

        # Set objective
        model.setObjective(
            gp.quicksum(setup_cost * y[i,m,t] + startup_cost * z[i,m,t] 
                       for i in items for m in machines for t in periods) +
            gp.quicksum(holding_costs[i] * s[i,t] + backlog_costs[i] * b[i,t] 
                       for i in items for t in periods),
            GRB.MINIMIZE
        )

        # Add constraints
        # Flow balance
        for i in items:
            for t in periods:
                if t == 0:
                    model.addConstr(
                        gp.quicksum(x[i,m,t] for m in machines) == 
                        demands[i,t] + s[i,t] - b[i,t],
                        name=f"FlowBalance_{i}_{t}",
                    )
                else:
                    model.addConstr(
                        s[i,t-1] - b[i,t-1] + gp.quicksum(x[i,m,t] for m in machines) == 
                        demands[i,t] + s[i,t] - b[i,t],
                        name=f"FlowBalance_{i}_{t}",
                    )

        # Capacity constraints
        for i in items:
            for m in machines:
                for t in periods:
                    model.addConstr(
                        x[i,m,t] + startup_times[m] * z[i,m,t] <= capacities[m] * y[i,m,t],
                        name=f"Capacity_{i}_{m}_{t}",
                    )

        # Machine constraints
        for m in machines:
            for t in periods:
                model.addConstr(
                    gp.quicksum(y[i,m,t] for i in items) <= 1,
                    name=f"OneItemPerMachinePeriod_{m}_{t}",
                )

        # Startup constraints
        for i in items:
            for m in machines:
                # First period
                model.addConstr(z[i,m,0] == y[i,m,0], name=f"StartupInitial_{i}_{m}")
                # Other periods
                for t in periods:
                    if t > 0:
                        model.addConstr(z[i,m,t] >= y[i,m,t] - y[i,m,t-1], name=f"StartupTransition_{i}_{m}_{t}")
                        model.addConstr(y[i,m,t-1] + z[i,m,t] <= 1, name=f"NoStartupIfAlreadyRunning_{i}_{m}_{t}")

        return model

    def _record_instance_metadata(
        self,
        items,
        machines,
        periods,
        setup_cost,
        startup_cost,
        holding_costs,
        backlog_costs,
        startup_times,
        capacities,
        demands,
    ):
        self.parameters.update(
            {
                "items": list(items),
                "machines": list(machines),
                "periods": list(periods),
                "setup_cost": setup_cost,
                "startup_cost": startup_cost,
                "holding_costs": {str(i): holding_costs[i] for i in items},
                "backlog_costs": {str(i): backlog_costs[i] for i in items},
                "startup_times": {str(m): startup_times[m] for m in machines},
                "machine_capacities": {str(m): capacities[m] for m in machines},
                "demands": {f"{i}|{t}": demands[i, t] for i in items for t in periods},
                "compact_lotsizing_tables": {
                    "sets": {
                        "items": list(items),
                        "machines": list(machines),
                        "periods": list(periods),
                    },
                    "global_costs": {
                        "setup_cost_per_active_item_machine_period": setup_cost,
                        "startup_cost_per_start_event": startup_cost,
                        "omitted_objective_terms": ["per-unit production cost"],
                        "source_objective_note": "There is no per-unit production cost term in this source model.",
                    },
                    "item_cost_table": {
                        "columns": ["item", "holding_cost", "backlog_cost"],
                        "rows": [
                            [i, holding_costs[i], backlog_costs[i]]
                            for i in items
                        ],
                    },
                    "machine_table": {
                        "columns": ["machine", "capacity", "startup_time"],
                        "rows": [
                            [m, capacities[m], startup_times[m]]
                            for m in machines
                        ],
                    },
                    "demand_matrix": {
                        "columns": list(periods),
                        "rows": [
                            {
                                "item": i,
                                "values": [demands[i, t] for t in periods],
                            }
                            for i in items
                        ],
                    },
                },
                "decision_variables": {
                    "Amount[i,m,t]": "continuous production quantity of item i on machine m in period t",
                    "Production[i,m,t]": "binary; 1 if machine m is set to produce item i in period t",
                    "Startup[i,m,t]": "binary; 1 if item i starts on machine m in period t",
                    "Stock[i,t]": "continuous ending inventory of item i after period t",
                    "Backlog[i,t]": "continuous backlog of item i after period t",
                },
                "objective_terms": [
                    "setup_cost_times_binary_production_state",
                    "startup_cost_times_binary_startup_event",
                    "holding_cost_times_inventory",
                    "backlog_cost_times_backlog",
                ],
                "required_constraints": [
                    "net_inventory_balance_with_backlog_carryover",
                    "production_capacity_with_startup_time",
                    "one_item_per_machine_period",
                    "startup_initial_equals_production_initial",
                    "startup_transition_from_previous_period",
                    "no_startup_if_machine_already_running_same_item",
                ],
                "required_parameter_presentation": [
                    "list item, machine, and period sets",
                    "list the full item-period demand matrix",
                    "list holding and backlog cost for every item",
                    "list machine capacity and startup time for every machine",
                    "state setup cost and startup cost are global scalar costs",
                    "state there is no per-unit production cost term in the source objective",
                    "state net inventory is Stock minus Backlog and carries over period to period",
                    "state each machine can produce at most one item in each period",
                ],
                "structured_problem_data": {
                    "sets": {
                        "items": list(items),
                        "machines": list(machines),
                        "periods": list(periods),
                    },
                    "parameters": {
                        "demand_matrix": "see compact_lotsizing_tables.demand_matrix",
                        "item_cost_table": "see compact_lotsizing_tables.item_cost_table",
                        "machine_table": "see compact_lotsizing_tables.machine_table",
                        "global_costs": "see compact_lotsizing_tables.global_costs",
                    },
                    "constraints": {
                        "flow_balance": "previous Stock - previous Backlog + production equals demand + current Stock - current Backlog",
                        "capacity": "Amount + startup_time * Startup <= machine_capacity * Production",
                        "machine_exclusivity": "sum_i Production[i,m,t] <= 1",
                        "startup_logic": "Startup tracks transitions from not producing to producing an item on a machine",
                    },
                    "objective": "minimize setup, startup, holding, and backlog costs; no unit production cost term",
                },
                "business_interpretation_guardrails": [
                    "This is a single-level small-bucket dynamic lot-sizing problem with machines, periods, inventory, and backlog.",
                    "Do not invent per-unit production costs; the source objective has setup, startup, holding, and backlog terms only.",
                    "Do not use placeholder demand, capacity, startup time, or cost values; use compact_lotsizing_tables exactly.",
                    "Do not collapse machine-period binary setup/startup decisions into a single order setup variable.",
                ],
            }
        )

if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("dlsp.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    test_generator()
