import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Multi-Factory Schedule Problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_factories: Number of factories
                - n_months: Number of months
                - fixed_cost_range: Tuple of (min, max) for fixed costs
                - min_production_range: Tuple of (min, max) for minimum production levels
                - max_production_range: Tuple of (min, max) for maximum production levels
                - unit_cost_range: Tuple of (min, max) for unit production costs
                - demand_range: Tuple of (min, max) for monthly demand
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "multi_factory_schedule"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are F factories and M months. Each factory f (where f = 1, 2, ..., F) has the following attributes:
        - **Fixed Cost** a_f: The cost of running factory f during a month.
        - **Minimum Production Level** l_f: The minimum production level of factory f.
        - **Maximum Production Level** u_f: The maximum production level of factory f.
        - **Unit Production Cost** c_f: The cost per unit produced by factory f.
        - **Demand** d_m: The demand for month m.

        Decision Variables:
        - **z_{m,f}**: Binary variable indicating whether factory f is running in month m.
        - **x_{m,f}**: Continuous variable representing the units produced by factory f in month m.

        Objective:
        Minimize the total cost, which includes fixed costs and variable production costs:
        $$
        \text{Minimize} \quad \sum_{m \in M} \sum_{f \in F} \left( a_f \cdot z_{m,f} + c_f \cdot x_{m,f} \right)
        $$

        Constraints:
        1. **Minimum Production Level**:
           If a factory f is running in month m, the production must be at least the minimum production level:
           $$
           x_{m,f} \geq l_f \cdot z_{m,f} \quad \forall m \in M, f \in F
           $$

        2. **Maximum Production Level**:
           If a factory f is running in month m, the production must not exceed the maximum production level:
           $$
           x_{m,f} \leq u_f \cdot z_{m,f} \quad \forall m \in M, f \in F
           $$

        3. **Demand Satisfaction**:
           The total production across all factories in each month must meet or exceed the demand:
           $$
           \sum_{f \in F} x_{m,f} \geq d_m \quad \forall m \in M
           $$

        4. **Binary Decision Variable**:
           The decision variable z_{m,f} is binary:
           $$
           z_{m,f} \in \{0, 1\} \quad \forall m \in M, f \in F
           $$
        """
        default_parameters = {
            "n_factories": (3, 5),
            "n_months": (3, 5),
            "fixed_cost_range": (100, 500),
            "min_production_range": (20, 40),
            "max_production_range": (70, 120),
            "unit_cost_range": (1, 10),
            "demand_range": (100, 200)
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
        Generate a Multi-Factory Schedule Problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (factories, months, costs, production levels, demand)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the multi-factory schedule problem
        """
        
        # Randomly select number of factories and months
        if isinstance(self.n_factories, (tuple, list)):
            self.n_factories = random.randint(*self.n_factories)
        if isinstance(self.n_months, (tuple, list)):
            self.n_months = random.randint(*self.n_months)
        
        # Generate factories and months
        factories = [f"factory_{f}" for f in range(self.n_factories)]
        months = [f"month_{m}" for m in range(self.n_months)]
        
        # Generate problem data
        fixed_costs = {f: random.randint(*self.fixed_cost_range) for f in factories}
        min_production = {f: random.randint(*self.min_production_range) for f in factories}
        max_production = {f: random.randint(*self.max_production_range) for f in factories}
        unit_costs = {f: random.randint(*self.unit_cost_range) for f in factories}
        total_max_capacity = sum(max_production.values())
        largest_single_factory_capacity = max(max_production.values())
        demand = {}
        for m in months:
            sampled_demand = random.randint(*self.demand_range)
            # Keep the instance meaningful: each month should normally need a
            # combination of factories, not just the single largest plant.
            demand_floor = min(total_max_capacity, largest_single_factory_capacity + 1)
            demand[m] = min(total_max_capacity, max(sampled_demand, demand_floor))

        self.parameters.update(
            {
                "n_factories": self.n_factories,
                "n_months": self.n_months,
                "factories": factories,
                "months": months,
                "fixed_costs": dict(fixed_costs),
                "min_production": dict(min_production),
                "max_production": dict(max_production),
                "unit_costs": dict(unit_costs),
                "demand": dict(demand),
                "factory_table": {
                    f: {
                        "fixed_run_cost_per_month": fixed_costs[f],
                        "unit_production_cost": unit_costs[f],
                        "minimum_production_if_running": min_production[f],
                        "maximum_production_if_running": max_production[f],
                    }
                    for f in factories
                },
                "month_demand_table": {
                    m: {
                        "required_units": demand[m],
                        "requires_multiple_factories": demand[m] > largest_single_factory_capacity,
                    }
                    for m in months
                },
                "capacity_summary": {
                    "total_max_capacity_per_month": total_max_capacity,
                    "largest_single_factory_capacity": largest_single_factory_capacity,
                    "months_requiring_multiple_factories": [
                        m for m in months if demand[m] > largest_single_factory_capacity
                    ],
                },
                "compact_multi_factory_schedule_tables": {
                    "sets": {
                        "months": months,
                        "factories": factories,
                    },
                    "factory_table": {
                        "columns": [
                            "factory",
                            "fixed_run_cost_per_month",
                            "unit_production_cost",
                            "minimum_production_if_running",
                            "maximum_production_if_running",
                        ],
                        "rows": [
                            [
                                f,
                                fixed_costs[f],
                                unit_costs[f],
                                min_production[f],
                                max_production[f],
                            ]
                            for f in factories
                        ],
                    },
                    "month_demand_table": {
                        "columns": ["month", "required_units", "requires_multiple_factories"],
                        "rows": [
                            [m, demand[m], demand[m] > largest_single_factory_capacity]
                            for m in months
                        ],
                    },
                    "required_decision_layers": [
                        "RunDecision[m,f] binary factory run decision",
                        "Production[m,f] continuous nonnegative production quantity",
                    ],
                },
                "decision_variables": {
                    "RunDecision[m,f]": "binary; 1 if factory f is operated in month m",
                    "Production[m,f]": "continuous nonnegative units produced by factory f in month m",
                },
                "objective_terms": {
                    "fixed_run_cost": "sum_m,f fixed_costs[f] * RunDecision[m,f]",
                    "variable_production_cost": "sum_m,f unit_costs[f] * Production[m,f]",
                    "sense": "minimize",
                },
                "required_constraints": [
                    "minimum_production_if_running",
                    "maximum_production_if_running",
                    "monthly_demand_satisfaction",
                    "binary_factory_run_decision",
                    "nonnegative_monthly_production",
                ],
                "required_parameter_presentation": [
                    "list factories and months",
                    "for every factory, list fixed run cost per month",
                    "for every factory, list unit production cost",
                    "for every factory, list minimum production if running",
                    "for every factory, list maximum production if running",
                    "for every month, list required demand units",
                ],
                "structured_problem_data": {
                    "sets": {
                        "months": months,
                        "factories": factories,
                    },
                    "parameters": {
                        "factory_table": {
                            f: {
                                "fixed_run_cost_per_month": fixed_costs[f],
                                "unit_production_cost": unit_costs[f],
                                "minimum_production_if_running": min_production[f],
                                "maximum_production_if_running": max_production[f],
                            }
                            for f in factories
                        },
                        "monthly_demand": dict(demand),
                    },
                    "constraints": {
                        "minimum_production_if_running": "Production[m,f] >= minimum_production[f] * RunDecision[m,f]",
                        "maximum_production_if_running": "Production[m,f] <= maximum_production[f] * RunDecision[m,f]",
                        "monthly_demand_satisfaction": "sum_f Production[m,f] >= demand[m]",
                    },
                    "objective": "minimize monthly factory fixed run costs plus variable production costs",
                },
                "business_interpretation_guardrails": [
                    "This is a fixed-charge multi-factory monthly production planning model.",
                    "It is not a job-shop, flow-shop, aircraft landing, or machine sequencing problem.",
                    "There are no products, inventory carryover, backlog, setup inventory balances, or due-date penalties.",
                    "Demand is monthly aggregate production demand and must be met or exceeded.",
                    "RunDecision[m,f] is a binary plant-operation decision; Production[m,f] is continuous output quantity.",
                    "The lower and upper production bounds only apply when a factory is running.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("MultiFactorySchedule")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        z = model.addVars(months, factories, vtype=GRB.BINARY, name="RunDecision")
        x = model.addVars(months, factories, vtype=GRB.CONTINUOUS, name="Production")
        
        # Set objective: minimize total cost (fixed + variable)
        model.setObjective(
            gp.quicksum(fixed_costs[f] * z[m, f] + unit_costs[f] * x[m, f] for m in months for f in factories),
            GRB.MINIMIZE
        )
        
        # Add constraints
        # 1. Minimum production level
        model.addConstrs(
            (x[m, f] >= min_production[f] * z[m, f] for m in months for f in factories),
            name="MinProduction"
        )
        
        # 2. Maximum production level
        model.addConstrs(
            (x[m, f] <= max_production[f] * z[m, f] for m in months for f in factories),
            name="MaxProduction"
        )
        
        # 3. Demand satisfaction
        model.addConstrs(
            (gp.quicksum(x[m, f] for f in factories) >= demand[m] for m in months),
            name="DemandSatisfaction"
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
        
        model.write("multi_factory_schedule.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
