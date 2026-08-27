import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize the Electrical Power Problem (Unit Commitment Problem).

        Parameters:
            parameters (dict): Dictionary containing:
                - n_generator_types: Number of generator types
                - n_time_periods: Number of time periods
                - demand_range: Tuple of (min, max) for demand in each time period
                - min_output_range: Tuple of (min, max) for minimum output of generators
                - max_output_range: Tuple of (min, max) for maximum output of generators
                - base_cost_range: Tuple of (min, max) for base cost of generators
                - per_mw_cost_range: Tuple of (min, max) for cost per MW of generators
                - startup_cost_range: Tuple of (min, max) for startup cost of generators
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "electrical_power"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are T generator types and P time periods. Each generator type t has the following attributes:
        - **BaseCost**: Base operating cost per hour
        - **PerMWCost**: Cost per MW generated per hour
        - **StartupCost**: Cost to start a generator
        - **MinOutput**: Minimum power output when the generator is on
        - **MaxOutput**: Maximum power output when the generator is on

        Decision Variables:
        - x_{t,p}: Number of generators of type t that are on in time period p (integer)
        - y_{t,p}: Total power output from generators of type t in time period p (continuous)
        - z_{t,p}: Number of generators of type t to start in time period p (integer)

        Objective:
        Minimize the total cost:
        $$
        \text{Minimize} \quad \sum_{t=1}^T \sum_{p=1}^P \left( \text{BaseCost}_t \cdot x_{t,p} + \text{PerMWCost}_t \cdot y_{t,p} + \text{StartupCost}_t \cdot z_{t,p} \right)
        $$

        Constraints:
        1. **Available Generators**: Number of generators used must be less than or equal to the number available.
        2. **Demand**: Total power generated must meet the demand for each time period.
        3. **MinGeneration**: Power generation must respect the minimum output of each generator type.
        4. **MaxGeneration**: Power generation must respect the maximum output of each generator type.
        5. **Reserve Requirement**: Selected generators must be able to satisfy a demand that is 15% above the predicted demand.
        6. **Startup Constraints**: Relationship between the number of active generators and the number of startups.
        """
        default_parameters = {
            "n_generator_types": (3, 4),
            "n_time_periods": (5, 8),
            "demand_range": (100, 500),
            "min_output_range": (10, 50),
            "max_output_range": (100, 200),
            "base_cost_range": (50, 100),
            "per_mw_cost_range": (1, 5),
            "startup_cost_range": (200, 500)
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
        Generate an Electrical Power Problem instance and create its corresponding Gurobi model.

        Returns:
            gp.Model: Configured Gurobi model for the Electrical Power Problem
        """
        self.n_generator_types = random.randint(*self.n_generator_types)
        self.n_time_periods = random.randint(*self.n_time_periods)

        # Generate generator types and time periods
        generator_types = [f"type_{t}" for t in range(self.n_generator_types)]
        time_periods = [f"period_{p}" for p in range(self.n_time_periods)]

        min_output = {t: random.randint(*self.min_output_range) for t in generator_types}
        max_output = {}
        for t in generator_types:
            sampled_max = random.randint(*self.max_output_range)
            max_output[t] = max(sampled_max, min_output[t] + random.randint(30, 120))
        base_cost = {t: random.randint(*self.base_cost_range) for t in generator_types}
        per_mw_cost = {t: random.randint(*self.per_mw_cost_range) for t in generator_types}
        startup_cost = {t: random.randint(*self.startup_cost_range) for t in generator_types}
        generators_available = {t: random.randint(1, 5) for t in generator_types}
        on_start = {t: random.randint(0, generators_available[t]) for t in generator_types}
        reserve_margin = 1.15
        total_available_capacity = sum(max_output[t] * generators_available[t] for t in generator_types)
        safe_upper = max(1, int(total_available_capacity / reserve_margin * 0.85))
        safe_lower = max(1, min(self.demand_range[0], safe_upper))
        demand_upper = max(safe_lower, min(self.demand_range[1], safe_upper))
        demand = {p: random.randint(safe_lower, demand_upper) for p in time_periods}
        generator_type_table = [
            {
                "generator_type": t,
                "base_operating_cost_per_period": base_cost[t],
                "per_mw_generation_cost": per_mw_cost[t],
                "startup_cost": startup_cost[t],
                "minimum_output_mw": min_output[t],
                "maximum_output_mw": max_output[t],
                "generators_available": generators_available[t],
                "generators_on_initially": on_start[t],
            }
            for t in generator_types
        ]
        period_demand_table = [
            {
                "period": p,
                "demand_mw": demand[p],
                "reserve_required_capacity_mw": round(reserve_margin * demand[p], 4),
            }
            for p in time_periods
        ]
        self.parameters.update(
            {
                "generator_types": generator_types,
                "time_periods": time_periods,
                "demand": demand,
                "min_output": min_output,
                "max_output": max_output,
                "base_cost": base_cost,
                "per_mw_cost": per_mw_cost,
                "startup_cost": startup_cost,
                "generators_available": generators_available,
                "on_start": on_start,
                "reserve_margin": reserve_margin,
                "compact_electrical_power_tables": {
                    "model_family": "unit_commitment_electrical_power",
                    "sets": {
                        "generator_types": generator_types,
                        "time_periods": time_periods,
                    },
                    "generator_type_table": generator_type_table,
                    "period_demand_table": period_demand_table,
                    "decision_variables": {
                        "NumGenerators[t,p]": "nonnegative integer number of generators of type t that are on in period p",
                        "PowerOutput[t,p]": "continuous nonnegative total MW output from generator type t in period p",
                        "NumStart[t,p]": "nonnegative integer number of generators of type t started in period p",
                    },
                    "constraints": [
                        "NumGenerators[t,p] <= generators_available[t]",
                        "sum_t PowerOutput[t,p] >= demand[p]",
                        "PowerOutput[t,p] >= minimum_output_mw[t] * NumGenerators[t,p]",
                        "PowerOutput[t,p] <= maximum_output_mw[t] * NumGenerators[t,p]",
                        "sum_t maximum_output_mw[t] * NumGenerators[t,p] >= reserve_margin * demand[p]",
                        "NumGenerators[t,first_period] <= generators_on_initially[t] + NumStart[t,first_period]",
                        "NumGenerators[t,p] <= NumGenerators[t,previous_period] + NumStart[t,p]",
                    ],
                    "objective": "minimize base operating cost, per-MW generation cost, and startup cost",
                    "source_contract_note": (
                        "This is a deterministic unit-commitment style MILP with integer on-count and startup-count "
                        "variables by generator type and period. It has no battery storage, power-flow network, "
                        "fuel inventory, emissions cap, unit-specific identity, or minimum up/down time constraints."
                    ),
                },
            }
        )

        # Create Gurobi model
        model = gp.Model("ElectricalPower")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create decision variables
        x = model.addVars(generator_types, time_periods, vtype=GRB.INTEGER, name="NumGenerators")
        y = model.addVars(generator_types, time_periods, vtype=GRB.CONTINUOUS, name="PowerOutput")
        z = model.addVars(generator_types, time_periods, vtype=GRB.INTEGER, name="NumStart")

        # Set objective: minimize total cost
        model.setObjective(
            gp.quicksum(base_cost[t] * x[t, p] + per_mw_cost[t] * y[t, p] + startup_cost[t] * z[t, p]
                        for t in generator_types for p in time_periods),
            GRB.MINIMIZE
        )

        # Add constraints
        # 1. Available Generators
        for t in generator_types:
            for p in time_periods:
                model.addConstr(x[t, p] <= generators_available[t], name=f"Available_{t}_{p}")

        # 2. Demand
        for p in time_periods:
            model.addConstr(gp.quicksum(y[t, p] for t in generator_types) >= demand[p], name=f"Demand_{p}")

        # 3. Min Generation
        for t in generator_types:
            for p in time_periods:
                model.addConstr(y[t, p] >= min_output[t] * x[t, p], name=f"MinGeneration_{t}_{p}")

        # 4. Max Generation
        for t in generator_types:
            for p in time_periods:
                model.addConstr(y[t, p] <= max_output[t] * x[t, p], name=f"MaxGeneration_{t}_{p}")

        # 5. Reserve Requirement
        for p in time_periods:
            model.addConstr(
                gp.quicksum(max_output[t] * x[t, p] for t in generator_types) >= reserve_margin * demand[p],
                name=f"Reserve_{p}",
            )

        # 6. Startup Constraints
        for t in generator_types:
            model.addConstr(
                x[t, time_periods[0]] <= on_start[t] + z[t, time_periods[0]],
                name=f"StartupInitial_{t}",
            )
            for p in range(1, self.n_time_periods):
                model.addConstr(
                    x[t, time_periods[p]] <= x[t, time_periods[p - 1]] + z[t, time_periods[p]],
                    name=f"StartupTransition_{t}_{time_periods[p]}",
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
        
        model.write("electrical_power.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
