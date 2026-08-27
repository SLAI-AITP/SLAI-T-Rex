import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Revenue Maximization optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_flight_legs: Number of flight legs
                - n_packages: Number of packages
                - demand_range: Tuple of (min, max) for package demand
                - revenue_range: Tuple of (min, max) for package revenue
                - available_seats_range: Tuple of (min, max) for available seats per flight leg
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "revenue_maximization"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are:
        - FlightLegs: Set of flight legs (one-way non-stop flights)
        - Packages: Set of flight packages (itineraries) that can be sold

        Parameters:
        - AvailableSeats_r: Number of available seats for flight leg r
        - Demand_p: Estimated demand for package p
        - Revenue_p: Revenue gained from selling one unit of package p
        - Delta_p,r: Binary parameter indicating if package p uses flight leg r

        Decision Variable:
        - Sell_p: Number of units of package p to sell (integer)

        Objective:
        Maximize the total revenue:
        $$
        \text{Maximize} \quad \sum_{p \in \text{Packages}} \text{Revenue}_p \times \text{Sell}_p
        $$

        Constraints:
        1. Demand Constraint:
        $$
        \text{Sell}_p \leq \text{Demand}_p \quad \forall p \in \text{Packages}
        $$

        2. Capacity Constraint:
        $$
        \sum_{p \in \text{Packages}} \Delta_{p,r} \times \text{Sell}_p \leq \text{AvailableSeats}_r \quad \forall r \in \text{FlightLegs}
        $$
        """
        default_parameters = {
            "n_flight_legs": (3, 5),
            "n_packages": (5, 8),
            "demand_range": (4, 18),
            "revenue_range": (120, 650),
            "available_seats_range": (8, 45)
        }
        # Use default parameters if none are provided or if an empty dict is given
        if parameters is None or not parameters:
            parameters = default_parameters
        self.input_parameters = dict(parameters)
        self.parameters = {}
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Revenue Maximization problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (flight legs, packages, demand, revenue, available seats)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the revenue maximization problem
        """
        
        # Randomly select number of flight legs and packages
        n_flight_legs = self._sample_int(self.n_flight_legs)
        n_packages = self._sample_int(self.n_packages)
        self.n_flight_legs = n_flight_legs
        self.n_packages = n_packages
        
        # Generate flight legs and packages
        flight_legs = [f"flight_leg_{i}" for i in range(n_flight_legs)]
        packages = [f"package_{i}" for i in range(n_packages)]
        
        # Generate random parameters
        demand = {p: random.randint(*self.demand_range) for p in packages}
        revenue = {p: random.randint(*self.revenue_range) for p in packages}
        
        # Generate Delta_p,r: Binary parameter indicating if package p uses flight leg r.
        # Repair isolated packages/resources and set capacities after usage is known
        # so the problem has real allocation tradeoffs instead of always selling all demand.
        delta = {}
        for p in packages:
            used_leg_count = random.randint(1, min(3, len(flight_legs)))
            used_legs = set(random.sample(flight_legs, used_leg_count))
            for r in flight_legs:
                delta[p, r] = 1 if r in used_legs else 0
        for r in flight_legs:
            if not any(delta[p, r] for p in packages):
                delta[random.choice(packages), r] = 1
        available_seats = {}
        for r in flight_legs:
            resource_total_demand = sum(demand[p] * delta[p, r] for p in packages)
            if resource_total_demand <= 0:
                available_seats[r] = self._sample_int(self.available_seats_range)
            else:
                available_seats[r] = max(1, int(resource_total_demand * random.uniform(0.45, 0.78)))
        usage_matrix = {
            "rows": packages,
            "columns": flight_legs,
            "values": [[delta[p, r] for r in flight_legs] for p in packages],
        }
        self.parameters = {
            "problem_type": self.problem_type,
            "input_parameters": self.input_parameters,
            "flight_legs": flight_legs,
            "packages": packages,
            "available_seats": available_seats,
            "demand": demand,
            "revenue": revenue,
            "delta": {f"{p}|{r}": delta[p, r] for p in packages for r in flight_legs},
            "compact_revenue_tables": {
                "resource_capacity_table": [
                    {"resource": r, "available_seats": available_seats[r]}
                    for r in flight_legs
                ],
                "package_table": [
                    {
                        "package": p,
                        "demand_upper_bound": demand[p],
                        "revenue_per_unit": revenue[p],
                        "resources_used": [r for r in flight_legs if delta[p, r] == 1],
                    }
                    for p in packages
                ],
                "package_resource_usage_matrix": usage_matrix,
                "objective": "maximize total revenue from accepted package sales",
                "decision_variables": [
                    "Sell[p] is a nonnegative integer number of accepted sales for package p"
                ],
                "constraints": [
                    "Sell[p] <= demand[p] for every package",
                    "sum_p usage[p,r] * Sell[p] <= available_seats[r] for every flight leg",
                ],
            },
            "compact_revenue_management_tables": {
                "resource_capacity_table": [
                    {"resource": r, "capacity": available_seats[r]}
                    for r in flight_legs
                ],
                "package_table": [
                    {
                        "package": p,
                        "demand_upper_bound": demand[p],
                        "revenue_per_unit": revenue[p],
                        "resources_used": [r for r in flight_legs if delta[p, r] == 1],
                    }
                    for p in packages
                ],
                "package_resource_usage_matrix": usage_matrix,
                "objective": "maximize total revenue from accepted package sales",
                "decision_variables": [
                    "Sell[p] is a nonnegative integer number of accepted sales for package p"
                ],
                "constraints": [
                    "Sell[p] <= demand[p] for every package",
                    "sum_p usage[p,r] * Sell[p] <= capacity[r] for every resource",
                ],
            },
            "structured_problem_data": {
                "sets": {"resources": flight_legs, "packages": packages},
                "parameters": {
                    "available_seats": available_seats,
                    "demand_upper_bounds": demand,
                    "revenue_per_unit": revenue,
                    "usage_matrix": usage_matrix,
                },
                "objective_sense": "maximize",
                "variable_domain": "nonnegative integer package sales",
            },
            "required_parameter_presentation": [
                "list every flight leg/resource capacity",
                "list every package with demand upper bound and revenue per unit",
                "list the complete package-resource usage matrix",
                "state Sell[p] is nonnegative integer",
                "state Sell[p] cannot exceed package demand",
                "state each resource capacity is consumed according to the usage matrix",
                "state the objective maximizes total package revenue",
            ],
            "business_interpretation_guardrails": [
                "Do not introduce dynamic pricing, stochastic arrivals, booking periods, or bid-price controls.",
                "Do not change the objective into cost minimization.",
                "Do not omit demand upper bounds or resource capacity consumption constraints.",
            ],
            "objective_terms": ["package_revenue_times_package_sales"],
            "required_constraints": [
                "package_demand_upper_bound",
                "resource_capacity_consumption_limit",
                "nonnegative_integer_package_sales",
            ],
        }

        # Create Gurobi model
        model = gp.Model("RevenueMaximization")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create integer decision variables (Sell_p: Number of units of package p to sell)
        sell = model.addVars(packages, vtype=GRB.INTEGER, name="Sell")

        # Set objective: maximize total revenue
        model.setObjective(
            gp.quicksum(revenue[p] * sell[p] for p in packages),
            GRB.MAXIMIZE
        )

        # Add demand constraints
        for p in packages:
            model.addConstr(
                sell[p] <= demand[p],
                name=f"Demand_{p}"
            )

        # Add capacity constraints
        for r in flight_legs:
            model.addConstr(
                gp.quicksum(delta[p, r] * sell[p] for p in packages) <= available_seats[r],
                name=f"Capacity_{r}"
            )

        return model

    @staticmethod
    def _sample_int(value):
        if isinstance(value, int):
            return value
        return random.randint(*value)


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("revenue_maximization.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
