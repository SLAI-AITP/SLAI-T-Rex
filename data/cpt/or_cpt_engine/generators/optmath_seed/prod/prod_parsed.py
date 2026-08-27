import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import reciprocal_rounded, range_uniform_rounded, round_business_float


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Profit Maximization optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_items: Number of items
                - a_range: Tuple of (min, max) for parameter a[j]
                - c_range: Tuple of (min, max) for parameter c[j]
                - u_range: Tuple of (min, max) for parameter u[j]
                - b_ratio: Ratio of b to total (sum of (1/a[j]) * u[j])
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "profit_maximization"
        self.mathematical_formulation = r"""
        ### Mathematical Model\nSuppose there are m items, each item j (where j = 1, 2, ..., m) has the following attributes:\n- **Parameter** a_j: A parameter for each item j.\n- **Parameter** c_j: The profit coefficient for item j.\n- **Parameter** u_j: The upper bound for the variable X_j.\n- **Variable** X_j: A continuous variable representing the quantity or level of activity for item j.\n\nAdditionally, there is a global parameter b representing a resource or capacity constraint.\n$$\n\\begin{aligned}\n&\\text{Maximize} && \\sum_{j=1}^m c_j X_j \\\\\n&\\text{Subject to} && \\sum_{j=1}^m \\frac{1}{a_j} X_j \\leq b \\\\\n& && 0 \\leq X_j \\leq u_j \\quad \\forall j = 1, 2, \\ldots, m\n\\end{aligned}\n$$
        """
        default_parameters = {
            "n_items": (5, 9),
            "a_range": (1, 10),
            "c_range": (5, 60),
            "u_range": (20, 120),
            "active_item_count": (2, 4),
            "partial_fill_range": (0.35, 0.75),
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
        Generate a Profit Maximization problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (items, a[j], c[j], u[j], b)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the profit maximization problem
        """
        
        self.n_items = random.randint(*self.n_items)
        
        # Generate items and their properties
        items = [f"item_{i}" for i in range(self.n_items)]
        a_values = {item: random.randint(*self.a_range) for item in items}
        c_values = {item: random.randint(*self.c_range) for item in items}
        u_values = {item: random.randint(*self.u_range) for item in items}
        
        # Choose capacity from the sorted profit-per-resource frontier so the
        # optimum normally uses several products, not only the single best item.
        resource_coefficients = {item: reciprocal_rounded(a_values[item]) for item in items}
        resource_usage_at_ub = {item: resource_coefficients[item] * u_values[item] for item in items}
        efficiency_order = sorted(items, key=lambda item: (c_values[item] * a_values[item], c_values[item]), reverse=True)
        active_min, active_max = _as_range(self.active_item_count)
        active_count = random.randint(active_min, min(active_max, self.n_items - 1))
        partial_fill = _sample_range(self.partial_fill_range)
        full_prefix = efficiency_order[: active_count - 1]
        marginal_item = efficiency_order[active_count - 1]
        b = round_business_float(
            sum(resource_usage_at_ub[item] for item in full_prefix)
            + partial_fill * resource_usage_at_ub[marginal_item]
        )
        self.parameters.update(
            {
                "n_items": self.n_items,
                "a_values": a_values,
                "c_values": c_values,
                "u_values": u_values,
                "resource_coefficients": resource_coefficients,
                "resource_capacity": b,
                "efficiency_order": efficiency_order,
                "target_active_item_count": active_count,
                "partial_fill": partial_fill,
            }
        )

        # Create Gurobi model
        model = gp.Model("ProfitMaximization")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (X[j] represents the quantity for item j)
        X = model.addVars(items, vtype=GRB.CONTINUOUS, name="X")

        # Set objective: maximize total profit
        model.setObjective(
            gp.quicksum(c_values[i] * X[i] for i in items),
            GRB.MAXIMIZE
        )

        # Add resource constraint: sum of (1/a[j]) * X[j] must not exceed b
        model.addConstr(
            gp.quicksum(resource_coefficients[i] * X[i] for i in items) <= b,
            name="ResourceConstraint"
        )

        # Add variable bounds: 0 <= X[j] <= u[j]
        for i in items:
            model.addConstr(X[i] >= 0, name=f"LowerBound_{i}")
            model.addConstr(X[i] <= u_values[i], name=f"UpperBound_{i}")

        return model


def _as_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def _sample_range(value):
    return range_uniform_rounded(value)


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("profit_maximization.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
