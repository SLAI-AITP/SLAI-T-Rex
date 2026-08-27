
import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import range_uniform_rounded


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Knapsack optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_items: Number of items
                - value_range: Tuple of (min, max) for item values
                - weight_range: Tuple of (min, max) for item weights
                - capacity_ratio: Ratio of capacity to total weight (between 0 and 1)
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "knapsack"
        self.mathematical_formulation = r"""
        ### Mathematical Model\nSuppose there are m items, each item j (where j = 1, 2, ..., m) has the following attributes:\n- **Value** c_j: The value of item j.\n- **Weight** w_j: The weight of item j.\n- **Selection Variable** x_j: A binary variable indicating whether item j is selected.\n- x_j = 1: Item j is selected.\n- x_j = 0: Item j is not selected.\n\nAdditionally, the knapsack has a maximum weight capacity denoted by the constant K.\n$$\n\\begin{aligned}\n&\\text{Maximize} && \\sum_{j=1}^m c_j x_j \\\\\n&\\text{Subject to} && \\sum_{j=1}^m w_j x_j \\leq K \\\\\n& && x_j \\in \\{0, 1\\} \\quad \\forall j = 1, 2, \\ldots, m\n\\end{aligned}\n$$
        """
        default_parameters = {
            "n_items": (8, 30),
            "value_range": (10, 300),
            "weight_range": (5, 60),
            "capacity_ratio": (0.35, 0.55),
            "max_generation_attempts": 20,
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
        Generate a Knapsack problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (items, values, weights, capacity)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the knapsack problem
        """
        
        chosen_data = None
        for _ in range(getattr(self, "max_generation_attempts", 20)):
            n_items = random.randint(*self.n_items)
            items = [f"item_{i}" for i in range(n_items)]
            item_values = {item: random.randint(*self.value_range) for item in items}
            item_weights = {item: random.randint(*self.weight_range) for item in items}
            total_weight = sum(item_weights.values())
            ratio = _sample_ratio(self.capacity_ratio)
            provisional_capacity = max(1, int(total_weight * ratio))
            selected_items, used_weight = _solve_knapsack_by_dp(items, item_values, item_weights, provisional_capacity)
            if 2 <= len(selected_items) <= n_items - 2 and used_weight > 0:
                chosen_data = (n_items, items, item_values, item_weights, used_weight, selected_items)
                break
        if chosen_data is None:
            n_items = random.randint(*self.n_items)
            items = [f"item_{i}" for i in range(n_items)]
            item_values = {item: random.randint(*self.value_range) for item in items}
            item_weights = {item: random.randint(*self.weight_range) for item in items}
            total_weight = sum(item_weights.values())
            provisional_capacity = max(1, int(total_weight * _sample_ratio(self.capacity_ratio)))
            selected_items, used_weight = _solve_knapsack_by_dp(items, item_values, item_weights, provisional_capacity)
            chosen_data = (n_items, items, item_values, item_weights, max(1, used_weight), selected_items)

        self.n_items, items, item_values, item_weights, knapsack_capacity, selected_items = chosen_data
        self.parameters.update(
            {
                "n_items": self.n_items,
                "item_values": item_values,
                "item_weights": item_weights,
                "knapsack_capacity": knapsack_capacity,
                "reference_selected_items": selected_items,
            }
        )

        # Create Gurobi model
        model = gp.Model("Knapsack")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create binary decision variables (x[i] = 1 if item i is selected)
        x = model.addVars(items, vtype=GRB.BINARY, name="Items")

        # Set objective: maximize total value of selected items
        model.setObjective(
            gp.quicksum(item_values[i] * x[i] for i in items),
            GRB.MAXIMIZE
        )

        # Add capacity constraint: total weight must not exceed capacity
        model.addConstr(
            gp.quicksum(item_weights[i] * x[i] for i in items) <= knapsack_capacity,
            name="WeightCapacity"
        )

        return model


def _sample_ratio(value):
    return range_uniform_rounded(value)


def _solve_knapsack_by_dp(items, item_values, item_weights, capacity):
    # Exact dynamic programming is cheap at the default item/weight scale and
    # lets us tighten the final capacity without invoking the solver twice.
    states = {0: (0, [])}
    for item in items:
        weight = item_weights[item]
        value = item_values[item]
        next_states = dict(states)
        for used_weight, (used_value, selected) in states.items():
            candidate_weight = used_weight + weight
            if candidate_weight > capacity:
                continue
            candidate_value = used_value + value
            if candidate_value > next_states.get(candidate_weight, (-1, []))[0]:
                next_states[candidate_weight] = (candidate_value, [*selected, item])
        states = next_states
    best_weight, (best_value, best_selected) = max(
        states.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    return best_selected, best_weight


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("knapsack.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
