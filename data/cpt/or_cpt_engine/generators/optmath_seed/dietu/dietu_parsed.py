import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Diet Problem optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_foods: Number of foods
                - n_nutrients: Number of nutrients
                - cost_range: Tuple of (min, max) for food costs
                - nutrient_amount_range: Tuple of (min, max) for nutrient amounts in foods
                - min_requirement_range: Tuple of (min, max) for minimum nutrient requirements
                - max_requirement_range: Tuple of (min, max) for maximum nutrient requirements
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "diet"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are m foods and n nutrients. Each food j (where j = 1, 2, ..., m) has the following attributes:
        - **Cost** c_j: The cost of food j.
        - **Nutrient Amount** a_{i,j}: The amount of nutrient i in food j.
        
        Each nutrient i (where i = 1, 2, ..., n) has:
        - **Minimum Requirement** l_i: The minimum required amount of nutrient i.
        - **Maximum Requirement** u_i: The maximum allowed amount of nutrient i.
        
        The decision variable x_j represents the amount of food j to buy.
        
        The objective is to minimize the total cost of the diet:
        $$
        \begin{aligned}
        &\text{Minimize} && \sum_{j=1}^m c_j x_j \\
        &\text{Subject to} && \sum_{j=1}^m a_{i,j} x_j \geq l_i \quad \forall i \in \text{MinRequirements} \\
        & && \sum_{j=1}^m a_{i,j} x_j \leq u_i \quad \forall i \in \text{MaxRequirements} \\
        & && x_j \geq 0 \quad \forall j = 1, 2, \ldots, m
        \end{aligned}
        $$
        """
        default_parameters = {
            "n_foods": (5, 9),
            "n_nutrients": (2, 3),
            "cost_range": (1, 10),
            "nutrient_amount_range": (1, 50),
            "min_requirement_range": (100, 200),
            "max_requirement_range": (300, 400),
            "serving_upper_bound": 2.0
        }
        # Use default parameters if none are provided or if an empty dict is given
        if parameters is None or not parameters:
            parameters = default_parameters
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Diet Problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (foods, costs, nutrient amounts, requirements)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the diet problem
        """
        
        # Randomly select number of foods and nutrients
        self.n_foods = random.randint(*self.n_foods)
        self.n_nutrients = random.randint(*self.n_nutrients)
        
        # Generate foods and nutrients
        foods = [f"food_{i}" for i in range(self.n_foods)]
        nutrients = [f"nutrient_{i}" for i in range(self.n_nutrients)]
        
        # Generate random costs for foods
        food_costs = {food: random.randint(*self.cost_range) for food in foods}
        
        # Generate random nutrient amounts in foods
        nutrient_amounts = {
            (nutrient, food): random.randint(*self.nutrient_amount_range)
            for nutrient in nutrients for food in foods
        }
        
        target_food_count = random.randint(2, min(4, self.n_foods))
        target_foods = random.sample(foods, target_food_count)
        target_servings = {food: 0.0 for food in foods}
        for food in target_foods:
            target_servings[food] = uniform_rounded(0.8, self.serving_upper_bound)

        # Generate nutrient requirements around a known feasible target diet.
        min_requirements = {}
        max_requirements = {}
        for nutrient in nutrients:
            target_amount = sum(nutrient_amounts[nutrient, food] * target_servings[food] for food in foods)
            min_requirements[nutrient] = max(1, int(target_amount * 0.85))
            max_requirements[nutrient] = max(min_requirements[nutrient] + 1, int(target_amount * 1.30))

        # Create Gurobi model
        model = gp.Model("DietProblem")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (x[j] = amount of food j to buy)
        x = model.addVars(foods, vtype=GRB.CONTINUOUS, lb=0, ub=self.serving_upper_bound, name="Foods")

        # Set objective: minimize total cost of the diet
        model.setObjective(
            gp.quicksum(food_costs[j] * x[j] for j in foods),
            GRB.MINIMIZE
        )

        # Add minimum nutrient requirement constraints
        for i in nutrients:
            model.addConstr(
                gp.quicksum(nutrient_amounts[i, j] * x[j] for j in foods) >= min_requirements[i],
                name=f"MinRequirement_{i}"
            )

        # Add maximum nutrient requirement constraints
        for i in nutrients:
            model.addConstr(
                gp.quicksum(nutrient_amounts[i, j] * x[j] for j in foods) <= max_requirements[i],
                name=f"MaxRequirement_{i}"
            )

        self.target_servings = target_servings
        self.min_requirements = min_requirements
        self.max_requirements = max_requirements
        return model


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("diet_problem.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Cost: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
