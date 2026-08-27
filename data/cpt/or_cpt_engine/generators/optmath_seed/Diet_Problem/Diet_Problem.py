import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded
class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Diet optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_foods: Number of food items
                - n_nutrients: Number of nutrients to track
                - cost_range: Tuple of (min, max) for food costs
                - nutrient_range: Tuple of (min, max) for nutrient content
                - volume_range: Tuple of (min, max) for food volumes
                - volume_capacity_ratio: Ratio for maximum total volume
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "diet"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Consider sets F of foods and N of nutrients with the following:
        - **Decision Variable**:
          - x_i: Number of servings of food i to consume
        - **Parameters**:
          - c_i: Cost per serving of food i
          - a_{i,j}: Amount of nutrient j in food i
          - n^{min}_j: Minimum required amount of nutrient j
          - n^{max}_j: Maximum allowed amount of nutrient j
          - v_i: Volume per serving of food i
          - m: Maximum total volume allowed
        """
        default_parameters = {
            "n_foods": (5, 10),
            "n_nutrients": (3, 5),
            "cost_range": (1, 10),
            "nutrient_range": (1, 50),
            "volume_range": (1, 5),
            "volume_capacity_ratio": 1.2,
            "serving_upper_bound": 4.0
        }
        
        if parameters is None or not parameters:
            parameters = default_parameters
        for key, value in parameters.items():
            setattr(self, key, value)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
    def generate_instance(self):
        """
        Generate a Diet problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for the diet problem
        """
        # Generate random numbers of foods and nutrients
        self.n_foods = random.randint(*self.n_foods)
        self.n_nutrients = random.randint(*self.n_nutrients)
        
        # Create sets
        foods = [f"food_{i}" for i in range(self.n_foods)]
        nutrients = [f"nutrient_{i}" for i in range(self.n_nutrients)]
        
        # Generate parameters
        costs = {f: random.randint(*self.cost_range) for f in foods}
        nutrient_content = {(f,n): random.randint(*self.nutrient_range) 
                          for f in foods for n in nutrients}
        volumes = {f: random.randint(*self.volume_range) for f in foods}
        
        target_food_count = random.randint(2, min(4, self.n_foods))
        target_foods = random.sample(foods, target_food_count)
        target_servings = {f: 0.0 for f in foods}
        for f in target_foods:
            target_servings[f] = uniform_rounded(0.8, self.serving_upper_bound)

        # Generate nutrient bounds around a known feasible target diet.
        nutrient_mins = {}
        nutrient_maxs = {}
        for n in nutrients:
            target_amount = sum(nutrient_content[f,n] * target_servings[f] for f in foods)
            nutrient_mins[n] = max(1, int(target_amount * 0.85))
            nutrient_maxs[n] = max(nutrient_mins[n] + 1, int(target_amount * 1.30))
        
        # Calculate volume capacity
        target_volume = sum(volumes[f] * target_servings[f] for f in foods)
        max_volume = max(1, int(target_volume * self.volume_capacity_ratio))
        
        # Create Gurobi model
        model = gp.Model("Diet")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        x = model.addVars(foods, name="Servings", vtype=GRB.CONTINUOUS, lb=0, ub=self.serving_upper_bound)
        
        # Set objective: minimize total cost
        model.setObjective(
            gp.quicksum(costs[f] * x[f] for f in foods),
            GRB.MINIMIZE
        )
        
        # Add nutrient requirement constraints
        for n in nutrients:
            model.addConstr(
                gp.quicksum(nutrient_content[f,n] * x[f] for f in foods) >= nutrient_mins[n],
                name=f"MinNutrient_{n}"
            )
            model.addConstr(
                gp.quicksum(nutrient_content[f,n] * x[f] for f in foods) <= nutrient_maxs[n],
                name=f"MaxNutrient_{n}"
            )
        
        # Add volume constraint
        model.addConstr(
            gp.quicksum(volumes[f] * x[f] for f in foods) <= max_volume,
            name="VolumeLimit"
        )

        self.target_servings = target_servings
        self.nutrient_mins = nutrient_mins
        self.nutrient_maxs = nutrient_maxs
        self.volumes = volumes
        return model
if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("diet.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Cost: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
