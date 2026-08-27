import gurobipy as gp
from gurobipy import GRB
import random
import json

from or_cpt_engine.generators.numeric import range_uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Set Covering optimization problem.
        Parameters:
            parameters (dict): Dictionary containing:
                - n_sets: Number of sets
                - n_elements: Number of elements
                - density: Density of set-element associations (between 0 and 1)
                - cost_range: Tuple of (min, max) for set costs
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "set_covering"
        self.mathematical_formulation = r"""
        \begin{align*}
        \min & \sum_{i=1}^{n} c_i x_i \\
        \text{s.t.} & \sum_{i \in I_j} x_i \geq 1 \quad \forall j \in M \\
        & x_i \in \{0,1\} \quad \forall i \in N
        \end{align*}
        """
        
        default_parameters = {
            "n_sets": (8, 22),
            "n_elements": (12, 35),
            "density": (0.35, 0.55),
            "cost_range": (1, 100),
            "max_generation_attempts": 20,
        }
        
        # Use default parameters if none are provided
        if parameters is None or not parameters:
            parameters = default_parameters
        parameters = dict(parameters)
            
        # Set attributes from parameters
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Set Covering problem instance and create its corresponding Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the set covering problem
        """
        self._n_sets_range = _as_range(self.n_sets)
        self._n_elements_range = _as_range(self.n_elements)
        last_model = None
        last_payload = None
        for _ in range(getattr(self, "max_generation_attempts", 20)):
            model, elements, sets, costs = self._build_candidate_model()
            last_model = model
            last_payload = (elements, sets, costs)
            model.optimize()
            selected_count = sum(var.X > 0.5 for var in model.getVars() if var.VarName.startswith("Selected"))
            if 2 <= selected_count <= self.n_sets - 1:
                self.parameters.update(
                    {
                        "n_sets": self.n_sets,
                        "n_elements": self.n_elements,
                        "sets": {str(key): sorted(value) for key, value in sets.items()},
                        "costs": costs,
                        "reference_selected_count": int(selected_count),
                    }
                )
                return model
        if last_payload is not None:
            elements, sets, costs = last_payload
            self.parameters.update(
                {
                    "n_sets": self.n_sets,
                    "n_elements": self.n_elements,
                    "sets": {str(key): sorted(value) for key, value in sets.items()},
                    "costs": costs,
                }
            )
        return last_model

    def _build_candidate_model(self):
        self.n_sets = random.randint(*self._n_sets_range)
        self.n_elements = random.randint(*self._n_elements_range)
        elements = [f"e{j}" for j in range(1, self.n_elements + 1)]
        density = _sample_density(self.density)
        min_cover = 2 if self.n_sets >= 3 else 1
        max_cover = max(min_cover, min(self.n_sets - 1, int(round(self.n_sets * density))))
        sets = {i: set() for i in range(1, self.n_sets + 1)}
        for e in elements:
            cover_count = random.randint(min_cover, max_cover)
            for set_id in random.sample(list(sets.keys()), cover_count):
                sets[set_id].add(e)
        # Avoid completely unused sets; unused sets create trivial variables that
        # add little training value and can distort duplicate signatures.
        for set_id, covered in sets.items():
            if not covered:
                covered.add(random.choice(elements))
        costs = {i: random.randint(*self.cost_range) for i in range(1, self.n_sets + 1)}
        model = gp.Model("SetCovering")
        model.Params.OutputFlag = 0
        x = model.addVars(sets.keys(), vtype=GRB.BINARY, name="Selected")
        model.setObjective(gp.quicksum(costs[i] * x[i] for i in sets.keys()), GRB.MINIMIZE)
        for e in elements:
            model.addConstr(
                gp.quicksum(x[i] for i in sets.keys() if e in sets[i]) >= 1,
                f"Cover_{e}",
            )
        return model, elements, sets, costs


def _sample_density(value):
    return range_uniform_rounded(value)


def _as_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value

if __name__ == '__main__':
    import time
    
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("set_covering.lp")
        
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    
    test_generator()
