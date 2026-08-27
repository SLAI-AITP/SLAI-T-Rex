import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Bin Packing optimization problem.
        Parameters:
            parameters (dict): Dictionary containing:
                - n_items: Number of items (tuple of min, max)
                - weight_range: Tuple of (min, max) for item weights
                - bin_capacity: Capacity of each bin
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "binpacking"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Consider n items, where each item i has:
        - **Weight** s_i: The weight of item i
        - **Bin Capacity** c: The uniform capacity of each bin
        - **Bin Usage Variable** y_j: A binary variable indicating whether bin j is used
        - **Assignment Variable** x_{i,j}: A binary variable indicating whether item i is assigned to bin j

        $$
        \begin{aligned}
        &\text{Minimize} && \sum_{j=1}^n y_j \\
        &\text{Subject to} && \sum_{i=1}^n s_i x_{i,j} \leq c y_j && \forall j = 1,\ldots,n \\
        & && \sum_{j=1}^n x_{i,j} = 1 && \forall i = 1,\ldots,n \\
        & && x_{i,j}, y_j \in \{0,1\} && \forall i,j = 1,\ldots,n
        \end{aligned}
        $$
        """
        
        default_parameters = {
            "n_items": (6, 14),
            "weight_range": (10, 60),
            "bin_capacity": None,
            "target_bins": (2, 5),
            "max_generation_attempts": 20,
        }

        # Use default parameters if none are provided
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
        Generate a Bin Packing problem instance and create its corresponding Gurobi model.
        
        Returns:
            gp.Model: Configured Gurobi model for the bin packing problem
        """
        n_items_range = _as_range(self.n_items)
        target_bins_range = _as_range(self.target_bins)
        last_model = None
        last_payload = None
        for _ in range(getattr(self, "max_generation_attempts", 20)):
            n_items = random.randint(*n_items_range)
            items = list(range(n_items))
            item_weights = {i: random.randint(*self.weight_range) for i in items}
            if self.bin_capacity is None:
                target_bins = random.randint(target_bins_range[0], min(target_bins_range[1], max(2, n_items - 1)))
                total_weight = sum(item_weights.values())
                bin_capacity = max(max(item_weights.values()), int(total_weight / target_bins * 1.08))
            else:
                bin_capacity = int(self.bin_capacity)
            model = self._build_model(items, item_weights, bin_capacity)
            last_model = model
            last_payload = (items, item_weights, bin_capacity)
            model.optimize()
            used_bins = sum(var.X > 0.5 for var in model.getVars() if var.VarName.startswith("y["))
            if 2 <= used_bins <= n_items - 1:
                self.n_items = n_items
                self.bin_capacity = bin_capacity
                self.parameters.update(
                    {
                        "n_items": n_items,
                        "item_weights": item_weights,
                        "bin_capacity": bin_capacity,
                        "reference_used_bins": int(used_bins),
                    }
                )
                return model
        if last_payload is not None:
            items, item_weights, bin_capacity = last_payload
            self.n_items = len(items)
            self.bin_capacity = bin_capacity
            self.parameters.update(
                {
                    "n_items": len(items),
                    "item_weights": item_weights,
                    "bin_capacity": bin_capacity,
                }
            )
        return last_model

    def _build_model(self, items, item_weights, bin_capacity):
        model = gp.Model("BinPacking")
        model.Params.OutputFlag = 0
        x = model.addVars(items, items, vtype=GRB.BINARY, name="x")
        y = model.addVars(items, vtype=GRB.BINARY, name="y")
        model.setObjective(gp.quicksum(y[j] for j in items), GRB.MINIMIZE)
        for j in items:
            model.addConstr(
                gp.quicksum(item_weights[i] * x[i, j] for i in items) <= bin_capacity * y[j],
                name=f"Capacity_{j}",
            )
        for i in items:
            model.addConstr(
                gp.quicksum(x[i, j] for j in items) == 1,
                name=f"Assignment_{i}",
            )
        for j in items[:-1]:
            model.addConstr(y[j] >= y[j + 1], name=f"UsedBinOrder_{j}")
        model._items = items
        model._weights = item_weights
        model._bin_capacity = bin_capacity
        return model


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
        
        model.write("binpacking.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal number of bins: {model.ObjVal:.0f}")
        else:
            print("No optimal solution found")
            
    test_generator()
