import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Cutting Stock optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_widths: Number of widths to be cut
                - roll_width: Width of the raw rolls
                - orders_range: Tuple of (min, max) for number of orders per width
                - num_patterns: Number of cutting patterns
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "cutting_stock"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are:
        - A set of widths `Width` to be cut.
        - A set of patterns `Patterns`.
        - Each pattern `j` has a certain number of rolls of each width `i` (`NumRollsWidth_{i,j}`).
        - Each width `i` has a certain number of orders (`Orders_i`).
        - The raw roll has a fixed width `RollWidth`.
        Decision Variables:
        - `Cut_j`: Number of rolls cut using pattern `j`.
        Objective:
        - Minimize the total number of raw rolls cut:
        $$\text{Minimize} \quad \sum_{j \in \text{Patterns}} \text{Cut}_j$$
        Constraints:
        1. For each width `i`, the planned cuts exactly meet the orders:
        $$\sum_{j \in \text{Patterns}} \text{NumRollsWidth}_{i,j} \cdot \text{Cut}_j = \text{Orders}_i \quad \forall i \in \text{Width}$$
        2. For each pattern `j`, the total width of rolls must not exceed the raw roll width:
        $$\sum_{i \in \text{Width}} i \cdot \text{NumRollsWidth}_{i,j} \leq \text{RollWidth} \quad \forall j \in \text{Patterns}$$
        """
        default_parameters = {
            "n_widths": (3, 5),
            "roll_width": (50, 120),
            "orders_range": (10, 50),
            "num_patterns": (10, 20),
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
        Generate a Cutting Stock problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (widths, orders, patterns, and rolls per pattern)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the Cutting Stock problem
        """
        
        # Generate widths as numbers (integers)
        self.n_widths = random.randint(*self.n_widths)  
        self.roll_width = random.randint(*self.roll_width)
        self.num_patterns = random.randint(*self.num_patterns)
        
        widths = sorted(random.sample(range(5, max(6, self.roll_width // 2)), self.n_widths))
        
        # Generate patterns and their rolls per width
        patterns = [f"pattern_{j}" for j in range(self.num_patterns)]
        num_rolls_width = {}
        for pattern in patterns:
            remaining_width = self.roll_width
            has_piece = False
            for width in sorted(widths, reverse=True):
                max_count = remaining_width // width
                count = random.randint(0, min(3, max_count)) if max_count > 0 else 0
                num_rolls_width[width, pattern] = count
                remaining_width -= width * count
                has_piece = has_piece or count > 0
            if not has_piece:
                width = random.choice(widths)
                num_rolls_width[width, pattern] = 1

        target_pattern_count = random.randint(3, min(5, self.num_patterns))
        target_patterns = random.sample(patterns, target_pattern_count)
        target_pattern_counts = {
            pattern: random.randint(1, 3)
            for pattern in target_patterns
        }
        orders = {
            width: sum(num_rolls_width[width, pattern] * target_pattern_counts[pattern] for pattern in target_patterns)
            for width in widths
        }
        self.widths = widths
        self.orders = orders
        self.num_rolls_width = num_rolls_width
        self.target_patterns = target_patterns
        self.target_pattern_counts = target_pattern_counts
        
        # Create Gurobi model
        model = gp.Model("CuttingStock")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables (Cut_j = number of stock rolls cut using pattern j)
        cut = model.addVars(patterns, lb=0, vtype=GRB.INTEGER, name="Cut")
        
        # Set objective: minimize total number of raw rolls cut
        model.setObjective(
            gp.quicksum(cut[j] for j in patterns),
            GRB.MINIMIZE
        )
        
        # Add constraints:
        # 1. For each width, the planned cuts exactly meet the orders. The
        # sampled target patterns are known feasible; equality avoids trivial
        # overcut solutions whose demand constraints are all slack.
        for width in widths:
            model.addConstr(
                gp.quicksum(num_rolls_width[width, j] * cut[j] for j in patterns) == orders[width],
                name=f"Fill_{width}"
            )
        
        # 2. For each pattern, the total width of rolls must not exceed the raw roll width
        for pattern in patterns:
            model.addConstr(
                gp.quicksum(width * num_rolls_width[width, pattern] for width in widths) <= self.roll_width,
                name=f"Check_{pattern}"
            )
        pattern_table = {
            pattern: {
                "pattern_width_used": sum(width * num_rolls_width[width, pattern] for width in widths),
                "unused_width": self.roll_width - sum(width * num_rolls_width[width, pattern] for width in widths),
                "pieces_by_width": {
                    str(width): num_rolls_width[width, pattern]
                    for width in widths
                    if num_rolls_width[width, pattern] > 0
                },
            }
            for pattern in patterns
        }
        self.parameters.update(
            {
                "widths": widths,
                "patterns": patterns,
                "roll_width": self.roll_width,
                "orders": {str(width): orders[width] for width in widths},
                "num_rolls_width": {
                    f"{width}|{pattern}": num_rolls_width[width, pattern]
                    for width in widths
                    for pattern in patterns
                },
                "target_pattern_count": target_pattern_count,
                "compact_cutting_stock_tables": {
                    "sets": {
                        "order_widths": widths,
                        "cutting_patterns": patterns,
                    },
                    "roll_width": self.roll_width,
                    "order_table": {
                        "columns": ["order_width", "required_piece_count"],
                        "rows": [
                            [width, orders[width]]
                            for width in widths
                        ],
                    },
                    "pattern_table": {
                        "columns": ["pattern", "pattern_width_used", "unused_width", "pieces_by_width"],
                        "rows": [
                            [
                                pattern,
                                pattern_table[pattern]["pattern_width_used"],
                                pattern_table[pattern]["unused_width"],
                                pattern_table[pattern]["pieces_by_width"],
                            ]
                            for pattern in patterns
                        ],
                    },
                    "required_decision_layers": [
                        "Cut[j] nonnegative integer number of stock rolls cut using pattern j",
                    ],
                },
                "decision_variables": {
                    "Cut[j]": "nonnegative integer number of stock rolls cut using cutting pattern j",
                },
                "objective_terms": {
                    "sense": "minimize",
                    "raw_roll_count": "sum_j Cut[j]",
                },
                "required_constraints": [
                    "exact_order_width_fulfillment",
                    "pattern_width_feasibility",
                    "nonnegative_integer_pattern_counts",
                ],
                "required_parameter_presentation": [
                    "list the stock roll width",
                    "list every required order width and required piece count",
                    "list every cutting pattern and the number of pieces of each width it produces",
                    "state Cut[j] is a nonnegative integer count of stock rolls using pattern j",
                    "state each order width must be matched exactly by the selected pattern counts",
                    "state each listed pattern already respects the stock roll width",
                ],
                "structured_problem_data": {
                    "sets": {
                        "order_widths": widths,
                        "cutting_patterns": patterns,
                    },
                    "parameters": {
                        "roll_width": self.roll_width,
                        "orders": {str(width): orders[width] for width in widths},
                        "pattern_table": pattern_table,
                    },
                    "variables": {
                        "Cut[j]": "nonnegative integer pattern-count variable",
                    },
                    "objective": "minimize the total number of stock rolls cut",
                    "constraints": {
                        "exact_fulfillment": "for every order width, the produced pieces equal the required piece count",
                        "pattern_feasibility": "every listed pattern fits within the stock roll width",
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a cutting-stock pattern-count model, not set cover, bin packing, or routing.",
                    "Do not make Cut[j] binary; it is a nonnegative integer roll count.",
                    "Do not replace exact order fulfillment with at-least fulfillment unless explicitly stated.",
                    "Do not invent cutting patterns or piece counts not listed in pattern_table.",
                    "Do not add assignment, facility, vehicle, or scheduling constraints.",
                ],
            }
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
        
        model.write("cutting_stock.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    test_generator()
