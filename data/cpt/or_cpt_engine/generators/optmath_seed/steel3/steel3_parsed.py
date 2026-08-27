import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import reciprocal_rounded, round_business_float, uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize the Production Planning optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_products: Number of products
                - production_rate_range: Tuple of (min, max) for production rates (tons per hour)
                - profit_range: Tuple of (min, max) for profit per ton
                - min_sold_range: Tuple of (min, max) for minimum tons sold
                - max_sold_range: Tuple of (min, max) for maximum tons sold
                - available_hours: Total available hours in a week
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "production_planning"
        self.mathematical_formulation = r"""
        ### Mathematical Model\nSuppose there are n products, each product p (where p = 1, 2, ..., n) has the following attributes:\n- **Production Rate** r_p: The production rate of product p (tons per hour).\n- **Profit** c_p: The profit per ton of product p.\n- **Minimum Sold** min_p: The minimum tons of product p that must be sold.\n- **Maximum Sold** max_p: The maximum tons of product p that can be sold.\n- **Production Variable** x_p: The tons of product p to be produced.\n\nAdditionally, the total available production hours in a week is denoted by the constant H.\n$$\n\\begin{aligned}\n&\\text{Maximize} && \\sum_{p=1}^n c_p x_p \\\\\n&\\text{Subject to} && \\sum_{p=1}^n \\frac{x_p}{r_p} \\leq H \\\\\n& && x_p \\geq \\text{min}_p \\quad \\forall p = 1, 2, \\ldots, n \\\\\n& && x_p \\leq \\text{max}_p \\quad \\forall p = 1, 2, \\ldots, n\n\\end{aligned}\n$$
        """
        default_parameters = {
            "n_products": (3, 5),
            "production_rate_range": (1, 10),
            "profit_range": (1, 100),
            "min_sold_range": (1, 50),
            "max_sold_range": (50, 100),
            "available_hours": 168  # 168 hours in a week
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
        Generate a Production Planning problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (products, production rates, profits, min/max sold)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the production planning problem
        """
        
        # Randomly select number of products
        self.n_products = random.randint(*self.n_products)
        
        # Generate products and their properties. The capacity is sampled after
        # min/max bounds so the resource constraint creates a real trade-off.
        products = [f"product_{i}" for i in range(self.n_products)]
        production_rates = {p: random.randint(*self.production_rate_range) for p in products}
        profits = {p: random.randint(*self.profit_range) for p in products}
        min_sold = {p: random.randint(*self.min_sold_range) for p in products}
        max_sold = {p: random.randint(*self.max_sold_range) for p in products}
        for p in products:
            if max_sold[p] <= min_sold[p]:
                max_sold[p] = min_sold[p] + random.randint(10, 50)

        processing_hours_per_unit = {p: reciprocal_rounded(production_rates[p]) for p in products}
        min_hours = sum(min_sold[p] * processing_hours_per_unit[p] for p in products)
        max_hours = sum(max_sold[p] * processing_hours_per_unit[p] for p in products)
        tightness_range = self.parameters.get("capacity_tightness_range", (0.45, 0.85))
        tightness = uniform_rounded(*tightness_range)
        if max_hours > min_hours:
            available_hours = min_hours + tightness * (max_hours - min_hours)
        else:
            available_hours = min_hours * 1.05
        explicit_hours = self.parameters.get("available_hours")
        if isinstance(explicit_hours, tuple):
            available_hours = uniform_rounded(*explicit_hours)
        elif isinstance(explicit_hours, (int, float)) and explicit_hours > 0:
            available_hours = min(float(explicit_hours), available_hours)
        self.available_hours = round_business_float(max(available_hours, min_hours * 1.01))
        self.parameters.update(
            {
                "products": products,
                "production_rates": production_rates,
                "processing_hours_per_unit": processing_hours_per_unit,
                "profits": profits,
                "min_sold": min_sold,
                "max_sold": max_sold,
                "available_hours": self.available_hours,
                "capacity_tightness": round_business_float(tightness),
                "compact_steel_product_mix_tables": {
                    "model_family": "single_stage_continuous_product_mix",
                    "sets": {"products": products},
                    "product_table": {
                        "columns": [
                            "product",
                            "profit_per_ton",
                            "production_rate_tons_per_hour",
                            "processing_hours_per_ton",
                            "minimum_commitment_tons",
                            "maximum_market_tons",
                        ],
                        "rows": [
                            [
                                p,
                                profits[p],
                                production_rates[p],
                                processing_hours_per_unit[p],
                                min_sold[p],
                                max_sold[p],
                            ]
                            for p in products
                        ],
                    },
                    "time_capacity": {
                        "available_hours": self.available_hours,
                        "constraint": "sum_p processing_hours_per_ton[p] * Production[p] <= available_hours",
                    },
                    "source_model_note": (
                        "This is a single-stage continuous product-mix LP. It has one shared production-hour "
                        "capacity, product minimum commitments, and product maximum market bounds. It has no "
                        "inventory, setup, binary product-selection, sequencing, or multi-stage capacity matrix."
                    ),
                },
                "decision_variables": {
                    "Production[p]": "continuous tons of product p to produce",
                },
                "objective_terms": ["profit_per_ton_times_production"],
                "required_constraints": [
                    "single_shared_production_hour_capacity",
                    "product_minimum_commitment",
                    "product_maximum_market_bound",
                ],
                "required_parameter_presentation": [
                    "list profit, production rate, processing hours per ton, minimum commitment, and maximum market bound for every product",
                    "list the single shared available production hours value",
                    "state there are no inventory, setup, binary product-selection, sequencing, or multi-stage capacity variables",
                ],
                "structured_problem_data": {
                    "sets": {"products": products},
                    "parameters": {
                        "product_table": [
                            {
                                "product": p,
                                "profit_per_ton": profits[p],
                                "production_rate_tons_per_hour": production_rates[p],
                                "processing_hours_per_ton": processing_hours_per_unit[p],
                                "minimum_commitment_tons": min_sold[p],
                                "maximum_market_tons": max_sold[p],
                            }
                            for p in products
                        ],
                        "available_hours": self.available_hours,
                    },
                    "objective": "maximize total profit",
                    "constraints": {
                        "time_capacity": "sum_p processing_hours_per_ton[p] * Production[p] <= available_hours",
                        "product_bounds": "minimum_commitment[p] <= Production[p] <= maximum_market[p]",
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a single-stage product-mix LP.",
                    "Do not introduce inventory, setup, binary product-selection, sequencing, or stage-specific capacity variables.",
                    "Do not confuse product market bounds with the shared available production hours.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("ProductionPlanning")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (x[p] = tons of product p to produce)
        x = model.addVars(products, vtype=GRB.CONTINUOUS, name="Production")

        # Set objective: maximize total profit from produced products
        model.setObjective(
            gp.quicksum(profits[p] * x[p] for p in products),
            GRB.MAXIMIZE
        )

        # Add time constraint: total production time must not exceed available hours
        model.addConstr(
            gp.quicksum(processing_hours_per_unit[p] * x[p] for p in products) <= self.available_hours,
            name="TimeCapacity"
        )

        # Add minimum and maximum production constraints
        for p in products:
            model.addConstr(x[p] >= min_sold[p], name=f"MinSold_{p}")
            model.addConstr(x[p] <= max_sold[p], name=f"MaxSold_{p}")

        return model


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("production_planning.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
