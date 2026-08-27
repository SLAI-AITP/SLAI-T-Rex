import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import reciprocal_rounded, round_business_float, uniform_rounded


def _sample_count(value):
    if isinstance(value, int):
        return value
    return random.randint(int(value[0]), int(value[1]))


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Production Planning optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_products: Number of products
                - n_stages: Number of stages
                - rate_range: Tuple of (min, max) for production rates (tons per hour)
                - available_range: Tuple of (min, max) for available hours per stage
                - profit_range: Tuple of (min, max) for profit per ton
                - commit_range: Tuple of (min, max) for minimum production commitment
                - market_range: Tuple of (min, max) for maximum market demand
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "production_planning"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are:
        - Products: Set of products \( p \)
        - Stages: Set of stages \( s \)
        
        Parameters:
        - \( \text{Rate}_{p,s} \): Production rate (tons per hour) of product \( p \) in stage \( s \)
        - \( \text{Available}_{s} \): Available hours per week in stage \( s \)
        - \( \text{Profit}_{p} \): Profit per ton for product \( p \)
        - \( \text{Commit}_{p} \): Minimum production commitment for product \( p \)
        - \( \text{Market}_{p} \): Maximum market demand for product \( p \)
        
        Decision Variable:
        - \( \text{Production}_{p} \): Tons of product \( p \) to be produced (continuous variable)
        
        Objective:
        Maximize total profit:
        \[
        \text{Maximize} \quad \sum_{p \in \text{Products}} \text{Profit}_{p} \cdot \text{Production}_{p}
        \]
        
        Constraints:
        1. Time Constraint:
        \[
        \sum_{p \in \text{Products}} \left( \frac{1}{\text{Rate}_{p,s}} \cdot \text{Production}_{p} \right) \leq \text{Available}_{s} \quad \forall s \in \text{Stages}
        \]
        2. Commit Constraint:
        \[
        \text{Commit}_{p} \leq \text{Production}_{p} \quad \forall p \in \text{Products}
        \]
        3. Market Constraint:
        \[
        \text{Production}_{p} \leq \text{Market}_{p} \quad \forall p \in \text{Products}
        \]
        """
        default_parameters = {
            "n_products": (3, 5),
            "n_stages": (3, 6),
            "rate_range": (3, 10),
            "available_range": (150, 300),
            "profit_range": (100, 500),
            "commit_range": (10, 50),
            "market_range": (120, 200),
            "capacity_tightness_range": (0.38, 0.72),
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
        Generate a Production Planning problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (products, stages, rates, available hours, profits, commitments, market demands)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the production planning problem
        """
        
        # Randomly select number of products and stages
        self.n_products = _sample_count(self.n_products)
        self.n_stages = _sample_count(self.n_stages)
        
        # Generate products and stages
        products = [f"product_{i}" for i in range(self.n_products)]
        stages = [f"stage_{j}" for j in range(self.n_stages)]
        
        # Generate random parameters. Stage availability is sampled from the
        # generated commit/market envelope so at least one stage becomes a real
        # bottleneck instead of letting every product hit its market upper bound.
        rate = {(p, s): random.randint(*self.rate_range) for p in products for s in stages}
        profit = {p: random.randint(*self.profit_range) for p in products}
        commit = {p: random.randint(*self.commit_range) for p in products}
        market = {p: random.randint(*self.market_range) for p in products}
        for p in products:
            if market[p] <= commit[p]:
                market[p] = commit[p] + random.randint(30, 100)
        tightness_range = self.parameters.get("capacity_tightness_range", (0.45, 0.85))
        available = {}
        stage_tightness = {}
        processing_hours_per_unit = {(p, s): reciprocal_rounded(rate[p, s]) for p in products for s in stages}
        for s in stages:
            min_hours = sum(commit[p] * processing_hours_per_unit[p, s] for p in products)
            max_hours = sum(market[p] * processing_hours_per_unit[p, s] for p in products)
            tightness = uniform_rounded(*tightness_range)
            if max_hours > min_hours:
                sampled_available = min_hours + tightness * (max_hours - min_hours)
            else:
                sampled_available = min_hours * 1.05
            available[s] = round_business_float(max(sampled_available, min_hours * 1.01))
            stage_tightness[s] = round_business_float(tightness)
        capacity_envelope_by_stage = {}
        for s in stages:
            commit_hours = round_business_float(sum(commit[p] * processing_hours_per_unit[p, s] for p in products))
            market_hours = round_business_float(sum(market[p] * processing_hours_per_unit[p, s] for p in products))
            capacity_envelope_by_stage[s] = {
                "commit_hours_required": commit_hours,
                "market_hours_required": market_hours,
                "available_hours": available[s],
                "slack_above_commit_hours": round_business_float(available[s] - commit_hours),
                "shortfall_below_full_market_hours": round_business_float(max(0.0, market_hours - available[s])),
                "capacity_tightness": stage_tightness[s],
            }
        planned_bottleneck_candidates = [
            s
            for s, envelope in capacity_envelope_by_stage.items()
            if envelope["shortfall_below_full_market_hours"] > 0
        ]
        self.parameters.update(
            {
                "products": products,
                "stages": stages,
                "rate": {f"{p}|{s}": rate[p, s] for p in products for s in stages},
                "processing_hours_per_unit": {
                    f"{p}|{s}": processing_hours_per_unit[p, s] for p in products for s in stages
                },
                "available": available,
                "profit": profit,
                "commit": commit,
                "market": market,
                "stage_capacity_tightness": stage_tightness,
                "product_table": {
                    p: {
                        "profit_per_ton": profit[p],
                        "minimum_commitment_tons": commit[p],
                        "maximum_market_tons": market[p],
                    }
                    for p in products
                },
                "stage_capacity_table": {
                    s: {
                        "available_hours": available[s],
                        "capacity_tightness": stage_tightness[s],
                    }
                    for s in stages
                },
                "capacity_envelope_by_stage": capacity_envelope_by_stage,
                "planned_bottleneck_candidate_stages": planned_bottleneck_candidates,
                "compact_steel_product_mix_tables": {
                    "sets": {
                        "products": products,
                        "production_stages": stages,
                    },
                    "product_table": {
                        "columns": ["product", "profit_per_ton", "minimum_commitment_tons", "maximum_market_tons"],
                        "rows": [
                            [p, profit[p], commit[p], market[p]]
                            for p in products
                        ],
                    },
                    "stage_capacity_table": {
                        "columns": ["stage", "available_hours"],
                        "rows": [
                            [s, available[s]]
                            for s in stages
                        ],
                    },
                    "processing_hours_per_ton_matrix": {
                        "columns": stages,
                        "rows": [
                            {
                                "product": p,
                                "values": [processing_hours_per_unit[p, s] for s in stages],
                            }
                            for p in products
                        ],
                    },
                    "source_model_note": (
                        "Stage capacity is sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]. "
                        "Product market upper bounds are separate from stage available hours."
                    ),
                },
                "decision_variables": [
                    "Production[p] continuous tons of product p to produce",
                ],
                "objective_terms": [
                    "profit_per_ton_times_production_tons",
                ],
                "required_constraints": [
                    "stage_time_capacity_by_processing_hours_per_ton",
                    "minimum_product_commitment",
                    "maximum_product_market_bound",
                    "continuous_nonnegative_production_tons",
                ],
                "structured_problem_data": {
                    "sets": {
                        "products": "P",
                        "production_stages": "S",
                    },
                    "parameters": [
                        "profit[p]",
                        "commit[p]",
                        "market[p]",
                        "rate[p,s]",
                        "processing_hours_per_unit[p,s]=1/rate[p,s]",
                        "available[s]",
                    ],
                    "constraints": [
                        "sum_p processing_hours_per_unit[p,s] * Production[p] <= available[s] for every production stage",
                        "Production[p] >= commit[p] for every product",
                        "Production[p] <= market[p] for every product",
                    ],
                    "objective": "maximize total profit across products",
                },
                "required_parameter_presentation": [
                    "List products and production stages.",
                    "List profit, minimum commitment, and maximum market demand for each product.",
                    "List production rate or processing hours per ton for every product-stage pair.",
                    "List available hours for each stage.",
                    "State stage capacity uses processing hours per ton times production tons.",
                    "Do not confuse stage available hours with product maximum market tons.",
                    "State production variables are continuous tons, not binary product selection.",
                    "State this model has no inventory, periods, setup variables, or sequencing constraints.",
                ],
                "business_interpretation_guardrails": [
                    "This is a continuous product-mix production planning LP over production stages.",
                    "Do not reinterpret stages as time periods with inventory carryover.",
                    "Do not add binary product selection, setup, or fixed-charge variables.",
                    "Do not add sequencing, machine-ordering, or job-shop constraints.",
                    "Preserve both minimum commitment lower bounds and maximum market upper bounds.",
                    "Do not copy product market bounds into stage available-hour constraints; use the listed stage_capacity_table.",
                ],
                "generation_notes": [
                    "Stage capacities are sampled from the commit-to-market envelope so at least some stages are expected to be meaningful bottleneck candidates.",
                    "Capacity envelope metadata is for explanation and quality review; the model itself uses the listed stage time constraints.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("ProductionPlanning")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create continuous decision variables (Production[p] = tons of product p to produce)
        production = model.addVars(products, vtype=GRB.CONTINUOUS, name="Production")
        
        # Set objective: maximize total profit
        model.setObjective(
            gp.quicksum(profit[p] * production[p] for p in products),
            GRB.MAXIMIZE
        )
        
        # Add time constraints for each stage
        for s in stages:
            model.addConstr(
                gp.quicksum(processing_hours_per_unit[p, s] * production[p] for p in products) <= available[s],
                name=f"Time_{s}"
            )
        
        # Add commit and market constraints for each product
        for p in products:
            model.addConstr(
                production[p] >= commit[p],
                name=f"Commit_{p}"
            )
            model.addConstr(
                production[p] <= market[p],
                name=f"Market_{p}"
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
        
        model.write("production_planning.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
