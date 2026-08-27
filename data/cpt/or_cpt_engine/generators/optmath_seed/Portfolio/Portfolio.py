import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import round_business_float, uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Portfolio Optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_assets: Number of assets
            - budget: Total investment budget
            - return_range: Tuple of (min, max) for expected returns
            - risk_range: Tuple of (min, max) for risk (std dev)
            - weight_bounds: Tuple of (min, max) for asset weights
        """
        self.problem_type = "portfolio_optimization"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Continuous Mean-Variance Portfolio Optimization

        Sets:
        - A: candidate assets.

        Parameters:
        - r_i: expected return of asset i.
        - Sigma_{i,j}: positive-semidefinite covariance between assets i and j.
        - R: minimum required portfolio expected return.
        - l_i, u_i: lower and upper bounds on the portfolio weight of asset i.

        Decision variables:
        - w_i: continuous portfolio weight assigned to asset i.

        Minimize:
            sum_{i in A} sum_{j in A} Sigma_{i,j} w_i w_j

        Subject to:
        - Full-investment budget: sum_i w_i = 1.
        - Target return: sum_i r_i w_i >= R.
        - Weight bounds: l_i <= w_i <= u_i.

        This generator does not include cardinality constraints or binary asset
        selection variables.
        """

        default_parameters = {
            "n_assets": (6, 10),
            "budget": 1000000,  # $1M investment
            "return_range": (0.05, 0.15),  # 5-15% annual return
            "risk_range": (0.10, 0.30),  # 10-30% annual volatility
            "weight_bounds": (0.0, 0.3),  # 0-30% per asset
            "off_diagonal_covariance_range": (-0.03, 0.05),
            "diagonal_dominance_margin": 0.05,
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
        Generate a Portfolio Optimization instance and create its Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for portfolio optimization
        """
        
        self.n_assets = random.randint(*self.n_assets)
        
        # Generate assets
        self.assets = [f"asset_{i}" for i in range(self.n_assets)]

        # Generate expected returns
        self.returns = {i: uniform_rounded(*self.return_range) for i in self.assets}

        # Generate a rounded positive-semidefinite covariance matrix. The matrix
        # is made strictly diagonally dominant after rounding, which keeps the
        # QP convex while preserving readable two-decimal business coefficients.
        self.volatilities = {i: uniform_rounded(*self.risk_range) for i in self.assets}
        self.covariance = {}
        off_diagonal_abs_sum = {asset: 0.0 for asset in self.assets}
        for i, asset1 in enumerate(self.assets):
            for asset2 in self.assets[i + 1 :]:
                value = uniform_rounded(*self.off_diagonal_covariance_range)
                self.covariance[(asset1, asset2)] = value
                self.covariance[(asset2, asset1)] = value
                off_diagonal_abs_sum[asset1] += abs(value)
                off_diagonal_abs_sum[asset2] += abs(value)
        for asset in self.assets:
            variance_floor = round_business_float(self.volatilities[asset] ** 2)
            dominance_floor = round_business_float(
                off_diagonal_abs_sum[asset] + self.diagonal_dominance_margin
            )
            self.covariance[(asset, asset)] = max(variance_floor, dominance_floor)
        target_return = _target_return(self.returns, self.weight_bounds)
        self.parameters.update(
            {
                "n_assets": self.n_assets,
                "assets": self.assets,
                "returns": self.returns,
                "volatilities": self.volatilities,
                "covariance": {
                    f"{asset_i}|{asset_j}": self.covariance[asset_i, asset_j]
                    for asset_i in self.assets
                    for asset_j in self.assets
                },
                "target_return": target_return,
                "weight_bounds": self.weight_bounds,
                "off_diagonal_covariance_range": self.off_diagonal_covariance_range,
                "diagonal_dominance_margin": self.diagonal_dominance_margin,
                "objective_terms": [
                    "portfolio_variance_quadratic_form",
                ],
                "required_constraints": [
                    "full_investment_budget_weights_sum_to_one",
                    "minimum_expected_return_requirement",
                    "continuous_asset_weight_bounds",
                    "positive_semidefinite_covariance_risk_objective",
                ],
                "business_interpretation_guardrails": [
                    "This is a continuous mean-variance allocation model, not a binary asset-selection model.",
                    "Do not add cardinality, minimum-lot, transaction-cost, or buy/sell integer constraints.",
                    "The objective minimizes portfolio variance using the full covariance matrix.",
                    "Expected returns are used in a minimum-return constraint, not as a maximization objective.",
                ],
                "compact_portfolio_tables": {
                    "assets": self.assets,
                    "expected_return_table": {
                        "columns": ["asset", "expected_return"],
                        "rows": [[asset, self.returns[asset]] for asset in self.assets],
                    },
                    "volatility_table": {
                        "columns": ["asset", "volatility"],
                        "rows": [[asset, self.volatilities[asset]] for asset in self.assets],
                    },
                    "covariance_matrix": {
                        "row_assets": self.assets,
                        "column_assets": self.assets,
                        "values": [
                            [self.covariance[asset_i, asset_j] for asset_j in self.assets]
                            for asset_i in self.assets
                        ],
                    },
                    "target_return": target_return,
                    "weight_bounds": {
                        "lower_bound": self.weight_bounds[0],
                        "upper_bound": self.weight_bounds[1],
                    },
                    "required_decision_layers": [
                        "Weights[i] continuous portfolio allocation weight",
                    ],
                    "no_binary_asset_selection": True,
                },
            }
        )

        # Create Gurobi model
        model = gp.Model("PortfolioOptimization")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Decision variables: portfolio weights
        weights = model.addVars(
            self.assets,
            lb=self.weight_bounds[0],
            ub=self.weight_bounds[1],
            name="Weights",
        )

        # Objective: minimize portfolio variance directly. With the PSD
        # covariance construction above, this is a convex QP and solves quickly
        # at seed-generation scale.
        model.setObjective(
            gp.quicksum(
                self.covariance[(i, j)] * weights[i] * weights[j]
                for i in self.assets
                for j in self.assets
            ),
            GRB.MINIMIZE,
        )

        # Constraints
        # Budget constraint (weights sum to 1)
        model.addConstr(gp.quicksum(weights[i] for i in self.assets) == 1, name="Budget")

        # Target return constraint
        model.addConstr(
            gp.quicksum(weights[i] * self.returns[i] for i in self.assets)
            >= target_return,
            name="TargetReturn",
        )

        return model

    def print_solution(self, model):
        """
        Print the solution of the Portfolio Optimization in a readable format.
        Parameters:
            model (gp.Model): Solved Gurobi model
        """
        if model.Status != GRB.OPTIMAL:
            print("No optimal solution found.")
            return

        print("\n=== Portfolio Optimization Results ===")
        # Get portfolio weights
        weights = {}
        for v in model.getVars():
            if v.VarName.startswith("Weights"):
                asset = v.VarName.split("[")[1].split("]")[0]
                weights[asset] = v.X

        # Calculate portfolio return
        port_return = sum(weights[i] * self.returns[i] for i in self.assets)

        print("\nPortfolio Allocation:")
        print(f"{'Asset':^12} {'Weight':^10} {'Return':^10} {'Risk':^10}")
        print("-" * 45)

        for asset in self.assets:
            print(
                f"{asset:^12} {weights[asset]:^10.4f} "
                f"{self.returns[asset]:^10.4f} "
                f"{self.volatilities[asset]:^10.4f}"
            )

        print("\nPortfolio Statistics:")
        print(f"Number of assets: {len(self.assets)}")
        print(f"Portfolio expected return: {port_return:.4f}")
        print(f"Highest individual weight: {max(weights.values()):.4f}")
        print(f"Lowest individual weight: {min(weights.values()):.4f}")

        # Calculate diversification metrics
        herfindahl = sum(w * w for w in weights.values())
        print(f"Herfindahl Index (concentration): {herfindahl:.4f}")


def _target_return(returns, weight_bounds):
    upper_bound = float(weight_bounds[1])
    sorted_returns = sorted((float(value) for value in returns.values()), reverse=True)
    remaining_weight = 1.0
    max_return = 0.0
    for value in sorted_returns:
        weight = min(upper_bound, remaining_weight)
        max_return += weight * value
        remaining_weight -= weight
        if remaining_weight <= 1e-9:
            break
    average_return = sum(sorted_returns) / len(sorted_returns)
    aspirational = max(average_return * 1.03, sorted_returns[min(len(sorted_returns) - 1, len(sorted_returns) // 3)])
    return round_business_float(min(aspirational, max_return * 0.85))


if __name__ == "__main__":
    import time

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(
                f"Optimal Portfolio Variance: {model.ObjVal:.6f}"
            )
            generator.print_solution(model)
        else:
            print("No optimal solution found")

    test_generator()
