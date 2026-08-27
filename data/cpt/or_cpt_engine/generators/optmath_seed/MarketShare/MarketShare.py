import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Market Sharing optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_companies: Number of companies
                - n_markets: Number of markets
                - n_products: Number of products
                - demand_range: Tuple of (min, max) for market demand
                - cost_range: Tuple of (min, max) for unit costs
                - revenue_range: Tuple of (min, max) for unit revenues
            seed (int, optional): Random seed for repro
        """
        self.problem_type = "market_sharing"
        
        self.mathematical_formulation = r"""# Market Allocation Problem Mathematical Formulation
## Sets and Indices
- $I$: Set of companies, $i \in I$
- $J$: Set of markets, $j \in J$
- $K$: Set of products, $k \in K$
## Parameters
- $d_{jk}$: Demand of market $j$ for product $k$
- $c_{ijk}$: Unit cost for company $i$ to supply product $k$ to market $j$
- $r_{ijk}$: Unit revenue for company $i$ selling product $k$ in market $j$
## Decision Variables
- $x_{ijk}$: Quantity of product $k$ supplied by company $i$ to market $j$ (continuous)
## Objective Function
Maximize total profit:
$$\max \sum_{i\in I}\sum_{j\in J}\sum_{k\in K} (r_{ijk} - c_{ijk})x_{ijk}$$
## Constraints
1. Demand Satisfaction:
$$\sum_{i\in I} x_{ijk} = d_{jk} \quad \forall j\in J, \forall k\in K$$
2. Non-negativity:
$$x_{ijk} \geq 0 \quad \forall i\in I, \forall j\in J, \forall k\in K$$
"""
        
        default_parameters = {
            "n_companies": (2,5),
            "n_markets": (2,6),
            "n_products": (2,5),
            "demand_range": (10, 30),    
            "cost_range": (1, 5),        
            "revenue_range": (8, 15),     
        }

        if parameters is None or not parameters:
            parameters = default_parameters
        for key, value in parameters.items():
            setattr(self, key, value)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        """
        Generate a Market Sharing problem instance and create its Gurobi model.
        """
        
        # Randomly select number of companies, markets, and products
        self.n_companies = random.randint(*self.n_companies)
        self.n_markets = random.randint(*self.n_markets)
        self.n_products = random.randint(*self.n_products)
        
        # Create sets
        companies = range(self.n_companies)
        markets = range(self.n_markets)
        products = range(self.n_products)

        # Generate random parameters
        demand = {}
        for j in markets:
            for k in products:
                demand[j,k] = random.randint(*self.demand_range)

        # Generate random costs and revenues, ensuring that revenues exceed costs
        costs = {}
        revenues = {}
        for i in companies:
            for j in markets:
                for k in products:
                    cost = random.randint(*self.cost_range)
                    costs[i,j,k] = cost
                    # Ensure that revenue is at least 1 unit higher than cost
                    revenues[i,j,k] = cost + random.randint(3, 10)
        profit = {
            (i, j, k): revenues[i, j, k] - costs[i, j, k]
            for i in companies for j in markets for k in products
        }
        companies_list = list(companies)
        markets_list = list(markets)
        products_list = list(products)
        demand_table = [
            {
                "market": j,
                "product": k,
                "demand": demand[j, k],
            }
            for j in markets_list for k in products_list
        ]
        demand_matrix = {
            "columns": products_list,
            "rows": [
                {
                    "market": j,
                    "values": [demand[j, k] for k in products_list],
                }
                for j in markets_list
            ],
        }
        unit_profit_matrices_by_company = {
            "columns": products_list,
            "rows": [
                {
                    "company": i,
                    "market": j,
                    "values": [profit[i, j, k] for k in products_list],
                }
                for i in companies_list for j in markets_list
            ],
        }
        self.parameters = {
            "problem_type": self.problem_type,
            "companies": companies_list,
            "markets": markets_list,
            "products": products_list,
            "demand": {f"{j}|{k}": demand[j, k] for j in markets for k in products},
            "costs": {f"{i}|{j}|{k}": costs[i, j, k] for i in companies for j in markets for k in products},
            "revenues": {f"{i}|{j}|{k}": revenues[i, j, k] for i in companies for j in markets for k in products},
            "unit_profit": {f"{i}|{j}|{k}": profit[i, j, k] for i in companies for j in markets for k in products},
            "compact_marketshare_tables": {
                "model_family": "integer_market_share_allocation",
                "sets": {
                    "companies": companies_list,
                    "markets": markets_list,
                    "products": products_list,
                },
                "demand_matrix": demand_matrix,
                "unit_profit_matrices_by_company": unit_profit_matrices_by_company,
                "demand_table": demand_table,
                "profit_table_format": "see unit_profit_matrices_by_company; each row is a company-market pair with product columns",
                "decision_variables": [
                    "Supply[i,j,k] is the nonnegative integer quantity supplied by company i to market j for product k",
                ],
                "objective": "maximize sum_{i,j,k} unit_profit[i,j,k] * Supply[i,j,k]",
                "constraints": [
                    "for every market-product pair (j,k), sum_i Supply[i,j,k] == demand[j,k]",
                    "Supply[i,j,k] is nonnegative integer for every declared company-market-product triple",
                ],
                "source_contract_note": (
                    "This source model has exact market-product demand fulfillment and no resource capacity, "
                    "budget, channel capacity, eligibility, or nonlinear market-response constraints."
                ),
            },
            "decision_variables": {
                "Supply[i,j,k]": "nonnegative integer quantity supplied by company i to market j for product k",
            },
            "objective_terms": [
                "net_profit_times_integer_supply",
            ],
            "required_constraints": [
                "market_product_demand_exactly_satisfied",
                "nonnegative_integer_supply",
            ],
            "required_parameter_presentation": [
                "list companies, markets, and products",
                "list demand for every market-product pair",
                "list unit net profit for every company-market-product triple",
                "state Supply[i,j,k] is a nonnegative integer decision variable",
                "state there are no resource capacity, budget, channel capacity, or eligibility constraints in the source model",
                "prefer compact_marketshare_tables when writing the natural-language data tables",
            ],
            "structured_problem_data": {
                "sets": {
                    "companies": companies_list,
                    "markets": markets_list,
                    "products": products_list,
                },
                "parameters": {
                    "market_product_demand": "see compact_marketshare_tables.demand_table",
                    "company_market_product_profit": "see compact_marketshare_tables.unit_profit_matrices_by_company",
                },
                "constraints": {
                    "demand_fulfillment": "sum_i Supply[i,j,k] equals demand[j,k] for every market-product pair",
                    "nonnegativity_and_integrality": "Supply[i,j,k] is a nonnegative integer quantity",
                },
                "objective": "maximize net profit from supplied quantities",
            },
            "business_interpretation_guardrails": [
                "This is an integer market-share allocation model, not resource-capacity revenue management.",
                "Do not introduce resource capacities, budget limits, eligibility restrictions, nonlinear demand response, or price decisions.",
                "Do not relax exact demand fulfillment into optional market coverage.",
            ],
        }

        # Create Gurobi model
        model = gp.Model("MarketSharing")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Decision variables, x[i,j,k] = supply of product k by company i to market j
        x = model.addVars(companies, markets, products, 
                         vtype=GRB.INTEGER, 
                         name="supply")

        # Objective: maximize profit
        model.setObjective(
            gp.quicksum((revenues[i,j,k] - costs[i,j,k]) * x[i,j,k]
                       for i in companies for j in markets for k in products),
            GRB.MAXIMIZE
        )

        # Demand satisfaction constraints
        for j in markets:
            for k in products:
                model.addConstr(
                    gp.quicksum(x[i,j,k] for i in companies) == demand[j,k],
                    f"demand_{j}_{k}"
                )

        # Non-negativity constraints
        for i in companies:
            for j in markets:
                for k in products:
                    model.addConstr(x[i,j,k] >= 0, f"nonneg_{i}_{j}_{k}")

        # Store the parameters for solution analysis
        self.demand = demand
        self.costs = costs
        self.revenues = revenues

        return model

    def print_solution(self, model):
        """Print the solution details"""
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.ObjVal:.2f}")
            
            # Print summary statistics
            total_supply = 0
            total_revenue = 0
            total_cost = 0
            
            for i in range(self.n_companies):
                for j in range(self.n_markets):
                    for k in range(self.n_products):
                        var_name = f"supply[{i},{j},{k}]"
                        for v in model.getVars():
                            if v.VarName == var_name and v.X > 1e-6:  # Non-zero supply
                                supply = v.X
                                revenue = self.revenues[i,j,k] * supply
                                cost = self.costs[i,j,k] * supply
                                total_supply += supply
                                total_revenue += revenue
                                total_cost += cost
                                print(f"Company {i} -> Market {j}, Product {k}: "
                                      f"Supply = {supply:.2f}, "
                                      f"Revenue = {revenue:.2f}, "
                                      f"Cost = {cost:.2f}")
            
            print(f"\nSummary:")
            print(f"Total Supply: {total_supply:.2f}")
            print(f"Total Revenue: {total_revenue:.2f}")
            print(f"Total Cost: {total_cost:.2f}")
            print(f"Total Profit: {total_revenue - total_cost:.2f}")
        else:
            print("No optimal solution found")
            print(f"Status code: {model.Status}")
            # Print model statistics
            print(f"\nModel Statistics:")
            print(f"Number of variables: {model.NumVars}")
            print(f"Number of constraints: {model.NumConstrs}")
            # Try to compute IIS
            try:
                model.computeIIS()
                print("\nInfeasible constraints:")
                for c in model.getConstrs():
                    if c.IISConstr:
                        print(f"Constraint {c.ConstrName} is infeasible")
            except:
                print("Could not compute IIS")

if __name__ == '__main__':
    import time
    
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        generator.print_solution(model)
        
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        
        print(model.NumVars, model.NumConstrs)

    test_generator()
