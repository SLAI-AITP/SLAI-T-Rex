import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Capacitated Lot-Sizing Problem generator.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_products: Number of products
                - n_periods: Number of periods
                - setup_cost_range: Tuple of (min, max) for setup costs
                - production_cost_range: Tuple of (min, max) for production costs
                - holding_cost_range: Tuple of (min, max) for holding costs
                - capacity_range: Tuple of (min, max) for period capacities
                - demand_range: Tuple of (min, max) for product demands
                - resource_usage_range: Tuple of (min, max) for resource usage
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "capacitated_lot_sizing"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        I_{it} = Σ_{k=1}^t X_{ik} - Σ_{k=1}^t d_{ik}  ∀i,t
        
        Minimize Σ_{i=1}^n Σ_{t=1}^T (S_{it}Y_{it} + C_{it}X_{it} + h_{it}I_{it})
        Subject to:
        Σ_{i=1}^n a_i X_{it} ≤ R_t                 ∀t
        X_{it} ≤ M_{it}Y_{it}                      ∀i,t
        I_{it} ≥ 0                                 ∀i,t
        Y_{it} ∈ {0,1}, X_{it} ≥ 0                ∀i,t
        """
        default_parameters = {
            "n_products": (2, 2),
            "n_periods": (6, 6),
            "setup_cost_range": (1000, 2000),
            "production_cost_range": (40, 50),
            "holding_cost_range": (4, 5),
            "capacity_range": (800, 800),
            "demand_range": (50, 100),
            "resource_usage_range": (1.5, 2)
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
        Generate a CLSP instance and create its corresponding Gurobi model.
        
        Returns:
            gp.Model: Configured Gurobi model for the CLSP
        """
        # Generate problem dimensions
        n_products = random.randint(*self.n_products)
        n_periods = random.randint(*self.n_periods)
        
        # Generate sets
        products = range(n_products)
        periods = range(n_periods)
        
        # Generate parameters
        setup_costs = {(i,t): random.randint(*self.setup_cost_range) 
                      for i in products for t in periods}
        prod_costs = {(i,t): random.randint(*self.production_cost_range) 
                     for i in products for t in periods}
        hold_costs = {(i,t): random.randint(*self.holding_cost_range) 
                     for i in products for t in periods}
        demands = {(i,t): random.randint(*self.demand_range) 
                  for i in products for t in periods}
        capacities = {t: random.randint(*self.capacity_range) for t in periods}
        resource_usage = {i: uniform_rounded(*self.resource_usage_range)
                         for i in products}
        
        # Calculate big-M values
        big_M = {(i,t): sum(demands[i,k] for k in range(t, n_periods)) 
                for i in products for t in periods}
        
        # Create Gurobi model
        model = gp.Model("CLSP")
        model.Params.OutputFlag = 0
        
        # Create variables
        X = model.addVars(products, periods, name="Production")
        Y = model.addVars(products, periods, vtype=GRB.BINARY, name="Setup")
        
        # Create inventory variables and expressions
        I = {}
        for i in products:
            for t in periods:
                # Inventory at t = Total production up to t - Total demand up to t
                I[i,t] = model.addVar(name=f"Inventory_{i}_{t}")
                model.addConstr(
                    I[i,t] == gp.quicksum(X[i,k] for k in range(t+1)) - 
                             gp.quicksum(demands[i,k] for k in range(t+1)),
                    name=f"InventoryBalance_{i}_{t}"
                )
        
        # Set objective
        model.setObjective(
            gp.quicksum(setup_costs[i,t] * Y[i,t] + 
                       prod_costs[i,t] * X[i,t] + 
                       hold_costs[i,t] * I[i,t] 
                       for i in products for t in periods),
            GRB.MINIMIZE
        )
        
        # Add capacity constraints
        for t in periods:
            model.addConstr(
                gp.quicksum(resource_usage[i] * X[i,t] for i in products) <= capacities[t],
                name=f"Capacity_{t}"
            )
        
        # Add setup forcing constraints
        for i in products:
            for t in periods:
                model.addConstr(
                    X[i,t] <= big_M[i,t] * Y[i,t],
                    name=f"Setup_{i}_{t}"
                )
        
        # Add non-negativity constraints for inventory
        for i in products:
            for t in periods:
                model.addConstr(I[i,t] >= 0, name=f"NonNegInventory_{i}_{t}")

        product_list = list(products)
        period_list = list(periods)
        period_demand_table = [
            {
                "product": i,
                "period": t,
                "period_demand": demands[i, t],
                "cumulative_demand_through_period": sum(demands[i, k] for k in range(t + 1)),
                "setup_cost": setup_costs[i, t],
                "unit_production_cost": prod_costs[i, t],
                "unit_holding_cost": hold_costs[i, t],
                "capacity_consumption_per_unit": resource_usage[i],
                "period_capacity": capacities[t],
                "setup_big_m_remaining_demand": big_M[i, t],
            }
            for i in product_list
            for t in period_list
        ]
        self.parameters.update(
            {
                "n_products": n_products,
                "n_periods": n_periods,
                "products": product_list,
                "periods": period_list,
                "period_demands": {f"{i}|{t}": demands[i, t] for i in product_list for t in period_list},
                "cumulative_demands": {
                    f"{i}|{t}": sum(demands[i, k] for k in range(t + 1))
                    for i in product_list
                    for t in period_list
                },
                "setup_costs": {f"{i}|{t}": setup_costs[i, t] for i in product_list for t in period_list},
                "unit_production_costs": {f"{i}|{t}": prod_costs[i, t] for i in product_list for t in period_list},
                "unit_holding_costs": {f"{i}|{t}": hold_costs[i, t] for i in product_list for t in period_list},
                "period_capacities": capacities,
                "capacity_consumption_per_unit": resource_usage,
                "setup_big_m_remaining_demand": {
                    f"{i}|{t}": big_M[i, t] for i in product_list for t in period_list
                },
                "compact_lotsizing_tables": {
                    "model_family": "capacitated_lot_sizing_without_backlog",
                    "products": product_list,
                    "periods": period_list,
                    "period_demand_table": period_demand_table,
                    "period_capacity_table": [
                        {"period": t, "capacity": capacities[t]} for t in period_list
                    ],
                    "capacity_consumption_per_product": [
                        {"product": i, "capacity_consumption_per_unit": resource_usage[i]}
                        for i in product_list
                    ],
                    "source_model_note": (
                        "Inventory[i,t] is cumulative production through t minus cumulative demand through t. "
                        "Equivalently, Inventory[i,t] = Inventory[i,t-1] + Production[i,t] - period_demand[i,t]. "
                        "There are no backlog variables, no backlog penalties, and no capacity-expansion decision variables."
                    ),
                },
                "decision_variables": {
                    "Production[i,t]": "continuous nonnegative production quantity for product i in period t",
                    "Setup[i,t]": "binary setup indicator for product i in period t",
                    "Inventory[i,t]": "continuous nonnegative ending inventory after period t",
                },
                "objective_terms": [
                    "fixed_setup_cost_times_setup",
                    "unit_production_cost_times_production",
                    "inventory_holding_cost_times_inventory",
                ],
                "required_constraints": [
                    "cumulative_inventory_balance_without_backlog",
                    "period_capacity_limit",
                    "production_quantity_linked_to_binary_setup_by_remaining_demand_big_m",
                    "nonnegative_inventory",
                ],
                "required_parameter_presentation": [
                    "list period demand and cumulative demand for every product-period pair",
                    "list setup cost, unit production cost, holding cost, and setup big-M for every product-period pair",
                    "list capacity consumption for every product and capacity for every period",
                    "state there are no backlog variables, no backlog penalty costs, and no capacity-expansion variables",
                    "state final inventory is not a separate hard constraint in the source model",
                ],
                "structured_problem_data": {
                    "sets": {"products": product_list, "periods": period_list},
                    "parameters": {
                        "period_demand_table": period_demand_table,
                        "period_capacity_table": [
                            {"period": t, "capacity": capacities[t]} for t in period_list
                        ],
                        "capacity_consumption_per_product": [
                            {"product": i, "capacity_consumption_per_unit": resource_usage[i]}
                            for i in product_list
                        ],
                    },
                    "constraints": {
                        "inventory_balance": "Inventory is cumulative production minus cumulative demand; no backlog.",
                        "capacity": "sum_i capacity_consumption[i] * Production[i,t] <= capacity[t]",
                        "setup_linking": "Production[i,t] <= remaining_demand_big_m[i,t] * Setup[i,t]",
                    },
                    "objective": "minimize setup, production, and inventory holding costs",
                },
                "business_interpretation_guardrails": [
                    "This is capacitated lot sizing without backlog.",
                    "Do not introduce backlog variables, backlog penalties, lost sales, or unmet-demand slack.",
                    "Do not introduce capacity-expansion variables; period capacities are fixed parameters.",
                    "Do not require zero final inventory as a separate hard constraint.",
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
        
        model.write("clsp.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    
    test_generator()
