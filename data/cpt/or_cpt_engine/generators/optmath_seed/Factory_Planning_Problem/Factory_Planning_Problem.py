import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Factory Planning optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_periods: Number of time periods
                - n_products: Number of products
                - n_machines: Dictionary of machine types and quantities
                - profit_range: Tuple of (min, max) for product profits
                - holding_cost: Storage cost per unit per period
                - machine_time_range: Tuple of (min, max) for processing times
                - machine_hours: Available hours per machine per period
                - max_inventory: Maximum storage capacity per product
                - sales_limit_range: Tuple of (min, max) for sales limits
                - target_inventory_ratio: Ratio for end-horizon inventory targets
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "factory_planning"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Multi-Period Factory Production Planning

        Sets:
        - \(T\): planning periods
        - \(P\): products
        - \(M\): machine types

        Parameters:
        - \(r_p\): revenue or profit per unit sold for product \(p\)
        - \(h\): inventory holding cost per unit per period
        - \(a_{m,p}\): machine hours of type \(m\) required to produce one unit of product \(p\)
        - \(H_{t,m}\): available machine hours of type \(m\) in period \(t\), after maintenance
        - \(U_{t,p}\): sales upper bound for product \(p\) in period \(t\)
        - \(I^{max}\): inventory upper bound for each product and period
        - \(I^{target}_p\): required final inventory for product \(p\)

        Decision variables:
        - \(x_{t,p} \ge 0\): production quantity of product \(p\) in period \(t\)
        - \(s_{t,p} \ge 0\): sales quantity of product \(p\) in period \(t\)
        - \(i_{t,p} \ge 0\): end-of-period inventory of product \(p\) in period \(t\)

        Objective:
        \[
        \max \sum_{t \in T}\sum_{p \in P} r_p s_{t,p}
             - \sum_{t \in T}\sum_{p \in P} h i_{t,p}
        \]

        Constraints:
        \[
        x_{0,p} = s_{0,p} + i_{0,p} \quad \forall p \in P
        \]
        \[
        i_{t-1,p} + x_{t,p} = s_{t,p} + i_{t,p}
        \quad \forall t \in T \setminus \{0\}, p \in P
        \]
        \[
        \sum_{p \in P} a_{m,p} x_{t,p} \le H_{t,m}
        \quad \forall t \in T, m \in M
        \]
        \[
        i_{t,p} \le I^{max}, \quad s_{t,p} \le U_{t,p}
        \quad \forall t \in T, p \in P
        \]
        \[
        i_{\max(T),p} = I^{target}_p \quad \forall p \in P
        \]

        Required interpretation:
        This is a multi-period production, sales, and inventory planning model.
        It is not a static product-mix model and not an unconstrained allocation
        problem. The product-period sales limits, machine-hour capacity after
        maintenance, inventory upper bounds, inventory flow-balance equations,
        and final inventory targets are all mandatory source constraints.
        """
        default_parameters = {
            "n_periods": 6,
            "n_products": (3, 5),
            "n_machines": {"grinder": 4, "drill": 2, "borer": 1},
            "profit_range": (150, 300),
            "holding_cost": 20,
            "machine_time_range": (1, 3),
            "machine_hours": 160,
            "max_inventory": 100,
            "sales_limit_range": (30, 80),
            "target_inventory_ratio": 0.3
        }
        
        if parameters is None or not parameters:
            parameters = default_parameters
        else:
            parameters = {**default_parameters, **dict(parameters)}
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Factory Planning problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for the factory planning problem
        """
        # Generate random number of products
        if isinstance(self.n_products, (tuple, list)):
            self.n_products = random.randint(*self.n_products)
        
        # Create sets
        periods = range(self.n_periods)
        products = [f"product_{i}" for i in range(self.n_products)]
        machines = list(self.n_machines.keys())
        
        # Generate parameters
        profits = {p: random.randint(*self.profit_range) for p in products}
        machine_time = {(m,p): random.randint(*self.machine_time_range) 
                       for m in machines for p in products}
        sales_limits = {(t,p): random.randint(*self.sales_limit_range) 
                       for t in periods for p in products}
        
        # Generate maintenance schedule (randomly select periods for maintenance)
        maintenance = {(t,m): 1 if random.random() < 0.1 else 0 
                      for t in periods for m in machines}
        
        # Calculate target inventory levels
        target_inventory = {p: int(self.max_inventory * self.target_inventory_ratio) 
                          for p in products}
        available_machine_hours = {
            (t, m): self.machine_hours * (self.n_machines[m] - maintenance[t, m])
            for t in periods
            for m in machines
        }
        product_table = [
            {
                "product": p,
                "profit_per_unit_sold": profits[p],
                "initial_inventory": 0,
                "required_final_inventory": target_inventory[p],
                "max_inventory_each_period": self.max_inventory,
            }
            for p in products
        ]
        machine_product_processing_table = [
            {"machine": m, **{p: machine_time[m, p] for p in products}}
            for m in machines
        ]
        period_machine_capacity_table = [
            {
                "period": t,
                **{m: available_machine_hours[t, m] for m in machines},
            }
            for t in periods
        ]
        period_product_sales_limit_table = [
            {
                "period": t,
                **{p: sales_limits[t, p] for p in products},
            }
            for t in periods
        ]

        # Expose complete source data to downstream backtranslation and forward
        # modeling. Without this, later LLM stages only see an LP preview and may
        # incorrectly describe the instance as having no capacity or sales limits.
        self.parameters.update(
            {
                "n_periods": self.n_periods,
                "n_products": self.n_products,
                "periods": list(periods),
                "products": products,
                "machines": machines,
                "machine_counts": dict(self.n_machines),
                "profits": dict(profits),
                "holding_cost": self.holding_cost,
                "machine_time": {
                    m: {p: machine_time[m, p] for p in products}
                    for m in machines
                },
                "sales_limits": {
                    str(t): {p: sales_limits[t, p] for p in products}
                    for t in periods
                },
                "available_machine_hours": {
                    str(t): {m: available_machine_hours[t, m] for m in machines}
                    for t in periods
                },
                "compact_factory_planning_tables": {
                    "product_table": product_table,
                    "machine_product_processing_hours": machine_product_processing_table,
                    "period_machine_available_hours": period_machine_capacity_table,
                    "period_product_sales_upper_bounds": period_product_sales_limit_table,
                },
                "max_inventory_per_product_period": self.max_inventory,
                "target_inventory": dict(target_inventory),
                "initial_inventory": {p: 0 for p in products},
                "quantity_unit": "units",
                "capacity_unit": "machine hours",
                "money_unit": "dollars",
                "decision_variables": {
                    "Production[t,p]": "continuous units of product p manufactured in period t",
                    "Sales[t,p]": "continuous units of product p sold in period t",
                    "Inventory[t,p]": "continuous units of product p held at the end of period t",
                },
                "objective_terms": {
                    "sales_revenue": "sum_t,p profits[p] * Sales[t,p]",
                    "inventory_holding_cost": "sum_t,p holding_cost * Inventory[t,p]",
                    "sense": "maximize",
                },
                "required_constraints": [
                    "initial_inventory_balance",
                    "period_to_period_inventory_balance",
                    "machine_hour_capacity_after_maintenance",
                    "sales_upper_bound_by_period_product",
                    "inventory_upper_bound_by_period_product",
                    "required_final_inventory_target",
                    "nonnegative_production_sales_inventory",
                ],
                "required_parameter_presentation": [
                    "list products and periods",
                    "list profit per product",
                    "list holding cost",
                    "list machine types and installed counts",
                    "list processing hours a[m,p] for every machine-product pair",
                    "list available machine hours H[t,m] for every period-machine pair after maintenance",
                    "list sales upper bound U[t,p] for every period-product pair",
                    "state max inventory per product-period",
                    "state required final inventory target for every product",
                    "prefer compact_factory_planning_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "periods": list(periods),
                        "products": products,
                        "machines": machines,
                    },
                    "parameters": {
                        "compact_tables": "see top-level compact_factory_planning_tables",
                        "profit_per_unit_sold": "see compact_factory_planning_tables.product_table",
                        "holding_cost_per_unit_inventory": self.holding_cost,
                        "machine_time_hours_per_unit": "see compact_factory_planning_tables.machine_product_processing_hours",
                        "available_machine_hours_after_maintenance": "see compact_factory_planning_tables.period_machine_available_hours",
                        "sales_upper_bounds": "see compact_factory_planning_tables.period_product_sales_upper_bounds",
                        "max_inventory_per_product_period": self.max_inventory,
                        "required_final_inventory": "see compact_factory_planning_tables.product_table",
                        "initial_inventory": {p: 0 for p in products},
                    },
                    "constraints": {
                        "inventory_balance": "Inventory[t-1,p] + Production[t,p] = Sales[t,p] + Inventory[t,p], with initial inventory 0 in period 0",
                        "machine_capacity": "sum_p machine_time[m,p] * Production[t,p] <= available_machine_hours[t,m] for each period and machine",
                        "sales_limits": "Sales[t,p] <= sales_upper_bounds[t,p] for each period and product",
                        "inventory_limits": "Inventory[t,p] <= max_inventory_per_product_period",
                        "final_inventory_targets": "Inventory[last_period,p] = required_final_inventory[p]",
                    },
                    "objective": "maximize sales revenue minus inventory holding cost",
                },
                "business_interpretation_guardrails": [
                    "This is a multi-period factory production-sales-inventory planning model.",
                    "The business story must use product-period planning, not one-shot product mix.",
                    "Do not describe this instance as having no production capacity constraints.",
                    "Do not describe sales as unbounded; every product-period pair has a sales upper bound.",
                    "Do not aggregate sales limits into one total demand number per product.",
                    "Do not infer demand totals from the reference answer; use the listed product-period sales upper bounds.",
                    "Do not replace machine-hour capacity with budgets, workforce headcount, service coverage, or site capacity.",
                    "Do not omit the required final inventory target.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("FactoryPlanning")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        make = model.addVars(periods, products, name="Production", vtype=GRB.CONTINUOUS, lb=0)
        store = model.addVars(periods, products, name="Inventory", vtype=GRB.CONTINUOUS, lb=0)
        sell = model.addVars(periods, products, name="Sales", vtype=GRB.CONTINUOUS, lb=0)
        
        # Set objective: maximize profit minus holding costs
        model.setObjective(
            gp.quicksum(profits[p] * sell[t,p] for t in periods for p in products) -
            gp.quicksum(self.holding_cost * store[t,p] for t in periods for p in products),
            GRB.MAXIMIZE
        )
        
        # Add constraints
        # Initial balance constraints
        for p in products:
            model.addConstr(
                make[0,p] == sell[0,p] + store[0,p],
                name=f"InitialBalance_{p}"
            )
        
        # Balance constraints for remaining periods
        for t in periods[1:]:
            for p in products:
                model.addConstr(
                    store[t-1,p] + make[t,p] == sell[t,p] + store[t,p],
                    name=f"Balance_{t}_{p}"
                )
        
        # Machine capacity constraints
        for t in periods:
            for m in machines:
                model.addConstr(
                    gp.quicksum(machine_time[m,p] * make[t,p] for p in products) <= 
                    available_machine_hours[t,m],
                    name=f"Capacity_{t}_{m}"
                )
        
        # Inventory and sales limits
        for t in periods:
            for p in products:
                model.addConstr(store[t,p] <= self.max_inventory, name=f"Storage_{t}_{p}")
                model.addConstr(sell[t,p] <= sales_limits[t,p], name=f"Sales_{t}_{p}")
        
        # End-horizon inventory targets
        for p in products:
            model.addConstr(
                store[self.n_periods-1,p] == target_inventory[p],
                name=f"Target_{p}"
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
        
        model.write("factory_planning.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Profit: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
