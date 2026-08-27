import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Uncapacitated Lot-Sizing (ULS) optimization problem.

        Parameters:
            parameters (dict): Dictionary containing:
                - n_periods: Number of periods
                - demand_range: Tuple of (min, max) for demand in each period
                - fixed_cost_range: Tuple of (min, max) for fixed ordering cost in each period
                - unit_order_cost_range: Tuple of (min, max) for unit ordering cost in each period
                - unit_holding_cost_range: Tuple of (min, max) for unit holding cost in each period
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "uncapacitated_lot_sizing"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are T periods, each period t (where t = 1, 2, ..., T) has the following attributes:
        - **Demand** D_t: Demand in period t.
        - **Fixed Cost** F_t: Fixed ordering cost in period t.
        - **Unit Order Cost** c_t: Unit ordering cost in period t.
        - **Unit Holding Cost** h_t: Unit holding cost in period t.
        - **Ordered Amount** x_t: Continuous variable representing the amount ordered in period t.
        - **Ending Inventory** I_t: Continuous variable representing the ending inventory in period t.
        - **Order Indicator** y_t: Binary variable indicating whether an order is placed in period t.

        Objective:
        Minimize the total cost, which includes fixed ordering costs, unit ordering costs, and holding costs:
        $$
        \text{Minimize } Z = \sum_{t=1}^T \left( F_t y_t + c_t x_t + h_t I_t \right)
        $$

        Constraints:
        1. Flow Balance: Ending inventory in period t is equal to the ending inventory from the previous period plus the ordered amount minus the demand in period t:
        $$
        I_{t-1} + x_t = I_t + D_t \quad \forall t \in T
        $$
        2. Ordered Amount Upper Bound: The ordered amount in period t is zero if no order is placed (y_t = 0), and otherwise, it is bounded by the total demand over all periods:
        $$
        x_t \leq y_t \cdot \sum_{i=1}^T D_i \quad \forall t \in T
        $$
        3. Stock Loss of Generality: Starting and ending inventories are zero:
        $$
        I_0 = 0, \quad I_T = 0
        $$
        """
        default_parameters = {
            "n_periods": (3, 5),
            "demand_range": (1, 10),
            "fixed_cost_range": (10, 50),
            "unit_order_cost_range": (1, 5),
            "unit_holding_cost_range": (1, 3),
        }
        # Use default parameters if none are provided or if an empty dict is given
        if parameters is None or not parameters:
            parameters = default_parameters
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)

        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate an Uncapacitated Lot-Sizing problem instance and create its corresponding Gurobi model.

        This method does two things:
        1. Generates random problem data (demands, costs, etc.)
        2. Creates and returns a configured Gurobi model ready to solve

        Returns:
            gp.Model: Configured Gurobi model for the ULS problem
        """
        # Randomly select number of periods
        self.n_periods = random.randint(*self.n_periods)

        # Generate periods and their properties
        periods = [f"period_{t}" for t in range(1, self.n_periods + 1)]
        demands = {t: random.randint(*self.demand_range) for t in periods}
        fixed_costs = {t: random.randint(*self.fixed_cost_range) for t in periods}
        unit_order_costs = {t: random.randint(*self.unit_order_cost_range) for t in periods}
        unit_holding_costs = {t: random.randint(*self.unit_holding_cost_range) for t in periods}
        total_demand = sum(demands.values())
        period_cost_table = [
            {
                "period": t,
                "demand": demands[t],
                "fixed_ordering_cost": fixed_costs[t],
                "unit_order_cost": unit_order_costs[t],
                "unit_holding_cost": unit_holding_costs[t],
                "setup_big_m_remaining_demand": total_demand,
            }
            for t in periods
        ]
        balance_transition_by_period = []
        for index, t in enumerate(periods):
            if index == 0:
                previous_state = "initial_inventory = 0"
                equation = f"OrderedAmount[{t}] = demand[{t}] + EndingInventory[{t}]"
            else:
                prev_t = periods[index - 1]
                previous_state = f"EndingInventory[{prev_t}]"
                equation = (
                    f"EndingInventory[{prev_t}] + OrderedAmount[{t}] = "
                    f"demand[{t}] + EndingInventory[{t}]"
                )
            balance_transition_by_period.append(
                {
                    "period": t,
                    "previous_inventory_state": previous_state,
                    "demand": demands[t],
                    "equation": equation,
                }
            )
        self.parameters.update(
            {
                "periods": periods,
                "period_cost_table": period_cost_table,
                "compact_lotsizing_tables": {
                    "model_family": "uncapacitated_lot_sizing_without_backlog",
                    "periods": periods,
                    "period_cost_table": period_cost_table,
                    "total_demand_big_m": total_demand,
                    "initial_inventory": 0,
                    "required_final_inventory": 0,
                    "balance_transition_by_period": balance_transition_by_period,
                    "setup_linking_by_period": {
                        t: f"OrderedAmount[{t}] <= {total_demand} * OrderIsPlaced[{t}]"
                        for t in periods
                    },
                    "source_model_note": (
                        "This is uncapacitated lot sizing without backlog or lost sales. "
                        "Inventory is nonnegative ending inventory. The only horizon-end hard constraint is "
                        "EndingInventory[last_period] == 0. There is no production capacity constraint."
                    ),
                },
                "demands": demands,
                "fixed_ordering_costs": fixed_costs,
                "unit_order_costs": unit_order_costs,
                "unit_holding_costs": unit_holding_costs,
                "initial_inventory": 0,
                "required_final_inventory": 0,
                "big_m_order_upper_bound": total_demand,
                "balance_transition_by_period": balance_transition_by_period,
                "setup_linking_by_period": {
                    t: f"OrderedAmount[{t}] <= {total_demand} * OrderIsPlaced[{t}]"
                    for t in periods
                },
                "decision_variables": {
                    "OrderedAmount[t]": "continuous nonnegative quantity ordered in period t",
                    "EndingInventory[t]": "continuous nonnegative inventory held after period t",
                    "OrderIsPlaced[t]": "binary setup variable equal to 1 if any order is placed in period t",
                },
                "objective_terms": [
                    "fixed_ordering_cost_if_order_is_placed",
                    "unit_order_cost",
                    "ending_inventory_holding_cost",
                ],
                "required_constraints": [
                    "inventory_balance_without_backlog",
                    "order_quantity_linked_to_binary_setup",
                    "zero_initial_inventory_as_balance_constant",
                    "zero_final_inventory",
                    "binary_order_setup_variables",
                    "nonnegative_order_and_inventory_quantities",
                ],
                "required_parameter_presentation": [
                    "list periods in order",
                    "list demand, fixed ordering cost, unit order cost, and holding cost for every period",
                    "state initial inventory is zero as the first balance constant",
                    "state final inventory must be zero",
                    "state there are no backlog, lost-sales, or capacity variables in the source model",
                    "state OrderedAmount[t] is linked to OrderIsPlaced[t] by total-demand big-M",
                    "prefer compact_lotsizing_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "periods": periods,
                    },
                    "parameters": {
                        "period_cost_table": period_cost_table,
                        "initial_inventory": 0,
                        "required_final_inventory": 0,
                        "big_m_order_upper_bound": total_demand,
                    },
                    "constraints": {
                        "inventory_balance": "previous inventory plus OrderedAmount[t] equals demand[t] plus EndingInventory[t]",
                        "setup_linking": "OrderedAmount[t] <= total_demand * OrderIsPlaced[t]",
                        "final_inventory": "EndingInventory[last_period] == 0",
                    },
                    "objective": "minimize fixed ordering, unit ordering, and holding cost",
                },
                "business_interpretation_guardrails": [
                    "This is uncapacitated lot sizing without backlog.",
                    "Do not introduce backlog variables, backlog penalties, lost sales, unmet-demand slack, or production capacity constraints.",
                    "Do not add a separate zero final backlog constraint.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("UncapacitatedLotSizing")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create decision variables
        x = model.addVars(periods, vtype=GRB.CONTINUOUS, name="OrderedAmount")
        I = model.addVars(periods, vtype=GRB.CONTINUOUS, name="EndingInventory")
        y = model.addVars(periods, vtype=GRB.BINARY, name="OrderIsPlaced")

        # Set objective: minimize total cost
        model.setObjective(
            gp.quicksum(
                fixed_costs[t] * y[t] + unit_order_costs[t] * x[t] + unit_holding_costs[t] * I[t]
                for t in periods
            ),
            GRB.MINIMIZE,
        )

        # Add constraints
        for t in periods:
            # Flow balance constraint
            if t == "period_1":
                model.addConstr(x[t] == I[t] + demands[t], name=f"FlowBalance_{t}")
            else:
                prev_t = f"period_{int(t.split('_')[1]) - 1}"
                model.addConstr(I[prev_t] + x[t] == I[t] + demands[t], name=f"FlowBalance_{t}")

            # Ordered amount upper bound
            model.addConstr(
                x[t] <= y[t] * total_demand, name=f"OrderedUpperBound_{t}"
            )

        # The first-period flow balance already assumes zero starting
        # inventory. Do not force I_period_1 to zero; that suppresses normal
        # carryover decisions and makes many instances structurally repetitive.
        model.addConstr(I[f"period_{self.n_periods}"] == 0, name="EndingInventory")

        return model


if __name__ == "__main__":
    import time

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()

        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        model.write("uncapacitated_lot_sizing.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
