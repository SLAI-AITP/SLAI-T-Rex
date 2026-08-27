import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Uncapacitated Lot-Sizing with Backlogging (ULSB) problem.

        Parameters:
            parameters (dict): Dictionary containing:
                - n_periods: Number of periods
                - demand_range: Tuple of (min, max) for demand in each period
                - fixed_cost_range: Tuple of (min, max) for fixed ordering cost
                - unit_order_cost_range: Tuple of (min, max) for unit ordering cost
                - unit_holding_cost_range: Tuple of (min, max) for unit holding cost
                - unit_backlog_penalty_range: Tuple of (min, max) for unit backlogging penalty
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "uncapacitated_lot_sizing_backlogging"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are T periods, each period t (where t = 1, 2, ..., T) has the following attributes:
        - **Demand** D_t: Demand in period t.
        - **Fixed Ordering Cost** F_t: Fixed cost if an order is placed in period t.
        - **Unit Ordering Cost** c_t: Cost per unit ordered in period t.
        - **Unit Holding Cost** h_t: Cost per unit held in inventory at the end of period t.
        - **Unit Backlogging Penalty** p_t: Penalty per unit backlogged in period t.

        Decision Variables:
        - **x_t**: Ordered amount in period t (continuous).
        - **I_t**: Ending inventory in period t (continuous).
        - **y_t**: Binary variable indicating whether an order is placed in period t (1 if placed, 0 otherwise).
        - **B_t**: Backlogged amount in period t (continuous).

        Objective:
        Minimize the total cost:
        $$
        \text{Minimize} \quad Z = \sum_{t=1}^T \left( F_t y_t + c_t x_t + h_t I_t + p_t B_t \right)
        $$

        Constraints:
        1. **Net Inventory Flow Balance**:
           $$
           I_{t-1} + x_t - B_{t-1} = I_t + D_t - B_t \quad \forall t \in T
           $$
           with constants I_0 = 0 and B_0 = 0.
        2. **Ordered Upper Bound**:
           $$
           x_t \leq y_t \cdot \sum_{i=1}^T D_i \quad \forall t \in T
           $$
        3. **Final Inventory Requirement**:
           $$
           I_T = 0
           $$
        4. **Final Backlog Requirement**:
           $$
           B_T = 0
           $$
        5. **Domains**:
           $$
           y_t \in \{0,1\}, \quad x_t, I_t, B_t \geq 0
           $$
        """
        default_parameters = {
            "n_periods": (4, 6),
            "demand_range": (10, 80),
            "fixed_cost_range": (50, 200),
            "unit_order_cost_range": (1, 10),
            "unit_holding_cost_range": (1, 5),
            "unit_backlog_penalty_range": (5, 20),
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
        Generate a ULSB problem instance and create its corresponding Gurobi model.

        This method does two things:
        1. Generates random problem data (demands, costs, penalties)
        2. Creates and returns a configured Gurobi model ready to solve

        Returns:
            gp.Model: Configured Gurobi model for the ULSB problem
        """
        # Randomly select number of periods
        self.n_periods = _sample_int(self.n_periods)

        # Generate periods
        periods = [f"period_{t}" for t in range(1, self.n_periods + 1)]

        # Generate random data
        demands = {t: random.randint(*self.demand_range) for t in periods}
        fixed_costs = {t: random.randint(*self.fixed_cost_range) for t in periods}
        unit_order_costs = {t: random.randint(*self.unit_order_cost_range) for t in periods}
        unit_holding_costs = {t: random.randint(*self.unit_holding_cost_range) for t in periods}
        unit_backlog_penalties = {t: random.randint(*self.unit_backlog_penalty_range) for t in periods}

        # Create Gurobi model
        model = gp.Model("UncapacitatedLotSizingBacklogging")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Decision variables
        x = model.addVars(periods, vtype=GRB.CONTINUOUS, name="OrderedAmount")
        I = model.addVars(periods, vtype=GRB.CONTINUOUS, name="EndingInventory")
        y = model.addVars(periods, vtype=GRB.BINARY, name="OrderIsPlaced")
        B = model.addVars(periods, vtype=GRB.CONTINUOUS, name="BackloggedAmount")

        # Objective: Minimize total cost
        model.setObjective(
            gp.quicksum(
                fixed_costs[t] * y[t]
                + unit_order_costs[t] * x[t]
                + unit_holding_costs[t] * I[t]
                + unit_backlog_penalties[t] * B[t]
                for t in periods
            ),
            GRB.MINIMIZE,
        )

        # Constraints
        # Net-inventory flow balance:
        # I_t - B_t = I_{t-1} - B_{t-1} + x_t - D_t.
        # Initial inventory and backlog are fixed constants equal to zero.
        for index, t in enumerate(periods):
            if index == 0:
                model.addConstr(
                    I[t] - B[t] == x[t] - demands[t],
                    name=f"FlowBalance_{t}",
                )
            else:
                prev_t = periods[index - 1]
                model.addConstr(
                    I[t] - B[t] == I[prev_t] - B[prev_t] + x[t] - demands[t],
                    name=f"FlowBalance_{t}",
                )

        # Ordered upper bound constraint
        for t in periods:
            model.addConstr(
                x[t] <= y[t] * sum(demands.values()),
                name=f"OrderedUpperBound_{t}",
            )

        # End the horizon cleanly: all demand is eventually satisfied and no
        # leftover inventory is carried beyond the planning window.
        model.addConstr(I[f"period_{self.n_periods}"] == 0, name="EndingInventoryZero")
        model.addConstr(B[f"period_{self.n_periods}"] == 0, name="EndingBacklogZero")

        self.demands = demands
        self.fixed_costs = fixed_costs
        self.unit_order_costs = unit_order_costs
        self.unit_holding_costs = unit_holding_costs
        self.unit_backlog_penalties = unit_backlog_penalties
        total_demand = sum(demands.values())
        period_cost_table = [
            {
                "period": t,
                "demand": demands[t],
                "fixed_ordering_cost": fixed_costs[t],
                "unit_order_cost": unit_order_costs[t],
                "unit_holding_cost": unit_holding_costs[t],
                "unit_backlog_penalty": unit_backlog_penalties[t],
            }
            for t in periods
        ]
        balance_transition_by_period = []
        for index, t in enumerate(periods):
            if index == 0:
                previous_state = "initial_inventory - initial_backlog = 0"
                equation = (
                    f"EndingInventory[{t}] - BackloggedAmount[{t}] = "
                    f"OrderedAmount[{t}] - demand[{t}]"
                )
            else:
                prev_t = periods[index - 1]
                previous_state = f"EndingInventory[{prev_t}] - BackloggedAmount[{prev_t}]"
                equation = (
                    f"EndingInventory[{t}] - BackloggedAmount[{t}] = "
                    f"EndingInventory[{prev_t}] - BackloggedAmount[{prev_t}] + "
                    f"OrderedAmount[{t}] - demand[{t}]"
                )
            balance_transition_by_period.append(
                {
                    "period": t,
                    "previous_net_inventory_state": previous_state,
                    "demand": demands[t],
                    "equation": equation,
                }
            )
        self.parameters.update(
            {
                "periods": periods,
                "period_cost_table": period_cost_table,
                "compact_lotsizing_tables": {
                    "model_family": "uncapacitated_lot_sizing_with_backlogging",
                    "periods": periods,
                    "period_cost_table": period_cost_table,
                    "total_demand_big_m": total_demand,
                    "initial_inventory": 0,
                    "initial_backlog": 0,
                    "required_final_inventory": 0,
                    "required_final_backlog": 0,
                    "balance_transition_by_period": balance_transition_by_period,
                    "setup_linking_by_period": {
                        t: f"OrderedAmount[{t}] <= {total_demand} * OrderIsPlaced[{t}]"
                        for t in periods
                    },
                    "source_model_note": (
                        "Demand values are period-by-period demand, not cumulative demand. "
                        "Net inventory is EndingInventory[t] - BackloggedAmount[t]. "
                        "There is no production capacity constraint."
                    ),
                },
                "demands": demands,
                "fixed_ordering_costs": fixed_costs,
                "unit_order_costs": unit_order_costs,
                "unit_holding_costs": unit_holding_costs,
                "unit_backlog_penalties": unit_backlog_penalties,
                "initial_inventory": 0,
                "initial_backlog": 0,
                "required_final_inventory": 0,
                "required_final_backlog": 0,
                "big_m_order_upper_bound": total_demand,
                "balance_transition_by_period": balance_transition_by_period,
                "setup_linking_by_period": {
                    t: f"OrderedAmount[{t}] <= {total_demand} * OrderIsPlaced[{t}]"
                    for t in periods
                },
                "decision_variables": {
                    "OrderedAmount[t]": "continuous nonnegative quantity ordered in period t",
                    "EndingInventory[t]": "continuous nonnegative inventory held after period t",
                    "BackloggedAmount[t]": "continuous nonnegative demand backlog carried after period t",
                    "OrderIsPlaced[t]": "binary setup variable equal to 1 if any order is placed in period t",
                },
                "objective_terms": [
                    "fixed_ordering_cost_if_order_is_placed",
                    "unit_order_cost",
                    "ending_inventory_holding_cost",
                    "ending_backlog_penalty",
                ],
                "required_constraints": [
                    "net_inventory_balance_with_backlog_carryover",
                    "order_quantity_linked_to_binary_setup",
                    "zero_initial_inventory_and_backlog_as_balance_constants",
                    "zero_final_inventory",
                    "zero_final_backlog",
                    "binary_order_setup_variables",
                    "nonnegative_order_inventory_backlog_quantities",
                ],
                "required_parameter_presentation": [
                    "list periods in order",
                    "list demand, fixed ordering cost, unit order cost, holding cost, and backlog penalty for every period",
                    "state initial inventory and initial backlog are both zero constants",
                    "state final inventory and final backlog must both be zero",
                    "state net inventory is ending inventory minus backlog",
                    "state OrderedAmount[t] is linked to OrderIsPlaced[t] by total-demand big-M",
                    "state there is no production capacity constraint",
                ],
                "structured_problem_data": {
                    "sets": {
                        "periods": periods,
                    },
                    "parameters": {
                        "period_cost_table": period_cost_table,
                        "initial_inventory": 0,
                        "initial_backlog": 0,
                        "required_final_inventory": 0,
                        "required_final_backlog": 0,
                        "big_m_order_upper_bound": total_demand,
                    },
                    "constraints": {
                        "net_inventory_balance": "EndingInventory[t] - BackloggedAmount[t] equals previous net inventory plus ordered amount minus demand",
                        "setup_linking": "OrderedAmount[t] <= total_demand * OrderIsPlaced[t]",
                        "final_state": "ending inventory and ending backlog are both zero in the final period",
                    },
                    "objective": "minimize fixed setup cost, unit order cost, holding cost, and backlog penalty",
                },
                "business_interpretation_guardrails": [
                    "This is uncapacitated lot sizing with backlogging; there is no production capacity constraint.",
                    "Do not add a per-period capacity limit unless it is explicitly listed in the source data.",
                    "Do not add capacity_if_present; this source data has no capacity table.",
                    "The setup binary only activates ordering through the total-demand big-M upper bound.",
                    "Backlog is a carried state variable with a per-period penalty, not lost sales.",
                    "Do not replace backlog with unmet-demand slack that disappears at the end of each period.",
                ],
            }
        )

        return model


def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


if __name__ == "__main__":
    import time

    def test_generator():
        generator = Generator()
        model = generator.generate_instance()

        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time

        model.write("ulsb.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
