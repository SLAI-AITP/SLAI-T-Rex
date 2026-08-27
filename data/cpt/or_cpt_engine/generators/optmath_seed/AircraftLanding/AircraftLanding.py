import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Aircraft Landing Problem optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_aircrafts: Number of self.aricrafts
            - time_window: Tuple of (min, max) for time window
            - penalty_range: Tuple of (min, max) for penalties
            - separation_range: Tuple of (min, max) for separation times
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "aircraft_landing"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Minimize total cost of early and late landings subject to:
        - Order constraints between self.aricrafts
        - Separation time requirements
        - Time window constraints
        - Early/Late landing calculations
        """

        default_parameters = {
            "n_aircrafts": (4, 6),
            "time_window": (10, 300),  # 3-hour window in minutes
            "penalty_range": (10, 100),
            "separation_range": (1, 5),  # minutes
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
        Generate an Aircraft Landing Problem instance and create its Gurobi model.
        Returns:
        gp.Model: Configured Gurobi model for the aircraft landing problem
        """
        # Generate random number of self.aricrafts
        self.n_aircrafts = _sample_int(self.n_aircrafts)
        self.aricrafts = [f"aircraft_{i}" for i in range(self.n_aircrafts)]
        self.aircraft = list(self.aricrafts)

        # Generate a congested arrival bank instead of fully independent target
        # times. This creates meaningful early/late tradeoffs and avoids many
        # zero-penalty schedules.
        min_sep, max_sep = self.separation_range
        horizon = max(1, self.time_window[1] - self.time_window[0])
        bank_width = max(3, min(horizon, int(max_sep * max(2, self.n_aircrafts) * 0.35)))
        latest_bank_start = max(self.time_window[0], self.time_window[1] - bank_width)
        bank_start = random.randint(self.time_window[0], latest_bank_start)
        self.target_landing = {
            i: random.randint(bank_start, min(self.time_window[1], bank_start + bank_width))
            for i in self.aricrafts
        }
        time_window_size = 20  # +/- 20 minutes around target time

        self.earliest_landing = {
            i: max(self.time_window[0], self.target_landing[i] - time_window_size)
            for i in self.aricrafts
        }
        self.latest_landing = {
            i: min(self.time_window[1], self.target_landing[i] + time_window_size)
            for i in self.aricrafts
        }

        self.penalty_before = {i: random.randint(*self.penalty_range) for i in self.aricrafts}
        self.penalty_after = {i: random.randint(*self.penalty_range) for i in self.aricrafts}

        # Generate separation times between self.aricrafts
        separation_time = {
            (i, j): random.randint(min_sep, max_sep)
            for i in self.aricrafts
            for j in self.aricrafts
            if i != j
        }
        big_M = horizon + max_sep + 2 * time_window_size
        compact_aircraft_landing_tables = {
            "sets": {
                "aircraft": self.aircraft,
                "ordered_aircraft_pairs": [
                    [i, j] for i in self.aricrafts for j in self.aricrafts if i != j
                ],
            },
            "aircraft_time_penalty_table": [
                {
                    "aircraft": i,
                    "earliest_landing": self.earliest_landing[i],
                    "target_landing": self.target_landing[i],
                    "latest_landing": self.latest_landing[i],
                    "early_penalty_per_minute": self.penalty_before[i],
                    "late_penalty_per_minute": self.penalty_after[i],
                }
                for i in self.aricrafts
            ],
            "ordered_pair_separation_matrix": {
                "columns": self.aricrafts,
                "rows": [
                    {
                        "aircraft_before": i,
                        "values": [
                            None if i == j else separation_time[i, j]
                            for j in self.aricrafts
                        ],
                    }
                    for i in self.aricrafts
                ],
            },
            "decision_variables": {
                "Landing[i]": "continuous actual landing time for aircraft i",
                "AircraftOrder[i,j]": "binary; 1 if aircraft i lands before aircraft j",
                "Early[i]": "continuous nonnegative minutes before the target time",
                "Late[i]": "continuous nonnegative minutes after the target time",
            },
            "objective": "minimize total early_landing_penalty * Early plus late_landing_penalty * Late",
            "constraints": [
                "for each unordered pair {i,j}, exactly one of AircraftOrder[i,j] and AircraftOrder[j,i] equals 1",
                "if i lands before j, Landing[j] must be at least Landing[i] plus separation_time[i,j]",
                "each Landing[i] must stay within earliest_landing[i] and latest_landing[i]",
                "Early[i] >= target_landing[i] - Landing[i] and Late[i] >= Landing[i] - target_landing[i]",
            ],
            "source_contract_note": (
                "This is runway landing-time sequencing. It is not aircraft type assignment, route allocation, "
                "tail routing, or a makespan-only scheduling model."
            ),
        }
        self.parameters.update(
            {
                "aircraft": self.aircraft,
                "compact_aircraft_landing_tables": compact_aircraft_landing_tables,
                "generation_note": (
                    "A sufficiently large big-M is used internally for conditional separation constraints, "
                    "but big-M is not a business parameter that should be listed in the natural-language problem."
                ),
                "decision_variables": {
                    "Landing[i]": "continuous actual landing time for aircraft i",
                    "AircraftOrder[i,j]": "binary, 1 if aircraft i lands before aircraft j",
                    "Early[i]": "continuous nonnegative minutes aircraft i lands before target",
                    "Late[i]": "continuous nonnegative minutes aircraft i lands after target",
                },
                "objective_terms": [
                    "early_landing_penalty_times_early_minutes",
                    "late_landing_penalty_times_late_minutes",
                ],
                "required_constraints": [
                    "one_precedence_order_for_each_aircraft_pair",
                    "minimum_separation_time_between_ordered_landings",
                    "aircraft_landing_time_window",
                    "early_deviation_lower_bound",
                    "late_deviation_lower_bound",
                ],
                "required_parameter_presentation": [
                    "list aircraft",
                    "list earliest, target, and latest landing time for every aircraft",
                    "list early and late penalty per minute for every aircraft",
                    "list required separation time for every ordered aircraft pair",
                    "state the objective minimizes early plus late landing penalty",
                    "state binary order variables decide which aircraft lands first for every pair",
                    "prefer compact_aircraft_landing_tables when writing the natural-language data tables",
                    "do not present big-M as a business parameter",
                ],
                "structured_problem_data": {
                    "sets": {
                        "aircraft": self.aircraft,
                    },
                    "parameters": {
                        "aircraft_time_penalty_table": "see compact_aircraft_landing_tables.aircraft_time_penalty_table",
                        "ordered_pair_separation_matrix": "see compact_aircraft_landing_tables.ordered_pair_separation_matrix",
                        "compact_tables": "see compact_aircraft_landing_tables",
                    },
                    "constraints": {
                        "pair_order": "for each unordered pair exactly one aircraft lands before the other",
                        "separation": "if i lands before j, landing[j] must be at least landing[i] plus separation_time[i,j]",
                        "time_windows": "each Landing[i] lies between earliest_landing[i] and latest_landing[i]",
                        "early_late_deviation": "Early[i] and Late[i] lower-bound deviations from target_landing[i]",
                    },
                    "objective": "minimize total early and late landing penalty",
                },
                "business_interpretation_guardrails": [
                    "This is a static aircraft landing sequencing model, not fleet assignment or aircraft routing.",
                    "Do not add aircraft type availability, route demand, passenger capacity, or operating cost matrices.",
                    "Do not omit pairwise landing separation constraints.",
                    "Do not omit earliest/target/latest landing times or early/late deviation penalties.",
                    "Do not change the objective to minimize completion time or makespan.",
                    "Do not expose big-M as an operational business input; it is only a linearization device.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("AircraftLanding")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Decision variables
        landing = model.addVars(self.aricrafts, vtype=GRB.CONTINUOUS, name="Landing")
        aircraft_order = model.addVars(
            [(i, j) for i in self.aricrafts for j in self.aricrafts if i != j],
            vtype=GRB.BINARY,
            name="AircraftOrder",
        )
        early = model.addVars(self.aricrafts, vtype=GRB.CONTINUOUS, name="Early")
        late = model.addVars(self.aricrafts, vtype=GRB.CONTINUOUS, name="Late")

        # Objective: minimize total penalty
        model.setObjective(
            gp.quicksum(
                self.penalty_before[i] * early[i] + self.penalty_after[i] * late[i]
                for i in self.aricrafts
            ),
            GRB.MINIMIZE,
        )

        # Constraints
        # Order constraints
        for left_index, i in enumerate(self.aricrafts):
            for j in self.aricrafts[left_index + 1:]:
                model.addConstr(aircraft_order[i, j] + aircraft_order[j, i] == 1, name=f"OrderPair_{i}_{j}")

        # Separation constraints
        for i in self.aricrafts:
            for j in self.aricrafts:
                if i != j:
                    model.addConstr(
                        landing[j]
                        >= landing[i]
                        + separation_time[i, j] * aircraft_order[i, j]
                        - big_M * aircraft_order[j, i],
                        name=f"Separation_{i}_{j}",
                    )

        # Time window constraints
        for i in self.aricrafts:
            model.addConstr(landing[i] >= self.earliest_landing[i], name=f"Earliest_{i}")
            model.addConstr(landing[i] <= self.latest_landing[i], name=f"Latest_{i}")

        # Early/Late constraints
        for i in self.aricrafts:
            model.addConstr(early[i] >= self.target_landing[i] - landing[i], name=f"EarlyDeviation_{i}")
            model.addConstr(late[i] >= landing[i] - self.target_landing[i], name=f"LateDeviation_{i}")

        return model
    
    def print_solution(self, model):
        """
        Print the solution of the Aircraft Landing Problem in a readable format.

        Parameters:
            model (gp.Model): Solved Gurobi model
        """
        if model.Status != GRB.OPTIMAL:
            print("No optimal solution found.")
            return

        print("\n=== Aircraft Landing Schedule ===")
        print(f"Total Cost: {model.ObjVal:.2f}")
        print("\nLanding Sequence:")
        print(f"{'Aircraft':^12} {'Landing Time':^15} {'Target Time':^15} {'Early':^10} {'Late':^10} {'Cost':^10}")
        print("-" * 75)

        # Get all landing times and sort aircraft by landing time
        landing_times = {}
        for v in model.getVars():
            if v.VarName.startswith('Landing'):
                aircraft = v.VarName.split('[')[1].split(']')[0]
                landing_times[aircraft] = v.X

        sorted_aircrafts = sorted(landing_times.keys(), key=lambda x: landing_times[x])

        for aircraft in sorted_aircrafts:
            # Get variable values
            actual_time = landing_times[aircraft]
            target_time = self.target_landing[aircraft]
            
            early_val = 0
            late_val = 0
            for v in model.getVars():
                if v.VarName == f'Early[{aircraft}]':
                    early_val = v.X
                elif v.VarName == f'Late[{aircraft}]':
                    late_val = v.X
            
            cost = (self.penalty_before[aircraft] * early_val + 
                    self.penalty_after[aircraft] * late_val)
            
            print(f"{aircraft:^12} {actual_time:^15.2f} {target_time:^15.2f} "
                    f"{early_val:^10.2f} {late_val:^10.2f} {cost:^10.2f}")

        print("\nSeparation Times:")
        print(f"{'Aircraft Pair':^25} {'Separation Time':^15}")
        print("-" * 40)

        for i in sorted_aircrafts:
            for j in sorted_aircrafts:
                if i != j and landing_times[j] > landing_times[i]:
                    sep_time = landing_times[j] - landing_times[i]
                    print(f"{i:>10} → {j:<10} {sep_time:^15.2f}")

        print("\nStatistics:")
        print(f"Number of aircraft: {len(sorted_aircrafts)}")
        print(f"Total schedule span: {landing_times[sorted_aircrafts[-1]] - landing_times[sorted_aircrafts[0]]:.2f} minutes")

        # Calculate average deviation
        total_deviation = 0
        for aircraft in sorted_aircrafts:
            for v in model.getVars():
                if v.VarName == f'Early[{aircraft}]':
                    total_deviation += v.X
                elif v.VarName == f'Late[{aircraft}]':
                    total_deviation += v.X

        avg_deviation = total_deviation / len(sorted_aircrafts)
        print(f"Average deviation from target: {avg_deviation:.2f} minutes")


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

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
            generator.print_solution(model)
        else:
            print("No optimal solution found")
        
        print(model.NumVars, model.NumConstrs)

    test_generator()
