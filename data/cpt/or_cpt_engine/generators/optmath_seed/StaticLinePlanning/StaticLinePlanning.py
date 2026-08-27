import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import round_business_float, uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Static Line Planning optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_nodes: Number of nodes
            - n_lines: Number of lines
            - n_od_pairs: Number of OD pairs
            - fixed_cost_range: Tuple of (min, max) for fixed costs
            - operational_cost_range: Tuple of (min, max) for operational costs
            - capacity_range: Tuple of (min, max) for capacity
            - demand_range: Tuple of (min, max) for demand
            - penalty_range: Tuple of (min, max) for penalty
            - trip_time_range: Tuple of (min, max) for trip time
            - density: Network density (between 0 and 1)
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "static_line_planning"
        self.mathematical_formulation = r"""
        \begin{align*}
        \min & \sum_{l \in L} (fc_l x_l + oc_l f_l) + \sum_{c \in C} p_c s_c \\
        \text{s.t.} & \sum_{l \in L} cap_l service_{lc} f_l + s_c \geq d_c & \forall c \in C \\
        & f_l \leq max\_freq_l x_l & \forall l \in L \\
        & f_l \geq min\_freq_l x_l & \forall l \in L \\
        & \sum_{l \in L} tt_l f_l \leq total\_vehicle\_hours \\
        & \sum_{l \in L} x_l pass_{ln} \geq 2r_n & \forall n \in N
        \end{align*}
        """
        
        default_parameters = {
            "n_nodes": (5, 8),
            "n_lines": (6, 10),
            "n_od_pairs": (6, 12),
            "fixed_cost_range": (1000, 5000),
            "operational_cost_range": (100, 500),
            "capacity_range": (25, 55),
            "demand_range": (60, 160),
            "penalty_range": (8000, 20000),
            "trip_time_range": (15, 55),
            "target_line_activation_ratio": (0.55, 0.85),
            "service_lines_per_od": (2, 4),
            "transfer_requirement_ratio": (0.25, 0.45),
        }

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
        Generate a Static Line Planning problem instance and create its corresponding Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the static line planning problem
        """
        
        # Randomly select number of nodes, lines, and OD pairs
        self.n_nodes = random.randint(*self.n_nodes)
        self.n_lines = random.randint(*self.n_lines)
        self.n_od_pairs = random.randint(*self.n_od_pairs)
    
        # Generate nodes
        nodes = list(range(self.n_nodes))

        # Generate lines and commodities
        lines = [f"L_{i}" for i in range(self.n_lines)]
        commodities = set()
        while len(commodities) < min(self.n_od_pairs, self.n_nodes * (self.n_nodes-1)):
            origin = random.choice(nodes)
            dest = random.choice([n for n in nodes if n != origin])
            commodities.add(f"OD_{origin}_{dest}")
        commodities = list(commodities)

        # Generate parameters. The default generator used a vehicle budget that
        # was far too small compared with trip_time * frequency, which made the
        # all-reserve solution artificially attractive. We size the budget from
        # feasible line frequencies so selected lines can realistically serve
        # demand while still creating a fleet trade-off.
        line_info = {l: {
            "fixed_cost": random.randint(*self.fixed_cost_range),
            "operational_cost": random.randint(*self.operational_cost_range),
            "capacity": random.randint(*self.capacity_range),
            "trip_time": random.randint(*self.trip_time_range),
            "min_freq": 1,
            "max_freq": 10
        } for l in lines}

        activation_ratio = uniform_rounded(*self.target_line_activation_ratio)
        target_active_lines = max(1, min(self.n_lines, int(round(self.n_lines * activation_ratio))))
        target_active_set = set(random.sample(lines, target_active_lines))
        planned_frequency = {
            l: random.randint(line_info[l]["min_freq"] + 2, line_info[l]["max_freq"] - 1)
            for l in target_active_set
        }

        # Generate service and pass-through matrices. Every OD pair is served by
        # at least one planned active line so a zero-shortage solution is present
        # in the generated instance. This prevents the generator from producing
        # mathematically valid but training-poor reserve-only models.
        service_matrix = {f"{l}_{c}": 0 for l in lines for c in commodities}
        min_service, max_service = self.service_lines_per_od
        od_service_table = []
        for c in commodities:
            service_count = min(
                self.n_lines,
                random.randint(min(min_service, self.n_lines), min(max_service, self.n_lines)),
            )
            serving_lines = set(random.sample(lines, service_count))
            serving_lines.add(random.choice(sorted(target_active_set)))
            for l in sorted(serving_lines):
                service_matrix[f"{l}_{c}"] = 1
            od_service_table.append({"od_pair": c, "serving_lines": sorted(serving_lines)})
        for l in lines:
            if not any(service_matrix[f"{l}_{c}"] for c in commodities):
                c = random.choice(commodities)
                service_matrix[f"{l}_{c}"] = 1

        # Demand is sized from the planned active service envelope, rather than
        # independently sampled, so the instance can be fully served within the
        # generated fleet budget while still requiring nontrivial line choices.
        od_info = {}
        for service_row in od_service_table:
            c = service_row["od_pair"]
            active_serving = [l for l in service_row["serving_lines"] if l in target_active_set]
            minimum_capacity = sum(line_info[l]["capacity"] * line_info[l]["min_freq"] for l in active_serving)
            planned_capacity = sum(line_info[l]["capacity"] * planned_frequency[l] for l in active_serving)
            lower, upper = self.demand_range
            demand_lower = max(lower, int(round(minimum_capacity * uniform_rounded(1.05, 1.25))))
            demand_upper = min(upper, int(round(planned_capacity * uniform_rounded(0.65, 0.9))))
            if demand_lower <= demand_upper:
                demand = random.randint(demand_lower, demand_upper)
            else:
                demand = max(lower, min(upper, int(round(planned_capacity * uniform_rounded(0.55, 0.75)))))
            od_info[c] = {
                "demand": demand,
                "penalty": random.randint(*self.penalty_range),
            }

        line_pass_nodes = {}
        for l in lines:
            pass_count = random.randint(2, min(self.n_nodes, 4))
            line_pass_nodes[l] = set(random.sample(nodes, pass_count))
        pass_through = {f"{l}_{n}": 1 if n in line_pass_nodes[l] else 0 for l in lines for n in nodes}
        for l in lines:
            if not any(pass_through[f"{l}_{n}"] for n in nodes):
                chosen_node = random.choice(nodes)
                pass_through[f"{l}_{chosen_node}"] = 1
                line_pass_nodes[l].add(chosen_node)

        transfer_required = {n: 0 for n in nodes}
        required_count = 0
        if len(target_active_set) >= 2:
            required_count = max(1, int(round(self.n_nodes * uniform_rounded(*self.transfer_requirement_ratio))))
        required_nodes = random.sample(nodes, min(required_count, self.n_nodes))
        for n in required_nodes:
            transfer_required[n] = 1
            passing_active = [l for l in target_active_set if pass_through[f"{l}_{n}"] == 1]
            if len(passing_active) < 2:
                missing_active = [l for l in sorted(target_active_set) if pass_through[f"{l}_{n}"] == 0]
                for l in random.sample(missing_active, 2 - len(passing_active)):
                    pass_through[f"{l}_{n}"] = 1
                    line_pass_nodes[l].add(n)

        planned_vehicle_use = sum(line_info[l]["trip_time"] * planned_frequency[l] for l in target_active_set)
        total_vehicle_hours = round_business_float(
            planned_vehicle_use * uniform_rounded(1.02, 1.15)
        )

        line_table = [
            {
                "line": l,
                "fixed_cost": line_info[l]["fixed_cost"],
                "operating_cost_per_frequency": line_info[l]["operational_cost"],
                "capacity_per_frequency": line_info[l]["capacity"],
                "vehicle_hours_per_frequency": line_info[l]["trip_time"],
                "minimum_frequency_if_selected": line_info[l]["min_freq"],
                "maximum_frequency_if_selected": line_info[l]["max_freq"],
            }
            for l in lines
        ]
        od_table = [
            {
                "od_pair": c,
                "origin": int(c.split("_")[1]),
                "destination": int(c.split("_")[2]),
                "demand": od_info[c]["demand"],
                "unmet_demand_penalty": od_info[c]["penalty"],
            }
            for c in commodities
        ]
        line_pass_table = [
            {"line": l, "passes_nodes": sorted(line_pass_nodes[l])}
            for l in lines
        ]
        node_transfer_table = [
            {
                "node": n,
                "transfer_required": transfer_required[n],
                "lines_passing": [l for l in lines if pass_through[f"{l}_{n}"] == 1],
            }
            for n in nodes
        ]
        self.parameters.update(
            {
                "nodes": nodes,
                "lines": lines,
                "commodities": commodities,
                "required_transfer_nodes": [n for n in nodes if transfer_required[n] == 1],
                "total_vehicle_hours": total_vehicle_hours,
                "line_table": line_table,
                "od_table": od_table,
                "od_service_table": od_service_table,
                "line_pass_table": line_pass_table,
                "node_transfer_table": node_transfer_table,
                "structured_problem_data": {
                    "schema": "static_line_frequency_planning",
                    "complete_tables": [
                        "line_table",
                        "od_table",
                        "od_service_table",
                        "line_pass_table",
                        "node_transfer_table",
                    ],
                    "compact_matrix_policy": "Use od_service_table and line_pass_table instead of dense binary matrices.",
                    "total_vehicle_hours": total_vehicle_hours,
                },
                "required_parameter_presentation": [
                    "List every candidate line with fixed cost, operating cost per frequency, capacity per frequency, vehicle-hours per frequency, min frequency, and max frequency.",
                    "List every OD pair with origin, destination, demand, unmet-demand penalty, and the candidate lines that can serve it.",
                    "List every transfer-required node and the lines that pass through it.",
                    "State x[l] is binary line activation, f[l] is continuous nonnegative service frequency, and s[c] is continuous unmet demand.",
                    "State total_vehicle_hours is a fleet budget on sum(vehicle_hours_per_frequency[l] * f[l]).",
                    "State transfer requirements are fixed parameters r[n], not decision variables.",
                ],
                "business_interpretation_guardrails": [
                    "Do not introduce passenger flow, commodity flow, route sequencing, subtour, time-window, or arc-continuity variables.",
                    "Do not invent z[i,j,l] arc-use variables; the source instance already gives candidate lines and coverage indicators.",
                    "Do not make f[l] integer unless the source explicitly says frequencies must be integer.",
                    "Do not omit unmet-demand variables s[c] or their penalty terms.",
                    "Do not omit the line activation-frequency linking constraints f[l] <= max_freq[l] x[l] and f[l] >= min_freq[l] x[l].",
                    "Do not replace the fixed transfer requirement parameter r[n] with an optional transfer-station decision.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("StaticLinePlanning")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create variables
        x = model.addVars(lines, vtype=GRB.BINARY, name="x")
        f = model.addVars(lines, vtype=GRB.CONTINUOUS, name="f")
        s = model.addVars(commodities, vtype=GRB.CONTINUOUS, name="s")

        # Set objective
        obj = (gp.quicksum(line_info[l]["fixed_cost"] * x[l] + 
                          line_info[l]["operational_cost"] * f[l] for l in lines) +
               gp.quicksum(od_info[c]["penalty"] * s[c] for c in commodities))
        model.setObjective(obj, GRB.MINIMIZE)

        # Add constraints
        for c in commodities:
            model.addConstr(
                gp.quicksum(line_info[l]["capacity"] * service_matrix[f"{l}_{c}"] * f[l]
                           for l in lines) + s[c] >= od_info[c]["demand"]
                , name=f"Demand_{c}"
            )

        for l in lines:
            model.addConstr(f[l] <= line_info[l]["max_freq"] * x[l], name=f"MaxFreq_{l}")
            model.addConstr(f[l] >= line_info[l]["min_freq"] * x[l], name=f"MinFreq_{l}")

        model.addConstr(
            gp.quicksum(line_info[l]["trip_time"] * f[l] for l in lines) <= 
            total_vehicle_hours,
            name="FleetCapacity"
        )

        for n in nodes:
            model.addConstr(
                gp.quicksum(x[l] * pass_through[f"{l}_{n}"] for l in lines) >= 2 * transfer_required[n],
                name=f"Transfer_{n}"
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

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
        
        print(model.NumVars, model.NumConstrs)

    test_generator()
