import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).

        Parameters:
            parameters (dict): Dictionary containing:
                - n_customers: Number of customers
                - demand_range: Tuple of (min, max) for customer demands
                - time_window_range: Tuple of (min, max) for time windows
                - distance_range: Tuple of (min, max) for distances between customers
                - service_time_range: Tuple of (min, max) for service times
                - vehicle_capacity: Capacity of the single vehicle
                - M: A large constant for constraints
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "CVRPTW"
        self.mathematical_formulation = r"""
        ### Mathematical Model for a Single-Vehicle CVRPTW Seed

        Sets:
        - \(N = \{0, 1, \ldots, n\}\): depot plus customers
        - \(C = N \setminus \{0\}\): customers

        Parameters:
        - \(c_{i,j}\): travel distance/time from node \(i\) to node \(j\)
        - \(d_i\): demand at customer \(i\)
        - \(s_i\): service time at customer \(i\)
        - \([a_i,b_i]\): service start time window at customer \(i\)
        - \(Q\): vehicle capacity
        - \(M\): big-M constant

        Decision variables:
        - \(x_{i,j} \in \{0,1\}\): 1 if the vehicle travels directly from node \(i\) to node \(j\)
        - \(t_i\): service start time at customer \(i\)
        - \(u_i\): cumulative load after serving customer \(i\)

        Objective:
        \[
        \min \sum_{i \in N}\sum_{j \in N, j \ne i} c_{i,j}x_{i,j}
        \]

        Constraints:
        - exactly one arc enters and leaves each customer
        - exactly one route leaves and returns to the depot
        - time propagation for depot-to-customer and customer-to-customer arcs
        - customer time windows
        - load propagation and vehicle capacity

        Instances are generated from a latent feasible route, then time windows
        are derived from that route. Do not simplify this seed to pure
        transportation flow or omit route continuity/time-window logic.
        """
        default_parameters = {
            "n_customers": (5, 7),
            "coordinate_range": (0, 60),
            "demand_range": (2, 8),
            "service_time_range": (3, 8),
            "time_window_early_slack_range": (2, 10),
            "time_window_late_slack_range": (18, 35),
            "capacity_slack_range": (0.05, 0.20),
            "M": None
        }
        # Use default parameters if none are provided or if an empty dict is given
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
        Generate a CVRPTW problem instance and create its corresponding Gurobi model.

        Returns:
            gp.Model: Configured Gurobi model for the CVRPTW problem
        """
        # Randomly select number of customers and build a latent feasible route.
        self.n_customers = _sample_int(self.n_customers)

        nodes = [f"node_{i}" for i in range(self.n_customers + 1)]
        depot = nodes[0]
        customers = nodes[1:]
        coordinates = {
            node: (
                random.randint(*self.coordinate_range),
                random.randint(*self.coordinate_range),
            )
            for node in nodes
        }
        distances = {
            (i, j): _manhattan_distance(coordinates[i], coordinates[j])
            for i in nodes
            for j in nodes
            if i != j
        }
        demands = {customer: random.randint(*self.demand_range) for customer in customers}
        service_times = {customer: random.randint(*self.service_time_range) for customer in customers}
        latent_route = random.sample(customers, len(customers))
        total_demand = sum(demands.values())
        capacity_slack = random.uniform(*self.capacity_slack_range)
        self.vehicle_capacity = max(total_demand, int(round(total_demand * (1.0 + capacity_slack))))

        arrival_times: dict[str, int] = {}
        current_node = depot
        current_time = 0
        cumulative_load = 0
        latent_loads: dict[str, int] = {}
        for customer in latent_route:
            current_time += distances[current_node, customer]
            arrival_times[customer] = current_time
            cumulative_load += demands[customer]
            latent_loads[customer] = cumulative_load
            current_time += service_times[customer]
            current_node = customer

        lower_time_windows = {}
        upper_time_windows = {}
        for customer in customers:
            early_slack = random.randint(*self.time_window_early_slack_range)
            late_slack = random.randint(*self.time_window_late_slack_range)
            lower_time_windows[customer] = max(0, arrival_times[customer] - early_slack)
            upper_time_windows[customer] = arrival_times[customer] + late_slack

        max_time = max(upper_time_windows.values()) if upper_time_windows else 0
        max_distance = max(distances.values()) if distances else 0
        max_service = max(service_times.values()) if service_times else 0
        self.M = int(self.M or (max_time + max_distance + max_service + self.vehicle_capacity + 100))

        self.parameters.update(
            {
                "n_customers": self.n_customers,
                "nodes": nodes,
                "depot": depot,
                "customers": customers,
                "coordinates": {node: list(value) for node, value in coordinates.items()},
                "distance_matrix": {
                    i: {j: distances[i, j] for j in nodes if i != j}
                    for i in nodes
                },
                "demands": dict(demands),
                "service_times": dict(service_times),
                "time_windows": {
                    customer: [lower_time_windows[customer], upper_time_windows[customer]]
                    for customer in customers
                },
                "vehicle_capacity": self.vehicle_capacity,
                "vehicle_count": 1,
                "route_policy": "single vehicle starts at the depot, visits every customer exactly once, and returns to the depot",
                "big_m": self.M,
                "decision_variables": {
                    "ArcVisit[i,j]": "binary, 1 if the single vehicle travels directly from node i to node j",
                    "ServiceStartTime[i]": "continuous service start time at customer i",
                    "CumulativeLoad[i]": "continuous cumulative load after serving customer i",
                },
                "objective_terms": [
                    "travel_distance_times_arc_visit",
                ],
                "required_constraints": [
                    "each_customer_has_one_incoming_arc",
                    "each_customer_has_one_outgoing_arc",
                    "single_route_leaves_and_returns_to_depot",
                    "time_propagation_on_used_arcs",
                    "customer_time_windows",
                    "load_propagation_on_used_arcs",
                    "vehicle_capacity",
                    "no_self_arcs",
                ],
                "required_parameter_presentation": [
                    "state there is exactly one vehicle",
                    "list depot and customer nodes",
                    "list customer demand for every customer",
                    "list service time for every customer",
                    "list time window for every customer",
                    "list vehicle capacity",
                    "list complete directed distance or travel-time matrix",
                    "state that each customer must be visited exactly once",
                    "state that the route starts and ends at the depot",
                ],
                "structured_problem_data": {
                    "sets": {
                        "nodes": nodes,
                        "depot": depot,
                        "customers": customers,
                    },
                    "parameters": {
                        "vehicle_count": 1,
                        "vehicle_capacity": self.vehicle_capacity,
                        "customer_demands": demands,
                        "service_times": service_times,
                        "time_windows": {
                            customer: [lower_time_windows[customer], upper_time_windows[customer]]
                            for customer in customers
                        },
                        "directed_distance_matrix": {
                            i: {j: distances[i, j] for j in nodes if i != j}
                            for i in nodes
                        },
                        "big_m": self.M,
                    },
                    "constraints": {
                        "visit_once": "each customer has exactly one incoming and one outgoing arc",
                        "single_depot_route": "exactly one arc leaves the depot and exactly one arc returns to the depot",
                        "time_windows": "service start time at each customer must lie in its [early, late] window",
                        "time_propagation": "if arc i->j is used, service at j cannot start before service at i plus service time and travel time",
                        "load_propagation": "cumulative load increases by the demand of the next visited customer",
                        "vehicle_capacity": "cumulative load at every customer is at most the single vehicle capacity",
                    },
                    "objective": "minimize total directed travel distance of the single route",
                    "generation_note": "The instance was internally generated from a feasible route, but no route certificate should be included in the natural-language problem.",
                },
                "business_interpretation_guardrails": [
                    "This is a single-vehicle CVRPTW seed, not a fleet-sizing or multi-vehicle VRPTW seed.",
                    "Do not introduce multiple vehicles, vehicle indices, or route-assignment variables.",
                    "Do not simplify this instance to transportation flow.",
                    "Do not omit depot start/end route continuity.",
                    "Do not omit customer time windows.",
                    "Do not omit vehicle capacity and cumulative load logic.",
                    "Do not change the customer count or create extra depot nodes.",
                    "Do not reveal or constrain the model to any internally used feasible route certificate.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("CVRPTW")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create binary decision variables (x[i,j] = 1 if arc from i to j is in the route)
        x = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="ArcVisit")

        # Create continuous decision variables (t[i] = departure time at customer i)
        t = model.addVars(customers, vtype=GRB.CONTINUOUS, name="ServiceStartTime")

        # Create continuous decision variables (load after serving customer i)
        load = model.addVars(customers, vtype=GRB.CONTINUOUS, name="CumulativeLoad")

        # Set objective: minimize total distance
        model.setObjective(
            gp.quicksum(distances[i, j] * x[i, j] for i in nodes for j in nodes if i != j),
            GRB.MINIMIZE
        )

        # Add constraints
        # 1. No self arcs.
        for i in nodes:
            model.addConstr(x[i, i] == 0, name=f"NoSelfArc_{i}")

        # 2. Each customer is entered and left exactly once.
        for i in customers:
            model.addConstr(
                gp.quicksum(x[i, j] for j in nodes if i != j) == 1,
                name=f"DepartCustomer_{i}"
            )
            model.addConstr(
                gp.quicksum(x[j, i] for j in nodes if i != j) == 1,
                name=f"EnterCustomer_{i}"
            )

        # 3. A single route starts and ends at the depot.
        model.addConstr(gp.quicksum(x[depot, j] for j in customers) == 1, name="DepotDeparture")
        model.addConstr(gp.quicksum(x[i, depot] for i in customers) == 1, name="DepotReturn")

        # 4. Schedule feasibility. Depot-to-customer arcs use zero depot time.
        for j in customers:
            model.addConstr(
                distances[depot, j] - t[j] <= self.M * (1 - x[depot, j]),
                name=f"ScheduleFromDepot_{j}"
            )
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                model.addConstr(
                    t[i] + service_times[i] + distances[i, j] - t[j] <= self.M * (1 - x[i, j]),
                    name=f"ScheduleFeasibility_{i}_{j}"
                )

        # 5. Time window constraints
        for i in customers:
            model.addConstr(
                lower_time_windows[i] <= t[i],
                name=f"TimeWindowLower_{i}"
            )
            model.addConstr(
                t[i] <= upper_time_windows[i],
                name=f"TimeWindowUpper_{i}"
            )

        # 6. Load feasibility and capacity.
        for j in customers:
            model.addConstr(load[j] >= demands[j], name=f"LoadLowerBound_{j}")
            model.addConstr(load[j] <= self.vehicle_capacity, name=f"VehicleCapacity_{j}")
            model.addConstr(
                demands[j] - load[j] <= self.M * (1 - x[depot, j]),
                name=f"LoadFromDepot_{j}"
            )
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                model.addConstr(
                    load[i] + demands[j] - load[j] <= self.M * (1 - x[i, j]),
                    name=f"LoadFeasibility_{i}_{j}"
                )

        return model


def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


def _manhattan_distance(left, right):
    distance = abs(left[0] - right[0]) + abs(left[1] - right[1])
    return max(1, int(distance))


if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("cvrptw.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
