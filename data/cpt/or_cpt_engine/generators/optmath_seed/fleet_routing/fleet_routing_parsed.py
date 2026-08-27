import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Fleet Routing Problem optimization.
        Parameters:
            parameters (dict): Dictionary containing problem setup details:
                - n_locations: Number of locations/cities
                - n_planes: Number of plane types
                - time_periods: Total time periods
                - max_passengers: Max number of passengers per flight
                - capacity_range: Range for plane capacity
                - cost_range: Range for plane costs
                - max_available_planes: Max number of available planes per type
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "time_expanded_fleet_flow"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Time-Expanded Fleet Flow

        Sets:
        - V: fleet types.
        - L: locations.
        - T: time periods.
        - A: feasible flight legs (i,t,j,h) with i != j and h > t.

        Parameters:
        - q_v: passenger capacity of one vehicle or aircraft of type v.
        - c_v: operating cost per vehicle of type v assigned to a flight leg.
        - a_v: number of available vehicles of type v.
        - delta_{i,t,j,h} in {0,1}: whether flight leg (i,t,j,h) is active/allowed.
        - P_{i,t,j,h}: passenger demand on active flight leg (i,t,j,h).

        Decision variables:
        - n_{v,i,t,j,h} >= 0 integer: number of vehicles of type v assigned to leg (i,t,j,h).
        - idle_{v,i,t} >= 0 integer: idle vehicles of type v at location i after period t balance.
        - init_{v,i} >= 0 integer: initial vehicles of type v placed at location i.

        Minimize:
            sum_{v in V, (i,t,j,h) in A} c_v * n_{v,i,t,j,h}

        Subject to:
        - Initial balance at period 1:
            init_{v,i} = idle_{v,i,1} + outgoing vehicles from i in period 1.
        - Time-expanded fleet conservation for t > 1:
            idle_{v,i,t} = idle_{v,i,t-1} + arrivals to (i,t) - departures from (i,t).
        - Fleet availability:
            sum_i init_{v,i} <= a_v for each fleet type v.
        - Active leg demand satisfaction:
            sum_v q_v * n_{v,i,t,j,h} >= P_{i,t,j,h} for every active leg.
        - Route restriction:
            n_{v,i,t,j,h} <= a_v * delta_{i,t,j,h}.

        This is not a vehicle-routing problem with visit-once or subtour
        constraints. It is an integer fleet-flow model on a time-expanded
        network.
        """
        default_parameters = {
            "n_locations": (3, 4),
            "n_planes": (2, 4),
            "time_periods": (2, 3),
            "max_passengers": 80,
            "capacity_range": (100, 160),
            "cost_range": (5, 20),
            "max_available_planes": 3,
            "route_density": 0.35,
            "max_active_routes": 2
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
        Generate a Fleet Routing Problem instance and create its corresponding Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the fleet routing problem
        """
        # Randomly set number of locations, planes, and periods
        self.n_locations = random.randint(*self.n_locations) if isinstance(self.n_locations, tuple) else int(self.n_locations)
        self.n_planes = random.randint(*self.n_planes) if isinstance(self.n_planes, tuple) else int(self.n_planes)
        self.time_periods = random.randint(*self.time_periods) if isinstance(self.time_periods, tuple) else int(self.time_periods)
        
        locations = [f"location_{i}" for i in range(self.n_locations)]
        planes = [f"plane_{i}" for i in range(self.n_planes)]
        periods = [t for t in range(1, self.time_periods + 1)]
        
        # Generate random data for problem parameters
        capacity = {v: random.randint(*self.capacity_range) for v in planes}
        cost = {v: random.randint(*self.cost_range) for v in planes}
        available_planes = {v: self.max_available_planes for v in planes}
        feasible_arcs = [
            (i, t, j, h)
            for i in locations for t in periods
            for j in locations for h in periods
            if i != j and h > t
        ]
        active_route_count = random.randint(1, min(self.max_active_routes, len(feasible_arcs)))
        active_arcs = set(random.sample(feasible_arcs, active_route_count))
        delta = {arc: 1 if arc in active_arcs else 0 for arc in feasible_arcs}
        passengers = {(i, t, j, h): random.randint(1, self.max_passengers) if delta[i, t, j, h] else 0
                      for i, t, j, h in delta}
        
        # Create Gurobi model
        model = gp.Model("FleetRouting")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Decision variables
        num_planes = model.addVars(planes, feasible_arcs, lb=0, vtype=GRB.INTEGER, name="NumPlanes")
        num_idle_planes = model.addVars(planes, locations, periods, vtype=GRB.INTEGER, name="NumIdlePlanes")
        num_idle_planes_init = model.addVars(planes, locations, vtype=GRB.INTEGER, name="NumIdlePlanesInit")
        
        # Objective function: minimize total cost
        model.setObjective(
            gp.quicksum(cost[v] * num_planes[v, i, t, j, h] for v in planes for i, t, j, h in feasible_arcs),
            GRB.MINIMIZE
        )
        
        # Constraints
        # Flow balance constraints
        for v in planes:
            for i in locations:
                # Initial flow balance
                model.addConstr(
                    num_idle_planes_init[v, i] ==
                    num_idle_planes[v, i, 1] + gp.quicksum(
                        num_planes[v, arc_i, arc_t, arc_j, arc_h]
                        for arc_i, arc_t, arc_j, arc_h in feasible_arcs
                        if arc_i == i and arc_t == 1
                    ),
                    name=f"FlowBalanceInit_{v}_{i}"
                )
                for t in periods[1:]:
                    model.addConstr(
                        num_idle_planes[v, i, t] ==
                        num_idle_planes[v, i, t - 1]
                        + gp.quicksum(
                            num_planes[v, arc_i, arc_t, arc_j, arc_h]
                            for arc_i, arc_t, arc_j, arc_h in feasible_arcs
                            if arc_j == i and arc_h == t
                        )
                        - gp.quicksum(
                            num_planes[v, arc_i, arc_t, arc_j, arc_h]
                            for arc_i, arc_t, arc_j, arc_h in feasible_arcs
                            if arc_i == i and arc_t == t
                        ),
                        name=f"FlowBalance_{v}_{i}_{t}"
                    )
        
        # Plane availability constraints
        for v in planes:
            model.addConstr(
                gp.quicksum(num_idle_planes_init[v, i] for i in locations) <= available_planes[v],
                name=f"PlanesAvailability_{v}"
            )
        
        # Demand satisfaction constraints
        for i, t, j, h in active_arcs:
            model.addConstr(
                gp.quicksum(capacity[v] * num_planes[v, i, t, j, h] for v in planes) >= passengers[i, t, j, h],
                name=f"DemandSatisfaction_{i}_{t}_{j}_{h}"
            )
        
        # Route restriction constraints
        for v in planes:
            for i, t, j, h in feasible_arcs:
                model.addConstr(
                    num_planes[v, i, t, j, h] <= available_planes[v] * delta[i, t, j, h],
                    name=f"RouteRestriction_{v}_{i}_{t}_{j}_{h}"
                )

        self.capacity = capacity
        self.cost = cost
        self.available_planes = available_planes
        self.feasible_arcs = feasible_arcs
        self.active_arcs = sorted(active_arcs)
        self.delta = delta
        self.passengers = passengers
        self.parameters.update(
            {
                "locations": locations,
                "fleet_types": planes,
                "periods": periods,
                "capacity_by_fleet_type": capacity,
                "operating_cost_by_fleet_type": cost,
                "available_fleet_by_type": available_planes,
                "fleet_type_table": {
                    v: {
                        "capacity": capacity[v],
                        "operating_cost_per_assigned_leg": cost[v],
                        "available_units": available_planes[v],
                    }
                    for v in planes
                },
                "feasible_flight_legs": [
                    f"{i}|{t}|{j}|{h}" for i, t, j, h in feasible_arcs
                ],
                "active_flight_legs": [
                    f"{i}|{t}|{j}|{h}" for i, t, j, h in sorted(active_arcs)
                ],
                "active_leg_table": {
                    f"{i}|{t}|{j}|{h}": {
                        "origin": i,
                        "departure_period": t,
                        "destination": j,
                        "arrival_period": h,
                        "passenger_demand": passengers[i, t, j, h],
                    }
                    for i, t, j, h in sorted(active_arcs)
                },
                "leg_is_active": {
                    f"{i}|{t}|{j}|{h}": delta[i, t, j, h]
                    for i, t, j, h in feasible_arcs
                },
                "passenger_demand": {
                    f"{i}|{t}|{j}|{h}": passengers[i, t, j, h]
                    for i, t, j, h in sorted(active_arcs)
                },
                "compact_fleet_flow_tables": {
                    "sets": {
                        "fleet_types": planes,
                        "locations": locations,
                        "periods": periods,
                    },
                    "fleet_type_table": {
                        "columns": ["fleet_type", "capacity", "operating_cost_per_assigned_leg", "available_units"],
                        "rows": [[v, capacity[v], cost[v], available_planes[v]] for v in planes],
                    },
                    "feasible_leg_table": {
                        "columns": [
                            "origin",
                            "departure_period",
                            "destination",
                            "arrival_period",
                            "leg_is_active",
                            "passenger_demand",
                        ],
                        "rows": [
                            [i, t, j, h, delta[i, t, j, h], passengers[i, t, j, h]]
                            for i, t, j, h in feasible_arcs
                        ],
                    },
                    "active_leg_table": {
                        "columns": ["origin", "departure_period", "destination", "arrival_period", "passenger_demand"],
                        "rows": [
                            [i, t, j, h, passengers[i, t, j, h]]
                            for i, t, j, h in sorted(active_arcs)
                        ],
                    },
                    "required_decision_layers": [
                        "NumPlanes[v,i,t,j,h] nonnegative integer fleet units assigned to feasible leg",
                        "NumIdlePlanes[v,i,t] nonnegative integer idle fleet units after period balance",
                        "NumIdlePlanesInit[v,i] nonnegative integer initial fleet-unit placement",
                    ],
                    "not_a_vrp": True,
                },
                "time_expanded_network_summary": {
                    "locations": len(locations),
                    "periods": len(periods),
                    "fleet_types": len(planes),
                    "feasible_leg_count": len(feasible_arcs),
                    "active_leg_count": len(active_arcs),
                    "inactive_leg_count": len(feasible_arcs) - len(active_arcs),
                    "leg_time_rule": "A feasible leg has origin != destination and arrival_period > departure_period.",
                },
                "decision_variables": [
                    "NumPlanes[v,i,t,j,h] nonnegative integer count of fleet units of type v assigned to feasible leg (i,t,j,h)",
                    "NumIdlePlanes[v,i,t] nonnegative integer idle fleet units of type v at location i after period t balance",
                    "NumIdlePlanesInit[v,i] nonnegative integer initial fleet units of type v placed at location i",
                ],
                "objective": "minimize_fleet_operating_cost_on_time_expanded_legs",
                "required_constraints": [
                    "initial_fleet_balance_by_location",
                    "time_expanded_fleet_flow_conservation",
                    "fleet_availability_by_type",
                    "active_leg_passenger_demand_satisfaction",
                    "inactive_leg_route_restriction",
                    "nonnegative_integer_flight_leg_counts",
                    "nonnegative_integer_idle_fleet_counts",
                ],
                "structured_problem_data": {
                    "sets": {
                        "fleet_types": "V",
                        "locations": "L",
                        "periods": "T",
                        "feasible_legs": "A={(i,t,j,h): i != j and h > t}",
                        "active_legs": "A_active subset of A with passenger demand",
                    },
                    "parameters": [
                        "capacity_by_fleet_type[v]",
                        "operating_cost_by_fleet_type[v]",
                        "available_fleet_by_type[v]",
                        "leg_is_active[i,t,j,h]",
                        "passenger_demand[i,t,j,h] only for active legs",
                    ],
                    "constraints": [
                        "NumIdlePlanesInit[v,i] = NumIdlePlanes[v,i,1] + outgoing NumPlanes from i at period 1",
                        "NumIdlePlanes[v,i,t] = NumIdlePlanes[v,i,t-1] + arrivals to (i,t) - departures from (i,t) for t > 1",
                        "sum_i NumIdlePlanesInit[v,i] <= available_fleet_by_type[v]",
                        "sum_v capacity_by_fleet_type[v] * NumPlanes[v,i,t,j,h] >= passenger_demand[i,t,j,h] for active legs",
                        "NumPlanes[v,i,t,j,h] <= available_fleet_by_type[v] * leg_is_active[i,t,j,h] for every feasible leg",
                    ],
                },
                "required_parameter_presentation": [
                    "List fleet types with capacity, operating cost per assigned leg, and available fleet count.",
                    "List locations and planning periods.",
                    "List feasible legs as origin, departure period, destination, and arrival period.",
                    "List active legs and passenger demand only for those active legs.",
                    "State inactive feasible legs have leg_is_active=0 and therefore NumPlanes must be zero.",
                    "State all decision variables are nonnegative integer counts, not binary visit variables.",
                    "State this is time-expanded fleet flow conservation, not customer-stop vehicle routing.",
                ],
                "business_interpretation_guardrails": [
                    "This is an integer time-expanded fleet-flow model, not a VRP.",
                    "Do not add visit-once, subtour elimination, customer service, or time-window constraints.",
                    "Do not require binary route-arc variables; the decision is the integer number of fleet units assigned to each active leg.",
                    "Only listed active flight legs have passenger demand.",
                    "Do not omit idle fleet variables or initial fleet placement variables; they are required for time-expanded conservation.",
                ],
                "generation_notes": [
                    "The active-leg set is intentionally sparse so the full active demand table can be stated compactly.",
                    "RouteRestriction constraints force all inactive feasible legs to carry zero assigned fleet units.",
                    "This source has no customer visit-once, subtour, service-time, or time-window constraints.",
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
        model.write("fleet_routing.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    test_generator()
