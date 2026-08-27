import gurobipy as gp 
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Aircraft Assignment optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_aircraft: Number of aircraft types
            - n_routes: Number of routes
            - availability_range: Tuple of (min, max) for aircraft availability
            - demand_range: Tuple of (min, max) for route demand
            - capabilities_range: Tuple of (min, max) for aircraft capabilities
            - cost_range: Tuple of (min, max) for assignment costs
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "aircraft_assignment"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Sets:
        - A: Set of aircraft types
        - R: Set of routes
        
        Parameters:
        - availability_a: Availability of aircraft type a
        - demand_r: Demand for route r
        - capabilities_ar: Capabilities of aircraft a for route r
        - costs_ar: Cost of assigning aircraft a to route r
        
        Variables:
        - allocation_ar: Number of aircraft type a assigned to route r (integer)
        
        $$
        \begin{aligned}
        &\text{Minimize} && \sum_{a \in A}\sum_{r \in R} costs_{ar} \cdot allocation_{ar} \\
        &\text{Subject to} && \sum_{r \in R} allocation_{ar} \leq availability_a && \forall a \in A \\
        & && \sum_{a \in A} allocation_{ar} \cdot capabilities_{ar} \geq demand_r && \forall r \in R \\
        & && allocation_{ar} \geq 0, \text{ integer} && \forall a \in A, r \in R
        \end{aligned}
        $$
        """
        
        default_parameters = {
            "n_aircraft": (4, 6),
            "n_routes": (3, 4),
            "availability_range": (1, 10),
            "demand_range": (100, 500),
            "capabilities_range": (50, 200),
            "cost_range": (1000, 5000)
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
        Generate an Aircraft Assignment problem instance and create its corresponding Gurobi model.
        Returns:
        gp.Model: Configured Gurobi model for the aircraft assignment problem
        """
        # Randomly select number of aircraft types and routes
        self.n_aircraft = _sample_int(self.n_aircraft)
        self.n_routes = _sample_int(self.n_routes)

        # Generate sets
        self.aircraft = [f"aircraft_{i}" for i in range(self.n_aircraft)]
        self.routes = [f"route_{i}" for i in range(self.n_routes)]

        # Generate parameters and save them as class attributes. Route demand is
        # derived from a hidden feasible allocation, instead of being sampled
        # independently from fleet capacity.
        self.availability = {a: random.randint(*self.availability_range) for a in self.aircraft}
        while sum(self.availability.values()) < self.n_routes:
            self.availability[random.choice(self.aircraft)] += 1
        self.capabilities = {(a,r): random.randint(*self.capabilities_range) 
                            for a in self.aircraft for r in self.routes}
        self.costs = {(a,r): random.randint(*self.cost_range) 
                    for a in self.aircraft for r in self.routes}
        remaining_availability = dict(self.availability)
        hidden_feasible_allocation = {
            (a, r): 0 for a in self.aircraft for r in self.routes
        }
        feasible_route_capacity = {r: 0 for r in self.routes}
        min_demand, max_demand = self.demand_range
        for r in self.routes:
            available_aircraft = [a for a in self.aircraft if remaining_availability[a] > 0]
            if not available_aircraft:
                break
            first_aircraft = random.choice(available_aircraft)
            remaining_availability[first_aircraft] -= 1
            hidden_feasible_allocation[first_aircraft, r] += 1
            feasible_route_capacity[r] += self.capabilities[first_aircraft, r]
            while feasible_route_capacity[r] < min_demand and any(value > 0 for value in remaining_availability.values()):
                candidates = [a for a in self.aircraft if remaining_availability[a] > 0]
                best_aircraft = max(candidates, key=lambda a: self.capabilities[a, r])
                remaining_availability[best_aircraft] -= 1
                hidden_feasible_allocation[best_aircraft, r] += 1
                feasible_route_capacity[r] += self.capabilities[best_aircraft, r]
        self.demand = {}
        for r in self.routes:
            route_capacity = max(1, feasible_route_capacity[r])
            demand_upper = min(max_demand, route_capacity)
            demand_lower = min(min_demand, demand_upper)
            self.demand[r] = random.randint(demand_lower, demand_upper)
        screening_model = self._build_model()
        screening_model.optimize()
        self._tighten_instance_to_screened_solution(screening_model)
        self._postprocess_iterations = 1
        model = self._build_model()
        # Re-tighten around the final optimum if the first tightening still
        # leaves all core constraints slack. This keeps the instance feasible
        # while making the route-capacity and fleet-availability constraints
        # meaningful for downstream training.
        for iteration in range(5):
            model.optimize()
            if model.Status != GRB.OPTIMAL or self._core_binding_count(model) >= 1:
                self._postprocess_iterations = iteration + 1
                break
            self._tighten_instance_to_screened_solution(model)
            model = self._build_model()
        self._record_instance_metadata()
        return model

    def _tighten_instance_to_screened_solution(self, model):
        if model.Status != GRB.OPTIMAL:
            return
        tightened_demand = {}
        tightened_availability = {}
        for a in self.aircraft:
            used_count = 0
            for r in self.routes:
                variable = model.getVarByName(f"Allocation[{a},{r}]")
                if variable is None:
                    continue
                used_count += int(round(variable.X))
            # Keep every aircraft type visible in the data table, but make
            # aircraft types used by the screened solution fleet-tight.
            tightened_availability[a] = max(1, used_count)
        for r in self.routes:
            provided_capacity = 0
            for a in self.aircraft:
                variable = model.getVarByName(f"Allocation[{a},{r}]")
                if variable is None:
                    continue
                provided_capacity += int(round(variable.X)) * self.capabilities[a, r]
            tightened_demand[r] = max(1, int(round(provided_capacity)))
        if all(value > 0 for value in tightened_demand.values()):
            self.demand = tightened_demand
        if any(value > 0 for value in tightened_availability.values()):
            self.availability = tightened_availability

    def _core_binding_count(self, model):
        if model.Status != GRB.OPTIMAL:
            return 0
        binding = 0
        for constraint in model.getConstrs():
            if constraint.ConstrName.startswith(("Availability_", "Demand_")) and abs(float(constraint.Slack)) <= 1e-6:
                binding += 1
        return binding

    def _record_instance_metadata(self):
        compact_aircraft_assignment_tables = {
            "sets": {
                "aircraft_types": self.aircraft,
                "routes": self.routes,
            },
            "aircraft_type_table": [
                {
                    "aircraft_type": a,
                    "available_count": self.availability[a],
                }
                for a in self.aircraft
            ],
            "route_demand_table": [
                {
                    "route": r,
                    "required_capacity": self.demand[r],
                }
                for r in self.routes
            ],
            "capacity_contribution_matrix": {
                "columns": self.routes,
                "rows": [
                    {
                        "aircraft_type": a,
                        "values": [self.capabilities[a, r] for r in self.routes],
                    }
                    for a in self.aircraft
                ],
            },
            "operating_cost_matrix": {
                "columns": self.routes,
                "rows": [
                    {
                        "aircraft_type": a,
                        "values": [self.costs[a, r] for r in self.routes],
                    }
                    for a in self.aircraft
                ],
            },
            "decision_variable": "Allocation[a,r] is a nonnegative integer count of aircraft type a assigned to route r",
            "objective": "minimize sum of operating_cost[a,r] * Allocation[a,r]",
            "constraints": [
                "for each aircraft type a, sum_r Allocation[a,r] <= available_count[a]",
                "for each route r, sum_a capacity_contribution[a,r] * Allocation[a,r] >= required_capacity[r]",
            ],
            "source_contract_note": (
                "This is fleet-type capacity allocation. It is not tail routing, passenger spill modeling, "
                "or binary one-aircraft-to-one-route matching."
            ),
        }
        self.parameters.update(
            {
                "aircraft": self.aircraft,
                "routes": self.routes,
                "availability": self.availability,
                "demand": self.demand,
                "compact_aircraft_assignment_tables": compact_aircraft_assignment_tables,
                "generation_note": (
                    "Route demand and aircraft availability are tightened around a screened optimal allocation so that "
                    "at least one core capacity or availability requirement is binding. The screened allocation itself "
                    "is not part of the business problem and should not be presented to the model."
                ),
                "postprocess_iterations": getattr(self, "_postprocess_iterations", 1),
                "decision_variables": {
                    "Allocation[a,r]": "nonnegative integer number of aircraft of type a assigned to route r",
                },
                "objective_terms": [
                    "aircraft_route_operating_cost_times_allocation",
                ],
                "required_constraints": [
                    "aircraft_type_availability_limit",
                    "route_capacity_meets_or_exceeds_demand",
                    "nonnegative_integer_allocation",
                ],
                "required_parameter_presentation": [
                    "list aircraft types and available aircraft count for every type",
                    "list routes and passenger demand for every route",
                    "list seat or payload capacity contribution for every aircraft-route pair",
                    "list operating cost for every aircraft-route pair",
                    "state that assigned capacity on each route must meet or exceed demand",
                    "state that each aircraft type cannot be used more times than its availability",
                    "prefer compact_aircraft_assignment_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "aircraft_types": self.aircraft,
                        "routes": self.routes,
                    },
                    "parameters": {
                        "availability": self.availability,
                        "route_demand": self.demand,
                        "capacity_contribution_table": "see compact_aircraft_assignment_tables.capacity_contribution_matrix",
                        "assignment_cost_table": "see compact_aircraft_assignment_tables.operating_cost_matrix",
                        "compact_tables": "see compact_aircraft_assignment_tables",
                    },
                    "constraints": {
                        "availability": "sum of Allocation[a,r] over routes is at most availability[a]",
                        "route_demand": "sum of capacity[a,r] * Allocation[a,r] over aircraft types must meet or exceed demand[r]",
                    },
                    "objective": "minimize total aircraft-route assignment cost",
                },
                "business_interpretation_guardrails": [
                    "This is an integer fleet-to-route allocation model, not a binary one-aircraft-to-one-route assignment model.",
                    "Do not introduce time-expanded aircraft balance or aircraft routing continuity constraints.",
                    "Do not introduce passenger itinerary spill, recapture, or revenue-management variables.",
                    "Do not change route demand into an equality unless the statement explicitly requires exact capacity.",
                    "Do not omit the aircraft availability limits or aircraft-route capacity contribution table.",
                    "Do not expose or rely on any internally generated feasible allocation certificate in the business statement.",
                ],
            }
        )

    def _build_model(self):
        model = gp.Model("AircraftAssignment")
        model.Params.OutputFlag = 0
        allocation = model.addVars(self.aircraft, self.routes, vtype=GRB.INTEGER, name="Allocation")
        model.setObjective(
            gp.quicksum(self.costs[a, r] * allocation[a, r] for a in self.aircraft for r in self.routes),
            GRB.MINIMIZE,
        )
        for a in self.aircraft:
            model.addConstr(
                gp.quicksum(allocation[a, r] for r in self.routes) <= self.availability[a],
                name=f"Availability_{a}",
            )
        for r in self.routes:
            model.addConstr(
                gp.quicksum(allocation[a, r] * self.capabilities[a, r] for a in self.aircraft) >= self.demand[r],
                name=f"Demand_{r}",
            )
        return model
    
    def print_solution(self, model):
        """Print the solution details for the Aircraft Assignment Problem"""
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal total cost: {model.ObjVal:.2f}")
            
            # Get sets from variable names
            aircraft = [f"aircraft_{i}" for i in range(self.n_aircraft)]
            routes = [f"route_{i}" for i in range(self.n_routes)]
            
            # Print allocation details and calculate statistics
            total_aircraft_used = 0
            total_capacity_provided = 0
            
            print("\nDetailed Allocation:")
            print("-" * 60)
            for a in aircraft:
                aircraft_count = 0
                for r in routes:
                    var_name = f"Allocation[{a},{r}]"
                    for v in model.getVars():
                        if v.VarName == var_name and v.X > 0:  # Non-zero allocation
                            allocation = int(v.X)  # Should be integer
                            aircraft_count += allocation
                            capacity = allocation * self.capabilities[a,r]
                            cost = allocation * self.costs[a,r]
                            total_aircraft_used += allocation
                            total_capacity_provided += capacity
                            
                            print(f"{a} -> {r}: "
                                f"Allocated: {allocation} aircraft, "
                                f"Capacity: {capacity}, "
                                f"Cost: {cost:.2f}")
                
                print(f"Total {a} used: {aircraft_count}")
                print("-" * 30)
            
            print("\nSummary Statistics:")
            print("-" * 60)
            print(f"Total Aircraft Used: {total_aircraft_used}")
            print(f"Total Capacity Provided: {total_capacity_provided}")
            print(f"Total Cost: {model.ObjVal:.2f}")
            
            # Check capacity vs demand
            print("\nRoute Demand Satisfaction:")
            print("-" * 60)
            for r in routes:
                total_route_capacity = sum(
                    model.getVarByName(f"Allocation[{a},{r}]").X * self.capabilities[a,r]
                    for a in aircraft
                )
                print(f"{r}: Demand = {self.demand[r]}, "
                    f"Capacity Provided = {total_route_capacity}")
                
        else:
            print("No optimal solution found")
            print(f"Status code: {model.Status}")
        
        # Print model statistics
        print(f"\nModel Statistics:")
        print("-" * 60)
        print(f"Number of variables: {model.NumVars}")
        print(f"Number of constraints: {model.NumConstrs}")
        
        # If model is infeasible, try to compute IIS
        if model.Status == GRB.INFEASIBLE:
            print("\nAnalyzing Infeasibility:")
            print("-" * 60)
            try:
                model.computeIIS()
                print("Infeasible constraints:")
                for c in model.getConstrs():
                    if c.IISConstr:
                        print(f"Constraint {c.ConstrName} is infeasible")
                        
                # Additional analysis for availability constraints
                print("\nAvailability Analysis:")
                for a in aircraft:
                    total_allocated = sum(
                        model.getVarByName(f"Allocation[{a},{r}]").X 
                        for r in routes
                    )
                    print(f"{a}: Available = {self.availability[a]}, "
                        f"Allocated = {total_allocated}")
                    
                # Analysis for demand constraints
                print("\nDemand Analysis:")
                for r in routes:
                    total_capacity = sum(
                        model.getVarByName(f"Allocation[{a},{r}]").X * self.capabilities[a,r]
                        for a in aircraft
                    )
                    print(f"{r}: Required = {self.demand[r]}, "
                        f"Provided = {total_capacity}")
                    
            except:
                print("Could not compute IIS")
                
        # Print computation time if available
        if hasattr(model, "Runtime"):
            print(f"\nSolution time: {model.Runtime:.2f} seconds")
def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


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
        
        generator.print_solution(model)

    test_generator()
