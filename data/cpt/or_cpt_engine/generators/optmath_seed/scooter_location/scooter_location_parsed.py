import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Scooter Location optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_demand_points: Number of demand points (tuple of min, max)
                - n_candidate_locations: Number of candidate locations (tuple of min, max)
                - demand_range: Tuple of (min, max) for demand at each demand point
                - distance_range: Tuple of (min, max) for distance between demand points and candidate locations
                - num_available_scooters: Number of scooters already available (tuple of min, max)
                - max_selected_locations: Maximum number of selected locations (tuple of min, max)
                - new_max: Maximum number of new scooters that can be added (tuple of min, max)
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "scooter_location"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are:
        - Demand points: i ∈ DemandPoints
        - Candidate locations: j ∈ CandidateLocations
        - Estimated demand: EstimatedDemand_i
        - Distance: Distance_{i,j}
                - Existing scooters at each candidate location: ExistingScooters_j
                - Maximum number of selected locations: MaxSelectedLocations
                - Maximum number of new scooters: NewMax
        Decision Variables:
        - SelectedLocation_j: Binary variable, 1 if candidate location j is selected; 0 otherwise
        - Assign_{i,j}: Binary variable, 1 if demand point i is assigned to candidate location j; 0 otherwise
        - NewScooters_j: Integer variable representing new scooters added at candidate location j
        Objective:
        Minimize weighted user travel distance, opening costs, and new scooter deployment costs:
        $$
        \text{Minimize} \quad
        \sum_i \sum_j \text{EstimatedDemand}_i \text{Distance}_{i,j} \text{Assign}_{i,j}
        + \sum_j \text{OpenCost}_j \text{SelectedLocation}_j
        + \sum_j \text{NewScooterCost}_j \text{NewScooters}_j
        $$
        Constraints:
        1. Assign_{i,j} ≤ SelectedLocation_j ∀ i ∈ DemandPoints, j ∈ CandidateLocations
        2. \sum_{j \in \text{CandidateLocations}} Assign_{i,j} = 1 ∀ i ∈ DemandPoints
        3. \sum_i \text{EstimatedDemand}_i Assign_{i,j} ≤ ExistingScooters_j SelectedLocation_j + NewScooters_j ∀ j
        4. \sum_{j \in \text{CandidateLocations}} \text{SelectedLocation}_j ≤ \text{MaxSelectedLocations}
        5. \sum_j \text{NewScooters}_j ≤ \text{NewMax}
        """
        # Default parameters with adjusted ranges
        default_parameters = {
            "n_demand_points": (8, 10),                
            "n_candidate_locations": (5, 8),          
            "demand_range": (10, 100),             
            "distance_range": (1, 20),                
            "max_selected_locations": (2, 10),        
            "existing_capacity_fraction": (0.35, 0.60),
            "new_capacity_buffer_fraction": (0.08, 0.20),
            "opening_cost_range": (40, 220),
            "new_scooter_cost_range": (2, 12),
        }
        
        # Use provided parameters if they exist, otherwise fallback to defaults
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
        Generate a Scooter Location problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (demand points, candidate locations, distances, etc.)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the scooter location problem
        """
        # 1. Randomly determine parameters within ranges
        self.n_demand_points = random.randint(*self.n_demand_points)
        self.n_candidate_locations = random.randint(*self.n_candidate_locations)
        self.max_selected_locations = random.randint(*self.max_selected_locations)

        # 2. Generate data for demand points and candidate locations
        demand_points = [f"demand_{i}" for i in range(self.n_demand_points)]
        candidate_locations = [f"location_{j}" for j in range(self.n_candidate_locations)]
        
        estimated_demand = {i: random.randint(*self.demand_range) for i in demand_points}
        distance = {(i, j): random.randint(*self.distance_range) for i in demand_points for j in candidate_locations}
        total_demand = sum(estimated_demand.values())
        self.max_selected_locations = max(2, min(self.max_selected_locations, self.n_candidate_locations))
        existing_fraction = uniform_rounded(*self.existing_capacity_fraction)
        target_existing_capacity = max(self.n_candidate_locations, int(round(total_demand * existing_fraction)))
        raw_weights = [random.random() + 0.2 for _ in candidate_locations]
        weight_sum = sum(raw_weights)
        existing_scooters = {
            j: max(1, int(round(target_existing_capacity * raw_weights[index] / weight_sum)))
            for index, j in enumerate(candidate_locations)
        }
        top_existing_capacity = sum(sorted(existing_scooters.values(), reverse=True)[:self.max_selected_locations])
        required_new = max(0, total_demand - top_existing_capacity)
        buffer = int(round(total_demand * uniform_rounded(*self.new_capacity_buffer_fraction)))
        self.new_max = required_new + buffer
        opening_cost = {j: random.randint(*self.opening_cost_range) for j in candidate_locations}
        new_scooter_cost = {j: random.randint(*self.new_scooter_cost_range) for j in candidate_locations}
        self.parameters.update(
            {
                "demand_points": demand_points,
                "candidate_locations": candidate_locations,
                "estimated_demand": estimated_demand,
                "distance": {f"{i}|{j}": distance[i, j] for i in demand_points for j in candidate_locations},
                "existing_scooters": existing_scooters,
                "opening_cost": opening_cost,
                "new_scooter_cost": new_scooter_cost,
                "total_demand": total_demand,
                "max_selected_locations": self.max_selected_locations,
                "new_max": self.new_max,
                "compact_scooter_location_tables": {
                    "sets": {
                        "demand_points": demand_points,
                        "candidate_locations": candidate_locations,
                    },
                    "demand_table": {
                        "columns": ["demand_point", "estimated_demand"],
                        "rows": [[i, estimated_demand[i]] for i in demand_points],
                    },
                    "candidate_location_table": {
                        "columns": [
                            "candidate_location",
                            "existing_scooters",
                            "opening_cost",
                            "new_scooter_cost",
                        ],
                        "rows": [
                            [
                                j,
                                existing_scooters[j],
                                opening_cost[j],
                                new_scooter_cost[j],
                            ]
                            for j in candidate_locations
                        ],
                    },
                    "distance_table": {
                        "columns": ["demand_point", "candidate_location", "walking_distance"],
                        "rows": [
                            [i, j, distance[i, j]]
                            for i in demand_points
                            for j in candidate_locations
                        ],
                    },
                    "global_limits": {
                        "max_selected_locations": self.max_selected_locations,
                        "max_new_scooters_total": self.new_max,
                        "total_demand": total_demand,
                    },
                    "required_decision_layers": [
                        "SelectedLocation[j] binary, 1 if scooter station j is opened or selected",
                        "Assign[i,j] binary, 1 if demand point i is assigned to station j",
                        "NewScooters[j] nonnegative integer number of new scooters placed at station j",
                    ],
                },
                "decision_variables": {
                    "SelectedLocation[j]": "binary station-opening decision for candidate location j",
                    "Assign[i,j]": "binary assignment of demand point i to selected station j",
                    "NewScooters[j]": "nonnegative integer new scooters deployed at selected station j",
                },
                "objective_terms": [
                    "demand_weighted_walking_distance",
                    "station_opening_cost",
                    "new_scooter_deployment_cost",
                ],
                "required_constraints": [
                    "assign_only_to_selected_location",
                    "each_demand_point_assigned_exactly_once",
                    "selected_location_capacity_with_existing_and_new_scooters",
                    "maximum_selected_locations",
                    "total_new_scooter_budget",
                    "new_scooters_only_at_selected_locations",
                ],
                "required_parameter_presentation": [
                    "list every demand point with estimated demand",
                    "list every candidate station with existing scooters, opening cost, and new scooter cost",
                    "list the complete demand-point-to-candidate-station distance table",
                    "state each demand point must be assigned to exactly one selected station",
                    "state assigned demand at a station cannot exceed existing scooters at selected stations plus new scooters",
                    "state the total number of selected stations and the total number of new scooters are limited",
                    "state this is not a route, tour, or vehicle sequencing problem",
                ],
                "structured_problem_data": {
                    "sets": {
                        "demand_points": demand_points,
                        "candidate_locations": candidate_locations,
                    },
                    "parameters": {
                        "demand_table": "see compact_scooter_location_tables.demand_table",
                        "candidate_location_table": "see compact_scooter_location_tables.candidate_location_table",
                        "distance_table": "see compact_scooter_location_tables.distance_table",
                        "global_limits": "see compact_scooter_location_tables.global_limits",
                    },
                    "objective": "minimize demand-weighted walking distance plus station opening cost plus new scooter deployment cost",
                    "constraints": {
                        "assignment": "each demand point assigned exactly once and only to selected stations",
                        "capacity": "assigned demand cannot exceed existing scooters at selected stations plus new scooters",
                        "budgets": "limit selected station count and total new scooters",
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a micromobility station-location and demand-assignment MILP.",
                    "Do not reinterpret it as vehicle routing, a traveling-salesperson tour, or a shortest-path problem.",
                    "Do not drop station opening costs, new scooter deployment costs, assignment linking, or station capacity constraints.",
                    "Do not allow new scooters at a location unless that location is selected.",
                ],
            }
        )
        
        # 3. Create Gurobi model
        model = gp.Model("ScooterLocation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create variables
        selected_location = model.addVars(candidate_locations, vtype=GRB.BINARY, name="SelectedLocation")
        assign = model.addVars(demand_points, candidate_locations, vtype=GRB.BINARY, name="Assign")
        new_scooters = model.addVars(candidate_locations, vtype=GRB.INTEGER, lb=0, name="NewScooters")
        
        # Set objective to minimize total user travel distance
        model.setObjective(
            gp.quicksum(estimated_demand[i] * distance[i, j] * assign[i, j] 
                        for i in demand_points for j in candidate_locations)
            + gp.quicksum(opening_cost[j] * selected_location[j] for j in candidate_locations)
            + gp.quicksum(new_scooter_cost[j] * new_scooters[j] for j in candidate_locations),
            GRB.MINIMIZE
        )
        
        # Add constraints
        model.addConstrs(
            (assign[i, j] <= selected_location[j] for i in demand_points for j in candidate_locations),
            name="AssignOpenLink"
        )
        model.addConstrs(
            (gp.quicksum(assign[i, j] for j in candidate_locations) == 1 for i in demand_points),
            name="DemandAssignment"
        )
        model.addConstrs(
            (
                gp.quicksum(estimated_demand[i] * assign[i, j] for i in demand_points)
                <= existing_scooters[j] * selected_location[j] + new_scooters[j]
                for j in candidate_locations
            ),
            name="LocationCapacity"
        )
        model.addConstr(
            gp.quicksum(selected_location[j] for j in candidate_locations) <= self.max_selected_locations,
            name="LimitLocation"
        )
        model.addConstr(
            gp.quicksum(new_scooters[j] for j in candidate_locations) <= self.new_max,
            name="NewScooterBudget"
        )
        model.addConstrs(
            (new_scooters[j] <= total_demand * selected_location[j] for j in candidate_locations),
            name="NewScooterOpenLink"
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
        
        print(f"\nTest with adjusted parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            print(generator.n_demand_points, generator.n_candidate_locations)

    test_generator()
