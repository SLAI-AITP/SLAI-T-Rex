import gurobipy as gp 
from gurobipy import GRB
import random
import numpy as np

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Traveling Salesman Problem optimization problem.
        Parameters:
        parameters (dict): Dictionary containing:
            - n_cities: Number of cities
            - distance_range: Tuple of (min, max) for distances between cities
        seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "traveling_salesman"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Minimize total travel distance subject to:
        - Each city must be visited exactly once
        - Each city must be departed from exactly once  
        - Subtour elimination constraints
        - Start and end at the same city
        """
        
        default_parameters = {
            "n_cities": (5, 8),
            "distance_range": (10, 1000)
        }
        
        parameters = {**default_parameters, **dict(parameters or {})}
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def generate_instance(self):
        """
        Generate a TSP instance and create its Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the TSP
        """
        # Generate random number of cities
        if isinstance(self.n_cities, tuple):
            self.n_cities = random.randint(*self.n_cities)
            
        # Generate cities
        self.cities = [f"city_{i}" for i in range(self.n_cities)]
        
        # Generate random distances between cities
        self.distances = {
            (i, j): random.randint(*self.distance_range)
            for i in self.cities
            for j in self.cities 
            if i != j
        }
        arcs = sorted(self.distances)
        depot = self.cities[0]
        self.parameters.update(
            {
                "cities": self.cities,
                "depot": depot,
                "directed_arcs": [[i, j] for i, j in arcs],
                "distance": {f"{i}|{j}": self.distances[i, j] for i, j in arcs},
                "compact_tsp_tables": {
                    "sets": {
                        "cities": self.cities,
                        "depot": depot,
                        "directed_arcs": [[i, j] for i, j in arcs],
                    },
                    "distance_table": {
                        "columns": ["from_city", "to_city", "travel_distance"],
                        "rows": [[i, j, self.distances[i, j]] for i, j in arcs],
                    },
                    "required_decision_layers": [
                        "route[i,j] binary, 1 if the tour travels directly from city i to city j",
                        "u[i] integer MTZ ordering variable for non-depot city i",
                    ],
                },
                "decision_variables": {
                    "route[i,j]": "binary arc-use variable for directed travel from city i to city j",
                    "u[i]": "integer MTZ order variable used only for subtour elimination on non-depot cities",
                },
                "objective_terms": [
                    "directed_arc_distance_times_route_binary",
                ],
                "required_constraints": [
                    "exactly_one_inbound_arc_per_city",
                    "exactly_one_outbound_arc_per_city",
                    "single_tour_mtz_subtour_elimination",
                ],
                "required_parameter_presentation": [
                    "list all cities and identify the depot/start city",
                    "list the complete directed travel-distance table for every ordered city pair i != j",
                    "state route[i,j] is binary and self-loops are not allowed",
                    "state every city has exactly one incoming selected arc and exactly one outgoing selected arc",
                    "state MTZ order constraints eliminate subtours and force one single tour returning to the depot",
                    "state there are no vehicle capacities, time windows, service times, or multiple vehicles",
                ],
                "structured_problem_data": {
                    "sets": {
                        "cities": self.cities,
                        "depot": depot,
                        "directed_arcs": [[i, j] for i, j in arcs],
                    },
                    "parameters": {
                        "distance_table": "see compact_tsp_tables.distance_table",
                    },
                    "objective": "minimize total directed travel distance over selected tour arcs",
                    "constraints": {
                        "visit_once": "exactly one inbound arc and exactly one outbound arc for every city",
                        "subtour_elimination": "MTZ ordering constraints for non-depot cities",
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a single traveling-salesperson tour over all listed cities.",
                    "Do not add vehicle capacity, time windows, service times, multiple vehicles, or delivery quantities.",
                    "Do not relax the tour into a shortest path or transportation flow.",
                    "Do not omit subtour-elimination logic.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("TSP")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Decision variables
        # x[i,j] = 1 if we travel from city i to j
        x = model.addVars(self.distances.keys(), vtype=GRB.BINARY, name="route")
        
        # Additional variables for subtour elimination
        u = model.addVars(self.cities, vtype=GRB.INTEGER, name="u")
        
        # Objective: minimize total distance
        model.setObjective(
            gp.quicksum(self.distances[i,j] * x[i,j] for i,j in self.distances.keys()),
            GRB.MINIMIZE
        )
        
        # Constraints
        # Each city must be visited exactly once
        for j in self.cities:
            model.addConstr(
                gp.quicksum(x[i,j] for i in self.cities if i != j) == 1,
                f"visit_{j}"
            )
            
        # Must depart from each city exactly once
        for i in self.cities:
            model.addConstr(
                gp.quicksum(x[i,j] for j in self.cities if i != j) == 1,
                f"depart_{i}"
            )
            
        # Subtour elimination constraints
        for i in self.cities:
            if i != self.cities[0]:
                model.addConstr(u[i] >= 0, name=f"MTZLower_{i}")
                model.addConstr(u[i] <= len(self.cities)-1, name=f"MTZUpper_{i}")
        
        for i,j in self.distances.keys():
            if i != self.cities[0] and j != self.cities[0]:
                model.addConstr(
                    u[i] - u[j] + len(self.cities)*x[i,j] <= len(self.cities)-1,
                    name=f"MTZSubtour_{i}_{j}",
                )
                
        return model

    def print_solution(self, model):
        """
        Print the solution of the TSP in a readable format.
        Parameters:
            model (gp.Model): Solved Gurobi model
        """
        if model.Status != GRB.OPTIMAL:
            print("No optimal solution found.")
            return
            
        print("\n=== Traveling Salesman Solution ===")
        print(f"Total Distance: {model.ObjVal:.2f}")
        
        # Extract the route
        route = []
        current_city = self.cities[0]
        for _ in range(len(self.cities)):
            next_city = None
            for v in model.getVars():
                if v.VarName.startswith('route'):
                    city1, city2 = v.VarName.split('[')[1].split(']')[0].split(',')
                    if city1 == current_city and v.X > 0.5:
                        next_city = city2
                        break
            if next_city is None:
                break
            route.append(current_city)
            current_city = next_city
            
        route.append(route[0])  # Return to start
        
        print("\nOptimal Route:")
        print(" -> ".join(route))
        
        print("\nSegment Details:")
        print(f"{'From':^10} {'To':^10} {'Distance':^10}")
        print("-" * 30)
        total_distance = 0
        for i in range(len(route)-1):
            distance = self.distances[route[i], route[i+1]]
            total_distance += distance
            print(f"{route[i]:^10} {route[i+1]:^10} {distance:^10}")
            
        print("\nStatistics:")
        print(f"Number of cities: {len(self.cities)}")
        print(f"Total distance: {total_distance:.2f}")
        print(f"Average segment distance: {total_distance/(len(self.cities)):.2f}")

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
