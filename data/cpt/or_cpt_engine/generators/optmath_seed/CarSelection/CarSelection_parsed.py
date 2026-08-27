import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Car Selection Assignment problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_participants: Number of participants
                - n_cars: Number of cars
                - preference_density: Probability of a participant being interested in a car
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "car_selection"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Consider sets P of participants and C of cars with the following:
        - **Decision Variable**:
          - x_{p,c}: Binary variable indicating if participant p is assigned to car c
        - **Parameters**:
          - a_{p,c}: Binary parameter indicating if participant p is interested in car c
        
        $$
        \begin{aligned}
        &\text{Maximize} && \sum_{p \in P}\sum_{c \in C} x_{p,c} \\
        &\text{Subject to} && x_{p,c} \leq a_{p,c} && \forall p \in P, c \in C \\
        & && \sum_{c \in C} x_{p,c} \leq 1 && \forall p \in P \\
        & && \sum_{p \in P} x_{p,c} \leq 1 && \forall c \in C \\
        & && x_{p,c} \in \{0,1\} && \forall p \in P, c \in C
        \end{aligned}
        $$
        """
        default_parameters = {
            "n_participants": (6, 12),
            "n_cars": (5, 11),
            "preference_density": 0.38
        }
        
        if parameters is None or not parameters:
            parameters = default_parameters
        self.input_parameters = dict(parameters)
        self.parameters = {}
        for key, value in parameters.items():
            setattr(self, key, value)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Car Selection Assignment problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for the car selection problem
        """
        # Generate random number of participants and cars.
        n_participants = self._sample_int(self.n_participants)
        n_cars = self._sample_int(self.n_cars)
        self.n_participants = n_participants
        self.n_cars = n_cars
        
        # Create sets
        participants = [f"participant_{i}" for i in range(n_participants)]
        cars = [f"car_{i}" for i in range(n_cars)]
        
        # Generate random eligibility/preferences and repair isolated rows/columns
        # so the matching instance is useful rather than empty or one-sided.
        preferences = {(p, c): 1 if random.random() < self.preference_density else 0 
                     for p in participants for c in cars}
        for p in participants:
            if not any(preferences[p, c] for c in cars):
                preferences[p, random.choice(cars)] = 1
        for c in cars:
            if not any(preferences[p, c] for p in participants):
                preferences[random.choice(participants), c] = 1
        eligibility_rows = [
            {
                "participant": p,
                "eligible_cars": [c for c in cars if preferences[p, c] == 1],
            }
            for p in participants
        ]
        eligibility_matrix = {
            "rows": participants,
            "columns": cars,
            "values": [[preferences[p, c] for c in cars] for p in participants],
        }
        self.parameters = {
            "problem_type": self.problem_type,
            "input_parameters": self.input_parameters,
            "participants": participants,
            "cars": cars,
            "preferences": {f"{p}|{c}": preferences[p, c] for p in participants for c in cars},
            "compact_car_selection_tables": {
                "participant_table": [
                    {"participant": p, "eligible_car_count": sum(preferences[p, c] for c in cars)}
                    for p in participants
                ],
                "car_table": [
                    {"car": c, "eligible_participant_count": sum(preferences[p, c] for p in participants)}
                    for c in cars
                ],
                "eligibility_by_participant": eligibility_rows,
                "eligibility_matrix": eligibility_matrix,
                "objective": "maximize the number of assigned participant-car pairs",
                "decision_variables": [
                    "Assignments[p,c] is binary and equals 1 if participant p is assigned to car c"
                ],
                "constraints": [
                    "Assignments[p,c] <= eligibility[p,c] for every participant-car pair",
                    "each participant is assigned to at most one car",
                    "each car is assigned to at most one participant",
                ],
            },
            "structured_problem_data": {
                "sets": {"participants": participants, "cars": cars},
                "parameters": {"eligibility_matrix": eligibility_matrix},
                "objective_sense": "maximize",
                "variable_domain": "binary assignment",
            },
            "required_parameter_presentation": [
                "list every participant and every car",
                "list the complete participant-car eligibility matrix or equivalent eligible-car lists",
                "state Assignments[p,c] is binary",
                "state assignments are allowed only for eligible participant-car pairs",
                "state each participant can receive at most one car",
                "state each car can be assigned to at most one participant",
                "state the objective maximizes the number of assigned eligible pairs",
            ],
            "business_interpretation_guardrails": [
                "Do not introduce assignment scores, costs, travel distances, routing, or time windows.",
                "Do not require every participant or every car to be assigned; both sides have at-most-one constraints.",
                "Do not turn the model into a transportation flow or vehicle-routing problem.",
            ],
            "objective_terms": ["one unit of benefit for each eligible assignment selected"],
            "required_constraints": [
                "assignment_only_if_eligible",
                "participant_at_most_one_car",
                "car_at_most_one_participant",
            ],
        }
        
        # Create Gurobi model
        model = gp.Model("CarSelection")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create binary decision variables
        x = model.addVars(participants, cars, vtype=GRB.BINARY, name="Assignments")
        
        # Set objective: maximize total assignments
        model.setObjective(
            gp.quicksum(x[p,c] for p in participants for c in cars),
            GRB.MAXIMIZE
        )
        
        # Add constraints
        # Assignment only possible if participant is interested
        for p in participants:
            for c in cars:
                model.addConstr(x[p,c] <= preferences[p,c], name=f"Preference_{p}_{c}")
        
        # One car per participant at most
        for p in participants:
            model.addConstr(
                gp.quicksum(x[p,c] for c in cars) <= 1,
                name=f"OneCarPerParticipant_{p}"
            )
        
        # One participant per car at most
        for c in cars:
            model.addConstr(
                gp.quicksum(x[p,c] for p in participants) <= 1,
                name=f"OneParticipantPerCar_{c}"
            )
            
        return model

    @staticmethod
    def _sample_int(value):
        if isinstance(value, int):
            return value
        return random.randint(*value)

if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("car_selection.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
