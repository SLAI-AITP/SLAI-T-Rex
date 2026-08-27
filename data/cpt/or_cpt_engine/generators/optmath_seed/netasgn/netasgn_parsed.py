import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Project Assignment optimization problem.

        Parameters:
            parameters (dict): Dictionary containing:
                - n_people: Number of people
                - n_projects: Number of projects
                - supply_range: Tuple of (min, max) for available hours per person
                - demand_range: Tuple of (min, max) for required hours per project
                - cost_range: Tuple of (min, max) for cost per hour
                - limit_range: Tuple of (min, max) for person's max contribution to a project
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "project_assignment"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Continuous Project Assignment

        Sets:
        - \(I\): people or resources
        - \(J\): projects or requests

        Parameters:
        - \(S_i\): available hours from person \(i\)
        - \(D_j\): required hours for project \(j\)
        - \(c_{i,j}\): cost per assigned hour from person \(i\) to project \(j\)
        - \(u_{i,j}\): maximum hours person \(i\) can contribute to project \(j\)

        Decision variable:
        - \(x_{i,j} \ge 0\): hours assigned from person \(i\) to project \(j\)

        Objective:
        \[
        \min \sum_{i \in I}\sum_{j \in J} c_{i,j}x_{i,j}
        \]

        Constraints:
        \[
        \sum_{j \in J} x_{i,j} = S_i \quad \forall i \in I
        \]
        \[
        \sum_{i \in I} x_{i,j} = D_j \quad \forall j \in J
        \]
        \[
        0 \le x_{i,j} \le u_{i,j} \quad \forall i \in I, j \in J
        \]

        This is a continuous allocation model, not a one-to-one binary
        assignment model. Do not convert the source instance into binary
        matching unless the generated data explicitly says so.
        """
        default_parameters = {
            # Keep the full cost and upper-bound matrices compact enough for
            # the LLM stages; missing matrix entries invite schema guessing.
            "n_people": (5, 6),
            "n_projects": (5, 6),
            "allocation_range": (1, 6),
            "cost_range": (10, 50),
            "limit_slack_range": (2, 8),
            "assignments_per_person": (2, 4),
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
        Generate a Project Assignment problem instance and create its corresponding Gurobi model.

        This method does two things:
        1. Generates random problem data (people, projects, costs, supplies, demands, limits)
        2. Creates and returns a configured Gurobi model ready to solve

        Returns:
            gp.Model: Configured Gurobi model for the project assignment problem
        """
        
        # Randomly select number of people and projects
        self.n_people = _sample_int(self.n_people)
        self.n_projects = _sample_int(self.n_projects)
        
        # Define sets for people and projects
        people = [f"person_{i}" for i in range(self.n_people)]
        projects = [f"project_{j}" for j in range(self.n_projects)]
        
        # Build a latent feasible allocation first, then derive supply, demand,
        # and upper limits from it. This prevents the random upper-bound matrix
        # from making the balanced transportation-style assignment infeasible.
        latent_allocation = {(i, j): 0 for i in people for j in projects}
        min_assign, max_assign = _as_range(self.assignments_per_person)
        for person in people:
            assignment_count = min(self.n_projects, random.randint(min_assign, max_assign))
            for project in random.sample(projects, assignment_count):
                latent_allocation[person, project] += random.randint(*self.allocation_range)
        for project in projects:
            if sum(latent_allocation[person, project] for person in people) <= 0:
                person = random.choice(people)
                latent_allocation[person, project] += random.randint(*self.allocation_range)

        supply = {person: sum(latent_allocation[person, project] for project in projects) for person in people}
        demand = {project: sum(latent_allocation[person, project] for person in people) for project in projects}
        cost = {(i, j): random.randint(*self.cost_range) for i in people for j in projects}
        limit = {
            (i, j): latent_allocation[i, j] + random.randint(*self.limit_slack_range)
            for i in people
            for j in projects
        }
        pair_table_summary = {
            "entry_count": len(people) * len(projects),
            "fields": ["person", "project", "cost_per_hour", "max_contribution_hours"],
            "storage_note": "Complete pair data is stored in cost_per_hour and max_contribution_hours matrices.",
        }
        person_table = {
            person: {
                "available_hours": supply[person],
            }
            for person in people
        }
        project_table = {
            project: {
                "required_hours": demand[project],
            }
            for project in projects
        }
        compact_project_assignment_tables = {
            "model_family": "continuous_resource_assignment_hours",
            "sets": {
                "people": people,
                "projects": projects,
            },
            "person_supply_table": [
                {"person": person, "available_hours": supply[person]}
                for person in people
            ],
            "project_demand_table": [
                {"project": project, "required_hours": demand[project]}
                for project in projects
            ],
            "cost_per_hour_matrix": {
                "columns": projects,
                "rows": [
                    {"person": person, "values": [cost[person, project] for project in projects]}
                    for person in people
                ],
            },
            "max_contribution_hours_matrix": {
                "columns": projects,
                "rows": [
                    {"person": person, "values": [limit[person, project] for project in projects]}
                    for person in people
                ],
            },
            "total_supply_hours": sum(supply.values()),
            "total_demand_hours": sum(demand.values()),
            "source_model_note": (
                "Continuous nonnegative assignment hours. Person supply and project demand are equalities; "
                "every person-project upper bound is enforced. No binary one-to-one matching."
            ),
        }

        self.parameters.update(
            {
                "n_people": self.n_people,
                "n_projects": self.n_projects,
                "people": people,
                "projects": projects,
                "supply_hours": dict(supply),
                "demand_hours": dict(demand),
                "cost_per_hour": {
                    i: {j: cost[i, j] for j in projects}
                    for i in people
                },
                "max_contribution_hours": {
                    i: {j: limit[i, j] for j in projects}
                    for i in people
                },
                "total_supply_hours": sum(supply.values()),
                "total_demand_hours": sum(demand.values()),
                "person_table": person_table,
                "project_table": project_table,
                "person_project_pair_table": pair_table_summary,
                "compact_project_assignment_tables": compact_project_assignment_tables,
                "balance_summary": {
                    "total_supply_hours": sum(supply.values()),
                    "total_demand_hours": sum(demand.values()),
                    "balanced": sum(supply.values()) == sum(demand.values()),
                    "supply_constraints_are_equalities": True,
                    "demand_constraints_are_equalities": True,
                    "all_person_project_pairs_have_upper_bounds": True,
                },
                "decision_variables": {
                    "Assign[i,j]": "continuous nonnegative hours assigned from person/resource i to project/request j",
                },
                "objective_terms": {
                    "sense": "minimize",
                    "total_assignment_cost": "sum over all person-project pairs of cost_per_hour[i,j] * Assign[i,j]",
                },
                "required_constraints": [
                    "person_supply_hours_must_be_fully_allocated",
                    "project_demand_hours_must_be_fully_satisfied",
                    "person_project_contribution_upper_bounds",
                    "nonnegative_continuous_assignment_hours",
                ],
                "required_parameter_presentation": [
                    "list every person or resource and its available hours",
                    "list every project or request and its required hours",
                    "state total available hours equals total required hours",
                    "list cost per assigned hour for every person-project pair",
                    "list maximum contribution hours for every person-project pair",
                    "state Assign[i,j] is continuous nonnegative assigned hours, not binary matching",
                    "state every person's available hours must be fully allocated with equality",
                    "state every project's required hours must be exactly satisfied with equality",
                    "state all listed contribution upper bounds must be enforced",
                ],
                "structured_problem_data": {
                    "sets": {
                        "people": people,
                        "projects": projects,
                    },
                    "parameters": {
                        "supply_hours": dict(supply),
                        "demand_hours": dict(demand),
                        "cost_per_hour": "see top-level complete cost_per_hour matrix",
                        "max_contribution_hours": "see top-level complete max_contribution_hours matrix",
                        "total_supply_hours": sum(supply.values()),
                        "total_demand_hours": sum(demand.values()),
                    },
                    "variables": {
                        "Assign[i,j]": "continuous nonnegative assigned hours",
                    },
                    "objective": "minimize total cost per assigned hour times assigned hours",
                    "constraints": {
                        "supply_balance": "sum_j Assign[i,j] = supply_hours[i] for every person/resource",
                        "demand_balance": "sum_i Assign[i,j] = demand_hours[j] for every project/request",
                        "contribution_limit": "Assign[i,j] <= max_contribution_hours[i,j] for every pair",
                    },
                    "generation_note": "The instance was internally generated from a feasible allocation, but no allocation certificate should be included in the natural-language problem.",
                },
                "business_interpretation_guardrails": [
                    "Do not model this as binary one-to-one matching.",
                    "Decision variables are continuous assigned hours.",
                    "Every person's available hours and every project's required hours are equality constraints.",
                    "Do not relax supply equality to a less-than-or-equal capacity constraint.",
                    "Do not relax demand equality to greater-than-or-equal or optional coverage.",
                    "Do not omit or invent contribution limits; every person-project pair has a listed maximum contribution.",
                    "Do not add route continuity, sequencing, time windows, eligibility binaries, or vehicle paths.",
                    "Do not reveal or constrain the model to any internally used feasible allocation certificate.",
                ],
            }
        )

        assert sum(supply.values()) == sum(demand.values()), "Total supply and demand must be equal"
        
        # Create Gurobi model
        model = gp.Model("ProjectAssignment")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables (Assign_{i,j} = hours assigned)
        assign = model.addVars(people, projects, vtype=GRB.CONTINUOUS, name="Assign")
        
        # Set objective: Minimize total cost of assignment
        model.setObjective(
            gp.quicksum(cost[i, j] * assign[i, j] for i in people for j in projects),
            GRB.MINIMIZE
        )
        
        # Add supply constraints: every person's hours are fully allocated.
        model.addConstrs(
            (gp.quicksum(assign[i, j] for j in projects) == supply[i] for i in people),
            name="SupplyConstraint"
        )
        
        # Add demand constraints: every project's hours are exactly satisfied.
        model.addConstrs(
            (gp.quicksum(assign[i, j] for i in people) == demand[j] for j in projects),
            name="DemandConstraint"
        )
        
        # Add capacity constraints: Assigned hours cannot exceed limit for person-project pair
        model.addConstrs(
            (assign[i, j] <= limit[i, j] for i in people for j in projects),
            name="CapacityConstraint"
        )
        
        return model


def _sample_int(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)


def _as_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


if __name__ == '__main__':
    import time
    
    def test_generator():
        generator = Generator()  # Fix seed for reproducibility
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("project_assignment.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
