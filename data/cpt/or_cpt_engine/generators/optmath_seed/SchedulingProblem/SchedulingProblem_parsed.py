import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize a staff shift coverage optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_restaurants: Number of restaurants
                - n_employees: Number of employees
                - n_shifts: Number of shifts
                - n_skills: Number of skills
                - skill_probability: Probability of employee having a skill
                - availability_probability: Probability of employee being available for a shift
                - preference_range: Tuple of (min, max) for preference costs
                - unfulfilled_cost: Cost of unfulfilled position
                - extra_shortage_slots_range: Optional extra demand slots not guaranteed by the latent roster
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "staff_shift_coverage_assignment"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Staff Shift Coverage Assignment

        Sets:
        - R: work sites or restaurants.
        - E: employees.
        - S: shifts.
        - K: skill categories.

        Parameters:
        - d[r,s,k]: required number of employees with skill k at site r in shift s.
        - a[e,s] in {0,1}: whether employee e is available for shift s.
        - h[e,k] in {0,1}: whether employee e has skill k.
        - c[e,k]: preference or mismatch cost for assigning employee e to work skill k.
        - w: penalty cost per unfilled required position.

        Decision variables:
        - x[r,e,s,k] in {0,1}: 1 if employee e is assigned to site r, shift s,
          using skill k.
        - u[r,s,k] >= 0 integer: number of unfilled positions for site r, shift s,
          and skill k.

        Minimize:
            sum_{r,s,k} w * u[r,s,k] + sum_{r,e,s,k} c[e,k] * x[r,e,s,k]

        Subject to:
        - Coverage with shortage slack:
            sum_e x[r,e,s,k] + u[r,s,k] = d[r,s,k] for all r,s,k.
        - Shift availability:
            sum_{r,k} x[r,e,s,k] <= a[e,s] for all e,s.
        - Skill eligibility:
            x[r,e,s,k] <= h[e,k] for all r,e,s,k.
        - At most one total assignment per employee:
            sum_{r,s,k} x[r,e,s,k] <= 1 for all e.
        """
        default_parameters = {
            "n_restaurants": (2, 3),
            "n_employees": (6, 9),
            "n_shifts": 2,
            "n_skills": 2,
            "skill_probability": 0.7,
            "availability_probability": 0.8,
            "preference_range": (1, 5),
            "unfulfilled_cost": 100,
            "extra_shortage_slots_range": (0, 1),
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
        Generate an Employee Assignment problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for employee assignment
        """
        self.n_restaurants = _sample_int(self.n_restaurants)
        self.n_employees = _sample_int(self.n_employees)
        self.n_shifts = _sample_int(self.n_shifts)
        self.n_skills = _sample_int(self.n_skills)

        # Create named sets so downstream LLM prompts and Gurobi code share the
        # same index labels. This reduces off-by-one and table-remapping errors.
        restaurants = [f"site_{r}" for r in range(self.n_restaurants)]
        employees = [f"employee_{e}" for e in range(self.n_employees)]
        shifts = [f"shift_{s}" for s in range(self.n_shifts)]
        skills = [f"skill_{k}" for k in range(self.n_skills)]
        
        # Generate skill and availability data first, then derive mostly feasible
        # coverage demand from a latent roster. This keeps the shortage variable
        # meaningful without making the whole instance dominated by shortage cost.
        employee_has_skill = {
            (e, k): 1 if random.random() < self.skill_probability else 0
            for e in employees for k in skills
        }
        for e in employees:
            if not any(employee_has_skill[e, k] for k in skills):
                employee_has_skill[e, random.choice(list(skills))] = 1
        for k in skills:
            if not any(employee_has_skill[e, k] for e in employees):
                employee_has_skill[random.choice(list(employees)), k] = 1

        employee_does_shift = {
            (e, s): 1 if random.random() < self.availability_probability else 0
            for e in employees for s in shifts
        }
        for e in employees:
            if not any(employee_does_shift[e, s] for s in shifts):
                employee_does_shift[e, random.choice(list(shifts))] = 1

        preference_cost = {
            (e, k): random.randint(*self.preference_range)
            for e in employees for k in skills
        }

        demand = {(r, s, k): 0 for r in restaurants for s in shifts for k in skills}
        latent_roster = []
        employees_in_random_order = list(employees)
        random.shuffle(employees_in_random_order)
        target_assignments = random.randint(max(1, self.n_employees // 2), self.n_employees)
        for e in employees_in_random_order[:target_assignments]:
            feasible_pairs = [
                (s, k)
                for s in shifts
                for k in skills
                if employee_does_shift[e, s] and employee_has_skill[e, k]
            ]
            if not feasible_pairs:
                continue
            s, k = random.choice(feasible_pairs)
            r = random.choice(restaurants)
            demand[r, s, k] += 1
            latent_roster.append({"employee": e, "restaurant": r, "shift": s, "skill": k})

        # Add a small number of extra demand slots so the unfulfilled-position
        # variable remains visible in some seeds while staying easy to explain.
        extra_min, extra_max = _as_range(self.extra_shortage_slots_range)
        extra_shortage_slots = random.randint(extra_min, extra_max)
        for _ in range(extra_shortage_slots):
            r = random.choice(restaurants)
            s = random.choice(shifts)
            k = random.choice(skills)
            demand[r, s, k] += 1

        if not any(demand.values()):
            demand[restaurants[0], shifts[0], skills[0]] = 1
        
        # Create Gurobi model
        model = gp.Model("EmployeeAssignment")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        x = model.addVars(restaurants, employees, shifts, skills, 
                         vtype=GRB.BINARY, name="Assignment")
        u = model.addVars(restaurants, shifts, skills, 
                         vtype=GRB.INTEGER, name="Unfulfilled")
        
        # Set objective: minimize total cost
        model.setObjective(
            gp.quicksum(self.unfulfilled_cost * u[r,s,k] 
                       for r in restaurants for s in shifts for k in skills) +
            gp.quicksum(preference_cost[e,k] * x[r,e,s,k] 
                       for r in restaurants for e in employees 
                       for s in shifts for k in skills),
            GRB.MINIMIZE
        )
        
        # Add constraints
        # Satisfy demand
        for r in restaurants:
            for s in shifts:
                for k in skills:
                    model.addConstr(
                        gp.quicksum(x[r,e,s,k] for e in employees) + u[r,s,k] == demand[r,s,k],
                        name=f"Demand_{r}_{s}_{k}"
                    )
        
        # Assignment satisfies shifts
        for e in employees:
            for s in shifts:
                model.addConstr(
                    gp.quicksum(x[r,e,s,k] for r in restaurants for k in skills) 
                    <= employee_does_shift[e,s],
                    name=f"ShiftAvail_{e}_{s}"
                )
        
        # Assignment satisfies skills
        for r in restaurants:
            for e in employees:
                for s in shifts:
                    for k in skills:
                        model.addConstr(
                            x[r,e,s,k] <= employee_has_skill[e,k],
                            name=f"SkillReq_{r}_{e}_{s}_{k}"
                        )
        
        # Maximum one shift per employee
        for e in employees:
            model.addConstr(
                gp.quicksum(x[r,e,s,k] 
                           for r in restaurants for s in shifts for k in skills) <= 1,
                name=f"OneShift_{e}"
            )
        
        # Store problem data
        self.demand = demand
        self.employee_has_skill = employee_has_skill
        self.employee_does_shift = employee_does_shift
        self.preference_cost = preference_cost
        self.parameters.update(
            {
                "restaurants": restaurants,
                "employees": employees,
                "shifts": shifts,
                "skills": skills,
                "demand": {
                    f"{r}|{s}|{k}": demand[r, s, k]
                    for r in restaurants for s in shifts for k in skills
                },
                "positive_demand": {
                    f"{r}|{s}|{k}": value
                    for (r, s, k), value in demand.items()
                    if value > 0
                },
                "employee_has_skill": {
                    f"{e}|{k}": employee_has_skill[e, k]
                    for e in employees for k in skills
                },
                "employee_availability": {
                    f"{e}|{s}": employee_does_shift[e, s]
                    for e in employees for s in shifts
                },
                "preference_cost": {
                    f"{e}|{k}": preference_cost[e, k]
                    for e in employees for k in skills
                },
                "qualified_employees_by_skill": {
                    k: [e for e in employees if employee_has_skill[e, k]]
                    for k in skills
                },
                "available_employees_by_shift": {
                    s: [e for e in employees if employee_does_shift[e, s]]
                    for s in shifts
                },
                "feasible_assignment_arc_count": sum(
                    1
                    for r in restaurants
                    for e in employees
                    for s in shifts
                    for k in skills
                    if employee_does_shift[e, s] and employee_has_skill[e, k]
                ),
                "unfulfilled_position_penalty": self.unfulfilled_cost,
                "extra_shortage_slots_added": extra_shortage_slots,
                "demand_generation_summary": {
                    "base_staff_slots_created": len(latent_roster),
                    "extra_shortage_slots_added": extra_shortage_slots,
                    "note": (
                        "Demand was generated from an internal staff sample plus a small number "
                        "of extra uncovered slots; the internal staff sample is not part of the "
                        "business problem statement."
                    ),
                },
                "total_staff_demand": sum(demand.values()),
                "decision_variables": {
                    "Assignment[site,employee,shift,skill]": "binary assignment to a site-shift-skill role",
                    "Unfulfilled[site,shift,skill]": "nonnegative integer unfilled positions",
                },
                "objective_terms": [
                    "employee_skill_preference_assignment_cost",
                    "unfilled_position_penalty",
                ],
                "required_constraints": [
                    "coverage_with_integer_shortage_slack",
                    "employee_shift_availability",
                    "employee_skill_eligibility",
                    "at_most_one_total_assignment_per_employee",
                    "binary_assignment_variables",
                    "integer_unfulfilled_position_variables",
                ],
                "required_parameter_presentation": [
                    "list sites, employees, shifts, and skills",
                    "list only positive site-shift-skill demand and state omitted combinations have zero demand",
                    "list employee availability by shift",
                    "list employee skill eligibility",
                    "list preference cost by employee and skill",
                    "state the unfilled-position penalty",
                    "state each employee can receive at most one total assignment",
                    "state this is staff coverage assignment with shortage slack, not machine sequencing",
                ],
                "structured_problem_data": {
                    "sets": {
                        "sites": restaurants,
                        "employees": employees,
                        "shifts": shifts,
                        "skills": skills,
                    },
                    "parameters": {
                        "positive_demand": {
                            f"{r}|{s}|{k}": value
                            for (r, s, k), value in demand.items()
                            if value > 0
                        },
                        "employee_availability": "see top-level employee_availability[employee|shift]",
                        "employee_skill_eligibility": "see top-level employee_has_skill[employee|skill]",
                        "preference_cost": "see top-level preference_cost[employee|skill]",
                        "unfulfilled_position_penalty": self.unfulfilled_cost,
                    },
                    "constraints": {
                        "coverage": "assigned employees plus unfilled positions equals demand for each site-shift-skill role",
                        "availability": "an employee can work a shift only if available for that shift",
                        "skill_eligibility": "an employee can cover a skill role only if qualified for that skill",
                        "one_assignment": "each employee receives at most one total assignment",
                    },
                    "objective": "minimize assignment preference cost plus unfilled-position penalties",
                },
                "business_interpretation_guardrails": [
                    "This is a staff coverage assignment model, not a machine sequencing model.",
                    "Do not introduce operation processing times, machine non-overlap, due dates, or routing constraints.",
                    "The objective minimizes assignment preference costs plus an explicit penalty for unfilled required positions.",
                    "Do not reveal or force any internal demand-generation staff sample; solve only from the listed demand, availability, skill, and cost tables.",
                ],
                "generation_notes": [
                    "The internal staff sample used to shape demand is generator-only bookkeeping.",
                    "The LLM-facing problem should present positive demand roles, availability, skills, preference costs, and the unfilled-position penalty.",
                ],
            }
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
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("employee_assignment.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Cost: ${model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
