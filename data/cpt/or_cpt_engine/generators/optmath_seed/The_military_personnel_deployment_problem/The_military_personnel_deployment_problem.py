import gurobipy as gp
from gurobipy import GRB
import random


class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize the Military Personnel Deployment Problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_tasks: Number of tasks
                - n_skills: Number of skills
                - cost_range: Tuple of (min, max) for deployment costs
                - skill_requirement_range: Tuple of (min, max) for skill requirements per task
                - total_soldiers: Total number of soldiers available
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "military_personnel_deployment"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose there are tasks \(i \in T\) and skill pools \(s \in S\).

        Parameters:
        - \(c_{i,s}\): cost of deploying one soldier with skill \(s\) to task \(i\)
        - \(r_{i,s}\): minimum number of soldiers with skill \(s\) required by task \(i\)
        - \(R_i\): total staffing requirement for task \(i\)
        - \(A_s\): available soldiers in skill pool \(s\)
        - \(A\): total deployable soldiers

        Decision variable:
        - \(x_{i,s}\): soldiers with skill \(s\) assigned to task \(i\)

        Objective:
        \[
        \min \sum_{i \in T}\sum_{s \in S} c_{i,s} x_{i,s}
        \]

        Constraints:
        \[
        \sum_{i \in T}\sum_{s \in S} x_{i,s} \leq A
        \]
        \[
        \sum_{i \in T} x_{i,s} \leq A_s \quad \forall s \in S
        \]
        \[
        \sum_{s \in S} x_{i,s} \geq R_i \quad \forall i \in T
        \]
        \[
        x_{i,s} \geq r_{i,s} \quad \forall i \in T, s \in S
        \]
        \[
        x_{i,s} \in \mathbb{Z}_+
        \]
        """
        default_parameters = {
            "n_tasks": (3, 5),
            "n_skills": (2, 4),
            "cost_range": (1, 10),
            "skill_requirement_range": (1, 5),
            "total_soldiers": (80, 120)
        }
        # Use default parameters if none are provided or if an empty dict is given
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
        Generate a Military Personnel Deployment problem instance and create its corresponding Gurobi model.
        
        This method does two things:
        1. Generates random problem data (tasks, costs, skill requirements, total soldiers)
        2. Creates and returns a configured Gurobi model ready to solve
        
        Returns:
            gp.Model: Configured Gurobi model for the personnel deployment problem
        """
        
        # Randomly select number of tasks and skills
        self.n_tasks = random.randint(*self.n_tasks)
        self.n_skills = random.randint(*self.n_skills)
        
        # Generate tasks and skills
        tasks = [f"task_{i}" for i in range(self.n_tasks)]
        skills = [f"skill_{s}" for s in range(self.n_skills)]
        
        # Generate deployment costs for each task-skill pair.
        task_costs = {(task, skill): random.randint(*self.cost_range) for task in tasks for skill in skills}
        
        # Generate skill requirements for each task and skill
        skill_requirements = {
            (task, skill): random.randint(*self.skill_requirement_range)
            for task in tasks for skill in skills
        }
        task_extra_staff = {task: random.randint(1, max(2, self.n_skills)) for task in tasks}
        task_staffing_requirement = {
            task: sum(skill_requirements[task, skill] for skill in skills) + task_extra_staff[task]
            for task in tasks
        }
        hidden_extra_by_skill = {skill: 0 for skill in skills}
        for extra in task_extra_staff.values():
            for _ in range(extra):
                hidden_extra_by_skill[random.choice(skills)] += 1

        skill_availability = {}
        for skill in skills:
            required = sum(skill_requirements[task, skill] for task in tasks)
            skill_availability[skill] = required + hidden_extra_by_skill[skill] + random.randint(0, max(2, required // 2))
        required_total = sum(task_staffing_requirement.values())
        total_upper = max(self.total_soldiers)
        total_soldiers = min(total_upper, required_total + random.randint(0, max(3, required_total // 3)))
        total_soldiers = max(required_total, total_soldiers)
        self.parameters.update(
            {
                "tasks": tasks,
                "skills": skills,
                "task_skill_costs": {f"{task}|{skill}": task_costs[task, skill] for task in tasks for skill in skills},
                "skill_requirements": {f"{task}|{skill}": skill_requirements[task, skill] for task in tasks for skill in skills},
                "task_staffing_requirement": task_staffing_requirement,
                "skill_availability": skill_availability,
                "total_soldiers": total_soldiers,
            }
        )

        # Create Gurobi model
        model = gp.Model("MilitaryPersonnelDeployment")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create integer decision variables (x[i,s] = soldiers with skill s assigned to task i)
        x = model.addVars(tasks, skills, vtype=GRB.INTEGER, name="Soldiers")

        # Set objective: minimize total deployment cost
        model.setObjective(
            gp.quicksum(task_costs[i, s] * x[i, s] for i in tasks for s in skills),
            GRB.MINIMIZE
        )

        # Add constraints:
        # 1. Total number of soldiers deployed must not exceed available soldiers
        model.addConstr(
            gp.quicksum(x[i, s] for i in tasks for s in skills) <= total_soldiers,
            name="TotalSoldiers"
        )

        # 2. Each skill pool cannot be over-deployed
        for s in skills:
            model.addConstr(
                gp.quicksum(x[i, s] for i in tasks) <= skill_availability[s],
                name=f"SkillAvailability_{s}"
            )

        # 3. Each task must meet total staffing and skill-specific requirements
        for i in tasks:
            model.addConstr(
                gp.quicksum(x[i, s] for s in skills) >= task_staffing_requirement[i],
                name=f"TaskStaffing_{i}"
            )
        for i in tasks:
            for s in skills:
                model.addConstr(
                    x[i, s] >= skill_requirements[i, s],
                    name=f"SkillRequirement_{i}_{s}"
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
        
        model.write("military_personnel_deployment.lp")

        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")

    test_generator()
