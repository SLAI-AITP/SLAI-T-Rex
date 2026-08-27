import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Flow Shop Scheduling optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_jobs: Number of jobs to schedule
                - n_machines: Number of machines in series
                - processing_time_range: Tuple of (min, max) for processing times
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "flow_shop_scheduling"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Permutation Flow Shop Scheduling

        Sets:
        - J: jobs.
        - P: sequence positions, one position for each job.
        - M: machines or processing stages in the fixed technological order.

        Parameters:
        - p[j,m]: processing time of job j on machine m.

        Decision variables:
        - x[j,p] in {0,1}: 1 if job j is placed in sequence position p.
        - t[p,m] >= 0: start time of the job in position p on machine m.
        - Cmax >= 0: completion time of the final shared sequence position on
          the final machine.

        Minimize the makespan:
            Cmax

        Subject to:
        - One job per sequence position: sum_j x[j,p] = 1 for every p.
        - One sequence position per job: sum_p x[j,p] = 1 for every j.
        - Stage precedence for each position:
            t[p,m+1] >= t[p,m] + sum_j p[j,m] * x[j,p].
        - Machine order between adjacent sequence positions:
            t[p+1,m] >= t[p,m] + sum_j p[j,m] * x[j,p].
        - Makespan completion bound:
            Cmax >= t[last_position,last_machine]
                    + sum_j p[j,last_machine] * x[j,last_position].
        - x is binary and t is nonnegative.

        This is a permutation flow shop: every machine processes jobs in the
        same sequence chosen by x[j,p].
        """
        default_parameters = {
            "n_jobs": (4, 6),
            "n_machines": 3,
            "processing_time_range": (1, 5)
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
        Generate a Flow Shop Scheduling problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for the flow shop scheduling problem
        """
        # Generate random number of jobs if range is provided
        if isinstance(self.n_jobs, tuple):
            self.n_jobs = random.randint(*self.n_jobs)
        
        # Create sets
        jobs = range(self.n_jobs)
        schedules = range(self.n_jobs)  # Sequence positions equal the number of jobs.
        machines = range(self.n_machines)
        
        # Generate processing times
        process_times = {(j,m): random.randint(*self.processing_time_range) 
                        for j in jobs for m in machines}
        
        # Create Gurobi model
        model = gp.Model("FlowShopScheduling")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        x = model.addVars(jobs, schedules, vtype=GRB.BINARY, name="JobSchedule")
        t = model.addVars(schedules, machines, lb=0, vtype=GRB.CONTINUOUS, name="StartTime")
        c_max = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Makespan")
        
        # Set objective: minimize the explicit makespan variable. Keeping this
        # variable visible makes downstream LLM reconstruction less likely to
        # mistake the final-machine processing-time coefficients for costs.
        model.setObjective(c_max, GRB.MINIMIZE)
        
        # Add constraints
        # One job per schedule position
        for s in schedules:
            model.addConstr(
                gp.quicksum(x[j,s] for j in jobs) == 1,
                name=f"OneJobPerSchedule_{s}"
            )
        
        # One schedule position per job
        for j in jobs:
            model.addConstr(
                gp.quicksum(x[j,s] for s in schedules) == 1,
                name=f"OneSchedulePerJob_{j}"
            )
        
        # Machine precedence constraints
        for s in schedules:
            for m in range(self.n_machines-1):
                model.addConstr(
                    t[s,m+1] >= t[s,m] + 
                    gp.quicksum(process_times[j,m] * x[j,s] for j in jobs),
                    name=f"MachinePrecedence_{s}_{m}"
                )
        
        # Job precedence constraints
        for s in range(self.n_jobs-1):
            for m in machines:
                model.addConstr(
                    t[s+1,m] >= t[s,m] + 
                    gp.quicksum(process_times[j,m] * x[j,s] for j in jobs),
                    name=f"JobPrecedence_{s}_{m}"
                )

        model.addConstr(
            c_max
            >= t[self.n_jobs - 1, self.n_machines - 1]
            + gp.quicksum(process_times[j, self.n_machines - 1] * x[j, self.n_jobs - 1] for j in jobs),
            name="MakespanCompletion",
        )

        self.process_times = process_times
        job_labels = [f"job_{j}" for j in jobs]
        position_labels = [f"position_{s}" for s in schedules]
        machine_labels = [f"machine_{m}" for m in machines]
        processing_time_table_by_job = {
            f"job_{j}": {f"machine_{m}": process_times[j, m] for m in machines}
            for j in jobs
        }
        processing_time_table_by_machine = {
            f"machine_{m}": {f"job_{j}": process_times[j, m] for j in jobs}
            for m in machines
        }
        last_position = f"position_{self.n_jobs - 1}"
        last_machine = f"machine_{self.n_machines - 1}"
        self.parameters.update(
            {
                "jobs": job_labels,
                "sequence_positions": position_labels,
                "machines": machine_labels,
                "processing_times": {
                    f"job_{j}|machine_{m}": process_times[j, m]
                    for j in jobs for m in machines
                },
                "processing_time_table_by_job": processing_time_table_by_job,
                "processing_time_table_by_machine": processing_time_table_by_machine,
                "machine_stage_order": machine_labels,
                "sequence_position_table": {
                    f"position_{s}": {
                        "position_index": s,
                        "meaning": "shared sequence position used on every machine",
                    }
                    for s in schedules
                },
                "last_sequence_position": last_position,
                "last_machine": last_machine,
                "objective": "minimize_makespan_completion_time_of_last_sequence_position_on_last_machine",
                "objective_terms": {
                    "sense": "minimize",
                    "primary_objective_variable": "Makespan",
                    "makespan_expression": (
                        f"Makespan is at least StartTime[{last_position},{last_machine}] plus the processing time "
                        f"on {last_machine} of whichever job is assigned to {last_position}"
                    ),
                    "lp_objective_note": (
                        "The processing-time coefficients appear in the MakespanCompletion constraint, not as "
                        "assignment costs, delay costs, or weighted completion costs."
                    ),
                },
                "decision_variables": {
                    "JobSchedule[j,k]": "binary; 1 if job j is assigned to shared sequence position k",
                    "StartTime[k,m]": "continuous nonnegative start time of the job in sequence position k on machine m",
                    "Makespan": "continuous nonnegative completion time of the final shared sequence position on the final machine",
                },
                "required_constraints": [
                    "one_job_per_sequence_position",
                    "one_sequence_position_per_job",
                    "same_sequence_used_on_every_machine",
                    "same_job_follows_machine_order",
                    "adjacent_sequence_positions_do_not_overlap_on_each_machine",
                    "makespan_completion_bound",
                    "binary_job_position_assignment_variables",
                    "nonnegative_start_time_variables",
                ],
                "required_parameter_presentation": [
                    "list all jobs",
                    "list all shared sequence positions, one position for each job",
                    "list machines in their fixed processing order",
                    "list processing time p[j,m] for every job-machine pair",
                    "state JobSchedule[j,k] is indexed only by job and shared sequence position, not by machine",
                    "state every machine uses the same job sequence",
                    "state StartTime[k,m] is the start time of the job in shared position k on machine m",
                    "state Makespan is an explicit variable minimized by the objective",
                    "state Makespan is bounded below by the completion time of the final sequence position on the final machine",
                ],
                "structured_problem_data": {
                    "sets": {
                        "jobs": job_labels,
                        "shared_sequence_positions": position_labels,
                        "ordered_machines": machine_labels,
                    },
                    "parameters": {
                        "processing_time_by_job_machine": processing_time_table_by_job,
                    },
                    "variables": {
                        "JobSchedule[j,k]": "binary shared permutation assignment",
                        "StartTime[k,m]": "continuous nonnegative start time",
                        "Makespan": "continuous nonnegative makespan variable",
                    },
                    "objective": "minimize Makespan",
                    "constraints": {
                        "one_job_per_position": "sum_j JobSchedule[j,k] = 1 for every shared sequence position",
                        "one_position_per_job": "sum_k JobSchedule[j,k] = 1 for every job",
                        "machine_stage_precedence": (
                            "StartTime[k,m+1] >= StartTime[k,m] + sum_j p[j,m] JobSchedule[j,k]"
                        ),
                        "same_machine_adjacent_position_ordering": (
                            "StartTime[k+1,m] >= StartTime[k,m] + sum_j p[j,m] JobSchedule[j,k]"
                        ),
                        "makespan_completion": (
                            "Makespan >= StartTime[last_position,last_machine] plus the last-machine processing time "
                            "of the job assigned to the last shared sequence position"
                        ),
                    },
                },
                "business_interpretation_guardrails": [
                    "This is a permutation flow shop: all machines use the same job sequence.",
                    "JobSchedule[j,k] is not machine-specific; do not create JobSchedule[j,k,m].",
                    "Do not allow each machine to choose an independent job sequence.",
                    "The objective is to minimize the explicit Makespan variable.",
                    "Final-machine processing times belong in the MakespanCompletion constraint, not in an assignment-cost objective.",
                    "Do not add due dates, release dates, machine choices, or unrelated staffing constraints.",
                    "The objective is makespan, not total processing cost or total tardiness.",
                ],
                "generation_notes": [
                    "Use the phrase shared sequence position rather than time slot when describing k.",
                    "Use an explicit Makespan variable to avoid misreading the final-position processing-time expression as an assignment-cost objective.",
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
        
        model.write("flow_shop.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Makespan: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
