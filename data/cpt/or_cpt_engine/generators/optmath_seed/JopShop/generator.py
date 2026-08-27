import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Job Shop Problem instance.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - num_jobs: Number of jobs
                - num_machines: Number of machines
                - operations_per_job: Number of operations per job
                - processing_time_range: Tuple (min_time, max_time) for processing times
            seed (int): Random seed for reproducibility
        """
        self.problem_type = "job_shop_scheduling"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Job Shop Scheduling

        Sets:
        - J: jobs.
        - O_j: ordered operations belonging to job j.
        - M: machines.
        - Pairs(m): unordered pairs of operations that require the same machine m.

        Parameters:
        - p[j,i]: processing time of operation i in job j.
        - machine[j,i]: required machine for operation i in job j.
        - H: a valid Big-M scheduling horizon.

        Decision variables:
        - S[j,i] >= 0: start time of operation i of job j.
        - C_max >= 0: makespan.
        - X[(j,i),(k,l)] in {0,1}: ordering binary for each pair of operations
          that require the same machine.

        Minimize:
            C_max

        Subject to:
        - Job precedence:
            S[j,i+1] >= S[j,i] + p[j,i] for consecutive operations in each job.
        - Machine non-overlap for each same-machine operation pair:
            S[a] + p[a] <= S[b] + H * (1 - X[a,b])
            S[b] + p[b] <= S[a] + H * X[a,b]
        - Makespan definition:
            C_max >= S[j,i] + p[j,i] for every operation.
        """
        default_parameters = {
            'num_jobs': (4,7),
            'num_machines': (3,5),
            'operations_per_job': (2, 4),
            'processing_time_range': (1, 8)
        }
        parameters = {**default_parameters, **dict(parameters or {})}
         
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_instance(self):
        """
        Generate a Job Shop Problem instance.
        
        Generates:
            - self.jobs: Dictionary of jobs, each with a list of operations
            - self.p: Processing times p_{j,i}
            - self.m: Machines m_{j,i}
        """
        self.jobs = {}
        self.p = {}  # Processing times
        self.m = {}  # Machines for operations
        
        self.num_jobs = int(np.random.randint(*self.num_jobs)) if isinstance(self.num_jobs, tuple) else int(self.num_jobs)
        self.num_machines = (
            int(np.random.randint(*self.num_machines))
            if isinstance(self.num_machines, tuple)
            else int(self.num_machines)
        )

        for j in range(self.num_jobs):
            job_operations = []
            num_ops = (
                int(np.random.randint(*self.operations_per_job))
                if isinstance(self.operations_per_job, tuple)
                else int(self.operations_per_job)
            )
            num_ops = max(1, min(num_ops, self.num_machines))
            if num_ops <= self.num_machines:
                machine_route = list(np.random.choice(self.num_machines, size=num_ops, replace=False))
            else:
                machine_route = list(np.random.choice(self.num_machines, size=num_ops, replace=True))
            for i in range(num_ops):
                op_id = (j, i)
                # Random processing time
                p_j_i = int(np.random.randint(*self.processing_time_range))
                # Machine required by this operation in the job route.
                m_j_i = int(machine_route[i])
                # Store
                self.p[op_id] = p_j_i
                self.m[op_id] = m_j_i
                job_operations.append(op_id)
            self.jobs[j] = job_operations  # List of operations for job j

        # Create a new model
        model = gp.Model("JobShopProblem")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Variables:
        # Start times S_{j,i}
        S = {}
        for op_id in self.p.keys():
            S[op_id] = model.addVar(lb=0.0, vtype= GRB.INTEGER, name=f"S_{op_id}")

        # Makespan C_max
        C_max = model.addVar(lb=0.0,vtype= GRB.INTEGER, name="C_max")

        # Binary variables X_{(j,i),(k,l)} for pairs of operations on same machine
        X = {}
        # For each machine, get operations that require it
        machine_ops = {}
        for m in range(self.num_machines):
            machine_ops[m] = []

        for op_id, m_j_i in self.m.items():
            machine_ops[m_j_i].append(op_id)

        # Keep Big-M tied to the generated processing horizon. A huge fixed Big-M
        # weakens the model and can create noisy integer-feasibility diagnostics.
        big_M = sum(self.p.values()) + max(self.p.values(), default=0)

        # Add precedence constraints within jobs
        for j, operations in self.jobs.items():
            num_ops = len(operations)
            for idx in range(num_ops - 1):
                op1 = operations[idx]
                op2 = operations[idx + 1]
                model.addConstr(S[op2] >= S[op1] + self.p[op1], name=f"prec_{op1}_{op2}")

        # Add machine capacity constraints
        # For each machine
        for m in range(self.num_machines):
            ops = machine_ops[m]
            for idx1 in range(len(ops)):
                op1 = ops[idx1]
                for idx2 in range(idx1 + 1, len(ops)):
                    op2 = ops[idx2]
                    # Need to decide order between op1 and op2
                    var_name = f"X_{op1}_{op2}"
                    # X_{op1_op2} = 1 if op1 precedes op2
                    X[op1, op2] = model.addVar(vtype=GRB.BINARY, name=var_name)
                    # Constraints:
                    model.addConstr(
                        S[op1] + self.p[op1] <= S[op2] + big_M * (1 - X[op1, op2]),
                        name=f"machine_{op1}_{op2}_1")
                    model.addConstr(
                        S[op2] + self.p[op2] <= S[op1] + big_M * (X[op1, op2]),
                        name=f"machine_{op1}_{op2}_2")
        # Makespan definition constraints
        for op_id in self.p.keys():
            model.addConstr(C_max >= S[op_id] + self.p[op_id], name=f"makespan_{op_id}")

        # Set objective
        model.setObjective(C_max, GRB.MINIMIZE)

        self.model = model
        self.vars = {'S': S, 'C_max': C_max, 'X': X}
        self.parameters.update(
            {
                "jobs": [f"job_{j}" for j in range(self.num_jobs)],
                "machines": [f"machine_{m}" for m in range(self.num_machines)],
                "operations_by_job": {
                    f"job_{j}": [f"operation_{i}" for _, i in operations]
                    for j, operations in self.jobs.items()
                },
                "machine_route": {
                    f"job_{j}|operation_{i}": f"machine_{self.m[j, i]}"
                    for j, operations in self.jobs.items() for _, i in operations
                },
                "processing_times": {
                    f"job_{j}|operation_{i}": int(self.p[j, i])
                    for j, operations in self.jobs.items() for _, i in operations
                },
                "machine_operation_sets": {
                    f"machine_{m}": [f"job_{j}|operation_{i}" for j, i in ops]
                    for m, ops in machine_ops.items()
                },
                "big_m": int(big_M),
                "objective": "minimize_makespan",
                "required_constraints": [
                    "within_job_operation_precedence",
                    "same_machine_pairwise_nonoverlap_with_binary_ordering",
                    "makespan_at_least_every_operation_completion_time",
                    "nonnegative_operation_start_times",
                    "binary_same_machine_ordering_variables",
                ],
                "business_interpretation_guardrails": [
                    "This is a job shop: each job has its own ordered machine route.",
                    "Do not convert it into a permutation flow shop with one common sequence on every machine.",
                    "Do not add due dates, release dates, or tardiness costs unless they are explicitly listed.",
                    "The objective is makespan, not assignment cost or total processing time.",
                ],
            }
        )

        return model

    def solve(self):
        """
        Solve the Job Shop Problem using Gurobi.
        
        Returns:
            model.Status, solve_time
        """
        try:
            # Solve the model
            start_time = time.time()
            self.model.optimize()
            solve_time = time.time() - start_time
            return self.model.Status, solve_time
        except gp.GurobiError as e:
            print(f"Error: {e}")
            return GRB.LOADED, 0.0

if __name__ == '__main__':
    ################# Parameters #################
    

    # Create and solve instance
    import random
    jsp = Generator(seed=random.randint(0, 1000))
    
    model = jsp.generate_instance()
    solve_status, solve_time = jsp.solve()

    # Print results
    status_map = {
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED: "Unbounded",
        GRB.INF_OR_UNBD: "Infeasible or Unbounded",
        GRB.LOADED: "Model Loaded",
        GRB.TIME_LIMIT: "Time Limit Reached"
    }

    print(f"Solve Status: {status_map.get(solve_status, solve_status)}")
    print(f"Solve Time: {solve_time:.2f} seconds")

    # If solved to optimality, print solution
    if solve_status == GRB.OPTIMAL:
        S = jsp.vars['S']
        C_max = jsp.vars['C_max']

        print(f"\nOptimal Makespan: {C_max.X}\n")
        for op_id, var in S.items():
            print(f"Start time of operation {op_id}: {var.X}")
