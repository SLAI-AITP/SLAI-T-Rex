import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Revenue Management optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_resources: Number of flight legs
                - n_packages: Number of travel packages
                - capacity_range: Tuple of (min, max) for flight capacities
                - demand_range: Tuple of (min, max) for package demands
                - revenue_range: Tuple of (min, max) for package revenues
                - resource_use_probability: Probability of a package using a resource
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "revenue_management"
        self.mathematical_formulation = r"""
        ### Mathematical Model for Revenue Management
        Sets:
        - R: set of capacity-constrained resources
        - P: set of products or packages

        Parameters:
        - cap_r: available capacity of resource r
        - demand_p: maximum demand for package p
        - revenue_p: revenue from accepting one unit of package p
        - use_{p,r}: 1 if package p consumes one unit of resource r, otherwise 0

        Variables:
        - x_p: integer number of accepted bookings or sales of package p

        $$
        \begin{aligned}
        &\text{Maximize} && \sum_{p \in P} revenue_p x_p \\
        &\text{Subject to} && x_p \leq demand_p && \forall p \in P \\
        & && \sum_{p \in P} use_{p,r} x_p \leq cap_r && \forall r \in R \\
        & && x_p \geq 0,\; x_p \in \mathbb{Z} && \forall p \in P
        \end{aligned}
        $$
        """
        default_parameters = {
            "n_resources": 3,
            "n_packages": (4, 6),
            "capacity_range": (150, 250),
            "demand_range": (50, 150),
            "revenue_range": (200, 800),
            "resource_use_probability": 0.3
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
        Generate a Revenue Management problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for revenue management
        """
        # Generate number of packages if range is provided
        if isinstance(self.n_packages, tuple):
            self.n_packages = random.randint(*self.n_packages)
        
        # Create sets
        self.resources = [f"resource_{r}" for r in range(self.n_resources)]
        self.packages = [f"package_{p}" for p in range(self.n_packages)]
        
        # Generate parameters
        capacities = {r: random.randint(*self.capacity_range) 
                     for r in self.resources}
        demands = {p: random.randint(*self.demand_range) 
                  for p in self.packages}
        revenues = {p: random.randint(*self.revenue_range) 
                   for p in self.packages}
        
        # Generate resource usage matrix (ensuring each package uses at least one resource)
        resource_usage = {(p,r): 0 for p in self.packages for r in self.resources}
        for p in self.packages:
            # Ensure at least one resource is used
            first_resource = random.choice(list(self.resources))
            resource_usage[p,first_resource] = 1
            # Randomly use additional resources
            for r in self.resources:
                if r != first_resource and random.random() < self.resource_use_probability:
                    resource_usage[p,r] = 1
        compact_revenue_management_tables = {
            "sets": {
                "resources": self.resources,
                "packages": self.packages,
            },
            "resource_capacity_table": [
                {
                    "resource": r,
                    "capacity": capacities[r],
                }
                for r in self.resources
            ],
            "package_table": [
                {
                    "package": p,
                    "demand_upper_bound": demands[p],
                    "revenue_per_unit": revenues[p],
                }
                for p in self.packages
            ],
            "package_resource_usage_matrix": {
                "columns": self.resources,
                "rows": [
                    {
                        "package": p,
                        "values": [resource_usage[p, r] for r in self.resources],
                    }
                    for p in self.packages
                ],
            },
            "decision_variable": "PackageSales[p] is a nonnegative integer number of accepted sales/bookings for package p",
            "objective": "maximize sum of revenue_per_unit[p] * PackageSales[p]",
            "constraints": [
                "for each package p, PackageSales[p] <= demand_upper_bound[p]",
                "for each resource r, sum_p usage[p,r] * PackageSales[p] <= capacity[r]",
            ],
            "source_contract_note": (
                "This is a resource-capacity revenue management integer program. "
                "It is not price optimization, dynamic booking control, or nonlinear demand modeling."
            ),
        }
        self.parameters.update(
            {
                "resources": self.resources,
                "packages": self.packages,
                "capacities": capacities,
                "demands": demands,
                "revenues": revenues,
                "resource_usage": {f"{p}|{r}": resource_usage[p, r] for p in self.packages for r in self.resources},
                "compact_revenue_management_tables": compact_revenue_management_tables,
                "decision_variables": {
                    "PackageSales[p]": "nonnegative integer accepted sales/bookings for package p",
                },
                "objective_terms": [
                    "package_revenue_times_package_sales",
                ],
                "required_constraints": [
                    "package_demand_upper_bound",
                    "resource_capacity_consumption_limit",
                    "nonnegative_integer_package_sales",
                ],
                "required_parameter_presentation": [
                    "list resources and capacity for every resource",
                    "list packages with demand upper bound and revenue per unit",
                    "list complete package-resource usage matrix",
                    "state PackageSales[p] is a nonnegative integer decision variable",
                    "state package sales cannot exceed demand upper bounds",
                    "state resource capacity is consumed according to the usage matrix",
                    "prefer compact_revenue_management_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "resources": self.resources,
                        "packages": self.packages,
                    },
                    "parameters": {
                        "capacity": capacities,
                        "demand_upper_bound": demands,
                        "revenue_per_unit": revenues,
                        "package_resource_usage_matrix": "see compact_revenue_management_tables.package_resource_usage_matrix",
                    },
                    "constraints": {
                        "demand_upper_bound": "PackageSales[p] cannot exceed demand[p]",
                        "resource_capacity": "sum_p usage[p,r] * PackageSales[p] is at most capacity[r]",
                    },
                    "objective": "maximize accepted package revenue",
                },
                "business_interpretation_guardrails": [
                    "This is a static revenue management capacity-allocation model, not dynamic pricing.",
                    "Do not introduce nonlinear demand curves, bid prices, booking periods, or stochastic arrivals.",
                    "Do not omit package demand upper bounds.",
                    "Do not omit the package-resource usage matrix or resource capacity constraints.",
                    "Do not change the objective from revenue maximization to cost minimization.",
                ],
            }
        )
        
        # Create Gurobi model
        model = gp.Model("RevenueManagement")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        x = model.addVars(self.packages, vtype=GRB.INTEGER, name="PackageSales")
        
        # Set objective: maximize total revenue
        model.setObjective(
            gp.quicksum(revenues[p] * x[p] for p in self.packages),
            GRB.MAXIMIZE
        )
        
        # Add demand constraints
        for p in self.packages:
            model.addConstr(
                x[p] <= demands[p],
                name=f"DemandLimit_{p}"
            )
        
        # Add capacity constraints
        for r in self.resources:
            model.addConstr(
                gp.quicksum(resource_usage[p,r] * x[p] for p in self.packages) <= capacities[r],
                name=f"CapacityLimit_{r}"
            )
        
        # Store problem data
        self.capacities = capacities
        self.demands = demands
        self.revenues = revenues
        self.resource_usage = resource_usage
        
        return model

if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("revenue_management.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Revenue: ${model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
