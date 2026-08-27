import gurobipy as gp
from gurobipy import GRB
import random

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Capacitated Facility Location optimization problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_facilities: Number of potential facilities
                - n_customers: Number of customers
                - fixed_cost_range: Tuple of (min, max) for facility fixed costs
                - transport_cost_range: Tuple of (min, max) for transportation costs
                - demand_range: Tuple of (min, max) for customer demands
                - capacity_range: Tuple of (min, max) for facility capacities
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "capacitated_facility_location"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Suppose we have:
        - Set of facilities J = {1,...,n}
        - Set of customers I = {1,...,m}
        - f_j: Fixed cost of opening facility j
        - c_{ij}: Cost of serving customer i from facility j
        - d_i: Demand of customer i
        - K_j: Capacity of facility j
        $$
        \begin{aligned}
        &\text{Minimize} && \sum_{j \in J} f_j y_j + \sum_{i \in I}\sum_{j \in J} c_{ij} x_{ij} \\
        &\text{Subject to} && \sum_{j \in J} x_{ij} = d_i && \forall i \in I \\
        & && x_{ij} \leq d_i y_j && \forall i \in I, j \in J \\
        & && \sum_{i \in I} x_{ij} \leq K_j y_j && \forall j \in J \\
        & && y_j \in \{0,1\} && \forall j \in J \\
        & && x_{ij} \geq 0 && \forall i \in I, j \in J
        \end{aligned}
        $$
        """
        default_parameters = {
            "n_facilities": (3, 3),
            "n_customers": (3, 3),
            "fixed_cost_range": (80000, 120000),
            "transport_cost_range": (10, 100),
            "demand_range": (300, 500),
            "capacity_range": (800, 1200)
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
        Generate a Capacitated Facility Location problem instance and create its Gurobi model.
        
        Returns:
            gp.Model: Configured Gurobi model
        """
        # Randomly select number of facilities and customers
        self.n_facilities = _sample_int(self.n_facilities)
        self.n_customers = _sample_int(self.n_customers)
        
        # Generate facilities and customers
        self.facilities = [f"facility_{j}" for j in range(self.n_facilities)]
        self.customers = [f"customer_{i}" for i in range(self.n_customers)]
        
        # Generate problem parameters
        self.fixed_costs = {j: random.randint(*self.fixed_cost_range) for j in self.facilities}
        self.transport_costs = {(i, j): random.randint(*self.transport_cost_range) 
                         for i in self.customers for j in self.facilities}
        self.demands = {i: random.randint(*self.demand_range) for i in self.customers}
        total_demand = sum(self.demands.values())
        self.capacities = {j: random.randint(*self.capacity_range) for j in self.facilities}
        if sum(self.capacities.values()) < total_demand:
            deficit = total_demand - sum(self.capacities.values())
            for facility in self.facilities:
                if deficit <= 0:
                    break
                bump = min(deficit, max(1, self.capacity_range[1] - self.capacities[facility]))
                self.capacities[facility] += bump
                deficit -= bump
            if deficit > 0:
                self.capacities[self.facilities[0]] += deficit
        compact_cflp_tables = {
            "sets": {
                "facilities": self.facilities,
                "customers": self.customers,
            },
            "facility_table": [
                {
                    "facility": facility,
                    "fixed_opening_cost": self.fixed_costs[facility],
                    "capacity": self.capacities[facility],
                }
                for facility in self.facilities
            ],
            "customer_demand_table": [
                {
                    "customer": customer,
                    "demand": self.demands[customer],
                }
                for customer in self.customers
            ],
            "transport_cost_matrix": {
                "columns": self.facilities,
                "rows": [
                    {
                        "customer": customer,
                        "values": [self.transport_costs[customer, facility] for facility in self.facilities],
                    }
                    for customer in self.customers
                ],
            },
            "decision_variables": {
                "FacilityOpen[j]": "binary, 1 if facility j is opened",
                "ShippedAmount[i,j]": "continuous nonnegative amount of customer i demand served from facility j",
            },
            "objective": "minimize fixed opening costs plus customer-facility transportation costs",
            "constraints": [
                "for each customer i, sum_j ShippedAmount[i,j] equals demand[i]",
                "for each customer-facility pair (i,j), ShippedAmount[i,j] <= demand[i] * FacilityOpen[j]",
                "for each facility j, sum_i ShippedAmount[i,j] <= capacity[j] * FacilityOpen[j]",
            ],
            "source_contract_note": (
                "This is a capacitated facility-location MILP with binary opening decisions and continuous shipment/assignment quantities. "
                "It is not pure transportation and not vehicle routing."
            ),
        }
        self.parameters.update(
            {
                "facilities": self.facilities,
                "customers": self.customers,
                "fixed_costs": self.fixed_costs,
                "demands": self.demands,
                "capacities": self.capacities,
                "compact_cflp_tables": compact_cflp_tables,
                "decision_variables": {
                    "FacilityOpen[j]": "binary decision to open candidate facility j",
                    "ShippedAmount[i,j]": "continuous nonnegative quantity of customer i demand served from facility j",
                },
                "objective_terms": [
                    "facility_fixed_opening_cost",
                    "customer_facility_transport_cost_times_shipped_amount",
                ],
                "required_constraints": [
                    "customer_demand_exactly_satisfied",
                    "ship_only_from_open_facility",
                    "open_facility_capacity_limit",
                ],
                "required_parameter_presentation": [
                    "list candidate facilities with fixed opening cost and capacity",
                    "list customers and demand for every customer",
                    "list complete customer-facility transportation cost matrix",
                    "state FacilityOpen[j] is binary",
                    "state ShippedAmount[i,j] is continuous and nonnegative",
                    "state customer demand is exactly satisfied",
                    "state shipments can occur only from open facilities",
                    "state facility capacity applies only when the facility is opened",
                    "prefer compact_cflp_tables when writing the natural-language data tables",
                ],
                "structured_problem_data": {
                    "sets": {
                        "facilities": self.facilities,
                        "customers": self.customers,
                    },
                    "parameters": {
                        "fixed_opening_cost": self.fixed_costs,
                        "facility_capacity": self.capacities,
                        "customer_demand": self.demands,
                        "transport_cost_matrix": "see compact_cflp_tables.transport_cost_matrix",
                    },
                    "constraints": {
                        "customer_demand": "sum_j ShippedAmount[i,j] equals demand[i]",
                        "open_before_ship": "ShippedAmount[i,j] is at most demand[i] times FacilityOpen[j]",
                        "capacity": "sum_i ShippedAmount[i,j] is at most capacity[j] times FacilityOpen[j]",
                    },
                    "objective": "minimize fixed opening plus variable serving cost",
                },
                "business_interpretation_guardrails": [
                    "This is a capacitated facility-location model, not a pure transportation model.",
                    "Do not remove fixed opening costs or FacilityOpen[j] binary decisions.",
                    "Do not omit ShippedAmount[i,j] <= demand[i] * FacilityOpen[j] linking.",
                    "Do not omit capacity[j] * FacilityOpen[j] capacity linking.",
                    "Do not add vehicles, tours, time windows, subtour constraints, or route sequencing.",
                ],
            }
        )

        # Create Gurobi model
        model = gp.Model("CapacitatedFacilityLocation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output

        # Create decision variables
        y = model.addVars(self.facilities, vtype=GRB.BINARY, name="FacilityOpen")
        x = model.addVars(self.customers, self.facilities, name="ShippedAmount")

        # Set objective: minimize total costs
        model.setObjective(
            gp.quicksum(self.fixed_costs[j] * y[j] for j in self.facilities) +
            gp.quicksum(self.transport_costs[i,j] * x[i,j] for i in self.customers for j in self.facilities),
            GRB.MINIMIZE
        )

        # Add demand constraints
        for i in self.customers:
            model.addConstr(
                gp.quicksum(x[i,j] for j in self.facilities) == self.demands[i],
                name=f"Demand_{i}"
            )

        # Add valid constraints
        for i in self.customers:
            for j in self.facilities:
                model.addConstr(
                    x[i,j] <= self.demands[i] * y[j],
                    name=f"Valid_{i}_{j}"
                )

        # Add capacity constraints
        for j in self.facilities:
            model.addConstr(
                gp.quicksum(x[i,j] for i in self.customers) <= self.capacities[j] * y[j],
                name=f"Capacity_{j}"
            )

        return model

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
        
        model.write("facility_location.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    test_generator()
