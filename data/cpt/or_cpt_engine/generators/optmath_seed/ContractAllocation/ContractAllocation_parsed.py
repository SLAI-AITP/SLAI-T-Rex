import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Contract Allocation problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_producers: Number of producers/factories
                - n_contracts: Number of contracts
                - capacity_range: Tuple of (min, max) for producer capacities
                - contract_size_range: Tuple of (min, max) for contract sizes
                - min_delivery_ratio: Ratio for minimum delivery size
                - cost_range: Tuple of (min, max) for production costs
                - min_contributors_range: Tuple of (min, max) for minimum contributors
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "contract_allocation"
        self.mathematical_formulation = r"""
        ### Mathematical Model
        Consider sets P of producers and C of contracts with the following:
        - **Decision Variables**:
          - x_{p,c}: Amount of commodity delivered by producer p for contract c
          - y_{p,c}: Binary variable indicating if producer p delivers to contract c
        - **Parameters**:
          - c_{p,c}: Unit production cost for producer p to fulfill contract c
          - cap_p: Available capacity of producer p
          - d_c: Size of contract c
          - m_p: Minimal delivery size for producer p
          - n_c: Minimal number of contributors for contract c

        Minimize:
            sum_{p,c} c_{p,c} x_{p,c}

        Subject to:
        - Producer capacity: sum_c x_{p,c} <= cap_p
        - Exact contract fulfillment: sum_p x_{p,c} = d_c
        - Minimum contributors: sum_p y_{p,c} >= n_c
        - Delivery activation lower bound: x_{p,c} >= m_p y_{p,c}
        - Delivery activation upper bound: x_{p,c} <= d_c y_{p,c}
        """
        default_parameters = {
            "n_producers": (3, 5),
            "n_contracts": (4, 6),
            "capacity_range": (600, 1400),
            "contract_size_range": (250, 900),
            "min_delivery_ratio": 0.03,
            "cost_range": (10, 50),
            "min_contributors_range": (1, 2)
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
        Generate a Contract Allocation problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model for the contract allocation problem
        """
        # Generate random numbers of producers and contracts
        self.n_producers = random.randint(*self.n_producers)
        self.n_contracts = random.randint(*self.n_contracts)
        
        # Create sets
        producers = [f"producer_{i}" for i in range(self.n_producers)]
        contracts = [f"contract_{i}" for i in range(self.n_contracts)]
        
        # Generate parameters
        capacities = {p: random.randint(*self.capacity_range) for p in producers}
        total_capacity = sum(capacities.values())
        raw_contract_weights = {c: random.random() + 0.2 for c in contracts}
        total_weight = sum(raw_contract_weights.values())
        target_total_demand = uniform_rounded(0.45, 0.75) * total_capacity
        contract_sizes = {
            c: max(self.contract_size_range[0], int(target_total_demand * raw_contract_weights[c] / total_weight))
            for c in contracts
        }
        min_deliveries = {p: int(capacities[p] * self.min_delivery_ratio) for p in producers}
        min_contributors = {
            c: random.randint(
                self.min_contributors_range[0],
                min(self.min_contributors_range[1], self.n_producers),
            )
            for c in contracts
        }
        production_costs = {(p,c): random.randint(*self.cost_range) 
                          for p in producers for c in contracts}
        
        # Create Gurobi model
        model = gp.Model("ContractAllocation")
        model.Params.OutputFlag = 0  # Suppress Gurobi output
        
        # Create decision variables
        x = model.addVars(producers, contracts, name="Generation", vtype=GRB.CONTINUOUS)
        y = model.addVars(producers, contracts, name="GenerationIncidence", vtype=GRB.BINARY)
        
        # Set objective: minimize total production cost
        model.setObjective(
            gp.quicksum(production_costs[p,c] * x[p,c] 
                       for p in producers for c in contracts),
            GRB.MINIMIZE
        )
        
        # Add constraints
        # Capacity constraints
        for p in producers:
            model.addConstr(
                gp.quicksum(x[p,c] for c in contracts) <= capacities[p],
                name=f"Capacity_{p}"
            )
        
        # Contract fulfillment constraints
        for c in contracts:
            model.addConstr(
                gp.quicksum(x[p,c] for p in producers) == contract_sizes[c],
                name=f"ContractFulfillment_{c}"
            )
        
        # Minimum contributors constraints
        for c in contracts:
            model.addConstr(
                gp.quicksum(y[p,c] for p in producers) >= min_contributors[c],
                name=f"MinContributors_{c}"
            )
        
        # Minimum delivery size constraints
        for p in producers:
            for c in contracts:
                model.addConstr(
                    x[p,c] >= min_deliveries[p] * y[p,c],
                    name=f"MinDelivery_{p}_{c}"
                )
                model.addConstr(
                    x[p,c] <= contract_sizes[c] * y[p,c],
                    name=f"DeliveryActivation_{p}_{c}"
                )
        self.capacities = capacities
        self.contract_sizes = contract_sizes
        self.min_deliveries = min_deliveries
        self.min_contributors = min_contributors
        self.production_costs = production_costs
        self.parameters.update(
            {
                "producers": producers,
                "contracts": contracts,
                "producer_capacities": capacities,
                "contract_sizes": contract_sizes,
                "minimum_delivery_by_producer": min_deliveries,
                "minimum_contributors_by_contract": min_contributors,
                "production_costs": {
                    f"{p}|{c}": production_costs[p, c]
                    for p in producers
                    for c in contracts
                },
                "compact_contract_allocation_tables": {
                    "sets": {
                        "producers": producers,
                        "contracts": contracts,
                    },
                    "producer_table": {
                        "columns": ["producer", "capacity", "minimum_delivery_if_used"],
                        "rows": [
                            [p, capacities[p], min_deliveries[p]]
                            for p in producers
                        ],
                    },
                    "contract_table": {
                        "columns": ["contract", "contract_size", "minimum_contributors"],
                        "rows": [
                            [c, contract_sizes[c], min_contributors[c]]
                            for c in contracts
                        ],
                    },
                    "production_cost_table": {
                        "columns": ["producer", "contract", "unit_production_cost"],
                        "rows": [
                            [p, c, production_costs[p, c]]
                            for p in producers
                            for c in contracts
                        ],
                    },
                    "required_decision_layers": [
                        "Generation[p,c] continuous nonnegative delivery quantity from producer p to contract c",
                        "GenerationIncidence[p,c] binary indicator that producer p contributes positive delivery to contract c",
                    ],
                },
                "total_capacity": total_capacity,
                "total_contract_demand": sum(contract_sizes.values()),
                "objective_terms": [
                    "unit_production_cost_times_delivered_quantity",
                ],
                "required_constraints": [
                    "producer_capacity",
                    "exact_contract_fulfillment",
                    "minimum_contributors_per_contract",
                    "delivery_activation_lower_bound_minimum_delivery",
                    "delivery_activation_upper_bound_contract_size",
                    "binary_producer_contract_incidence",
                    "nonnegative_continuous_delivery_quantity",
                ],
                "business_interpretation_guardrails": [
                    "This is a contract quantity allocation model, not a facility-location or routing model.",
                    "Do not omit the binary producer-contract incidence variables.",
                    "Do not weaken exact contract fulfillment into optional or partial fulfillment.",
                    "Each positive delivery must respect the producer-specific minimum delivery size.",
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
        
        model.write("contract_allocation.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Cost: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
