import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np

from or_cpt_engine.generators.numeric import uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Structure Based Assignment Problem.
        
        Parameters:
            parameters (dict): Dictionary containing:
                - n_peaks: Number of peaks
                - n_acids: Number of amino acids
                - n_assignments: Number of required assignments
                - noe_density: Density of NOE relations (0-1)
                - nth: Distance threshold for NOE relations
                - cost_range: Tuple of (min, max) for assignment costs
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "structure_based_assignment"
        self.mathematical_formulation = r"""
        Min sum(i in P, j in A) c[i,j]*x[i,j]
        s.t.
        sum(i in P) x[i,j] <= 1 for all j in A
        sum(j in A) x[i,j] <= 1 for all i in P
        sum(i in P, j in A) x[i,j] = N
        x[i,j] + x[k,l] <= b[j,l] + 1 for all j,l in A, i in P, k in NOE[i]
        x[i,j] binary
        """
        
        default_parameters = {
            "n_peaks": 6,
            "n_acids": 8,
            "n_assignments": 5,
            "noe_density": 0.15,
            "nth": 5.0,
            "cost_range": (0.0, 1.0)
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
        Generate a Structure Based Assignment problem instance.
        
        Returns:
            gp.Model: Configured Gurobi model
        """
        # Create sets
        peaks = list(range(self.n_peaks))
        acids = list(range(self.n_acids))
        
        # Generate NOE relations for peaks
        noe_relations = {i: [] for i in peaks}
        for i in peaks:
            for j in peaks:
                if i != j and random.random() < self.noe_density:
                    noe_relations[i].append(j)
        if not any(noe_relations.values()) and self.n_peaks >= 2:
            noe_relations[0].append(1)
        
        # Generate distances between amino acids
        distances = {}
        for i in acids:
            for j in acids:
                if i != j:
                    distances[i,j] = uniform_rounded(2.0, 8.0)
                    distances[j,i] = distances[i,j]
                else:
                    distances[i,j] = 0.0
        
        # Generate binary distance indicators
        b = {(i,j): 1 if distances[i,j] < self.nth else 0 
             for i in acids for j in acids}
        
        # Generate assignment costs
        costs = {(i,j): uniform_rounded(*self.cost_range)
                for i in peaks for j in acids}
        self._record_instance_metadata(peaks, acids, noe_relations, distances, b, costs)
        
        # Create Gurobi model
        model = gp.Model("SBA_Problem")
        model.Params.OutputFlag = 0
        
        # Decision variables
        x = model.addVars(peaks, acids, vtype=GRB.BINARY, name="x")
        
        # Objective: minimize total assignment cost
        model.setObjective(
            gp.quicksum(costs[i,j] * x[i,j] for i in peaks for j in acids),
            GRB.MINIMIZE
        )
        
        # Constraints
        # Each amino acid gets at most one peak
        model.addConstrs(
            (gp.quicksum(x[i,j] for i in peaks) <= 1 for j in acids),
            name="amino_acid_assignment"
        )
        
        # Each peak gets at most one amino acid
        model.addConstrs(
            (gp.quicksum(x[i,j] for j in acids) <= 1 for i in peaks),
            name="peak_assignment"
        )
        
        # Total number of assignments
        model.addConstr(
            gp.quicksum(x[i,j] for i in peaks for j in acids) == self.n_assignments,
            name="total_assignments"
        )
        
        # NOE constraints
        for i in peaks:
            for k in noe_relations[i]:
                for j in acids:
                    for l in acids:
                        if j != l:
                            model.addConstr(
                                x[i,j] + x[k,l] <= b[j,l] + 1,
                                name=f"NOE_{i}_{k}_{j}_{l}"
                            )
        
        return model

    def _record_instance_metadata(self, peaks, acids, noe_relations, distances, compatibility, costs):
        noe_pairs = [
            [int(peak), int(related_peak)]
            for peak, related_peaks in noe_relations.items()
            for related_peak in related_peaks
        ]
        self.parameters.update(
            {
                "peaks": list(peaks),
                "amino_acids": list(acids),
                "required_assignment_count": self.n_assignments,
                "noe_relations": {str(peak): list(related_peaks) for peak, related_peaks in noe_relations.items()},
                "noe_relation_pairs": noe_pairs,
                "distance_threshold": self.nth,
                "assignment_cost": {f"{peak}|{acid}": costs[peak, acid] for peak in peaks for acid in acids},
                "acid_distance": {f"{left}|{right}": distances[left, right] for left in acids for right in acids},
                "acid_compatibility": {
                    f"{left}|{right}": compatibility[left, right]
                    for left in acids
                    for right in acids
                },
                "compact_structure_assignment_tables": {
                    "sets": {
                        "peaks": list(peaks),
                        "amino_acids": list(acids),
                    },
                    "required_assignment_count": self.n_assignments,
                    "distance_threshold": self.nth,
                    "assignment_cost_matrix": {
                        "columns": list(acids),
                        "rows": [
                            {
                                "peak": peak,
                                "values": [costs[peak, acid] for acid in acids],
                            }
                            for peak in peaks
                        ],
                    },
                    "acid_distance_matrix": {
                        "columns": list(acids),
                        "rows": [
                            {
                                "amino_acid": acid,
                                "values": [distances[acid, other] for other in acids],
                            }
                            for acid in acids
                        ],
                    },
                    "acid_compatibility_matrix": {
                        "columns": list(acids),
                        "rows": [
                            {
                                "amino_acid": acid,
                                "values": [compatibility[acid, other] for other in acids],
                            }
                            for acid in acids
                        ],
                    },
                    "noe_relation_pairs": noe_pairs,
                    "source_model_note": (
                        "Select exactly required_assignment_count binary peak-amino-acid assignments. "
                        "NOE-related peak pairs may only be assigned to amino-acid pairs marked compatible by the compatibility matrix."
                    ),
                },
                "decision_variables": {
                    "x[peak,acid]": "binary; 1 if peak is assigned to amino acid, 0 otherwise",
                },
                "objective_terms": [
                    "assignment_cost_times_binary_assignment",
                ],
                "required_constraints": [
                    "each_amino_acid_at_most_one_peak",
                    "each_peak_at_most_one_amino_acid",
                    "exact_total_assignment_count",
                    "noe_compatibility_pair_constraints",
                    "binary_peak_acid_assignment",
                ],
                "required_parameter_presentation": [
                    "list every peak and amino acid index",
                    "list the complete peak-amino-acid assignment cost matrix",
                    "list the required total assignment count",
                    "list all NOE-related peak pairs",
                    "list the complete amino-acid compatibility matrix or distance matrix with threshold",
                    "state x[peak,acid] is binary",
                ],
                "structured_problem_data": {
                    "sets": {
                        "peaks": list(peaks),
                        "amino_acids": list(acids),
                    },
                    "parameters": {
                        "assignment_cost_matrix": "see compact_structure_assignment_tables.assignment_cost_matrix",
                        "noe_relation_pairs": "see compact_structure_assignment_tables.noe_relation_pairs",
                        "acid_compatibility_matrix": "see compact_structure_assignment_tables.acid_compatibility_matrix",
                        "required_assignment_count": self.n_assignments,
                    },
                    "constraints": {
                        "amino_acid_assignment": "each amino acid receives at most one peak",
                        "peak_assignment": "each peak is assigned to at most one amino acid",
                        "total_assignments": "exactly N peak-acid assignments are selected",
                        "noe_compatibility": "if two peaks have an NOE relation, incompatible amino-acid pairs cannot both be selected",
                    },
                    "objective": "minimize total peak-amino-acid assignment cost",
                },
                "business_interpretation_guardrails": [
                    "This is a binary structure-based peak-to-amino-acid assignment problem.",
                    "Do not invent compatibility or distance values not listed in compact_structure_assignment_tables.",
                    "Do not replace the exact assignment count with optional assignment.",
                    "Do not reinterpret the NOE compatibility constraints as routes, flows, or capacity planning.",
                ],
            }
        )

if __name__ == '__main__':
    import time
    def test_generator():
        generator = Generator()
        model = generator.generate_instance()
        
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        
        model.write("sba_problem.lp")
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
            
    test_generator()
