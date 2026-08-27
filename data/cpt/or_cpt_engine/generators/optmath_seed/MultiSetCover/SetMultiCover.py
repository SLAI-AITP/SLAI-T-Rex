import gurobipy as gp
from gurobipy import GRB
import random

from or_cpt_engine.generators.numeric import range_uniform_rounded

class Generator:
    def __init__(self, parameters=None, seed=None):
        """
        Initialize Set Multi-Cover optimization problem.
        Parameters:
            parameters (dict): Dictionary containing:
                - n_sets: Number of sets
                - n_elements: Number of elements
                - density: Density of set-element associations (between 0 and 1)
                - coverage_range: Tuple of (min, max) for coverage requirements
                - cost_range: Tuple of (min, max) for set costs
            seed (int, optional): Random seed for reproducibility
        """
        self.problem_type = "set_multi_cover"
        self.mathematical_formulation = r"""
        \begin{align*}
        \min & \sum_{i=1}^{n} c_i x_i \\
        \text{s.t.} & \sum_{i \in I_j} x_i \geq r_j \quad \forall j \in M \\
        & x_i \in \{0,1\} \quad \forall i \in N
        \end{align*}
        """
        
        default_parameters = {
            # Keep the full coverage matrix compact enough for the LLM stages.
            # Truncated coverage data is the main source of infeasible repairs.
            "n_sets": (8, 14),
            "n_elements": (8, 16),
            "density": (0.35, 0.60),
            "coverage_range": (1, 3),
            "cost_range": (1, 20),
            "max_generation_attempts": 25,
        }
        
        parameters = {**default_parameters, **dict(parameters or {})}
            
        # Set attributes from parameters
        for key, value in parameters.items():
            setattr(self, key, value)
        self.parameters = dict(parameters)
            
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)

    def generate_instance(self):
        """
        Generate a Set Multi-Cover problem instance and create its corresponding Gurobi model.
        Returns:
            gp.Model: Configured Gurobi model for the set multi-cover problem
        """
        
        self._n_sets_range = _as_range(self.n_sets)
        self._n_elements_range = _as_range(self.n_elements)
        last_model = None
        last_payload = None
        for _ in range(getattr(self, "max_generation_attempts", 25)):
            model, elements, sets, costs, coverage_requirements = self._build_candidate_model()
            last_model = model
            last_payload = (elements, sets, costs, coverage_requirements)
            model.optimize()
            selected_count = sum(var.X > 0.5 for var in model.getVars() if var.VarName.startswith("Selected"))
            if 2 <= selected_count <= self.n_sets - 1:
                self._record_instance_metadata(elements, sets, costs, coverage_requirements, selected_count)
                return model
        if last_payload is not None:
            elements, sets, costs, coverage_requirements = last_payload
            self._record_instance_metadata(elements, sets, costs, coverage_requirements, None)
        return last_model

    def _record_instance_metadata(self, elements, sets, costs, coverage_requirements, selected_count):
        set_table = {
            f"set_{set_id}": {
                "cost": costs[set_id],
                "covered_elements": sorted(covered),
            }
            for set_id, covered in sets.items()
        }
        element_coverage_table = {
            element: {
                "required_coverage_count": coverage_requirements[element],
                "available_covering_set_count": sum(1 for set_id in sets if element in sets[set_id]),
            }
            for element in elements
        }
        coverage_matrix_summary = {
            "set_count": self.n_sets,
            "element_count": self.n_elements,
            "one_entries": sum(len(covered) for covered in sets.values()),
            "density": round(sum(len(covered) for covered in sets.values()) / (self.n_sets * self.n_elements), 2),
            "all_elements_have_requirement_below_available_count": all(
                coverage_requirements[element] < element_coverage_table[element]["available_covering_set_count"]
                for element in elements
            ),
        }
        payload = {
            "n_sets": self.n_sets,
            "n_elements": self.n_elements,
            "candidate_sets": [f"set_{set_id}" for set_id in sets],
            "elements": list(elements),
            "set_table": set_table,
            "element_coverage_table": element_coverage_table,
            "coverage_matrix_summary": coverage_matrix_summary,
            "compact_multisetcover_tables": {
                "sets": {
                    "candidate_sets": [f"set_{set_id}" for set_id in sets],
                    "elements": list(elements),
                },
                "candidate_set_table": {
                    "columns": ["candidate_set", "selection_cost"],
                    "rows": [
                        [f"set_{set_id}", costs[set_id]]
                        for set_id in sets
                    ],
                },
                "element_requirement_table": {
                    "columns": ["element", "required_coverage_count", "available_covering_set_count"],
                    "rows": [
                        [
                            element,
                            coverage_requirements[element],
                            element_coverage_table[element]["available_covering_set_count"],
                        ]
                        for element in elements
                    ],
                },
                "coverage_membership_table": {
                    "columns": ["candidate_set", "covered_elements"],
                    "rows": [
                        [f"set_{set_id}", sorted(covered)]
                        for set_id, covered in sets.items()
                    ],
                },
                "required_decision_layers": [
                    "Selected[i] binary candidate-set selection variable",
                ],
            },
            "decision_variables": {
                "Selected[i]": "binary; 1 if candidate set i is selected, 0 otherwise",
            },
            "objective_terms": {
                "sense": "minimize",
                "selected_set_cost": "sum_i cost[i] * Selected[i]",
            },
            "required_constraints": [
                "element_multicover_requirement",
                "binary_set_selection",
            ],
            "required_parameter_presentation": [
                "list every candidate set and its selection cost",
                "list every element and its required coverage count",
                "list the complete set-to-element coverage membership",
                "state Selected[i] is binary",
                "state each element must be covered by at least its required coverage count",
                "state requirements are lower bounds, not exact coverage counts",
                "state every element has more available covering sets than its requirement",
            ],
            "structured_problem_data": {
                "sets": {
                    "candidate_sets": [f"set_{set_id}" for set_id in sets],
                    "elements": list(elements),
                },
                "parameters": {
                    "set_table": "see top-level complete set_table",
                    "element_coverage_table": "see top-level element_coverage_table",
                },
                "variables": {
                    "Selected[i]": "binary set-selection variable",
                },
                "objective": "minimize total cost of selected sets",
                "constraints": {
                    "multicover": "for every element e, sum of Selected[i] over sets covering e >= required_coverage_count[e]",
                },
            },
            "business_interpretation_guardrails": [
                "This is a set multicover model, not ordinary single-cover set cover.",
                "Do not omit any candidate set or any element from the coverage matrix.",
                "Do not invent coverage memberships not listed in set_table.",
                "Do not replace >= required coverage with exact equality.",
                "Do not make Selected[i] continuous or integer counts; it is binary.",
                "Do not add routing, assignment, capacity, scheduling, or flow constraints.",
            ],
        }
        if selected_count is not None:
            payload["reference_selected_count"] = int(selected_count)
        self.parameters.update(payload)

    def _build_candidate_model(self):
        self.n_sets = random.randint(*self._n_sets_range)
        self.n_elements = random.randint(*self._n_elements_range)
        elements = [f"e{j}" for j in range(1, self.n_elements + 1)]
        density = _sample_density(self.density)
        min_cover = min(self.n_sets, max(3, int(self.coverage_range[0]) + 1))
        max_cover = max(min_cover, min(self.n_sets, int(round(self.n_sets * density))))
        sets = {i: set() for i in range(1, self.n_sets + 1)}
        coverage_requirements = {}
        for e in elements:
            cover_count = random.randint(min_cover, max_cover)
            candidate_sets = random.sample(list(sets.keys()), cover_count)
            for set_id in candidate_sets:
                sets[set_id].add(e)
            max_requirement = min(int(self.coverage_range[1]), max(1, cover_count - 1))
            min_requirement = min(int(self.coverage_range[0]), max_requirement)
            coverage_requirements[e] = random.randint(min_requirement, max_requirement)
        for set_id, covered in sets.items():
            if not covered:
                covered.add(random.choice(elements))
        costs = {i: random.randint(*self.cost_range) for i in range(1, self.n_sets + 1)}
        model = gp.Model("SetMultiCover")
        model.Params.OutputFlag = 0
        x = model.addVars(sets.keys(), vtype=GRB.BINARY, name="Selected")
        model.setObjective(gp.quicksum(costs[i] * x[i] for i in sets.keys()), GRB.MINIMIZE)
        for e in elements:
            model.addConstr(
                gp.quicksum(x[i] for i in sets.keys() if e in sets[i]) >= coverage_requirements[e],
                f"MultiCover_{e}",
            )
        return model, elements, sets, costs, coverage_requirements


def _sample_density(value):
    return range_uniform_rounded(value)


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
        
        model.write("set_multi_cover.lp")
        
        print(f"\nTest with default parameters:")
        print(f"Solve Time: {solve_time:.2f} seconds")
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal Value: {model.ObjVal:.2f}")
        else:
            print("No optimal solution found")
    
    test_generator()
