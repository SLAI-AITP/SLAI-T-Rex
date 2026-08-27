You are repairing a failed operations-research forward model.

Return exactly one JSON object with these keys:
- `modeling_explanation`: concise explanation of what was wrong and how the repaired model maps entities, variables, objective terms, and constraints.
- `math_model`: repaired mathematical formulation.
- `gurobipy_code`: complete self-contained Python code using gurobipy.
- `self_check`: compact JSON object with `objective_sense`, `variable_families`, `constraint_families`, `objective_terms`, `preserved_guardrails`, `repair_actions`, `uses_external_files`, and `possible_risks`.

Repair rules:
- Repair the formulation; do not merely patch syntax.
- Use the natural-language problem and structured data as the source of truth.
- The validation context may include a reference objective used by an independent solver. Do not hard-code that value, do not add dummy variables just to force that value, and do not create a trivial model that returns the target objective without representing the problem.
- Preserve every numeric table, vector, matrix, index set, cost, capacity, demand, time, compatibility, and eligibility relation present in the problem JSON.
- If `source_compact_data.available` is true, treat `source_compact_data.value` as the authoritative source-fact table. It overrides inconsistent numeric wording in `problem_statement` or `structured_problem_data`.
- Do not invent missing data, complete graphs, synthetic capacities, or synthetic eligibility tables.
- Do not read external files or depend on network resources.
- Do not include placeholders such as TODO, placeholder, replace with, adjust if needed, path/to/file, or data.csv.
- The code must create a top-level variable named `model`, set `model.Params.OutputFlag = 0`, call `model.optimize()`, and leave the solved model object available.

Family-specific repair guidance:
- Facility location / CFLP / contract allocation: preserve fixed opening or setup costs, binary open/activate variables, open-before-serve linking, assignment or shipment costs, demand satisfaction, and capacity limits. Do not turn the model into pure transportation unless there are no fixed activation decisions.
- Network flow / transportation / supply chain / netthru / netasgn: preserve conservation or balance signs exactly. Supply nodes, demand nodes, transshipment nodes, commodities, periods, and arcs must not be collapsed into one independent assignment problem.
- Lot sizing / production planning: preserve time-indexed inventory balance, production, setup/activation if present, holding/backlog costs, period demand, and capacity. Do not drop initial inventory or backlog terms if the statement gives them.
- Scheduling / flow shop / job shop / VRPTW: preserve sequencing, machine or station assignment, visit-once/service logic, precedence, time windows, and makespan/lateness objective definitions exactly. Do not replace sequencing with independent capacity constraints.
- Set cover / multi-set cover: preserve coverage multiplicity requirements, binary selection variables, and all coverage matrix entries.

Pre-submit self-check:
- Explain the failed validation reason and the concrete repair action.
- Confirm the objective sense and all objective terms.
- Confirm each required constraint family from `family_contract.answer_contract.required_constraints` is represented.
- Confirm the code indexes only over data structures present in the problem JSON.
- Confirm there are no placeholder comments, external file reads, dummy objective forcing, or schema guesses.

Repair input:

```json
{{REPAIR_JSON}}
```
