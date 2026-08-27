You are solving an operations research modeling task.

Return exactly one JSON object with these keys:
- `modeling_explanation`: concise but explicit explanation of how business entities map to sets, parameters, decision variables, objective, and each constraint family.
- `math_model`: a readable mathematical formulation.
- `gurobipy_code`: complete self-contained Python code using gurobipy.
- `self_check`: a compact JSON object with `objective_sense`, `variable_families`, `constraint_families`, `objective_terms`, `preserved_guardrails`, `uses_external_files`, and `possible_risks`.

Modeling rules:
- Preserve the variable domain implied by the problem. Use binary variables for take-or-leave selection, assignment, and yes/no activation. Use continuous nonnegative variables for standard transportation shipment amounts, diet/blending ingredient amounts, flows, and divisible production quantities unless indivisibility is explicitly stated.
- Explain why the objective direction is minimize or maximize based on the business goal.
- Explain how each constraint family comes from the natural-language problem statement.
- If the problem gives a vector, matrix, or indexed table, use all entries exactly. Do not simplify, average, or omit coefficients.
- If `source_compact_data.available` is true, treat `source_compact_data.value` as the authoritative source-fact table. It overrides any inconsistent numeric wording in `problem_statement` or `structured_problem_data`.
- If `generator_profile.canonical_math_signature` is present, treat it as a hard modeling checklist. Preserve the listed variable types, objective sense, required tables, and core constraint families.
- If `family_contract` is present, treat `family_contract.answer_contract` as a hard family-level contract. It defines valid objective senses, required variable families, required constraint families, valid index policies, and forbidden changes for this OR task family.
- Do not introduce any forbidden changes listed in `generator_profile.canonical_math_signature.forbidden_changes`.
- Do not introduce any forbidden changes listed in `family_contract.answer_contract.forbidden_changes`.
- Use `generator_profile.concept_tags` to make the explanation more explicit, but do not add unsupported modeling concepts.
- Treat `modeling_guardrails` as hard constraints. Before writing code, verify that every listed guardrail is represented in the formulation and executable model.
- In `self_check.preserved_guardrails`, list the guardrails and family-contract requirements that are explicitly preserved. In `self_check.possible_risks`, list any uncertainty; do not hide uncertainty by inventing missing data.
- Use only declared sets, parameters, arcs, lanes, resources, periods, and compatibility/index relations. Do not create a complete graph, missing nested dictionary, synthetic capacity, or synthetic eligibility table unless it is explicitly present in the problem JSON.
- For facility-location style problems, preserve fixed opening/setup/operating costs, binary opening decisions, open-before-serve linking, demand satisfaction, and capacity constraints.
- For supply-chain, network-flow, and network-design problems, preserve flow balance/conservation, arc or echelon capacity, activation/linking logic, and all cost components.
- For portfolio problems, preserve budget, return, risk, allocation bounds, and any cardinality or selection structure stated in the problem.
- For routing/time-window problems, preserve visit-once logic, route continuity, capacity and time-window constraints when they are present.
- For transportation and network assignment problems, preserve the sign and direction of every supply, demand, and transshipment balance. Do not convert a flow-balance model into independent binary matching unless the problem explicitly says each item is assigned once.
- For lot-sizing and production-planning problems, preserve period-by-period inventory balance, production, setup or activation variables if present, holding/backlog costs, initial inventory, period demand, and capacity. Do not drop setup/fixed-charge terms or replace inventory recursion with independent period constraints.
- For scheduling, flow-shop, job-shop, and vehicle-routing problems, preserve sequencing, machine/station assignment, route continuity, visit-once/service constraints, time windows, and makespan/lateness/earliness objective definitions. Do not replace sequencing or routing structure with a loose capacity allocation model.
- For fixed-cost facility, contract-allocation, and activation models, include both variable assignment/flow cost and fixed open/activation/setup cost exactly once. Do not double-count fixed cost and do not omit it.
- For set-cover and multi-set-cover problems, preserve coverage multiplicity. If an item must be covered at least k times, do not weaken it to simple coverage.
- If a generated objective value would be correct only after omitting, flipping, or relaxing a required constraint, that is not acceptable. Build the faithful model, not an easier surrogate.

Rules for `gurobipy_code`:
- It must be complete and executable as a standalone script.
- It must not read external files or depend on network resources.
- Do not include placeholders such as TODO, placeholder, replace with, adjust if needed, path/to/file, or data.csv.
- It must create a `gurobipy.Model`, add all variables and constraints, set the objective, call `model.optimize()`, and leave the solved model object in a top-level variable named `model`.
- Suppress solver chatter with `model.Params.OutputFlag = 0`.
- Do not hard-code the known optimum unless it follows from the model solution.
- Set `self_check.uses_external_files` to `false`. If you cannot write a complete self-contained model from the provided problem data, return the most faithful self-contained model and note the uncertainty in `self_check.possible_risks` rather than inventing external data dependencies.

Pre-submit self-check:
- Confirm the objective sense is allowed by `family_contract.answer_contract.objective_sense_options`.
- Confirm the code uses the variable domains required by the family contract.
- Confirm every `family_contract.answer_contract.required_constraints` item is represented in either the math model or code.
- Confirm the code indexes only over data structures that are actually present in the problem JSON.
- Confirm every objective coefficient family is represented exactly once: fixed costs, variable costs, revenues, holding costs, backlog costs, setup costs, penalty costs, and travel/time costs.
- Confirm balance equations use the correct side and sign convention: inflow + production + initial inventory equals outflow + demand + ending inventory, or the equivalent algebraic form.
- Confirm there are no placeholder comments, external file reads, or schema guesses.

Problem:

```json
{{PROBLEM_JSON}}
```
