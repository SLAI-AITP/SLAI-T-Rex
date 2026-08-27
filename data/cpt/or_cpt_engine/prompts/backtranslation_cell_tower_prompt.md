You are converting a verified cell-tower budgeted coverage instance into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 800 and 2000 characters whenever possible.
- Use one business paragraph followed by compact tower, region, and coverage data.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve tower costs, region populations, the tower-region coverage matrix, and the budget if available.
- Do not reveal solver status, optimal objective value, selected towers, covered regions, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as a business wrapper. Do not add structures absent from the source.

Mathematical rules for `optmath_cell_tower`:
- Describe a budgeted maximum-coverage MILP.
- Decision variables: binary `Build[t]` for candidate tower sites and binary `Covered[r]` for regions.
- Preserve every tower setup cost.
- Preserve every region population.
- Preserve the complete tower-region coverage matrix.
- Preserve the total tower-building budget.
- State the objective is to maximize total population covered.
- State a region can be marked covered only if at least one selected tower covers it.
- State total selected-tower setup cost cannot exceed the budget.
- Do not require all regions to be covered.
- Do not introduce customer assignments, shipments, facility capacities, vehicles, routes, time windows, or a cost-minimization facility-location model.

Verified instance:

```json
{{INSTANCE_JSON}}
```
