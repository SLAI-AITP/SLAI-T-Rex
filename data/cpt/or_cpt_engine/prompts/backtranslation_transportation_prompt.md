You are converting a verified balanced transportation LP into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 800 and 2200 characters whenever possible.
- Use one business paragraph followed by compact tables or indexed lists.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve the complete origin supply table, destination demand table, and lane cost matrix if available.
- Do not reveal solver status, optimal objective value, selected shipment quantities, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as a business wrapper. Do not add structures absent from the source.

Mathematical rules for `optmath_transp`:
- Describe a balanced transportation LP with continuous nonnegative `Transport[i,j]` shipment quantities.
- Preserve every origin and its exact supply.
- Preserve every destination and its exact demand.
- Preserve the complete origin-destination lane cost matrix.
- State that total supply equals total demand.
- State the objective is to minimize total lane shipping cost.
- State each origin ships exactly its supply and each destination receives exactly its demand.
- Do not add vehicles, routes, depot tours, subtour constraints, service times, time windows, lane capacities, fixed charges, binary lane activation, facility opening, unmet-demand slack, or optional unused supply.
- Do not mention any internally generated feasible shipment table.

Verified instance:

```json
{{INSTANCE_JSON}}
```
