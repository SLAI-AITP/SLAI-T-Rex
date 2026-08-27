You are converting a verified capacitated facility-location instance into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 900 and 2400 characters whenever possible.
- Use one business paragraph followed by compact tables or indexed lists.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve complete facility, customer-demand, and customer-facility cost tables if available.
- Do not reveal solver status, optimal objective value, selected facilities, shipment values, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as a business wrapper. Do not add structures absent from the source.

Mathematical rules for `optmath_cflp`:
- Describe a capacitated facility-location MILP.
- Decision variables: binary `FacilityOpen[j]` and continuous nonnegative `ShippedAmount[i,j]`.
- Preserve facility fixed opening costs and capacities.
- Preserve customer demand values.
- Preserve the complete customer-facility transportation cost matrix.
- State the objective is to minimize fixed opening costs plus variable customer-facility service costs.
- State each customer's demand is exactly satisfied.
- State shipments can occur only from opened facilities.
- State facility capacity is available only if the facility is opened.
- Do not reduce the model to pure transportation.
- Do not add vehicles, tours, subtours, service times, time windows, inventory periods, or production decisions.

Verified instance:

```json
{{INSTANCE_JSON}}
```
