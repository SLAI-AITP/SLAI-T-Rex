You are converting a verified static revenue-management capacity-allocation instance into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 800 and 2200 characters whenever possible.
- Use one business paragraph followed by compact tables or indexed lists.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve resource capacities, package revenue/demand bounds, and the package-resource usage matrix if available.
- Do not reveal solver status, optimal objective value, accepted sales values, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as a business wrapper. Do not add structures absent from the source.

Mathematical rules for `optmath_revenue_management` and `optmath_revenue`:
- Describe a static resource-capacity revenue-management integer program.
- Decision variable: nonnegative integer `PackageSales[p]`, the accepted sales or bookings for package `p`.
- Preserve every resource capacity.
- Preserve every package's demand upper bound and revenue per unit.
- Preserve the complete package-resource usage matrix.
- State the objective is to maximize total accepted package revenue.
- State package sales cannot exceed package demand upper bounds.
- State each resource capacity is consumed according to the usage matrix.
- Do not introduce dynamic pricing, booking periods, stochastic arrivals, nonlinear demand curves, bid-price controls, overbooking recapture, customer choice modeling, or cost minimization.

Verified instance:

```json
{{INSTANCE_JSON}}
```
