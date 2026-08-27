You are converting a verified aircraft-operations optimization instance into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 900 and 2400 characters whenever possible.
- Use one business paragraph followed by compact tables or indexed lists.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve complete tables if they are available.
- Do not reveal solver status, optimal objective value, selected decision values, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as the wrapper. Do not add business structures absent from the source.

If `generator_id` is `optmath_aircraftassignment`:
- Describe aircraft type-to-route fleet-capacity allocation.
- Decision variable: nonnegative integer `Allocation[a,r]`, the count of aircraft type `a` assigned to route `r`.
- Preserve aircraft-type availability, route demand, aircraft-route capacity contribution matrix, and aircraft-route operating cost matrix.
- State the objective is to minimize total operating cost.
- State each aircraft type cannot be used more than its available count.
- State each route's assigned capacity must meet or exceed demand.
- Do not turn this into binary one-aircraft matching, tail routing, runway landing sequencing, gate assignment, passenger spill/recapture, or time-expanded fleet balance.
- Do not mention any internal feasible allocation certificate.

If `generator_id` is `optmath_aircraftlanding`:
- Describe runway landing-time sequencing for a set of aircraft.
- Decision variables: continuous `Landing[i]`, binary pairwise `AircraftOrder[i,j]`, and nonnegative continuous `Early[i]` and `Late[i]`.
- Preserve earliest, target, and latest landing time for every aircraft.
- Preserve early and late penalty per minute for every aircraft.
- Preserve ordered-pair separation times.
- `problem_statement` itself must include the complete aircraft time/penalty table and the complete ordered-pair separation-time matrix. Do not put these numeric tables only in `structured_problem_data`.
- Use a compact format: one row per aircraft for earliest/target/latest/early-penalty/late-penalty, then one separation matrix with row = aircraft landing first and column = aircraft landing second.
- State the objective is to minimize total early plus late landing penalty.
- State exactly one precedence direction is chosen for each unordered aircraft pair.
- State conditional separation and landing-window constraints.
- Do not turn this into aircraft type assignment, route allocation, tail routing, gate assignment, or makespan-only scheduling.
- Do not present big-M as business data.

Verified instance:

```json
{{INSTANCE_JSON}}
```
