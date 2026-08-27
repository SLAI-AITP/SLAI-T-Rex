You are converting a verified car-selection matching instance into a concise natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact structured entities, sets, parameters, and units that appear in the statement.

Output rules:
- Keep `problem_statement` between 700 and 1800 characters whenever possible.
- Use one business paragraph followed by compact indexed lists or a small eligibility table.
- Use `source_artifacts.compact_source_data` as the primary numeric source. Preserve the complete participant-car eligibility data if available.
- Do not reveal solver status, optimal objective value, selected assignments, LP/MPS text, Python code, Gurobi code, or training-data/audit commentary.
- Use `scenario_guidance` only as a business wrapper. Do not add structures absent from the source.

Mathematical rules for `optmath_carselection`:
- Describe a maximum-cardinality bipartite matching or assignment model.
- Decision variable: binary `Assignments[p,c]`, equal to 1 if participant `p` is assigned to car `c`.
- Preserve every participant, every car, and the complete eligibility matrix or eligible-car lists.
- State the objective is to maximize the number of assigned eligible participant-car pairs.
- State assignments are allowed only for eligible participant-car pairs.
- State each participant can be assigned to at most one car.
- State each car can be assigned to at most one participant.
- Do not introduce assignment scores, costs, travel distances, capacities beyond one car/participant, vehicle routing, time windows, or required full assignment.

Semantic framing rules:
- Use neutral entity names such as `participants`, `requesters`, `approved users`, `cars`, `vehicles`, `pool cars`, or `transport assets`.
- Do not use staffing or service-dispatch vocabulary. Forbidden terms include `technician`, `technicians`, `worker`, `workers`, `job`, `jobs`, `shift`, `shifts`, `staffing`, `service job`, `inspection job`, `field-service workforce`, and `technician-job matching`.
- Do not describe the problem as assigning people to jobs, workers to shifts, technicians to service calls, crews to tasks, or staff to roles.
- If the scenario guidance uses broad operations language, reinterpret it strictly as eligible participant-to-vehicle access matching.

Verified instance:

```json
{{INSTANCE_JSON}}
```
