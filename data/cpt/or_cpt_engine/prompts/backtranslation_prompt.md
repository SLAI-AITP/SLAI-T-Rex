You are converting a verified optimization instance into a natural-language business problem.

Return exactly one JSON object with these keys:
- `problem_background`: a short realistic business setting.
- `problem_statement`: a self-contained problem statement in English.
- `structured_problem_data`: key entities, parameters, sets, and units that appear in the statement.

Rules:
- Do not reveal the optimal objective value, solver status, or variable solution.
- Do not include mathematical formulation, LP text, MPS text, Python code, or Gurobi code.
- Preserve every meaningful coefficient, limit, demand, capacity, cost, profit, bound, and unit that is explicitly available in `source_artifacts.generation_params`, `math_formula`, or the non-truncated artifact previews.
- If a vector, matrix, table, or indexed coefficient family is fully available, reproduce it completely in the problem statement using readable lists or tables. Do not summarize it as "various costs" or "several limits".
- If `source_artifacts.lp_summary.truncated` or `source_artifacts.generation_params.truncated` is true and the full coefficient table is not available, do not invent omitted coefficients. State the indexed structure, dimensions, available ranges/previews, and all explicitly available values. The pipeline may reject or audit such oversized instances later; faithfulness is more important than pretending the full table was provided.
- Use `scenario_guidance.selected_scenario` as the primary business wrapper for the problem. Keep the selected industry, background, entities, and units unless they conflict with the source coefficients.
- Use `scenario_guidance.scenario_variant` as mandatory diversification guidance: follow its industry lens, narrative angle, organization profile, planning horizon, and entity naming style so that different candidates do not all read like the same generic planning story.
- Treat `scenario_guidance.selected_scenario.base_scenario_id` only as a mathematical-wrapper trace. Do not force the old base industry into the text when the selected industry lens provides a different industry.
- Use `scenario_guidance.selected_business_trigger` as the concrete reason this planning problem arises, such as a weekly planning cycle, budget review, demand surge, disruption recovery, or capacity shortage. Blend it naturally into `problem_background`.
- Also include the concrete organization/stakeholder, business trigger, decision owner, and operating context inside `problem_statement` itself. Later stages consume `problem_statement` directly, so it must remain business-grounded even without reading `problem_background`.
- Start `problem_statement` with a natural business paragraph before listing numeric data. Avoid opening directly with "There are N items" unless the preceding sentence already names the organization, trigger, and decision context.
- Use `scenario_guidance.units` to keep decision quantities, capacities, costs, and objective units internally consistent. Do not mix incompatible units.
- You may use `scenario_guidance.entity_suggestions` and `scenario_guidance.modeling_notes` to improve naming and realism, but do not alter the optimization structure.
- Use `scenario_guidance.applicability.required_source_features` as a checklist for preserving the source structure in business language.
- Do not introduce any situation listed in `scenario_guidance.applicability.incompatible_source_features`. For example, do not turn a transportation matrix into a vehicle-routing story unless routes, vehicles, time windows, and sequencing variables are actually present.
- Use `generator_profile.canonical_math_signature` as a hard mathematical signature. Preserve variable types, objective sense, required tables, and core constraint families; do not introduce any listed forbidden changes.
- Use `generator_profile.concept_tags` only to clarify the business-to-model mapping. Do not add concepts that are not supported by the source instance.
- Include a concrete business background in `problem_background`; avoid generic wording like "a planning team needs to optimize resources" unless the source is truly generic.
- Avoid stale template openings such as "A company has..." when the scenario variant provides a more specific stakeholder, operating context, or trigger.
- Keep the natural-language problem faithful to the source mathematical structure. Do not invent extra constraints, side conditions, or objectives.
- If `scenario_guidance.forbidden_misframings` is present, avoid those misframings.
- The result must be suitable as the prompt side of a CPT training document.

Verified instance. Large LP/code artifacts may be summarized for prompt safety; use the structured fields first and do not infer omitted source data:

```json
{{INSTANCE_JSON}}
```
