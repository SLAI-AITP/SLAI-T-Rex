You are converting a verified optimization seed into a concise business problem.

Return exactly one JSON object with:
- `problem_background`: one short sentence naming the organization, trigger, and decision owner.
- `problem_statement`: a self-contained English problem statement.
- `structured_problem_data`: compact entities, sets, parameters, and units that appear in the statement.

Hard rules:
- Keep `problem_statement` concise and self-contained. Prefer 1200-3200 characters and never exceed 3800 characters unless the source fact card itself is larger.
- Use compact indexed rows instead of prose repetition for dense numeric tables.
- Use `source_artifacts.compact_source_data.value` as the authoritative numeric fact card.
- Preserve every number in the compact fact card exactly. Do not change, round, aggregate, infer, or invent coefficients.
- Do not reveal optimal objective value, solver status, solution values, LP text, Python code, Gurobi code, or training/audit commentary.
- Keep the business wrapper realistic, but never alter the mathematical structure in `generator_profile.canonical_math_signature`.
- Avoid unsupported structures listed in `scenario_guidance.forbidden_misframings` and `generator_profile.canonical_math_signature.forbidden_changes`.

Generator-specific rules:
- For `optmath_clsp_expand_capacity`: this is capacitated lot sizing without backlog. Include period demand, cumulative demand, setup cost, unit production cost, holding cost, setup big-M, product capacity consumption, and fixed period capacity. Use one compact row per product-period like `p=0,t=0,demand=...,cum=...,setup=...,prod=...,hold=...,M=...`, plus short capacity-consumption and period-capacity rows. Do not introduce backlog, lost sales, capacity-expansion variables, expansion costs, overtime purchases, or zero-final-inventory as a hard constraint.
- For `optmath_uncapacitatedlotsizing`: this is uncapacitated lot sizing without backlog. Include period demand, fixed ordering cost, unit order cost, holding cost, total-demand big-M, initial inventory 0, and final inventory 0 exactly. Use compact period rows. Do not introduce backlog, backlog penalty, lost sales, unmet-demand slack, production capacity, capacity expansion, or zero-final-backlog constraints.
- For `optmath_uncapacitatedlotsizingbacklogging`: this is uncapacitated lot sizing with carried backlog. Demand values are period-by-period demand, not cumulative demand. Preserve total-demand big-M, zero initial inventory/backlog constants, zero final inventory/backlog constraints, and net inventory as ending inventory minus backlog. Do not add capacity or lost sales.
- For `optmath_net1`: this is continuous capacitated minimum-cost network flow. List every node's separate supply and demand, every directed arc, every arc unit cost, and every arc capacity. Preserve `supply + inbound = demand + outbound`; do not aggregate demand nodes into one sink and do not add binary activation, vehicles, routes, time windows, or unmet-demand slack.
- For `optmath_structure_based_assignment`: this is binary NMR peak-to-amino-acid residue assignment. Include every peak, every amino acid residue, the exact required assignment count, the full peak-residue assignment-cost matrix, every NOE-related peak pair, and the full amino-acid compatibility matrix or distance matrix with threshold. Keep the structural-biology/NMR spectroscopy framing. Do not invent compatibility values, use `perm?`, assume missing matrix entries, or replace exact assignment count with optional assignment. Do not reframe this as staffing, technician-job matching, worker assignment, crew assignment, equipment allocation, gate assignment, vehicle routing, project scheduling, or service dispatch.
- For `optmath_steel3`: this is a single-stage continuous product-mix LP. Include the product profit/rate/processing-hours/minimum-commitment/maximum-market table and the single shared available production-hour capacity exactly. Do not add inventory, setup, binary product selection, sequencing, or multi-stage capacity variables.
- For `optmath_steel4`: this is continuous product-mix production planning. Include the product profit/minimum-commitment/maximum-market table, stage available-hour table, and processing-hours-per-ton matrix exactly. Stage capacity is `sum_p processing_hours_per_ton[p,s] * Production[p] <= available_hours[s]`. Do not copy product market bounds into stage capacities and do not add inventory, setup, binary product selection, or sequencing.
- For `optmath_singlelevelsmallbucket`: this is small-bucket dynamic lot sizing with machines, periods, inventory, and backlog. Include item holding/backlog costs, machine capacity/startup time, scalar setup/startup costs, and the full item-period demand matrix. The source objective has no per-unit production cost term. Do not use placeholder demand, add production unit costs, or collapse machine-period binary setup/startup decisions into one order setup variable.

Verified seed:

```json
{{INSTANCE_JSON}}
```
