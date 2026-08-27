# Static Line Frequency Planning Generator

This internalized OptMATH generator creates compact line-planning MILP seeds for OR-CPT production.

The generated model chooses which candidate transit or shuttle lines to activate and how much continuous service frequency to operate on each selected line. OD-pair demand can be served only by explicitly listed candidate lines. Unserved demand is allowed through a continuous slack variable with a high penalty, but the generator sizes each instance so a zero-shortage solution is available. A total vehicle-hours budget limits frequency, and selected transfer-required nodes must be covered by at least two activated lines.

## Default Production Scale

- `n_nodes`: 5 to 8 stops or network nodes
- `n_lines`: 6 to 10 candidate lines
- `n_od_pairs`: 6 to 12 origin-destination demand pairs
- `service_lines_per_od`: 2 to 4 candidate lines per OD pair
- `transfer_requirement_ratio`: 25% to 45% of nodes require transfer coverage

These defaults are intentionally prompt-safe. The instance includes complete compact tables for lines, OD demand, service eligibility, pass-through nodes, transfer requirements, and fleet vehicle-hours.

## Core Variables

- `x[l]`: binary line activation
- `f[l]`: continuous nonnegative service frequency
- `s[c]`: continuous unmet demand for OD pair `c`

## Core Constraints

- OD demand coverage with unmet-demand slack
- Frequency activation linking: `min_freq[l] * x[l] <= f[l] <= max_freq[l] * x[l]`
- Fleet vehicle-hours budget
- Fixed transfer-node coverage: each required node must be served by at least two activated passing lines

## Modeling Guardrails

- Do not introduce passenger-flow, commodity-flow, vehicle-routing, subtour, time-window, or arc-continuity variables.
- Do not invent `z[i,j,l]` arc-use variables; the generator already provides candidate line coverage indicators.
- Do not make frequency integer unless a future source variant explicitly requests integer frequencies.
- Do not replace fixed transfer requirements with optional transfer-station decisions.
