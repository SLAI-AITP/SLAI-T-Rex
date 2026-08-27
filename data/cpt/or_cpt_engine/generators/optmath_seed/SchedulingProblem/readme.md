## Parameters

### n_restaurants
- **Description**: The number of sites or restaurant locations to be staffed.
- **Type**: Integer or tuple `(min_sites, max_sites)`
- **Default**: `(2, 3)`
- **Reasonable Range**: 1 to 20
- **Note**: Represents different locations in the restaurant chain

### n_employees
- **Description**: The number of available employees in the staff pool.
- **Type**: Integer or Tuple `(min_employees, max_employees)`
- **Default**: `(6, 9)`
- **Reasonable Range**:
  - Small operations: 5 to 20 employees
  - Medium operations: 20 to 50 employees
  - Large operations: 50 to 200 employees

### n_shifts
- **Description**: The number of different shifts in a day.
- **Type**: Integer or tuple `(min_shifts, max_shifts)`
- **Default**: `2`
- **Reasonable Range**: 1 to 4
- **Note**: Typically represents morning/evening or morning/afternoon/evening shifts

### n_skills
- **Description**: The number of different skills or positions to be filled.
- **Type**: Integer
- **Default**: `2`
- **Reasonable Range**: 1 to 10
- **Note**: Examples include cook, server, bartender, host

### skill_probability
- **Description**: Probability that an employee possesses a particular skill.
- **Type**: Float between 0 and 1
- **Default**: `0.7`
- **Reasonable Range**: 0.3 to 0.9
- **Note**: Higher values mean more flexible workforce

### availability_probability
- **Description**: Probability that an employee is available for a particular shift.
- **Type**: Float between 0 and 1
- **Default**: `0.8`
- **Reasonable Range**: 0.5 to 0.9
- **Note**: Reflects typical employee availability patterns

### preference_range
- **Description**: Range for employee preference costs (lower means more preferred).
- **Type**: Tuple of two integers `(min_preference, max_preference)`
- **Default**: `(1, 5)`
- **Reasonable Range**:
  - `min_preference`: 1 to 5
  - `max_preference`: `min_preference` to 10
- **Unit**: Cost units per assignment

### unfulfilled_cost
- **Description**: Penalty cost for each unfilled position.
- **Type**: Integer
- **Default**: `100`
- **Reasonable Range**: 50 to 1000
- **Note**: Should be significantly higher than preference costs

### extra_shortage_slots_range
- **Description**: Optional extra demand slots added after the latent feasible roster is generated. These keep the unfilled-position slack variable meaningful without letting shortage dominate every instance.
- **Type**: Tuple of two integers `(min_extra, max_extra)`
- **Default**: `(0, 1)`
- **Reasonable Range**:
  - `min_extra`: 0 to 5
  - `max_extra`: `min_extra` to 10

### seed
- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer

## Modeling Contract

This generator creates a compact staff coverage assignment MILP.

- Binary `Assignment[site,employee,shift,skill]` variables assign employees to site-shift-skill roles.
- Integer `Unfulfilled[site,shift,skill]` variables count uncovered required positions.
- Coverage is an equality: assigned employees plus unfilled positions equals demand.
- Employee availability and skill eligibility are hard constraints.
- Each employee receives at most one total assignment.
- The objective minimizes assignment preference cost plus unfilled-position penalties.
- This is not machine sequencing, job-shop scheduling, flow-shop scheduling, routing, or due-date scheduling.
- Demand is shaped by generator-only internal bookkeeping, but that internal staff sample is not part of the LLM-facing problem statement and should never be revealed or forced in forward modeling.
