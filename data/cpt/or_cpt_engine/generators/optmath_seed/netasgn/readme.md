# Continuous Project Assignment

This internal OR-CPT generator creates a balanced continuous resource-hour allocation model. It is transportation-style assignment, not binary one-to-one matching.

## Parameters

### n_people

- **Description**: Number or range for available people/resources.
- **Type**: Integer or `(min, max)` range
- **Default**: `(5, 6)`

### n_projects

- **Description**: Number or range for projects/requests.
- **Type**: Integer or `(min, max)` range
- **Default**: `(5, 6)`

### allocation_range

- **Description**: Range used internally when constructing a feasible person-project allocation. The allocation certificate is not exposed to LLM-facing generation parameters.
- **Type**: Tuple of two integers `(min_hours, max_hours)`
- **Default**: `(1, 6)`

### assignments_per_person

- **Description**: Range for the number of projects used internally while constructing feasible balanced supply and demand.
- **Type**: Tuple of two integers `(min_assignments, max_assignments)`
- **Default**: `(2, 4)`

### cost_range

- **Description**: Minimum and maximum cost per assigned hour for each person-project pair.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(10, 50)`

### limit_slack_range

- **Description**: Extra upper-bound slack added above the latent feasible allocation for every person-project pair.
- **Type**: Tuple of two integers `(min_slack, max_slack)`
- **Default**: `(2, 8)`

### seed

- **Description**: Random seed for reproducibility.
- **Type**: Integer, optional

## OR-CPT Internal Notes

- Decision variable `Assign[i,j]` is continuous nonnegative assigned hours.
- Every person's available hours are fully allocated: `sum_j Assign[i,j] = supply[i]`.
- Every project's required hours are exactly satisfied: `sum_i Assign[i,j] = demand[j]`.
- Every person-project pair has a listed upper bound: `Assign[i,j] <= max_contribution[i,j]`.
- Total supply equals total demand by construction.
- The internal feasible allocation certificate is not part of the natural-language problem and should not be revealed as an answer.
- Do not relax supply to `<=` or demand to `>=`.
- Do not convert the model to binary one-to-one matching.
- Do not omit the contribution-limit matrix; missing upper bounds are a common source of objective mismatch.
