## OR-CPT Internal Notes

This internal version should be interpreted as a compact integer
time-expanded fleet-flow model, not as customer-stop vehicle routing.

The intended decision structure is:

- `NumPlanes[v,i,t,j,h]`: nonnegative integer number of fleet units of type
  `v` assigned to feasible leg `(i,t,j,h)`.
- `NumIdlePlanes[v,i,t]`: nonnegative integer idle fleet units at location
  `i` after period `t`.
- `NumIdlePlanesInit[v,i]`: nonnegative integer initial fleet units placed at
  location `i`.

Core constraints are initial fleet balance, time-expanded fleet conservation,
fleet availability, active-leg passenger-demand satisfaction, and inactive-leg
route restriction. There are no binary route variables, visit-once customer
constraints, subtour constraints, service times, load propagation, or customer
time windows.
The generator emits `compact_fleet_flow_tables`; downstream prompts should use
those tables as the complete time-expanded fleet-flow source instead of trying
to reconstruct the model from LP previews.

## Parameters

### n_locations

- **Description**: The number of locations/cities to include in the fleet routing problem.
- **Type**: Integer
- **Default**: `(3, 4)`
- **Reasonable Range**: 1 to 100

### n_planes

- **Description**: The number of plane types to include in the fleet routing problem.
- **Type**: Integer
- **Default**: `(2, 4)`
- **Reasonable Range**: 1 to 50

### time_periods

- **Description**: The number of periods (time slots) to include in the fleet routing problem.
- **Type**: Integer
- **Default**: `(2, 3)`
- **Reasonable Range**: 1 to 100

### capacity_range

- **Description**: A tuple specifying the minimum and maximum capacities for the planes.
- **Type**: Tuple of two integers `(min_capacity, max_capacity)`
- **Default**: `(100, 160)`
- **Reasonable Range**:
  - `min_capacity`: 1 to 10,000
  - `max_capacity`: `min_capacity` to 10,000

### cost_range

- **Description**: A tuple specifying the minimum and maximum costs for using the planes.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(5, 20)`
- **Reasonable Range**:
  - `min_cost`: 1 to 1,000,000
  - `max_cost`: `min_cost` to 1,000,000

### max_available_planes

- **Description**: The number of available planes for each plane type.
- **Type**: Integer
- **Default**: `3`
- **Reasonable Range**:
  - `min_available`: 1 to 100
  - `max_available`: `min_available` to 100

### max_passengers

- **Description**: Maximum passenger demand on each active flight leg.
- **Type**: Integer
- **Default**: `80`
- **Reasonable Range**:
  - 1 to 10,000

### route_density

- **Description**: Legacy sparsity parameter retained for compatibility. The
  current internal generator caps the active leg count directly.
- **Type**: Float
- **Default**: `0.35`

### max_active_routes

- **Description**: Maximum number of active legs with passenger demand.
- **Type**: Integer
- **Default**: `2`

### seed

- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer
