## Parameters

### n_origins

- **Description**: The number of origins (supply points) in the multi-commodity transportation problem.
- **Type**: Integer
- **Default**: `(3, 4)` (a tuple specifying the sampled origin-count range)
- **Reasonable Range**: 1 to 100

### n_destinations

- **Description**: The number of destinations (demand points) in the multi-commodity transportation problem.
- **Type**: Integer
- **Default**: `(4, 5)` (a tuple specifying the sampled destination-count range)
- **Reasonable Range**: 1 to 100

### n_products

- **Description**: The number of products (commodities) to be transported in the multi-commodity transportation problem.
- **Type**: Integer
- **Default**: `(2, 2)` (two products by default; kept compact for prompt-safe complete tables)
- **Reasonable Range**: 1 to 100

### demand_range

- **Description**: A tuple specifying the minimum and maximum demand values for each product at each destination.
- **Type**: Tuple of two integers `(min_demand, max_demand)`
- **Default**: `(20, 70)`
- **Reasonable Range**:
  - `min_demand`: 1 to 10,000
  - `max_demand`: `min_demand` to 10,000

### limit_slack_range

- **Description**: Extra shared lane capacity added above the latent feasible product flow on an active origin-destination lane.
- **Type**: Tuple of two integers `(min_limit, max_limit)`
- **Default**: `(5, 30)`
- **Reasonable Range**:
  - `min_limit`: 0 to 10,000
  - `max_limit`: `min_limit` to 10,000

### inactive_lane_limit_range

- **Description**: Shared lane capacity range for lanes that are not used by the latent feasible shipment.
- **Type**: Tuple of two integers `(min_limit, max_limit)`
- **Default**: `(0, 20)`
- **Reasonable Range**:
  - `min_limit`: 0 to 10,000
  - `max_limit`: `min_limit` to 10,000

### cost_range

- **Description**: A tuple specifying the minimum and maximum shipping costs for transporting one unit of a product from an origin to a destination.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(1, 10)`
- **Reasonable Range**:
  - `min_cost`: 1 to 10,000
  - `max_cost`: `min_cost` to 10,000

### seed

- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer
