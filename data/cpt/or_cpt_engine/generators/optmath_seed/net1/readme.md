## Parameters

### n_cities

- **Description**: The number of nodes in the capacitated minimum-cost network-flow problem.
- **Type**: Integer or tuple `(min_nodes, max_nodes)`
- **Default**: `(5, 7)`
- **Reasonable Range**: 4 to 12 for CPT seed generation

### shipment_range

- **Description**: A tuple specifying the minimum and maximum quantity for each latent source-to-sink shipment used to construct a feasible balanced flow instance.
- **Type**: Tuple of two integers `(min_quantity, max_quantity)`
- **Default**: `(10, 60)`
- **Reasonable Range**:
  - `min_quantity`: 1 to 1,000
  - `max_quantity`: `min_quantity` to 10,000

### shipping_cost_range

- **Description**: A tuple specifying the minimum and maximum per-unit shipping costs for each directed arc.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(1, 10)`
- **Reasonable Range**:
  - `min_cost`: 1 to 1,000
  - `max_cost`: `min_cost` to 1,000

### capacity_slack_range

- **Description**: A tuple specifying the minimum and maximum capacity slack added to the latent feasible flow on each directed arc. Arc capacities are always at least 1.
- **Type**: Tuple of two integers `(min_slack, max_slack)`
- **Default**: `(0, 15)`
- **Reasonable Range**:
  - `min_slack`: 0 to 100
  - `max_slack`: `min_slack` to 10,000

### seed

- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer

## Modeling Contract

This generator creates a continuous capacitated minimum-cost network-flow LP.

- Nodes have supply and demand values, and total supply equals total demand.
- Directed arcs have per-unit shipping costs and hard capacities.
- The decision variable is continuous nonnegative flow on each listed directed arc.
- Every node must satisfy exact flow balance: supply plus inbound flow equals demand plus outbound flow.
- This is not a vehicle-routing, facility-location, or fixed-charge network design model.
