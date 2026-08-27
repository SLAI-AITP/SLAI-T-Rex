## Parameters

### n_cities

- **Description**: The number of nodes in the node-throughput capacitated minimum-cost flow problem.
- **Type**: Integer or tuple `(min_nodes, max_nodes)`
- **Default**: `(5, 8)`
- **Reasonable Range**: 4 to 12 for CPT seed generation

### shipment_range

- **Description**: A tuple specifying the minimum and maximum quantity for each internally sampled source-to-sink shipment used to construct a feasible balanced network-flow instance. The flow certificate is not exposed to LLM-facing generation parameters.
- **Type**: Tuple of two integers `(min_quantity, max_quantity)`
- **Default**: `(8, 45)`
- **Reasonable Range**:
  - `min_quantity`: 1 to 1,000
  - `max_quantity`: `min_quantity` to 10,000

### cost_range

- **Description**: A tuple specifying the minimum and maximum per-unit costs for shipping over each directed link.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(1, 10)`
- **Reasonable Range**:
  - `min_cost`: 1 to 10,000
  - `max_cost`: `min_cost` to 10,000

### city_capacity_slack_range

- **Description**: A tuple specifying throughput slack added above the internally constructed feasible node throughput. Node throughput capacity is always at least 1.
- **Type**: Tuple of two integers `(min_slack, max_slack)`
- **Default**: `(0, 12)`
- **Reasonable Range**:
  - `min_slack`: 0 to 10,000
  - `max_slack`: `min_slack` to 10,000

### link_capacity_slack_range

- **Description**: A tuple specifying capacity slack added above the internally constructed feasible flow on each directed link. Link capacity is always at least 1.
- **Type**: Tuple of two integers `(min_slack, max_slack)`
- **Default**: `(0, 12)`
- **Reasonable Range**:
  - `min_slack`: 0 to 10,000
  - `max_slack`: `min_slack` to 10,000

### reverse_arc_probability

- **Description**: Probability of adding a reverse directed arc next to each backbone arc to create alternate routing options.
- **Type**: Float
- **Default**: `0.25`
- **Reasonable Range**: 0.0 to 1.0

### seed

- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer

## Modeling Contract

This generator creates a continuous capacitated minimum-cost flow LP with node throughput limits.

- Nodes have supply and demand values, and total supply equals total demand.
- Directed links have per-unit shipping costs and hard capacities.
- Each node has a throughput capacity on supply plus inbound flow.
- The decision variable is continuous nonnegative shipment on each listed directed link.
- The internal feasible flow certificate is not part of the natural-language problem and should not be revealed as an answer.
- This is not a maximum-flow, vehicle-routing, route-sequencing, or fixed-charge lane activation model.
