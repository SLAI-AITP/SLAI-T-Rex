## OR-CPT Internal Notes

This internal version is tuned for seed-instance generation before LLM
backtranslation. Defaults are intentionally compact so the complete
commodity-plant-distribution-center-zone shipping-cost table can be included
in the natural-language problem without creating very long prompts.
The generator now emits `compact_facility_location_tables`, which should be
treated as the primary LLM-facing numeric source for CPT generation.

The intended model is a multi-commodity distribution-center selection MILP:

- `Shipped[c,p,d,z]` is a continuous nonnegative commodity flow from plant `p`
  through distribution center `d` to customer zone `z`.
- `Selected[d]` is a binary open/closed decision for candidate distribution
  center `d`.
- `Served[d,z]` is a binary zone-assignment decision.
- Core constraints include supply limits, exact served-demand equalities,
  exactly-one zone assignment, explicit `Served[d,z] <= Selected[d]` linking,
  distribution-center throughput bounds, and a small-instance facility limit.

Do not reinterpret this generator as vehicle routing, customer visit
sequencing, pure transportation, or a one-index facility assignment model.

## Parameters

### n_locations

- **Description**: The number of locations to include in the facility location problem.
- **Type**: Integer
- **Default**: `(8, 9)`
- **Reasonable Range**: 1 to 100

### n_commodities

- **Description**: The number of commodities to include in the facility location problem.
- **Type**: Integer
- **Default**: `(2, 2)`
- **Reasonable Range**: 1 to 50

### n_product_plants

- **Description**: The number of product plants to include in the facility location problem.
- **Type**: Integer
- **Default**: `(2, 2)`
- **Reasonable Range**: 1 to 20

### n_distribution_centers

- **Description**: The number of distribution centers to include in the facility location problem.
- **Type**: Integer
- **Default**: `(3, 3)`
- **Reasonable Range**: 1 to 20

### n_customer_zones

- **Description**: The number of customer zones to include in the facility location problem.
- **Type**: Integer
- **Default**: `(3, 3)`
- **Reasonable Range**: 1 to 50

### supply_range

- **Description**: A tuple specifying the minimum and maximum supply values for the commodities at product plants.
- **Type**: Tuple of two integers `(min_supply, max_supply)`
- **Default**: `(80, 320)`
- **Reasonable Range**:
  - `min_supply`: 1 to 10,000
  - `max_supply`: `min_supply` to 10,000

### demand_range

- **Description**: A tuple specifying the minimum and maximum demand values for the commodities at customer zones.
- **Type**: Tuple of two integers `(min_demand, max_demand)`
- **Default**: `(20, 80)`
- **Reasonable Range**:
  - `min_demand`: 1 to 10,000
  - `max_demand`: `min_demand` to 10,000

### max_throughput_range

- **Description**: A tuple specifying the minimum and maximum throughput values for the distribution centers.
- **Type**: Tuple of two integers `(min_throughput, max_throughput)`
- **Default**: `(350, 900)`
- **Reasonable Range**:
  - `min_throughput`: 1 to 10,000
  - `max_throughput`: `min_throughput` to 10,000

### min_throughput_range

- **Description**: A tuple specifying the minimum and minimum throughput values for the distribution centers.
- **Type**: Tuple of two integers `(min_throughput, max_throughput)`
- **Default**: `(20, 160)`
- **Reasonable Range**:
  - `min_throughput`: 1 to 10,000
  - `max_throughput`: `min_throughput` to 10,000

### unit_throughput_cost_range

- **Description**: A tuple specifying the minimum and maximum unit throughput costs for the distribution centers.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(1, 10)`
- **Reasonable Range**:
  - `min_cost`: 1 to 10,000
  - `max_cost`: `min_cost` to 10,000

### fixed_throughput_cost_range

- **Description**: A tuple specifying the minimum and maximum fixed throughput costs for the distribution centers.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(600, 3000)`
- **Reasonable Range**:
  - `min_cost`: 1 to 10,000
  - `max_cost`: `min_cost` to 10,000

### variable_cost_range

- **Description**: A tuple specifying the minimum and maximum variable costs for shipping commodities from product plants through distribution centers to customer zones.
- **Type**: Tuple of two integers `(min_cost, max_cost)`
- **Default**: `(1, 20)`
- **Reasonable Range**:
  - `min_cost`: 1 to 10,000
  - `max_cost`: `min_cost` to 10,000

### seed

- **Description**: Random seed for reproducibility of the generated problem instance.
- **Type**: Integer (optional)
- **Default**: `None` (random seed not set)
- **Reasonable Range**: Any valid integer
