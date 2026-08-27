# Single-Vehicle CVRPTW Generator

This internalized OptMATH generator creates compact single-vehicle capacitated vehicle routing problem with time windows (CVRPTW) seeds for OR-CPT production.

The generated model has one depot, one vehicle, and a small set of customers. The vehicle must leave the depot once, visit every customer exactly once, respect customer service-start time windows, accumulate customer demand without exceeding capacity, and return to the depot. The objective is to minimize directed travel distance.

## Default Production Scale

- `n_customers`: 5 to 7 customers, not including the depot
- `coordinate_range`: 0 to 60 for generated grid coordinates
- `demand_range`: 2 to 8 units per customer
- `service_time_range`: 3 to 8 minutes per customer
- `time_window_early_slack_range`: 2 to 10 minutes before the latent arrival
- `time_window_late_slack_range`: 18 to 35 minutes after the latent arrival
- `capacity_slack_range`: 5% to 20% above total customer demand
- `M`: automatically derived from time, distance, service, and capacity scale

The instance is generated from an internal feasible route so the source MILP is feasible, but that route is not part of the natural-language problem and should not be revealed as an answer.

## Core Variables

- `ArcVisit[i,j]`: binary, 1 if the single vehicle travels directly from node `i` to node `j`
- `ServiceStartTime[i]`: continuous service start time at customer `i`
- `CumulativeLoad[i]`: continuous cumulative load after serving customer `i`

## Core Constraints

- No self arcs
- Each customer has exactly one incoming and one outgoing arc
- Exactly one arc leaves the depot and exactly one arc returns to the depot
- Time propagation on used arcs, including service time and travel time
- Customer service-start time windows
- Load propagation on used arcs
- Single-vehicle capacity

## Modeling Guardrails

- Do not introduce multiple vehicles, vehicle indices, fleet sizing, or customer-to-vehicle assignment.
- Do not split customers across several routes.
- Do not simplify the model to transportation flow.
- Do not omit service times, time windows, cumulative load, or single-vehicle capacity.
- Do not reveal or force the internally generated feasible route.
