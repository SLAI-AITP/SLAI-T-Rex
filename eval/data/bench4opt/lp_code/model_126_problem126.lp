\ Model Capital Budgeting Problem with Risk Management
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  871.9597849603 x[0] + 396.8232972211 x[1] + 724.3063065850999 x[2]
   + 359.3960158253 x[3] + 950.6741856069 x[4]
Subject To
 Budget: 542.5532219719 x[0] + 182.1181656644 x[1] + 413.1649506652 x[2]
   + 399.9775542986 x[3] + 703.1201585691 x[4] <= 3539.1388778163
 Regulatory_0: x[0] <= 0
 Regulatory_1: x[1] <= 1
 Regulatory_2: x[2] <= 0
 Regulatory_3: x[3] <= 1
 Regulatory_4: x[4] <= 1
 Risk: 0.358619029 x[0] + 0.8761477814 x[1] + 0.4233400456 x[2]
   + 0.5116770019 x[3] + 0.8904953718 x[4] <= 2.315320773
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
