\ Model Capital Budgeting Problem with Risk Management
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  6.9128033885 x[0] + 2.2741902223 x[1] + 3.3103990707 x[2]
   + 7.7063044182 x[3] + 0.6537856807 x[4]
Subject To
 Budget: 5.3879847826 x[0] + 2.523628159 x[1] + 4.6763760973 x[2]
   + 2.7724834852 x[3] + 9.261372678300001 x[4] <= 6.3862850532
 Regulatory_0: x[0] <= 0
 Regulatory_1: x[1] <= 1
 Regulatory_2: x[2] <= 1
 Regulatory_3: x[3] <= 1
 Regulatory_4: x[4] <= 0
 Resource_0: x[0] <= 0
 Resource_1: x[1] <= 1
 Resource_2: x[2] <= 0
 Resource_3: x[3] <= 1
 Resource_4: x[4] <= 1
 RiskThreshold: 8.8583632235 x[0] + 6.464233515 x[1] + 1.5144287454 x[2]
   + 1.4823401633 x[3] + 7.4753909227 x[4] <= 5.4358751677
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
