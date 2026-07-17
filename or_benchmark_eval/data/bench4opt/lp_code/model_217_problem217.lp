\ Model Capital Budgeting Problem with Environmental Impact Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  4.0719698892 x[0] + 25.4810470696 x[1] + 84.0836557822 x[2]
   + 62.512230002 x[3] + 24.615949247 x[4]
Subject To
 budget: 47.8825450337 x[0] + 12.3245551648 x[1] + 94.82794519940001 x[2]
   + 47.0383643601 x[3] + 26.3681259597 x[4] <= 187.60311129
 dependency_0_2: - x[0] + x[2] <= 0
 dependency_1_0: x[0] - x[1] <= 0
 dependency_1_2: - x[1] + x[2] <= 0
 dependency_3_0: x[0] - x[3] <= 0
 dependency_3_1: x[1] - x[3] <= 0
 dependency_3_2: x[2] - x[3] <= 0
 dependency_3_4: - x[3] + x[4] <= 0
 dependency_4_1: x[1] - x[4] <= 0
 dependency_4_2: x[2] - x[4] <= 0
 environmental_impact: 96.1179691542 x[0] + 63.922925504 x[1]
   + 83.0420237892 x[2] + 71.0235557269 x[3] + 44.1132273558 x[4]
   <= 368.2286754865
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
