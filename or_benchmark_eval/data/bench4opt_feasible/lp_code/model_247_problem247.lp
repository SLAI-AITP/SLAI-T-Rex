\ Model Capital Budgeting Problem with Project Dependencies and Multi-Period Budget Constraint
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  55.6728894963 x[0,0] + 55.6728894963 x[0,1] + 55.6728894963 x[0,2]
   + 10.2277109269 x[1,0] + 10.2277109269 x[1,1] + 10.2277109269 x[1,2]
   + 99.2334568001 x[2,0] + 99.2334568001 x[2,1] + 99.2334568001 x[2,2]
   + 91.38005796989999 x[3,0] + 91.38005796989999 x[3,1]
   + 91.38005796989999 x[3,2] + 46.6833415237 x[4,0] + 46.6833415237 x[4,1]
   + 46.6833415237 x[4,2]
Subject To
 Budget_0: 36.816156556 x[0,0] + 76.23906307 x[1,0] + 92.8396033236 x[2,0]
   + 13.0211822529 x[3,0] + 24.6904487704 x[4,0] <= 54.7167658414
 Budget_1: 75.8878722515 x[0,1] + 21.1522655967 x[1,1]
   + 44.3734934122 x[2,1] + 97.3415347663 x[3,1] + 16.6794382167 x[4,1]
   <= 41.9115095091
 Budget_2: 62.9154654724 x[0,2] + 55.3727442694 x[1,2]
   + 70.1267528831 x[2,2] + 61.2782950411 x[3,2] + 55.5330616933 x[4,2]
   <= 19.5706715981
 Regulatory_0_0: x[0,0] <= 0
 Regulatory_0_1: x[0,1] <= 0
 Regulatory_0_2: x[0,2] <= 0
 Regulatory_1_0: x[1,0] <= 1
 Regulatory_1_1: x[1,1] <= 1
 Regulatory_1_2: x[1,2] <= 1
 Regulatory_2_0: x[2,0] <= 1
 Regulatory_2_1: x[2,1] <= 1
 Regulatory_2_2: x[2,2] <= 1
 Regulatory_3_0: x[3,0] <= 1
 Regulatory_3_1: x[3,1] <= 1
 Regulatory_3_2: x[3,2] <= 1
 Regulatory_4_0: x[4,0] <= 1
 Regulatory_4_1: x[4,1] <= 1
 Regulatory_4_2: x[4,2] <= 1
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[1,0] x[1,1] x[1,2] x[2,0] x[2,1] x[2,2] x[3,0]
 x[3,1] x[3,2] x[4,0] x[4,1] x[4,2]
End
