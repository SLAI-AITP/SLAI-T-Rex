\ Model Assignment Problem with Skill Levels
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0,0] + x[0,1] + x[0,2] + x[1,0] + x[1,1] + x[1,2] + x[2,0] + x[2,1]
   + x[2,2] + x[3,0] + x[3,1] + x[3,2] + x[4,0] + x[4,1] + x[4,2]
Subject To
 coverage_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] >= 1
 coverage_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] >= 1
 coverage_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] >= 1
 assignment_0: x[0,0] + x[0,1] + x[0,2] <= 1
 assignment_1: x[1,0] + x[1,1] + x[1,2] <= 1
 assignment_2: x[2,0] + x[2,1] + x[2,2] <= 1
 assignment_3: x[3,0] + x[3,1] + x[3,2] <= 1
 assignment_4: x[4,0] + x[4,1] + x[4,2] <= 1
 skill_0_0: 10 x[0,0] >= 0
 skill_0_1: 10 x[0,1] >= 0
 skill_0_2: 10 x[0,2] >= 0
 skill_1_0: 10 x[1,0] >= 0
 skill_1_1: 10 x[1,1] >= 0
 skill_1_2: 10 x[1,2] >= 0
 skill_2_0: 10 x[2,0] >= 0
 skill_2_1: 10 x[2,1] >= 0
 skill_2_2: 10 x[2,2] >= 0
 skill_3_0: 10 x[3,0] >= 0
 skill_3_1: 10 x[3,1] >= 0
 skill_3_2: 10 x[3,2] >= 0
 skill_4_0: 10 x[4,0] >= 0
 skill_4_1: 10 x[4,1] >= 0
 skill_4_2: 10 x[4,2] >= 0
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[1,0] x[1,1] x[1,2] x[2,0] x[2,1] x[2,2] x[3,0]
 x[3,1] x[3,2] x[4,0] x[4,1] x[4,2]
End
