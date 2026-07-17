\ Model Production Planning Problem with Time-Dependent Quality Standards
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  185760.7925017837 flow_rate[0,0] + 185760.7925017837 flow_rate[0,1]
   + 185760.7925017837 flow_rate[0,2] + 185760.7925017837 flow_rate[0,3]
   + 185760.7925017837 flow_rate[0,4] + 185760.7925017837 flow_rate[0,5]
   + 185760.7925017837 flow_rate[0,6] + 871680.2477542313 flow_rate[1,0]
   + 871680.2477542313 flow_rate[1,1] + 871680.2477542313 flow_rate[1,2]
   + 871680.2477542313 flow_rate[1,3] + 871680.2477542313 flow_rate[1,4]
   + 871680.2477542313 flow_rate[1,5] + 871680.2477542313 flow_rate[1,6]
   + 703431.7550506816 flow_rate[2,0] + 703431.7550506816 flow_rate[2,1]
   + 703431.7550506816 flow_rate[2,2] + 703431.7550506816 flow_rate[2,3]
   + 703431.7550506816 flow_rate[2,4] + 703431.7550506816 flow_rate[2,5]
   + 703431.7550506816 flow_rate[2,6] + 464296.8818550048 flow_rate[3,0]
   + 464296.8818550048 flow_rate[3,1] + 464296.8818550048 flow_rate[3,2]
   + 464296.8818550048 flow_rate[3,3] + 464296.8818550048 flow_rate[3,4]
   + 464296.8818550048 flow_rate[3,5] + 464296.8818550048 flow_rate[3,6]
   + 584076.450902142 flow_rate[4,0] + 584076.450902142 flow_rate[4,1]
   + 584076.450902142 flow_rate[4,2] + 584076.450902142 flow_rate[4,3]
   + 584076.450902142 flow_rate[4,4] + 584076.450902142 flow_rate[4,5]
   + 584076.450902142 flow_rate[4,6] + 976444.3576662686 flow_rate[5,0]
   + 976444.3576662686 flow_rate[5,1] + 976444.3576662686 flow_rate[5,2]
   + 976444.3576662686 flow_rate[5,3] + 976444.3576662686 flow_rate[5,4]
   + 976444.3576662686 flow_rate[5,5] + 976444.3576662686 flow_rate[5,6]
   + 139485.2907607319 flow_rate[6,0] + 139485.2907607319 flow_rate[6,1]
   + 139485.2907607319 flow_rate[6,2] + 139485.2907607319 flow_rate[6,3]
   + 139485.2907607319 flow_rate[6,4] + 139485.2907607319 flow_rate[6,5]
   + 139485.2907607319 flow_rate[6,6]
Subject To
 capacity_constraint_0_0: flow_rate[0,0] <= 704198.1609595485
 capacity_constraint_0_1: flow_rate[0,1] <= 704198.1609595485
 capacity_constraint_0_2: flow_rate[0,2] <= 704198.1609595485
 capacity_constraint_0_3: flow_rate[0,3] <= 704198.1609595485
 capacity_constraint_0_4: flow_rate[0,4] <= 704198.1609595485
 capacity_constraint_0_5: flow_rate[0,5] <= 704198.1609595485
 capacity_constraint_0_6: flow_rate[0,6] <= 704198.1609595485
 capacity_constraint_1_0: flow_rate[1,0] <= 531389.4066265334
 capacity_constraint_1_1: flow_rate[1,1] <= 531389.4066265334
 capacity_constraint_1_2: flow_rate[1,2] <= 531389.4066265334
 capacity_constraint_1_3: flow_rate[1,3] <= 531389.4066265334
 capacity_constraint_1_4: flow_rate[1,4] <= 531389.4066265334
 capacity_constraint_1_5: flow_rate[1,5] <= 531389.4066265334
 capacity_constraint_1_6: flow_rate[1,6] <= 531389.4066265334
 capacity_constraint_2_0: flow_rate[2,0] <= 627184.1455238423
 capacity_constraint_2_1: flow_rate[2,1] <= 627184.1455238423
 capacity_constraint_2_2: flow_rate[2,2] <= 627184.1455238423
 capacity_constraint_2_3: flow_rate[2,3] <= 627184.1455238423
 capacity_constraint_2_4: flow_rate[2,4] <= 627184.1455238423
 capacity_constraint_2_5: flow_rate[2,5] <= 627184.1455238423
 capacity_constraint_2_6: flow_rate[2,6] <= 627184.1455238423
 capacity_constraint_3_0: flow_rate[3,0] <= 691059.8880020769
 capacity_constraint_3_1: flow_rate[3,1] <= 691059.8880020769
 capacity_constraint_3_2: flow_rate[3,2] <= 691059.8880020769
 capacity_constraint_3_3: flow_rate[3,3] <= 691059.8880020769
 capacity_constraint_3_4: flow_rate[3,4] <= 691059.8880020769
 capacity_constraint_3_5: flow_rate[3,5] <= 691059.8880020769
 capacity_constraint_3_6: flow_rate[3,6] <= 691059.8880020769
 capacity_constraint_4_0: flow_rate[4,0] <= 385494.3886439577
 capacity_constraint_4_1: flow_rate[4,1] <= 385494.3886439577
 capacity_constraint_4_2: flow_rate[4,2] <= 385494.3886439577
 capacity_constraint_4_3: flow_rate[4,3] <= 385494.3886439577
 capacity_constraint_4_4: flow_rate[4,4] <= 385494.3886439577
 capacity_constraint_4_5: flow_rate[4,5] <= 385494.3886439577
 capacity_constraint_4_6: flow_rate[4,6] <= 385494.3886439577
 capacity_constraint_5_0: flow_rate[5,0] <= 2362.285275178132
 capacity_constraint_5_1: flow_rate[5,1] <= 2362.285275178132
 capacity_constraint_5_2: flow_rate[5,2] <= 2362.285275178132
 capacity_constraint_5_3: flow_rate[5,3] <= 2362.285275178132
 capacity_constraint_5_4: flow_rate[5,4] <= 2362.285275178132
 capacity_constraint_5_5: flow_rate[5,5] <= 2362.285275178132
 capacity_constraint_5_6: flow_rate[5,6] <= 2362.285275178132
 capacity_constraint_6_0: flow_rate[6,0] <= 627677.7188750199
 capacity_constraint_6_1: flow_rate[6,1] <= 627677.7188750199
 capacity_constraint_6_2: flow_rate[6,2] <= 627677.7188750199
 capacity_constraint_6_3: flow_rate[6,3] <= 627677.7188750199
 capacity_constraint_6_4: flow_rate[6,4] <= 627677.7188750199
 capacity_constraint_6_5: flow_rate[6,5] <= 627677.7188750199
 capacity_constraint_6_6: flow_rate[6,6] <= 627677.7188750199
 quality_standard_constraint_0: 998952.7076232343 flow_rate[0,0]
   + 239823.3336507718 flow_rate[1,0] + 459823.5539128198 flow_rate[2,0]
   + 162641.030718395 flow_rate[3,0] + 16914.23429029551 flow_rate[4,0]
   + 669366.5727491063 flow_rate[5,0] + 570810.4167918817 flow_rate[6,0]
   >= 207939.2805166851
 quality_standard_constraint_1: 998952.7076232343 flow_rate[0,1]
   + 239823.3336507718 flow_rate[1,1] + 459823.5539128198 flow_rate[2,1]
   + 162641.030718395 flow_rate[3,1] + 16914.23429029551 flow_rate[4,1]
   + 669366.5727491063 flow_rate[5,1] + 570810.4167918817 flow_rate[6,1]
   >= 575654.1382050832
 quality_standard_constraint_2: 998952.7076232343 flow_rate[0,2]
   + 239823.3336507718 flow_rate[1,2] + 459823.5539128198 flow_rate[2,2]
   + 162641.030718395 flow_rate[3,2] + 16914.23429029551 flow_rate[4,2]
   + 669366.5727491063 flow_rate[5,2] + 570810.4167918817 flow_rate[6,2]
   >= 297293.3912312712
 quality_standard_constraint_3: 998952.7076232343 flow_rate[0,3]
   + 239823.3336507718 flow_rate[1,3] + 459823.5539128198 flow_rate[2,3]
   + 162641.030718395 flow_rate[3,3] + 16914.23429029551 flow_rate[4,3]
   + 669366.5727491063 flow_rate[5,3] + 570810.4167918817 flow_rate[6,3]
   >= 753135.122951266
 quality_standard_constraint_4: 998952.7076232343 flow_rate[0,4]
   + 239823.3336507718 flow_rate[1,4] + 459823.5539128198 flow_rate[2,4]
   + 162641.030718395 flow_rate[3,4] + 16914.23429029551 flow_rate[4,4]
   + 669366.5727491063 flow_rate[5,4] + 570810.4167918817 flow_rate[6,4]
   >= 698356.9288264915
 quality_standard_constraint_5: 998952.7076232343 flow_rate[0,5]
   + 239823.3336507718 flow_rate[1,5] + 459823.5539128198 flow_rate[2,5]
   + 162641.030718395 flow_rate[3,5] + 16914.23429029551 flow_rate[4,5]
   + 669366.5727491063 flow_rate[5,5] + 570810.4167918817 flow_rate[6,5]
   >= 684980.5161645865
 quality_standard_constraint_6: 998952.7076232343 flow_rate[0,6]
   + 239823.3336507718 flow_rate[1,6] + 459823.5539128198 flow_rate[2,6]
   + 162641.030718395 flow_rate[3,6] + 16914.23429029551 flow_rate[4,6]
   + 669366.5727491063 flow_rate[5,6] + 570810.4167918817 flow_rate[6,6]
   >= 612888.7366197456
Bounds
End
