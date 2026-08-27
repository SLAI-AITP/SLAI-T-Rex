\ Model Multi-Commodity Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 flow[0,0] + 2 flow[0,1] + 2 flow[0,2] + 2 flow[0,3] + 2 flow[0,4]
   + 2 flow[0,5] + 2 flow[0,6] + 2 flow[0,7] + 2 flow[0,8] + 2 flow[0,9]
   + 2 flow[0,10] + 2 flow[0,11] + 2 flow[0,12] + 2 flow[0,13]
   + 2 flow[0,14] + 2 flow[0,15] + 2 flow[0,16] + 2 flow[0,17]
   + 2 flow[0,18] + 2 flow[0,19] + 2 flow[1,0] + 2 flow[1,1] + 2 flow[1,2]
   + 2 flow[1,3] + 2 flow[1,4] + 2 flow[1,5] + 2 flow[1,6] + 2 flow[1,7]
   + 2 flow[1,8] + 2 flow[1,9] + 2 flow[1,10] + 2 flow[1,11] + 2 flow[1,12]
   + 2 flow[1,13] + 2 flow[1,14] + 2 flow[1,15] + 2 flow[1,16]
   + 2 flow[1,17] + 2 flow[1,18] + 2 flow[1,19] + 2 flow[2,0] + 2 flow[2,1]
   + 2 flow[2,2] + 2 flow[2,3] + 2 flow[2,4] + 2 flow[2,5] + 2 flow[2,6]
   + 2 flow[2,7] + 2 flow[2,8] + 2 flow[2,9] + 2 flow[2,10] + 2 flow[2,11]
   + 2 flow[2,12] + 2 flow[2,13] + 2 flow[2,14] + 2 flow[2,15]
   + 2 flow[2,16] + 2 flow[2,17] + 2 flow[2,18] + 2 flow[2,19]
   + 2 flow[3,0] + 2 flow[3,1] + 2 flow[3,2] + 2 flow[3,3] + 2 flow[3,4]
   + 2 flow[3,5] + 2 flow[3,6] + 2 flow[3,7] + 2 flow[3,8] + 2 flow[3,9]
   + 2 flow[3,10] + 2 flow[3,11] + 2 flow[3,12] + 2 flow[3,13]
   + 2 flow[3,14] + 2 flow[3,15] + 2 flow[3,16] + 2 flow[3,17]
   + 2 flow[3,18] + 2 flow[3,19] + 2 flow[4,0] + 2 flow[4,1] + 2 flow[4,2]
   + 2 flow[4,3] + 2 flow[4,4] + 2 flow[4,5] + 2 flow[4,6] + 2 flow[4,7]
   + 2 flow[4,8] + 2 flow[4,9] + 2 flow[4,10] + 2 flow[4,11] + 2 flow[4,12]
   + 2 flow[4,13] + 2 flow[4,14] + 2 flow[4,15] + 2 flow[4,16]
   + 2 flow[4,17] + 2 flow[4,18] + 2 flow[4,19]
Subject To
 cap_0: flow[0,0] + flow[1,0] + flow[2,0] + flow[3,0] + flow[4,0] <= 1000
 cap_1: flow[0,1] + flow[1,1] + flow[2,1] + flow[3,1] + flow[4,1] <= 1000
 cap_2: flow[0,2] + flow[1,2] + flow[2,2] + flow[3,2] + flow[4,2] <= 1000
 cap_3: flow[0,3] + flow[1,3] + flow[2,3] + flow[3,3] + flow[4,3] <= 1000
 cap_4: flow[0,4] + flow[1,4] + flow[2,4] + flow[3,4] + flow[4,4] <= 1000
 cap_5: flow[0,5] + flow[1,5] + flow[2,5] + flow[3,5] + flow[4,5] <= 1000
 cap_6: flow[0,6] + flow[1,6] + flow[2,6] + flow[3,6] + flow[4,6] <= 1000
 cap_7: flow[0,7] + flow[1,7] + flow[2,7] + flow[3,7] + flow[4,7] <= 1000
 cap_8: flow[0,8] + flow[1,8] + flow[2,8] + flow[3,8] + flow[4,8] <= 1000
 cap_9: flow[0,9] + flow[1,9] + flow[2,9] + flow[3,9] + flow[4,9] <= 1000
 cap_10: flow[0,10] + flow[1,10] + flow[2,10] + flow[3,10] + flow[4,10]
   <= 1000
 cap_11: flow[0,11] + flow[1,11] + flow[2,11] + flow[3,11] + flow[4,11]
   <= 1000
 cap_12: flow[0,12] + flow[1,12] + flow[2,12] + flow[3,12] + flow[4,12]
   <= 1000
 cap_13: flow[0,13] + flow[1,13] + flow[2,13] + flow[3,13] + flow[4,13]
   <= 1000
 cap_14: flow[0,14] + flow[1,14] + flow[2,14] + flow[3,14] + flow[4,14]
   <= 1000
 cap_15: flow[0,15] + flow[1,15] + flow[2,15] + flow[3,15] + flow[4,15]
   <= 1000
 cap_16: flow[0,16] + flow[1,16] + flow[2,16] + flow[3,16] + flow[4,16]
   <= 1000
 cap_17: flow[0,17] + flow[1,17] + flow[2,17] + flow[3,17] + flow[4,17]
   <= 1000
 cap_18: flow[0,18] + flow[1,18] + flow[2,18] + flow[3,18] + flow[4,18]
   <= 1000
 cap_19: flow[0,19] + flow[1,19] + flow[2,19] + flow[3,19] + flow[4,19]
   <= 1000
 flow_conservation_0_0: - flow[0,5] + flow[0,7] + flow[0,9] - flow[0,19]
   = 1
 flow_conservation_0_1: - flow[0,4] + flow[0,6] - flow[0,8] - flow[0,9]
   = 1
 flow_conservation_0_2: - flow[0,0] - flow[0,11] - flow[0,14] - flow[0,15]
   = -9
 flow_conservation_0_3: - flow[0,3] + flow[0,8] + flow[0,12] + flow[0,15]
   + flow[0,18] = 1
 flow_conservation_0_4: flow[0,1] - flow[0,7] + flow[0,14] + flow[0,17]
   = 1
 flow_conservation_0_5: flow[0,3] - flow[0,6] + flow[0,11] - flow[0,16]
   - flow[0,17] - flow[0,18] = 1
 flow_conservation_0_6: flow[0,4] + flow[0,13] + flow[0,16] = 1
 flow_conservation_0_7: - flow[0,2] - flow[0,10] - flow[0,12] + flow[0,19]
   = 1
 flow_conservation_0_8: - flow[0,1] + flow[0,5] + flow[0,10] = 1
 flow_conservation_0_9: flow[0,0] + flow[0,2] - flow[0,13] = 1
 flow_conservation_1_0: - flow[1,5] + flow[1,7] + flow[1,9] - flow[1,19]
   = 1
 flow_conservation_1_1: - flow[1,4] + flow[1,6] - flow[1,8] - flow[1,9]
   = 1
 flow_conservation_1_2: - flow[1,0] - flow[1,11] - flow[1,14] - flow[1,15]
   = -9
 flow_conservation_1_3: - flow[1,3] + flow[1,8] + flow[1,12] + flow[1,15]
   + flow[1,18] = 1
 flow_conservation_1_4: flow[1,1] - flow[1,7] + flow[1,14] + flow[1,17]
   = 1
 flow_conservation_1_5: flow[1,3] - flow[1,6] + flow[1,11] - flow[1,16]
   - flow[1,17] - flow[1,18] = 1
 flow_conservation_1_6: flow[1,4] + flow[1,13] + flow[1,16] = 1
 flow_conservation_1_7: - flow[1,2] - flow[1,10] - flow[1,12] + flow[1,19]
   = 1
 flow_conservation_1_8: - flow[1,1] + flow[1,5] + flow[1,10] = 1
 flow_conservation_1_9: flow[1,0] + flow[1,2] - flow[1,13] = 1
 flow_conservation_2_0: - flow[2,5] + flow[2,7] + flow[2,9] - flow[2,19]
   = 1
 flow_conservation_2_1: - flow[2,4] + flow[2,6] - flow[2,8] - flow[2,9]
   = 1
 flow_conservation_2_2: - flow[2,0] - flow[2,11] - flow[2,14] - flow[2,15]
   = -9
 flow_conservation_2_3: - flow[2,3] + flow[2,8] + flow[2,12] + flow[2,15]
   + flow[2,18] = 1
 flow_conservation_2_4: flow[2,1] - flow[2,7] + flow[2,14] + flow[2,17]
   = 1
 flow_conservation_2_5: flow[2,3] - flow[2,6] + flow[2,11] - flow[2,16]
   - flow[2,17] - flow[2,18] = 1
 flow_conservation_2_6: flow[2,4] + flow[2,13] + flow[2,16] = 1
 flow_conservation_2_7: - flow[2,2] - flow[2,10] - flow[2,12] + flow[2,19]
   = 1
 flow_conservation_2_8: - flow[2,1] + flow[2,5] + flow[2,10] = 1
 flow_conservation_2_9: flow[2,0] + flow[2,2] - flow[2,13] = 1
 flow_conservation_3_0: - flow[3,5] + flow[3,7] + flow[3,9] - flow[3,19]
   = 1
 flow_conservation_3_1: - flow[3,4] + flow[3,6] - flow[3,8] - flow[3,9]
   = 1
 flow_conservation_3_2: - flow[3,0] - flow[3,11] - flow[3,14] - flow[3,15]
   = -9
 flow_conservation_3_3: - flow[3,3] + flow[3,8] + flow[3,12] + flow[3,15]
   + flow[3,18] = 1
 flow_conservation_3_4: flow[3,1] - flow[3,7] + flow[3,14] + flow[3,17]
   = 1
 flow_conservation_3_5: flow[3,3] - flow[3,6] + flow[3,11] - flow[3,16]
   - flow[3,17] - flow[3,18] = 1
 flow_conservation_3_6: flow[3,4] + flow[3,13] + flow[3,16] = 1
 flow_conservation_3_7: - flow[3,2] - flow[3,10] - flow[3,12] + flow[3,19]
   = 1
 flow_conservation_3_8: - flow[3,1] + flow[3,5] + flow[3,10] = 1
 flow_conservation_3_9: flow[3,0] + flow[3,2] - flow[3,13] = 1
 flow_conservation_4_0: - flow[4,5] + flow[4,7] + flow[4,9] - flow[4,19]
   = 1
 flow_conservation_4_1: - flow[4,4] + flow[4,6] - flow[4,8] - flow[4,9]
   = 1
 flow_conservation_4_2: - flow[4,0] - flow[4,11] - flow[4,14] - flow[4,15]
   = -9
 flow_conservation_4_3: - flow[4,3] + flow[4,8] + flow[4,12] + flow[4,15]
   + flow[4,18] = 1
 flow_conservation_4_4: flow[4,1] - flow[4,7] + flow[4,14] + flow[4,17]
   = 1
 flow_conservation_4_5: flow[4,3] - flow[4,6] + flow[4,11] - flow[4,16]
   - flow[4,17] - flow[4,18] = 1
 flow_conservation_4_6: flow[4,4] + flow[4,13] + flow[4,16] = 1
 flow_conservation_4_7: - flow[4,2] - flow[4,10] - flow[4,12] + flow[4,19]
   = 1
 flow_conservation_4_8: - flow[4,1] + flow[4,5] + flow[4,10] = 1
 flow_conservation_4_9: flow[4,0] + flow[4,2] - flow[4,13] = 1
Bounds
End
