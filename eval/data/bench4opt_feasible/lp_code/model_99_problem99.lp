\ Model Production Planning Problem with Ramp Rate Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  power_output[0,0] + power_output[0,1] + power_output[0,2]
   + power_output[0,3] + power_output[0,4] + power_output[0,5]
   + power_output[0,6] + power_output[0,7] + power_output[0,8]
   + power_output[0,9] + power_output[0,10] + power_output[0,11]
   + power_output[0,12] + power_output[0,13] + power_output[1,0]
   + power_output[1,1] + power_output[1,2] + power_output[1,3]
   + power_output[1,4] + power_output[1,5] + power_output[1,6]
   + power_output[1,7] + power_output[1,8] + power_output[1,9]
   + power_output[1,10] + power_output[1,11] + power_output[1,12]
   + power_output[1,13] + power_output[2,0] + power_output[2,1]
   + power_output[2,2] + power_output[2,3] + power_output[2,4]
   + power_output[2,5] + power_output[2,6] + power_output[2,7]
   + power_output[2,8] + power_output[2,9] + power_output[2,10]
   + power_output[2,11] + power_output[2,12] + power_output[2,13]
   + power_output[3,0] + power_output[3,1] + power_output[3,2]
   + power_output[3,3] + power_output[3,4] + power_output[3,5]
   + power_output[3,6] + power_output[3,7] + power_output[3,8]
   + power_output[3,9] + power_output[3,10] + power_output[3,11]
   + power_output[3,12] + power_output[3,13] + power_output[4,0]
   + power_output[4,1] + power_output[4,2] + power_output[4,3]
   + power_output[4,4] + power_output[4,5] + power_output[4,6]
   + power_output[4,7] + power_output[4,8] + power_output[4,9]
   + power_output[4,10] + power_output[4,11] + power_output[4,12]
   + power_output[4,13] + power_output[5,0] + power_output[5,1]
   + power_output[5,2] + power_output[5,3] + power_output[5,4]
   + power_output[5,5] + power_output[5,6] + power_output[5,7]
   + power_output[5,8] + power_output[5,9] + power_output[5,10]
   + power_output[5,11] + power_output[5,12] + power_output[5,13]
Subject To
 demand_0: power_output[0,0] + power_output[1,0] + power_output[2,0]
   + power_output[3,0] + power_output[4,0] + power_output[5,0] >= 1
 demand_1: power_output[0,1] + power_output[1,1] + power_output[2,1]
   + power_output[3,1] + power_output[4,1] + power_output[5,1] >= 1
 demand_2: power_output[0,2] + power_output[1,2] + power_output[2,2]
   + power_output[3,2] + power_output[4,2] + power_output[5,2] >= 1
 demand_3: power_output[0,3] + power_output[1,3] + power_output[2,3]
   + power_output[3,3] + power_output[4,3] + power_output[5,3] >= 1
 demand_4: power_output[0,4] + power_output[1,4] + power_output[2,4]
   + power_output[3,4] + power_output[4,4] + power_output[5,4] >= 1
 demand_5: power_output[0,5] + power_output[1,5] + power_output[2,5]
   + power_output[3,5] + power_output[4,5] + power_output[5,5] >= 1
 demand_6: power_output[0,6] + power_output[1,6] + power_output[2,6]
   + power_output[3,6] + power_output[4,6] + power_output[5,6] >= 1
 demand_7: power_output[0,7] + power_output[1,7] + power_output[2,7]
   + power_output[3,7] + power_output[4,7] + power_output[5,7] >= 1
 demand_8: power_output[0,8] + power_output[1,8] + power_output[2,8]
   + power_output[3,8] + power_output[4,8] + power_output[5,8] >= 1
 demand_9: power_output[0,9] + power_output[1,9] + power_output[2,9]
   + power_output[3,9] + power_output[4,9] + power_output[5,9] >= 1
 demand_10: power_output[0,10] + power_output[1,10] + power_output[2,10]
   + power_output[3,10] + power_output[4,10] + power_output[5,10] >= 1
 demand_11: power_output[0,11] + power_output[1,11] + power_output[2,11]
   + power_output[3,11] + power_output[4,11] + power_output[5,11] >= 1
 demand_12: power_output[0,12] + power_output[1,12] + power_output[2,12]
   + power_output[3,12] + power_output[4,12] + power_output[5,12] >= 1
 demand_13: power_output[0,13] + power_output[1,13] + power_output[2,13]
   + power_output[3,13] + power_output[4,13] + power_output[5,13] >= 1
 capacity_0_0: power_output[0,0] <= 1e+06
 capacity_0_1: power_output[0,1] <= 1e+06
 capacity_0_2: power_output[0,2] <= 1e+06
 capacity_0_3: power_output[0,3] <= 1e+06
 capacity_0_4: power_output[0,4] <= 1e+06
 capacity_0_5: power_output[0,5] <= 1e+06
 capacity_0_6: power_output[0,6] <= 1e+06
 capacity_0_7: power_output[0,7] <= 1e+06
 capacity_0_8: power_output[0,8] <= 1e+06
 capacity_0_9: power_output[0,9] <= 1e+06
 capacity_0_10: power_output[0,10] <= 1e+06
 capacity_0_11: power_output[0,11] <= 1e+06
 capacity_0_12: power_output[0,12] <= 1e+06
 capacity_0_13: power_output[0,13] <= 1e+06
 capacity_1_0: power_output[1,0] <= 1e+06
 capacity_1_1: power_output[1,1] <= 1e+06
 capacity_1_2: power_output[1,2] <= 1e+06
 capacity_1_3: power_output[1,3] <= 1e+06
 capacity_1_4: power_output[1,4] <= 1e+06
 capacity_1_5: power_output[1,5] <= 1e+06
 capacity_1_6: power_output[1,6] <= 1e+06
 capacity_1_7: power_output[1,7] <= 1e+06
 capacity_1_8: power_output[1,8] <= 1e+06
 capacity_1_9: power_output[1,9] <= 1e+06
 capacity_1_10: power_output[1,10] <= 1e+06
 capacity_1_11: power_output[1,11] <= 1e+06
 capacity_1_12: power_output[1,12] <= 1e+06
 capacity_1_13: power_output[1,13] <= 1e+06
 capacity_2_0: power_output[2,0] <= 1e+06
 capacity_2_1: power_output[2,1] <= 1e+06
 capacity_2_2: power_output[2,2] <= 1e+06
 capacity_2_3: power_output[2,3] <= 1e+06
 capacity_2_4: power_output[2,4] <= 1e+06
 capacity_2_5: power_output[2,5] <= 1e+06
 capacity_2_6: power_output[2,6] <= 1e+06
 capacity_2_7: power_output[2,7] <= 1e+06
 capacity_2_8: power_output[2,8] <= 1e+06
 capacity_2_9: power_output[2,9] <= 1e+06
 capacity_2_10: power_output[2,10] <= 1e+06
 capacity_2_11: power_output[2,11] <= 1e+06
 capacity_2_12: power_output[2,12] <= 1e+06
 capacity_2_13: power_output[2,13] <= 1e+06
 capacity_3_0: power_output[3,0] <= 1e+06
 capacity_3_1: power_output[3,1] <= 1e+06
 capacity_3_2: power_output[3,2] <= 1e+06
 capacity_3_3: power_output[3,3] <= 1e+06
 capacity_3_4: power_output[3,4] <= 1e+06
 capacity_3_5: power_output[3,5] <= 1e+06
 capacity_3_6: power_output[3,6] <= 1e+06
 capacity_3_7: power_output[3,7] <= 1e+06
 capacity_3_8: power_output[3,8] <= 1e+06
 capacity_3_9: power_output[3,9] <= 1e+06
 capacity_3_10: power_output[3,10] <= 1e+06
 capacity_3_11: power_output[3,11] <= 1e+06
 capacity_3_12: power_output[3,12] <= 1e+06
 capacity_3_13: power_output[3,13] <= 1e+06
 capacity_4_0: power_output[4,0] <= 1e+06
 capacity_4_1: power_output[4,1] <= 1e+06
 capacity_4_2: power_output[4,2] <= 1e+06
 capacity_4_3: power_output[4,3] <= 1e+06
 capacity_4_4: power_output[4,4] <= 1e+06
 capacity_4_5: power_output[4,5] <= 1e+06
 capacity_4_6: power_output[4,6] <= 1e+06
 capacity_4_7: power_output[4,7] <= 1e+06
 capacity_4_8: power_output[4,8] <= 1e+06
 capacity_4_9: power_output[4,9] <= 1e+06
 capacity_4_10: power_output[4,10] <= 1e+06
 capacity_4_11: power_output[4,11] <= 1e+06
 capacity_4_12: power_output[4,12] <= 1e+06
 capacity_4_13: power_output[4,13] <= 1e+06
 capacity_5_0: power_output[5,0] <= 1e+06
 capacity_5_1: power_output[5,1] <= 1e+06
 capacity_5_2: power_output[5,2] <= 1e+06
 capacity_5_3: power_output[5,3] <= 1e+06
 capacity_5_4: power_output[5,4] <= 1e+06
 capacity_5_5: power_output[5,5] <= 1e+06
 capacity_5_6: power_output[5,6] <= 1e+06
 capacity_5_7: power_output[5,7] <= 1e+06
 capacity_5_8: power_output[5,8] <= 1e+06
 capacity_5_9: power_output[5,9] <= 1e+06
 capacity_5_10: power_output[5,10] <= 1e+06
 capacity_5_11: power_output[5,11] <= 1e+06
 capacity_5_12: power_output[5,12] <= 1e+06
 capacity_5_13: power_output[5,13] <= 1e+06
 emission_cap: 1.948448013637469 power_output[0,0]
   + 1.948448013637469 power_output[0,1]
   + 1.948448013637469 power_output[0,2]
   + 1.948448013637469 power_output[0,3]
   + 1.948448013637469 power_output[0,4]
   + 1.948448013637469 power_output[0,5]
   + 1.948448013637469 power_output[0,6]
   + 1.948448013637469 power_output[0,7]
   + 1.948448013637469 power_output[0,8]
   + 1.948448013637469 power_output[0,9]
   + 1.948448013637469 power_output[0,10]
   + 1.948448013637469 power_output[0,11]
   + 1.948448013637469 power_output[0,12]
   + 1.948448013637469 power_output[0,13]
   + 1.912718456196204 power_output[1,0]
   + 1.912718456196204 power_output[1,1]
   + 1.912718456196204 power_output[1,2]
   + 1.912718456196204 power_output[1,3]
   + 1.912718456196204 power_output[1,4]
   + 1.912718456196204 power_output[1,5]
   + 1.912718456196204 power_output[1,6]
   + 1.912718456196204 power_output[1,7]
   + 1.912718456196204 power_output[1,8]
   + 1.912718456196204 power_output[1,9]
   + 1.912718456196204 power_output[1,10]
   + 1.912718456196204 power_output[1,11]
   + 1.912718456196204 power_output[1,12]
   + 1.912718456196204 power_output[1,13]
   + 1.569122730345508 power_output[2,0]
   + 1.569122730345508 power_output[2,1]
   + 1.569122730345508 power_output[2,2]
   + 1.569122730345508 power_output[2,3]
   + 1.569122730345508 power_output[2,4]
   + 1.569122730345508 power_output[2,5]
   + 1.569122730345508 power_output[2,6]
   + 1.569122730345508 power_output[2,7]
   + 1.569122730345508 power_output[2,8]
   + 1.569122730345508 power_output[2,9]
   + 1.569122730345508 power_output[2,10]
   + 1.569122730345508 power_output[2,11]
   + 1.569122730345508 power_output[2,12]
   + 1.569122730345508 power_output[2,13]
   + 1.524220926732872 power_output[3,0]
   + 1.524220926732872 power_output[3,1]
   + 1.524220926732872 power_output[3,2]
   + 1.524220926732872 power_output[3,3]
   + 1.524220926732872 power_output[3,4]
   + 1.524220926732872 power_output[3,5]
   + 1.524220926732872 power_output[3,6]
   + 1.524220926732872 power_output[3,7]
   + 1.524220926732872 power_output[3,8]
   + 1.524220926732872 power_output[3,9]
   + 1.524220926732872 power_output[3,10]
   + 1.524220926732872 power_output[3,11]
   + 1.524220926732872 power_output[3,12]
   + 1.524220926732872 power_output[3,13]
   + 1.275221478847895 power_output[4,0]
   + 1.275221478847895 power_output[4,1]
   + 1.275221478847895 power_output[4,2]
   + 1.275221478847895 power_output[4,3]
   + 1.275221478847895 power_output[4,4]
   + 1.275221478847895 power_output[4,5]
   + 1.275221478847895 power_output[4,6]
   + 1.275221478847895 power_output[4,7]
   + 1.275221478847895 power_output[4,8]
   + 1.275221478847895 power_output[4,9]
   + 1.275221478847895 power_output[4,10]
   + 1.275221478847895 power_output[4,11]
   + 1.275221478847895 power_output[4,12]
   + 1.275221478847895 power_output[4,13]
   + 1.705125910685309 power_output[5,0]
   + 1.705125910685309 power_output[5,1]
   + 1.705125910685309 power_output[5,2]
   + 1.705125910685309 power_output[5,3]
   + 1.705125910685309 power_output[5,4]
   + 1.705125910685309 power_output[5,5]
   + 1.705125910685309 power_output[5,6]
   + 1.705125910685309 power_output[5,7]
   + 1.705125910685309 power_output[5,8]
   + 1.705125910685309 power_output[5,9]
   + 1.705125910685309 power_output[5,10]
   + 1.705125910685309 power_output[5,11]
   + 1.705125910685309 power_output[5,12]
   + 1.705125910685309 power_output[5,13] <= 1e+06
 ramp_up_0_1: - power_output[0,0] + power_output[0,1] <= 228.9708956654567
 ramp_down_0_1: power_output[0,0] - power_output[0,1] <= 321.7626731397392
 ramp_up_0_2: - power_output[0,1] + power_output[0,2] <= 228.9708956654567
 ramp_down_0_2: power_output[0,1] - power_output[0,2] <= 321.7626731397392
 ramp_up_0_3: - power_output[0,2] + power_output[0,3] <= 228.9708956654567
 ramp_down_0_3: power_output[0,2] - power_output[0,3] <= 321.7626731397392
 ramp_up_0_4: - power_output[0,3] + power_output[0,4] <= 228.9708956654567
 ramp_down_0_4: power_output[0,3] - power_output[0,4] <= 321.7626731397392
 ramp_up_0_5: - power_output[0,4] + power_output[0,5] <= 228.9708956654567
 ramp_down_0_5: power_output[0,4] - power_output[0,5] <= 321.7626731397392
 ramp_up_0_6: - power_output[0,5] + power_output[0,6] <= 228.9708956654567
 ramp_down_0_6: power_output[0,5] - power_output[0,6] <= 321.7626731397392
 ramp_up_0_7: - power_output[0,6] + power_output[0,7] <= 228.9708956654567
 ramp_down_0_7: power_output[0,6] - power_output[0,7] <= 321.7626731397392
 ramp_up_0_8: - power_output[0,7] + power_output[0,8] <= 228.9708956654567
 ramp_down_0_8: power_output[0,7] - power_output[0,8] <= 321.7626731397392
 ramp_up_0_9: - power_output[0,8] + power_output[0,9] <= 228.9708956654567
 ramp_down_0_9: power_output[0,8] - power_output[0,9] <= 321.7626731397392
 ramp_up_0_10: - power_output[0,9] + power_output[0,10]
   <= 228.9708956654567
 ramp_down_0_10: power_output[0,9] - power_output[0,10]
   <= 321.7626731397392
 ramp_up_0_11: - power_output[0,10] + power_output[0,11]
   <= 228.9708956654567
 ramp_down_0_11: power_output[0,10] - power_output[0,11]
   <= 321.7626731397392
 ramp_up_0_12: - power_output[0,11] + power_output[0,12]
   <= 228.9708956654567
 ramp_down_0_12: power_output[0,11] - power_output[0,12]
   <= 321.7626731397392
 ramp_up_0_13: - power_output[0,12] + power_output[0,13]
   <= 228.9708956654567
 ramp_down_0_13: power_output[0,12] - power_output[0,13]
   <= 321.7626731397392
 initial_ramp_up_0: power_output[0,0] <= 798.8116496139123
 initial_ramp_down_0: - power_output[0,0] <= -248.0780808087164
 ramp_up_1_1: - power_output[1,0] + power_output[1,1] <= 118.2892379019528
 ramp_down_1_1: power_output[1,0] - power_output[1,1] <= 297.129063142002
 ramp_up_1_2: - power_output[1,1] + power_output[1,2] <= 118.2892379019528
 ramp_down_1_2: power_output[1,1] - power_output[1,2] <= 297.129063142002
 ramp_up_1_3: - power_output[1,2] + power_output[1,3] <= 118.2892379019528
 ramp_down_1_3: power_output[1,2] - power_output[1,3] <= 297.129063142002
 ramp_up_1_4: - power_output[1,3] + power_output[1,4] <= 118.2892379019528
 ramp_down_1_4: power_output[1,3] - power_output[1,4] <= 297.129063142002
 ramp_up_1_5: - power_output[1,4] + power_output[1,5] <= 118.2892379019528
 ramp_down_1_5: power_output[1,4] - power_output[1,5] <= 297.129063142002
 ramp_up_1_6: - power_output[1,5] + power_output[1,6] <= 118.2892379019528
 ramp_down_1_6: power_output[1,5] - power_output[1,6] <= 297.129063142002
 ramp_up_1_7: - power_output[1,6] + power_output[1,7] <= 118.2892379019528
 ramp_down_1_7: power_output[1,6] - power_output[1,7] <= 297.129063142002
 ramp_up_1_8: - power_output[1,7] + power_output[1,8] <= 118.2892379019528
 ramp_down_1_8: power_output[1,7] - power_output[1,8] <= 297.129063142002
 ramp_up_1_9: - power_output[1,8] + power_output[1,9] <= 118.2892379019528
 ramp_down_1_9: power_output[1,8] - power_output[1,9] <= 297.129063142002
 ramp_up_1_10: - power_output[1,9] + power_output[1,10]
   <= 118.2892379019528
 ramp_down_1_10: power_output[1,9] - power_output[1,10]
   <= 297.129063142002
 ramp_up_1_11: - power_output[1,10] + power_output[1,11]
   <= 118.2892379019528
 ramp_down_1_11: power_output[1,10] - power_output[1,11]
   <= 297.129063142002
 ramp_up_1_12: - power_output[1,11] + power_output[1,12]
   <= 118.2892379019528
 ramp_down_1_12: power_output[1,11] - power_output[1,12]
   <= 297.129063142002
 ramp_up_1_13: - power_output[1,12] + power_output[1,13]
   <= 118.2892379019528
 ramp_down_1_13: power_output[1,12] - power_output[1,13]
   <= 297.129063142002
 initial_ramp_up_1: power_output[1,0] <= 714.6773708857385
 initial_ramp_down_1: - power_output[1,0] <= -299.2590698417837
 ramp_up_2_1: - power_output[2,0] + power_output[2,1] <= 490.7787910950464
 ramp_down_2_1: power_output[2,0] - power_output[2,1] <= 479.347234500992
 ramp_up_2_2: - power_output[2,1] + power_output[2,2] <= 490.7787910950464
 ramp_down_2_2: power_output[2,1] - power_output[2,2] <= 479.347234500992
 ramp_up_2_3: - power_output[2,2] + power_output[2,3] <= 490.7787910950464
 ramp_down_2_3: power_output[2,2] - power_output[2,3] <= 479.347234500992
 ramp_up_2_4: - power_output[2,3] + power_output[2,4] <= 490.7787910950464
 ramp_down_2_4: power_output[2,3] - power_output[2,4] <= 479.347234500992
 ramp_up_2_5: - power_output[2,4] + power_output[2,5] <= 490.7787910950464
 ramp_down_2_5: power_output[2,4] - power_output[2,5] <= 479.347234500992
 ramp_up_2_6: - power_output[2,5] + power_output[2,6] <= 490.7787910950464
 ramp_down_2_6: power_output[2,5] - power_output[2,6] <= 479.347234500992
 ramp_up_2_7: - power_output[2,6] + power_output[2,7] <= 490.7787910950464
 ramp_down_2_7: power_output[2,6] - power_output[2,7] <= 479.347234500992
 ramp_up_2_8: - power_output[2,7] + power_output[2,8] <= 490.7787910950464
 ramp_down_2_8: power_output[2,7] - power_output[2,8] <= 479.347234500992
 ramp_up_2_9: - power_output[2,8] + power_output[2,9] <= 490.7787910950464
 ramp_down_2_9: power_output[2,8] - power_output[2,9] <= 479.347234500992
 ramp_up_2_10: - power_output[2,9] + power_output[2,10]
   <= 490.7787910950464
 ramp_down_2_10: power_output[2,9] - power_output[2,10]
   <= 479.347234500992
 ramp_up_2_11: - power_output[2,10] + power_output[2,11]
   <= 490.7787910950464
 ramp_down_2_11: power_output[2,10] - power_output[2,11]
   <= 479.347234500992
 ramp_up_2_12: - power_output[2,11] + power_output[2,12]
   <= 490.7787910950464
 ramp_down_2_12: power_output[2,11] - power_output[2,12]
   <= 479.347234500992
 ramp_up_2_13: - power_output[2,12] + power_output[2,13]
   <= 490.7787910950464
 ramp_down_2_13: power_output[2,12] - power_output[2,13]
   <= 479.347234500992
 initial_ramp_up_2: power_output[2,0] <= 595.4005301619953
 initial_ramp_down_2: - power_output[2,0] <= 374.7254954340431
 ramp_up_3_1: - power_output[3,0] + power_output[3,1] <= 252.87505458636
 ramp_down_3_1: power_output[3,0] - power_output[3,1] <= 352.7254230742298
 ramp_up_3_2: - power_output[3,1] + power_output[3,2] <= 252.87505458636
 ramp_down_3_2: power_output[3,1] - power_output[3,2] <= 352.7254230742298
 ramp_up_3_3: - power_output[3,2] + power_output[3,3] <= 252.87505458636
 ramp_down_3_3: power_output[3,2] - power_output[3,3] <= 352.7254230742298
 ramp_up_3_4: - power_output[3,3] + power_output[3,4] <= 252.87505458636
 ramp_down_3_4: power_output[3,3] - power_output[3,4] <= 352.7254230742298
 ramp_up_3_5: - power_output[3,4] + power_output[3,5] <= 252.87505458636
 ramp_down_3_5: power_output[3,4] - power_output[3,5] <= 352.7254230742298
 ramp_up_3_6: - power_output[3,5] + power_output[3,6] <= 252.87505458636
 ramp_down_3_6: power_output[3,5] - power_output[3,6] <= 352.7254230742298
 ramp_up_3_7: - power_output[3,6] + power_output[3,7] <= 252.87505458636
 ramp_down_3_7: power_output[3,6] - power_output[3,7] <= 352.7254230742298
 ramp_up_3_8: - power_output[3,7] + power_output[3,8] <= 252.87505458636
 ramp_down_3_8: power_output[3,7] - power_output[3,8] <= 352.7254230742298
 ramp_up_3_9: - power_output[3,8] + power_output[3,9] <= 252.87505458636
 ramp_down_3_9: power_output[3,8] - power_output[3,9] <= 352.7254230742298
 ramp_up_3_10: - power_output[3,9] + power_output[3,10] <= 252.87505458636
 ramp_down_3_10: power_output[3,9] - power_output[3,10]
   <= 352.7254230742298
 ramp_up_3_11: - power_output[3,10] + power_output[3,11]
   <= 252.87505458636
 ramp_down_3_11: power_output[3,10] - power_output[3,11]
   <= 352.7254230742298
 ramp_up_3_12: - power_output[3,11] + power_output[3,12]
   <= 252.87505458636
 ramp_down_3_12: power_output[3,11] - power_output[3,12]
   <= 352.7254230742298
 ramp_up_3_13: - power_output[3,12] + power_output[3,13]
   <= 252.87505458636
 ramp_down_3_13: power_output[3,12] - power_output[3,13]
   <= 352.7254230742298
 initial_ramp_up_3: power_output[3,0] <= 438.810600590778
 initial_ramp_down_3: - power_output[3,0] <= 166.7898770698118
 ramp_up_4_1: - power_output[4,0] + power_output[4,1] <= 204.6354999809793
 ramp_down_4_1: power_output[4,0] - power_output[4,1] <= 279.0580533757618
 ramp_up_4_2: - power_output[4,1] + power_output[4,2] <= 204.6354999809793
 ramp_down_4_2: power_output[4,1] - power_output[4,2] <= 279.0580533757618
 ramp_up_4_3: - power_output[4,2] + power_output[4,3] <= 204.6354999809793
 ramp_down_4_3: power_output[4,2] - power_output[4,3] <= 279.0580533757618
 ramp_up_4_4: - power_output[4,3] + power_output[4,4] <= 204.6354999809793
 ramp_down_4_4: power_output[4,3] - power_output[4,4] <= 279.0580533757618
 ramp_up_4_5: - power_output[4,4] + power_output[4,5] <= 204.6354999809793
 ramp_down_4_5: power_output[4,4] - power_output[4,5] <= 279.0580533757618
 ramp_up_4_6: - power_output[4,5] + power_output[4,6] <= 204.6354999809793
 ramp_down_4_6: power_output[4,5] - power_output[4,6] <= 279.0580533757618
 ramp_up_4_7: - power_output[4,6] + power_output[4,7] <= 204.6354999809793
 ramp_down_4_7: power_output[4,6] - power_output[4,7] <= 279.0580533757618
 ramp_up_4_8: - power_output[4,7] + power_output[4,8] <= 204.6354999809793
 ramp_down_4_8: power_output[4,7] - power_output[4,8] <= 279.0580533757618
 ramp_up_4_9: - power_output[4,8] + power_output[4,9] <= 204.6354999809793
 ramp_down_4_9: power_output[4,8] - power_output[4,9] <= 279.0580533757618
 ramp_up_4_10: - power_output[4,9] + power_output[4,10]
   <= 204.6354999809793
 ramp_down_4_10: power_output[4,9] - power_output[4,10]
   <= 279.0580533757618
 ramp_up_4_11: - power_output[4,10] + power_output[4,11]
   <= 204.6354999809793
 ramp_down_4_11: power_output[4,10] - power_output[4,11]
   <= 279.0580533757618
 ramp_up_4_12: - power_output[4,11] + power_output[4,12]
   <= 204.6354999809793
 ramp_down_4_12: power_output[4,11] - power_output[4,12]
   <= 279.0580533757618
 ramp_up_4_13: - power_output[4,12] + power_output[4,13]
   <= 204.6354999809793
 ramp_down_4_13: power_output[4,12] - power_output[4,13]
   <= 279.0580533757618
 initial_ramp_up_4: power_output[4,0] <= 388.8233596397764
 initial_ramp_down_4: - power_output[4,0] <= 94.87019371696465
 ramp_up_5_1: - power_output[5,0] + power_output[5,1] <= 380.4642472408214
 ramp_down_5_1: power_output[5,0] - power_output[5,1] <= 197.4407015945449
 ramp_up_5_2: - power_output[5,1] + power_output[5,2] <= 380.4642472408214
 ramp_down_5_2: power_output[5,1] - power_output[5,2] <= 197.4407015945449
 ramp_up_5_3: - power_output[5,2] + power_output[5,3] <= 380.4642472408214
 ramp_down_5_3: power_output[5,2] - power_output[5,3] <= 197.4407015945449
 ramp_up_5_4: - power_output[5,3] + power_output[5,4] <= 380.4642472408214
 ramp_down_5_4: power_output[5,3] - power_output[5,4] <= 197.4407015945449
 ramp_up_5_5: - power_output[5,4] + power_output[5,5] <= 380.4642472408214
 ramp_down_5_5: power_output[5,4] - power_output[5,5] <= 197.4407015945449
 ramp_up_5_6: - power_output[5,5] + power_output[5,6] <= 380.4642472408214
 ramp_down_5_6: power_output[5,5] - power_output[5,6] <= 197.4407015945449
 ramp_up_5_7: - power_output[5,6] + power_output[5,7] <= 380.4642472408214
 ramp_down_5_7: power_output[5,6] - power_output[5,7] <= 197.4407015945449
 ramp_up_5_8: - power_output[5,7] + power_output[5,8] <= 380.4642472408214
 ramp_down_5_8: power_output[5,7] - power_output[5,8] <= 197.4407015945449
 ramp_up_5_9: - power_output[5,8] + power_output[5,9] <= 380.4642472408214
 ramp_down_5_9: power_output[5,8] - power_output[5,9] <= 197.4407015945449
 ramp_up_5_10: - power_output[5,9] + power_output[5,10]
   <= 380.4642472408214
 ramp_down_5_10: power_output[5,9] - power_output[5,10]
   <= 197.4407015945449
 ramp_up_5_11: - power_output[5,10] + power_output[5,11]
   <= 380.4642472408214
 ramp_down_5_11: power_output[5,10] - power_output[5,11]
   <= 197.4407015945449
 ramp_up_5_12: - power_output[5,11] + power_output[5,12]
   <= 380.4642472408214
 ramp_down_5_12: power_output[5,11] - power_output[5,12]
   <= 197.4407015945449
 ramp_up_5_13: - power_output[5,12] + power_output[5,13]
   <= 380.4642472408214
 ramp_down_5_13: power_output[5,12] - power_output[5,13]
   <= 197.4407015945449
 initial_ramp_up_5: power_output[5,0] <= 941.4754654886585
 initial_ramp_down_5: - power_output[5,0] <= -363.5705166532922
Bounds
End
