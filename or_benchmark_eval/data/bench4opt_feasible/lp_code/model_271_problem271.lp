\ Model Production Planning Problem with Time-of-Use Pricing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  power_output[0,0] + power_output[0,1] + power_output[0,2]
   + power_output[0,3] + power_output[0,4] + power_output[0,5]
   + power_output[0,6] + power_output[0,7] + power_output[0,8]
   + power_output[0,9] + power_output[0,10] + power_output[0,11]
   + power_output[0,12] + power_output[0,13] + power_output[0,14]
   + power_output[1,0] + power_output[1,1] + power_output[1,2]
   + power_output[1,3] + power_output[1,4] + power_output[1,5]
   + power_output[1,6] + power_output[1,7] + power_output[1,8]
   + power_output[1,9] + power_output[1,10] + power_output[1,11]
   + power_output[1,12] + power_output[1,13] + power_output[1,14]
   + power_output[2,0] + power_output[2,1] + power_output[2,2]
   + power_output[2,3] + power_output[2,4] + power_output[2,5]
   + power_output[2,6] + power_output[2,7] + power_output[2,8]
   + power_output[2,9] + power_output[2,10] + power_output[2,11]
   + power_output[2,12] + power_output[2,13] + power_output[2,14]
   + power_output[3,0] + power_output[3,1] + power_output[3,2]
   + power_output[3,3] + power_output[3,4] + power_output[3,5]
   + power_output[3,6] + power_output[3,7] + power_output[3,8]
   + power_output[3,9] + power_output[3,10] + power_output[3,11]
   + power_output[3,12] + power_output[3,13] + power_output[3,14]
   + power_output[4,0] + power_output[4,1] + power_output[4,2]
   + power_output[4,3] + power_output[4,4] + power_output[4,5]
   + power_output[4,6] + power_output[4,7] + power_output[4,8]
   + power_output[4,9] + power_output[4,10] + power_output[4,11]
   + power_output[4,12] + power_output[4,13] + power_output[4,14]
   + power_output[5,0] + power_output[5,1] + power_output[5,2]
   + power_output[5,3] + power_output[5,4] + power_output[5,5]
   + power_output[5,6] + power_output[5,7] + power_output[5,8]
   + power_output[5,9] + power_output[5,10] + power_output[5,11]
   + power_output[5,12] + power_output[5,13] + power_output[5,14]
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
 demand_14: power_output[0,14] + power_output[1,14] + power_output[2,14]
   + power_output[3,14] + power_output[4,14] + power_output[5,14] >= 1
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
 capacity_0_14: power_output[0,14] <= 1e+06
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
 capacity_1_14: power_output[1,14] <= 1e+06
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
 capacity_2_14: power_output[2,14] <= 1e+06
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
 capacity_3_14: power_output[3,14] <= 1e+06
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
 capacity_4_14: power_output[4,14] <= 1e+06
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
 capacity_5_14: power_output[5,14] <= 1e+06
 emission_cap_0: 1.559626520851306 power_output[0,0]
   + 2.796194280269926 power_output[1,0]
   + 2.397486243656689 power_output[2,0]
   + 2.391119827615344 power_output[3,0]
   + 1.456467183664871 power_output[4,0]
   + 2.05563046633204 power_output[5,0] <= 1e+06
 emission_cap_1: 1.559626520851306 power_output[0,1]
   + 2.796194280269926 power_output[1,1]
   + 2.397486243656689 power_output[2,1]
   + 2.391119827615344 power_output[3,1]
   + 1.456467183664871 power_output[4,1]
   + 2.05563046633204 power_output[5,1] <= 1e+06
 emission_cap_2: 1.559626520851306 power_output[0,2]
   + 2.796194280269926 power_output[1,2]
   + 2.397486243656689 power_output[2,2]
   + 2.391119827615344 power_output[3,2]
   + 1.456467183664871 power_output[4,2]
   + 2.05563046633204 power_output[5,2] <= 1e+06
 emission_cap_3: 1.559626520851306 power_output[0,3]
   + 2.796194280269926 power_output[1,3]
   + 2.397486243656689 power_output[2,3]
   + 2.391119827615344 power_output[3,3]
   + 1.456467183664871 power_output[4,3]
   + 2.05563046633204 power_output[5,3] <= 1e+06
 emission_cap_4: 1.559626520851306 power_output[0,4]
   + 2.796194280269926 power_output[1,4]
   + 2.397486243656689 power_output[2,4]
   + 2.391119827615344 power_output[3,4]
   + 1.456467183664871 power_output[4,4]
   + 2.05563046633204 power_output[5,4] <= 1e+06
 emission_cap_5: 1.559626520851306 power_output[0,5]
   + 2.796194280269926 power_output[1,5]
   + 2.397486243656689 power_output[2,5]
   + 2.391119827615344 power_output[3,5]
   + 1.456467183664871 power_output[4,5]
   + 2.05563046633204 power_output[5,5] <= 1e+06
 emission_cap_6: 1.559626520851306 power_output[0,6]
   + 2.796194280269926 power_output[1,6]
   + 2.397486243656689 power_output[2,6]
   + 2.391119827615344 power_output[3,6]
   + 1.456467183664871 power_output[4,6]
   + 2.05563046633204 power_output[5,6] <= 1e+06
 emission_cap_7: 1.559626520851306 power_output[0,7]
   + 2.796194280269926 power_output[1,7]
   + 2.397486243656689 power_output[2,7]
   + 2.391119827615344 power_output[3,7]
   + 1.456467183664871 power_output[4,7]
   + 2.05563046633204 power_output[5,7] <= 1e+06
 emission_cap_8: 1.559626520851306 power_output[0,8]
   + 2.796194280269926 power_output[1,8]
   + 2.397486243656689 power_output[2,8]
   + 2.391119827615344 power_output[3,8]
   + 1.456467183664871 power_output[4,8]
   + 2.05563046633204 power_output[5,8] <= 1e+06
 emission_cap_9: 1.559626520851306 power_output[0,9]
   + 2.796194280269926 power_output[1,9]
   + 2.397486243656689 power_output[2,9]
   + 2.391119827615344 power_output[3,9]
   + 1.456467183664871 power_output[4,9]
   + 2.05563046633204 power_output[5,9] <= 1e+06
 emission_cap_10: 1.559626520851306 power_output[0,10]
   + 2.796194280269926 power_output[1,10]
   + 2.397486243656689 power_output[2,10]
   + 2.391119827615344 power_output[3,10]
   + 1.456467183664871 power_output[4,10]
   + 2.05563046633204 power_output[5,10] <= 1e+06
 emission_cap_11: 1.559626520851306 power_output[0,11]
   + 2.796194280269926 power_output[1,11]
   + 2.397486243656689 power_output[2,11]
   + 2.391119827615344 power_output[3,11]
   + 1.456467183664871 power_output[4,11]
   + 2.05563046633204 power_output[5,11] <= 1e+06
 emission_cap_12: 1.559626520851306 power_output[0,12]
   + 2.796194280269926 power_output[1,12]
   + 2.397486243656689 power_output[2,12]
   + 2.391119827615344 power_output[3,12]
   + 1.456467183664871 power_output[4,12]
   + 2.05563046633204 power_output[5,12] <= 1e+06
 emission_cap_13: 1.559626520851306 power_output[0,13]
   + 2.796194280269926 power_output[1,13]
   + 2.397486243656689 power_output[2,13]
   + 2.391119827615344 power_output[3,13]
   + 1.456467183664871 power_output[4,13]
   + 2.05563046633204 power_output[5,13] <= 1e+06
 emission_cap_14: 1.559626520851306 power_output[0,14]
   + 2.796194280269926 power_output[1,14]
   + 2.397486243656689 power_output[2,14]
   + 2.391119827615344 power_output[3,14]
   + 1.456467183664871 power_output[4,14]
   + 2.05563046633204 power_output[5,14] <= 1e+06
Bounds
End
