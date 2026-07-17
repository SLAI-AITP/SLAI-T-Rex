\ Model Production Planning Problem with Ramp Rate Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  19.87284852200974 power_output[0,0] + 19.87284852200974 power_output[0,1]
   + 19.87284852200974 power_output[0,2]
   + 19.87284852200974 power_output[0,3]
   + 19.87284852200974 power_output[0,4]
   + 19.87284852200974 power_output[0,5]
   + 19.87284852200974 power_output[0,6]
   + 19.87284852200974 power_output[0,7]
   + 19.87284852200974 power_output[0,8]
   + 19.87284852200974 power_output[0,9]
   + 19.87284852200974 power_output[0,10]
   + 19.87284852200974 power_output[0,11]
   + 19.87284852200974 power_output[0,12]
   + 19.87284852200974 power_output[0,13]
   + 35.39883217149352 power_output[1,0]
   + 35.39883217149352 power_output[1,1]
   + 35.39883217149352 power_output[1,2]
   + 35.39883217149352 power_output[1,3]
   + 35.39883217149352 power_output[1,4]
   + 35.39883217149352 power_output[1,5]
   + 35.39883217149352 power_output[1,6]
   + 35.39883217149352 power_output[1,7]
   + 35.39883217149352 power_output[1,8]
   + 35.39883217149352 power_output[1,9]
   + 35.39883217149352 power_output[1,10]
   + 35.39883217149352 power_output[1,11]
   + 35.39883217149352 power_output[1,12]
   + 35.39883217149352 power_output[1,13]
   + 12.80662679055113 power_output[2,0]
   + 12.80662679055113 power_output[2,1]
   + 12.80662679055113 power_output[2,2]
   + 12.80662679055113 power_output[2,3]
   + 12.80662679055113 power_output[2,4]
   + 12.80662679055113 power_output[2,5]
   + 12.80662679055113 power_output[2,6]
   + 12.80662679055113 power_output[2,7]
   + 12.80662679055113 power_output[2,8]
   + 12.80662679055113 power_output[2,9]
   + 12.80662679055113 power_output[2,10]
   + 12.80662679055113 power_output[2,11]
   + 12.80662679055113 power_output[2,12]
   + 12.80662679055113 power_output[2,13]
   + 48.85439510047412 power_output[3,0]
   + 48.85439510047412 power_output[3,1]
   + 48.85439510047412 power_output[3,2]
   + 48.85439510047412 power_output[3,3]
   + 48.85439510047412 power_output[3,4]
   + 48.85439510047412 power_output[3,5]
   + 48.85439510047412 power_output[3,6]
   + 48.85439510047412 power_output[3,7]
   + 48.85439510047412 power_output[3,8]
   + 48.85439510047412 power_output[3,9]
   + 48.85439510047412 power_output[3,10]
   + 48.85439510047412 power_output[3,11]
   + 48.85439510047412 power_output[3,12]
   + 48.85439510047412 power_output[3,13]
   + 63.11474871251098 power_output[4,0]
   + 63.11474871251098 power_output[4,1]
   + 63.11474871251098 power_output[4,2]
   + 63.11474871251098 power_output[4,3]
   + 63.11474871251098 power_output[4,4]
   + 63.11474871251098 power_output[4,5]
   + 63.11474871251098 power_output[4,6]
   + 63.11474871251098 power_output[4,7]
   + 63.11474871251098 power_output[4,8]
   + 63.11474871251098 power_output[4,9]
   + 63.11474871251098 power_output[4,10]
   + 63.11474871251098 power_output[4,11]
   + 63.11474871251098 power_output[4,12]
   + 63.11474871251098 power_output[4,13]
   + 66.26928730779483 power_output[5,0]
   + 66.26928730779483 power_output[5,1]
   + 66.26928730779483 power_output[5,2]
   + 66.26928730779483 power_output[5,3]
   + 66.26928730779483 power_output[5,4]
   + 66.26928730779483 power_output[5,5]
   + 66.26928730779483 power_output[5,6]
   + 66.26928730779483 power_output[5,7]
   + 66.26928730779483 power_output[5,8]
   + 66.26928730779483 power_output[5,9]
   + 66.26928730779483 power_output[5,10]
   + 66.26928730779483 power_output[5,11]
   + 66.26928730779483 power_output[5,12]
   + 66.26928730779483 power_output[5,13]
Subject To
 demand_0: power_output[0,0] + power_output[1,0] + power_output[2,0]
   + power_output[3,0] + power_output[4,0] + power_output[5,0]
   >= 15676.60477565113
 demand_1: power_output[0,1] + power_output[1,1] + power_output[2,1]
   + power_output[3,1] + power_output[4,1] + power_output[5,1]
   >= 11090.48981758561
 demand_2: power_output[0,2] + power_output[1,2] + power_output[2,2]
   + power_output[3,2] + power_output[4,2] + power_output[5,2]
   >= 11527.85072607384
 demand_3: power_output[0,3] + power_output[1,3] + power_output[2,3]
   + power_output[3,3] + power_output[4,3] + power_output[5,3]
   >= 8168.241120407829
 demand_4: power_output[0,4] + power_output[1,4] + power_output[2,4]
   + power_output[3,4] + power_output[4,4] + power_output[5,4]
   >= 7601.151593946451
 demand_5: power_output[0,5] + power_output[1,5] + power_output[2,5]
   + power_output[3,5] + power_output[4,5] + power_output[5,5]
   >= 7086.032985325177
 demand_6: power_output[0,6] + power_output[1,6] + power_output[2,6]
   + power_output[3,6] + power_output[4,6] + power_output[5,6]
   >= 13331.32368154023
 demand_7: power_output[0,7] + power_output[1,7] + power_output[2,7]
   + power_output[3,7] + power_output[4,7] + power_output[5,7]
   >= 17649.91506090187
 demand_8: power_output[0,8] + power_output[1,8] + power_output[2,8]
   + power_output[3,8] + power_output[4,8] + power_output[5,8]
   >= 10404.31832968772
 demand_9: power_output[0,9] + power_output[1,9] + power_output[2,9]
   + power_output[3,9] + power_output[4,9] + power_output[5,9]
   >= 11258.53242214031
 demand_10: power_output[0,10] + power_output[1,10] + power_output[2,10]
   + power_output[3,10] + power_output[4,10] + power_output[5,10]
   >= 9327.063075913789
 demand_11: power_output[0,11] + power_output[1,11] + power_output[2,11]
   + power_output[3,11] + power_output[4,11] + power_output[5,11]
   >= 18091.03537388884
 demand_12: power_output[0,12] + power_output[1,12] + power_output[2,12]
   + power_output[3,12] + power_output[4,12] + power_output[5,12]
   >= 6883.237302408827
 demand_13: power_output[0,13] + power_output[1,13] + power_output[2,13]
   + power_output[3,13] + power_output[4,13] + power_output[5,13]
   >= 5174.041286678536
 capacity_0_0: power_output[0,0] <= 1561.695026777729
 capacity_0_1: power_output[0,1] <= 1561.695026777729
 capacity_0_2: power_output[0,2] <= 1561.695026777729
 capacity_0_3: power_output[0,3] <= 1561.695026777729
 capacity_0_4: power_output[0,4] <= 1561.695026777729
 capacity_0_5: power_output[0,5] <= 1561.695026777729
 capacity_0_6: power_output[0,6] <= 1561.695026777729
 capacity_0_7: power_output[0,7] <= 1561.695026777729
 capacity_0_8: power_output[0,8] <= 1561.695026777729
 capacity_0_9: power_output[0,9] <= 1561.695026777729
 capacity_0_10: power_output[0,10] <= 1561.695026777729
 capacity_0_11: power_output[0,11] <= 1561.695026777729
 capacity_0_12: power_output[0,12] <= 1561.695026777729
 capacity_0_13: power_output[0,13] <= 1561.695026777729
 capacity_1_0: power_output[1,0] <= 9109.681255668149
 capacity_1_1: power_output[1,1] <= 9109.681255668149
 capacity_1_2: power_output[1,2] <= 9109.681255668149
 capacity_1_3: power_output[1,3] <= 9109.681255668149
 capacity_1_4: power_output[1,4] <= 9109.681255668149
 capacity_1_5: power_output[1,5] <= 9109.681255668149
 capacity_1_6: power_output[1,6] <= 9109.681255668149
 capacity_1_7: power_output[1,7] <= 9109.681255668149
 capacity_1_8: power_output[1,8] <= 9109.681255668149
 capacity_1_9: power_output[1,9] <= 9109.681255668149
 capacity_1_10: power_output[1,10] <= 9109.681255668149
 capacity_1_11: power_output[1,11] <= 9109.681255668149
 capacity_1_12: power_output[1,12] <= 9109.681255668149
 capacity_1_13: power_output[1,13] <= 9109.681255668149
 capacity_2_0: power_output[2,0] <= 9711.095649586214
 capacity_2_1: power_output[2,1] <= 9711.095649586214
 capacity_2_2: power_output[2,2] <= 9711.095649586214
 capacity_2_3: power_output[2,3] <= 9711.095649586214
 capacity_2_4: power_output[2,4] <= 9711.095649586214
 capacity_2_5: power_output[2,5] <= 9711.095649586214
 capacity_2_6: power_output[2,6] <= 9711.095649586214
 capacity_2_7: power_output[2,7] <= 9711.095649586214
 capacity_2_8: power_output[2,8] <= 9711.095649586214
 capacity_2_9: power_output[2,9] <= 9711.095649586214
 capacity_2_10: power_output[2,10] <= 9711.095649586214
 capacity_2_11: power_output[2,11] <= 9711.095649586214
 capacity_2_12: power_output[2,12] <= 9711.095649586214
 capacity_2_13: power_output[2,13] <= 9711.095649586214
 capacity_3_0: power_output[3,0] <= 1007.589826957785
 capacity_3_1: power_output[3,1] <= 1007.589826957785
 capacity_3_2: power_output[3,2] <= 1007.589826957785
 capacity_3_3: power_output[3,3] <= 1007.589826957785
 capacity_3_4: power_output[3,4] <= 1007.589826957785
 capacity_3_5: power_output[3,5] <= 1007.589826957785
 capacity_3_6: power_output[3,6] <= 1007.589826957785
 capacity_3_7: power_output[3,7] <= 1007.589826957785
 capacity_3_8: power_output[3,8] <= 1007.589826957785
 capacity_3_9: power_output[3,9] <= 1007.589826957785
 capacity_3_10: power_output[3,10] <= 1007.589826957785
 capacity_3_11: power_output[3,11] <= 1007.589826957785
 capacity_3_12: power_output[3,12] <= 1007.589826957785
 capacity_3_13: power_output[3,13] <= 1007.589826957785
 capacity_4_0: power_output[4,0] <= 6040.27232070848
 capacity_4_1: power_output[4,1] <= 6040.27232070848
 capacity_4_2: power_output[4,2] <= 6040.27232070848
 capacity_4_3: power_output[4,3] <= 6040.27232070848
 capacity_4_4: power_output[4,4] <= 6040.27232070848
 capacity_4_5: power_output[4,5] <= 6040.27232070848
 capacity_4_6: power_output[4,6] <= 6040.27232070848
 capacity_4_7: power_output[4,7] <= 6040.27232070848
 capacity_4_8: power_output[4,8] <= 6040.27232070848
 capacity_4_9: power_output[4,9] <= 6040.27232070848
 capacity_4_10: power_output[4,10] <= 6040.27232070848
 capacity_4_11: power_output[4,11] <= 6040.27232070848
 capacity_4_12: power_output[4,12] <= 6040.27232070848
 capacity_4_13: power_output[4,13] <= 6040.27232070848
 capacity_5_0: power_output[5,0] <= 9808.266067752254
 capacity_5_1: power_output[5,1] <= 9808.266067752254
 capacity_5_2: power_output[5,2] <= 9808.266067752254
 capacity_5_3: power_output[5,3] <= 9808.266067752254
 capacity_5_4: power_output[5,4] <= 9808.266067752254
 capacity_5_5: power_output[5,5] <= 9808.266067752254
 capacity_5_6: power_output[5,6] <= 9808.266067752254
 capacity_5_7: power_output[5,7] <= 9808.266067752254
 capacity_5_8: power_output[5,8] <= 9808.266067752254
 capacity_5_9: power_output[5,9] <= 9808.266067752254
 capacity_5_10: power_output[5,10] <= 9808.266067752254
 capacity_5_11: power_output[5,11] <= 9808.266067752254
 capacity_5_12: power_output[5,12] <= 9808.266067752254
 capacity_5_13: power_output[5,13] <= 9808.266067752254
 emission_cap: 0.6835503184358553 power_output[0,0]
   + 0.6835503184358553 power_output[0,1]
   + 0.6835503184358553 power_output[0,2]
   + 0.6835503184358553 power_output[0,3]
   + 0.6835503184358553 power_output[0,4]
   + 0.6835503184358553 power_output[0,5]
   + 0.6835503184358553 power_output[0,6]
   + 0.6835503184358553 power_output[0,7]
   + 0.6835503184358553 power_output[0,8]
   + 0.6835503184358553 power_output[0,9]
   + 0.6835503184358553 power_output[0,10]
   + 0.6835503184358553 power_output[0,11]
   + 0.6835503184358553 power_output[0,12]
   + 0.6835503184358553 power_output[0,13]
   + 0.2295962978095211 power_output[1,0]
   + 0.2295962978095211 power_output[1,1]
   + 0.2295962978095211 power_output[1,2]
   + 0.2295962978095211 power_output[1,3]
   + 0.2295962978095211 power_output[1,4]
   + 0.2295962978095211 power_output[1,5]
   + 0.2295962978095211 power_output[1,6]
   + 0.2295962978095211 power_output[1,7]
   + 0.2295962978095211 power_output[1,8]
   + 0.2295962978095211 power_output[1,9]
   + 0.2295962978095211 power_output[1,10]
   + 0.2295962978095211 power_output[1,11]
   + 0.2295962978095211 power_output[1,12]
   + 0.2295962978095211 power_output[1,13]
   + 0.8963998141738116 power_output[2,0]
   + 0.8963998141738116 power_output[2,1]
   + 0.8963998141738116 power_output[2,2]
   + 0.8963998141738116 power_output[2,3]
   + 0.8963998141738116 power_output[2,4]
   + 0.8963998141738116 power_output[2,5]
   + 0.8963998141738116 power_output[2,6]
   + 0.8963998141738116 power_output[2,7]
   + 0.8963998141738116 power_output[2,8]
   + 0.8963998141738116 power_output[2,9]
   + 0.8963998141738116 power_output[2,10]
   + 0.8963998141738116 power_output[2,11]
   + 0.8963998141738116 power_output[2,12]
   + 0.8963998141738116 power_output[2,13]
   + 0.6921340271313607 power_output[3,0]
   + 0.6921340271313607 power_output[3,1]
   + 0.6921340271313607 power_output[3,2]
   + 0.6921340271313607 power_output[3,3]
   + 0.6921340271313607 power_output[3,4]
   + 0.6921340271313607 power_output[3,5]
   + 0.6921340271313607 power_output[3,6]
   + 0.6921340271313607 power_output[3,7]
   + 0.6921340271313607 power_output[3,8]
   + 0.6921340271313607 power_output[3,9]
   + 0.6921340271313607 power_output[3,10]
   + 0.6921340271313607 power_output[3,11]
   + 0.6921340271313607 power_output[3,12]
   + 0.6921340271313607 power_output[3,13]
   + 0.9774775243639638 power_output[4,0]
   + 0.9774775243639638 power_output[4,1]
   + 0.9774775243639638 power_output[4,2]
   + 0.9774775243639638 power_output[4,3]
   + 0.9774775243639638 power_output[4,4]
   + 0.9774775243639638 power_output[4,5]
   + 0.9774775243639638 power_output[4,6]
   + 0.9774775243639638 power_output[4,7]
   + 0.9774775243639638 power_output[4,8]
   + 0.9774775243639638 power_output[4,9]
   + 0.9774775243639638 power_output[4,10]
   + 0.9774775243639638 power_output[4,11]
   + 0.9774775243639638 power_output[4,12]
   + 0.9774775243639638 power_output[4,13]
   + 0.2221981709912203 power_output[5,0]
   + 0.2221981709912203 power_output[5,1]
   + 0.2221981709912203 power_output[5,2]
   + 0.2221981709912203 power_output[5,3]
   + 0.2221981709912203 power_output[5,4]
   + 0.2221981709912203 power_output[5,5]
   + 0.2221981709912203 power_output[5,6]
   + 0.2221981709912203 power_output[5,7]
   + 0.2221981709912203 power_output[5,8]
   + 0.2221981709912203 power_output[5,9]
   + 0.2221981709912203 power_output[5,10]
   + 0.2221981709912203 power_output[5,11]
   + 0.2221981709912203 power_output[5,12]
   + 0.2221981709912203 power_output[5,13] <= 15488.8107306489
 ramp_up_0_1: - power_output[0,0] + power_output[0,1] <= 422.3766458507044
 ramp_down_0_1: power_output[0,0] - power_output[0,1] <= 281.7577436846939
 ramp_up_0_2: - power_output[0,1] + power_output[0,2] <= 422.3766458507044
 ramp_down_0_2: power_output[0,1] - power_output[0,2] <= 281.7577436846939
 ramp_up_0_3: - power_output[0,2] + power_output[0,3] <= 422.3766458507044
 ramp_down_0_3: power_output[0,2] - power_output[0,3] <= 281.7577436846939
 ramp_up_0_4: - power_output[0,3] + power_output[0,4] <= 422.3766458507044
 ramp_down_0_4: power_output[0,3] - power_output[0,4] <= 281.7577436846939
 ramp_up_0_5: - power_output[0,4] + power_output[0,5] <= 422.3766458507044
 ramp_down_0_5: power_output[0,4] - power_output[0,5] <= 281.7577436846939
 ramp_up_0_6: - power_output[0,5] + power_output[0,6] <= 422.3766458507044
 ramp_down_0_6: power_output[0,5] - power_output[0,6] <= 281.7577436846939
 ramp_up_0_7: - power_output[0,6] + power_output[0,7] <= 422.3766458507044
 ramp_down_0_7: power_output[0,6] - power_output[0,7] <= 281.7577436846939
 ramp_up_0_8: - power_output[0,7] + power_output[0,8] <= 422.3766458507044
 ramp_down_0_8: power_output[0,7] - power_output[0,8] <= 281.7577436846939
 ramp_up_0_9: - power_output[0,8] + power_output[0,9] <= 422.3766458507044
 ramp_down_0_9: power_output[0,8] - power_output[0,9] <= 281.7577436846939
 ramp_up_0_10: - power_output[0,9] + power_output[0,10]
   <= 422.3766458507044
 ramp_down_0_10: power_output[0,9] - power_output[0,10]
   <= 281.7577436846939
 ramp_up_0_11: - power_output[0,10] + power_output[0,11]
   <= 422.3766458507044
 ramp_down_0_11: power_output[0,10] - power_output[0,11]
   <= 281.7577436846939
 ramp_up_0_12: - power_output[0,11] + power_output[0,12]
   <= 422.3766458507044
 ramp_down_0_12: power_output[0,11] - power_output[0,12]
   <= 281.7577436846939
 ramp_up_0_13: - power_output[0,12] + power_output[0,13]
   <= 422.3766458507044
 ramp_down_0_13: power_output[0,12] - power_output[0,13]
   <= 281.7577436846939
 initial_ramp_up_0: power_output[0,0] <= 429.8066373488357
 initial_ramp_down_0: - power_output[0,0] <= 274.3277521865626
 ramp_up_1_1: - power_output[1,0] + power_output[1,1] <= 444.0611048222441
 ramp_down_1_1: power_output[1,0] - power_output[1,1] <= 269.649873829037
 ramp_up_1_2: - power_output[1,1] + power_output[1,2] <= 444.0611048222441
 ramp_down_1_2: power_output[1,1] - power_output[1,2] <= 269.649873829037
 ramp_up_1_3: - power_output[1,2] + power_output[1,3] <= 444.0611048222441
 ramp_down_1_3: power_output[1,2] - power_output[1,3] <= 269.649873829037
 ramp_up_1_4: - power_output[1,3] + power_output[1,4] <= 444.0611048222441
 ramp_down_1_4: power_output[1,3] - power_output[1,4] <= 269.649873829037
 ramp_up_1_5: - power_output[1,4] + power_output[1,5] <= 444.0611048222441
 ramp_down_1_5: power_output[1,4] - power_output[1,5] <= 269.649873829037
 ramp_up_1_6: - power_output[1,5] + power_output[1,6] <= 444.0611048222441
 ramp_down_1_6: power_output[1,5] - power_output[1,6] <= 269.649873829037
 ramp_up_1_7: - power_output[1,6] + power_output[1,7] <= 444.0611048222441
 ramp_down_1_7: power_output[1,6] - power_output[1,7] <= 269.649873829037
 ramp_up_1_8: - power_output[1,7] + power_output[1,8] <= 444.0611048222441
 ramp_down_1_8: power_output[1,7] - power_output[1,8] <= 269.649873829037
 ramp_up_1_9: - power_output[1,8] + power_output[1,9] <= 444.0611048222441
 ramp_down_1_9: power_output[1,8] - power_output[1,9] <= 269.649873829037
 ramp_up_1_10: - power_output[1,9] + power_output[1,10]
   <= 444.0611048222441
 ramp_down_1_10: power_output[1,9] - power_output[1,10]
   <= 269.649873829037
 ramp_up_1_11: - power_output[1,10] + power_output[1,11]
   <= 444.0611048222441
 ramp_down_1_11: power_output[1,10] - power_output[1,11]
   <= 269.649873829037
 ramp_up_1_12: - power_output[1,11] + power_output[1,12]
   <= 444.0611048222441
 ramp_down_1_12: power_output[1,11] - power_output[1,12]
   <= 269.649873829037
 ramp_up_1_13: - power_output[1,12] + power_output[1,13]
   <= 444.0611048222441
 ramp_down_1_13: power_output[1,12] - power_output[1,13]
   <= 269.649873829037
 initial_ramp_up_1: power_output[1,0] <= 942.2180684597442
 initial_ramp_down_1: - power_output[1,0] <= -228.507089808463
 ramp_up_2_1: - power_output[2,0] + power_output[2,1] <= 457.432987656976
 ramp_down_2_1: power_output[2,0] - power_output[2,1] <= 428.028569966691
 ramp_up_2_2: - power_output[2,1] + power_output[2,2] <= 457.432987656976
 ramp_down_2_2: power_output[2,1] - power_output[2,2] <= 428.028569966691
 ramp_up_2_3: - power_output[2,2] + power_output[2,3] <= 457.432987656976
 ramp_down_2_3: power_output[2,2] - power_output[2,3] <= 428.028569966691
 ramp_up_2_4: - power_output[2,3] + power_output[2,4] <= 457.432987656976
 ramp_down_2_4: power_output[2,3] - power_output[2,4] <= 428.028569966691
 ramp_up_2_5: - power_output[2,4] + power_output[2,5] <= 457.432987656976
 ramp_down_2_5: power_output[2,4] - power_output[2,5] <= 428.028569966691
 ramp_up_2_6: - power_output[2,5] + power_output[2,6] <= 457.432987656976
 ramp_down_2_6: power_output[2,5] - power_output[2,6] <= 428.028569966691
 ramp_up_2_7: - power_output[2,6] + power_output[2,7] <= 457.432987656976
 ramp_down_2_7: power_output[2,6] - power_output[2,7] <= 428.028569966691
 ramp_up_2_8: - power_output[2,7] + power_output[2,8] <= 457.432987656976
 ramp_down_2_8: power_output[2,7] - power_output[2,8] <= 428.028569966691
 ramp_up_2_9: - power_output[2,8] + power_output[2,9] <= 457.432987656976
 ramp_down_2_9: power_output[2,8] - power_output[2,9] <= 428.028569966691
 ramp_up_2_10: - power_output[2,9] + power_output[2,10]
   <= 457.432987656976
 ramp_down_2_10: power_output[2,9] - power_output[2,10]
   <= 428.028569966691
 ramp_up_2_11: - power_output[2,10] + power_output[2,11]
   <= 457.432987656976
 ramp_down_2_11: power_output[2,10] - power_output[2,11]
   <= 428.028569966691
 ramp_up_2_12: - power_output[2,11] + power_output[2,12]
   <= 457.432987656976
 ramp_down_2_12: power_output[2,11] - power_output[2,12]
   <= 428.028569966691
 ramp_up_2_13: - power_output[2,12] + power_output[2,13]
   <= 457.432987656976
 ramp_down_2_13: power_output[2,12] - power_output[2,13]
   <= 428.028569966691
 initial_ramp_up_2: power_output[2,0] <= 1188.346367719183
 initial_ramp_down_2: - power_output[2,0] <= -302.8848100955163
 ramp_up_3_1: - power_output[3,0] + power_output[3,1] <= 378.1562078131124
 ramp_down_3_1: power_output[3,0] - power_output[3,1] <= 253.8858871320105
 ramp_up_3_2: - power_output[3,1] + power_output[3,2] <= 378.1562078131124
 ramp_down_3_2: power_output[3,1] - power_output[3,2] <= 253.8858871320105
 ramp_up_3_3: - power_output[3,2] + power_output[3,3] <= 378.1562078131124
 ramp_down_3_3: power_output[3,2] - power_output[3,3] <= 253.8858871320105
 ramp_up_3_4: - power_output[3,3] + power_output[3,4] <= 378.1562078131124
 ramp_down_3_4: power_output[3,3] - power_output[3,4] <= 253.8858871320105
 ramp_up_3_5: - power_output[3,4] + power_output[3,5] <= 378.1562078131124
 ramp_down_3_5: power_output[3,4] - power_output[3,5] <= 253.8858871320105
 ramp_up_3_6: - power_output[3,5] + power_output[3,6] <= 378.1562078131124
 ramp_down_3_6: power_output[3,5] - power_output[3,6] <= 253.8858871320105
 ramp_up_3_7: - power_output[3,6] + power_output[3,7] <= 378.1562078131124
 ramp_down_3_7: power_output[3,6] - power_output[3,7] <= 253.8858871320105
 ramp_up_3_8: - power_output[3,7] + power_output[3,8] <= 378.1562078131124
 ramp_down_3_8: power_output[3,7] - power_output[3,8] <= 253.8858871320105
 ramp_up_3_9: - power_output[3,8] + power_output[3,9] <= 378.1562078131124
 ramp_down_3_9: power_output[3,8] - power_output[3,9] <= 253.8858871320105
 ramp_up_3_10: - power_output[3,9] + power_output[3,10]
   <= 378.1562078131124
 ramp_down_3_10: power_output[3,9] - power_output[3,10]
   <= 253.8858871320105
 ramp_up_3_11: - power_output[3,10] + power_output[3,11]
   <= 378.1562078131124
 ramp_down_3_11: power_output[3,10] - power_output[3,11]
   <= 253.8858871320105
 ramp_up_3_12: - power_output[3,11] + power_output[3,12]
   <= 378.1562078131124
 ramp_down_3_12: power_output[3,11] - power_output[3,12]
   <= 253.8858871320105
 ramp_up_3_13: - power_output[3,12] + power_output[3,13]
   <= 378.1562078131124
 ramp_down_3_13: power_output[3,12] - power_output[3,13]
   <= 253.8858871320105
 initial_ramp_up_3: power_output[3,0] <= 1090.286206778143
 initial_ramp_down_3: - power_output[3,0] <= -458.2441118330204
 ramp_up_4_1: - power_output[4,0] + power_output[4,1] <= 472.124404983831
 ramp_down_4_1: power_output[4,0] - power_output[4,1] <= 190.2963673644991
 ramp_up_4_2: - power_output[4,1] + power_output[4,2] <= 472.124404983831
 ramp_down_4_2: power_output[4,1] - power_output[4,2] <= 190.2963673644991
 ramp_up_4_3: - power_output[4,2] + power_output[4,3] <= 472.124404983831
 ramp_down_4_3: power_output[4,2] - power_output[4,3] <= 190.2963673644991
 ramp_up_4_4: - power_output[4,3] + power_output[4,4] <= 472.124404983831
 ramp_down_4_4: power_output[4,3] - power_output[4,4] <= 190.2963673644991
 ramp_up_4_5: - power_output[4,4] + power_output[4,5] <= 472.124404983831
 ramp_down_4_5: power_output[4,4] - power_output[4,5] <= 190.2963673644991
 ramp_up_4_6: - power_output[4,5] + power_output[4,6] <= 472.124404983831
 ramp_down_4_6: power_output[4,5] - power_output[4,6] <= 190.2963673644991
 ramp_up_4_7: - power_output[4,6] + power_output[4,7] <= 472.124404983831
 ramp_down_4_7: power_output[4,6] - power_output[4,7] <= 190.2963673644991
 ramp_up_4_8: - power_output[4,7] + power_output[4,8] <= 472.124404983831
 ramp_down_4_8: power_output[4,7] - power_output[4,8] <= 190.2963673644991
 ramp_up_4_9: - power_output[4,8] + power_output[4,9] <= 472.124404983831
 ramp_down_4_9: power_output[4,8] - power_output[4,9] <= 190.2963673644991
 ramp_up_4_10: - power_output[4,9] + power_output[4,10]
   <= 472.124404983831
 ramp_down_4_10: power_output[4,9] - power_output[4,10]
   <= 190.2963673644991
 ramp_up_4_11: - power_output[4,10] + power_output[4,11]
   <= 472.124404983831
 ramp_down_4_11: power_output[4,10] - power_output[4,11]
   <= 190.2963673644991
 ramp_up_4_12: - power_output[4,11] + power_output[4,12]
   <= 472.124404983831
 ramp_down_4_12: power_output[4,11] - power_output[4,12]
   <= 190.2963673644991
 ramp_up_4_13: - power_output[4,12] + power_output[4,13]
   <= 472.124404983831
 ramp_down_4_13: power_output[4,12] - power_output[4,13]
   <= 190.2963673644991
 initial_ramp_up_4: power_output[4,0] <= 741.3748531126937
 initial_ramp_down_4: - power_output[4,0] <= -78.9540807643636
 ramp_up_5_1: - power_output[5,0] + power_output[5,1] <= 349.2993926399868
 ramp_down_5_1: power_output[5,0] - power_output[5,1] <= 269.9087429182924
 ramp_up_5_2: - power_output[5,1] + power_output[5,2] <= 349.2993926399868
 ramp_down_5_2: power_output[5,1] - power_output[5,2] <= 269.9087429182924
 ramp_up_5_3: - power_output[5,2] + power_output[5,3] <= 349.2993926399868
 ramp_down_5_3: power_output[5,2] - power_output[5,3] <= 269.9087429182924
 ramp_up_5_4: - power_output[5,3] + power_output[5,4] <= 349.2993926399868
 ramp_down_5_4: power_output[5,3] - power_output[5,4] <= 269.9087429182924
 ramp_up_5_5: - power_output[5,4] + power_output[5,5] <= 349.2993926399868
 ramp_down_5_5: power_output[5,4] - power_output[5,5] <= 269.9087429182924
 ramp_up_5_6: - power_output[5,5] + power_output[5,6] <= 349.2993926399868
 ramp_down_5_6: power_output[5,5] - power_output[5,6] <= 269.9087429182924
 ramp_up_5_7: - power_output[5,6] + power_output[5,7] <= 349.2993926399868
 ramp_down_5_7: power_output[5,6] - power_output[5,7] <= 269.9087429182924
 ramp_up_5_8: - power_output[5,7] + power_output[5,8] <= 349.2993926399868
 ramp_down_5_8: power_output[5,7] - power_output[5,8] <= 269.9087429182924
 ramp_up_5_9: - power_output[5,8] + power_output[5,9] <= 349.2993926399868
 ramp_down_5_9: power_output[5,8] - power_output[5,9] <= 269.9087429182924
 ramp_up_5_10: - power_output[5,9] + power_output[5,10]
   <= 349.2993926399868
 ramp_down_5_10: power_output[5,9] - power_output[5,10]
   <= 269.9087429182924
 ramp_up_5_11: - power_output[5,10] + power_output[5,11]
   <= 349.2993926399868
 ramp_down_5_11: power_output[5,10] - power_output[5,11]
   <= 269.9087429182924
 ramp_up_5_12: - power_output[5,11] + power_output[5,12]
   <= 349.2993926399868
 ramp_down_5_12: power_output[5,11] - power_output[5,12]
   <= 269.9087429182924
 ramp_up_5_13: - power_output[5,12] + power_output[5,13]
   <= 349.2993926399868
 ramp_down_5_13: power_output[5,12] - power_output[5,13]
   <= 269.9087429182924
 initial_ramp_up_5: power_output[5,0] <= 1275.863844291011
 initial_ramp_down_5: - power_output[5,0] <= -656.6557087327313
Bounds
End
