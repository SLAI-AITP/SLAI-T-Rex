\ Model Network Flow Problem with Evaporation Losses
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  53.11536797183472 flow[0] + 23.6298636963737 flow[1]
   + 7.228720104033537 flow[2] + 56.12425647010208 flow[3]
   + 37.95369974284428 flow[4] + 32.53481941735498 flow[5]
   + 13.87131963168337 flow[6] + 70.09130124265727 flow[7]
   + 72.80696548977306 flow[8] + 96.85224423964796 flow[9]
   + 15.450982549962 flow[10] + 2.25180383641329 flow[11]
   + 53.11536797183472 flow[12] + 23.6298636963737 flow[13]
   + 7.228720104033537 flow[14] + 56.12425647010208 flow[15]
   + 37.95369974284428 flow[16] + 32.53481941735498 flow[17]
   + 13.87131963168337 flow[18] + 70.09130124265727 flow[19]
Subject To
 cap_0: flow[0] <= 1000
 cap_1: flow[1] <= 1000
 cap_2: flow[2] <= 1000
 cap_3: flow[3] <= 1000
 cap_4: flow[4] <= 1000
 cap_5: flow[5] <= 1000
 cap_6: flow[6] <= 1000
 cap_7: flow[7] <= 1000
 cap_8: flow[8] <= 1000
 cap_9: flow[9] <= 1000
 cap_10: flow[10] <= 1000
 cap_11: flow[11] <= 1000
 cap_12: flow[12] <= 1000
 cap_13: flow[13] <= 1000
 cap_14: flow[14] <= 1000
 cap_15: flow[15] <= 1000
 cap_16: flow[16] <= 1000
 cap_17: flow[17] <= 1000
 cap_18: flow[18] <= 1000
 cap_19: flow[19] <= 1000
 flow_conservation_0: - flow[4] + flow[5] + flow[7] - flow[8] - flow[9]
   - flow[16] + flow[17] + flow[19] = 0
 flow_conservation_1: <= 0
 flow_conservation_2: flow[0] - flow[1] - flow[11] + flow[12] - flow[13]
   = 10
 flow_conservation_3: - flow[6] + flow[8] + flow[9] - flow[18] = 0
 flow_conservation_4: flow[1] + flow[3] + flow[6] + flow[13] + flow[15]
   + flow[18] = 0
 flow_conservation_5: - flow[2] + flow[11] - flow[14] = 0
 flow_conservation_6: flow[4] - flow[7] + flow[16] - flow[19] = 0
 flow_conservation_7: - flow[5] + flow[10] - flow[17] = 0
 flow_conservation_8: - flow[0] + flow[2] - flow[3] - flow[10] - flow[12]
   + flow[14] - flow[15] = -10
 flow_conservation_9: <= 0
Bounds
End
