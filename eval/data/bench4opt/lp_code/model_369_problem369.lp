\ Model Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  53.75675002607009 flow[0] + 22.69778815704039 flow[1]
   + 87.7109820560474 flow[2] + 86.66916478055091 flow[3]
   + 52.50744557052623 flow[4] + 57.92297233766054 flow[5]
   + 78.94160608772842 flow[6] + 91.92945139666479 flow[7]
   + 11.93320783734555 flow[8] + 8.156571602897861 flow[9]
   + 31.65400734127256 flow[10] + 61.0383259453092 flow[11]
   + 23.09574927008243 flow[12] + 11.69425222633907 flow[13]
   + 13.21412114283677 flow[14] + 33.49276951522223 flow[15]
   + 65.97286643665875 flow[16] + 59.2317044441134 flow[17]
   + 48.51339854016491 flow[18] + 69.60403252383634 flow[19]
Subject To
 capacity_0: flow[0] <= 685934.5630115375
 capacity_1: flow[1] <= 421168.5041176524
 capacity_2: flow[2] <= 599256.7654869115
 capacity_3: flow[3] <= 505635.9015095776
 capacity_4: flow[4] <= 137444.3563600016
 capacity_5: flow[5] <= 231468.2301125651
 capacity_6: flow[6] <= 799089.587879856
 capacity_7: flow[7] <= 260226.5461019165
 capacity_8: flow[8] <= 562737.6374417323
 capacity_9: flow[9] <= 316429.3884745147
 capacity_10: flow[10] <= 657623.8731550232
 capacity_11: flow[11] <= 856552.623651003
 capacity_12: flow[12] <= 142015.9717057545
 capacity_13: flow[13] <= 1341.715504887865
 capacity_14: flow[14] <= 324915.1727624613
 capacity_15: flow[15] <= 372181.1755486118
 capacity_16: flow[16] <= 605153.5375692684
 capacity_17: flow[17] <= 475042.3066173914
 capacity_18: flow[18] <= 613170.3118436623
 capacity_19: flow[19] <= 970652.0812832478
 flow_conservation_0: - flow[4] - flow[7] + flow[13] - flow[14]
   = 171.1486370772598
 flow_conservation_1: flow[1] - flow[2] + flow[3] - flow[6] + flow[7]
   + flow[15] = -171.1782902797498
 flow_conservation_2: flow[8] - flow[17] = 898.8261551919475
 flow_conservation_3: - flow[5] + flow[11] + flow[12] - flow[18] - flow[19]
   = -577.2135495116637
 flow_conservation_4: - flow[0] - flow[8] - flow[10] - flow[16]
   = -353.2385953003796
 flow_conservation_5: - flow[1] - flow[3] - flow[9] - flow[11] - flow[15]
   + flow[19] = 216.7357399872481
 flow_conservation_6: flow[5] + flow[14] + flow[17] = -372.9984144808376
 flow_conservation_7: flow[0] + flow[2] + flow[4] + flow[6] + flow[9]
   - flow[13] + flow[16] = 608.1778907899886
 flow_conservation_8: flow[10] - flow[12] + flow[18] = -84.25228018411497
 flow_conservation_9: <= -1
Bounds
End
