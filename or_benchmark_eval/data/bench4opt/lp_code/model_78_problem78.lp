\ Model Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  30.55526866705014 flow[0] + 88.61247824104655 flow[1]
   + 9.600361696281832 flow[2] + 4.817313820139342 flow[3]
   + 85.55641414945059 flow[4] + 55.61006126304833 flow[5]
   + 49.65816139731093 flow[6] + 24.01184977157876 flow[7]
   + 10.24295842596161 flow[8] + 27.33135558599814 flow[9]
   + 90.99889288526529 flow[10] + 61.92961318892784 flow[11]
   + 36.9390683326678 flow[12] + 4.296726144643425 flow[13]
   + 67.57846416537376 flow[14] + 38.03363005440075 flow[15]
   + 79.77311406536502 flow[16] + 78.16932672458209 flow[17]
   + 50.23194945216619 flow[18] + 25.53953965207644 flow[19]
Subject To
 cap_0: flow[0] <= 612437.0868665522
 cap_1: flow[1] <= 791515.1664170029
 cap_2: flow[2] <= 570081.989580609
 cap_3: flow[3] <= 416412.8751497786
 cap_4: flow[4] <= 866379.8459480993
 cap_5: flow[5] <= 935288.019007945
 cap_6: flow[6] <= 656917.8813360111
 cap_7: flow[7] <= 527289.2285823686
 cap_8: flow[8] <= 501692.8181177936
 cap_9: flow[9] <= 578375.1407405842
 cap_10: flow[10] <= 431202.0445675749
 cap_11: flow[11] <= 87948.4196166195
 cap_12: flow[12] <= 610073.2994995383
 cap_13: flow[13] <= 443474.6083981163
 cap_14: flow[14] <= 825808.5683589765
 cap_15: flow[15] <= 107997.4177994327
 cap_16: flow[16] <= 924160.5129778242
 cap_17: flow[17] <= 936199.3635537494
 cap_18: flow[18] <= 90455.31127029019
 cap_19: flow[19] <= 372798.4343000991
 flow_conservation_0: flow[0] - flow[1] - flow[2] + flow[5]
   = 491.2840758221546
 flow_conservation_1: flow[3] + flow[10] + flow[15] = 303.2036997758833
 flow_conservation_2: - flow[3] - flow[5] = 675.835010492412
 flow_conservation_3: flow[1] + flow[7] - flow[14] - flow[15]
   = 944.6226223654771
 flow_conservation_4: - flow[0] + flow[13] - flow[16] = -842.3725555164543
 flow_conservation_5: flow[2] + flow[4] + flow[19] = -664.7165516856757
 flow_conservation_6: flow[6] + flow[8] - flow[10] + flow[17] + flow[18]
   = -435.0239154130197
 flow_conservation_7: - flow[12] + flow[14] - flow[17]
   = -848.3352143556974
 flow_conservation_8: - flow[4] - flow[6] - flow[8] - flow[9] - flow[11]
   + flow[16] = -793.1031654670535
 flow_conservation_9: - flow[7] + flow[9] + flow[11] + flow[12] - flow[13]
   - flow[18] - flow[19] = 784.0468630793057
Bounds
End
