\ Model Network Flow Problem with Perishable Goods
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  8.723083238626542 flow[0] + 1.775458477706408 flow[1]
   + 6.934320729064493 flow[2] + 3.784955906840536 flow[3]
   + 7.175364866877161 flow[4] + 2.658007125573129 flow[5]
   + 4.983289981745116 flow[6] + 5.634095399692089 flow[7]
   + 4.881361798028912 flow[8] + 2.295384196787211 flow[9]
   + 3.83777106426801 flow[10] + 2.190810396579217 flow[11]
   + 1.440728408607095 flow[12] + 7.088479899981314 flow[13]
   + 6.032749340548333 flow[14] + 1.96107817882982 flow[15]
   + 7.632779968330152 flow[16]
Subject To
 cap_0: flow[0] <= 25.28012961929583
 cap_1: flow[1] <= 91.00397443672088
 cap_2: flow[2] <= 97.06816103941146
 cap_3: flow[3] <= 10.92314606686223
 cap_4: flow[4] <= 53.15107644790289
 cap_5: flow[5] <= 72.94969786922614
 cap_6: flow[6] <= 80.99127993857162
 cap_7: flow[7] <= 60.72605813200224
 cap_8: flow[8] <= 38.40963784616488
 cap_9: flow[9] <= 3.050875633161629
 cap_10: flow[10] <= 82.82966257886331
 cap_11: flow[11] <= 76.58112985060509
 cap_12: flow[12] <= 64.80372918049318
 cap_13: flow[13] <= 5.330264211339687
 cap_14: flow[14] <= 25.58595150144134
 cap_15: flow[15] <= 9.275959063306384
 cap_16: flow[16] <= 15.12092497141093
 flow_conservation_0: flow[16] = 2.387026436465426
 flow_conservation_1: - flow[6] - flow[7] - flow[15] = 34.0474921838955
 flow_conservation_2: - flow[2] + flow[4] + flow[7] + flow[10] + flow[11]
   - flow[12] = -39.98791453707672
 flow_conservation_3: flow[0] + flow[5] + flow[9] - flow[13]
   = 7.27790957841758
 flow_conservation_4: flow[8] - flow[11] - flow[14] - flow[16]
   = -15.84998896750085
 flow_conservation_5: flow[3] - flow[8] = -2.282012774364433
 flow_conservation_6: - flow[3] - flow[10] + flow[12] + flow[15]
   = 15.0242235492796
 flow_conservation_7: flow[2] - flow[4] + flow[6] - flow[9]
   = 6.57117558814549
 flow_conservation_8: - flow[0] = -13.16466679822381
 flow_conservation_9: - flow[5] + flow[13] + flow[14] = -13.31964184577494
Bounds
End
