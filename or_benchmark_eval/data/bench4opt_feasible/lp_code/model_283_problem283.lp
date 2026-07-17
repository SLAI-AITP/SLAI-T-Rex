\ Model Network Flow Problem with Time-Dependent Costs and Capacities
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2.236066599369885 flow[0,0] + 4.627584815286707 flow[0,1]
   + 3.950711173276606 flow[0,2] + 7.21608988504091 flow[0,3]
   + 10.81865533710675 flow[0,4] + 5.357795417937272 flow[1,0]
   + 7.145443213399981 flow[1,1] + 6.22430886790673 flow[1,2]
   + 8.992305624317886 flow[1,3] + 4.683002567648197 flow[1,4]
   + 9.819930984395906 flow[2,0] + 7.998728526487265 flow[2,1]
   + 10.93881064980186 flow[2,2] + 10.83771288747875 flow[2,3]
   + 10.57582807319175 flow[2,4] + 9.709301723053507 flow[3,0]
   + 7.70139879370263 flow[3,1] + 4.559365085976461 flow[3,2]
   + 9.605262548234334 flow[3,3] + 4.71116052186065 flow[3,4]
   + 7.992567007673602 flow[4,0] + 6.84327294957679 flow[4,1]
   + 5.773803285912884 flow[4,2] + 5.231998822137334 flow[4,3]
   + 7.904341989086782 flow[4,4] + 3.681812789433419 flow[5,0]
   + 6.441713429849533 flow[5,1] + 4.697754528899371 flow[5,2]
   + 8.571684662337963 flow[5,3] + 9.545251370572725 flow[5,4]
   + 6.363784111197496 flow[6,0] + 2.882010589285099 flow[6,1]
   + 8.575331789453699 flow[6,2] + 9.181255272715713 flow[6,3]
   + 7.254278432560533 flow[6,4] + 9.255424847460466 flow[7,0]
   + 4.916981076682784 flow[7,1] + 3.161638838498566 flow[7,2]
   + 7.087820801196874 flow[7,3] + 10.62443423384614 flow[7,4]
   + 3.708786727580475 flow[8,0] + 2.623181833121677 flow[8,1]
   + 8.532057123555106 flow[8,2] + 5.202639079480507 flow[8,3]
   + 8.51849996593733 flow[8,4] + 7.961403190351263 flow[9,0]
   + 3.257076935132474 flow[9,1] + 4.691244111312463 flow[9,2]
   + 9.136118499416657 flow[9,3] + 3.340280660584176 flow[9,4]
   + 6.795302128584552 flow[10,0] + 10.32818787527896 flow[10,1]
   + 4.089418811036815 flow[10,2] + 3.611143360936184 flow[10,3]
   + 9.592385025096483 flow[10,4] + 4.828339888999867 flow[11,0]
   + 6.105406894080138 flow[11,1] + 4.18243522475041 flow[11,2]
   + 7.683468538199098 flow[11,3] + 4.64920022548041 flow[11,4]
   + 4.36531797597294 flow[12,0] + 3.095431644887949 flow[12,1]
   + 4.72066607725235 flow[12,2] + 8.756658490252681 flow[12,3]
   + 3.83603006190376 flow[12,4] + 6.077864745525344 flow[13,0]
   + 7.948092553290538 flow[13,1] + 10.21532541461485 flow[13,2]
   + 8.207565735696736 flow[13,3] + 8.23026360066866 flow[13,4]
   + 7.356032242455206 flow[14,0] + 8.399663535812834 flow[14,1]
   + 10.45404742506149 flow[14,2] + 5.27786243095562 flow[14,3]
   + 3.884584274402454 flow[14,4] + 2.364943581380133 flow[15,0]
   + 3.42666880502486 flow[15,1] + 7.613358418878679 flow[15,2]
   + 6.770268193519909 flow[15,3] + 9.62608692307527 flow[15,4]
   + 7.745950463756865 flow[16,0] + 3.491339784931896 flow[16,1]
   + 4.117604118574366 flow[16,2] + 4.092376563223059 flow[16,3]
   + 6.625511029227508 flow[16,4] + 2.526125042710116 flow[17,0]
   + 10.29767468249136 flow[17,1] + 4.423983105323071 flow[17,2]
   + 8.063420825384041 flow[17,3] + 9.513947580509356 flow[17,4]
   + 3.607316870895723 flow[18,0] + 10.44941703453757 flow[18,1]
   + 5.81742001984815 flow[18,2] + 4.742532315948969 flow[18,3]
   + 2.805007258832272 flow[18,4] + 7.450759926529855 flow[19,0]
   + 6.569313980227483 flow[19,1] + 3.772670104146731 flow[19,2]
   + 6.016993757340846 flow[19,3] + 9.705401018749139 flow[19,4]
Subject To
 capacity_0_0: flow[0,0] <= 1000
 capacity_0_1: flow[0,1] <= 1000
 capacity_0_2: flow[0,2] <= 1000
 capacity_0_3: flow[0,3] <= 1000
 capacity_0_4: flow[0,4] <= 1000
 capacity_1_0: flow[1,0] <= 1000
 capacity_1_1: flow[1,1] <= 1000
 capacity_1_2: flow[1,2] <= 1000
 capacity_1_3: flow[1,3] <= 1000
 capacity_1_4: flow[1,4] <= 1000
 capacity_2_0: flow[2,0] <= 1000
 capacity_2_1: flow[2,1] <= 1000
 capacity_2_2: flow[2,2] <= 1000
 capacity_2_3: flow[2,3] <= 1000
 capacity_2_4: flow[2,4] <= 1000
 capacity_3_0: flow[3,0] <= 1000
 capacity_3_1: flow[3,1] <= 1000
 capacity_3_2: flow[3,2] <= 1000
 capacity_3_3: flow[3,3] <= 1000
 capacity_3_4: flow[3,4] <= 1000
 capacity_4_0: flow[4,0] <= 1000
 capacity_4_1: flow[4,1] <= 1000
 capacity_4_2: flow[4,2] <= 1000
 capacity_4_3: flow[4,3] <= 1000
 capacity_4_4: flow[4,4] <= 1000
 capacity_5_0: flow[5,0] <= 1000
 capacity_5_1: flow[5,1] <= 1000
 capacity_5_2: flow[5,2] <= 1000
 capacity_5_3: flow[5,3] <= 1000
 capacity_5_4: flow[5,4] <= 1000
 capacity_6_0: flow[6,0] <= 1000
 capacity_6_1: flow[6,1] <= 1000
 capacity_6_2: flow[6,2] <= 1000
 capacity_6_3: flow[6,3] <= 1000
 capacity_6_4: flow[6,4] <= 1000
 capacity_7_0: flow[7,0] <= 1000
 capacity_7_1: flow[7,1] <= 1000
 capacity_7_2: flow[7,2] <= 1000
 capacity_7_3: flow[7,3] <= 1000
 capacity_7_4: flow[7,4] <= 1000
 capacity_8_0: flow[8,0] <= 1000
 capacity_8_1: flow[8,1] <= 1000
 capacity_8_2: flow[8,2] <= 1000
 capacity_8_3: flow[8,3] <= 1000
 capacity_8_4: flow[8,4] <= 1000
 capacity_9_0: flow[9,0] <= 1000
 capacity_9_1: flow[9,1] <= 1000
 capacity_9_2: flow[9,2] <= 1000
 capacity_9_3: flow[9,3] <= 1000
 capacity_9_4: flow[9,4] <= 1000
 capacity_10_0: flow[10,0] <= 1000
 capacity_10_1: flow[10,1] <= 1000
 capacity_10_2: flow[10,2] <= 1000
 capacity_10_3: flow[10,3] <= 1000
 capacity_10_4: flow[10,4] <= 1000
 capacity_11_0: flow[11,0] <= 1000
 capacity_11_1: flow[11,1] <= 1000
 capacity_11_2: flow[11,2] <= 1000
 capacity_11_3: flow[11,3] <= 1000
 capacity_11_4: flow[11,4] <= 1000
 capacity_12_0: flow[12,0] <= 1000
 capacity_12_1: flow[12,1] <= 1000
 capacity_12_2: flow[12,2] <= 1000
 capacity_12_3: flow[12,3] <= 1000
 capacity_12_4: flow[12,4] <= 1000
 capacity_13_0: flow[13,0] <= 1000
 capacity_13_1: flow[13,1] <= 1000
 capacity_13_2: flow[13,2] <= 1000
 capacity_13_3: flow[13,3] <= 1000
 capacity_13_4: flow[13,4] <= 1000
 capacity_14_0: flow[14,0] <= 1000
 capacity_14_1: flow[14,1] <= 1000
 capacity_14_2: flow[14,2] <= 1000
 capacity_14_3: flow[14,3] <= 1000
 capacity_14_4: flow[14,4] <= 1000
 capacity_15_0: flow[15,0] <= 1000
 capacity_15_1: flow[15,1] <= 1000
 capacity_15_2: flow[15,2] <= 1000
 capacity_15_3: flow[15,3] <= 1000
 capacity_15_4: flow[15,4] <= 1000
 capacity_16_0: flow[16,0] <= 1000
 capacity_16_1: flow[16,1] <= 1000
 capacity_16_2: flow[16,2] <= 1000
 capacity_16_3: flow[16,3] <= 1000
 capacity_16_4: flow[16,4] <= 1000
 capacity_17_0: flow[17,0] <= 1000
 capacity_17_1: flow[17,1] <= 1000
 capacity_17_2: flow[17,2] <= 1000
 capacity_17_3: flow[17,3] <= 1000
 capacity_17_4: flow[17,4] <= 1000
 capacity_18_0: flow[18,0] <= 1000
 capacity_18_1: flow[18,1] <= 1000
 capacity_18_2: flow[18,2] <= 1000
 capacity_18_3: flow[18,3] <= 1000
 capacity_18_4: flow[18,4] <= 1000
 capacity_19_0: flow[19,0] <= 1000
 capacity_19_1: flow[19,1] <= 1000
 capacity_19_2: flow[19,2] <= 1000
 capacity_19_3: flow[19,3] <= 1000
 capacity_19_4: flow[19,4] <= 1000
 flow_conservation_0_0: flow[15,0] + flow[18,0] = 0
 flow_conservation_0_1: flow[15,1] + flow[18,1] = 0
 flow_conservation_0_2: flow[15,2] + flow[18,2] = 0
 flow_conservation_0_3: flow[15,3] + flow[18,3] = 0
 flow_conservation_0_4: flow[15,4] + flow[18,4] = 0
 flow_conservation_1_0: flow[6,0] - flow[9,0] - flow[12,0] - flow[17,0]
   = 0
 flow_conservation_1_1: flow[6,1] - flow[9,1] - flow[12,1] - flow[17,1]
   = 0
 flow_conservation_1_2: flow[6,2] - flow[9,2] - flow[12,2] - flow[17,2]
   = 0
 flow_conservation_1_3: flow[6,3] - flow[9,3] - flow[12,3] - flow[17,3]
   = 0
 flow_conservation_1_4: flow[6,4] - flow[9,4] - flow[12,4] - flow[17,4]
   = 0
 flow_conservation_2_0: flow[12,0] - flow[14,0] - flow[16,0] + flow[17,0]
   = 0
 flow_conservation_2_1: flow[12,1] - flow[14,1] - flow[16,1] + flow[17,1]
   = 0
 flow_conservation_2_2: flow[12,2] - flow[14,2] - flow[16,2] + flow[17,2]
   = 0
 flow_conservation_2_3: flow[12,3] - flow[14,3] - flow[16,3] + flow[17,3]
   = 0
 flow_conservation_2_4: flow[12,4] - flow[14,4] - flow[16,4] + flow[17,4]
   = 0
 flow_conservation_3_0: flow[0,0] - flow[7,0] + flow[8,0] - flow[11,0]
   - flow[18,0] = 10
 flow_conservation_3_1: flow[0,1] - flow[7,1] + flow[8,1] - flow[11,1]
   - flow[18,1] = 10
 flow_conservation_3_2: flow[0,2] - flow[7,2] + flow[8,2] - flow[11,2]
   - flow[18,2] = 10
 flow_conservation_3_3: flow[0,3] - flow[7,3] + flow[8,3] - flow[11,3]
   - flow[18,3] = 10
 flow_conservation_3_4: flow[0,4] - flow[7,4] + flow[8,4] - flow[11,4]
   - flow[18,4] = 10
 flow_conservation_4_0: flow[2,0] - flow[6,0] - flow[10,0] = 0
 flow_conservation_4_1: flow[2,1] - flow[6,1] - flow[10,1] = 0
 flow_conservation_4_2: flow[2,2] - flow[6,2] - flow[10,2] = 0
 flow_conservation_4_3: flow[2,3] - flow[6,3] - flow[10,3] = 0
 flow_conservation_4_4: flow[2,4] - flow[6,4] - flow[10,4] = 0
 flow_conservation_5_0: - flow[2,0] + flow[4,0] + flow[11,0] + flow[19,0]
   = 0
 flow_conservation_5_1: - flow[2,1] + flow[4,1] + flow[11,1] + flow[19,1]
   = 0
 flow_conservation_5_2: - flow[2,2] + flow[4,2] + flow[11,2] + flow[19,2]
   = 0
 flow_conservation_5_3: - flow[2,3] + flow[4,3] + flow[11,3] + flow[19,3]
   = 0
 flow_conservation_5_4: - flow[2,4] + flow[4,4] + flow[11,4] + flow[19,4]
   = 0
 flow_conservation_6_0: flow[7,0] = 0
 flow_conservation_6_1: flow[7,1] = 0
 flow_conservation_6_2: flow[7,2] = 0
 flow_conservation_6_3: flow[7,3] = 0
 flow_conservation_6_4: flow[7,4] = 0
 flow_conservation_7_0: - flow[1,0] - flow[4,0] - flow[5,0] + flow[14,0]
   - flow[15,0] - flow[19,0] = 0
 flow_conservation_7_1: - flow[1,1] - flow[4,1] - flow[5,1] + flow[14,1]
   - flow[15,1] - flow[19,1] = 0
 flow_conservation_7_2: - flow[1,2] - flow[4,2] - flow[5,2] + flow[14,2]
   - flow[15,2] - flow[19,2] = 0
 flow_conservation_7_3: - flow[1,3] - flow[4,3] - flow[5,3] + flow[14,3]
   - flow[15,3] - flow[19,3] = 0
 flow_conservation_7_4: - flow[1,4] - flow[4,4] - flow[5,4] + flow[14,4]
   - flow[15,4] - flow[19,4] = 0
 flow_conservation_8_0: - flow[0,0] + flow[3,0] + flow[9,0] + flow[16,0]
   = -10
 flow_conservation_8_1: - flow[0,1] + flow[3,1] + flow[9,1] + flow[16,1]
   = -10
 flow_conservation_8_2: - flow[0,2] + flow[3,2] + flow[9,2] + flow[16,2]
   = -10
 flow_conservation_8_3: - flow[0,3] + flow[3,3] + flow[9,3] + flow[16,3]
   = -10
 flow_conservation_8_4: - flow[0,4] + flow[3,4] + flow[9,4] + flow[16,4]
   = -10
 flow_conservation_9_0: flow[1,0] - flow[3,0] + flow[5,0] - flow[8,0]
   + flow[10,0] = 0
 flow_conservation_9_1: flow[1,1] - flow[3,1] + flow[5,1] - flow[8,1]
   + flow[10,1] = 0
 flow_conservation_9_2: flow[1,2] - flow[3,2] + flow[5,2] - flow[8,2]
   + flow[10,2] = 0
 flow_conservation_9_3: flow[1,3] - flow[3,3] + flow[5,3] - flow[8,3]
   + flow[10,3] = 0
 flow_conservation_9_4: flow[1,4] - flow[3,4] + flow[5,4] - flow[8,4]
   + flow[10,4] = 0
Bounds
End
