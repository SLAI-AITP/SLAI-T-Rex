\ Model Network Flow Problem with Time-Dependent Flow
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  6.703236515803272 flow[0,0] + 6.703236515803272 flow[0,1]
   + 6.703236515803272 flow[0,2] + 6.703236515803272 flow[0,3]
   + 6.703236515803272 flow[0,4] + 37.03952212109096 flow[1,0]
   + 37.03952212109096 flow[1,1] + 37.03952212109096 flow[1,2]
   + 37.03952212109096 flow[1,3] + 37.03952212109096 flow[1,4]
   + 83.93884228148443 flow[2,0] + 83.93884228148443 flow[2,1]
   + 83.93884228148443 flow[2,2] + 83.93884228148443 flow[2,3]
   + 83.93884228148443 flow[2,4] + 32.48288151816824 flow[3,0]
   + 32.48288151816824 flow[3,1] + 32.48288151816824 flow[3,2]
   + 32.48288151816824 flow[3,3] + 32.48288151816824 flow[3,4]
   + 64.89606614097747 flow[4,0] + 64.89606614097747 flow[4,1]
   + 64.89606614097747 flow[4,2] + 64.89606614097747 flow[4,3]
   + 64.89606614097747 flow[4,4] + 98.77859954496613 flow[5,0]
   + 98.77859954496613 flow[5,1] + 98.77859954496613 flow[5,2]
   + 98.77859954496613 flow[5,3] + 98.77859954496613 flow[5,4]
   + 82.3336262726459 flow[6,0] + 82.3336262726459 flow[6,1]
   + 82.3336262726459 flow[6,2] + 82.3336262726459 flow[6,3]
   + 82.3336262726459 flow[6,4] + 64.77678229044935 flow[7,0]
   + 64.77678229044935 flow[7,1] + 64.77678229044935 flow[7,2]
   + 64.77678229044935 flow[7,3] + 64.77678229044935 flow[7,4]
   + 56.38757292361248 flow[8,0] + 56.38757292361248 flow[8,1]
   + 56.38757292361248 flow[8,2] + 56.38757292361248 flow[8,3]
   + 56.38757292361248 flow[8,4] + 30.38812232611714 flow[9,0]
   + 30.38812232611714 flow[9,1] + 30.38812232611714 flow[9,2]
   + 30.38812232611714 flow[9,3] + 30.38812232611714 flow[9,4]
   + 34.48553925132111 flow[10,0] + 34.48553925132111 flow[10,1]
   + 34.48553925132111 flow[10,2] + 34.48553925132111 flow[10,3]
   + 34.48553925132111 flow[10,4] + 21.02509899448442 flow[11,0]
   + 21.02509899448442 flow[11,1] + 21.02509899448442 flow[11,2]
   + 21.02509899448442 flow[11,3] + 21.02509899448442 flow[11,4]
   + 53.45279304100445 flow[12,0] + 53.45279304100445 flow[12,1]
   + 53.45279304100445 flow[12,2] + 53.45279304100445 flow[12,3]
   + 53.45279304100445 flow[12,4] + 73.22035938050045 flow[13,0]
   + 73.22035938050045 flow[13,1] + 73.22035938050045 flow[13,2]
   + 73.22035938050045 flow[13,3] + 73.22035938050045 flow[13,4]
   + 85.47773187272539 flow[14,0] + 85.47773187272539 flow[14,1]
   + 85.47773187272539 flow[14,2] + 85.47773187272539 flow[14,3]
   + 85.47773187272539 flow[14,4] + 61.70459942252604 flow[15,0]
   + 61.70459942252604 flow[15,1] + 61.70459942252604 flow[15,2]
   + 61.70459942252604 flow[15,3] + 61.70459942252604 flow[15,4]
   + 23.69958087482802 flow[16,0] + 23.69958087482802 flow[16,1]
   + 23.69958087482802 flow[16,2] + 23.69958087482802 flow[16,3]
   + 23.69958087482802 flow[16,4] + 85.23087753302308 flow[17,0]
   + 85.23087753302308 flow[17,1] + 85.23087753302308 flow[17,2]
   + 85.23087753302308 flow[17,3] + 85.23087753302308 flow[17,4]
   + 47.14426475142773 flow[18,0] + 47.14426475142773 flow[18,1]
   + 47.14426475142773 flow[18,2] + 47.14426475142773 flow[18,3]
   + 47.14426475142773 flow[18,4] + 26.2531759407153 flow[19,0]
   + 26.2531759407153 flow[19,1] + 26.2531759407153 flow[19,2]
   + 26.2531759407153 flow[19,3] + 26.2531759407153 flow[19,4]
Subject To
 cap_0_0: flow[0,0] <= 1000
 cap_0_1: flow[0,1] <= 1000
 cap_0_2: flow[0,2] <= 1000
 cap_0_3: flow[0,3] <= 1000
 cap_0_4: flow[0,4] <= 1000
 cap_1_0: flow[1,0] <= 1000
 cap_1_1: flow[1,1] <= 1000
 cap_1_2: flow[1,2] <= 1000
 cap_1_3: flow[1,3] <= 1000
 cap_1_4: flow[1,4] <= 1000
 cap_2_0: flow[2,0] <= 1000
 cap_2_1: flow[2,1] <= 1000
 cap_2_2: flow[2,2] <= 1000
 cap_2_3: flow[2,3] <= 1000
 cap_2_4: flow[2,4] <= 1000
 cap_3_0: flow[3,0] <= 1000
 cap_3_1: flow[3,1] <= 1000
 cap_3_2: flow[3,2] <= 1000
 cap_3_3: flow[3,3] <= 1000
 cap_3_4: flow[3,4] <= 1000
 cap_4_0: flow[4,0] <= 1000
 cap_4_1: flow[4,1] <= 1000
 cap_4_2: flow[4,2] <= 1000
 cap_4_3: flow[4,3] <= 1000
 cap_4_4: flow[4,4] <= 1000
 cap_5_0: flow[5,0] <= 1000
 cap_5_1: flow[5,1] <= 1000
 cap_5_2: flow[5,2] <= 1000
 cap_5_3: flow[5,3] <= 1000
 cap_5_4: flow[5,4] <= 1000
 cap_6_0: flow[6,0] <= 1000
 cap_6_1: flow[6,1] <= 1000
 cap_6_2: flow[6,2] <= 1000
 cap_6_3: flow[6,3] <= 1000
 cap_6_4: flow[6,4] <= 1000
 cap_7_0: flow[7,0] <= 1000
 cap_7_1: flow[7,1] <= 1000
 cap_7_2: flow[7,2] <= 1000
 cap_7_3: flow[7,3] <= 1000
 cap_7_4: flow[7,4] <= 1000
 cap_8_0: flow[8,0] <= 1000
 cap_8_1: flow[8,1] <= 1000
 cap_8_2: flow[8,2] <= 1000
 cap_8_3: flow[8,3] <= 1000
 cap_8_4: flow[8,4] <= 1000
 cap_9_0: flow[9,0] <= 1000
 cap_9_1: flow[9,1] <= 1000
 cap_9_2: flow[9,2] <= 1000
 cap_9_3: flow[9,3] <= 1000
 cap_9_4: flow[9,4] <= 1000
 cap_10_0: flow[10,0] <= 1000
 cap_10_1: flow[10,1] <= 1000
 cap_10_2: flow[10,2] <= 1000
 cap_10_3: flow[10,3] <= 1000
 cap_10_4: flow[10,4] <= 1000
 cap_11_0: flow[11,0] <= 1000
 cap_11_1: flow[11,1] <= 1000
 cap_11_2: flow[11,2] <= 1000
 cap_11_3: flow[11,3] <= 1000
 cap_11_4: flow[11,4] <= 1000
 cap_12_0: flow[12,0] <= 1000
 cap_12_1: flow[12,1] <= 1000
 cap_12_2: flow[12,2] <= 1000
 cap_12_3: flow[12,3] <= 1000
 cap_12_4: flow[12,4] <= 1000
 cap_13_0: flow[13,0] <= 1000
 cap_13_1: flow[13,1] <= 1000
 cap_13_2: flow[13,2] <= 1000
 cap_13_3: flow[13,3] <= 1000
 cap_13_4: flow[13,4] <= 1000
 cap_14_0: flow[14,0] <= 1000
 cap_14_1: flow[14,1] <= 1000
 cap_14_2: flow[14,2] <= 1000
 cap_14_3: flow[14,3] <= 1000
 cap_14_4: flow[14,4] <= 1000
 cap_15_0: flow[15,0] <= 1000
 cap_15_1: flow[15,1] <= 1000
 cap_15_2: flow[15,2] <= 1000
 cap_15_3: flow[15,3] <= 1000
 cap_15_4: flow[15,4] <= 1000
 cap_16_0: flow[16,0] <= 1000
 cap_16_1: flow[16,1] <= 1000
 cap_16_2: flow[16,2] <= 1000
 cap_16_3: flow[16,3] <= 1000
 cap_16_4: flow[16,4] <= 1000
 cap_17_0: flow[17,0] <= 1000
 cap_17_1: flow[17,1] <= 1000
 cap_17_2: flow[17,2] <= 1000
 cap_17_3: flow[17,3] <= 1000
 cap_17_4: flow[17,4] <= 1000
 cap_18_0: flow[18,0] <= 1000
 cap_18_1: flow[18,1] <= 1000
 cap_18_2: flow[18,2] <= 1000
 cap_18_3: flow[18,3] <= 1000
 cap_18_4: flow[18,4] <= 1000
 cap_19_0: flow[19,0] <= 1000
 cap_19_1: flow[19,1] <= 1000
 cap_19_2: flow[19,2] <= 1000
 cap_19_3: flow[19,3] <= 1000
 cap_19_4: flow[19,4] <= 1000
 flow_conservation_0_0: flow[5,0] + flow[8,0] - flow[16,0] - flow[17,0]
   = 0
 flow_conservation_0_1: flow[5,1] + flow[8,1] - flow[16,1] - flow[17,1]
   = 0
 flow_conservation_0_2: flow[5,2] + flow[8,2] - flow[16,2] - flow[17,2]
   = 0
 flow_conservation_0_3: flow[5,3] + flow[8,3] - flow[16,3] - flow[17,3]
   = 0
 flow_conservation_0_4: flow[5,4] + flow[8,4] - flow[16,4] - flow[17,4]
   = 0
 flow_conservation_1_0: flow[13,0] + flow[17,0] = 0
 flow_conservation_1_1: flow[13,1] + flow[17,1] = 0
 flow_conservation_1_2: flow[13,2] + flow[17,2] = 0
 flow_conservation_1_3: flow[13,3] + flow[17,3] = 0
 flow_conservation_1_4: flow[13,4] + flow[17,4] = 0
 flow_conservation_2_0: flow[0,0] - flow[1,0] + flow[9,0] - flow[11,0]
   + flow[12,0] + flow[14,0] = 10
 flow_conservation_2_1: flow[0,1] - flow[1,1] + flow[9,1] - flow[11,1]
   + flow[12,1] + flow[14,1] = 10
 flow_conservation_2_2: flow[0,2] - flow[1,2] + flow[9,2] - flow[11,2]
   + flow[12,2] + flow[14,2] = 10
 flow_conservation_2_3: flow[0,3] - flow[1,3] + flow[9,3] - flow[11,3]
   + flow[12,3] + flow[14,3] = 10
 flow_conservation_2_4: flow[0,4] - flow[1,4] + flow[9,4] - flow[11,4]
   + flow[12,4] + flow[14,4] = 10
 flow_conservation_3_0: flow[1,0] - flow[19,0] = 0
 flow_conservation_3_1: flow[1,1] - flow[19,1] = 0
 flow_conservation_3_2: flow[1,2] - flow[19,2] = 0
 flow_conservation_3_3: flow[1,3] - flow[19,3] = 0
 flow_conservation_3_4: flow[1,4] - flow[19,4] = 0
 flow_conservation_4_0: flow[6,0] = 0
 flow_conservation_4_1: flow[6,1] = 0
 flow_conservation_4_2: flow[6,2] = 0
 flow_conservation_4_3: flow[6,3] = 0
 flow_conservation_4_4: flow[6,4] = 0
 flow_conservation_5_0: - flow[0,0] + flow[2,0] + flow[4,0] - flow[13,0]
   = -10
 flow_conservation_5_1: - flow[0,1] + flow[2,1] + flow[4,1] - flow[13,1]
   = -10
 flow_conservation_5_2: - flow[0,2] + flow[2,2] + flow[4,2] - flow[13,2]
   = -10
 flow_conservation_5_3: - flow[0,3] + flow[2,3] + flow[4,3] - flow[13,3]
   = -10
 flow_conservation_5_4: - flow[0,4] + flow[2,4] + flow[4,4] - flow[13,4]
   = -10
 flow_conservation_6_0: - flow[4,0] - flow[9,0] + flow[10,0] + flow[16,0]
   + flow[18,0] + flow[19,0] = 0
 flow_conservation_6_1: - flow[4,1] - flow[9,1] + flow[10,1] + flow[16,1]
   + flow[18,1] + flow[19,1] = 0
 flow_conservation_6_2: - flow[4,2] - flow[9,2] + flow[10,2] + flow[16,2]
   + flow[18,2] + flow[19,2] = 0
 flow_conservation_6_3: - flow[4,3] - flow[9,3] + flow[10,3] + flow[16,3]
   + flow[18,3] + flow[19,3] = 0
 flow_conservation_6_4: - flow[4,4] - flow[9,4] + flow[10,4] + flow[16,4]
   + flow[18,4] + flow[19,4] = 0
 flow_conservation_7_0: - flow[2,0] - flow[7,0] - flow[12,0] = 0
 flow_conservation_7_1: - flow[2,1] - flow[7,1] - flow[12,1] = 0
 flow_conservation_7_2: - flow[2,2] - flow[7,2] - flow[12,2] = 0
 flow_conservation_7_3: - flow[2,3] - flow[7,3] - flow[12,3] = 0
 flow_conservation_7_4: - flow[2,4] - flow[7,4] - flow[12,4] = 0
 flow_conservation_8_0: - flow[3,0] + flow[7,0] - flow[8,0] - flow[14,0]
   - flow[15,0] - flow[18,0] = 0
 flow_conservation_8_1: - flow[3,1] + flow[7,1] - flow[8,1] - flow[14,1]
   - flow[15,1] - flow[18,1] = 0
 flow_conservation_8_2: - flow[3,2] + flow[7,2] - flow[8,2] - flow[14,2]
   - flow[15,2] - flow[18,2] = 0
 flow_conservation_8_3: - flow[3,3] + flow[7,3] - flow[8,3] - flow[14,3]
   - flow[15,3] - flow[18,3] = 0
 flow_conservation_8_4: - flow[3,4] + flow[7,4] - flow[8,4] - flow[14,4]
   - flow[15,4] - flow[18,4] = 0
 flow_conservation_9_0: flow[3,0] - flow[5,0] - flow[6,0] - flow[10,0]
   + flow[11,0] + flow[15,0] = 0
 flow_conservation_9_1: flow[3,1] - flow[5,1] - flow[6,1] - flow[10,1]
   + flow[11,1] + flow[15,1] = 0
 flow_conservation_9_2: flow[3,2] - flow[5,2] - flow[6,2] - flow[10,2]
   + flow[11,2] + flow[15,2] = 0
 flow_conservation_9_3: flow[3,3] - flow[5,3] - flow[6,3] - flow[10,3]
   + flow[11,3] + flow[15,3] = 0
 flow_conservation_9_4: flow[3,4] - flow[5,4] - flow[6,4] - flow[10,4]
   + flow[11,4] + flow[15,4] = 0
Bounds
End
