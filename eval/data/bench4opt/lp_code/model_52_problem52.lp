\ Model Multi-Commodity Network Flow Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  54.21359563100408 flow[0,0] + 41.12772098536733 flow[0,1]
   + 70.37344904955395 flow[0,2] + 39.98585794903304 flow[0,3]
   + 81.92882147509799 flow[0,4] + 78.56398101454548 flow[0,5]
   + 53.47867843037667 flow[0,6] + 7.329234610283448 flow[0,7]
   + 83.87725002234751 flow[0,8] + 54.35919678703817 flow[0,9]
   + 82.06393291217611 flow[0,10] + 65.51836548957856 flow[0,11]
   + 75.59731336344653 flow[0,12] + 21.53891331406252 flow[0,13]
   + 66.19965573297908 flow[0,14] + 12.61528065715546 flow[0,15]
   + 86.33834082120029 flow[0,16] + 69.82637307654186 flow[0,17]
   + 1.527076273541555 flow[0,18] + 45.27605494298625 flow[0,19]
   + 32.11531217531476 flow[1,0] + 4.875172933584317 flow[1,1]
   + 57.93302192849487 flow[1,2] + 65.61140953719479 flow[1,3]
   + 9.694271971934466 flow[1,4] + 26.41928795198768 flow[1,5]
   + 75.55380431909504 flow[1,6] + 9.376422464943788 flow[1,7]
   + 13.34699489257248 flow[1,8] + 30.55921430880132 flow[1,9]
   + 96.66134037746703 flow[1,10] + 17.98871665360093 flow[1,11]
   + 97.73320678306833 flow[1,12] + 66.15777740633838 flow[1,13]
   + 25.58537139564304 flow[1,14] + 5.570460048878532 flow[1,15]
   + 94.63366477507996 flow[1,16] + 90.55733181906776 flow[1,17]
   + 30.04628597391099 flow[1,18] + 53.8761374506884 flow[1,19]
   + 14.84564959354754 flow[2,0] + 49.55383968215471 flow[2,1]
   + 31.98066886323659 flow[2,2] + 91.16753250819393 flow[2,3]
   + 99.68291903182562 flow[2,4] + 61.01104121865882 flow[2,5]
   + 51.54844427685521 flow[2,6] + 69.10510059298502 flow[2,7]
   + 66.59908266482178 flow[2,8] + 5.428819712049771 flow[2,9]
   + 4.636660960460389 flow[2,10] + 15.02506576978574 flow[2,11]
   + 85.75243208132798 flow[2,12] + 22.30608114439897 flow[2,13]
   + 14.28659256258825 flow[2,14] + 55.65499204686251 flow[2,15]
   + 59.5865307416577 flow[2,16] + 28.46217615694342 flow[2,17]
   + 73.18133602745057 flow[2,18] + 42.7794861745954 flow[2,19]
   + 55.4350124466551 flow[3,0] + 26.80676977929807 flow[3,1]
   + 65.32712509827563 flow[3,2] + 17.16861808751433 flow[3,3]
   + 46.15070714981968 flow[3,4] + 94.52930787327706 flow[3,5]
   + 67.97604137740278 flow[3,6] + 94.07973348913357 flow[3,7]
   + 15.1811510626564 flow[3,8] + 21.17252679158186 flow[3,9]
   + 63.32725202690672 flow[3,10] + 27.67056907371109 flow[3,11]
   + 63.67886432247974 flow[3,12] + 64.911551356838 flow[3,13]
   + 71.7300518217838 flow[3,14] + 56.8005734985656 flow[3,15]
   + 65.12934001706773 flow[3,16] + 51.63131698138658 flow[3,17]
   + 17.2402641292584 flow[3,18] + 82.33521267172434 flow[3,19]
   + 32.31967360714413 flow[4,0] + 78.9906224901161 flow[4,1]
   + 38.51524822089878 flow[4,2] + 96.43775241641042 flow[4,3]
   + 25.50227089729111 flow[4,4] + 27.74239812297747 flow[4,5]
   + 53.61554638213681 flow[4,6] + 68.71394546544931 flow[4,7]
   + 41.27134514994901 flow[4,8] + 33.60639345182159 flow[4,9]
   + 9.808441660695735 flow[4,10] + 91.01529353125555 flow[4,11]
   + 78.0290241543356 flow[4,12] + 15.4866336585334 flow[4,13]
   + 80.01163926236913 flow[4,14] + 28.09103190136306 flow[4,15]
   + 10.69673867030771 flow[4,16] + 97.96903613886666 flow[4,17]
   + 23.00033162895293 flow[4,18] + 95.22650320377402 flow[4,19]
Subject To
 cap_0: flow[0,0] + flow[1,0] + flow[2,0] + flow[3,0] + flow[4,0]
   <= 1104.421757757689
 cap_1: flow[0,1] + flow[1,1] + flow[2,1] + flow[3,1] + flow[4,1]
   <= 2049.648707685864
 cap_2: flow[0,2] + flow[1,2] + flow[2,2] + flow[3,2] + flow[4,2]
   <= 2098.46406730239
 cap_3: flow[0,3] + flow[1,3] + flow[2,3] + flow[3,3] + flow[4,3]
   <= 6772.683948420956
 cap_4: flow[0,4] + flow[1,4] + flow[2,4] + flow[3,4] + flow[4,4]
   <= 9176.306050924482
 cap_5: flow[0,5] + flow[1,5] + flow[2,5] + flow[3,5] + flow[4,5]
   <= 3024.415360445088
 cap_6: flow[0,6] + flow[1,6] + flow[2,6] + flow[3,6] + flow[4,6]
   <= 6281.654418683328
 cap_7: flow[0,7] + flow[1,7] + flow[2,7] + flow[3,7] + flow[4,7]
   <= 9886.144020808302
 cap_8: flow[0,8] + flow[1,8] + flow[2,8] + flow[3,8] + flow[4,8]
   <= 5315.485294636512
 cap_9: flow[0,9] + flow[1,9] + flow[2,9] + flow[3,9] + flow[4,9]
   <= 7243.532670350623
 cap_10: flow[0,10] + flow[1,10] + flow[2,10] + flow[3,10] + flow[4,10]
   <= 1850.979143001835
 cap_11: flow[0,11] + flow[1,11] + flow[2,11] + flow[3,11] + flow[4,11]
   <= 6335.911055131773
 cap_12: flow[0,12] + flow[1,12] + flow[2,12] + flow[3,12] + flow[4,12]
   <= 4455.638903064453
 cap_13: flow[0,13] + flow[1,13] + flow[2,13] + flow[3,13] + flow[4,13]
   <= 9999.587720642758
 cap_14: flow[0,14] + flow[1,14] + flow[2,14] + flow[3,14] + flow[4,14]
   <= 7324.632859996767
 cap_15: flow[0,15] + flow[1,15] + flow[2,15] + flow[3,15] + flow[4,15]
   <= 3685.270124247729
 cap_16: flow[0,16] + flow[1,16] + flow[2,16] + flow[3,16] + flow[4,16]
   <= 2127.12793719002
 cap_17: flow[0,17] + flow[1,17] + flow[2,17] + flow[3,17] + flow[4,17]
   <= 3927.060772266124
 cap_18: flow[0,18] + flow[1,18] + flow[2,18] + flow[3,18] + flow[4,18]
   <= 7423.913316279089
 cap_19: flow[0,19] + flow[1,19] + flow[2,19] + flow[3,19] + flow[4,19]
   <= 8759.894394554372
 flow_conservation_0_0: - flow[0,3] + flow[0,7] + flow[0,8] - flow[0,10]
   = 31.25599460491804
 flow_conservation_0_1: - flow[0,1] - flow[0,9] + flow[0,10] - flow[0,14]
   - flow[0,17] = 255.586593984602
 flow_conservation_0_2: flow[0,1] + flow[0,4] + flow[0,5] + flow[0,6]
   - flow[0,8] - flow[0,13] = -224.8912304579258
 flow_conservation_0_3: - flow[0,7] + flow[0,14] = -317.7375918689467
 flow_conservation_0_4: flow[0,3] - flow[0,6] - flow[0,16] + flow[0,18]
   = -247.5967208155622
 flow_conservation_0_5: - flow[0,2] + flow[0,9] + flow[0,11] + flow[0,19]
   = -115.740732968625
 flow_conservation_0_6: flow[0,0] + flow[0,12] + flow[0,13] - flow[0,15]
   - flow[0,18] = -5.477718210036514
 flow_conservation_0_7: - flow[0,4] + flow[0,15] + flow[0,17] - flow[0,19]
   = -31.5246736492804
 flow_conservation_0_8: - flow[0,0] - flow[0,11] = 178.7154660651541
 flow_conservation_0_9: flow[0,2] - flow[0,5] - flow[0,12] + flow[0,16]
   = 263.1750975653929
 flow_conservation_1_0: - flow[1,3] + flow[1,7] + flow[1,8] - flow[1,10]
   = -130.1796423096233
 flow_conservation_1_1: - flow[1,1] - flow[1,9] + flow[1,10] - flow[1,14]
   - flow[1,17] = 424.6652667500577
 flow_conservation_1_2: flow[1,1] + flow[1,4] + flow[1,5] + flow[1,6]
   - flow[1,8] - flow[1,13] = 314.5348889303764
 flow_conservation_1_3: - flow[1,7] + flow[1,14] = 307.2225145410264
 flow_conservation_1_4: flow[1,3] - flow[1,6] - flow[1,16] + flow[1,18]
   = -307.7222664278275
 flow_conservation_1_5: - flow[1,2] + flow[1,9] + flow[1,11] + flow[1,19]
   = 158.5663600632496
 flow_conservation_1_6: flow[1,0] + flow[1,12] + flow[1,13] - flow[1,15]
   - flow[1,18] = -60.48156597965476
 flow_conservation_1_7: - flow[1,4] + flow[1,15] + flow[1,17] - flow[1,19]
   = -0.5188587304940597
 flow_conservation_1_8: - flow[1,0] - flow[1,11] = -128.1690256066627
 flow_conservation_1_9: flow[1,2] - flow[1,5] - flow[1,12] + flow[1,16]
   = -213.4900547159669
 flow_conservation_2_0: - flow[2,3] + flow[2,7] + flow[2,8] - flow[2,10]
   = -434.0604403367021
 flow_conservation_2_1: - flow[2,1] - flow[2,9] + flow[2,10] - flow[2,14]
   - flow[2,17] = -106.2336287125031
 flow_conservation_2_2: flow[2,1] + flow[2,4] + flow[2,5] + flow[2,6]
   - flow[2,8] - flow[2,13] = 119.543647834454
 flow_conservation_2_3: - flow[2,7] + flow[2,14] = -125.8853871201627
 flow_conservation_2_4: flow[2,3] - flow[2,6] - flow[2,16] + flow[2,18]
   = -130.8397196642075
 flow_conservation_2_5: - flow[2,2] + flow[2,9] + flow[2,11] + flow[2,19]
   = -89.44891916894665
 flow_conservation_2_6: flow[2,0] + flow[2,12] + flow[2,13] - flow[2,15]
   - flow[2,18] = -156.2511544009377
 flow_conservation_2_7: - flow[2,4] + flow[2,15] + flow[2,17] - flow[2,19]
   = 425.7759018249157
 flow_conservation_2_8: - flow[2,0] - flow[2,11] = 46.8066678026542
 flow_conservation_2_9: flow[2,2] - flow[2,5] - flow[2,12] + flow[2,16]
   = 326.5445261094673
 flow_conservation_3_0: - flow[3,3] + flow[3,7] + flow[3,8] - flow[3,10]
   = 40.36313725602861
 flow_conservation_3_1: - flow[3,1] - flow[3,9] + flow[3,10] - flow[3,14]
   - flow[3,17] = 152.8739465535119
 flow_conservation_3_2: flow[3,1] + flow[3,4] + flow[3,5] + flow[3,6]
   - flow[3,8] - flow[3,13] = 156.9637019250214
 flow_conservation_3_3: - flow[3,7] + flow[3,14] = -367.5860898261446
 flow_conservation_3_4: flow[3,3] - flow[3,6] - flow[3,16] + flow[3,18]
   = -114.7815907152578
 flow_conservation_3_5: - flow[3,2] + flow[3,9] + flow[3,11] + flow[3,19]
   = -124.8557047257743
 flow_conservation_3_6: flow[3,0] + flow[3,12] + flow[3,13] - flow[3,15]
   - flow[3,18] = 236.1993852959053
 flow_conservation_3_7: - flow[3,4] + flow[3,15] + flow[3,17] - flow[3,19]
   = 162.3752211677661
 flow_conservation_3_8: - flow[3,0] - flow[3,11] = 274.8986014224577
 flow_conservation_3_9: flow[3,2] - flow[3,5] - flow[3,12] + flow[3,16]
   = -146.9258123155939
 flow_conservation_4_0: - flow[4,3] + flow[4,7] + flow[4,8] - flow[4,10]
   = 40.11563794160622
 flow_conservation_4_1: - flow[4,1] - flow[4,9] + flow[4,10] - flow[4,14]
   - flow[4,17] = -409.6487608331464
 flow_conservation_4_2: flow[4,1] + flow[4,4] + flow[4,5] + flow[4,6]
   - flow[4,8] - flow[4,13] = -83.65156497505831
 flow_conservation_4_3: - flow[4,7] + flow[4,14] = -333.7801227668875
 flow_conservation_4_4: flow[4,3] - flow[4,6] - flow[4,16] + flow[4,18]
   = -214.152047168197
 flow_conservation_4_5: - flow[4,2] + flow[4,9] + flow[4,11] + flow[4,19]
   = -29.14006982707879
 flow_conservation_4_6: flow[4,0] + flow[4,12] + flow[4,13] - flow[4,15]
   - flow[4,18] = 167.046577689469
 flow_conservation_4_7: - flow[4,4] + flow[4,15] + flow[4,17] - flow[4,19]
   = 125.5081555492823
 flow_conservation_4_8: - flow[4,0] - flow[4,11] = 186.7008352400751
 flow_conservation_4_9: flow[4,2] - flow[4,5] - flow[4,12] + flow[4,16]
   = 30.57400221520186
Bounds
End
