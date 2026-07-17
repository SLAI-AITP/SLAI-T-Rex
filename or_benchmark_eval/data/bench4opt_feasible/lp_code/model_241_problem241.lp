\ Model Capital Budgeting Problem with Multi-Period and Resource Allocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  871602.7102939819 x[0,0] + 4.0143173374982271e+06 x[0,1]
   + 5.4157626229245756e+06 x[0,2] + 1.4689949527006254e+06 x[1,0]
   + 1.8856573706248212e+06 x[1,1] + 9.2207523630795758e+06 x[1,2]
   + 451253.0200556683 x[2,0] + 4.8356134077080479e+06 x[2,1]
   + 8.5181232827806436e+06 x[2,2] + 5.8463694889884694e+06 x[3,0]
   + 5.7299465663833953e+06 x[3,1] + 8.5254893837627433e+06 x[3,2]
   + 7.0109267675002543e+06 x[4,0] + 1.7429780793445823e+06 x[4,1]
   + 1.616191874257925e+06 x[4,2]
Subject To
 Budget_0: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 1e+06
 Budget_1: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 1e+06
 Budget_2: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 1e+06
 Regulatory_0_0: x[0,0] <= 1e+06
 Regulatory_0_1: x[0,1] <= 1e+06
 Regulatory_0_2: x[0,2] <= 1e+06
 Regulatory_1_0: x[1,0] <= 1e+06
 Regulatory_1_1: x[1,1] <= 1e+06
 Regulatory_1_2: x[1,2] <= 1e+06
 Regulatory_2_0: x[2,0] <= 1e+06
 Regulatory_2_1: x[2,1] <= 1e+06
 Regulatory_2_2: x[2,2] <= 1e+06
 Regulatory_3_0: x[3,0] <= 1e+06
 Regulatory_3_1: x[3,1] <= 1e+06
 Regulatory_3_2: x[3,2] <= 1e+06
 Regulatory_4_0: x[4,0] <= 1e+06
 Regulatory_4_1: x[4,1] <= 1e+06
 Regulatory_4_2: x[4,2] <= 1e+06
 Resource_0_0: 1e+06 x[0,0] + 1e+06 x[1,0] + 1e+06 x[2,0] + 1e+06 x[3,0]
   + 1e+06 x[4,0] <= 1e+06
 Resource_0_1: 1e+06 x[0,1] + 1e+06 x[1,1] + 1e+06 x[2,1] + 1e+06 x[3,1]
   + 1e+06 x[4,1] <= 1e+06
 Resource_0_2: 1e+06 x[0,2] + 1e+06 x[1,2] + 1e+06 x[2,2] + 1e+06 x[3,2]
   + 1e+06 x[4,2] <= 1e+06
 Resource_1_0: 1e+06 x[0,0] + 1e+06 x[1,0] + 1e+06 x[2,0] + 1e+06 x[3,0]
   + 1e+06 x[4,0] <= 1e+06
 Resource_1_1: 1e+06 x[0,1] + 1e+06 x[1,1] + 1e+06 x[2,1] + 1e+06 x[3,1]
   + 1e+06 x[4,1] <= 1e+06
 Resource_1_2: 1e+06 x[0,2] + 1e+06 x[1,2] + 1e+06 x[2,2] + 1e+06 x[3,2]
   + 1e+06 x[4,2] <= 1e+06
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[1,0] x[1,1] x[1,2] x[2,0] x[2,1] x[2,2] x[3,0]
 x[3,1] x[3,2] x[4,0] x[4,1] x[4,2]
End
