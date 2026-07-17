\ Model Portfolio Optimization Problem with Soil Quality and Crop Rotation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  4688.819829368938 x[0] + 3444.233188983117 x[1] + 6687.196909255849 x[2]
   + 1417.835318004684 x[3] + 5346.374019522662 x[4]
   + 2535.470451026388 x[5] + 7154.639627714114 x[6]
   + 5727.027274866097 x[7] + 9485.249547278141 x[8]
   + 7320.137316192144 x[9]
Subject To
 Land_Constraint: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]
   + x[8] + x[9] <= 879925.2199974434
 Water_Constraint: 786.2845219126901 x[0] + 369.780756170681 x[1]
   + 468.7421708011885 x[2] + 863.8142975891889 x[3]
   + 794.1575213084072 x[4] + 381.0684721735122 x[5]
   + 313.8299930943439 x[6] + 398.9012651884419 x[7]
   + 475.5301257786429 x[8] + 547.4004192331065 x[9] <= 452867.4907313283
 Nutrient_Constraint: 905.6293029677794 x[0] + 956.1755146595229 x[1]
   + 871.4098095697174 x[2] + 417.9966682223741 x[3]
   + 207.4156497791883 x[4] + 344.3063212646256 x[5]
   + 891.0254075016986 x[6] + 663.0549465153186 x[7]
   + 937.7507565343815 x[8] + 437.465980861426 x[9] <= 943256.0809443694
 Soil_Quality_Constraint: 33.59704201690786 x[0] + 214.388748721115 x[1]
   + 571.7189765000611 x[2] + 180.6173820837347 x[3]
   + 983.7962296494244 x[4] + 683.8238648901162 x[5]
   + 771.5476300142491 x[6] + 644.8257596631178 x[7]
   + 579.3184513154649 x[8] + 802.8284349727008 x[9] <= 1591.764244971933
 Crop_Rotation_Constraint_0_2: [ x[0] * x[2] ] = 0
 Crop_Rotation_Constraint_0_3: [ x[0] * x[3] ] = 0
 Crop_Rotation_Constraint_0_6: [ x[0] * x[6] ] = 0
 Crop_Rotation_Constraint_0_7: [ x[0] * x[7] ] = 0
 Crop_Rotation_Constraint_0_8: [ x[0] * x[8] ] = 0
 Crop_Rotation_Constraint_0_9: [ x[0] * x[9] ] = 0
 Crop_Rotation_Constraint_1_0: [ x[0] * x[1] ] = 0
 Crop_Rotation_Constraint_1_2: [ x[1] * x[2] ] = 0
 Crop_Rotation_Constraint_1_4: [ x[1] * x[4] ] = 0
 Crop_Rotation_Constraint_1_7: [ x[1] * x[7] ] = 0
 Crop_Rotation_Constraint_1_8: [ x[1] * x[8] ] = 0
 Crop_Rotation_Constraint_1_9: [ x[1] * x[9] ] = 0
 Crop_Rotation_Constraint_2_1: [ x[1] * x[2] ] = 0
 Crop_Rotation_Constraint_2_7: [ x[2] * x[7] ] = 0
 Crop_Rotation_Constraint_3_0: [ x[0] * x[3] ] = 0
 Crop_Rotation_Constraint_3_1: [ x[1] * x[3] ] = 0
 Crop_Rotation_Constraint_3_2: [ x[2] * x[3] ] = 0
 Crop_Rotation_Constraint_3_5: [ x[3] * x[5] ] = 0
 Crop_Rotation_Constraint_3_7: [ x[3] * x[7] ] = 0
 Crop_Rotation_Constraint_4_0: [ x[0] * x[4] ] = 0
 Crop_Rotation_Constraint_4_1: [ x[1] * x[4] ] = 0
 Crop_Rotation_Constraint_4_2: [ x[2] * x[4] ] = 0
 Crop_Rotation_Constraint_4_5: [ x[4] * x[5] ] = 0
 Crop_Rotation_Constraint_4_6: [ x[4] * x[6] ] = 0
 Crop_Rotation_Constraint_5_1: [ x[1] * x[5] ] = 0
 Crop_Rotation_Constraint_5_2: [ x[2] * x[5] ] = 0
 Crop_Rotation_Constraint_5_7: [ x[5] * x[7] ] = 0
 Crop_Rotation_Constraint_6_1: [ x[1] * x[6] ] = 0
 Crop_Rotation_Constraint_6_3: [ x[3] * x[6] ] = 0
 Crop_Rotation_Constraint_6_4: [ x[4] * x[6] ] = 0
 Crop_Rotation_Constraint_6_5: [ x[5] * x[6] ] = 0
 Crop_Rotation_Constraint_6_8: [ x[6] * x[8] ] = 0
 Crop_Rotation_Constraint_6_9: [ x[6] * x[9] ] = 0
 Crop_Rotation_Constraint_7_0: [ x[0] * x[7] ] = 0
 Crop_Rotation_Constraint_7_3: [ x[3] * x[7] ] = 0
 Crop_Rotation_Constraint_7_6: [ x[6] * x[7] ] = 0
 Crop_Rotation_Constraint_7_8: [ x[7] * x[8] ] = 0
 Crop_Rotation_Constraint_8_1: [ x[1] * x[8] ] = 0
 Crop_Rotation_Constraint_8_3: [ x[3] * x[8] ] = 0
 Crop_Rotation_Constraint_8_6: [ x[6] * x[8] ] = 0
 Crop_Rotation_Constraint_8_7: [ x[7] * x[8] ] = 0
 Crop_Rotation_Constraint_9_4: [ x[4] * x[9] ] = 0
 Crop_Rotation_Constraint_9_7: [ x[7] * x[9] ] = 0
Bounds
End
