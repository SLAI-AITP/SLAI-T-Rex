\ Model Production Planning Problem with Quality Control
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  5543.152804337348 production[0] + 5881.943032834983 production[1]
   + 8285.334376521787 production[2] + 5864.760882553311 production[3]
   + 1159.033533731284 production[4] + 7875.684159132738 production[5]
   + 9916.404089880418 production[6] + 8921.131038922462 production[7]
Subject To
 demand_0: production[0] >= 1036.103488701093
 demand_1: production[1] >= 2525.51983606455
 demand_2: production[2] >= 4337.934301745255
 demand_3: production[3] >= 2491.502658042707
 demand_4: production[4] >= 661.9150351574201
 demand_5: production[5] >= 1295.911278612907
 demand_6: production[6] >= 3467.349621442133
 demand_7: production[7] >= 4007.221112874899
 storage_0: production[0] <= 8962.30206157664
 storage_1: production[1] <= 2477.298195833835
 storage_2: production[2] <= 4876.130060929017
 storage_3: production[3] <= 3812.137078058039
 storage_4: production[4] <= 4501.724417411493
 storage_5: production[5] <= 7876.100242021962
 storage_6: production[6] <= 6119.697012567677
 storage_7: production[7] <= 5414.983406221176
 shelf_life_0: production[0] <= 598.4055088224763
 shelf_life_1: production[1] <= 585.453147552827
 shelf_life_2: production[2] <= 522.8365415954586
 shelf_life_3: production[3] <= 606.0298701294982
 shelf_life_4: production[4] <= 240.6798612774651
 shelf_life_5: production[5] <= 150.1634132284398
 shelf_life_6: production[6] <= 337.0518859119863
 shelf_life_7: production[7] <= 631.9406067861891
 quality_0: 0.8285980848062889 production[0] >= 220.0089672775779
 quality_1: 0.625570280285253 production[1] >= 354.4195026810742
 quality_2: 0.7886746183202311 production[2] >= 394.9118453264409
 quality_3: 0.6626377001936052 production[3] >= 177.9968928114865
 quality_4: 0.4425160399646688 production[4] >= 77.64078760643409
 quality_5: 0.4853333712416047 production[5] >= 315.7085368059822
 quality_6: 0.992750970435362 production[6] >= 107.6891286419359
 quality_7: 0.4707366423021431 production[7] >= 214.8845423534106
Bounds
End
