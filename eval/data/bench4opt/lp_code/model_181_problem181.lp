\ Model Production Planning Problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  85870.43435631637 production[0] + 865335.1306463331 production[1]
   + 519400.8571444945 production[2] + 825144.352458033 production[3]
   + 373984.9897373148 production[4] + 520605.294932454 production[5]
   + 466109.5141225525 production[6] + 863673.6761630634 production[7]
   + 844852.3255795107 production[8]
Subject To
 demand_0: production[0] >= 910789.4646323631
 demand_1: production[1] >= 308297.3365095154
 demand_2: production[2] >= 33837.55740517275
 demand_3: production[3] >= 93287.73734442734
 demand_4: production[4] >= 54500.71585116161
 demand_5: production[5] >= 269295.3814396002
 demand_6: production[6] >= 210044.8174778762
 demand_7: production[7] >= 418846.6292096356
 demand_8: production[8] >= 504422.7752011791
 storage_0: production[0] <= 861724.5505187594
 storage_1: production[1] <= 582590.9535867156
 storage_2: production[2] <= 791501.47327402
 storage_3: production[3] <= 894834.4619310942
 storage_4: production[4] <= 940363.4596879148
 storage_5: production[5] <= 164546.7233893418
 storage_6: production[6] <= 498108.7194056853
 storage_7: production[7] <= 838192.9072623161
 storage_8: production[8] <= 56614.89287881115
 shelf_life_0: production[0] <= 612149.4367457523
 shelf_life_1: production[1] <= 514989.677488188
 shelf_life_2: production[2] <= 68250.0087052826
 shelf_life_3: production[3] <= 581385.4859117442
 shelf_life_4: production[4] <= 367102.9566386537
 shelf_life_5: production[5] <= 927963.9470497026
 shelf_life_6: production[6] <= 815016.8084217876
 shelf_life_7: production[7] <= 497807.7213372531
 shelf_life_8: production[8] <= 954833.5509676343
Bounds
End
