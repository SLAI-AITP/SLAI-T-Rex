\ Model Production Planning Problem with Nutritional Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  248366.3170292039 production[0] + 648434.7420617001 production[1]
   + 592939.1351257025 production[2] + 910478.6296956484 production[3]
   + 66434.73583512762 production[4]
Subject To
 demand_0: production[0] >= 305607.5437766076
 demand_1: production[1] >= 338256.6628128995
 demand_2: production[2] >= 20692.27820544233
 demand_3: production[3] >= 326790.7210017758
 demand_4: production[4] >= 461448.4770231695
 storage_0: production[0] <= 863505.153276403
 storage_1: production[1] <= 568481.3554752249
 storage_2: production[2] <= 901508.4353666861
 storage_3: production[3] <= 267352.2663666447
 storage_4: production[4] <= 696262.5178115829
 shelf_life_0: production[0] <= 588401.2443662748
 shelf_life_1: production[1] <= 985779.0167271928
 shelf_life_2: production[2] <= 378359.8558365046
 shelf_life_3: production[3] <= 390910.4895971636
 shelf_life_4: production[4] <= 572966.1061587031
 nutrition_0: 808518.2417994012 production[0]
   + 503265.7540637681 production[1] + 363970.971051293 production[2]
   + 787362.4590877134 production[3] + 48483.82975178168 production[4]
   >= 743145.1376004749
 nutrition_1: 100956.6265208504 production[0]
   + 765138.9966288221 production[1] + 436555.1411401992 production[2]
   + 840540.1082223948 production[3] + 996231.4656670926 production[4]
   >= 92865.87740472796
 nutrition_2: 675436.1840315342 production[0]
   + 340199.8854015811 production[1] + 497454.7597304638 production[2]
   + 536298.1894378812 production[3] + 402625.3248363615 production[4]
   >= 844342.6971183348
 nutrition_3: 336631.495023032 production[0]
   + 615213.0456806822 production[1] + 847486.7600585009 production[2]
   + 288068.2035005833 production[3] + 226660.0910570301 production[4]
   >= 95638.88154382096
Bounds
End
