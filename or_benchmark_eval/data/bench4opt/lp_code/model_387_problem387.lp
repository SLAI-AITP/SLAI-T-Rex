\ Model Production Planning Problem with Environmental Impact
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  337501.2231028942 production[0] + 68202.50673489169 production[1]
   + 801384.485800062 production[2] + 475793.0987676887 production[3]
   + 210529.4194964257 production[4] + 795856.0975982615 production[5]
   + 651885.2767416477 production[6] + 302545.167596153 production[7]
   + 804322.5852221928 production[8] + 433477.5740637871 production[9]
Subject To
 demand_0: production[0] >= 54040.76315763563
 demand_1: production[1] >= 20087.69084803755
 demand_2: production[2] >= 40289.86547813826
 demand_3: production[3] >= 62533.89120985109
 demand_4: production[4] >= 45377.53767213872
 demand_5: production[5] >= 76367.87727178255
 demand_6: production[6] >= 18142.02500359392
 demand_7: production[7] >= 1958.471879524058
 demand_8: production[8] >= 70362.83140533259
 demand_9: production[9] >= 86284.70969524414
 storage_0: production[0] <= 577266.4519451888
 storage_1: production[1] <= 213739.8522134528
 storage_2: production[2] <= 962154.9575335091
 storage_3: production[3] <= 742465.8029762275
 storage_4: production[4] <= 467828.0180742443
 storage_5: production[5] <= 937225.7626599701
 storage_6: production[6] <= 533186.6156915168
 storage_7: production[7] <= 614807.8755730529
 storage_8: production[8] <= 769350.0459784939
 storage_9: production[9] <= 434290.7603658832
 shelf_life_0: production[0] <= 56655.03262869152
 shelf_life_1: production[1] <= 62713.17131059004
 shelf_life_2: production[2] <= 52274.731614774
 shelf_life_3: production[3] <= 36491.03944899455
 shelf_life_4: production[4] <= 14859.59451357541
 shelf_life_5: production[5] <= 15132.21813491514
 shelf_life_6: production[6] <= 68562.13300955635
 shelf_life_7: production[7] <= 30097.53991162618
 shelf_life_8: production[8] <= 43595.3395402804
 shelf_life_9: production[9] <= 31933.45269734096
 emission_limit: 8303.841661753257 production[0]
   + 653.6310199866855 production[1] + 2043.544834563212 production[2]
   + 4130.259413470911 production[3] + 8238.869937938269 production[4]
   + 6463.966109709719 production[5] + 2571.901882452372 production[6]
   + 6016.310374058313 production[7] + 4064.600733089814 production[8]
   + 1144.450180516562 production[9] <= 1.9272409546879276e+06
Bounds
End
