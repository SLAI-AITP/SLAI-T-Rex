\ Model Production Planning Problem with Seasonal Variations
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  17609.21357834672 flow_rate[0,0] + 17609.21357834672 flow_rate[0,1]
   + 17609.21357834672 flow_rate[0,2] + 17609.21357834672 flow_rate[0,3]
   + 55923.16675720273 flow_rate[1,0] + 55923.16675720273 flow_rate[1,1]
   + 55923.16675720273 flow_rate[1,2] + 55923.16675720273 flow_rate[1,3]
   + 6793.890824316029 flow_rate[2,0] + 6793.890824316029 flow_rate[2,1]
   + 6793.890824316029 flow_rate[2,2] + 6793.890824316029 flow_rate[2,3]
   + 70962.77988165761 flow_rate[3,0] + 70962.77988165761 flow_rate[3,1]
   + 70962.77988165761 flow_rate[3,2] + 70962.77988165761 flow_rate[3,3]
   + 20550.78805034413 flow_rate[4,0] + 20550.78805034413 flow_rate[4,1]
   + 20550.78805034413 flow_rate[4,2] + 20550.78805034413 flow_rate[4,3]
   + 98127.03706293907 flow_rate[5,0] + 98127.03706293907 flow_rate[5,1]
   + 98127.03706293907 flow_rate[5,2] + 98127.03706293907 flow_rate[5,3]
   + 23939.40726525649 flow_rate[6,0] + 23939.40726525649 flow_rate[6,1]
   + 23939.40726525649 flow_rate[6,2] + 23939.40726525649 flow_rate[6,3]
   + 11444.41211476129 flow_rate[7,0] + 11444.41211476129 flow_rate[7,1]
   + 11444.41211476129 flow_rate[7,2] + 11444.41211476129 flow_rate[7,3]
   + 47901.49715516171 flow_rate[8,0] + 47901.49715516171 flow_rate[8,1]
   + 47901.49715516171 flow_rate[8,2] + 47901.49715516171 flow_rate[8,3]
Subject To
 capacity_constraint_0_0: flow_rate[0,0] <= 1.2116017778907301e+07
 capacity_constraint_0_1: flow_rate[0,1] <= 6.4247906017002854e+06
 capacity_constraint_0_2: flow_rate[0,2] <= 1.4696156122819098e+07
 capacity_constraint_0_3: flow_rate[0,3] <= 1.4252140518688422e+07
 capacity_constraint_1_0: flow_rate[1,0] <= 7.4890952721376289e+06
 capacity_constraint_1_1: flow_rate[1,1] <= 3.9712610032177805e+06
 capacity_constraint_1_2: flow_rate[1,2] <= 9.0839181112465281e+06
 capacity_constraint_1_3: flow_rate[1,3] <= 8.8094652982571535e+06
 capacity_constraint_2_0: flow_rate[2,0] <= 1.2373194915642114e+07
 capacity_constraint_2_1: flow_rate[2,1] <= 6.5611645556856049e+06
 capacity_constraint_2_2: flow_rate[2,2] <= 1.5008099817657029e+07
 capacity_constraint_2_3: flow_rate[2,3] <= 1.4554659445106592e+07
 capacity_constraint_3_0: flow_rate[3,0] <= 2.7165862697652536e+06
 capacity_constraint_3_1: flow_rate[3,1] <= 1.4405308949843671e+06
 capacity_constraint_3_2: flow_rate[3,2] <= 3.2950905710190753e+06
 capacity_constraint_3_3: flow_rate[3,3] <= 3.1955358562808055e+06
 capacity_constraint_4_0: flow_rate[4,0] <= 9.6141830132021941e+06
 capacity_constraint_4_1: flow_rate[4,1] <= 5.0981365159253431e+06
 capacity_constraint_4_2: flow_rate[4,2] <= 1.1661548962180324e+07
 capacity_constraint_4_3: flow_rate[4,3] <= 1.1309218076180678e+07
 capacity_constraint_5_0: flow_rate[5,0] <= 2.9181861406315472e+06
 capacity_constraint_5_1: flow_rate[5,1] <= 1.5474337552542344e+06
 capacity_constraint_5_2: flow_rate[5,2] <= 3.5396216727932109e+06
 capacity_constraint_5_3: flow_rate[5,3] <= 3.4326789292414482e+06
 capacity_constraint_6_0: flow_rate[6,0] <= 4.5584150020233216e+06
 capacity_constraint_6_1: flow_rate[6,1] <= 2.4172019551369706e+06
 capacity_constraint_6_2: flow_rate[6,2] <= 5.5291416507294988e+06
 capacity_constraint_6_3: flow_rate[6,3] <= 5.3620894535525264e+06
 capacity_constraint_7_0: flow_rate[7,0] <= 3.4584314488914502e+06
 capacity_constraint_7_1: flow_rate[7,1] <= 1.8339109660390744e+06
 capacity_constraint_7_2: flow_rate[7,2] <= 4.1949136622643666e+06
 capacity_constraint_7_3: flow_rate[7,3] <= 4.0681725533335609e+06
 capacity_constraint_8_0: flow_rate[8,0] <= 8.7792126486807913e+06
 capacity_constraint_8_1: flow_rate[8,1] <= 4.6553747233490385e+06
 capacity_constraint_8_2: flow_rate[8,2] <= 1.0648769428603213e+07
 capacity_constraint_8_3: flow_rate[8,3] <= 1.0327037694701182e+07
 quality_standard_constraint_0: 736025.0243493245 flow_rate[0,0]
   + 55756.8635361026 flow_rate[1,0] + 945376.9075579502 flow_rate[2,0]
   + 674978.3136220763 flow_rate[3,0] + 888089.9280496762 flow_rate[4,0]
   + 111855.5771825614 flow_rate[5,0] + 366225.268087443 flow_rate[6,0]
   + 619514.3169450272 flow_rate[7,0] + 366205.5728291464 flow_rate[8,0]
   >= 2.7786818865988798e+07
 quality_standard_constraint_1: 736025.0243493245 flow_rate[0,1]
   + 55756.8635361026 flow_rate[1,1] + 945376.9075579502 flow_rate[2,1]
   + 674978.3136220763 flow_rate[3,1] + 888089.9280496762 flow_rate[4,1]
   + 111855.5771825614 flow_rate[5,1] + 366225.268087443 flow_rate[6,1]
   + 619514.3169450272 flow_rate[7,1] + 366205.5728291464 flow_rate[8,1]
   >= 6.4305810879476089e+06
 quality_standard_constraint_2: 736025.0243493245 flow_rate[0,2]
   + 55756.8635361026 flow_rate[1,2] + 945376.9075579502 flow_rate[2,2]
   + 674978.3136220763 flow_rate[3,2] + 888089.9280496762 flow_rate[4,2]
   + 111855.5771825614 flow_rate[5,2] + 366225.268087443 flow_rate[6,2]
   + 619514.3169450272 flow_rate[7,2] + 366205.5728291464 flow_rate[8,2]
   >= 3.4588135520337693e+07
 quality_standard_constraint_3: 736025.0243493245 flow_rate[0,3]
   + 55756.8635361026 flow_rate[1,3] + 945376.9075579502 flow_rate[2,3]
   + 674978.3136220763 flow_rate[3,3] + 888089.9280496762 flow_rate[4,3]
   + 111855.5771825614 flow_rate[5,3] + 366225.268087443 flow_rate[6,3]
   + 619514.3169450272 flow_rate[7,3] + 366205.5728291464 flow_rate[8,3]
   >= 4.867406782429564e+07
Bounds
End
