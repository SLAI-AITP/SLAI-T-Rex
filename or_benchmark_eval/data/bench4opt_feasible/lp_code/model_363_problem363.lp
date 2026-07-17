\ Model electric_power_grid
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  90.96688206952432 flow[Node_5,Node_1]
   + 20.79858341848296 flow[Node_5,Node_3]
   + 2.587805887658822 flow[Node_3,Node_7]
   + 3.766472816018551 flow[Node_7,Node_8]
   + 2.441436472345488 flow[Node_0,Node_5]
   + 82.23212836699955 flow[Node_4,Node_0]
   + 68.3159930424839 flow[Node_7,Node_5]
   + 7.253977036745964 flow[Node_6,Node_8]
   + 95.91578026373922 flow[Node_4,Node_5]
   + 10.56371737193683 flow[Node_5,Node_6]
   + 83.70312320600908 flow[Node_6,Node_3]
   + 66.74676669106189 flow[Node_6,Node_1]
   + 72.67713819844781 flow[Node_7,Node_4]
   + 5.357238725962643 flow[Node_1,Node_2]
   + 51.73750996336862 flow[Node_1,Node_8]
   + 81.74680929919336 flow[Node_2,Node_0]
   + 32.94527802350638 flow[Node_7,Node_0]
   + 63.05185702268187 flow[Node_2,Node_3]
   + 62.84266196567424 flow[Node_4,Node_3]
   + 93.49840943238557 flow[Node_5,Node_8]
   + 40.11132821405977 flow[Node_6,Node_2]
   + 43.36014925658398 flow[Node_7,Node_1]
   + 24.00840563449426 flow[Node_4,Node_6]
   + 11.48264614716959 flow[Node_7,Node_2]
   + 33.01341496548454 flow[Node_1,Node_4]
   + 2.484462639777429 flow[Node_6,Node_7]
   + 65.74528319617943 flow[Node_0,Node_3]
   + 2.18615646309762 flow[Node_2,Node_4]
   + 22.06402603103829 flow[Node_8,Node_4]
   + 22.7895231808395 flow[Node_1,Node_3]
   + 66.1382085434601 flow[Node_8,Node_0]
   + 28.56730478513654 flow[Node_6,Node_0]
   + 62.76703002697987 flow[Node_8,Node_3]
Subject To
 flow_conservation_Node_0: - flow[Node_0,Node_5] + flow[Node_4,Node_0]
   + flow[Node_2,Node_0] + flow[Node_7,Node_0] - flow[Node_0,Node_3]
   + flow[Node_8,Node_0] + flow[Node_6,Node_0] + renewable[Node_0]
   = -912.1221144895069
 flow_conservation_Node_1: flow[Node_5,Node_1] + flow[Node_6,Node_1]
   - flow[Node_1,Node_2] - flow[Node_1,Node_8] + flow[Node_7,Node_1]
   - flow[Node_1,Node_4] - flow[Node_1,Node_3] + renewable[Node_1]
   = 770.9201010847219
 flow_conservation_Node_2: flow[Node_1,Node_2] - flow[Node_2,Node_0]
   - flow[Node_2,Node_3] + flow[Node_6,Node_2] + flow[Node_7,Node_2]
   - flow[Node_2,Node_4] + renewable[Node_2] = -177.8063282141949
 flow_conservation_Node_3: flow[Node_5,Node_3] - flow[Node_3,Node_7]
   + flow[Node_6,Node_3] + flow[Node_2,Node_3] + flow[Node_4,Node_3]
   + flow[Node_0,Node_3] + flow[Node_1,Node_3] + flow[Node_8,Node_3]
   + renewable[Node_3] = 535.0902616809171
 flow_conservation_Node_4: - flow[Node_4,Node_0] - flow[Node_4,Node_5]
   + flow[Node_7,Node_4] - flow[Node_4,Node_3] - flow[Node_4,Node_6]
   + flow[Node_1,Node_4] + flow[Node_2,Node_4] + flow[Node_8,Node_4]
   + renewable[Node_4] = -0.8479508295284859
 flow_conservation_Node_5: - flow[Node_5,Node_1] - flow[Node_5,Node_3]
   + flow[Node_0,Node_5] + flow[Node_7,Node_5] + flow[Node_4,Node_5]
   - flow[Node_5,Node_6] - flow[Node_5,Node_8] + renewable[Node_5]
   = 980.9485838087437
 flow_conservation_Node_6: - flow[Node_6,Node_8] + flow[Node_5,Node_6]
   - flow[Node_6,Node_3] - flow[Node_6,Node_1] - flow[Node_6,Node_2]
   + flow[Node_4,Node_6] - flow[Node_6,Node_7] - flow[Node_6,Node_0]
   + renewable[Node_6] = 425.0622173574793
 flow_conservation_Node_7: flow[Node_3,Node_7] - flow[Node_7,Node_8]
   - flow[Node_7,Node_5] - flow[Node_7,Node_4] - flow[Node_7,Node_0]
   - flow[Node_7,Node_1] - flow[Node_7,Node_2] + flow[Node_6,Node_7]
   + renewable[Node_7] = 582.5195460100663
 flow_conservation_Node_8: flow[Node_7,Node_8] + flow[Node_6,Node_8]
   + flow[Node_1,Node_8] + flow[Node_5,Node_8] - flow[Node_8,Node_4]
   - flow[Node_8,Node_0] - flow[Node_8,Node_3] + renewable[Node_8]
   = 844.4478191504631
Bounds
 flow[Node_5,Node_1] <= 713867.9577787847
 flow[Node_5,Node_3] <= 438285.9289207719
 flow[Node_3,Node_7] <= 471316.9447228779
 flow[Node_7,Node_8] <= 279862.7449551296
 flow[Node_0,Node_5] <= 162205.1645026989
 flow[Node_4,Node_0] <= 838657.3548469548
 flow[Node_7,Node_5] <= 607381.5688745275
 flow[Node_6,Node_8] <= 178059.367297966
 flow[Node_4,Node_5] <= 125027.2314706383
 flow[Node_5,Node_6] <= 741705.0365839798
 flow[Node_6,Node_3] <= 169284.9202193894
 flow[Node_6,Node_1] <= 671720.7637565244
 flow[Node_7,Node_4] <= 12199.40738160727
 flow[Node_1,Node_2] <= 273979.0125440283
 flow[Node_1,Node_8] <= 441217.3328677483
 flow[Node_2,Node_0] <= 178762.2884769421
 flow[Node_7,Node_0] <= 714697.8599608719
 flow[Node_2,Node_3] <= 892795.0180125913
 flow[Node_4,Node_3] <= 532766.5061350936
 flow[Node_5,Node_8] <= 442616.0855967225
 flow[Node_6,Node_2] <= 48708.26646435734
 flow[Node_7,Node_1] <= 635155.6212656222
 flow[Node_4,Node_6] <= 523351.5899358031
 flow[Node_7,Node_2] <= 662877.4019978495
 flow[Node_1,Node_4] <= 660380.3671979373
 flow[Node_6,Node_7] <= 103690.4933842105
 flow[Node_0,Node_3] <= 690656.3659684056
 flow[Node_2,Node_4] <= 444542.6977899882
 flow[Node_8,Node_4] <= 598607.1450238009
 flow[Node_1,Node_3] <= 49616.03012410762
 flow[Node_8,Node_0] <= 378018.1357588405
 flow[Node_6,Node_0] <= 202536.3134882728
 flow[Node_8,Node_3] <= 250450.0818645213
 renewable[Node_0] <= 569.2050618787043
 renewable[Node_1] <= 251.9401532339654
 renewable[Node_2] <= 94.74456757208061
 renewable[Node_3] <= 324.3542908237568
 renewable[Node_4] <= 630.873923721279
 renewable[Node_5] <= 388.5654410563447
 renewable[Node_6] <= 924.5534015623244
 renewable[Node_7] <= 771.4064098426755
 renewable[Node_8] <= 132.9314360920082
End
