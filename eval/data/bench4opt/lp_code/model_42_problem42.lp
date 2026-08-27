\ Model Project Selection
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  28.0134311999 select[0] + 46.8470606761 select[1]
   + 8.798514174200001 select[2] + 26.6823380107 select[3]
   + 21.8715605841 select[4]
Subject To
 Budget: 11.2564560265 select[0] + 51.2338421498 select[1]
   + 16.31828741 select[2] + 48.7367502242 select[3]
   + 84.8802099654 select[4] <= 271.6449643527
 Capacity_0: 6.0486913157 select[0] <= 11.0899491555
 Capacity_1: 7.9068927867 select[1] <= 11.57831228
 Capacity_2: 6.5955142492 select[2] <= 11.9347118786
 Capacity_3: 7.0747904016 select[3] <= 5.2895318106
 Capacity_4: 6.0106577965 select[4] <= 7.7327038871
 Risk: 7.647854228 select[0] + 9.223221458199999 select[1]
   + 3.5751318653 select[2] + 1.8959150263 select[3]
   + 1.5181897544 select[4] <= 36.2937853935
 Diversification: select[0] + select[1] + select[2] + select[3] + select[4]
   >= 1
Bounds
Binaries
 select[0] select[1] select[2] select[3] select[4]
End
