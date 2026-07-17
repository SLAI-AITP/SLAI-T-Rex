\ Model Portfolio Optimization Problem with Nutrient Management
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  914.0813382201558 x[0] + 442.0152399655984 x[1] + 132.3848420810658 x[2]
   + 704.7135259453603 x[3] + 727.1704825582187 x[4]
Subject To
 Land_Constraint: x[0] + x[1] + x[2] + x[3] + x[4] <= 2175.84870642692
 Water_Constraint: 4.020714210727401 x[0] + 8.228314277628495 x[1]
   + 8.682639948598128 x[2] + 8.057083188151031 x[3]
   + 4.664755332213079 x[4] <= 4079.158188336052
 Nutrient_Constraint: 9.486156300575672 x[0] + 7.472053962605383 x[1]
   + 5.971443844855806 x[2] + 5.399473775437252 x[3]
   + 3.400052769107782 x[4] <= 9220.216155416279
Bounds
End
