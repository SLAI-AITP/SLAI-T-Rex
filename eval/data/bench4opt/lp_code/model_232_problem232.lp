\ Model TokenExchangeMaximization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x1
Subject To
 value_conservation: 1.76 x0 - 5.64 x1 - 6.7 x2 = 0
Bounds
 x0 >= 1
Generals
 x0 x1 x2
End
