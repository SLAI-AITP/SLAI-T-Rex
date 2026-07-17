\ Model TokenExchangeMaximizeToken1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x1
Subject To
 value_conservation: 8.52 x0 - 3.95 x1 - 1.82 x2 = 0
Bounds
 x0 >= 1
 x2 >= 1
Generals
 x0 x1 x2
End
