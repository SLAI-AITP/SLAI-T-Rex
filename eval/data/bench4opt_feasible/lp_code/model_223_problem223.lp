\ Model ContainerConsolidation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x_small
Subject To
 Balance: 43500 x_small - 41010 n_A - 21120 n_B = 0
Bounds
 x_small >= 43
Generals
 x_small n_A n_B
End
