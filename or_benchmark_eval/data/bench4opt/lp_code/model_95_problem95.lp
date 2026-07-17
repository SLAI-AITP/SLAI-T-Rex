\ Model ContainerAllocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  n_B
Subject To
 Balance: 30496 x_small - 40711 n_A - 51026 n_B <= 0
Bounds
 x_small >= 1
Generals
 x_small n_A n_B
End
