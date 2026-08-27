\ Model ContainerAllocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  x_small + n_A + n_B
Subject To
 Balance: 25962 x_small - 18252 n_A - 47000 n_B <= 0
Bounds
 x_small >= 14
Generals
 x_small n_A n_B
End
