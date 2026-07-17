\ Model Capital Budgeting Problem with Energy Storage Consideration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  8168.5096883047 x[0] + 7058.5220788599 x[1] + 8832.792560321899 x[2]
   + 9921.8353390676 x[3] + 5115.9485985397 x[4]
Subject To
 BudgetConstraint: 1408.7562634398 x[0] + 4557.3705647887 x[1]
   + 6393.9246125001 x[2] + 1069.1837730989 x[3] + 3712.7742581659 x[4]
   <= 20562.2460975694
 CapacityConstraint_0: 162.0521185145 x[0] <= 266.670840484
 CapacityConstraint_1: 76.3542202814 x[1] <= 261.2400373134
 CapacityConstraint_2: 107.0311168573 x[2] <= 114.4150993184
 CapacityConstraint_3: 155.5506895074 x[3] <= 272.3528724045
 CapacityConstraint_4: 125.0393519834 x[4] <= 108.4604523126
 MinStorageInvestment: 4557.3705647887 x[1] + 6393.9246125001 x[2]
   + 1069.1837730989 x[3] + 3712.7742581659 x[4] >= 7895.1520558576
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
