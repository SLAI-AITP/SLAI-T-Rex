\ Model Capital Budgeting Problem with Renewable Energy Integration
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  11.2072035677 x[0] + 17.4112883717 x[1] + 10.9809527639 x[2]
   + 15.9441876145 x[3] + 9.68822525 x[4]
Subject To
 BudgetConstraint: 45.1082485226 x[0] + 32.0534091356 x[1]
   + 44.6196926281 x[2] + 27.2027976667 x[3] + 45.5236408795 x[4]
   <= 162.5047142284
 CapacityConstraint_0: 3.2940081683 x[0] <= 5.8536435733
 CapacityConstraint_1: 9.5823326586 x[1] <= 9.3187651855
 CapacityConstraint_2: 7.1813893192 x[2] <= 12.4458896919
 CapacityConstraint_3: 5.2684109703 x[3] <= 8.7264685938
 CapacityConstraint_4: 1.8210230953 x[4] <= 6.3011545357
 EmissionsConstraint: 8.7360737223 x[0] + 9.411298688500001 x[1]
   + 4.2293685576 x[2] + 9.1584367097 x[3] + 0.5144607997 x[4]
   <= 28.6288822294
 RenewableInvestmentConstraint: 32.0534091356 x[1] + 27.2027976667 x[3]
   + 45.5236408795 x[4] >= 44.995911927
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4]
End
