\ Model Production Planning Problem with Quality Control Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  344.341418369865 production_rates[0]
   + 723.8018692462622 production_rates[1]
   + 411.4527774379047 production_rates[2]
   + 219.5613939243008 production_rates[3]
   + 657.311435197616 production_rates[4]
Subject To
 R0: 3.96203677933634 production_rates[0]
   + 8.863620131199312 production_rates[1]
   + 0.3511095942960404 production_rates[2]
   + 7.948117801185438 production_rates[3]
   + 8.136603059233293 production_rates[4] <= 5990.279245033715
 R1: 1.157003365778244 production_rates[0]
   + 4.436039118317584 production_rates[1]
   + 9.964197240547536 production_rates[2]
   + 6.469735613066968 production_rates[3]
   + 9.669083984633929 production_rates[4] <= 9707.616645525939
 R2: 6.919541565751173 production_rates[0]
   + 1.975698506723606 production_rates[1]
   + 2.461002800191955 production_rates[2]
   + 5.046873099997686 production_rates[3]
   + 0.2625410308444687 production_rates[4] <= 6946.606797866735
 R3: 4.916385066808301 production_rates[0]
   + 0.4534130574138308 production_rates[1]
   + 3.948798581388794 production_rates[2]
   + 9.653043075346037 production_rates[3]
   + 3.45892254825636 production_rates[4] <= 3689.833106569502
 R4: 5.278884429021676 production_rates[0]
   + 9.406888917185146 production_rates[1]
   + 8.71498029273099 production_rates[2]
   + 5.411957794238931 production_rates[3]
   + 7.436409843977708 production_rates[4] >= 309.687093886243
 R5: 5.278884429021676 production_rates[0]
   + 9.406888917185146 production_rates[1]
   + 8.71498029273099 production_rates[2]
   + 5.411957794238931 production_rates[3]
   + 7.436409843977708 production_rates[4] <= 1648.301253705464
Bounds
End
