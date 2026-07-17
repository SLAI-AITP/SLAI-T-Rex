\ Model Network Flow Problem with Pressure Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  20.10704585044434 flow[0] + 16.8866717490463 flow[1]
   + 3.54202780817012 flow[2] + 12.58806476113195 flow[3]
   + 47.51849574326715 flow[4] + 28.10293319036623 flow[5]
   + 32.04352346400381 flow[6] + 29.72246217787097 flow[7]
   + 32.92599457856637 flow[8]
Subject To
 cap_0: flow[0] <= 6.791676443103018
 cap_1: flow[1] <= 41.86945838785239
 cap_2: flow[2] <= 90.05064883810027
 cap_3: flow[3] <= 60.10345729453026
 cap_4: flow[4] <= 50.93770393818482
 cap_5: flow[5] <= 32.51530811203666
 cap_6: flow[6] <= 3.861737343235945
 cap_7: flow[7] <= 55.85345105367894
 cap_8: flow[8] <= 76.53485141631988
 flow_conservation_0: - flow[0] + flow[5] - flow[6] - flow[7] - flow[8]
   = -37.44286574079631
 flow_conservation_1: flow[1] - flow[3] + flow[8] = 6.824563746710055
 flow_conservation_2: - flow[1] - flow[5] + flow[7] = -5.401774433323482
 flow_conservation_3: - flow[2] + flow[6] = 34.62778526231571
 flow_conservation_4: - flow[4] = 0.0432373609072698
 flow_conservation_5: flow[0] = 28.28815875046813
 flow_conservation_6: flow[2] + flow[3] + flow[4] = -35.7632969518712
 pressure_min_0: pressure[0] >= 5.863957791535395
 pressure_max_0: pressure[0] <= 13.99039496864601
 pressure_min_1: pressure[1] >= 2.51369498698572
 pressure_max_1: pressure[1] <= 12.12483941265889
 pressure_min_2: pressure[2] >= 3.206453366859411
 pressure_max_2: pressure[2] <= 9.413752546387277
 pressure_min_3: pressure[3] >= 9.573513640565961
 pressure_max_3: pressure[3] <= 16.10410213123702
 pressure_min_4: pressure[4] >= 9.652168983246643
 pressure_max_4: pressure[4] <= 11.32598572379994
 pressure_min_5: pressure[5] >= 5.99637023408471
 pressure_max_5: pressure[5] <= 6.598120543366267
 pressure_min_6: pressure[6] >= 1.376128718324076
 pressure_max_6: pressure[6] <= 10.59423584365004
Bounds
End
