\ Model warehouse_storage_optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  y#1#1 + y#1#2 + y#1#3 + y#1#4 + y#1#5 + y#1#6 + y#1#7 + y#1#8 + y#1#9
   + y#1#10 + y#2#1 + y#2#2 + y#2#3 + y#2#4 + y#2#5 + y#2#6 + y#2#7 + y#2#8
   + y#2#9 + y#2#10 + y#3#1 + y#3#2 + y#3#3 + y#3#4 + y#3#5 + y#3#6 + y#3#7
   + y#3#8 + y#3#9 + y#3#10 + y#4#1 + y#4#2 + y#4#3 + y#4#4 + y#4#5 + y#4#6
   + y#4#7 + y#4#8 + y#4#9 + y#4#10 + y#5#1 + y#5#2 + y#5#3 + y#5#4 + y#5#5
   + y#5#6 + y#5#7 + y#5#8 + y#5#9 + y#5#10 + y#6#1 + y#6#2 + y#6#3 + y#6#4
   + y#6#5 + y#6#6 + y#6#7 + y#6#8 + y#6#9 + y#6#10 + y#7#1 + y#7#2 + y#7#3
   + y#7#4 + y#7#5 + y#7#6 + y#7#7 + y#7#8 + y#7#9 + y#7#10 + y#8#1 + y#8#2
   + y#8#3 + y#8#4 + y#8#5 + y#8#6 + y#8#7 + y#8#8 + y#8#9 + y#8#10 + y#9#1
   + y#9#2 + y#9#3 + y#9#4 + y#9#5 + y#9#6 + y#9#7 + y#9#8 + y#9#9 + y#9#10
   + y#10#1 + y#10#2 + y#10#3 + y#10#4 + y#10#5 + y#10#6 + y#10#7 + y#10#8
   + y#10#9 + y#10#10
Subject To
 storage_accessibility_4_6: y#3#5 + y#3#6 + y#3#7 + y#4#5 + y#4#7 + y#5#5
   + y#5#6 + y#5#7 - s#4#6 >= 0
 storage_accessibility_5_1: y#4#1 + y#4#2 + y#5#2 + y#6#1 + y#6#2 - s#5#1
   >= 0
 storage_accessibility_9_5: y#8#4 + y#8#5 + y#8#6 + y#9#4 + y#9#6 + y#10#4
   + y#10#5 + y#10#6 - s#9#5 >= 0
 storage_accessibility_5_10: y#4#9 + y#4#10 + y#5#9 + y#6#9 + y#6#10
   - s#5#10 >= 0
 storage_accessibility_8_9: y#7#8 + y#7#9 + y#7#10 + y#8#8 + y#8#10 + y#9#8
   + y#9#9 + y#9#10 - s#8#9 >= 0
 storage_accessibility_9_8: y#8#7 + y#8#8 + y#8#9 + y#9#7 + y#9#9 + y#10#7
   + y#10#8 + y#10#9 - s#9#8 >= 0
 storage_accessibility_2_2: y#1#1 + y#1#2 + y#1#3 + y#2#1 + y#2#3 + y#3#1
   + y#3#2 + y#3#3 - s#2#2 >= 0
 storage_accessibility_1_6: y#1#5 + y#1#7 + y#2#5 + y#2#6 + y#2#7 - s#1#6
   >= 0
 storage_accessibility_6_2: y#5#1 + y#5#2 + y#5#3 + y#6#1 + y#6#3 + y#7#1
   + y#7#2 + y#7#3 - s#6#2 >= 0
 storage_accessibility_4_5: y#3#4 + y#3#5 + y#3#6 + y#4#4 + y#4#6 + y#5#4
   + y#5#5 + y#5#6 - s#4#5 >= 0
 storage_accessibility_3_9: y#2#8 + y#2#9 + y#2#10 + y#3#8 + y#3#10 + y#4#8
   + y#4#9 + y#4#10 - s#3#9 >= 0
 storage_accessibility_5_3: y#4#2 + y#4#3 + y#4#4 + y#5#2 + y#5#4 + y#6#2
   + y#6#3 + y#6#4 - s#5#3 >= 0
 storage_accessibility_9_1: y#8#1 + y#8#2 + y#9#2 + y#10#1 + y#10#2 - s#9#1
   >= 0
 storage_accessibility_9_7: y#8#6 + y#8#7 + y#8#8 + y#9#6 + y#9#8 + y#10#6
   + y#10#7 + y#10#8 - s#9#7 >= 0
 storage_accessibility_2_7: y#1#6 + y#1#7 + y#1#8 + y#2#6 + y#2#8 + y#3#6
   + y#3#7 + y#3#8 - s#2#7 >= 0
 storage_accessibility_3_5: y#2#4 + y#2#5 + y#2#6 + y#3#4 + y#3#6 + y#4#4
   + y#4#5 + y#4#6 - s#3#5 >= 0
 storage_accessibility_5_2: y#4#1 + y#4#2 + y#4#3 + y#5#1 + y#5#3 + y#6#1
   + y#6#2 + y#6#3 - s#5#2 >= 0
 storage_accessibility_3_8: y#2#7 + y#2#8 + y#2#9 + y#3#7 + y#3#9 + y#4#7
   + y#4#8 + y#4#9 - s#3#8 >= 0
 storage_accessibility_2_3: y#1#2 + y#1#3 + y#1#4 + y#2#2 + y#2#4 + y#3#2
   + y#3#3 + y#3#4 - s#2#3 >= 0
 storage_accessibility_2_9: y#1#8 + y#1#9 + y#1#10 + y#2#8 + y#2#10 + y#3#8
   + y#3#9 + y#3#10 - s#2#9 >= 0
 storage_capacity_4_6: s#4#6 <= 2.8
 storage_capacity_5_1: s#5#1 <= 3.01
 storage_capacity_9_5: s#9#5 <= 5.99
 storage_capacity_5_10: s#5#10 <= 8.59
 storage_capacity_8_9: s#8#9 <= 2.12
 storage_capacity_9_8: s#9#8 <= 4.41
 storage_capacity_2_2: s#2#2 <= 7
 storage_capacity_1_6: s#1#6 <= 4.28
 storage_capacity_6_2: s#6#2 <= 1.25
 storage_capacity_4_5: s#4#5 <= 5.24
 storage_capacity_3_9: s#3#9 <= 7.62
 storage_capacity_5_3: s#5#3 <= 4.59
 storage_capacity_9_1: s#9#1 <= 5.56
 storage_capacity_9_7: s#9#7 <= 6.98
 storage_capacity_2_7: s#2#7 <= 7.75
 storage_capacity_3_5: s#3#5 <= 9.32
 storage_capacity_5_2: s#5#2 <= 6.94
 storage_capacity_3_8: s#3#8 <= 7.03
 storage_capacity_2_3: s#2#3 <= 4.18
 storage_capacity_2_9: s#2#9 <= 3.13
Bounds
Binaries
 y#1#1 y#1#2 y#1#3 y#1#4 y#1#5 y#1#6 y#1#7 y#1#8 y#1#9 y#1#10 y#2#1 y#2#2
 y#2#3 y#2#4 y#2#5 y#2#6 y#2#7 y#2#8 y#2#9 y#2#10 y#3#1 y#3#2 y#3#3 y#3#4
 y#3#5 y#3#6 y#3#7 y#3#8 y#3#9 y#3#10 y#4#1 y#4#2 y#4#3 y#4#4 y#4#5 y#4#6
 y#4#7 y#4#8 y#4#9 y#4#10 y#5#1 y#5#2 y#5#3 y#5#4 y#5#5 y#5#6 y#5#7 y#5#8
 y#5#9 y#5#10 y#6#1 y#6#2 y#6#3 y#6#4 y#6#5 y#6#6 y#6#7 y#6#8 y#6#9 y#6#10
 y#7#1 y#7#2 y#7#3 y#7#4 y#7#5 y#7#6 y#7#7 y#7#8 y#7#9 y#7#10 y#8#1 y#8#2
 y#8#3 y#8#4 y#8#5 y#8#6 y#8#7 y#8#8 y#8#9 y#8#10 y#9#1 y#9#2 y#9#3 y#9#4
 y#9#5 y#9#6 y#9#7 y#9#8 y#9#9 y#9#10 y#10#1 y#10#2 y#10#3 y#10#4 y#10#5
 y#10#6 y#10#7 y#10#8 y#10#9 y#10#10 s#4#6 s#5#1 s#9#5 s#5#10 s#8#9 s#9#8
 s#2#2 s#1#6 s#6#2 s#4#5 s#3#9 s#5#3 s#9#1 s#9#7 s#2#7 s#3#5 s#5#2 s#3#8
 s#2#3 s#2#9
End
