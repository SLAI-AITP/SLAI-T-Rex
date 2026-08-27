\ Model workforce_scheduling
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  w#0 + w#1 + w#2 + w#3 + w#4 + w#5 + w#6 + w#7 + w#8 + w#9 + w#10 + w#11
   + w#12 + w#13 + w#14 + w#15 + w#16 + w#17 + w#18
Subject To
 task_covered#0: w#0 + w#1 + w#2 + w#6 + w#8 + w#9 + w#10 + w#12 + w#15
   + w#18 - t#0 >= 0
 task_covered#1: w#0 + w#1 + w#4 + w#6 + w#11 + w#12 + w#15 + w#16 + w#18
   - t#1 >= 0
 task_covered#2: w#2 + w#3 + w#4 + w#6 + w#8 + w#10 + w#11 + w#12 + w#14
   + w#18 - t#2 >= 0
 task_covered#3: w#4 + w#7 + w#8 + w#9 + w#10 + w#11 + w#12 + w#13 - t#3
   >= 0
 task_covered#4: w#1 + w#2 + w#4 + w#8 + w#9 + w#10 + w#12 + w#13 + w#15
   + w#17 - t#4 >= 0
 task_covered#5: w#1 + w#2 + w#3 + w#4 + w#10 + w#12 + w#13 + w#14 + w#15
   + w#16 + w#18 - t#5 >= 0
 task_covered#6: w#0 + w#2 + w#4 + w#6 + w#7 + w#8 + w#9 + w#10 + w#11
   + w#12 + w#18 - t#6 >= 0
 task_covered#7: w#1 + w#3 + w#4 + w#7 + w#9 + w#12 + w#14 + w#16 + w#18
   - t#7 >= 0
 task_covered#8: w#1 + w#2 + w#7 + w#8 + w#9 + w#11 + w#13 + w#15 + w#16
   - t#8 >= 0
 task_covered#9: w#2 + w#4 + w#6 + w#8 + w#15 + w#16 - t#9 >= 0
 task_covered#10: w#0 + w#1 + w#2 + w#3 + w#5 + w#6 + w#7 + w#8 + w#9
   + w#11 + w#13 + w#14 + w#16 + w#18 - t#10 >= 0
 task_covered#11: w#0 + w#1 + w#3 + w#6 + w#8 + w#9 + w#10 + w#11 + w#12
   + w#14 + w#16 - t#11 >= 0
 task_covered#12: w#0 + w#1 + w#2 + w#3 + w#4 + w#7 + w#8 + w#10 + w#11
   + w#13 + w#14 + w#16 + w#17 + w#18 - t#12 >= 0
 worker_limit: w#0 + w#1 + w#2 + w#3 + w#4 + w#5 + w#6 + w#7 + w#8 + w#9
   + w#10 + w#11 + w#12 + w#13 + w#14 + w#15 + w#16 + w#17 + w#18 <= 6
 task_compatibility#0: - w#0 - w#1 - w#2 - w#6 - w#8 - w#9 - w#10 - w#12
   - w#15 - w#18 + t#0 <= 0
 task_compatibility#1: - w#0 - w#1 - w#4 - w#6 - w#11 - w#12 - w#15 - w#16
   - w#18 + t#1 <= 0
 task_compatibility#2: - w#2 - w#3 - w#4 - w#6 - w#8 - w#10 - w#11 - w#12
   - w#14 - w#18 + t#2 <= 0
 task_compatibility#3: - w#4 - w#7 - w#8 - w#9 - w#10 - w#11 - w#12 - w#13
   + t#3 <= 0
 task_compatibility#4: - w#1 - w#2 - w#4 - w#8 - w#9 - w#10 - w#12 - w#13
   - w#15 - w#17 + t#4 <= 0
 task_compatibility#5: - w#1 - w#2 - w#3 - w#4 - w#10 - w#12 - w#13 - w#14
   - w#15 - w#16 - w#18 + t#5 <= 0
 task_compatibility#6: - w#0 - w#2 - w#4 - w#6 - w#7 - w#8 - w#9 - w#10
   - w#11 - w#12 - w#18 + t#6 <= 0
 task_compatibility#7: - w#1 - w#3 - w#4 - w#7 - w#9 - w#12 - w#14 - w#16
   - w#18 + t#7 <= 0
 task_compatibility#8: - w#1 - w#2 - w#7 - w#8 - w#9 - w#11 - w#13 - w#15
   - w#16 + t#8 <= 0
 task_compatibility#9: - w#2 - w#4 - w#6 - w#8 - w#15 - w#16 + t#9 <= 0
 task_compatibility#10: - w#0 - w#1 - w#2 - w#3 - w#5 - w#6 - w#7 - w#8
   - w#9 - w#11 - w#13 - w#14 - w#16 - w#18 + t#10 <= 0
 task_compatibility#11: - w#0 - w#1 - w#3 - w#6 - w#8 - w#9 - w#10 - w#11
   - w#12 - w#14 - w#16 + t#11 <= 0
 task_compatibility#12: - w#0 - w#1 - w#2 - w#3 - w#4 - w#7 - w#8 - w#10
   - w#11 - w#13 - w#14 - w#16 - w#17 - w#18 + t#12 <= 0
Bounds
Binaries
 w#0 w#1 w#2 w#3 w#4 w#5 w#6 w#7 w#8 w#9 w#10 w#11 w#12 w#13 w#14 w#15 w#16
 w#17 w#18 t#0 t#1 t#2 t#3 t#4 t#5 t#6 t#7 t#8 t#9 t#10 t#11 t#12
End
