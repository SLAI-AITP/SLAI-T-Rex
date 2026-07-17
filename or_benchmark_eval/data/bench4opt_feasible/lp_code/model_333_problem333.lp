\ Model Staff Scheduling Problem with Shift Overlap Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  22.6630851126598 x[0,0] + 83.00302049948115 x[0,1]
   + 68.64573656139297 x[0,2] + 24.15012682678364 x[0,3]
   + 43.24152643096723 x[0,4] + 29.72688315595149 x[0,5]
   + 33.20483663129038 x[0,6] + 34.81391008304684 x[0,7]
   + 29.55790448345587 x[1,0] + 88.99743180998907 x[1,1]
   + 65.41270151372805 x[1,2] + 56.28512367501685 x[1,3]
   + 14.09143565193116 x[1,4] + 46.20074717290034 x[1,5]
   + 25.13122584460826 x[1,6] + 44.77836901414766 x[1,7]
   + 72.48331568385403 x[2,0] + 59.95491133179899 x[2,1]
   + 74.9612218363535 x[2,2] + 52.40801604219545 x[2,3]
   + 90.63643577051327 x[2,4] + 70.68066529852847 x[2,5]
   + 88.77180339901689 x[2,6] + 18.49183763472311 x[2,7]
   + 73.16727312468029 x[3,0] + 97.80223189706425 x[3,1]
   + 21.58316428611635 x[3,2] + 18.11209595626977 x[3,3]
   + 70.00656388800894 x[3,4] + 83.98609360966702 x[3,5]
   + 45.15579274814107 x[3,6] + 31.75436220731941 x[3,7]
   + 99.62522146674334 x[4,0] + 60.57129129837345 x[4,1]
   + 70.71653800433332 x[4,2] + 47.14101666786244 x[4,3]
   + 87.29256338757443 x[4,4] + 70.57508352019158 x[4,5]
   + 24.54623032484447 x[4,6] + 57.87811258084815 x[4,7]
Subject To
 task_0_shift_0: x[0,0] >= 1
 task_0_shift_1: x[0,1] >= 1
 task_0_shift_2: x[0,2] >= 1
 task_0_shift_3: x[0,3] >= 1
 task_0_shift_4: x[0,4] >= 1
 task_0_shift_5: x[0,5] >= 1
 task_0_shift_6: x[0,6] >= 1
 task_0_shift_7: x[0,7] >= 1
 task_1_shift_0: x[1,0] >= 1
 task_1_shift_1: x[1,1] >= 1
 task_1_shift_2: x[1,2] >= 1
 task_1_shift_3: x[1,3] >= 1
 task_1_shift_4: x[1,4] >= 1
 task_1_shift_5: x[1,5] >= 1
 task_1_shift_6: x[1,6] >= 1
 task_1_shift_7: x[1,7] >= 1
 task_2_shift_0: x[2,0] >= 1
 task_2_shift_1: x[2,1] >= 1
 task_2_shift_2: x[2,2] >= 1
 task_2_shift_3: x[2,3] >= 1
 task_2_shift_4: x[2,4] >= 1
 task_2_shift_5: x[2,5] >= 1
 task_2_shift_6: x[2,6] >= 1
 task_2_shift_7: x[2,7] >= 1
 task_3_shift_0: x[3,0] >= 1
 task_3_shift_1: x[3,1] >= 1
 task_3_shift_2: x[3,2] >= 1
 task_3_shift_3: x[3,3] >= 1
 task_3_shift_4: x[3,4] >= 1
 task_3_shift_5: x[3,5] >= 1
 task_3_shift_6: x[3,6] >= 1
 task_3_shift_7: x[3,7] >= 1
 task_4_shift_0: x[4,0] >= 1
 task_4_shift_1: x[4,1] >= 1
 task_4_shift_2: x[4,2] >= 1
 task_4_shift_3: x[4,3] >= 1
 task_4_shift_4: x[4,4] >= 1
 task_4_shift_5: x[4,5] >= 1
 task_4_shift_6: x[4,6] >= 1
 task_4_shift_7: x[4,7] >= 1
 shift_0_availability: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 100
 shift_1_availability: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 100
 shift_2_availability: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 100
 shift_3_availability: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] <= 100
 shift_4_availability: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] <= 100
 shift_5_availability: x[0,5] + x[1,5] + x[2,5] + x[3,5] + x[4,5] <= 100
 shift_6_availability: x[0,6] + x[1,6] + x[2,6] + x[3,6] + x[4,6] <= 100
 shift_7_availability: x[0,7] + x[1,7] + x[2,7] + x[3,7] + x[4,7] <= 100
 overlap_task_0_shift_0_1: x[0,0] + x[0,1] <= 1e+06
 overlap_task_1_shift_0_1: x[1,0] + x[1,1] <= 1e+06
 overlap_task_2_shift_0_1: x[2,0] + x[2,1] <= 1e+06
 overlap_task_3_shift_0_1: x[3,0] + x[3,1] <= 1e+06
 overlap_task_4_shift_0_1: x[4,0] + x[4,1] <= 1e+06
 overlap_task_0_shift_1_2: x[0,1] + x[0,2] <= 1e+06
 overlap_task_1_shift_1_2: x[1,1] + x[1,2] <= 1e+06
 overlap_task_2_shift_1_2: x[2,1] + x[2,2] <= 1e+06
 overlap_task_3_shift_1_2: x[3,1] + x[3,2] <= 1e+06
 overlap_task_4_shift_1_2: x[4,1] + x[4,2] <= 1e+06
 overlap_task_0_shift_2_3: x[0,2] + x[0,3] <= 1e+06
 overlap_task_1_shift_2_3: x[1,2] + x[1,3] <= 1e+06
 overlap_task_2_shift_2_3: x[2,2] + x[2,3] <= 1e+06
 overlap_task_3_shift_2_3: x[3,2] + x[3,3] <= 1e+06
 overlap_task_4_shift_2_3: x[4,2] + x[4,3] <= 1e+06
 overlap_task_0_shift_3_4: x[0,3] + x[0,4] <= 1e+06
 overlap_task_1_shift_3_4: x[1,3] + x[1,4] <= 1e+06
 overlap_task_2_shift_3_4: x[2,3] + x[2,4] <= 1e+06
 overlap_task_3_shift_3_4: x[3,3] + x[3,4] <= 1e+06
 overlap_task_4_shift_3_4: x[4,3] + x[4,4] <= 1e+06
 overlap_task_0_shift_4_5: x[0,4] + x[0,5] <= 1e+06
 overlap_task_1_shift_4_5: x[1,4] + x[1,5] <= 1e+06
 overlap_task_2_shift_4_5: x[2,4] + x[2,5] <= 1e+06
 overlap_task_3_shift_4_5: x[3,4] + x[3,5] <= 1e+06
 overlap_task_4_shift_4_5: x[4,4] + x[4,5] <= 1e+06
 overlap_task_0_shift_5_6: x[0,5] + x[0,6] <= 1e+06
 overlap_task_1_shift_5_6: x[1,5] + x[1,6] <= 1e+06
 overlap_task_2_shift_5_6: x[2,5] + x[2,6] <= 1e+06
 overlap_task_3_shift_5_6: x[3,5] + x[3,6] <= 1e+06
 overlap_task_4_shift_5_6: x[4,5] + x[4,6] <= 1e+06
 overlap_task_0_shift_6_7: x[0,6] + x[0,7] <= 1e+06
 overlap_task_1_shift_6_7: x[1,6] + x[1,7] <= 1e+06
 overlap_task_2_shift_6_7: x[2,6] + x[2,7] <= 1e+06
 overlap_task_3_shift_6_7: x[3,6] + x[3,7] <= 1e+06
 overlap_task_4_shift_6_7: x[4,6] + x[4,7] <= 1e+06
Bounds
End
