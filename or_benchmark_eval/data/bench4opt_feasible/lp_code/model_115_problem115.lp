\ Model Staff Scheduling Problem with Skill Level Constraints
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  96.75553433427433 x[0,0] + 99.70018413017691 x[0,1]
   + 27.29158050359237 x[0,2] + 43.32129691239764 x[0,3]
   + 79.62246353639325 x[0,4] + 44.04871705301705 x[1,0]
   + 17.31799770040919 x[1,1] + 76.92297601178612 x[1,2]
   + 73.18665674809901 x[1,3] + 48.55154785083503 x[1,4]
   + 83.20956920607257 x[2,0] + 52.33523929398937 x[2,1]
   + 27.43100262357183 x[2,2] + 30.58845153817024 x[2,3]
   + 19.45926378310282 x[2,4] + 28.20371694200926 x[3,0]
   + 55.8881238246885 x[3,1] + 33.97495786186804 x[3,2]
   + 55.30137473455778 x[3,3] + 71.61607335805689 x[3,4]
   + 91.84835576301604 x[4,0] + 83.6469668996767 x[4,1]
   + 63.58977372635822 x[4,2] + 67.22760245531923 x[4,3]
   + 93.94775092308427 x[4,4] + 51.00461431727267 x[5,0]
   + 28.54078557711752 x[5,1] + 17.39960902705828 x[5,2]
   + 18.29070990118332 x[5,3] + 31.85947837031135 x[5,4]
   + 49.22339147840076 x[6,0] + 92.11421367464463 x[6,1]
   + 84.43201029956558 x[6,2] + 25.3331018732534 x[6,3]
   + 69.95619991118237 x[6,4] + 55.44015865133669 x[7,0]
   + 40.66153535468779 x[7,1] + 58.31331781670684 x[7,2]
   + 80.71751826944397 x[7,3] + 80.30009654500986 x[7,4]
   + 49.50035655809378 x[8,0] + 18.83982260362451 x[8,1]
   + 12.67460394024209 x[8,2] + 89.84418789422963 x[8,3]
   + 29.4425368688319 x[8,4] + 79.1014439122911 x[9,0]
   + 74.12987099648241 x[9,1] + 60.55533447272657 x[9,2]
   + 77.52292750312452 x[9,3] + 71.67932539372121 x[9,4]
Subject To
 task_0_shift_0: x[0,0] >= 1
 task_0_shift_1: x[0,1] >= 1
 task_0_shift_2: x[0,2] >= 1
 task_0_shift_3: x[0,3] >= 1
 task_0_shift_4: x[0,4] >= 1
 task_1_shift_0: x[1,0] >= 1
 task_1_shift_1: x[1,1] >= 1
 task_1_shift_2: x[1,2] >= 1
 task_1_shift_3: x[1,3] >= 1
 task_1_shift_4: x[1,4] >= 1
 task_2_shift_0: x[2,0] >= 1
 task_2_shift_1: x[2,1] >= 1
 task_2_shift_2: x[2,2] >= 1
 task_2_shift_3: x[2,3] >= 1
 task_2_shift_4: x[2,4] >= 1
 task_3_shift_0: x[3,0] >= 1
 task_3_shift_1: x[3,1] >= 1
 task_3_shift_2: x[3,2] >= 1
 task_3_shift_3: x[3,3] >= 1
 task_3_shift_4: x[3,4] >= 1
 task_4_shift_0: x[4,0] >= 1
 task_4_shift_1: x[4,1] >= 1
 task_4_shift_2: x[4,2] >= 1
 task_4_shift_3: x[4,3] >= 1
 task_4_shift_4: x[4,4] >= 1
 task_5_shift_0: x[5,0] >= 1
 task_5_shift_1: x[5,1] >= 1
 task_5_shift_2: x[5,2] >= 1
 task_5_shift_3: x[5,3] >= 1
 task_5_shift_4: x[5,4] >= 1
 task_6_shift_0: x[6,0] >= 1
 task_6_shift_1: x[6,1] >= 1
 task_6_shift_2: x[6,2] >= 1
 task_6_shift_3: x[6,3] >= 1
 task_6_shift_4: x[6,4] >= 1
 task_7_shift_0: x[7,0] >= 1
 task_7_shift_1: x[7,1] >= 1
 task_7_shift_2: x[7,2] >= 1
 task_7_shift_3: x[7,3] >= 1
 task_7_shift_4: x[7,4] >= 1
 task_8_shift_0: x[8,0] >= 1
 task_8_shift_1: x[8,1] >= 1
 task_8_shift_2: x[8,2] >= 1
 task_8_shift_3: x[8,3] >= 1
 task_8_shift_4: x[8,4] >= 1
 task_9_shift_0: x[9,0] >= 1
 task_9_shift_1: x[9,1] >= 1
 task_9_shift_2: x[9,2] >= 1
 task_9_shift_3: x[9,3] >= 1
 task_9_shift_4: x[9,4] >= 1
 shift_0_availability: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] + x[5,0]
   + x[6,0] + x[7,0] + x[8,0] + x[9,0] <= 100
 shift_1_availability: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] + x[5,1]
   + x[6,1] + x[7,1] + x[8,1] + x[9,1] <= 100
 shift_2_availability: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] + x[5,2]
   + x[6,2] + x[7,2] + x[8,2] + x[9,2] <= 100
 shift_3_availability: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] + x[5,3]
   + x[6,3] + x[7,3] + x[8,3] + x[9,3] <= 100
 shift_4_availability: x[0,4] + x[1,4] + x[2,4] + x[3,4] + x[4,4] + x[5,4]
   + x[6,4] + x[7,4] + x[8,4] + x[9,4] <= 100
 task_0_skill_0: 200 x[0,0] >= 1
 task_0_skill_1: 200 x[0,1] >= 1
 task_0_skill_2: 200 x[0,2] >= 1
 task_0_skill_3: 200 x[0,3] >= 1
 task_0_skill_4: 200 x[0,4] >= 1
 task_1_skill_0: 200 x[1,0] >= 1
 task_1_skill_1: 200 x[1,1] >= 1
 task_1_skill_2: 200 x[1,2] >= 1
 task_1_skill_3: 200 x[1,3] >= 1
 task_1_skill_4: 200 x[1,4] >= 1
 task_2_skill_0: 200 x[2,0] >= 1
 task_2_skill_1: 200 x[2,1] >= 1
 task_2_skill_2: 200 x[2,2] >= 1
 task_2_skill_3: 200 x[2,3] >= 1
 task_2_skill_4: 200 x[2,4] >= 1
 task_3_skill_0: 200 x[3,0] >= 1
 task_3_skill_1: 200 x[3,1] >= 1
 task_3_skill_2: 200 x[3,2] >= 1
 task_3_skill_3: 200 x[3,3] >= 1
 task_3_skill_4: 200 x[3,4] >= 1
 task_4_skill_0: 200 x[4,0] >= 1
 task_4_skill_1: 200 x[4,1] >= 1
 task_4_skill_2: 200 x[4,2] >= 1
 task_4_skill_3: 200 x[4,3] >= 1
 task_4_skill_4: 200 x[4,4] >= 1
 task_5_skill_0: 200 x[5,0] >= 1
 task_5_skill_1: 200 x[5,1] >= 1
 task_5_skill_2: 200 x[5,2] >= 1
 task_5_skill_3: 200 x[5,3] >= 1
 task_5_skill_4: 200 x[5,4] >= 1
 task_6_skill_0: 200 x[6,0] >= 1
 task_6_skill_1: 200 x[6,1] >= 1
 task_6_skill_2: 200 x[6,2] >= 1
 task_6_skill_3: 200 x[6,3] >= 1
 task_6_skill_4: 200 x[6,4] >= 1
 task_7_skill_0: 200 x[7,0] >= 1
 task_7_skill_1: 200 x[7,1] >= 1
 task_7_skill_2: 200 x[7,2] >= 1
 task_7_skill_3: 200 x[7,3] >= 1
 task_7_skill_4: 200 x[7,4] >= 1
 task_8_skill_0: 200 x[8,0] >= 1
 task_8_skill_1: 200 x[8,1] >= 1
 task_8_skill_2: 200 x[8,2] >= 1
 task_8_skill_3: 200 x[8,3] >= 1
 task_8_skill_4: 200 x[8,4] >= 1
 task_9_skill_0: 200 x[9,0] >= 1
 task_9_skill_1: 200 x[9,1] >= 1
 task_9_skill_2: 200 x[9,2] >= 1
 task_9_skill_3: 200 x[9,3] >= 1
 task_9_skill_4: 200 x[9,4] >= 1
Bounds
End
