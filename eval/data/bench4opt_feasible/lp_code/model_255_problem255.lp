\ Model Assignment Problem with Minimum Course Load and Prerequisites
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  48.48820668119532 x[0,0] + 79.66055979163441 x[0,1]
   + 21.63921678760715 x[0,2] + 96.65322319713601 x[0,3]
   + 33.67406228486922 x[1,0] + 15.55775868452924 x[1,1]
   + 99.26582523445747 x[1,2] + 16.21105154479048 x[1,3]
   + 88.34515052466303 x[2,0] + 86.11460906846524 x[2,1]
   + 100.8443511530426 x[2,2] + 42.53555160587909 x[2,3]
   + 63.149418551384 x[3,0] + 88.77403756832693 x[3,1]
   + 90.19683380115833 x[3,2] + 3.41953512434776 x[3,3]
   + 67.60929607963504 x[4,0] + 76.29208956928926 x[4,1]
   + 99.30822484682218 x[4,2] + 84.13396841988036 x[4,3]
Subject To
 student_0_min_course_load: x[0,0] + x[0,1] + x[0,2] + x[0,3] >= 1
 student_1_min_course_load: x[1,0] + x[1,1] + x[1,2] + x[1,3] >= 1
 student_2_min_course_load: x[2,0] + x[2,1] + x[2,2] + x[2,3] >= 1
 student_3_min_course_load: x[3,0] + x[3,1] + x[3,2] + x[3,3] >= 1
 student_4_min_course_load: x[4,0] + x[4,1] + x[4,2] + x[4,3] >= 1
 course_0_capacity: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 100
 course_1_capacity: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 100
 course_2_capacity: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 100
 course_3_capacity: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] <= 100
 student_0_course_0_prerequisite: x[0,0] <= 1
 student_0_course_1_prerequisite: x[0,1] <= 1
 student_0_course_2_prerequisite: x[0,2] <= 1
 student_0_course_3_prerequisite: x[0,3] <= 1
 student_1_course_0_prerequisite: x[1,0] <= 1
 student_1_course_1_prerequisite: x[1,1] <= 1
 student_1_course_2_prerequisite: x[1,2] <= 1
 student_1_course_3_prerequisite: x[1,3] <= 1
 student_2_course_0_prerequisite: x[2,0] <= 1
 student_2_course_1_prerequisite: x[2,1] <= 1
 student_2_course_2_prerequisite: x[2,2] <= 1
 student_2_course_3_prerequisite: x[2,3] <= 1
 student_3_course_0_prerequisite: x[3,0] <= 1
 student_3_course_1_prerequisite: x[3,1] <= 1
 student_3_course_2_prerequisite: x[3,2] <= 1
 student_3_course_3_prerequisite: x[3,3] <= 1
 student_4_course_0_prerequisite: x[4,0] <= 1
 student_4_course_1_prerequisite: x[4,1] <= 1
 student_4_course_2_prerequisite: x[4,2] <= 1
 student_4_course_3_prerequisite: x[4,3] <= 1
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[1,0] x[1,1] x[1,2] x[1,3] x[2,0] x[2,1]
 x[2,2] x[2,3] x[3,0] x[3,1] x[3,2] x[3,3] x[4,0] x[4,1] x[4,2] x[4,3]
End
