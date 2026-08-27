\ Model Assignment Problem with Minimum Course Load
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.534139877 x[0,0] + 0.2448109252 x[0,1] + 0.4622602119 x[0,2]
   + 0.2699478268 x[0,3] + 0.9253911795999999 x[1,0]
   + 0.6881619584000001 x[1,1] + 0.219615174 x[1,2] + 0.3242827344 x[1,3]
   + 0.7683135776 x[2,0] + 0.0559379475 x[2,1] + 0.821802594 x[2,2]
   + 0.8050458008 x[2,3] + 0.401164787 x[3,0] + 0.0661886052 x[3,1]
   + 0.9131361242 x[3,2] + 0.5671800514999999 x[3,3] + 0.7179142707 x[4,0]
   + 0.2126265441 x[4,1] + 0.4992312595 x[4,2] + 0.8846831539 x[4,3]
Subject To
 student_0_min_courses: x[0,0] + x[0,1] + x[0,2] + x[0,3] >= 2
 student_1_min_courses: x[1,0] + x[1,1] + x[1,2] + x[1,3] >= 3
 student_2_min_courses: x[2,0] + x[2,1] + x[2,2] + x[2,3] >= 4
 student_3_min_courses: x[3,0] + x[3,1] + x[3,2] + x[3,3] >= 4
 student_4_min_courses: x[4,0] + x[4,1] + x[4,2] + x[4,3] >= 3
 course_0_capacity: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 4
 course_1_capacity: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 2
 course_2_capacity: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 3
 course_3_capacity: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] <= 2
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[1,0] x[1,1] x[1,2] x[1,3] x[2,0] x[2,1]
 x[2,2] x[2,3] x[3,0] x[3,1] x[3,2] x[3,3] x[4,0] x[4,1] x[4,2] x[4,3]
End
