\ Model Assignment Problem with Minimum Course Load and Prerequisites
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.4594248797 x[0,0] + 0.1248261629 x[0,1] + 0.922295372 x[0,2]
   + 0.0788001981 x[0,3] + 0.2931782814 x[1,0] + 0.6286397995 x[1,1]
   + 0.8854517479 x[1,2] + 0.3616350262 x[1,3] + 0.1922885903 x[2,0]
   + 0.0695551488 x[2,1] + 0.6612633186 x[2,2] + 0.7730683408 x[2,3]
   + 0.9852215207 x[3,0] + 0.855317721 x[3,1] + 0.8664836668 x[3,2]
   + 0.380126225 x[3,3] + 0.4534103024 x[4,0] + 0.8341104266 x[4,1]
   + 0.1626540972 x[4,2] + 0.3552707002 x[4,3]
Subject To
 student_0_min_course_load: x[0,0] + x[0,1] + x[0,2] + x[0,3] >= 2
 student_1_min_course_load: x[1,0] + x[1,1] + x[1,2] + x[1,3] >= 2
 student_2_min_course_load: x[2,0] + x[2,1] + x[2,2] + x[2,3] >= 2
 student_3_min_course_load: x[3,0] + x[3,1] + x[3,2] + x[3,3] >= 2
 student_4_min_course_load: x[4,0] + x[4,1] + x[4,2] + x[4,3] >= 2
 course_0_capacity: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 3
 course_1_capacity: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 1
 course_2_capacity: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 5
 course_3_capacity: x[0,3] + x[1,3] + x[2,3] + x[3,3] + x[4,3] <= 2
 student_0_course_0_prerequisite: x[0,0] <= 0
 student_0_course_1_prerequisite: x[0,1] <= 0
 student_0_course_2_prerequisite: x[0,2] <= 1
 student_0_course_3_prerequisite: x[0,3] <= 1
 student_1_course_0_prerequisite: x[1,0] <= 1
 student_1_course_1_prerequisite: x[1,1] <= 0
 student_1_course_2_prerequisite: x[1,2] <= 1
 student_1_course_3_prerequisite: x[1,3] <= 0
 student_2_course_0_prerequisite: x[2,0] <= 0
 student_2_course_1_prerequisite: x[2,1] <= 0
 student_2_course_2_prerequisite: x[2,2] <= 1
 student_2_course_3_prerequisite: x[2,3] <= 1
 student_3_course_0_prerequisite: x[3,0] <= 1
 student_3_course_1_prerequisite: x[3,1] <= 0
 student_3_course_2_prerequisite: x[3,2] <= 0
 student_3_course_3_prerequisite: x[3,3] <= 1
 student_4_course_0_prerequisite: x[4,0] <= 0
 student_4_course_1_prerequisite: x[4,1] <= 1
 student_4_course_2_prerequisite: x[4,2] <= 1
 student_4_course_3_prerequisite: x[4,3] <= 1
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[0,3] x[1,0] x[1,1] x[1,2] x[1,3] x[2,0] x[2,1]
 x[2,2] x[2,3] x[3,0] x[3,1] x[3,2] x[3,3] x[4,0] x[4,1] x[4,2] x[4,3]
End
