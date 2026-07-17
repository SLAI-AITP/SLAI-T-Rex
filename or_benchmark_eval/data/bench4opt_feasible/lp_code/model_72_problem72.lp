\ Model Assignment Problem with Time Slots
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  2.1932075916 x[0,0] + 9.975376065000001 x[0,1] + 5.0952629368 x[0,2]
   + 0.9090941217 x[1,0] + 0.4711637542 x[1,1] + 1.0964913035 x[1,2]
   + 6.274460417 x[2,0] + 7.9207936436 x[2,1] + 4.221599668 x[2,2]
   + 0.6352770615 x[3,0] + 3.8161928651 x[3,1] + 9.9612138024 x[3,2]
   + 5.291143451 x[4,0] + 9.7107837761 x[4,1] + 8.6077970223 x[4,2]
Subject To
 student_0_timeslot_0: x[0,0] + x[0,1] + x[0,2] <= 1
 student_0_timeslot_1: x[0,0] + x[0,1] <= 1
 student_0_timeslot_2: x[0,2] <= 1
 student_1_timeslot_0: x[1,0] + x[1,1] + x[1,2] <= 1
 student_1_timeslot_1: x[1,0] + x[1,1] <= 1
 student_1_timeslot_2: x[1,2] <= 1
 student_2_timeslot_0: x[2,0] + x[2,1] + x[2,2] <= 1
 student_2_timeslot_1: x[2,0] + x[2,1] <= 1
 student_2_timeslot_2: x[2,2] <= 1
 student_3_timeslot_0: x[3,0] + x[3,1] + x[3,2] <= 1
 student_3_timeslot_1: x[3,0] + x[3,1] <= 1
 student_3_timeslot_2: x[3,2] <= 1
 student_4_timeslot_0: x[4,0] + x[4,1] + x[4,2] <= 1
 student_4_timeslot_1: x[4,0] + x[4,1] <= 1
 student_4_timeslot_2: x[4,2] <= 1
 course_0_capacity: x[0,0] + x[1,0] + x[2,0] + x[3,0] + x[4,0] <= 1
 course_1_capacity: x[0,1] + x[1,1] + x[2,1] + x[3,1] + x[4,1] <= 1
 course_2_capacity: x[0,2] + x[1,2] + x[2,2] + x[3,2] + x[4,2] <= 5
Bounds
Binaries
 x[0,0] x[0,1] x[0,2] x[1,0] x[1,1] x[1,2] x[2,0] x[2,1] x[2,2] x[3,0]
 x[3,1] x[3,2] x[4,0] x[4,1] x[4,2]
End
