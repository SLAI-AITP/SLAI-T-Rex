\ Model Healthcare_Staff_Assignment
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  27.2231364833 assign[0,0] + 25.4161232617 assign[0,1]
   + 56.5754452822 assign[0,2] + 27.0114192438 assign[1,0]
   + 58.8740130321 assign[1,1] + 89.8844654766 assign[1,2]
   + 40.5406500089 assign[2,0] + 22.7127551566 assign[2,1]
   + 99.75622304300001 assign[2,2] + 51.443103074 assign[3,0]
   + 10.0000318052 assign[3,1] + 5.664521167 assign[3,2]
   + 11.8552639047 assign[4,0] + 63.1171581286 assign[4,1]
   + 79.41585707190001 assign[4,2]
Subject To
 Coverage_shift_0: assign[0,0] + assign[1,0] + assign[2,0] + assign[3,0]
   + assign[4,0] >= 4
 Coverage_shift_1: assign[0,1] + assign[1,1] + assign[2,1] + assign[3,1]
   + assign[4,1] >= 5
 Coverage_shift_2: assign[0,2] + assign[1,2] + assign[2,2] + assign[3,2]
   + assign[4,2] >= 1
 Assignment_staff_0: assign[0,0] + assign[0,1] + assign[0,2] <= 1
 Assignment_staff_1: assign[1,0] + assign[1,1] + assign[1,2] <= 1
 Assignment_staff_2: assign[2,0] + assign[2,1] + assign[2,2] <= 1
 Assignment_staff_3: assign[3,0] + assign[3,1] + assign[3,2] <= 1
 Assignment_staff_4: assign[4,0] + assign[4,1] + assign[4,2] <= 1
Bounds
Binaries
 assign[0,0] assign[0,1] assign[0,2] assign[1,0] assign[1,1] assign[1,2]
 assign[2,0] assign[2,1] assign[2,2] assign[3,0] assign[3,1] assign[3,2]
 assign[4,0] assign[4,1] assign[4,2]
End
