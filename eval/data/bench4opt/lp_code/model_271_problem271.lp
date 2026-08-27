\ Model Production Planning Problem with Time-of-Use Pricing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  24.17464133074372 power_output[0,0] + 20.30143545100915 power_output[0,1]
   + 96.24682490576093 power_output[0,2]
   + 60.36631871957674 power_output[0,3]
   + 19.21377635443184 power_output[0,4]
   + 58.82758424824947 power_output[0,5]
   + 59.44290912645788 power_output[0,6]
   + 34.39468907371248 power_output[0,7]
   + 94.89500368678445 power_output[0,8]
   + 82.36025546114216 power_output[0,9]
   + 61.66746483768713 power_output[0,10]
   + 25.41474779257073 power_output[0,11]
   + 38.0239803730969 power_output[0,12]
   + 90.30187942450121 power_output[0,13]
   + 58.65193918293929 power_output[0,14]
   + 14.56535653551414 power_output[1,0]
   + 12.60934175823779 power_output[1,1]
   + 91.29631668921287 power_output[1,2]
   + 14.54104095349776 power_output[1,3]
   + 44.52396754057696 power_output[1,4]
   + 17.75076224194405 power_output[1,5]
   + 45.2725084581112 power_output[1,6]
   + 73.42098343112261 power_output[1,7]
   + 90.21483051291152 power_output[1,8]
   + 66.65236208778393 power_output[1,9]
   + 72.11124072181354 power_output[1,10]
   + 50.12526389560387 power_output[1,11]
   + 42.00618817001327 power_output[1,12]
   + 39.90988860703839 power_output[1,13]
   + 27.83324244738281 power_output[1,14]
   + 78.25622916503491 power_output[2,0]
   + 82.8369840831581 power_output[2,1]
   + 17.06910466873176 power_output[2,2]
   + 93.05055922282385 power_output[2,3]
   + 78.79854553872781 power_output[2,4]
   + 61.10103720323424 power_output[2,5]
   + 18.36245568958365 power_output[2,6]
   + 82.21154483237872 power_output[2,7]
   + 72.87394681680823 power_output[2,8]
   + 18.75366971643697 power_output[2,9]
   + 37.9352516055332 power_output[2,10]
   + 67.99767489671282 power_output[2,11]
   + 78.98441457663246 power_output[2,12]
   + 52.34912872949599 power_output[2,13]
   + 95.27034431801364 power_output[2,14]
   + 22.93495154007274 power_output[3,0]
   + 90.61307754758587 power_output[3,1]
   + 33.22474891064721 power_output[3,2]
   + 42.7104410923043 power_output[3,3]
   + 26.50177644912388 power_output[3,4]
   + 92.55379795928489 power_output[3,5]
   + 39.72951606653692 power_output[3,6]
   + 87.47124336225055 power_output[3,7]
   + 39.22849797305074 power_output[3,8]
   + 53.87560593131014 power_output[3,9]
   + 15.8828678934283 power_output[3,10]
   + 20.15983515701945 power_output[3,11]
   + 53.07868772381359 power_output[3,12]
   + 86.11896362432941 power_output[3,13]
   + 35.4031632029618 power_output[3,14]
   + 41.37538980245307 power_output[4,0]
   + 61.72104322512045 power_output[4,1]
   + 55.71546993601246 power_output[4,2]
   + 46.51173840150058 power_output[4,3]
   + 74.28457775494606 power_output[4,4]
   + 19.62831653154101 power_output[4,5]
   + 60.05348740853836 power_output[4,6]
   + 16.38315909593392 power_output[4,7]
   + 37.19063138922378 power_output[4,8]
   + 51.4877367472163 power_output[4,9]
   + 63.14823610491879 power_output[4,10]
   + 31.47177626102221 power_output[4,11]
   + 31.93219549443879 power_output[4,12]
   + 24.31182606257688 power_output[4,13]
   + 53.41371904718489 power_output[4,14]
   + 15.94539411860909 power_output[5,0]
   + 28.7414331231446 power_output[5,1]
   + 72.46538745751468 power_output[5,2]
   + 48.80336478790832 power_output[5,3]
   + 98.21847035871322 power_output[5,4]
   + 61.04900748177923 power_output[5,5]
   + 40.79807511722608 power_output[5,6]
   + 72.47198411266423 power_output[5,7]
   + 25.57769104566501 power_output[5,8] + 70.26446764924 power_output[5,9]
   + 76.54303404523203 power_output[5,10]
   + 74.07179394123662 power_output[5,11]
   + 85.30733744648641 power_output[5,12]
   + 20.28905744325132 power_output[5,13]
   + 63.64679442094086 power_output[5,14]
Subject To
 demand_0: power_output[0,0] + power_output[1,0] + power_output[2,0]
   + power_output[3,0] + power_output[4,0] + power_output[5,0]
   >= 9880.67141418394
 demand_1: power_output[0,1] + power_output[1,1] + power_output[2,1]
   + power_output[3,1] + power_output[4,1] + power_output[5,1]
   >= 6464.808637874914
 demand_2: power_output[0,2] + power_output[1,2] + power_output[2,2]
   + power_output[3,2] + power_output[4,2] + power_output[5,2]
   >= 11678.24348541058
 demand_3: power_output[0,3] + power_output[1,3] + power_output[2,3]
   + power_output[3,3] + power_output[4,3] + power_output[5,3]
   >= 9620.535853045074
 demand_4: power_output[0,4] + power_output[1,4] + power_output[2,4]
   + power_output[3,4] + power_output[4,4] + power_output[5,4]
   >= 11086.90545740809
 demand_5: power_output[0,5] + power_output[1,5] + power_output[2,5]
   + power_output[3,5] + power_output[4,5] + power_output[5,5]
   >= 13973.88772391654
 demand_6: power_output[0,6] + power_output[1,6] + power_output[2,6]
   + power_output[3,6] + power_output[4,6] + power_output[5,6]
   >= 15282.63022907824
 demand_7: power_output[0,7] + power_output[1,7] + power_output[2,7]
   + power_output[3,7] + power_output[4,7] + power_output[5,7]
   >= 13946.08122535162
 demand_8: power_output[0,8] + power_output[1,8] + power_output[2,8]
   + power_output[3,8] + power_output[4,8] + power_output[5,8]
   >= 17297.98593136419
 demand_9: power_output[0,9] + power_output[1,9] + power_output[2,9]
   + power_output[3,9] + power_output[4,9] + power_output[5,9]
   >= 9911.123643923007
 demand_10: power_output[0,10] + power_output[1,10] + power_output[2,10]
   + power_output[3,10] + power_output[4,10] + power_output[5,10]
   >= 14674.8367317205
 demand_11: power_output[0,11] + power_output[1,11] + power_output[2,11]
   + power_output[3,11] + power_output[4,11] + power_output[5,11]
   >= 5922.233323439918
 demand_12: power_output[0,12] + power_output[1,12] + power_output[2,12]
   + power_output[3,12] + power_output[4,12] + power_output[5,12]
   >= 8264.17660158503
 demand_13: power_output[0,13] + power_output[1,13] + power_output[2,13]
   + power_output[3,13] + power_output[4,13] + power_output[5,13]
   >= 7083.751062370262
 demand_14: power_output[0,14] + power_output[1,14] + power_output[2,14]
   + power_output[3,14] + power_output[4,14] + power_output[5,14]
   >= 13711.51837370426
 capacity_0_0: power_output[0,0] <= 2804.999610281816
 capacity_0_1: power_output[0,1] <= 2804.999610281816
 capacity_0_2: power_output[0,2] <= 2804.999610281816
 capacity_0_3: power_output[0,3] <= 2804.999610281816
 capacity_0_4: power_output[0,4] <= 2804.999610281816
 capacity_0_5: power_output[0,5] <= 2804.999610281816
 capacity_0_6: power_output[0,6] <= 2804.999610281816
 capacity_0_7: power_output[0,7] <= 2804.999610281816
 capacity_0_8: power_output[0,8] <= 2804.999610281816
 capacity_0_9: power_output[0,9] <= 2804.999610281816
 capacity_0_10: power_output[0,10] <= 2804.999610281816
 capacity_0_11: power_output[0,11] <= 2804.999610281816
 capacity_0_12: power_output[0,12] <= 2804.999610281816
 capacity_0_13: power_output[0,13] <= 2804.999610281816
 capacity_0_14: power_output[0,14] <= 2804.999610281816
 capacity_1_0: power_output[1,0] <= 3579.106394125863
 capacity_1_1: power_output[1,1] <= 3579.106394125863
 capacity_1_2: power_output[1,2] <= 3579.106394125863
 capacity_1_3: power_output[1,3] <= 3579.106394125863
 capacity_1_4: power_output[1,4] <= 3579.106394125863
 capacity_1_5: power_output[1,5] <= 3579.106394125863
 capacity_1_6: power_output[1,6] <= 3579.106394125863
 capacity_1_7: power_output[1,7] <= 3579.106394125863
 capacity_1_8: power_output[1,8] <= 3579.106394125863
 capacity_1_9: power_output[1,9] <= 3579.106394125863
 capacity_1_10: power_output[1,10] <= 3579.106394125863
 capacity_1_11: power_output[1,11] <= 3579.106394125863
 capacity_1_12: power_output[1,12] <= 3579.106394125863
 capacity_1_13: power_output[1,13] <= 3579.106394125863
 capacity_1_14: power_output[1,14] <= 3579.106394125863
 capacity_2_0: power_output[2,0] <= 5790.482251885146
 capacity_2_1: power_output[2,1] <= 5790.482251885146
 capacity_2_2: power_output[2,2] <= 5790.482251885146
 capacity_2_3: power_output[2,3] <= 5790.482251885146
 capacity_2_4: power_output[2,4] <= 5790.482251885146
 capacity_2_5: power_output[2,5] <= 5790.482251885146
 capacity_2_6: power_output[2,6] <= 5790.482251885146
 capacity_2_7: power_output[2,7] <= 5790.482251885146
 capacity_2_8: power_output[2,8] <= 5790.482251885146
 capacity_2_9: power_output[2,9] <= 5790.482251885146
 capacity_2_10: power_output[2,10] <= 5790.482251885146
 capacity_2_11: power_output[2,11] <= 5790.482251885146
 capacity_2_12: power_output[2,12] <= 5790.482251885146
 capacity_2_13: power_output[2,13] <= 5790.482251885146
 capacity_2_14: power_output[2,14] <= 5790.482251885146
 capacity_3_0: power_output[3,0] <= 7955.083876625005
 capacity_3_1: power_output[3,1] <= 7955.083876625005
 capacity_3_2: power_output[3,2] <= 7955.083876625005
 capacity_3_3: power_output[3,3] <= 7955.083876625005
 capacity_3_4: power_output[3,4] <= 7955.083876625005
 capacity_3_5: power_output[3,5] <= 7955.083876625005
 capacity_3_6: power_output[3,6] <= 7955.083876625005
 capacity_3_7: power_output[3,7] <= 7955.083876625005
 capacity_3_8: power_output[3,8] <= 7955.083876625005
 capacity_3_9: power_output[3,9] <= 7955.083876625005
 capacity_3_10: power_output[3,10] <= 7955.083876625005
 capacity_3_11: power_output[3,11] <= 7955.083876625005
 capacity_3_12: power_output[3,12] <= 7955.083876625005
 capacity_3_13: power_output[3,13] <= 7955.083876625005
 capacity_3_14: power_output[3,14] <= 7955.083876625005
 capacity_4_0: power_output[4,0] <= 7040.116377613377
 capacity_4_1: power_output[4,1] <= 7040.116377613377
 capacity_4_2: power_output[4,2] <= 7040.116377613377
 capacity_4_3: power_output[4,3] <= 7040.116377613377
 capacity_4_4: power_output[4,4] <= 7040.116377613377
 capacity_4_5: power_output[4,5] <= 7040.116377613377
 capacity_4_6: power_output[4,6] <= 7040.116377613377
 capacity_4_7: power_output[4,7] <= 7040.116377613377
 capacity_4_8: power_output[4,8] <= 7040.116377613377
 capacity_4_9: power_output[4,9] <= 7040.116377613377
 capacity_4_10: power_output[4,10] <= 7040.116377613377
 capacity_4_11: power_output[4,11] <= 7040.116377613377
 capacity_4_12: power_output[4,12] <= 7040.116377613377
 capacity_4_13: power_output[4,13] <= 7040.116377613377
 capacity_4_14: power_output[4,14] <= 7040.116377613377
 capacity_5_0: power_output[5,0] <= 3453.887860058892
 capacity_5_1: power_output[5,1] <= 3453.887860058892
 capacity_5_2: power_output[5,2] <= 3453.887860058892
 capacity_5_3: power_output[5,3] <= 3453.887860058892
 capacity_5_4: power_output[5,4] <= 3453.887860058892
 capacity_5_5: power_output[5,5] <= 3453.887860058892
 capacity_5_6: power_output[5,6] <= 3453.887860058892
 capacity_5_7: power_output[5,7] <= 3453.887860058892
 capacity_5_8: power_output[5,8] <= 3453.887860058892
 capacity_5_9: power_output[5,9] <= 3453.887860058892
 capacity_5_10: power_output[5,10] <= 3453.887860058892
 capacity_5_11: power_output[5,11] <= 3453.887860058892
 capacity_5_12: power_output[5,12] <= 3453.887860058892
 capacity_5_13: power_output[5,13] <= 3453.887860058892
 capacity_5_14: power_output[5,14] <= 3453.887860058892
 emission_cap_0: 1.534478171234223 power_output[0,0]
   + 1.63869479894899 power_output[1,0]
   + 1.538447663029866 power_output[2,0]
   + 1.355674648162061 power_output[3,0]
   + 0.9606513985200675 power_output[4,0]
   + 0.6538561816585589 power_output[5,0] <= 3462.288665113789
 emission_cap_1: 1.534478171234223 power_output[0,1]
   + 1.63869479894899 power_output[1,1]
   + 1.538447663029866 power_output[2,1]
   + 1.355674648162061 power_output[3,1]
   + 0.9606513985200675 power_output[4,1]
   + 0.6538561816585589 power_output[5,1] <= 2788.639074434835
 emission_cap_2: 1.534478171234223 power_output[0,2]
   + 1.63869479894899 power_output[1,2]
   + 1.538447663029866 power_output[2,2]
   + 1.355674648162061 power_output[3,2]
   + 0.9606513985200675 power_output[4,2]
   + 0.6538561816585589 power_output[5,2] <= 1822.646727607154
 emission_cap_3: 1.534478171234223 power_output[0,3]
   + 1.63869479894899 power_output[1,3]
   + 1.538447663029866 power_output[2,3]
   + 1.355674648162061 power_output[3,3]
   + 0.9606513985200675 power_output[4,3]
   + 0.6538561816585589 power_output[5,3] <= 4509.701410430602
 emission_cap_4: 1.534478171234223 power_output[0,4]
   + 1.63869479894899 power_output[1,4]
   + 1.538447663029866 power_output[2,4]
   + 1.355674648162061 power_output[3,4]
   + 0.9606513985200675 power_output[4,4]
   + 0.6538561816585589 power_output[5,4] <= 2198.941337176454
 emission_cap_5: 1.534478171234223 power_output[0,5]
   + 1.63869479894899 power_output[1,5]
   + 1.538447663029866 power_output[2,5]
   + 1.355674648162061 power_output[3,5]
   + 0.9606513985200675 power_output[4,5]
   + 0.6538561816585589 power_output[5,5] <= 4177.386638915666
 emission_cap_6: 1.534478171234223 power_output[0,6]
   + 1.63869479894899 power_output[1,6]
   + 1.538447663029866 power_output[2,6]
   + 1.355674648162061 power_output[3,6]
   + 0.9606513985200675 power_output[4,6]
   + 0.6538561816585589 power_output[5,6] <= 3718.235710398857
 emission_cap_7: 1.534478171234223 power_output[0,7]
   + 1.63869479894899 power_output[1,7]
   + 1.538447663029866 power_output[2,7]
   + 1.355674648162061 power_output[3,7]
   + 0.9606513985200675 power_output[4,7]
   + 0.6538561816585589 power_output[5,7] <= 4674.006490210691
 emission_cap_8: 1.534478171234223 power_output[0,8]
   + 1.63869479894899 power_output[1,8]
   + 1.538447663029866 power_output[2,8]
   + 1.355674648162061 power_output[3,8]
   + 0.9606513985200675 power_output[4,8]
   + 0.6538561816585589 power_output[5,8] <= 4746.705932460041
 emission_cap_9: 1.534478171234223 power_output[0,9]
   + 1.63869479894899 power_output[1,9]
   + 1.538447663029866 power_output[2,9]
   + 1.355674648162061 power_output[3,9]
   + 0.9606513985200675 power_output[4,9]
   + 0.6538561816585589 power_output[5,9] <= 3871.164887624392
 emission_cap_10: 1.534478171234223 power_output[0,10]
   + 1.63869479894899 power_output[1,10]
   + 1.538447663029866 power_output[2,10]
   + 1.355674648162061 power_output[3,10]
   + 0.9606513985200675 power_output[4,10]
   + 0.6538561816585589 power_output[5,10] <= 4660.471658237714
 emission_cap_11: 1.534478171234223 power_output[0,11]
   + 1.63869479894899 power_output[1,11]
   + 1.538447663029866 power_output[2,11]
   + 1.355674648162061 power_output[3,11]
   + 0.9606513985200675 power_output[4,11]
   + 0.6538561816585589 power_output[5,11] <= 4961.15702656768
 emission_cap_12: 1.534478171234223 power_output[0,12]
   + 1.63869479894899 power_output[1,12]
   + 1.538447663029866 power_output[2,12]
   + 1.355674648162061 power_output[3,12]
   + 0.9606513985200675 power_output[4,12]
   + 0.6538561816585589 power_output[5,12] <= 3330.907911304213
 emission_cap_13: 1.534478171234223 power_output[0,13]
   + 1.63869479894899 power_output[1,13]
   + 1.538447663029866 power_output[2,13]
   + 1.355674648162061 power_output[3,13]
   + 0.9606513985200675 power_output[4,13]
   + 0.6538561816585589 power_output[5,13] <= 3445.963024688412
 emission_cap_14: 1.534478171234223 power_output[0,14]
   + 1.63869479894899 power_output[1,14]
   + 1.538447663029866 power_output[2,14]
   + 1.355674648162061 power_output[3,14]
   + 0.9606513985200675 power_output[4,14]
   + 0.6538561816585589 power_output[5,14] <= 3041.695517426652
Bounds
End
