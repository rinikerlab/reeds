import unittest
import os
from reeds.function_libs.optimization import eds_energy_offsets
from reeds.function_libs.file_management import file_management as fM

import numpy as np

in_BRD4_7ligs = os.path.dirname(__file__) + "/data/7ligs"
in_PNMT_9ligs = os.path.dirname(__file__)+"/data/PNMT_9ligs"

out_result_BRD4_7ligs = os.path.dirname(__file__) + "/data/out_test_result_BRD4_7ligs.out"
out_result_PNMT_9ligs = os.path.dirname(__file__)+"/data/out_test_result_PNMT_9ligs.out"


class test_Eoff_wrapper(unittest.TestCase):

    def test_eoff_BRD4_7ligs(self):
        #result:
    
        expected_eoffs = np.array([
            [     0.,          -3477.665911,   -25277.106431,   -25243.526011, -19647.956147,   -26222.50097884, -19669.102572  ],
            [     0.,          32566.474782,   -17624.648377,   -17547.77742,   -6028.870275,   -18561.00373584, -14804.752015  ],
            [     0.,          51202.172245,   -13424.54721083, -13568.346403,  -2871.574164,   -14460.48970784, -11677.249527  ],
            [     0.,          32233.320004,   -16236.235581,   -16295.123735,  -7370.974142,   -17131.32000884, -14112.996993  ],
            [     0.,          21706.320631,   -25574.64824,    -25600.24727501, -15670.607396, -26438.71212484, -21713.395748  ],
            [     0.,          34834.060143,   -13764.013669,   -13883.155494,    -9439.909438, -14716.40340984, -11703.401339  ],
            [     0.,          52727.8412,     -10757.898227,   -10902.08184635,  -304.210506,  -11752.70209342,  -9669.599035  ],
            [     0.,          30267.947423,    -9751.32691253,  -9783.295176,    -195.599075,  -10685.44874403,  -7564.997107  ],
            [     0.,          43977.68671,     -5067.23676023,  -5506.5152002,   -3330.662978, -5793.48818846,  -4650.414262  ],
            [     0.,          2637.58545279,    266.12185815,    -89.57325054,   47.13542781,  -319.03035415,    159.90802968],
            [     0.,          321.49197543,    248.16835992,    -84.9513929,    37.60692756,   -318.24696274,    148.16364835],
            [     0.,          306.30322957,    242.4176958,     -82.05107038,   37.6572784,    -306.344127,    155.59786761],
            [     0.,          289.80169858,    230.86488193,   -103.46218717,   42.77049576,   -305.00825801,    131.59990836],
            [     0.,          304.25679099,    238.70062531,   -104.09826592,   26.36253701,   -314.72557004,   140.3880654 ],
            [     0.,          278.38106171,    232.13447009,    -87.00095251,   34.18917823,   -305.22691863,    149.8921765 ]
        ])



        #params:
        T = 298
        s_values = list(map(float, "1.0 0.56234133 0.31622777 0.17782794 0.1 0.05623413 0.03162278 0.01778279 0.01 0.00562341 0.00316228 0.00177828 0.001 0.00056234 0.00031623".split()))
        s_val_num = len(s_values)
        states_num = 7
        init_Eoff = [0.0 for x in range(states_num)]
        
        print ('testing we get the same results with BRD4 data')

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_BRD4_7ligs, ene_trajs_prefix="energies_")

        sampling_stat = {"state_undersampling_potTresh":  [0,0,0,0,0,0,0],
                         "undersampling_occurence_sampling_tresh": 0.9}
        new_eoffs, all_eoffs = eds_energy_offsets.estimate_energy_offsets(ene_ana_trajs, initial_offsets = init_Eoff, 
                                                                          s_values = s_values, out_path = None,  sampling_stat=sampling_stat,
                                                                          temp = T, trim_beg = 0.0, undersampling_idx = 0, 
                                                                          plot_results = False, calc_clara = False
                                                                          )
        # do the comparison with the previous data
 
        equal_criteria = 0.1 # kJ/mol, results must not differ by more than this value

        if not np.all( np.abs(all_eoffs - expected_eoffs) < equal_criteria):
            raise Exception('BRD4 offsets calculated to be different')
        else:
            print ('BRD4 data gives the same results')


    def test_eoff_PNMT_9ligs(self):
        
        expected_eoffs = np.array([
         [   0.0,           12.72507142,   20.05976081,  -25.62036539,  -13.15887897, -336.83665691, -504.19747211,   21.93712241,    4.79907405],
         [   0.0,            7.79728192,   30.30650457,   -5.42730397,   -1.87408498, -280.16669749, -499.16648857,  -26.3761377,   -72.31883187],
         [   0.0,           21.94475404,    4.77255521,   18.30944948,   -3.16837365, -289.52657702, -492.68223004,   16.46356148,  -39.16086421],
         [   0.0,          -12.59810399,   15.47408475,  -13.12023784,  -16.06243093, -325.21752006, -507.99624893,    7.31812804,  -78.99522206],
         [   0.0,            2.75649107,    8.13690798,  -13.84585243,  -21.85057136, -282.93737402, -500.27318313,   -2.0855101,   -11.57737438],
         [   0.0,            4.08140669,    3.21576768,   -1.20486334,   -5.35329057, -319.73353025, -508.81298842,   16.98748738,  -22.21220394],
         [   0.0,           15.66378689,   39.98890657,   24.538459  ,   -6.55874685, -309.19254098, -479.36416672,   37.51263524,  -29.10856717],
         [   0.0,            1.80585478,   -5.44434504,    0.29940514,   -5.4597886,  -299.61390276, -499.75284645,   -0.79510748,  -19.52586253],
         [   0.0,           11.38899217,   24.54550266,   31.97101691,  -22.32300735, -364.03968497, -494.10471298,   19.43628647,  -39.72010141],
         [   0.0,           -9.01614423,    0.90079861,   11.10510425,    1.32407133, -297.40412347, -493.46939524,   11.87810236,  -18.80446089],
         [   0.0,            8.02965178,    9.25176196,   -0.82255875,  -44.56403826, -333.37867756, -495.16741143,    8.18953278,    5.58434011],
         [   0.0,            6.28473162,   35.74675219,    0.02260035,  -54.79282894, -413.66590508, -498.64293198,    9.61728156,  -30.02387337],
         [   0.0,           21.59430564,    0.27520316,   20.13021646,  -53.60906903, -403.1916287,  -454.38539656,    1.07450116, -114.30080107],
         [   0.0,            7.00657794,    6.97119376,   11.79268054,  -25.84198252, -349.10150685, -415.21568473,    1.06692688, -115.66752471],
         [   0.0,          -11.90459409,   -2.91144683,    1.80999571,  -43.4216908,  -354.35081104, -417.23130005,  -12.48364889, -128.09082999],
         [   0.0,            4.16931893,   -2.15332892,   16.59053623,  -36.37201814, -339.59978434, -373.03866938,   -6.42293304, -120.75926452],
         [   0.0,            4.91842155,   -1.31944729,   16.07279018,  -33.38562859, -322.02948509, -381.04213761,  -12.58182375, -120.75448705],
         [   0.0,           -7.82538448,   -9.77108036,   6.19027003 , -34.55431285, -334.88915121,  -371.094358  ,   -15.96248935, -121.25245287],
         [   0.0,           -5.91136477,   -8.3543212 ,    2.16528193,  -46.61249371, -330.56653092, -373.4918628,    -5.9382177 , -133.95855808],
         [   0.0,            5.63037432,    3.77337435,    9.26545444,  -33.28069624, -325.4252408,  -368.66113113,   -3.86795046, -122.41915811],
         [   0.0,          -11.06604601,  -11.98836604,   -4.5436458 ,  -41.26514536, -331.03427956, -363.98161834,  -13.21456143, -127.61468884]
        ])

        #params:
        T = 298
        s_values = list(map(float, "1  0.7  0.5  0.35  0.25  0.18  0.13  0.089  0.063  0.044  0.031  0.022  0.016  0.011  0.008  0.0057  0.004  0.0028  0.002  0.0014  0.001".split()))
        s_val_num = len(s_values)
        states_num = 9
        init_Eoff = [0.0 for x in range(states_num)]

        print ('testing we get the same results with BRD4 data')

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_PNMT_9ligs, ene_trajs_prefix="energies_")

        sampling_stat = {"state_undersampling_potTresh":  [0,0,0,0,0,0,0,0,0],
                         "undersampling_occurence_sampling_tresh": 0.9}
        new_eoffs, all_eoffs = eds_energy_offsets.estimate_energy_offsets(ene_ana_trajs, initial_offsets = init_Eoff,
                                                                          s_values = s_values, out_path = None, sampling_stat=sampling_stat,
                                                                          temp = T, trim_beg = 0.0, undersampling_idx = 0,
                                                                          plot_results = False, calc_clara = False
                                                                          )
        # do the comparison with the previous data
        equal_criteria = 0.1 # kJ/mol, results must not differ by more than this value

        if not np.all( np.abs(all_eoffs - expected_eoffs) < equal_criteria):
            raise Exception('PNMT offsets calculated to be different')
        else:
            print ('PNMT data gives the same results')
 
    
        

    def test_undersampling_detection(self):
        #params:
        T = 298
        s_values = list(map(float, "1  0.7  0.5  0.35  0.25  0.18  0.13  0.089  0.063  0.044  0.031  0.022  0.016  0.011  0.008  0.0057  0.004  0.0028  0.002  0.0014  0.001".split()))
        s_val_num = len(s_values)
        states_num = 9
        init_Eoff = [0.0 for x in range(states_num)]
        sampling_stat = {"state_undersampling_potTresh":  [0,0,0,0,0,0,0,0,0],
                         "undersampling_occurence_sampling_tresh": 0.9}

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_PNMT_9ligs, ene_trajs_prefix="energies_")

        #sampling style
        import reeds.function_libs.analysis.sampling as sampling_ana
        (sampling_results, out_dir) = sampling_ana.detect_undersampling(out_path="",
                                                                        ene_traj_csvs=ene_ana_trajs,
                                                                        s_values=s_values,
                                                                        eoffs = init_Eoff,
                                                                        state_potential_treshold=sampling_stat['state_undersampling_potTresh'],
                                                                        _visualize=False,
                                                                        undersampling_occurence_sampling_tresh=sampling_stat['undersampling_occurence_sampling_tresh'])

        print(sampling_results)

    def test_whole(self):
        #params:
        expected_res = np.array([0.,-3.14132494, -4.67494518, 6.79295467, -38.41314081, -333.98504042, -378.36301104, -10.06737494, -124.97849135])
        T = 298
        s_values = list(map(float, "1  0.7  0.5  0.35  0.25  0.18  0.13  0.089  0.063  0.044  0.031  0.022  0.016  0.011  0.008  0.0057  0.004  0.0028  0.002  0.0014  0.001".split()))
        s_val_num = len(s_values)
        states_num = 9
        init_Eoff = [0.0 for x in range(states_num)]
        sampling_stat = {"state_undersampling_potTresh":  [0,0,0,0,0,0,0,0,0],
                         "undersampling_occurence_sampling_tresh": 0.9}

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_PNMT_9ligs, ene_trajs_prefix="energies_")

        #sampling style
        import reeds.function_libs.analysis.sampling as sampling_ana
        (sampling_results, out_dir) = sampling_ana.detect_undersampling(out_path="",
                                                                        ene_traj_csvs=ene_ana_trajs,
                                                                        s_values=s_values,
                                                                        eoffs=init_Eoff,
                                                                        state_potential_treshold=sampling_stat['state_undersampling_potTresh'],
                                                                        _visualize=False,
                                                                        undersampling_occurence_sampling_tresh=sampling_stat['undersampling_occurence_sampling_tresh'])

        print(sampling_results['undersamplingThreshold'])
        new_eoffs, all_eoffs = eds_energy_offsets.estimate_energy_offsets(ene_trajs=ene_ana_trajs,
                                                                          initial_offsets=init_Eoff,
                                                                          sampling_stat=sampling_results,
                                                                          s_values=s_values,
                                                                          out_path=None, temp=T, trim_beg=0.,
                                                                          undersampling_idx=sampling_results['undersamplingThreshold'],
                                                                          plot_results=False, calc_clara=False)

        print(new_eoffs)
        # do the comparison with the previous data
        equal_criteria = 0.1 # kJ/mol, results must not differ by more than this value
        np.testing.assert_almost_equal(expected_res, new_eoffs,  decimal=3)
