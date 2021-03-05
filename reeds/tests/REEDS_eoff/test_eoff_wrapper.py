import unittest
import os
from reeds.function_libs.optimization import eds_energy_offsets
from reeds.function_libs.file_management import file_management as fM

in_BRD4_7ligs = os.path.dirname(__file__) + "/data/7ligs"
in_PNMT_9ligs = os.path.dirname(__file__)+"/data/PNMT_9ligs"

out_result_BRD4_7ligs = os.path.dirname(__file__) + "/data/out_test_result_BRD4_7ligs.out"
out_result_PNMT_9ligs = os.path.dirname(__file__)+"/data/out_test_result_PNMT_9ligs.out"


class test_Eoff_wrapper(unittest.TestCase):
    def test_parser(self):
        #check_files

        #params:
        T = 298
        kb=0.00831451
        frac_tresh = 0.6
        rho = 0.0
        pot_tresh = 200
        max_iter = 20

        s_values = [1.0, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01, 0.0056, 0.0032, 0.0018, 0.001,
                    0.0006, 0.0003]
        s_val_num = len(s_values)
        states_num = 7

        init_Eoff = [0.0 for x in range(states_num)]

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_BRD4_7ligs, ene_trajs_prefix="energies_")
        print(ene_ana_trajs)
        #calc Eoffs and control
        parse = eds_energy_offsets.parse_args(ene_ana_trajs=ene_ana_trajs,
                                  s_values=s_values, Eoff=init_Eoff,
                                  Temp=T,  #kb=kb,
                                  frac_tresh=[frac_tresh], max_iter=max_iter, convergenceradius=rho, pot_tresh=pot_tresh)

    def test_eoff_BRD4_7ligs(self):
        #result:
        expected_Eoffs = [(0.0, 0.0), (290.8132, 10.5879), (233.9000, 3.4339), (-98.1871, 7.9141), (34.4407, 6.7009),
                          (-308.3202, 4.5301), (140.6267, 7.4697)]

        #params:
        T = 298
        kb=0.00831451
        frac_tresh = 0.6
        rho = 0.0
        pot_tresh = 200
        max_iter = 20

        s_values = list(map(float, "1.0 0.56234133 0.31622777 0.17782794 0.1 0.05623413 0.03162278 0.01778279 0.01 0.00562341 0.00316228 0.00177828 0.001 0.00056234 0.00031623".split()))
        s_val_num = len(s_values)
        states_num = 7
        init_Eoff = [0.0 for x in range(states_num)]

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_BRD4_7ligs, ene_trajs_prefix="energies_")

        #calc Eoffs and control
        self.estEoff(ene_ana_trajs=ene_ana_trajs, init_Eoff=init_Eoff, s_values=s_values, expected_Eoffs=expected_Eoffs, output_result=out_result_BRD4_7ligs,
                    T = T, kb = kb,
                    pot_tresh = pot_tresh, frac_tresh = frac_tresh,
                    rho = rho, max_iter = max_iter)

    def test_eoff_PNMT_9ligs(self):
        """
        Data from publication Sidler 2016 - 9ligands Table IV
        :return:
        """
        expected_Eoffs=[(0.0, 0.0), (-1.872856, 7.53874), (-3.2207 , 6.160522),
                        (7.41792, 6.93919), (-36.8417, 6.21895), (-335.875, 10.5098), (-382.97, 19.7235), (-8.67558, 5.40899), (-123.815, 5.35271)]

        #params:
        T = 298
        kb=0.00831451
        frac_tresh = 0.6
        rho = 0.0
        pot_tresh = 200
        max_iter = 20

        s_values = list(map(float, "1  0.7  0.5  0.35  0.25  0.18  0.13  0.089  0.063  0.044  0.031  0.022  0.016  0.011  0.008  0.0057  0.004  0.0028  0.002  0.0014  0.001".split()))

        s_val_num = len(s_values)
        states_num = 9
        init_Eoff = [0.0 for x in range(states_num)]

        #find Files
        ene_ana_trajs = fM.parse_csv_energy_trajectories(in_folder=in_PNMT_9ligs, ene_trajs_prefix="energies_")

        #calc Eoffs and control
        self.estEoff(ene_ana_trajs, init_Eoff=init_Eoff, s_values=s_values, expected_Eoffs=expected_Eoffs, output_result=out_result_BRD4_7ligs,
                    T = T, kb = kb,
                    pot_tresh = pot_tresh, frac_tresh = frac_tresh,
                    rho = rho, max_iter = max_iter)

    def estEoff(self, ene_ana_trajs:list, s_values:list, init_Eoff:list, expected_Eoffs:list, output_result:str, T:float, kb:float, pot_tresh:float, frac_tresh:float, rho:float, max_iter:int):
        #do
        eoffs = eds_energy_offsets.estEoff(ene_ana_trajs=ene_ana_trajs, out_path=output_result,
                                   s_values=s_values, Eoff=init_Eoff,
                                   Temp=T,
                                   pot_tresh=pot_tresh, frac_tresh=[frac_tresh], convergenceradius=rho, max_iter=max_iter)
        #check Result
        collect_error=[]
        for ind, offset in enumerate(eoffs.offsets):
            accuracy= -1
            try:
                print("decimal expected", 1/(10**(len(str(expected_Eoffs[ind][0]).split(".")[1]))))
                print("decimal calc", 1/(10**(len(str(offset.mean).split(".")[1]))))
                accuracy = max(1/(10**(len(str(expected_Eoffs[ind][0]).split(".")[1]))), 1/(10**(len(str(offset.mean).split(".")[1]))))
                accuracy = 0.1 if(accuracy>=1.0 or accuracy == 0) else accuracy
                print("used_acc", accuracy)
                self.assertAlmostEqual(first=offset.mean, second=expected_Eoffs[ind][0], delta= accuracy,
                                   msg= str(ind)+". Offset mean was calculated as: \n"+str(offset.mean)+"\tbut should be\t"+str(expected_Eoffs[ind][0])+"\n")
            except:
                msg= str(ind)+". Offset mean was calculated as: \n\t"+str(offset.mean)+"\tbut should be\t"+str(expected_Eoffs[ind][0])+ "\nthis is outside of the accuracy range: "+str(accuracy)
                collect_error.append(msg)

            try:
                print("decimal expected", 1/(10**(len(str(expected_Eoffs[ind][1]).split(".")[1]))))
                print("decimal calc", 1/(10**(len(str(offset.std).split(".")[1]))))
                accuracy = max(1/(10**(len(str(expected_Eoffs[ind][1]).split(".")[1]))), 1/(10**(len(str(offset.std).split(".")[1]))))
                accuracy = 0.1 if(accuracy>=1.0 or accuracy == 0) else accuracy
                self.assertAlmostEqual(first=offset.std, second=expected_Eoffs[ind][1], delta= accuracy,
                                       msg= str(ind)+". Offset mean was calculated as: \n"+str(offset.std)+"\tbut should be\t"+str(expected_Eoffs[ind][1]))
                print("used_acc", accuracy)

            except:
                msg = str(ind)+". Offset error was calculated as: \n\t"+str(offset.std)+"\t but should be\t"+str(expected_Eoffs[ind][1])+"\n"
                collect_error.append(msg)
            if(accuracy == -1):
                raise Exception("Accuracy was not set while checking Eoffs!")

        if(len(collect_error)> 0):
            print("The calculated Values are not equal to control!: \n"+"\n".join(collect_error))
            self.fail("The calculated Values are not equal to control!:  \n"+"\n".join(collect_error))

