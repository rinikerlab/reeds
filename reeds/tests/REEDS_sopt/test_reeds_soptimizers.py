import unittest
import os
from reeds.function_libs.optimization.src import sopt_Pathstatistic as stat, s_optimizer as opt

from pygromos.files.repdat import Repdat

in_repdat= os.path.dirname(__file__)+"/data/in_REEDS_repdat2_short.dat"
in_repdat2= os.path.dirname(__file__)+"/data/in_REEDS_repdat3_dsidler_iter1_sopt.dat"

class test_optimizerFunctions(unittest.TestCase):
    def test_RTO_nice_svals(self):
        repdat = Repdat(in_repdat)
        stat_file = stat.generate_PathStatistic_from_file(repdat)

        stat_file.s_values[-1] = 0.000316
        NLRTO = opt.N_LRTO(stat_file)
        NLRTO.optimize(3)
        new_s =NLRTO.get_new_replica_dist()
        print(NLRTO.orig_replica_parameters)
        print(new_s)

    def test_NGRTO_calc_c_prime(self):
        repdat = Repdat(in_repdat)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        NGRTO = opt.N_GRTO(stat_file)
        expected_c_prime = 2.414668163253696 #old c_prime if not divided by numstates in the weights: 0.9126587798163391

        verbose =False
        ds =0.00001
        f_n_list = NGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = NGRTO._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_NGRTO_calc_c_prime2(self):
        repdat = Repdat(in_repdat2)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        NGRTO = opt.N_GRTO(stat_file)
        expected_c_prime = 0.8419764136870403 #old c_prime if not divided by numstates in the weights:0.2806587438559297

        verbose =False
        ds =0.0001
        f_n_list = NGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = NGRTO._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_OneGRTO_calc_c_prime(self):
        repdat = Repdat(in_repdat)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        OneGRTO = opt.One_GRTO(stat_file)
        expected_c_prime =0.962742673126844

        verbose =False
        ds =0.00001
        f_n_list = OneGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = OneGRTO._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_OneGRTO_calc_c_prime2(self):
        repdat = Repdat(in_repdat2)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        OneGRTO = opt.One_GRTO(stat_file)
        expected_c_prime = 0.801909959350048

        verbose =False
        ds =0.0001
        f_n_list = OneGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = OneGRTO._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_NGRTO_calc_c_prime_old_integral(self):
        repdat = Repdat(in_repdat)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        NGRTO = opt.N_GRTO(stat_file)
        expected_c_prime = 2.414668163253711 #old c_prime if not divided by numstates in the weights: 0.9126587798163391

        verbose =False
        ds =0.00001
        f_n_list = NGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = NGRTO._calculate_normalisation_c_prime_integral(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_NGRTO_calc_c_prime2_old_integral(self):
        repdat = Repdat(in_repdat2)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        NGRTO = opt.N_GRTO(stat_file)
        expected_c_prime = 0.8419764136870506 #old c_prime if not divided by numstates in the weights:0.2806587438559297

        verbose =False
        ds =0.0001
        f_n_list = NGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        c_prime = NGRTO._calculate_normalisation_c_prime_integral(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        self.assertAlmostEqual(expected_c_prime,c_prime, len(str(ds)), msg="Values do not fit.")

    def test_NGRTO_calc_sDist_area(self):
        repdat = Repdat(in_repdat2)
        stat_file = stat.generate_PathStatistic_from_file(repdat)
        NGRTO = opt.N_GRTO(stat_file)
        expected_s = [ 1, 0.0629, 0.0257, 0.001]

        verbose =False
        add_n = 0
        ds =0.0001
        f_n_list = NGRTO._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        new_replica_num = add_n+len(old_s_dist)
        c_prime = NGRTO._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds, verbose=verbose)

        verbose =True
        s_fast = NGRTO._add_s_values_accord_to_flow_area(old_s_dist=old_s_dist, f_n_list=f_n_list, c_prime=c_prime, ds=ds, new_replica_num=new_replica_num, verbose=verbose)

        print(s_fast)
        nice_s_integral = NGRTO._nice_sval_list(svals_list=s_fast, sig_digits=4)
        print("integral")
        print(NGRTO)
        self.assertEqual(len(expected_s), len(nice_s_integral))
        self.assertEqual(expected_s, nice_s_integral)



if __name__ == '__main__':
    unittest.main()
