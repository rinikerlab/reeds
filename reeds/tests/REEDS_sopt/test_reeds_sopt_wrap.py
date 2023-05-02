import unittest

from reeds.function_libs.optimization import eds_s_values as opt
from reeds.function_libs.optimization.src.sopt_Pathstatistic import generate_PathStatistic_from_file
import os

from pygromos.files.repdat import Repdat

in_repdat= os.path.dirname(__file__)+"/data/in_REEDS_repdat2_short.dat"
in_repdat2= os.path.dirname(__file__)+"/data/in_REEDS_repdat3_dsidler_iter1_sopt.dat"

class test_RTOs(unittest.TestCase):
    def test_parse_parse_repdat(self):
        repdat = Repdat(in_repdat)
        stat = generate_PathStatistic_from_file(repdat)

    def test_parse_parse_repdat_skipping_offset(self):
        repdat = Repdat(in_repdat)
        stat = generate_PathStatistic_from_file(repdat, trial_range=10)

    def test_parse_parse_repdat_skipping_range(self):
        repdat = Repdat(in_repdat)
        stat = generate_PathStatistic_from_file(repdat, trial_range=(5, 10))

    def test_NLRTO_adding_5(self):
        
        repdat = Repdat(in_repdat)

        #add s_vals
        add_n_s=5
        expected_s = [1.0, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01, 0.0056, 0.0052, 0.0048, 0.0044, 0.004, 0.0036, 0.0032, 0.0018, 0.001, 0.0006, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_NLRTO(stat, add_n_s)
        print(LRTO)
        self.assertEqual(s_vals, expected_s)

    def test_NLRTO_adding_0(self):
        repdat = Repdat(in_repdat)
        #Don"t add a thing
        add_n_s=0
        expected_s = [1.0, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01, 0.0056, 0.0032, 0.0018, 0.001, 0.0006, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_NLRTO(stat, add_n_s)

        self.assertEqual(len(s_vals), len(expected_s), msg="Optimizing and adding "+str(add_n_s)+" s_values failed.")

    def test_NLRTO_dsidler_file_adding_0(self):
        
        repdat = Repdat(in_repdat2)
        
        add_n_s = 0
        expected_s = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.01, 0.001]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_NLRTO(stat, add_n_s)

        print(LRTO)
        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,
                         msg="Optimizing and adding " + str(add_n_s) + " s_values failed. Values not the same.")

    def test_NLRTO_dsidler_file_adding_4(self):
        
        repdat = Repdat(in_repdat2)
        
        add_n_s = 4
        expected_s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.082, 0.064, 0.046, 0.028, 0.01, 0.001]
        stat = generate_PathStatistic_from_file(repdat)

        s_vals, LRTO = opt.calc_NLRTO(stat, add_n_s)

        print(LRTO)
        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,
                         msg="Optimizing and adding " + str(add_n_s) + " s_values failed. Values not the same.")

    def test_NGRTO_adding_5(self):
        
        repdat = Repdat(in_repdat)
        
        add_n_s=5
        expected_s = [1.0, 0.00545, 0.00531, 0.00517, 0.00503, 0.00489, 0.00475, 0.00461, 0.00447, 0.00433, 0.00419, 0.00405, 0.00391, 0.00377, 0.00363, 0.00349, 0.00335, 0.00321, 0.00141, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, NGRTO = opt.calc_NGRTO(stat, add_n_s, ds=0.00001, verbose=True)

        print(s_vals)

        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_NGRTO_adding_0(self):
        repdat = Repdat(in_repdat)
        add_n_s=0
        expected_s = [1.0, 0.0054, 0.00521, 0.00502, 0.00483, 0.00464, 0.00445, 0.00426, 0.00407, 0.00388, 0.00369, 0.0035, 0.00331, 0.00157, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, NGRTO = opt.calc_NGRTO(stat, add_n_s, ds=0.00001, verbose=True)
        print(s_vals)
        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_NGRTO_dsidler_file_adding_0(self):
        
        repdat = Repdat(in_repdat2)
        add_n_s=0
        expected_s = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0629, 0.0257, 0.001]

        stat =  generate_PathStatistic_from_file(repdat)
        s_vals, NGRTO = opt.calc_NGRTO(stat, add_n_s, verbose=True)

        print(NGRTO)
        print(s_vals)
        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding "+str(add_n_s)+" s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,
                         msg="Optimizing and adding " + str(add_n_s) + " s_values failed. Values not the same.")

    def test_NGRTO_dsidler_file_adding_4(self):
        
        repdat = Repdat(in_repdat2)
        add_n_s=4
        expected_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0842, 0.0682, 0.0522, 0.0362, 0.0202, 0.0075, 0.001]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, NGRTO = opt.calc_NGRTO(stat, add_n_s, verbose=True)

        print(NGRTO)
        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding "+str(add_n_s)+" s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,
                         msg="Optimizing and adding " + str(add_n_s) + " s_values failed. Values not the same.")

    def test_one_GRTO_adding_5(self):

        repdat = Repdat(in_repdat)

        add_n_s=5
        expected_s = [1.0, 0.00545, 0.00531, 0.00517, 0.00503, 0.00489, 0.00475, 0.00461, 0.00447, 0.00433, 0.00419, 0.00405, 0.00391, 0.00377, 0.00363, 0.00349, 0.00335, 0.00321, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, GRTO = opt.calc_oneGRTO(stat=stat,  ds=0.00001, add_n_s=add_n_s, verbose=True)

        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_GRTO_adding_0(self):
        
        repdat = Repdat(in_repdat)
        add_n_s=0
        expected_s = [1.0, 0.00541, 0.00523, 0.00505, 0.00487, 0.00469, 0.00451, 0.00433, 0.00415, 0.00397, 0.00379, 0.00361, 0.00343, 0.00325, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, GRTO = opt.calc_oneGRTO(stat=stat, add_n_s=add_n_s, ds=0.00001, verbose=True)
        print(s_vals)

        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_GRTO_dsidler_file_adding_0(self):

        repdat = Repdat(in_repdat2)
        add_n_s = 0
        expected_s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0607,  0.0215, 0.001]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, GRTO = opt.calc_oneGRTO(stat=stat, add_n_s=add_n_s, verbose=True)


        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_GRTO_dsidler_file_adding_4(self):
        
        repdat = Repdat(in_repdat2)
        add_n_s = 4
        expected_s = [1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0831, 0.0663,  0.0495,  0.0327,  0.0159,  0.0064,  0.001]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, GRTO = opt.calc_oneGRTO(stat=stat, add_n_s=add_n_s, verbose=True)

        print("Length: expected/opt")
        print(len(expected_s), len(s_vals))

        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_LRTO_adding_5(self):
        repdat = Repdat(in_repdat)
        add_n_s=5
        expected_s = [1.0, 0.5623, 0.3162, 0.1778, 0.1, 0.0562, 0.0316, 0.0178, 0.01, 0.0056, 0.0052, 0.0048, 0.0044, 0.004, 0.0036, 0.0032, 0.0018, 0.001, 0.0006, 0.0003]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_oneLRTO(stat=stat, add_n_s=add_n_s)

        print(s_vals)
        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_LRTO_adding_0(self):
        repdat = Repdat(in_repdat)
        add_n_s=0
        stat = generate_PathStatistic_from_file(repdat)
        expected_s = stat.skipped_s_values+stat.s_values
        s_vals, LRTO = opt.calc_oneLRTO(stat=stat, add_n_s=add_n_s)
        print(s_vals)

        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_LRTO_dsidler_file_adding_0(self):
        repdat = Repdat(in_repdat2)
        add_n_s = 0
        stat = generate_PathStatistic_from_file(repdat)
        expected_s = stat.skipped_s_values+stat.s_values
        s_vals, LRTO = opt.calc_oneLRTO(stat=stat, add_n_s=add_n_s)


        self.assertEqual(len(stat.s_values)+len(stat.skipped_s_values), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_one_LRTO_dsidler_file_adding_4(self):
        
        repdat = Repdat(in_repdat2)
            
        add_n_s = 4
        expected_s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.082, 0.064, 0.046, 0.028, 0.01, 0.001]
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_oneLRTO(stat=stat, add_n_s=add_n_s)

        print("Length: expected/opt")
        print(len(expected_s), len(s_vals))
        print(s_vals)
        self.assertEqual(len(expected_s), len(s_vals), msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have now not the same value number")
        self.assertEqual(expected_s, s_vals,  msg="Optimizing and adding " + str(
            add_n_s) + " s_values failed. Have not the same value")

    def test_RTO_string(self):
        add_n_s=5

        repdat = Repdat(in_repdat)
        stat = generate_PathStatistic_from_file(repdat)
        s_vals, LRTO = opt.calc_NLRTO(stat, add_n_s)

        print(LRTO)
