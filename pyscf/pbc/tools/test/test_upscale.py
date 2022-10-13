#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ke Liao <ke.liao.whu@gmail.com>
#
import unittest
import numpy as np

from pyscf import gto, scf

from pyscf.lib import fp
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.tools.upscale import *

from pyscf.pbc import cc

from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.tools.make_test_cell as make_test_cell

def setUpModule():
    global cell
    '''
    Example calculation on H2 chain
    '''
    cell = pbcgto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'sto6g'
    cell.ke_cutoff = 50
    cell.atom='''
        H 2.00   2.00   1.20
        H 2.00   2.00   2.60
        '''
    cell.a = '''
        4.0   0.0   0.0
        0.0   4.0   0.0
        0.0   0.0   4.0
        '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_k222_K222(self):
        nks_mf_s = [2, 2, 2]
        kmf_s, kmp_s = set_up_method(cell, nks_mf_s)
        e_mp2_s, t2_mp2_s = kmp_s.kernel()

        mycc_s = cc.KCCSD(kmf_s)
        ecc_s, t1_cc_s, t2_cc_s = mycc_s.kernel()

        #for nks_mf_d in [[3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]: 
        for nks_mf_d in [[2, 2, 2]]: 
            kmf_d, kmp_d = set_up_method(cell, nks_mf_d)
            dist_nm = get_nn_dist(kmf_s.kpts, kmf_d.kpts)

            emp2_d, t2_d = kmp_d.kernel()


            # set up CCSD 
            e_us = 0.
            mycc_d = cc.KCCSD(kmf_d)
            ecc_d, t1_cc_d, t2_cc_d = mycc_d.kernel()
            e_us, t1_us, t2_us = upscale(t1_cc_s, t2_cc_s, t2_d, kmp_s, kmp_d, dist_nm, 1)
            abs_diff_sum = np.abs(t2_us) - np.abs(t2_cc_d)
            abs_diff_sum = np.einsum("xyzijab ->", abs_diff_sum)
            diff_sum = t2_us - t2_cc_d
            diff_sum = np.einsum("xyzijab ->", diff_sum)
            self.assertAlmostEqual(ecc_d, e_us, 7)
            print(nks_mf_d[0], abs_diff_sum, diff_sum)
            self.assertAlmostEqual(abs_diff_sum, 0., 12)
            self.assertAlmostEqual(diff_sum, 0., 12)

if __name__ == '__main__':
    print("Full test of upscale")
    unittest.main()