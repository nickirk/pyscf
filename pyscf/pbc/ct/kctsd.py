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
# Author: Ke Liao <ke.liao.whu@gmail.com>

import numpy

from pyscf import lib
from pyscf.ct import ctsd
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import RCCSD

kernel = pyscf.ct.ctsd.kernel

class RCTSD(ctsd.CTSD):
    """
    Restricted k-CTSD.
    """
    def __init__(self, mf, a_nmo=None, mo_coeff=None, mo_occ=None,
                 dm1=None, dm2=None, eri=None):
        assert (isinstance(mf, scf.khf.KSCF))
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)

    @property
    def nkpts(self):
        return len(self.kpts)
    
    def init_amps(self):
        if self._amps_algo == "mp2":
            return self.get_mp2_amps()
    
    def get_mp2_amps(self):
        pass
