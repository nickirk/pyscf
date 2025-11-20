#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import tempfile
import numpy as np
from functools import reduce

from pyscf import gto, lib
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import rccsd
from pyscf.cc import ccsd

MEMORYMIN = getattr(ccsd, 'MEMORYMIN', 2000)

def setUpModule():
    global mol, mf, mycc, eris
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    # CO2 in ccpvdz basis
    mol.atom = 'C 0 0 0; O 0 0 1.2; O 0 0 -1.2'
    #mol.atom = 'Li 0 0 0; H 0 0 2.0'
    #mol.atom = [
    #    [1, (0.0, 0.0, 0.0)],
    #    [1, (0.0, 0.0, 2.0)],
    #    [1, (0.0, 0.0, 4.0)],
    #    [1, (0.0, 0.0, 6.0)],
    #    [1, (0.0, 0.0, 8.0)],
    #    [1, (0.0, 0.0, 10.0)],
    #    [1, (0.0, 0.0, 12.0)],
    #    [1, (0.0, 0.0, 14.0)],
    #    [1, (0.0, 0.0, 16.0)],
    #    [1, (0.0, 0.0, 18.0)]
    #]

    mol.basis = 'ccpvdz'
    mol.verbose = 5
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    mycc = rccsd.RCCSD(mf)
    mycc.verbose = 5
    mycc.conv_tol = 1e-12
    mycc.max_cycle = 200
    eris = rccsd._make_eris_from_file('/Users/keliao/Work/project/pyscf/pyscf/cc/test/FCIDUMP.co2.ccpvdz.t.txt', mycc)
    #eris = rccsd._make_eris_from_file('/Users/keliao/Work/project/pyscf/pyscf/cc/test/FCIDUMP.LiH.321g', mycc)
    #mycc.kernel(eris=eris)

def tearDownModule():
    global mol, mf, mycc, eris
    mol.stdout.close()
    del mol, mf, mycc, eris

class KnownValues(unittest.TestCase):
    def test_rccsd(self):
        '''Test basic NHRCCSD calculation'''
        #t1 = np.zeros((mycc.nocc, mycc.nmo-mycc.nocc))
        e_corr, t1, t2 = mycc.kernel(eris=eris)
        #self.assertAlmostEqual(e_corr, -0.093860447736, 8)
        print(e_corr)
    #def test_eris(self):
    #    '''Test ERIS construction and storage'''
    #    np.random.seed(1)
    #    mo_coeff = np.random.random(mf.mo_coeff.shape)
    #    eris = rccsd._make_eris_incore(mycc, mo_coeff)

        # Test that all ERI components are stored without symmetry
        #self.assertAlmostEqual(lib.fp(eris.oooo), 4.963884938282539, 11)
        #self.assertAlmostEqual(lib.fp(eris.ovoo), -1.362368189698315, 11)
        #self.assertAlmostEqual(lib.fp(eris.ovov), 125.815506844421580, 11)
        #self.assertAlmostEqual(lib.fp(eris.oovv), 55.123681017639463, 11)
        #self.assertAlmostEqual(lib.fp(eris.ovvo), 133.480835278982620, 11)
        #self.assertAlmostEqual(lib.fp(eris.ovvv), 95.756230114113222, 11)
        #self.assertAlmostEqual(lib.fp(eris.vvvv), -10.450387490987071, 11)

    #def test_outcore_eris(self):
    #    '''Test outcore ERIS construction'''
    #    bak = MEMORYMIN
    #    ccsd.MEMORYMIN = 0
    #    mycc.max_memory = 0
    #    eris1 = mycc.ao2mo(mf.mo_coeff)
    #    ccsd.MEMORYMIN = bak

    #    # Test that outcore and incore ERIS match
    #    self.assertAlmostEqual(abs(np.array(eris1.oooo)-eris.oooo).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.ovoo)-eris.ovoo).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.ovov)-eris.ovov).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.oovv)-eris.oovv).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.ovvo)-eris.ovvo).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.ovvv)-eris.ovvv).max(), 0, 11)
    #    self.assertAlmostEqual(abs(np.array(eris1.vvvv)-eris.vvvv).max(), 0, 11)

    def test_file_eris(self):
        '''Test file-based ERIS construction. This is to test _make_eris_from_file '''
        #filename = 'fcidump'
        #eris = rccsd._make_eris_from_file(filename, mycc)
        # assert the two particle exchange symmetry is there
        self.assertTrue(np.allclose(eris.ovov, eris.ovov.transpose(2,3,0,1).conj()))
        self.assertTrue(np.allclose(eris.oooo, eris.oooo.transpose(2,3,0,1).conj()))
        self.assertTrue(np.allclose(eris.vvvv, eris.vvvv.transpose(2,3,0,1).conj()))

        e_fock = 2. * np.einsum('ii->', eris.fock[:mycc.nocc, :mycc.nocc])
        dirHFE = 2. * np.einsum('iijj->', eris.oooo)
        excHFE = -1. * np.einsum('ijij->', eris.oooo)

        e_fock = e_fock - (dirHFE + excHFE) + eris.e_core
        #self.assertAlmostEqual(e_fock,-5.5448830796638475, 6)
    #def test_complex_integrals(self):
    #    '''Test NHRCCSD with complex integrals'''
    #    mol = gto.M()
    #    mol.verbose = 0
    #    nocc, nvir = 5, 12
    #    nmo = nocc + nvir
    #    np.random.seed(1)
    #    
    #    # Create complex MO coefficients
    #    mo_coeff = np.random.random((nmo,nmo)) + 1j * np.random.random((nmo,nmo))
    #    
    #    # Create complex ERIs
    #    eri = np.random.random((nmo,nmo,nmo,nmo)) + 1j * np.random.random((nmo,nmo,nmo,nmo))
    #    eri = eri + eri.transpose(1,0,3,2).conj()
    #    
    #    mf = scf.RHF(mol)
    #    mf.mo_coeff = mo_coeff
    #    mf.mo_energy = np.arange(0., nmo)
    #    mf.mo_occ = np.zeros(nmo)
    #    mf.mo_occ[:nocc] = 2
    #    
    #    mycc = rccsd.RCCSD(mf)
    #    eris = mycc.ao2mo()
    #    
    #    # Test that ERIs maintain proper symmetry
    #    self.assertTrue(np.allclose(eris.ovov, eris.ovov.transpose(2,3,0,1).conj()))
    #    self.assertTrue(np.allclose(eris.oooo, eris.oooo.transpose(2,3,0,1).conj()))
    #    self.assertTrue(np.allclose(eris.vvvv, eris.vvvv.transpose(2,3,0,1).conj()))

    #def test_update_amps(self):
    #    '''Test amplitude update equations'''
    #    t1, t2 = mycc.t1, mycc.t2
    #    t1new, t2new = rccsd.update_amps(mycc, t1, t2, eris)
        
        # Test that amplitudes maintain proper symmetry
        #self.assertTrue(np.allclose(t2new, t2new.transpose(1,0,3,2)))
        
        ## Test against known values
        #self.assertAlmostEqual(lib.fp(t1new), 0.0177974121446, 6)
        #self.assertAlmostEqual(lib.fp(t2new), -0.0561112639789, 6)

    #def test_energy(self):
    #    '''Test energy calculation'''
    #    e_corr = rccsd.energy(mycc, mycc.t1, mycc.t2, eris)
    #    self.assertAlmostEqual(e_corr, -0.096069872087, 6)

if __name__ == "__main__":
    print("Full Tests for NHRCCSD")
    unittest.main() 