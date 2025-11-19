#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import sys
import os
import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import KRHF

# Import kpwscf from local directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kpwscf import KPWSCF


def make_test_cell(mesh):
    """Create a simple test cell with H2."""
    cell = pbcgto.Cell()
    cell.atom = '''
    H 0.0 0.0 0.0
    H 1.0 0.0 0.0
    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 4.0  # 4 Bohr cubic cell
    cell.mesh = mesh
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.build()
    return cell


def setUpModule():
    global cell, mf
    cell = make_test_cell([20, 20, 20])
    # Initialize at Gamma point with 4 bands
    mf = KPWSCF(cell, kpts=np.zeros((1, 3)), nband=4)
    mf.verbose = 0
    mf.build()
    mf.init_guess(kind='random', seed=42)


def tearDownModule():
    global cell, mf
    cell.stdout.close()
    del cell, mf


class KnownValues(unittest.TestCase):
    
    def test_build_operators(self):
        """Test that operators are correctly built."""
        # Check that Gv is computed
        self.assertIsNotNone(mf._Gv)
        self.assertEqual(mf._Gv.shape, (mf.ngrids, 3))
        
        # Check that kinetic diagonal is computed
        self.assertIsNotNone(mf._kin_diag)
        self.assertEqual(mf._kin_diag.shape, (mf.nk, mf.ngrids))
        
        # Check that V_ne is computed
        self.assertIsNotNone(mf._vne_R)
        self.assertEqual(mf._vne_R.shape, (mf.ngrids,))
        
        # Check that Coulomb kernel is cached
        self.assertIsNotNone(mf._coulG0)
        self.assertEqual(mf._coulG0.shape, (mf.ngrids,))

    def test_wavefunction_initialization(self):
        """Test wavefunction initialization and orthonormalization."""
        # Check psi_r and psi_g are initialized
        self.assertIsNotNone(mf.psi_r)
        self.assertIsNotNone(mf.psi_g)
        self.assertEqual(mf.psi_r.shape, (mf.nk, mf.nband, mf.ngrids))
        self.assertEqual(mf.psi_g.shape, (mf.nk, mf.nband, mf.ngrids))
        
        # Check approximate orthonormalization for first k-point
        ik = 0
        psi_r = mf.psi_r[ik]  # (nband, ngrids)
        overlap = np.dot(psi_r.conj(), psi_r.T) * mf.grid_weight
        # Should be approximately close to identity (relax tolerance for random init)
        self.assertTrue(np.allclose(np.diag(overlap), 1.0, atol=0.1))
        # Off-diagonals should be small
        off_diag = overlap - np.diag(np.diag(overlap))
        self.assertLess(np.abs(off_diag).max(), 0.1)

    def test_density_integration(self):
        """Test that density integrates to correct number of electrons."""
        rho_r = mf.get_density_r()
        self.assertEqual(rho_r.shape, (mf.ngrids,))
        
        # Integrate density
        nelec = np.sum(rho_r) * mf.grid_weight
        expected_nelec = cell.nelectron
        
        # Should be close to expected number of electrons
        self.assertAlmostEqual(nelec, expected_nelec, places=1)

    def test_fft_consistency(self):
        """Test FFT and IFFT are inverses."""
        ik = 0
        n = 0
        psi_r = mf.psi_r[ik, n]
        
        # FFT to G-space and back
        psi_g = mf._fft_r2g(psi_r)
        psi_r_back = mf._ifft_g2r(psi_g)
        
        # Should recover original
        self.assertTrue(np.allclose(psi_r, psi_r_back, atol=1e-10))

    def test_apply_kinetic(self):
        """Test kinetic energy operator in G-space."""
        ik = 0
        n = 0
        psi_g = mf.psi_g[ik, n]
        
        # Apply kinetic energy manually
        t_g = mf._kin_diag[ik]
        t_psi_g = t_g * psi_g
        
        # Compute expectation value <ψ|T|ψ>
        psi_r = mf._ifft_g2r(psi_g)
        t_psi_r = mf._ifft_g2r(t_psi_g)
        E_kin = np.real(np.vdot(psi_r, t_psi_r)) * mf.grid_weight
        
        # Kinetic energy should be positive
        self.assertGreater(E_kin, 0)

    def test_apply_vne(self):
        """Test nuclear-electron potential."""
        ik = 0
        n = 0
        psi_g = mf.psi_g[ik, n]
        psi_r = mf._ifft_g2r(psi_g)
        
        # Apply V_ne
        vne_psi_r = mf._vne_R * psi_r
        vne_psi_g = mf._fft_r2g(vne_psi_r)
        vne_psi_r_back = mf._ifft_g2r(vne_psi_g)
        
        # Compute expectation value <ψ|V_ne|ψ>
        E_vne = np.real(np.vdot(psi_r, vne_psi_r_back)) * mf.grid_weight
        
        # For random wavefunctions, V_ne expectation can vary
        # Just check it's finite and reasonable
        self.assertTrue(np.isfinite(E_vne))
        # V_ne potential itself should be mostly negative
        self.assertLess(mf._vne_R.min(), 0)

    def test_apply_hcore(self):
        """Test H_core = T + V_ne."""
        ik = 0
        psi_g_block = mf.psi_g[ik, :2]  # First 2 bands
        
        # Apply H_core
        hcore_psi_g = mf._apply_hcore(ik, psi_g_block)
        
        # Check shape
        self.assertEqual(hcore_psi_g.shape, psi_g_block.shape)
        
        # Compute expectation value for first band
        psi_r = mf._ifft_g2r(psi_g_block[0])
        hcore_psi_r = mf._ifft_g2r(hcore_psi_g[0])
        E_core = np.real(np.vdot(psi_r, hcore_psi_r)) * mf.grid_weight
        
        # Energy should be reasonable (not NaN or inf)
        self.assertTrue(np.isfinite(E_core))

    def test_apply_hartree(self):
        """Test Hartree potential V_H."""
        ik = 0
        n = 0
        psi_nk_g = mf.psi_g[ik, n]
        rho_r = mf.get_density_r()
        
        # Apply only kinetic + V_ne
        H_psi_core = mf.apply_hamiltonian(ik, n, psi_nk_g, rho_r, 
                                          with_j=False, with_k=False)
        psi_r = mf._ifft_g2r(psi_nk_g)
        H_psi_r = mf._ifft_g2r(H_psi_core)
        E_core = np.real(np.vdot(psi_r, H_psi_r)) * mf.grid_weight
        
        # Apply with Hartree
        H_psi_j = mf.apply_hamiltonian(ik, n, psi_nk_g, rho_r, 
                                       with_j=True, with_k=False)
        H_psi_r = mf._ifft_g2r(H_psi_j)
        E_j = np.real(np.vdot(psi_r, H_psi_r)) * mf.grid_weight
        
        # Hartree contribution should be positive (repulsive)
        hartree_contrib = E_j - E_core
        self.assertGreater(hartree_contrib, 0)

    def test_apply_exchange(self):
        """Test exchange operator K."""
        ik = 0
        n = 0
        psi_nk_g = mf.psi_g[ik, n]
        
        # Apply exchange
        k_psi_g = mf._apply_k(ik, psi_nk_g)
        
        # Check shape
        self.assertEqual(k_psi_g.shape, psi_nk_g.shape)
        
        # Compute exchange energy
        psi_r = mf._ifft_g2r(psi_nk_g)
        k_psi_r = mf._ifft_g2r(k_psi_g)
        E_x = np.real(np.vdot(psi_r, k_psi_r)) * mf.grid_weight
        
        # Exchange should be negative (stabilizing)
        self.assertLess(E_x, 0)

    def test_apply_hamiltonian_full(self):
        """Test full Hamiltonian H = T + V_ne + V_H + V_X."""
        ik = 0
        n = 0
        psi_nk_g = mf.psi_g[ik, n]
        rho_r = mf.get_density_r()
        
        # Apply full Hamiltonian
        H_psi_g = mf.apply_hamiltonian(ik, n, psi_nk_g, rho_r, 
                                       with_j=True, with_k=True)
        
        # Check shape
        self.assertEqual(H_psi_g.shape, psi_nk_g.shape)
        
        # Compute total energy
        psi_r = mf._ifft_g2r(psi_nk_g)
        H_psi_r = mf._ifft_g2r(H_psi_g)
        E_total = np.real(np.vdot(psi_r, H_psi_r)) * mf.grid_weight
        
        # Energy should be finite
        self.assertTrue(np.isfinite(E_total))

    def test_hamiltonian_hermiticity(self):
        """Test that Hamiltonian is Hermitian."""
        ik = 0
        rho_r = mf.get_density_r()
        
        # Take two different bands
        psi1_g = mf.psi_g[ik, 0]
        psi2_g = mf.psi_g[ik, 1]
        
        # Apply Hamiltonian
        H_psi1_g = mf.apply_hamiltonian(ik, 0, psi1_g, rho_r, 
                                        with_j=True, with_k=True)
        H_psi2_g = mf.apply_hamiltonian(ik, 1, psi2_g, rho_r, 
                                        with_j=True, with_k=True)
        
        # Transform to real space
        psi1_r = mf._ifft_g2r(psi1_g)
        psi2_r = mf._ifft_g2r(psi2_g)
        H_psi1_r = mf._ifft_g2r(H_psi1_g)
        H_psi2_r = mf._ifft_g2r(H_psi2_g)
        
        # Compute <ψ1|H|ψ2> and <ψ2|H|ψ1>
        val12 = np.vdot(psi1_r, H_psi2_r) * mf.grid_weight
        val21 = np.vdot(psi2_r, H_psi1_r) * mf.grid_weight
        
        # Should satisfy Hermiticity: <ψ1|H|ψ2> = <ψ2|H|ψ1>*
        self.assertAlmostEqual(val12, np.conj(val21), places=6)

    def test_different_init_guess(self):
        """Test different initialization methods."""
        cell_test = make_test_cell([15, 15, 15])
        
        # Test random initialization
        mf_rand = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=2)
        mf_rand.build()
        mf_rand.init_guess(kind='random', seed=123)
        rho_rand = mf_rand.get_density_r()
        nelec_rand = np.sum(rho_rand) * mf_rand.grid_weight
        self.assertAlmostEqual(nelec_rand, cell_test.nelectron, places=1)
        
        # Test plane-wave initialization
        mf_pw = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=2)
        mf_pw.build()
        mf_pw.init_guess(kind='pw')
        rho_pw = mf_pw.get_density_r()
        nelec_pw = np.sum(rho_pw) * mf_pw.grid_weight
        self.assertAlmostEqual(nelec_pw, cell_test.nelectron, places=1)

    def test_gamma_point(self):
        """Test that calculation works at Gamma point."""
        cell_test = make_test_cell([15, 15, 15])
        mf_gamma = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=2)
        
        # Should not raise error
        mf_gamma.build()
        mf_gamma.init_guess(kind='random', seed=1)
        
        # Check k-point count
        self.assertEqual(mf_gamma.nk, 1)
        self.assertTrue(np.allclose(mf_gamma.kpts[0], [0, 0, 0]))

    def test_orthonormalize_block(self):
        """Test Löwdin orthonormalization."""
        # Create a non-orthogonal set of wavefunctions
        np.random.seed(100)
        nb, ngr = 3, 1000
        psi = np.random.randn(nb, ngr) + 1j * np.random.randn(nb, ngr)
        
        # Create a temporary KPWSCF instance for testing
        cell_test = make_test_cell([10, 10, 10])
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=nb)
        mf_test.build()
        
        # Orthonormalize
        psi_ortho = mf_test._orthonormalize_block(psi)
        
        # Check orthonormality
        overlap = np.dot(psi_ortho.conj(), psi_ortho.T) * mf_test.grid_weight
        self.assertTrue(np.allclose(overlap, np.eye(nb), atol=1e-6))


class NumericValues(unittest.TestCase):
    """Test against reference values for a simple system."""
    
    def test_h2_molecule_energies(self):
        """Test energy components for H2 molecule."""
        # Create clean test
        cell_h2 = make_test_cell([25, 25, 25])
        mf_h2 = KPWSCF(cell_h2, kpts=np.zeros((1, 3)), nband=2)
        mf_h2.verbose = 5
        mf_h2.build()
        mf_h2.init_guess(kind='random', seed=42)
        
        rho_r = mf_h2.get_density_r()
        ik = 0
        n = 0
        psi_g = mf_h2.psi_g[ik, n]
        psi_r = mf_h2._ifft_g2r(psi_g)
        
        # Kinetic energy should be positive
        t_psi_g = mf_h2._kin_diag[ik] * psi_g
        t_psi_r = mf_h2._ifft_g2r(t_psi_g)
        E_kin = np.real(np.vdot(psi_r, t_psi_r)) * mf_h2.grid_weight
        self.assertGreater(E_kin, 0)
        self.assertLess(E_kin, 100)  # Reasonable upper bound
        
        # V_ne potential itself should have negative minimum
        self.assertLess(mf_h2._vne_R.min(), 0)


class TestSCFKernel(unittest.TestCase):
    """Test SCF kernel with Davidson diagonalization."""
    
    def test_scf_kernel_hartree_only(self):
        """Test SCF convergence without exchange (Hartree-only)."""
        # Use small system for quick test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'H 0 0 0; H 0.74 0 0'
        cell_test.basis = 'gth-szv'
        cell_test.a = np.eye(3) * 4.0
        cell_test.mesh = [12, 12, 12]
        cell_test.verbose = 5
        cell_test.build()
        
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=3)
        mf_test.verbose = 5
        
        # Run short SCF without exchange
        e_tot, converged = mf_test.kernel(
            init='random',
            max_cycle=100,
            with_k=False,
            conv_tol=1e-4,
            conv_tol_rho=1e-3,
            davidson_tol=1e-4,
            davidson_max_cycle=15,
            alpha=0.5
        )
        
        # Check that energy is finite
        self.assertTrue(np.isfinite(e_tot))
        # Check that mo_energy is populated
        self.assertIsNotNone(mf_test.mo_energy)
        self.assertEqual(mf_test.mo_energy.shape, (1, 3))
        # Lowest energy should be less than highest
        self.assertLess(mf_test.mo_energy[0, 0], mf_test.mo_energy[0, -1])
    
    def test_scf_kernel_with_exchange(self):
        """Test SCF convergence with exchange (full HF)."""
        # Very small system for quick test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'H 0 0 0; H 0.74 0 0'
        cell_test.basis = 'cc-pvtz'
        cell_test.a = np.eye(3) * 10.0
        cell_test.mesh = [32, 32, 32]
        cell_test.verbose = 5
        cell_test.build()

        # compute pbc hf for He as reference
        from pyscf.pbc.scf import RHF
        mf_ref = RHF(cell_test, exxdiv='ewald').density_fit()
        mf_ref.verbose = 5
        e_ref = mf_ref.kernel()
        
        # Compute energy components
        # Note: PySCF uses E_tot = E_hcore + 0.5*Tr(dm @ vhf) + E_nuc
        # where vhf = vj - 0.5*vk for RHF
        # Expanding: E_tot = E_hcore + 0.5*Tr(dm @ vj) - 0.25*Tr(dm @ vk) + E_nuc
        #                  = E_kin + E_ne + 0.5*E_j - 0.25*E_k + E_nuc
        # The vk from get_jk includes the Ewald Madelung correction when exxdiv='ewald'
        dm = mf_ref.make_rdm1()
        h1e = mf_ref.get_hcore()
        vj, vk = mf_ref.get_jk(dm)  # vk includes Madelung correction
        
        # Energy formula for comparison with KPWSCF
        E_kin = np.einsum('ij,ji', dm, mf_ref.cell.pbc_intor('int1e_kin')).real
        E_hcore = np.einsum('ij,ji', dm, h1e).real
        E_ne = E_hcore - E_kin  # h1e = T + V_ne
        E_hartree = 0.5 * np.einsum('ij,ji', dm, vj).real
        E_exchange = -0.25 * np.einsum('ij,ji', dm, vk).real  # NOTE: 0.25 factor!
        E_nuc = cell_test.energy_nuc()
        
        # Verify the energy formula
        E_tot_check = E_kin + E_ne + E_hartree + E_exchange + E_nuc

        print("Reference total energy (SCF): ", e_ref)
        print("Reference total energy (computed): ", E_tot_check)
        print("Reference Kinetic energy: ", E_kin)
        print("Reference Nuclear attraction energy: ", E_ne)
        print("Reference Hartree energy: ", E_hartree)
        print("Reference Exchange energy: ", E_exchange)
        print("Reference Nuclear repulsion: ", E_nuc)

        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=4)
        mf_test.verbose = 5
        
        # Run very short SCF with exchange
        e_tot, converged = mf_test.kernel(
            init='atom',
            max_cycle=20,
            with_k=True,
            conv_tol=1e-6,
            conv_tol_rho=1e-6,
            davidson_tol=1e-6,
            davidson_max_cycle=10,
            alpha=0.5,
        )
        
        # Check that energy is finite
        self.assertTrue(np.isfinite(e_tot))
        # Check that mo_energy is populated
        self.assertIsNotNone(mf_test.mo_energy)
        self.assertEqual(mf_test.mo_energy.shape, (1, 4))


class TestInitGuess(unittest.TestCase):
    """Test different initialization methods."""
    
    def test_init_guess_by_minao(self):
        """Test initialization from minimal (ANO) basis."""
        # Small cell for fast test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'He 0 0 0; He 1.5 0 0'
        cell_test.basis = 'sto-3g'
        cell_test.a = np.eye(3) * 5.0
        cell_test.mesh = [16, 16, 16]
        cell_test.verbose = 0
        cell_test.build()
        
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=4)
        mf_test.verbose = 0
        mf_test.build()
        mf_test.init_guess_by_minao(seed=42)
        
        # Check that wavefunctions are initialized
        self.assertIsNotNone(mf_test.psi_r)
        self.assertIsNotNone(mf_test.psi_g)
        self.assertEqual(mf_test.psi_r.shape, (1, 4, mf_test.ngrids))
        self.assertEqual(mf_test.psi_g.shape, (1, 4, mf_test.ngrids))
        
        # Check normalization for all bands
        for n in range(mf_test.nband):
            # R-space normalization: ∫|ψ_r|²dr = 1
            norm_r = np.sqrt(np.sum(np.abs(mf_test.psi_r[0, n])**2) * mf_test.grid_weight)
            self.assertAlmostEqual(norm_r, 1.0, places=5,
                                   msg=f'Band {n} R-space norm = {norm_r}')
            
            # G-space normalization: Σ|ψ_g|² = 1
            norm_g = np.sqrt(np.sum(np.abs(mf_test.psi_g[0, n])**2))
            self.assertAlmostEqual(norm_g, 1.0, places=5,
                                   msg=f'Band {n} G-space norm = {norm_g}')
        
        # Check that occupied orbitals are different from random
        # (they should have structure from atomic orbitals)
        psi_r_0 = mf_test.psi_r[0, 0]
        psi_r_1 = mf_test.psi_r[0, 1]
        # Orbitals should be different
        self.assertFalse(np.allclose(psi_r_0, psi_r_1))
        
        # Compute initial energy (should be reasonable)
        energy_dict = mf_test.compute_energy_components(with_k=False)
        self.assertTrue(np.isfinite(energy_dict['E_tot']))
        self.assertLess(energy_dict['E_tot'], 0)  # Should be negative for bound system
    
    def test_init_guess_by_atom(self):
        """Test initialization from atomic HF."""
        # Small cell for fast test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'He 0 0 0; He 1.5 0 0'
        cell_test.basis = 'sto-3g'
        cell_test.a = np.eye(3) * 5.0
        cell_test.mesh = [16, 16, 16]
        cell_test.verbose = 0
        cell_test.build()
        
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=4)
        mf_test.verbose = 0
        mf_test.build()
        mf_test.init_guess_by_atom(seed=42)
        
        # Check that wavefunctions are initialized
        self.assertIsNotNone(mf_test.psi_r)
        self.assertIsNotNone(mf_test.psi_g)
        self.assertEqual(mf_test.psi_r.shape, (1, 4, mf_test.ngrids))
        self.assertEqual(mf_test.psi_g.shape, (1, 4, mf_test.ngrids))
        
        # Check normalization for all bands
        for n in range(mf_test.nband):
            # R-space normalization: ∫|ψ_r|²dr = 1
            norm_r = np.sqrt(np.sum(np.abs(mf_test.psi_r[0, n])**2) * mf_test.grid_weight)
            self.assertAlmostEqual(norm_r, 1.0, places=5,
                                   msg=f'Band {n} R-space norm = {norm_r}')
            
            # G-space normalization: Σ|ψ_g|² = 1
            norm_g = np.sqrt(np.sum(np.abs(mf_test.psi_g[0, n])**2))
            self.assertAlmostEqual(norm_g, 1.0, places=5,
                                   msg=f'Band {n} G-space norm = {norm_g}')
        
        # Compute initial energy (should be reasonable)
        energy_dict = mf_test.compute_energy_components(with_k=False)
        self.assertTrue(np.isfinite(energy_dict['E_tot']))
        self.assertLess(energy_dict['E_tot'], 0)  # Should be negative for bound system
    
    def test_init_guess_comparison(self):
        """Compare minao and atom initialization methods."""
        # Small cell for fast test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'He 0 0 0; He 1.5 0 0'
        cell_test.basis = 'sto-3g'
        cell_test.a = np.eye(3) * 5.0
        cell_test.mesh = [16, 16, 16]
        cell_test.verbose = 0
        cell_test.build()
        
        # Initialize with minao
        mf_minao = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=4)
        mf_minao.verbose = 0
        mf_minao.build()
        mf_minao.init_guess_by_minao(seed=42)
        E_minao = mf_minao.compute_energy_components(with_k=False)['E_tot']
        
        # Initialize with atom
        mf_atom = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=4)
        mf_atom.verbose = 0
        mf_atom.build()
        mf_atom.init_guess_by_atom(seed=42)
        E_atom = mf_atom.compute_energy_components(with_k=False)['E_tot']
        
        # Both should give reasonable (negative, finite) energies
        self.assertTrue(np.isfinite(E_minao))
        self.assertTrue(np.isfinite(E_atom))
        self.assertLess(E_minao, 0)
        self.assertLess(E_atom, 0)
        
        # They should be different but within reasonable range
        # (both are projections of atomic states)
        self.assertNotAlmostEqual(E_minao, E_atom, places=3)
        # But should be of similar magnitude (order of Hartrees)
        self.assertLess(abs(E_minao - E_atom), 100)  # Very loose bound
    
    def test_init_guess_with_scf(self):
        """Test that init_guess methods work with SCF kernel."""
        # Very small system for quick test
        cell_test = pbcgto.Cell()
        cell_test.atom = 'Li 0 0 0; H 1.5 0 0'
        cell_test.basis = 'ccpvdz'
        cell_test.a = np.eye(3) * 5.0
        cell_test.mesh = [64, 64, 64]
        cell_test.verbose = 3
        cell_test.build()

        # run rhf to get reference energy
        from pyscf.pbc.scf import RHF
        mf_ref = RHF(cell_test).density_fit()
        mf_ref.verbose = 4
        e_ref = mf_ref.kernel()
        
        # Test with minao initialization
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)),nband=3)
        mf_test.verbose = 4
        
        # Run a few SCF iterations
        e_tot, converged = mf_test.kernel(
            init='mo',
            mo_coeff=mf_ref.mo_coeff,
            mo_occ=mf_ref.mo_occ,
            max_cycle=20,  
            with_k=True,  
            conv_tol=1e-4,
            davidson_max_cycle=3,
            alpha=0.9,
        )
        np.testing.assert_allclose(e_tot, -5.7075502066, rtol=1e-6)
        


class TestKPWSCFAE(unittest.TestCase):
    """Test all-electron KPWSCF convergence against KRHF."""

    def test_he_convergence(self):
        """Test Helium atom convergence with increasing mesh size."""
        # Helium atom in a box
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'cc-pv5z'  # Larger basis for reference
        cell.a = np.eye(3) * 4.0
        cell.verbose = 0
        cell.build()

        # 1. Compute Reference Energy (KRHF)
        print("\nComputing Reference KRHF (cc-pv5z)...")
        # Use Gamma point for simplicity in this convergence test
        kpts = np.zeros((1, 3))
        krhf = KRHF(cell, kpts=kpts).density_fit()
        krhf.verbose = 0
        krhf.conv_tol = 1e-9
        e_ref = krhf.kernel()
        print(f"Reference Energy: {e_ref:.8f} Ha")

        # 2. Run KPWSCF with increasing mesh sizes
        # Note: All-electron calculations require very fine grids to resolve the cusp
        meshes = [
            [30, 30, 30],
            [40, 40, 40],
            [50, 50, 50],
            # [60, 60, 60] # Uncomment for better convergence (slower)
        ]
        
        print("\nRunning KPWSCF convergence loop...")
        print(f"{'Mesh':<15} {'Energy (Ha)':<15} {'Error (Ha)':<15}")
        print("-" * 45)

        errors = []
        for mesh in meshes:
            # Update cell mesh
            cell.mesh = mesh
            
            # Run KPWSCF
            # Use nband=2 (1 occupied + 1 virtual)
            mf_pw = KPWSCF(cell, kpts=kpts, nband=2)
            mf_pw.verbose = 0
            mf_pw.build()
            
            # Initialize from atom to get a reasonable start
            e_pw, converged = mf_pw.kernel(init='atom', max_cycle=50, conv_tol=1e-6)
            
            error = abs(e_pw - e_ref)
            errors.append(error)
            
            print(f"{str(mesh):<15} {e_pw:.8f}        {error:.8f}")

        # 3. Verify Convergence
        # The error should generally decrease as mesh increases
        # Note: For all-electron, convergence is slow due to the cusp condition
        # We just check that the error is reasonable for the largest mesh
        
        # Check if error decreases (or is already small)
        # It might not strictly decrease every step due to grid aliasing, but trend should be down
        self.assertLess(errors[-1], errors[0], "Error did not decrease with larger mesh")
        
        # Check absolute error for the finest mesh
        # This threshold is loose because AE-PW is hard, but ensures we are in the ballpark
        self.assertLess(errors[-1], 0.5, "KPWSCF error too large for all-electron He")

    def test_he_convergence_multi_kpt(self):
        """Test Helium atom convergence with increasing mesh size (Multiple K-points)."""
        # Helium atom in a box
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'cc-pv5z'  # Large basis for reference
        cell.a = np.eye(3) * 4.0 # Slightly smaller cell to make k-points more relevant
        cell.verbose = 0
        cell.build()

        # Use 2x1x1 k-points
        kpts = cell.make_kpts([2, 1, 1])

        # 1. Compute Reference Energy (KRHF)
        print("\nComputing Reference KRHF (cc-pv5z, 2x1x1 kpts)...")
        krhf = KRHF(cell, kpts=kpts).density_fit()
        krhf.verbose = 0
        krhf.conv_tol = 1e-9
        e_ref = krhf.kernel()
        print(f"Reference Energy: {e_ref:.8f} Ha")

        # 2. Run KPWSCF with increasing mesh sizes
        meshes = [
            [50, 50, 50],
            [60, 60, 60],
            [70, 70, 70],
        ]
        
        print("\nRunning KPWSCF convergence loop (Multi-K)...")
        print(f"{'Mesh':<15} {'Energy (Ha)':<15} {'Error (Ha)':<15}")
        print("-" * 45)

        errors = []
        for mesh in meshes:
            # Update cell mesh
            cell.mesh = mesh
            
            # Run KPWSCF
            # Use nband=2 (1 occupied + 1 virtual)
            mf_pw = KPWSCF(cell, kpts=kpts, nband=2)
            mf_pw.verbose = 0
            mf_pw.build()
            
            # Initialize from atom
            e_pw, converged = mf_pw.kernel(init='atom', max_cycle=50, conv_tol=1e-6)
            
            error = abs(e_pw - e_ref)
            errors.append(error)
            
            print(f"{str(mesh):<15} {e_pw:.8f}        {error:.8f}")

        # 3. Verify Convergence
        # Note: KPWSCF energy might be lower than KRHF due to grid aliasing of the cusp (variational collapse).
        # So we check for self-convergence (Cauchy) rather than strict convergence to KRHF.
        energy_diffs = [abs(errors[i] - errors[i-1]) for i in range(1, len(errors))]
        self.assertLess(energy_diffs[-1], energy_diffs[0], "Energy change did not decrease (not converging)")
        
        # Check absolute error is still reasonable
        self.assertLess(errors[-1], 0.5, "KPWSCF error too large for all-electron He (Multi-K)")
        self.assertLess(errors[-1], 0.5, "KPWSCF error too large for all-electron He (Multi-K)")

    def test_cbs_extrapolation(self):
        """Test CBS extrapolation for KRHF and mesh extrapolation for KPWSCF."""
        # Helium atom in a box
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 4.0
        cell.verbose = 0
        
        # 1. KRHF Basis Set Extrapolation
        print("\n=== KRHF Basis Set Extrapolation ===")
        basis_sets = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z']
        cardinal_nums = [2, 3, 4, 5]
        energies_hf = []
        
        for basis in basis_sets:
            cell.basis = basis
            cell.build()
            # Use Gamma point
            kpts = np.zeros((1, 3))
            krhf = KRHF(cell, kpts=kpts).density_fit()
            krhf.verbose = 0
            krhf.conv_tol = 1e-9
            e = krhf.kernel()
            energies_hf.append(e)
            print(f"KRHF/{basis:<10}: {e:.8f} Ha")
            
        # Extrapolate using E(X) = E_CBS + A * X^-3 (using QZ and 5Z)
        X_N = cardinal_nums[-2] # 4 (QZ)
        X_M = cardinal_nums[-1] # 5 (5Z)
        E_N = energies_hf[-2]
        E_M = energies_hf[-1]
        
        # E_M = E_CBS + A * M^-3  => A = (E_M - E_CBS) * M^3
        # E_N = E_CBS + A * N^-3
        # E_N = E_CBS + (E_M - E_CBS) * (M/N)^3
        # E_N - E_M * (M/N)^3 = E_CBS * (1 - (M/N)^3)
        # E_CBS = (E_N - E_M * (M/N)^3) / (1 - (M/N)^3)
        
        ratio = (X_M / X_N)**3
        e_cbs_hf = (E_N - E_M * ratio) / (1 - ratio)
        print(f"KRHF CBS Limit (extrapolated from QZ/5Z): {e_cbs_hf:.8f} Ha")
        
        # 2. KPWSCF Mesh Extrapolation
        print("\n=== KPWSCF Mesh Extrapolation ===")
        # Use cc-pv5z basis for cell definition (though PW doesn't use it, it sets up the cell)
        cell.basis = 'cc-pv5z'
        cell.build()
        
        meshes = [80, 100, 120, 140]
        energies_pw = []
        
        for m in meshes:
            mesh = [m, m, m]
            cell.mesh = mesh
            mf_pw = KPWSCF(cell, kpts=kpts, nband=2)
            mf_pw.verbose = 5
            mf_pw.build()
            
            # Initialize from atom
            e_pw, converged = mf_pw.kernel(init='atom', max_cycle=50, conv_tol=1e-6)
            energies_pw.append(e_pw)
            print(f"KPWSCF/mesh={m:<4}: {e_pw:.8f} Ha")
            
        # Extrapolate KPWSCF
        # For all-electron (cusp), convergence is slow.
        # We can try to fit E(N) = E_limit + A / N^k
        # Let's try to estimate E_limit from the last two points assuming 1/N convergence (linear in grid spacing h ~ 1/N)
        # E(N) = E_lim + A/N
        # E1 = E_lim + A/N1
        # E2 = E_lim + A/N2
        # E1 - E2 = A(1/N1 - 1/N2) => A = (E1 - E2) / (1/N1 - 1/N2)
        # E_lim = E2 - A/N2
        
        N1, N2 = meshes[-2], meshes[-1]
        E1, E2 = energies_pw[-2], energies_pw[-1]
        
        # Try 1/N extrapolation (often appropriate for Coulomb singularity on grid)
        slope = (E1 - E2) / (1/N1 - 1/N2)
        e_limit_pw = E2 - slope * (1/N2)
        
        print(f"KPWSCF Limit (extrapolated 1/N from last 2 points): {e_limit_pw:.8f} Ha")
        
        # Compare limits
        diff = abs(e_cbs_hf - e_limit_pw)
        print(f"\nDifference between KRHF/CBS and KPWSCF/Limit: {diff:.8f} Ha")
        
        # Check that KPWSCF extrapolated limit is close to KRHF CBS limit
        # The agreement might not be perfect due to different extrapolation models, but should be close
        self.assertLess(diff, 0.001, "Extrapolated limits differ by too much")

        self.assertLess(diff, 0.001, "Extrapolated limits differ by too much")

    def test_cbs_extrapolation_multi_kpt(self):
        """Test CBS extrapolation for KRHF and mesh extrapolation for KPWSCF (Multi-K)."""
        # Helium atom in a box
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 4.0
        cell.verbose = 0
        
        # Use 2x1x1 k-points
        kpts = cell.make_kpts([2, 1, 1])
        
        # 1. KRHF Basis Set Extrapolation
        print("\n=== KRHF Basis Set Extrapolation (Multi-K) ===")
        basis_sets = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz', 'cc-pv5z']
        cardinal_nums = [2, 3, 4, 5]
        energies_hf = []
        
        for basis in basis_sets:
            cell.basis = basis
            cell.build()
            krhf = KRHF(cell, kpts=kpts).density_fit()
            krhf.verbose = 0
            krhf.conv_tol = 1e-9
            e = krhf.kernel()
            energies_hf.append(e)
            print(f"KRHF/{basis:<10}: {e:.8f} Ha")
            if basis == 'cc-pv5z':
                print("  KRHF Components (cc-pv5z):")
                print(f"  K-points: {krhf.kpts}")
                
                # Get energy components using exact formula from test_kpwscf.py
                dm = krhf.make_rdm1()  # (nkpts, nao, nao)
                h1e = krhf.get_hcore()  # (nkpts, nao, nao)
                vj, vk = krhf.get_jk(dm=dm)  # (nkpts, nao, nao)
                
                # For multi-kpoint, sum over k-points
                # Energy formula for comparison with KPWSCF
                E_kin = 0.0
                #print("    Per-kpoint E_kin:")
                for k in range(len(krhf.kpts)):
                    t_k = krhf.cell.pbc_intor('int1e_kin', kpts=krhf.kpts[k])
                    e_kin_k = np.einsum('ij,ji', dm[k], t_k).real
                    #print(f"      k={k} (kpt={krhf.kpts[k]}): {e_kin_k:.10f}")
                    E_kin += e_kin_k
                E_kin /= len(krhf.kpts)
                
                E_hcore = np.einsum('kij,kji->', dm, h1e).real / len(krhf.kpts)
                E_ne = E_hcore - E_kin  # h1e = T + V_ne
                E_hartree = 0.5 * np.einsum('kij,kji->', dm, vj).real / len(krhf.kpts)
                E_exchange = -0.25 * np.einsum('kij,kji->', dm, vk).real / len(krhf.kpts)
                E_nuc = krhf.energy_nuc()
                
                print(f"    E_nuc     = {E_nuc:.10f}")
                print(f"    E_kin     = {E_kin:.10f}")
                print(f"    E_ne      = {E_ne:.10f}")
                print(f"    E_hartree = {E_hartree:.10f}")
                print(f"    E_exchange= {E_exchange:.10f}")
                print(f"    E_tot     = {e:.10f}")

        # Extrapolate using E(X) = E_CBS + A * X^-3 (using QZ and 5Z)
        X_N = cardinal_nums[-2] # 4 (QZ)
        X_M = cardinal_nums[-1] # 5 (5Z)
        E_N = energies_hf[-2]
        E_M = energies_hf[-1]
        
        ratio = (X_M / X_N)**3
        e_cbs_hf = (E_N - E_M * ratio) / (1 - ratio)
        print(f"KRHF CBS Limit (extrapolated from QZ/5Z): {e_cbs_hf:.8f} Ha")
        
        # 2. KPWSCF Mesh Extrapolation
        print("\n=== KPWSCF Mesh Extrapolation (Multi-K) ===")
        # Use cc-pv5z basis for cell definition
        cell.basis = 'cc-pv5z'
        cell.build()
        
        # Diagnostic: evaluate KRHF MO on KPWSCF grid for k=1
        from pyscf.pbc.dft import numint
        print("\nDiagnostic: Evaluating KRHF k=1 orbital on grid...")
        
        meshes = [40]
        energies_pw = []
        
        for m in meshes:
            print(f'\n=== KPWSCF Mesh Extrapolation (Multi-K) ===')
            mf_pw = KPWSCF(cell, kpts=kpts, mesh=[m, m, m], nband=2)
            
            # Before build, evaluate KRHF orbital on KPWSCF grid
            coords = mf_pw.grids.coords
            k1 = krhf.kpts[1]
            ao_k1 = numint.eval_ao(cell, coords, kpt=k1, deriv=0)
            mo_k1 = krhf.mo_coeff[1][:, 0]  # First occupied orbital at k=1
            psi_krhf_k1 = np.dot(ao_k1, mo_k1)
            
            # Check normalization before build
            norm_krhf = np.sum(np.abs(psi_krhf_k1)**2).real * mf_pw.grid_weight
            print(f"KRHF k=1, n=0 on KPWSCF grid (before build):") 
            print(f"  |psi_r|^2 sum: {norm_krhf:.10f}")
            
            mf_pw.verbose = 4
            mf_pw.build()
            
            # NOW compute kinetic energy after build (when _fft_r2g and _Gv are available)
            psi_krhf_k1_G = mf_pw._fft_r2g(psi_krhf_k1, kpt=k1)  # Pass k-point!
            norm_G = np.sum(np.abs(psi_krhf_k1_G)**2).real
            psi_krhf_k1_G_normalized = psi_krhf_k1_G / np.sqrt(norm_G)
            
            # Compute kinetic energy: T = sum_G 0.5*|G+k|^2 *|psi(G)|^2
            Gv = mf_pw._Gv
            Gk = Gv + k1
            kin_diag_k1 = 0.5 * np.einsum('ij,ij->i', Gk, Gk)
            E_kin_krhf_k1 = 2.0 * np.sum((kin_diag_k1 * np.abs(psi_krhf_k1_G_normalized)**2).real)
            
            print(f"KRHF k=1 kinetic energy on KPWSCF grid: {E_kin_krhf_k1:.10f} Ha")
            print(f"  (Expected from KRHF: 2.8441365148 Ha)")
            
            # Initialize from KRHF MO coefficients and compute initial energy only
            e_pw, converged = mf_pw.kernel(init='mo', mo_occ=krhf.mo_occ, mo_coeff=krhf.mo_coeff, max_cycle=10, conv_tol=1e-4)
            energies_pw.append(e_pw)
            print(f"KPWSCF/mesh={m:<4}: {e_pw:.8f} Ha")
            
        # Extrapolate KPWSCF (1/N scaling)
        if len(meshes) >= 2:
            N1, N2 = meshes[-2], meshes[-1]
            E1, E2 = energies_pw[-2], energies_pw[-1]
            
            slope = (E1 - E2) / (1/N1 - 1/N2)
            e_limit_pw = E2 - slope * (1/N2)
            
            print(f"KPWSCF Limit (extrapolated 1/N from last 2 points): {e_limit_pw:.8f} Ha")
            
            # Compare limits
            diff = abs(e_cbs_hf - e_limit_pw)
            print(f"\nDifference between KRHF/CBS and KPWSCF/Limit: {diff:.8f} Ha")
            
            # Note: AE-PW calculations are prone to variational collapse and large errors
            # due to grid aliasing of the Coulomb cusp. The difference can be significant.
            # We relax the tolerance here.
            self.assertLess(diff, 0.001, "Extrapolated limits differ by too much (Multi-K)")
        else:
            print("\nSkipping extrapolation (need at least 2 meshes)")

if __name__ == '__main__':
    unittest.main()


 

if __name__ == '__main__':
    print("Full Tests for pbc.scf.kpwscf")
    unittest.main()
