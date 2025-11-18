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
        cell_test.pseudo = 'gth-pade'
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
        #cell_test.pseudo = 'gth-pade'
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
        cell_test.atom = 'He 0 0 0; He 1.5 0 0'
        cell_test.basis = 'ccpvqz'
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
        mf_test = KPWSCF(cell_test, kpts=np.zeros((1, 3)), nband=2)
        mf_test.verbose = 4
        
        # Run a few SCF iterations
        e_tot, converged = mf_test.kernel(
            init='minao',
            max_cycle=10,  # Just a few iterations to test it works
            with_k=True,  # Hartree only for speed
            conv_tol=1e-4,
            davidson_max_cycle=5,
        )
        self.assertEqual(e_tot, -5.7075502066)
        


if __name__ == '__main__':
    print("Full Tests for pbc.scf.kpwscf")
    unittest.main()
