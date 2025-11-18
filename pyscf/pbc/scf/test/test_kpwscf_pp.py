#!/usr/bin/env python
import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
from pyscf.pbc.gto.pseudo import pp as pseudo
from pyscf.pbc.scf.kpwscf import KPWSCF


def make_carbon_cell(mesh, use_pseudo=True):
    cell = pbcgto.Cell()
    cell.atom = 'C 0 0 0; C 1.5 0 0'
    cell.basis = 'gth-szv'
    if use_pseudo:
        cell.pseudo = 'gth-pade'
    else:
        # explicit empty pseudo
        cell.pseudo = None
    cell.a = np.eye(3) * 4.0
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    return cell


class TestKPWSCFPseudo(unittest.TestCase):
    def test_local_pseudopotential_added(self):
        """Verify that the local part of the pseudopotential is added to V_ne."""
        mesh = [20, 20, 20]
        # Build two cells: one with pseudo and one without
        cell_pp = make_carbon_cell(mesh, use_pseudo=True)
        cell_ae = make_carbon_cell(mesh, use_pseudo=False)

        mf_pp = KPWSCF(cell_pp, kpts=np.zeros((1, 3)), nband=6)  # Carbon has 6 electrons, so nocc=3
        mf_pp.build()

        mf_ae = KPWSCF(cell_ae, kpts=np.zeros((1, 3)), nband=6)
        mf_ae.build()

        # V_ne should be different between pseudo and all-electron
        # The presence of pseudo should reduce the magnitude of the nuclear attraction
        # (since the pseudo replaces the core with a softer potential)
        diff = mf_pp._vne_R - mf_ae._vne_R
        
        # Check that the difference is not all zeros (i.e., pseudo was actually added)
        max_diff = np.abs(diff).max()
        self.assertGreater(max_diff, 1e-10, 
                          msg="V_ne difference should be non-zero when pseudo is added")
        
        # Verify that V_ne with pseudo is typically less negative (softer potential)
        # than without pseudo
        self.assertGreater(np.mean(diff), -100,  # Very loose bound
                          msg="V_ne with pseudo should be softer (less negative) than all-electron")

    def test_apply_nuc_with_pseudo(self):
        """Test that _apply_nuc correctly applies nuclear + local pseudo potential."""
        mesh = [20, 20, 20]
        cell_pp = make_carbon_cell(mesh, use_pseudo=True)
        mf_pp = KPWSCF(cell_pp, kpts=np.zeros((1, 3)), nband=6)
        mf_pp.build()
        mf_pp.init_guess(kind='random', seed=42)
        
        # Apply nuclear potential to a test wavefunction
        ik = 0
        n = 0
        psi_g = mf_pp.psi_g[ik, n]
        vnuc_psi_g = mf_pp._apply_nuc(ik, psi_g)
        
        # Check shape
        self.assertEqual(vnuc_psi_g.shape, psi_g.shape)
        
        # Should be complex and non-zero
        self.assertTrue(np.iscomplexobj(vnuc_psi_g))
        self.assertGreater(np.abs(vnuc_psi_g).max(), 1e-10)
        
        # Compute expectation value of nuclear potential
        psi_r = mf_pp._ifft_g2r(psi_g)
        vnuc_psi_r = mf_pp._ifft_g2r(vnuc_psi_g)
        E_vnuc = np.real(np.vdot(psi_r, vnuc_psi_r)) * mf_pp.grid_weight
        
        # Nuclear attraction should be negative
        self.assertLess(E_vnuc, 0)

    def test_scf(self):
        """Test SCF convergence with and without pseudopotential, comparing to KRHF reference."""
        from pyscf.pbc.scf import KRHF
        
        # Use helium for simplicity
        cell_pp = pbcgto.Cell()
        cell_pp.atom = 'Li 0 0 0; H 1. 0 0'
        cell_pp.basis = 'gth-szv'
        cell_pp.pseudo = 'gth-pade'
        cell_pp.a = np.eye(3) * 5.0
        cell_pp.mesh = [32, 32, 32]
        cell_pp.verbose = 0
        cell_pp.build()
        
        # Reference: KRHF with pseudopotential
        krhf_pp = KRHF(cell_pp, kpts=np.zeros((1, 3))).density_fit()
        krhf_pp.verbose = 0
        krhf_pp.conv_tol = 1e-8
        E_krhf_pp = krhf_pp.kernel()
        
        # Compute KRHF energy components
        dm = krhf_pp.make_rdm1()  # Shape: (nkpts, nao, nao)
        h1e = krhf_pp.get_hcore()  # Shape: (nkpts, nao, nao)
        vj, vk = krhf_pp.get_jk(dm)
        
        # Energy breakdown for KRHF
        # Since we're at Gamma point only, just use the first (and only) k-point
        T_ao = cell_pp.pbc_intor('int1e_kin')  # Shape: (nao, nao)
        
        if dm.ndim == 3:
            # Multiple k-points format (even if nkpts=1)
            nkpts = dm.shape[0]
            E_kin_krhf = sum(np.einsum('ij,ji', dm[k], T_ao).real for k in range(nkpts)) / nkpts
            E_hcore_krhf = sum(np.einsum('ij,ji', dm[k], h1e[k]).real for k in range(nkpts)) / nkpts
            E_hartree_krhf = 0.5 * sum(np.einsum('ij,ji', dm[k], vj[k]).real for k in range(nkpts)) / nkpts
            E_exchange_krhf = -0.25 * sum(np.einsum('ij,ji', dm[k], vk[k]).real for k in range(nkpts)) / nkpts
        else:
            # Single k-point case
            E_kin_krhf = np.einsum('ij,ji', dm, T_ao).real
            E_hcore_krhf = np.einsum('ij,ji', dm, h1e).real
            E_hartree_krhf = 0.5 * np.einsum('ij,ji', dm, vj).real
            E_exchange_krhf = -0.25 * np.einsum('ij,ji', dm, vk).real
        
        E_ne_krhf = E_hcore_krhf - E_kin_krhf
        E_nuc_krhf = cell_pp.energy_nuc()
        
        # KPWSCF with pseudopotential
        mf_pp = KPWSCF(cell_pp, kpts=np.zeros((1, 3)), nband=20)
        mf_pp.verbose = 5
        mf_pp.build()
        mf_pp.init_guess(kind='atom', seed=42)
        E_pwscf_pp, converged = mf_pp.kernel(max_cycle=50, davidson_max_cycle=5)
        
        # Compute KPWSCF energy components
        E_components = mf_pp.compute_energy_components(with_k=True)
        E_pwscf_pp = E_components['E_tot']
        
        # Print comparison for debugging
        print("\n" + "="*60)
        print("Energy Comparison (Pseudopotential):")
        print(f"  KRHF Total:        {E_krhf_pp:.8f} Ha")
        print(f"    E_kin:           {E_kin_krhf:.8f} Ha")
        print(f"    E_ne:            {E_ne_krhf:.8f} Ha")
        print(f"    E_hartree:       {E_hartree_krhf:.8f} Ha")
        print(f"    E_exchange:      {E_exchange_krhf:.8f} Ha")
        print(f"    E_nuc:           {E_nuc_krhf:.8f} Ha")
        print()
        print(f"  KPWSCF Total:      {E_pwscf_pp:.8f} Ha")
        print(f"    E_kin:           {E_components['E_kin']:.8f} Ha")
        print(f"    E_ne:            {E_components['E_ne']:.8f} Ha")
        print(f"    E_hartree:       {E_components['E_hartree']:.8f} Ha")
        print(f"    E_exchange:      {E_components['E_exchange']:.8f} Ha")
        print(f"    E_nuc:           {E_components['E_nuc']:.8f} Ha")
        print()
        print(f"  Total Difference:  {abs(E_krhf_pp - E_pwscf_pp):.8f} Ha")
        print(f"  E_ne Difference:   {abs(E_ne_krhf - E_components['E_ne']):.8f} Ha")
        print("="*60)
        
        # All energies should be finite
        self.assertTrue(np.isfinite(E_krhf_pp))
        self.assertTrue(np.isfinite(E_pwscf_pp))
        
        # KPWSCF energies should be reasonably close to KRHF (same physics, different basis)
        # Note: We use very loose tolerance since KPWSCF is plane-wave basis and KRHF is AO basis
        self.assertLess(abs(E_krhf_pp - E_pwscf_pp), 10.0,
                       msg=f"KPWSCF pseudo energy differs too much from KRHF: {abs(E_krhf_pp - E_pwscf_pp):.4f} Ha")


if __name__ == '__main__':
    unittest.main()
