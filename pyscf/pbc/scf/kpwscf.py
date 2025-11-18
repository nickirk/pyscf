#!/usr/bin/env python
# Copyright 2025 The PySCF Developers.

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools as pbctools
from pyscf.pbc.dft import gen_grid as pbc_gen_grid
from pyscf.pbc.scf.khf import KRHF
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.lib.kpts import KPoints


class KPWSCF(lib.StreamObject):
    """K-point plane-wave SCF (initial milestone: T + V_ne only).

    This class operates directly in the plane-wave/Bloch representation on a
    uniform FFT grid defined by cell.mesh, without AO basis functions.

    """

    def __init__(self, cell, kpts=None, nband=None, mesh=None,
                 verbose=None, stdout=None):
        self.cell = cell
        self.stdout = stdout or cell.stdout
        self.verbose = logger.NOTE if verbose is None else verbose
        self.max_memory = cell.max_memory

        # K-points
        if kpts is None:
            kpts = np.zeros((1, 3))
        if isinstance(kpts, KPoints):
            kpts = kpts.kpts
        self.kpts = np.asarray(kpts, dtype=float)
        self.nk = len(self.kpts)

        # Mesh / grids (prefer kinetic_cutoff if provided, else explicit mesh, else cell.mesh)
        #self.mesh = self._select_mesh(mesh)
        # Use UniformGrids for consistent grid weights
        self.grids = pbc_gen_grid.UniformGrids(self.cell)
        self.mesh = self.grids.mesh
        logger.debug(self, 'Using mesh %s for FFT grids', str(self.mesh))
        #self.grids.mesh = np.asarray(self.mesh, dtype=int)
        self.ngrids = int(np.prod(self.grids.mesh))
        self.vol = float(self.cell.vol)
        self.grid_weight = self.grids.weights[0] if self.ngrids > 0 else 0.0  # ∫f ≈ weight * sum(f)

        # Number of bands
        nelec = sum(self.cell.nelec) if hasattr(self.cell, 'nelec') else self.cell.nelectron
        if nband is None:
            # Heuristic: filled + 20% extra, at least 1
            nocc = max(1, int(np.ceil(nelec/2)))
            nband = max(nocc, int(np.ceil(nocc * 1.2)))
        self.nband = int(nband)
        self.nocc = min(int(np.ceil(nelec/2)), self.nband)

        # Wavefunctions on grid (complex)
        self.psi_r = None  # shape (nk, nband, ngrids)
        self.psi_g = None  # shape (nk, nband, ngrids)
        self.mo_energy = None  # shape (nk, nband)

        # Cached operators
        self._Gv_cache = None       # (ngrids, 3)
        self._kin_diag = None       # (nk, ngrids)
        self._vne_R = None          # (ngrids,)
        self._coulG0 = None         # (ngrids,) Coulomb kernel at k=0

    def _select_mesh(self, mesh):
        """Choose FFT mesh from inputs or cell defaults.

        Priority: explicit mesh > kinetic_cutoff > cell.ke_cutoff > cell.mesh.
        """
        if mesh is not None:
            return tuple(np.asarray(mesh, dtype=int))
        # Use kinetic cutoff if provided
        # Fall back to cell.ke_cutoff if available
        if getattr(self.cell, 'ke_cutoff', None) is not None:
            Ecut = float(np.min(np.atleast_1d(self.cell.ke_cutoff)))
            m = self.cell.cutoff_to_mesh(Ecut)
            return tuple(np.asarray(m, dtype=int))
        # Default to cell.mesh
        return tuple(np.asarray(self.cell.mesh, dtype=int))

    @property
    def _Gv(self):
        """G-vectors on the FFT mesh (ngrids, 3)."""
        if self._Gv_cache is None:
            self._Gv_cache = self.cell.get_Gv(self.mesh)
        return self._Gv_cache

    # ----------------------------- utilities ----------------------------- #
    def _fft_r2g(self, psi_r_block):
        """FFT from real-space to G-space with consistent normalization.
        
        Convention:
        - R-space: ∫|ψ_r|²dr = Σ|ψ_r|² × (V/N) = 1
        - G-space: Σ|ψ_g|² = 1
        
        Numpy FFT Parseval: Σ|ψ_r|² = (1/N) Σ|FFT(ψ_r)|²
        
        Given Σ|ψ_r|² = N/V, we have Σ|FFT(ψ_r)|² = N²/V
        To normalize to Σ|ψ_g|² = 1, scale by: 1/√(N²/V) = √(V)/N
        
        psi_r_block: (nband, ngrids) or (ngrids,)
        Returns: (nband, ngrids) or (ngrids,) complex in G-space
        """
        if psi_r_block.ndim == 1:
            psi_r_2d = psi_r_block.reshape(1, -1)
            psi_g_2d = pbctools.fft(psi_r_2d, self.mesh)
            psi_g_2d *= np.sqrt(self.vol) / self.ngrids
            return psi_g_2d[0]
        else:
            psi_g = pbctools.fft(psi_r_block, self.mesh)
            psi_g *= np.sqrt(self.vol) / self.ngrids
            return psi_g

    def _ifft_g2r(self, psi_g_block):
        """IFFT from G-space to real-space with consistent normalization.
        
        Convention:
        - G-space: Σ|ψ_g|² = 1
        - R-space: ∫|ψ_r|²dr = Σ|ψ_r|² × (V/N) = 1
        
        Numpy IFFT Parseval: Σ|IFFT(ψ_g)|² = (1/N) Σ|ψ_g|²
        
        Given Σ|ψ_g|² = 1, we have Σ|IFFT(ψ_g)|² = 1/N
        To get Σ|ψ_r|² = N/V, scale by: √((N/V)/(1/N)) = √(N²/V) = N/√V
        
        psi_g_block: (nband, ngrids) or (ngrids,)
        Returns: (nband, ngrids) or (ngrids,) complex in real-space
        """
        if psi_g_block.ndim == 1:
            psi_g_2d = psi_g_block.reshape(1, -1)
            psi_r_2d = pbctools.ifft(psi_g_2d, self.mesh)
            psi_r_2d *= np.sqrt(self.ngrids)
            return psi_r_2d[0]
        else:
            psi_r = pbctools.ifft(psi_g_block, self.mesh)
            psi_r *= np.sqrt(self.ngrids)
            return psi_r

    def build(self):
        """Sanity checks and operator precompute."""
        log = logger.new_logger(self, self.verbose)
        # FFTDF supports 2D/3D with finite-vacuum; follow the same constraint
        if (self.cell.dimension < 2 or
            (self.cell.dimension == 2 and self.cell.low_dim_ft_type == 'inf_vacuum')):
            raise RuntimeError('PW-FFT solver requires 2D/3D periodicity with finite vacuum. '
                               'For 0D/1D or 2D inf_vacuum, use AFTDF/MDF paths instead.')

        log.info('KPWSCF: nk=%d, mesh=%s (ngrids=%d), nband=%d (nocc=%d)'
                 , self.nk, self.mesh, self.ngrids, self.nband, self.nocc)
        # Log implied kinetic cutoff for visibility
        try:
            ke_from_mesh = pbctools.mesh_to_cutoff(self.cell.lattice_vectors(), np.asarray(self.mesh))
            log.info('KPWSCF: implied kinetic cutoff (Eh) from mesh = %s (min=%.3f)',
                     str(np.asarray(ke_from_mesh)), float(np.min(ke_from_mesh)))
        except Exception:
            pass
        self._precompute_kinetic()
        self._build_vne()
        return self

    def _precompute_kinetic(self):
        """Diagonal kinetic operator in G for each k: 0.5*|G+k|^2."""
        Gv = self._Gv
        self._kin_diag = np.empty((self.nk, self.ngrids), dtype=float)
        for ik, kpt in enumerate(self.kpts):
            Gk = Gv + kpt  # broadcast
            self._kin_diag[ik] = 0.5 * np.einsum('ij,ij->i', Gk, Gk)

    def _build_vne(self):
        """Build nuclear-electron potential V_ne in real space."""
        log = logger.new_logger(self, self.verbose)
        mesh = self.mesh
        cell = self.cell
        
        # Get structure factors and atomic charges
        SI = cell.get_SI(mesh=mesh)  # (natm, ngrids) complex
        charge = cell.atom_charges()  # Nuclear charges (positive)
        rhoG = charge @ SI  # (ngrids,) nuclear charge density in G-space
        
        # Get Coulomb kernel and compute V_ne in G-space
        # V_ne(G) = -4π*Z(G)/|G|^2 (negative for attractive potential)
        Gv = self.cell.get_Gv()
        coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
        self._coulG0 = coulG  # Cache for later use
        vneG = -rhoG * coulG  # (ngrids,) negative sign for attractive potential
        
        # Transform to real space
        self._vne_R = pbctools.ifft(vneG, mesh).real
        log.debug1('Built V_ne on grid; min/max %.6g / %.6g', 
                   self._vne_R.min(), self._vne_R.max())
        return self._vne_R

    def _orthonormalize_block(self, psi_block):
        """Orthonormalize a block of wavefunctions.
        
        Args:
            psi_block: (nband, ngrids) complex array
        Returns:
            Orthonormalized psi_block with same shape
        """
        nb, ngr = psi_block.shape
        # Compute overlap matrix S_ij = <ψ_i|ψ_j> = Σ_r ψ_i*(r) ψ_j(r) * weight
        S = np.dot(psi_block.conj(), psi_block.T) * self.grid_weight
        
        # Löwdin orthonormalization: ψ' = S^(-1/2) ψ
        e, U = np.linalg.eigh(S)
        e = np.maximum(e, 1e-14)  # Avoid numerical issues
        S_mhalf = U @ (np.diag(1.0 / np.sqrt(e)) @ U.conj().T)
        
        return S_mhalf @ psi_block


    # ----------------------------- initialization ----------------------------- #
    def init_guess(self, kind='random', seed=1):
        """Initialize ψ on the real-space grid and orthonormalize (per k).

        kind: 'random' | 'pw' (lowest-|G| plane-waves per band) | 'khf-1e' | 'atom'
        """
        rng = np.random.default_rng(seed)
        nk, nb, ngr = self.nk, self.nband, self.ngrids
        self.psi_r = np.empty((nk, nb, ngr), dtype=np.complex128)
        if kind.lower().startswith('rand'):
            for ik in range(nk):
                # random complex with small imaginary part
                x = rng.standard_normal((nb, ngr))
                y = rng.standard_normal((nb, ngr))
                psi = (x + 1j * 0.1 * y)
                psi = self._orthonormalize_block(psi)
                self.psi_r[ik] = psi
        elif kind.lower().startswith('pw'):
            # Fill with single-G plane-waves of lowest kinetic energy at each k
            # Build kinetic ordering per k, select nb unique G indices
            for ik in range(nk):
                idx = np.argsort(self._kin_diag[ik])[:nb]
                psi = np.zeros((nb, ngr), dtype=np.complex128)
                for n, gi in enumerate(idx):
                    # exp(i G·r) has inverse FFT delta at G; in R-grid representation,
                    # we set in G-space and ifft to R for a smooth initial ψ.
                    pw_g = np.zeros((1, ngr), dtype=np.complex128)
                    pw_g[0, gi] = 1.0
                    psi[n] = self._ifft_g2r(pw_g)[0]
                psi = self._orthonormalize_block(psi)
                self.psi_r[ik] = psi
        elif kind.lower() in ('khf-1e', 'khf1e', 'hcore'):
            # Use AO-based k-point hcore diagonalization to seed ψ on uniform grids
            # 1) Build KRHF helper to get H_core and S in AO basis
            khf = KRHF(self.cell, self.kpts)
            Hk = khf.get_hcore()
            Sk = khf.get_ovlp()
            # 2) Diagonalize generalized eigenproblem per k
            mo_coeff = []
            for ik in range(nk):
                # scipy generalized Hermitian eigenproblem
                e, c = np.linalg.eigh(Hk[ik], Sk[ik])
                # sort by energy
                idx = np.argsort(e)
                mo_coeff.append(c[:, idx])
            # 3) Evaluate AO on our uniform grids and form ψ = AO @ C
            #    Use PySCF KNumInt to get AO(collocation) per k
            coords = self.grids.coords  # (ngr, 3)
            ni = pbc_numint.KNumInt()
            ao_list = ni.eval_ao(self.cell, coords, kpts=self.kpts)  # list of (ngr, nao)
            for ik in range(nk):
                ao = np.asarray(ao_list[ik])  # (ngr, nao)
                Ck = mo_coeff[ik][:, :min(nb, mo_coeff[ik].shape[0])]
                psi = (ao @ Ck).T  # (m<=nb, ngr)
                mcur = psi.shape[0]
                if mcur < nb:
                    add = rng.standard_normal((nb-mcur, ngr)) + 1j*0.1*rng.standard_normal((nb-mcur, ngr))
                    psi = np.vstack([psi, add])
                elif mcur > nb:
                    psi = psi[:nb]
                psi = self._orthonormalize_block(psi)
                self.psi_r[ik] = psi
        elif kind.lower() in ('atom', 'khf-atom', 'atomic'):
            # Superposition-of-atomic density matrix in AO basis -> natural orbitals
            khf = KRHF(self.cell, self.kpts)
            dm_kpts = khf.init_guess_by_atom(self.cell, self.kpts)
            Sk = khf.get_ovlp()
            coords = self.grids.coords
            ni = pbc_numint.KNumInt()
            ao_list = ni.eval_ao(self.cell, coords, kpts=self.kpts)
            for ik in range(nk):
                D = np.asarray(dm_kpts[ik])
                S = np.asarray(Sk[ik])
                # Compute natural orbitals from generalized eigenproblem D C = S C n
                # Use S^(1/2) transform to standard eigenproblem
                se, U = np.linalg.eigh(S)
                se = np.maximum(se, 1e-12)
                S_half = U @ (np.sqrt(se)[:, None] * U.conj().T)
                S_mhalf = U @ ((1.0/np.sqrt(se))[:, None] * U.conj().T)
                B = S_half.conj().T @ D @ S_half
                w, W = np.linalg.eigh((B + B.conj().T) * 0.5)
                # Natural orbital coefficients in AO basis
                Cnat = S_mhalf @ W  # columns are orbitals
                # Take the most occupied orbitals first
                idx = np.argsort(w)[::-1]
                m_take = min(nb, Cnat.shape[1])
                Ck = Cnat[:, idx[:m_take]]
                # Evaluate on grid
                ao = np.asarray(ao_list[ik])  # (ngr, nao)
                psi = (ao @ Ck).T  # (m_take, ngr)
                mcur = psi.shape[0]
                if mcur < nb:
                    add = rng.standard_normal((nb-mcur, ngr)) + 1j*0.1*rng.standard_normal((nb-mcur, ngr))
                    psi = np.vstack([psi, add])
                elif mcur > nb:
                    psi = psi[:nb]
                psi = self._orthonormalize_block(psi)
                self.psi_r[ik] = psi
        else:
            raise ValueError(f'Unknown init_guess kind: {kind}')
        # Keep G-space copy
        logger.debug(self, 'Initialized wavefunctions in R-space with kind=%s', kind)
        logger.debug(self, 'Wavefunction norms after initialization (R-space integral):')
        for ik in range(nk):
            for n in range(nb):
                norm_r = np.sqrt(np.sum(np.abs(self.psi_r[ik, n])**2) * self.grid_weight)
                logger.debug(self, f'  k-point {ik} band {n}: ∫|ψ_r|²dr = {norm_r:.6e}')
        self.psi_g = np.empty_like(self.psi_r)
        for ik in range(nk):
            self.psi_g[ik] = self._fft_r2g(self.psi_r[ik])
        logger.debug(self, 'Wavefunction norms after FFT to G-space (L2 norm):')
        for ik in range(nk):
            for n in range(nb):
                norm_g = np.sqrt(np.sum(np.abs(self.psi_g[ik, n])**2))
                logger.debug(self, f'  k-point {ik} band {n}: Σ|ψ_g|² = {norm_g:.6e}')
        
        # report the initial energy expectation values, using compute_energy_components
        logger.info(self, 'Initial energy expectation values per band (Eh):')
        energy_dict = self.compute_energy_components(with_k=True)

        return self

    def get_density_r(self):
        """Total electron density in real space from current ψ (closed shell).
        
        ρ(r) = Σ_{k,n≤nocc} f_kn |ψ_kn(r)|^2 with f_kn = 2/nk for closed shell.
        Returns (ngrids,) real array.
        """
        if self.psi_r is None:
            raise RuntimeError('Call init_guess() first')
        rho_r = np.zeros(self.ngrids, dtype=float)
        occ_weight = 2.0 / self.nk  # For closed shell, each orbital has weight 2
        for ik in range(self.nk):
            psi_r = self.psi_r[ik, :self.nocc]  # (nocc, ngr)
            rho_r += occ_weight * (np.abs(psi_r)**2).sum(axis=0)
        return rho_r

    def get_density_G(self):
        """Total electron density in G-space from current ψ̃ (closed shell).

        ρ̃(G) = FFT[ρ(r)]
        Returns (ngrids,) complex array.
        """
        rho_r = self.get_density_r()
        return self._fft_r2g(rho_r)


    def apply_hamiltonian(self, ik,  psi_nk_g, with_j=True, with_k=True):
        """Apply full Hamiltonian to a single wavefunction.
        
        H = T + V_ne + V_H + V_X (for HF)
        
        Following the pseudocode:
        1. Apply kinetic energy (diagonal in G-space)
        2. Apply local potential V_ne (nuclear-electron)
        3. Apply Hartree potential V_H (if with_j=True)
        4. Apply exchange potential V_X (if with_k=True)
        
        Args:
            ik: k-point index
            n: band index
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
            rho_total_r: (ngrids,) total electron density in real space
            with_j: whether to include Hartree term
            with_k: whether to include exchange term
        Returns:
            H|ψ⟩ in G-space (ngrids,)
        """
        H_psi_g = np.zeros_like(psi_nk_g)
        
        # 1. Apply Kinetic Energy (Diagonal in G-space)
        # T = 0.5 * |k + G|^2
        t_g = self._kin_diag[ik]  # (ngrids,)
        H_psi_g += t_g * psi_nk_g
        
        # 2. Apply Local Potential V_ne (nuclear-electron attraction)
        psi_nk_r = self._ifft_g2r(psi_nk_g)
        vne_psi_r = self._vne_R * psi_nk_r
        H_psi_g += self._fft_r2g(vne_psi_r)
        rho_total_r = self.get_density_r()  # Update density from current ψ
        # 3. Apply Hartree Potential V_H (electron-electron repulsion, direct term)
        if with_j:
            # V_H(r) from total density (precomputed)
            rho_g = self._fft_r2g(rho_total_r)
            coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
            vH_g = coulG * rho_g
            vH_r = self._ifft_g2r(vH_g).real
            vH_psi_r = vH_r * psi_nk_r
            H_psi_g += self._fft_r2g(vH_psi_r)
        
        # 4. Apply Exchange Potential V_X (electron-electron repulsion, exchange term)
        if with_k:
            H_psi_g += self._apply_k(ik, psi_nk_g)
        
        return H_psi_g



    def _apply_j(self, psi_nk_g, rho_total_r):
        """Apply Hartree potential V_H to wavefunction.
        
        V_H is computed from total density once per iteration and applied to all bands.
        
        Args:
            psi_nk_g: (ngrids,) wavefunction in G-space for single band
            rho_total_r: (ngrids,) total electron density in real space
        Returns:
            V_H * psi in G-space (ngrids,)
        """
        # Transform density to G-space
        rho_g = self._fft_r2g(rho_total_r)
        
        # Apply Coulomb kernel in G-space: V_H(G) = (4π/|G|^2) * ρ(G)
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        vH_g = coulG * rho_g  # (ngrids,)
        
        # Transform V_H to real space
        vH_r = self._ifft_g2r(vH_g).real
        
        # Transform wavefunction to real space, multiply, and transform back
        psi_r = self._ifft_g2r(psi_nk_g)
        vH_psi_r = vH_r * psi_r
        vH_psi_g = self._fft_r2g(vH_psi_r)
        
        return vH_psi_g

    def compute_energy_components(self, with_k=True):
        """Compute energy components efficiently using expectation values.
        
        E_HF = E_kin + E_ne + E_hartree + E_exchange + E_nuc
        
        Uses: E_op = Σ_{k,n≤nocc} f_kn <ψ_kn|O|ψ_kn>
        where f_kn = 2/nk for closed shell.
        
        Args:
            rho_r: total electron density in real space
            with_k: whether to include exchange
            
        Returns:
            dict with energy components: 
            {E_kin, E_ne, E_hartree, E_exchange, E_nuc, E_tot}
        """
        occ_weight = 2.0 / self.nk  # Closed shell
        
        # Sanity check on density
        
        # Precompute Hartree potential from density
        rho_g = self.get_density_G()
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        vH_g = coulG * rho_g
        vH_r = self._ifft_g2r(vH_g).real
        
        
        # Initialize energy components
        E_kin = 0.0
        E_ne = 0.0
        E_hartree = 0.0
        E_exchange = 0.0
        
        # Debug: check if we have occupied orbitals
        if self.nocc == 0:
            raise RuntimeError("No occupied orbitals (self.nocc = 0)!")
        
        # Loop over k-points and occupied orbitals
        for ik in range(self.nk):
            for n in range(self.nocc):
                logger.debug(self, f"Computing energy contributions for k-point {ik}, band {n}")
                psi_g = self.psi_g[ik, n].copy()
                psi_r = self._ifft_g2r(psi_g)
                
                # Verify normalization
                norm_g = np.sqrt(np.sum(np.abs(psi_g)**2))
                norm_r_integral = np.sqrt(np.sum(np.abs(psi_r)**2) * self.grid_weight)
                logger.debug(self, f"  Σ|ψ_g|² = {norm_g:.6e}, ∫|ψ_r|²dr = {norm_r_integral:.6e}")
                
                # Kinetic energy: <ψ|T|ψ> = Σ_G T(G) |ψ(G)|²
                # With L2 normalization (Σ|ψ_g|²=1), we compute directly without grid_weight
                t_diag = self._kin_diag[ik]
                E_kin_contrib = occ_weight * np.sum((t_diag * np.abs(psi_g)**2).real)
                E_kin += E_kin_contrib
                logger.debug(self, f"  E_kin contribution: {E_kin_contrib:.6e}")
                
                # Nuclear-electron energy: <ψ|V_ne|ψ> = ∫ ψ*(r) V_ne(r) ψ(r) dr
                #                                      = Σ_r ψ*(r) V_ne(r) ψ(r) * (V/N)
                vne_psi_r = self._vne_R * psi_r
                E_ne_contrib = occ_weight * np.sum(psi_r.conj() * vne_psi_r).real * self.grid_weight
                E_ne += E_ne_contrib
                logger.debug(self, f"  E_ne contribution: {E_ne_contrib:.6e}")
                
                # Hartree energy contribution: <ψ|V_H|ψ> = ∫ ψ*(r) V_H(r) ψ(r) dr
                vH_psi_r = vH_r * psi_r
                E_h_contrib = occ_weight * np.sum(psi_r.conj() * vH_psi_r).real * self.grid_weight
                E_hartree += E_h_contrib
                logger.debug(self, f"  E_hartree contribution: {E_h_contrib:.6e}, {self.grid_weight=:.6e}")
                
                # Exchange energy: <ψ|K|ψ> = ∫ ψ*(r) K[ψ](r) dr
                if with_k:
                    K_psi_g = self._apply_k(ik, psi_g)
                    K_psi_r = self._ifft_g2r(K_psi_g)
                    E_x_contrib = 0.5*occ_weight * np.sum(psi_r.conj() * K_psi_r).real * self.grid_weight
                    logger.debug(self, f"  E_exchange contribution: {E_x_contrib:.6e}")
                    E_exchange += E_x_contrib
        
        
        # Nuclear-nuclear repulsion (Ewald energy for periodic systems)
        if hasattr(self.cell, 'energy_nuc'):
            E_nuc = self.cell.energy_nuc()
        else:
            E_nuc = 0.0
        
        # Total energy
        E_tot = E_kin + E_ne + E_hartree + E_exchange + E_nuc
        
        return {
            'E_kin': E_kin,
            'E_ne': E_ne,
            'E_hartree': E_hartree,
            'E_exchange': E_exchange,
            'E_nuc': E_nuc,
            'E_tot': E_tot
        }

    def _apply_k(self, ik, psi_nk_g):
        """Apply exchange operator K to wavefunction (Gamma-point only for now).
        
        K|ψ_n⟩ = -Σ_{n'∈occ} ∫ dr' ψ_{n'}*(r') ψ_n(r') / |r-r'| ψ_{n'}(r)
        
        In Fourier space:
        1. Compute pair density ρ_{nn'}(r) = ψ_n*(r) ψ_{n'}(r)
        2. Solve Poisson in G-space: V_{nn'}(G) = (4π/|G|^2) FFT[ρ_{nn'}(r)]
        3. Transform back: V_{nn'}(r) = IFFT[V_{nn'}(G)]
        4. Accumulate: K|ψ_n⟩ += -V_{nn'}(r) * ψ_{n'}(r)
        
        Args:
            ik: k-point index (currently assumes Gamma point, ik=0)
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
        Returns:
            K|ψ_n⟩ in G-space (ngrids,)
        """
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh, exxdiv='ewald')
        
        # Transform input wavefunction to real space
        psi_n_r = self._ifft_g2r(psi_nk_g)
        
        # Initialize exchange contribution
        k_psi_g = np.zeros_like(psi_nk_g)
        
        # Loop over all occupied orbitals
        for n_occ in range(self.nocc):
            # Get occupied orbital in real space
            psi_occ_g = self.psi_g[ik, n_occ]  # (ngrids,)
            psi_occ_r = self._ifft_g2r(psi_occ_g)
            
            # Compute pair density: ρ_{n,n'}(r) = ψ_{n'}*(r) * ψ_n(r)
            rho_ij_r = psi_occ_r.conj() * psi_n_r  # (ngrids,)
            
            # Transform to G-space
            rho_ij_g = self._fft_r2g(rho_ij_r)  # (ngrids,)
            
            # Apply Coulomb kernel: V_{n,n'}(G) = (4π/|G|^2) * ρ_{n,n'}(G)
            V_ij_g = coulG * rho_ij_g  # (ngrids,)
            
            # Transform back to real space
            V_ij_r = self._ifft_g2r(V_ij_g)  # (ngrids,)
            
            # Accumulate exchange: -V_{n,n'}(r) * ψ_{n'}(r)
            k_psi_r = -V_ij_r * psi_occ_r  # (ngrids,)
            k_psi_g += self._fft_r2g(k_psi_r)  # (ngrids,)
        
        return k_psi_g*0.5


    def _scf_davidson(self, ik, psi_g_init, with_k=True, tol=1e-6, max_cycle=30):
        """Custom Davidson diagonalization with SCF density update.
        
        Unlike standard Davidson, this updates the density and Hamiltonian
        during the Davidson iterations to maintain self-consistency.
        
        Args:
            ik: k-point index
            psi_g_init: Initial guess wavefunctions (nband, ngrids) in G-space
            with_k: Whether to include exchange
            tol: Convergence tolerance for residuals
            max_cycle: Maximum Davidson iterations
            
        Returns:
            converged: (nband,) boolean array
            energies: (nband,) eigenvalues
            psi_g: (nband, ngrids) eigenvectors in G-space
        """
        nband = psi_g_init.shape[0]
        ngrids = psi_g_init.shape[1]
        logger.debug(self, f'Starting SCF Davidson for k-point {ik}, nband={nband}') 
        # Initialize subspace with input wavefunctions
        subspace = []
        for n in range(nband):
            psi = psi_g_init[n].copy()
            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi)**2))
            if norm > 1e-10:
                psi /= norm
            subspace.append(psi)
        
        # Diagonal preconditioner
        hdiag = self._get_hdiag(ik)
        
        energies = np.zeros(nband)
        converged = np.zeros(nband, dtype=bool)
        
        for davidson_iter in range(max_cycle):
            nv = len(subspace)  # Current subspace size
            
            # Orthonormalize subspace using QR decomposition
            # Stack vectors as columns
            subspace_matrix = np.column_stack([v.reshape(-1) for v in subspace])
            Q, R = np.linalg.qr(subspace_matrix)
            subspace = [Q[:, i] for i in range(Q.shape[1])]
            nv = len(subspace)
            
            # Build subspace Hamiltonian matrix
            # First, update density from current best wavefunctions
            if davidson_iter > 0:
                # Update orbitals at this k-point
                for n in range(min(nband, nv)):
                    self.psi_g[ik, n] = subspace[n].copy()
                    self.psi_r[ik, n] = self._ifft_g2r(subspace[n])
            
            # Build H in subspace by applying H to all basis vectors
            H_subspace = np.zeros((nv, nv), dtype=np.complex128)
            
            for i in range(nv):
                # Apply Hamiltonian
                H_psi_i = self.apply_hamiltonian(ik, subspace[i], 
                                                with_j=True, with_k=with_k)
                for j in range(nv):
                    H_subspace[j, i] = np.vdot(subspace[j].conj(), H_psi_i)
            
            # Since subspace is orthonormal, S = I, so just solve H c = E c
            e, c = np.linalg.eigh(H_subspace)
            
            # Extract lowest nband eigenpairs
            idx = np.argsort(e.real)[:nband]
            energies = e.real[idx]
            
            # Build eigenvectors in full space
            psi_g_new = np.zeros((nband, ngrids), dtype=np.complex128)
            for n in range(nband):
                for i in range(nv):
                    psi_g_new[n] += c[i, idx[n]] * subspace[i]
                # Normalize
                norm = np.sqrt(np.sum(np.abs(psi_g_new[n])**2))
                if norm > 1e-10:
                    psi_g_new[n] /= norm
            
            # Compute residuals for each eigenstate
            max_res = 0.0
            new_vectors = []
            
            for n in range(nband):
                # Apply H to eigenvector
                H_psi_n = self.apply_hamiltonian(ik, psi_g_new[n],
                                                with_j=True, with_k=with_k)
                # Residual: R = H|ψ⟩ - E|ψ⟩
                residual = H_psi_n - energies[n] * psi_g_new[n]
                res_norm = np.sqrt(np.sum(np.abs(residual)**2))
                max_res = max(max_res, res_norm)
                
                if res_norm < tol:
                    converged[n] = True
                else:
                    # Precondition residual: P = R / (H_diag - E)
                    diagd = hdiag - (energies[n] - 1e-3)
                    diagd[np.abs(diagd) < 1e-8] = 1e-8
                    precond_res = residual / diagd
                    
                    # Orthogonalize against existing subspace
                    for v in subspace:
                        precond_res -= np.vdot(v, precond_res) * v
                    
                    # Normalize and add to subspace
                    norm = np.sqrt(np.sum(np.abs(precond_res)**2))
                    if norm > 1e-10:
                        precond_res /= norm
                        new_vectors.append(precond_res)
            
            # Check convergence
            if np.all(converged) or max_res < tol:
                logger.debug(self, f'  Davidson converged at iteration {davidson_iter+1}, max_res={max_res:.3e}')
                break
            
            # Add new vectors to subspace (with size limit)
            for v in new_vectors[:nband]:  # Add at most nband new vectors
                subspace.append(v)
            
            # Restart if subspace gets too large
            if len(subspace) > 3 * nband:
                logger.debug(self, f'  Davidson restart at iteration {davidson_iter+1}, subspace size={len(subspace)}')
                subspace = [psi_g_new[n].copy() for n in range(nband)]
        
        # Update final wavefunctions
        subspace[:nband] = [psi_g_new[n] for n in range(nband)]
        
        return converged, energies, psi_g_new

    def _precondition_residual(self, ik, R_r, kappa=1.0):
        """Simple diagonal preconditioner in G-space: divide by T+κ."""
        Rg = self._fft_r2g(R_r)
        den = self._kin_diag[ik][None, :] + float(kappa)
        Rg /= den
        return self._ifft_g2r(Rg)

    def _get_hdiag(self, ik):
        """Diagonal elements of Hamiltonian for preconditioning.
        
        Returns kinetic energy diagonal (approximate diagonal of H).
        """
        return self._kin_diag[ik]

    # ----------------------------- SCF loop (G-space primary) ----------------------------- #
    def kernel(self, init='random', max_cycle=50, conv_tol=1e-7, conv_tol_rho=1e-6,
               alpha=0.3, with_k=True, davidson_tol=1e-6, davidson_max_cycle=30,
               ):
        """Self-consistent HF loop in G-space (J and optional K); no XC.
        
        Uses Davidson diagonalization to solve for eigenstates at each SCF iteration.

        Args:
            init: initialization method ('random', 'pw', 'khf-1e', 'atom')
            max_cycle: maximum number of SCF iterations
            conv_tol: energy convergence tolerance
            conv_tol_rho: density convergence tolerance
            alpha: linear mixing parameter for density (0<alpha<=1)
            with_k: include exchange operator (gamma-only for now)
            davidson_tol: tolerance for Davidson diagonalization
            davidson_max_cycle: max Davidson iterations
            trace: verbose output for each iteration

        Returns:
            (E_tot, converged)
        """
        # Build operators and initialize
        self.build()
        self.init_guess(kind=init)
        
        logger.info(self, '\n')
        logger.info(self, '******** %s SCF (plane-wave basis) ********', 
                    'HF' if with_k else 'Hartree-only')
        logger.info(self, 'nk = %d, mesh = %s, nband = %d, nocc = %d',
                    self.nk, self.mesh, self.nband, self.nocc)
        logger.info(self, 'Davidson tol = %.2e, max_cycle = %d', davidson_tol, davidson_max_cycle)
        logger.info(self, 'SCF conv_tol = %.2e, conv_tol_rho = %.2e', conv_tol, conv_tol_rho)
        
        # Allocate storage for energies
        self.mo_energy = np.zeros((self.nk, self.nband), dtype=float)
        
        # SCF loop
        e_tot_prev = 0.0
        rho_r_prev = None
        converged = False
        
        for scf_iter in range(1, max_cycle + 1):
            logger.info(self, '\n--- SCF Iteration %d ---', scf_iter)
            
            # 1. Compute current density in real space
            rho_r = self.get_density_r()
            
            # 2. Check density convergence
            logger.debug(self, "checking density convergence...")
            if rho_r_prev is not None:
                drho = rho_r - rho_r_prev
                rho_norm = np.linalg.norm(drho) * np.sqrt(self.grid_weight)
                logger.info(self, '  Density change: %.6e', rho_norm)
            else:
                rho_norm = 1.0
            
            # 3. Mix density (simple linear mixing)
            if scf_iter > 1:
                rho_r = alpha * rho_r + (1.0 - alpha) * rho_r_prev
            
            # 4. Diagonalize Hamiltonian at each k-point using custom SCF-aware Davidson
            for ik in range(self.nk):
                logger.info(self, '  k-point %d/%d: Diagonalizing H with SCF-Davidson...', ik + 1, self.nk)
                
                # Initial guess from current wavefunctions
                psi_g_init = self.psi_g[ik].copy()
                
                # Custom Davidson with density updates
                conv, e, psi_g_new = self._scf_davidson(
                    ik, psi_g_init,
                    with_k=with_k,
                    tol=davidson_tol,
                    max_cycle=davidson_max_cycle
                )
                
                # Update wavefunctions and energies
                for n in range(self.nband):
                    psi_g_n = psi_g_new[n]
                    
                    # Check normalization (should be ~1)
                    norm_g_check = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                    logger.debug(self, "k-point %d band %d: L2 norm(psi_g) = %.6e",
                                 ik, n, norm_g_check)

                    psi_r_n = self._ifft_g2r(psi_g_n)
                    
                    # For verification: R-space integral should also be ~1
                    norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2))
                    logger.debug(self, "k-point %d band %d: R-space integral = %.6e",
                                 ik, n, norm_r_integral)
                    
                    # Store both representations
                    self.psi_r[ik, n] = psi_r_n
                    self.psi_g[ik, n] = psi_g_n
                    self.mo_energy[ik, n] = np.real(e[n])
                
                logger.info(self, '    Converged: %s', conv[:min(4, self.nband)])
                logger.info(self, '    Lowest energies: %s', 
                            self.mo_energy[ik, :min(4, self.nband)])
                    
            
            # 5. Compute total energy using efficient method
            energy_dict = self.compute_energy_components(with_k=with_k)
            e_tot = energy_dict['E_tot']
            E_kin = energy_dict['E_kin']
            E_ne = energy_dict['E_ne']
            E_hartree = energy_dict['E_hartree']
            E_exchange = energy_dict['E_exchange']
            E_nuc = energy_dict['E_nuc']
            
            # 6. Check energy convergence
            de = e_tot - e_tot_prev
            logger.info(self, '  Cycle %3d: E = %16.10f  dE = %+.6e  |dρ| = %.6e',
                        scf_iter, e_tot, de, rho_norm)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, '    E_kin = %.10f', E_kin)
                logger.debug(self, '    E_ne = %.10f', E_ne)
                logger.debug(self, '    E_hartree = %.10f', E_hartree)
                logger.debug(self, '    E_exchange = %.10f', E_exchange)
                logger.debug(self, '    E_nuc = %.10f', E_nuc)
            
            # 7. Convergence check
            if scf_iter > 1 and abs(de) < conv_tol and rho_norm < conv_tol_rho:
                converged = True
                logger.info(self, '\n*** SCF Converged! ***')
                logger.info(self, 'E(HF) = %.10f Ha', e_tot)
                break
            
            # Update for next iteration
            e_tot_prev = e_tot
            rho_r_prev = rho_r.copy()
        
        if not converged:
            logger.warn(self, '\n*** SCF NOT converged after %d iterations ***', max_cycle)
        
        # Print final summary
        logger.info(self, '\n' + '=' * 60)
        logger.info(self, 'Final Results:')
        logger.info(self, '  Total Energy: %.10f Ha', e_tot)
        logger.info(self, '  Converged: %s', converged)
        for ik in range(self.nk):
            logger.info(self, '  k-point %d orbital energies (occupied):', ik)
            logger.info(self, '    %s', self.mo_energy[ik, :self.nocc])
        logger.info(self, '=' * 60)
        
        return e_tot, converged


__all__ = ['KPWSCF']


