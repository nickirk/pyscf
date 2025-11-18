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
        self.exxdiv = 'ewald'  # Default exchange divergence treatment for PBC

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
    def _ensure_grids_built(self):
        """Ensure grids object is synchronized with self.mesh and built."""
        if not np.array_equal(self.grids.mesh, self.mesh):
            self.grids.mesh = np.asarray(self.mesh, dtype=int)
            self.grids.build()
            self.ngrids = int(np.prod(self.mesh))  # Update ngrids too
            self.grid_weight = self.grids.weights[0] if self.ngrids > 0 else 0.0
        elif not hasattr(self.grids, 'coords') or self.grids.coords is None:
            self.grids.build()

    
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

    def _fft_density_r2g(self, rho_r):
        """FFT density from real-space to G-space.
        
        For density ρ(r), we want ρ(G) = ∫ ρ(r) e^{-iG·r} dr
        DFT gives: DFT[ρ] = Σ_r ρ_r e^{-iG·r}
        To convert: ρ(G) = DFT[ρ_r] * (V/N)
        
        Args:
            rho_r: density in real space (ngrids,) or shaped for mesh
        Returns:
            rho_g: density in G-space (ngrids,)
        """
        if rho_r.ndim == 1:
            rho_r_2d = rho_r.reshape(1, -1)
            rho_g_2d = pbctools.fft(rho_r_2d, self.mesh)
            rho_g_2d *= (self.vol / self.ngrids)
            return rho_g_2d[0]
        else:
            rho_g = pbctools.fft(rho_r, self.mesh)
            rho_g *= (self.vol / self.ngrids)
            return rho_g

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
            psi_r_2d *= self.ngrids / np.sqrt(self.vol)
            return psi_r_2d[0]
        else:
            psi_r = pbctools.ifft(psi_g_block, self.mesh)
            psi_r *= self.ngrids / np.sqrt(self.vol)
            return psi_r

    def _ifft_potential_g2r(self, V_g):
        """IFFT potential from G-space to real-space.
        
        For potential V(r), inverse FT is: V(r) = (1/V) Σ_G V(G) e^{iG·r}
        Numpy IFFT gives: IFFT[V] = (1/N) Σ_G V_G e^{iG·r}
        To convert: V(r) = (N/V) × IFFT[V(G)]
        
        Args:
            V_g: potential in G-space (ngrids,)
        Returns:
            V_r: potential in real-space (ngrids,)
        """
        if V_g.ndim == 1:
            V_g_2d = V_g.reshape(1, -1)
            V_r_2d = pbctools.ifft(V_g_2d, self.mesh)
            V_r_2d *= (self.ngrids / self.vol)
            return V_r_2d[0]
        else:
            V_r = pbctools.ifft(V_g, self.mesh)
            V_r *= (self.ngrids / self.vol)
            return V_r

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
        Gv = self.cell.get_Gv(mesh=mesh)  # Use the KPWSCF mesh, not cell.mesh!
        coulG = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
        self._coulG0 = coulG  # Cache for later use
        vneG = -rhoG * coulG  # (ngrids,) negative sign for attractive potential
        
        # Transform to real space
        # pbctools.ifft uses 1/N normalization, but we need 1/Ω for potentials
        # V(r) = (1/Ω) Σ_G V(G) e^(iG·r) = (N/Ω) × IFFT[V(G)]
        self._vne_R = pbctools.ifft(vneG, mesh).real * (self.ngrids / self.vol)
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
    def init_guess_by_minao(self, seed=1):
        """Initialize wavefunctions from projected ANO (minimal) basis.
        
        Projects atomic natural orbitals onto the real-space grid,
        then transforms to plane-wave representation via FFT.
        
        Occupied orbitals: projected from AO basis
        Virtual orbitals: random initialization in G-space
        
        Returns:
            self
        """
        from pyscf.scf import hf as mol_hf
        from pyscf.pbc.dft import numint
        
        log = logger.new_logger(self, self.verbose)
        log.info('init_guess_by_minao: Initializing from ANO basis')
        
        # Get initial guess in AO basis (gamma-point molecular guess)
        dm = mol_hf.init_guess_by_minao(self.cell)
        mo_coeff = dm.mo_coeff  # (nao, nao)
        mo_occ = dm.mo_occ      # (nao,)
        
        # Select occupied orbitals
        occ_idx = mo_occ > 0
        mo_coeff_occ = mo_coeff[:, occ_idx]
        nocc_from_ao = mo_coeff_occ.shape[1]
        
        log.info('  Projecting %d occupied AO-based orbitals to real-space grid', nocc_from_ao)
        log.info('  Bands 0-%d will be filled from AO projections', 
                 min(nocc_from_ao, self.nband) - 1)
        log.info('  Bands %d-%d will use random initialization', 
                 min(nocc_from_ao, self.nband), self.nband - 1)        # Ensure grids are built and synchronized BEFORE initializing arrays
        self._ensure_grids_built()
        
        # Initialize storage (must be after _ensure_grids_built which may update self.ngrids)
        self.psi_r = np.zeros((self.nk, self.nband, self.ngrids), dtype=complex)
        self.psi_g = np.zeros_like(self.psi_r)

        
        coords = self.grids.coords
        
        # Random number generator for virtual orbitals
        rng = np.random.RandomState(seed)
        
        for ik, kpt in enumerate(self.kpts):
            log.debug('  Processing k-point %d/%d', ik + 1, self.nk)
            
            # Evaluate AOs at k-point on grid: (ngrids, nao)
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            
            # Transform to occupied MOs on grid: (ngrids, nocc_from_ao)
            mo_on_grid = np.dot(ao_value, mo_coeff_occ)
            
            # Fill bands from projected AO orbitals (use all available, up to nband)
            n_fill = min(nocc_from_ao, self.nband)
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()  # (ngrids,)
                
                # Normalize: ∫|ψ_r|²dr = 1
                # Since grid_weight = V/N, we have: Σ|ψ_r|² * (V/N) = 1
                # So we need: Σ|ψ_r|² = N/V
                norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2) * self.grid_weight)
                if norm_r_integral > 1e-10:
                    psi_r_n /= norm_r_integral
                else:
                    log.warn('Orbital %d at k-point %d has near-zero norm, skipping', n, ik)
                    continue
                
                self.psi_r[ik, n] = psi_r_n
                
                # FFT to G-space
                self.psi_g[ik, n] = self._fft_r2g(psi_r_n)
            
            # Fill virtual bands with random values in G-space
            for n in range(n_fill, self.nband):
                # Random complex values in G-space
                psi_g_n = (rng.randn(self.ngrids) + 1j * rng.randn(self.ngrids))
                
                # Normalize in G-space: Σ|ψ_g|² = 1
                norm_g = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                if norm_g > 1e-10:
                    psi_g_n /= norm_g
                
                self.psi_g[ik, n] = psi_g_n
                
                # IFFT to R-space (will automatically have ∫|ψ_r|²dr = 1)
                self.psi_r[ik, n] = self._ifft_g2r(psi_g_n)
        
        log.info('  Initialization complete')
        if log.verbose >= logger.DEBUG:
            log.debug('  Checking normalization:')
            for ik in range(min(2, self.nk)):  # Check first 2 k-points
                for n in range(min(4, self.nband)):  # Check first 4 bands
                    norm_r = np.sqrt(np.sum(np.abs(self.psi_r[ik, n])**2) * self.grid_weight)
                    norm_g = np.sqrt(np.sum(np.abs(self.psi_g[ik, n])**2))
                    log.debug('    k=%d n=%d: ∫|ψ_r|²dr=%.6e, Σ|ψ_g|²=%.6e', 
                             ik, n, norm_r, norm_g)
        
        return self

    def init_guess_by_atom(self, seed=1):
        """Initialize from superposition of atomic HF densities.
        
        Projects atomic HF orbitals onto the real-space grid,
        then transforms to plane-wave representation via FFT.
        
        Occupied orbitals: projected from atomic HF
        Virtual orbitals: random initialization in G-space
        
        Returns:
            self
        """
        from pyscf.scf import hf as mol_hf
        from pyscf.pbc.dft import numint
        
        log = logger.new_logger(self, self.verbose)
        log.info('init_guess_by_atom: Initializing from atomic HF')
        
        # Get initial guess in AO basis (gamma-point molecular guess)
        dm = mol_hf.init_guess_by_atom(self.cell)
        mo_coeff = dm.mo_coeff  # (nao, nao)
        mo_occ = dm.mo_occ      # (nao,)
        
        # Select occupied orbitals
        occ_idx = mo_occ > 0
        mo_coeff_occ = mo_coeff[:, occ_idx]
        nocc_from_ao = mo_coeff_occ.shape[1]
        
        log.info('  Projecting %d occupied atomic orbitals to real-space grid', nocc_from_ao)
        log.info('  Bands 0-%d will be filled from AO projections', 
                 min(nocc_from_ao, self.nband) - 1)
        log.info('  Bands %d-%d will use random initialization', 
                 min(nocc_from_ao, self.nband), self.nband - 1)        # Ensure grids are built and synchronized BEFORE initializing arrays
        self._ensure_grids_built()
        
        # Initialize storage (must be after _ensure_grids_built which may update self.ngrids)
        self.psi_r = np.zeros((self.nk, self.nband, self.ngrids), dtype=complex)
        self.psi_g = np.zeros_like(self.psi_r)

        
        coords = self.grids.coords
        
        # Random number generator for virtual orbitals
        rng = np.random.RandomState(seed)
        
        for ik, kpt in enumerate(self.kpts):
            log.debug('  Processing k-point %d/%d', ik + 1, self.nk)
            
            # Evaluate AOs at k-point on grid: (ngrids, nao)
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            
            # Transform to occupied MOs on grid: (ngrids, nocc_from_ao)
            mo_on_grid = np.dot(ao_value, mo_coeff_occ)
            
            # Fill bands from projected AO orbitals (use all available, up to nband)
            n_fill = min(nocc_from_ao, self.nband)
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()  # (ngrids,)
                
                # Normalize: ∫|ψ_r|²dr = 1
                norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2) * self.grid_weight)
                if norm_r_integral > 1e-10:
                    psi_r_n /= norm_r_integral
                else:
                    log.warn('Orbital %d at k-point %d has near-zero norm, skipping', n, ik)
                    continue
                
                self.psi_r[ik, n] = psi_r_n
                
                # FFT to G-space
                self.psi_g[ik, n] = self._fft_r2g(psi_r_n)
            
            # Fill virtual bands with random values in G-space
            for n in range(n_fill, self.nband):
                # Random complex values in G-space
                psi_g_n = (rng.randn(self.ngrids) + 1j * rng.randn(self.ngrids))
                
                # Normalize in G-space: Σ|ψ_g|² = 1
                norm_g = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                if norm_g > 1e-10:
                    psi_g_n /= norm_g
                
                self.psi_g[ik, n] = psi_g_n
                
                # IFFT to R-space (will automatically have ∫|ψ_r|²dr = 1)
                self.psi_r[ik, n] = self._ifft_g2r(psi_g_n)
        
        log.info('  Initialization complete')
        if log.verbose >= logger.DEBUG:
            log.debug('  Checking normalization:')
            for ik in range(min(2, self.nk)):  # Check first 2 k-points
                for n in range(min(4, self.nband)):  # Check first 4 bands
                    norm_r = np.sqrt(np.sum(np.abs(self.psi_r[ik, n])**2) * self.grid_weight)
                    norm_g = np.sqrt(np.sum(np.abs(self.psi_g[ik, n])**2))
                    log.debug('    k=%d n=%d: ∫|ψ_r|²dr=%.6e, Σ|ψ_g|²=%.6e', 
                             ik, n, norm_r, norm_g)
        
        return self

    def init_guess_from_mo_coeff(self, mo_coeff, mo_occ=None, seed=1):
        """Initialize from provided MO coefficients (e.g., from converged KRHF).
        
        This method allows using converged orbitals from a standard AO-based
        calculation (like KRHF) as the initial guess. Useful for debugging
        and verifying energy calculations.
        
        Args:
            mo_coeff: MO coefficients in AO basis
                      - For single k-point: (nao, nmo) array
                      - For multiple k-points: list of (nao, nmo) arrays
            mo_occ: Optional orbital occupations
                    - If None, fills lowest nband orbitals
                    - For single k-point: (nmo,) array
                    - For multiple k-points: list of (nmo,) arrays
            seed: random seed for any remaining virtual orbitals
            
        Returns:
            self
        """
        from pyscf.pbc.dft import numint
        
        log = logger.new_logger(self, self.verbose)
        log.info('init_guess_from_mo_coeff: Initializing from provided MO coefficients')
        
        # Handle single k-point case
        if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
            mo_coeff = [mo_coeff]
            if mo_occ is not None:
                mo_occ = [mo_occ]
        
        if len(mo_coeff) != self.nk:
            raise ValueError(f'mo_coeff has {len(mo_coeff)} k-points but KPWSCF has {self.nk}')
        
        # Ensure grids are built and synchronized BEFORE initializing arrays
        self._ensure_grids_built()
        
        # Initialize storage (must be after _ensure_grids_built which may update self.ngrids)
        self.psi_r = np.zeros((self.nk, self.nband, self.ngrids), dtype=complex)
        self.psi_g = np.zeros_like(self.psi_r)

        
        coords = self.grids.coords
        
        # Random number generator for any remaining virtual orbitals
        rng = np.random.RandomState(seed)
        
        for ik, kpt in enumerate(self.kpts):
            log.debug('  Processing k-point %d/%d', ik + 1, self.nk)
            
            mo_k = mo_coeff[ik]  # (nao, nmo)
            nmo_available = mo_k.shape[1]
            
            # Determine which MOs to use
            if mo_occ is not None:
                # Use occupation to select orbitals (prioritize occupied)
                occ_k = mo_occ[ik]
                # Sort by occupancy (descending)
                idx_sorted = np.argsort(-occ_k)
                n_fill = min(nmo_available, self.nband)
                mo_to_use = mo_k[:, idx_sorted[:n_fill]]
            else:
                # Just use first nband orbitals
                n_fill = min(nmo_available, self.nband)
                mo_to_use = mo_k[:, :n_fill]
            
            log.info('  K-point %d: using %d MOs from provided coefficients', ik, n_fill)
            
            # Evaluate AOs at k-point on grid: (ngrids, nao)
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            
            # Transform to MOs on grid: (ngrids, n_fill)
            mo_on_grid = np.dot(ao_value, mo_to_use)
            
            # Fill bands from MO projections
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()  # (ngrids,)
                
                # Normalize: ∫|ψ_r|²dr = 1
                norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2) * self.grid_weight)
                if norm_r_integral > 1e-10:
                    psi_r_n /= norm_r_integral
                else:
                    log.warn('Orbital %d at k-point %d has near-zero norm, skipping', n, ik)
                    continue
                
                self.psi_r[ik, n] = psi_r_n
                
                # FFT to G-space
                self.psi_g[ik, n] = self._fft_r2g(psi_r_n)
            
            # Fill any remaining virtual bands with random values in G-space
            for n in range(n_fill, self.nband):
                psi_g_n = (rng.randn(self.ngrids) + 1j * rng.randn(self.ngrids))
                norm_g = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                if norm_g > 1e-10:
                    psi_g_n /= norm_g
                
                self.psi_g[ik, n] = psi_g_n
                self.psi_r[ik, n] = self._ifft_g2r(psi_g_n)
        
        log.info('  Initialization complete')
        if log.verbose >= logger.DEBUG:
            log.debug('  Checking normalization:')
            for ik in range(min(2, self.nk)):
                for n in range(min(4, self.nband)):
                    norm_r = np.sqrt(np.sum(np.abs(self.psi_r[ik, n])**2) * self.grid_weight)
                    norm_g = np.sqrt(np.sum(np.abs(self.psi_g[ik, n])**2))
                    log.debug('    k=%d n=%d: ∫|ψ_r|²dr=%.6e, Σ|ψ_g|²=%.6e', 
                             ik, n, norm_r, norm_g)
        
        return self

    def init_guess(self, kind='minao', seed=1, mo_coeff=None, mo_occ=None):
        """Initialize ψ on the real-space grid.

        Args:
            kind: 'atom' | 'minao' | 'random' | 'mo'
            seed: random seed for random/virtual orbital initialization
            mo_coeff: MO coefficients for kind='mo' (from converged calculation)
            mo_occ: MO occupations for kind='mo' (optional)
            
        Returns:
            self
        """
        if kind == 'minao':
            return self.init_guess_by_minao(seed=seed)
        elif kind == 'atom':
            return self.init_guess_by_atom(seed=seed)
        elif kind == 'mo':
            if mo_coeff is None:
                raise ValueError("mo_coeff must be provided for kind='mo'")
            return self.init_guess_from_mo_coeff(mo_coeff, mo_occ, seed=seed)
        elif kind == 'random':
            # Keep existing random initialization
            log = logger.new_logger(self, self.verbose)
            log.info('init_guess: Random initialization')
            rng = np.random.RandomState(seed)
            self.psi_r = np.zeros((self.nk, self.nband, self.ngrids), dtype=complex)
            self.psi_g = np.zeros_like(self.psi_r)
            
            for ik in range(self.nk):
                for n in range(self.nband):
                    # Random in G-space
                    psi_g_n = (rng.randn(self.ngrids) + 1j * rng.randn(self.ngrids))
                    norm_g = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                    if norm_g > 1e-10:
                        psi_g_n /= norm_g
                    self.psi_g[ik, n] = psi_g_n
                    self.psi_r[ik, n] = self._ifft_g2r(psi_g_n)
            return self
        else:
            raise ValueError(f"Unknown init_guess kind: {kind}. Use 'atom', 'minao', or 'random'")

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


    def apply_hamiltonian(self, ik,  psi_nk_g, rho_r=None, with_j=True, with_k=True):
        """Apply full Hamiltonian to a single wavefunction.
        
        H = T + V_ne + V_H + V_X (for HF)
        
        Following the pseudocode:
        1. Apply kinetic energy (diagonal in G-space)
        2. Apply local potential V_ne (nuclear-electron)
        3. Apply Hartree potential V_H (if with_j=True)
        4. Apply exchange potential V_X (if with_k=True)
        
        Args:
            ik: k-point index
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
            rho_r: (ngrids,) total electron density in real space (if None, computed from self.psi_r)
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
        
        # Get density (use provided or compute from current state)
        if rho_r is None:
            rho_r = self.get_density_r()
        
        # 3. Apply Hartree Potential V_H (electron-electron repulsion, direct term)
        if with_j:
            # V_H(r) from total density (precomputed)
            rho_g = self._fft_density_r2g(rho_r)
            coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
            vH_g = coulG * rho_g
            vH_r = self._ifft_potential_g2r(vH_g).real
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
        
        Energy formula (following PySCF convention for RHF with exxdiv='ewald'):
        E_tot = E_kin + E_ne + E_hartree + E_exchange + E_nuc
        where:
        - E_kin = Σ_{k,n} f_kn <ψ_kn|T|ψ_kn>
        - E_ne = Σ_{k,n} f_kn <ψ_kn|V_ne|ψ_kn>
        - E_hartree = 0.5 * ∫ ρ(r) V_H(r) dr  [computed ONCE from total density]
        - E_exchange = -0.5 * Σ_{k,n} f_kn <ψ_kn|K|ψ_kn>  [K includes factor of 2 for spin]
        - E_nuc = nuclear-nuclear repulsion (Ewald sum for PBC)
        - f_kn = 2/nk (occupation weight for closed-shell)
        
        This is equivalent to PySCF's: E_tot = E_h1e + 0.5*Tr(dm@vj) - 0.25*Tr(dm@vk) + E_nuc
        
        Args:
            with_k: whether to include exchange
            
        Returns:
            dict with energy components: 
            {E_kin, E_ne, E_hartree, E_exchange, E_nuc, E_tot}
        """
        occ_weight = 2.0 / self.nk  # Closed shell
        
        # Sanity check on density
        
        # Precompute Hartree potential and energy from total density
        # E_hartree = (1/2) ∫ ρ(r) V_H(r) dr
        # This is computed ONCE from the total density, not per-orbital!
        rho_r = self.get_density_r()  # Total density in real space
        rho_g = self.get_density_G()
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        vH_g = coulG * rho_g
        vH_r = self._ifft_g2r(vH_g).real
        
        # Compute Hartree energy ONCE from total density
        E_hartree = 0.5 * np.sum(rho_r * vH_r).real * self.grid_weight
        
        # Initialize other energy components
        E_kin = 0.0
        E_ne = 0.0
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
                if ik == 0 and n == 0:
                    logger.debug(self, f"  V_ne stats: min={self._vne_R.min():.6f}, max={self._vne_R.max():.6f}, mean={self._vne_R.mean():.6f}")
                    psi_r_sq = np.abs(psi_r)**2
                    logger.debug(self, f"  |ψ_r|² stats: min={psi_r_sq.min():.6e}, max={psi_r_sq.max():.6e}")
                vne_psi_r = self._vne_R * psi_r
                E_ne_contrib = occ_weight * np.sum(psi_r.conj() * vne_psi_r).real * self.grid_weight
                E_ne += E_ne_contrib
                logger.debug(self, f"  E_ne contribution: {E_ne_contrib:.6e}")
                
                # NOTE: Hartree energy is computed ONCE from total density above,
                # not as a sum over orbitals (that would double-count!)
                
                # Exchange energy: <ψ|K|ψ> = ∫ ψ*(r) K[ψ](r) dr
                # Note: K operator in _apply_k returns -2*Σ_m V_nm*ψ_m (negative)
                # So <ψ|K|ψ> < 0 for repulsive exchange
                # Energy formula: E_tot = ... - 0.25*Tr(dm@vk) + ...
                # Since Tr(dm@vk) = 2*Σ_n<ψ|vk|ψ> and vk = -K/2, we have:
                # E_exchange = -0.25*Tr(dm@vk) = -0.25*2*Σ_n<ψ|-K/2|ψ> = 0.25*Σ_n<ψ|K|ψ>
                # With occ_weight = 2/nk for closed shell:
                # E_exchange = 0.25 * (2/nk) * Σ_k,n <ψ|K|ψ> = (0.5/nk) * Σ_k,n <ψ|K|ψ>
                if with_k:
                    K_psi_g = self._apply_k(ik, psi_g)
                    K_psi_r = self._ifft_g2r(K_psi_g)
                    K_psi_integral = np.sum(psi_r.conj() * K_psi_r).real * self.grid_weight
                    E_x_contrib = 0.25 * occ_weight * K_psi_integral  # 0.5 factor, NO minus sign
                    if ik == 0 and n == 0:
                        logger.debug(self, f"  <ψ|K|ψ> = {K_psi_integral:.6e}, occ_weight={occ_weight:.3f}, E_x_contrib={E_x_contrib:.6e}")
                    logger.debug(self, f"  E_exchange contribution: {E_x_contrib:.6e}")
                    E_exchange += E_x_contrib
        
        
        # Nuclear-nuclear repulsion (Ewald energy for periodic systems)
        if hasattr(self.cell, 'energy_nuc'):
            E_nuc = self.cell.energy_nuc()
        else:
            E_nuc = 0.0
        
        # Add Ewald divergence correction for exchange if exxdiv='ewald'
        # This corrects for the G=0 divergence in periodic exchange
        # The correction is: -0.5 * N_elec * madelung
        if with_k and self.exxdiv == 'ewald':
            from pyscf.pbc import tools as pbctools
            madelung = pbctools.madelung(self.cell, np.zeros(3))
            E_ewald_correction = -0.5 * self.cell.nelectron * madelung
            logger.debug(self, f"Ewald exchange correction: madelung={madelung:.6e}, "
                        f"N_elec={self.cell.nelectron}, correction={E_ewald_correction:.6e}")
            E_exchange += E_ewald_correction
        
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
        """Apply exchange operator K to wavefunction for closed-shell system.
        
        For closed-shell RHF, each spatial orbital is occupied by 2 electrons (α and β).
        The exchange operator is:
        K|ψ_n⟩ = -Σ_{m∈occ,σ} ∫ dr' ψ_m^σ*(r') ψ_n(r') / |r-r'| ψ_m^σ(r)
               = -2 Σ_{m∈occ} ∫ dr' ψ_m*(r') ψ_n(r') / |r-r'| ψ_m(r)
        
        where the factor of 2 accounts for α and β spins.
        
        In Fourier space:
        1. Compute pair density ρ_{nm}(r) = ψ_m*(r) * ψ_n(r)
        2. Solve Poisson in G-space: V_{nm}(G) = (4π/|G|^2) FFT[ρ_{nm}(r)]
        3. Transform back: V_{nm}(r) = IFFT[V_{nm}(G)]
        4. Accumulate: K|ψ_n⟩ += -2 * V_{nm}(r) * ψ_m(r)  [factor of 2 for spin]
        
        Args:
            ik: k-point index (currently assumes Gamma point, ik=0)
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
        Returns:
            K|ψ_n⟩ in G-space (ngrids,)
        """
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh, exxdiv=self.exxdiv)
        
        # Transform input wavefunction to real space
        psi_n_r = self._ifft_g2r(psi_nk_g)
        
        # Initialize exchange contribution
        k_psi_g = np.zeros_like(psi_nk_g)
        
        # Loop over all occupied orbitals
        for n_occ in range(self.nocc):
            # Get occupied orbital in real space
            psi_occ_g = self.psi_g[ik, n_occ]  # (ngrids,)
            psi_occ_r = self._ifft_g2r(psi_occ_g)
            
            # Compute pair density: ρ_{n,m}(r) = ψ_m*(r) * ψ_n(r)
            rho_ij_r = psi_occ_r.conj() * psi_n_r  # (ngrids,)
            
            # Transform to G-space (use density FFT, not wavefunction FFT)
            rho_ij_g = self._fft_density_r2g(rho_ij_r)  # (ngrids,)
            
            # Apply Coulomb kernel: V_{n,m}(G) = (4π/|G|^2) * ρ_{n,m}(G)
            V_ij_g = coulG * rho_ij_g  # (ngrids,)
            
            # Transform back to real space (use potential IFFT, not wavefunction IFFT)
            V_ij_r = self._ifft_potential_g2r(V_ij_g)  # (ngrids,)
            
            # Accumulate exchange: -2 * V_{n,m}(r) * ψ_m(r)  [factor 2 for spin]
            k_psi_r = -2.0 * V_ij_r * psi_occ_r  # (ngrids,)
            k_psi_g += self._fft_r2g(k_psi_r)  # (ngrids,)
        
        return k_psi_g


    def _scf_davidson(self, ik, psi_g_init, with_k=True, tol=1e-6, max_cycle=30):
        """Custom Davidson diagonalization.
        
        Diagonalizes the Hamiltonian. The Hamiltonian is constructed using
        the current density from self.psi_g.
        
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
            # Note: We do NOT update self.psi_g here to keep the density stable
            # The density is mixed in the outer SCF loop
            
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
            idx = np.argsort(e)[:nband] 
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
                # we keep only the latest nband vectors
                subspace = subspace[-nband*3:]
            
            # update self.psi_g temporarily for next iteration
            for n in range(nband):
                self.psi_g[ik, n] = psi_g_new[n]
                self.psi_r[ik, n] = self._ifft_g2r(psi_g_new[n])
        # Update final wavefunctions
        #subspace[:nband] = [psi_g_new[n] for n in range(nband)]
        
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
    def kernel(self, init='minao', max_cycle=50, conv_tol=1e-7, conv_tol_rho=1e-6,
               alpha=0.3, with_k=True, davidson_tol=1e-6, davidson_max_cycle=1,
               ):
        """Self-consistent HF loop in G-space (J and optional K); no XC.
        
        Uses Davidson diagonalization to solve for eigenstates at each SCF iteration.
        
        Note: davidson_max_cycle defaults to 1 for better SCF stability.

        Args:
            init: initialization method ('random', 'minao', 'atom')
            max_cycle: maximum number of SCF iterations
            conv_tol: energy convergence tolerance
            conv_tol_rho: density convergence tolerance
            alpha: linear mixing parameter for density (0<alpha<=1)
            with_k: include exchange operator (gamma-only for now)
            davidson_tol: tolerance for Davidson diagonalization
            davidson_max_cycle: max Davidson iterations per SCF cycle (default=1)
        Returns:
            (E_tot, converged)
        """
        # Build operators if not already built
        if not hasattr(self, '_vne_R') or self._vne_R is None:
            self.build()
        
        # Initialize if requested
        if init is not None:
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
        
        # Compute initial energy if wavefunctions are already initialized
        if hasattr(self, 'psi_r') and self.psi_r is not None:
            logger.info(self, '\n--- Initial Energy (before SCF iterations) ---')
            energy_dict_init = self.compute_energy_components(with_k=with_k)
            e_init = energy_dict_init['E_tot']
            logger.info(self, '  Initial E_tot = %.10f Ha', e_init)
            if self.verbose >= logger.INFO:
                logger.info(self, '    E_kin     = %.10f', energy_dict_init['E_kin'])
                logger.info(self, '    E_ne      = %.10f', energy_dict_init['E_ne'])
                logger.info(self, '    E_hartree = %.10f', energy_dict_init['E_hartree'])
                logger.info(self, '    E_exchange= %.10f', energy_dict_init['E_exchange'])
                logger.info(self, '    E_nuc     = %.10f', energy_dict_init['E_nuc'])
            
            # Compute and print initial orbital energies <ψ|H|ψ>
            logger.info(self, '  Initial orbital energies (before Davidson):')
            for ik in range(self.nk):
                for n in range(self.nband):
                    psi_g_n = self.psi_g[ik, n]
                    # Apply Hamiltonian
                    H_psi_g = self.apply_hamiltonian(ik, psi_g_n, with_j=True, with_k=with_k)
                    # Compute <ψ|H|ψ>
                    e_orbital = np.vdot(psi_g_n, H_psi_g).real
                    self.mo_energy[ik, n] = e_orbital
                    if n < 4:  # Print first few
                        logger.info(self, '    k=%d, band %d: ε = %.6f Ha', ik, n, e_orbital)
            
            # Apply Madelung shift to occupied orbital energies (for exxdiv='ewald')
            if with_k and self.exxdiv == 'ewald':
                madelung = pbctools.madelung(self.cell, self.kpts)
                logger.info(self, '  Applying Madelung shift to occupied orbitals: %.6f Ha', madelung)
                for ik in range(self.nk):
                    for n in range(self.nocc):
                        self.mo_energy[ik, n] += madelung
                        if n < 4:  # Print shifted values
                            logger.info(self, '    k=%d, band %d: ε = %.6f Ha (after Madelung)', 
                                      ik, n, self.mo_energy[ik, n])
        
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
            # Pass the mixed density so the Hamiltonian uses it consistently
            
            # Skip Davidson on first iteration if orbitals are already initialized
            # This preserves the initial guess (e.g., from converged RHF)
            skip_davidson = (scf_iter == 1 and hasattr(self, 'psi_r') and 
                           self.psi_r is not None and init is None)
            
            if skip_davidson:
                logger.info(self, '  Skipping Davidson on first iteration (using initial guess)')
                # Just compute orbital energies from initial wavefunctions
                for ik in range(self.nk):
                    for n in range(self.nband):
                        psi_g_n = self.psi_g[ik, n]
                        H_psi_g = self.apply_hamiltonian(ik, psi_g_n, with_j=True, with_k=with_k)
                        self.mo_energy[ik, n] = np.vdot(psi_g_n, H_psi_g).real
                
                # Apply Madelung shift to occupied orbital energies (for exxdiv='ewald')
                if with_k and self.exxdiv == 'ewald':
                    madelung = pbctools.madelung(self.cell, self.kpts)
                    for ik in range(self.nk):
                        for n in range(self.nocc):
                            self.mo_energy[ik, n] += madelung
                    if self.verbose >= logger.DEBUG:
                        logger.debug(self, '  Applied Madelung shift to occupied orbitals: %.6f Ha', madelung)
            else:
                # Run Davidson diagonalization
                for ik in range(self.nk):
                    logger.info(self, '  k-point %d/%d: Diagonalizing H with SCF-Davidson...', ik + 1, self.nk)
                    
                    # Initial guess from current wavefunctions
                    psi_g_init = self.psi_g[ik].copy()
                    
                    # Custom Davidson diagonalization
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
                        norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2)*self.grid_weight)
                        logger.debug(self, "k-point %d band %d: R-space integral = %.6e",
                                     ik, n, norm_r_integral)
                        
                        # Store both representations
                        self.psi_r[ik, n] = psi_r_n
                        self.psi_g[ik, n] = psi_g_n
                        self.mo_energy[ik, n] = np.real(e[n])
                    
                    logger.info(self, '    Converged: %s', conv[:min(4, self.nband)])
                    logger.info(self, '    Lowest energies: %s', 
                                self.mo_energy[ik, :min(4, self.nband)])
            
            # Apply Madelung shift to occupied orbital energies (for exxdiv='ewald')
            if with_k and self.exxdiv == 'ewald':
                madelung = pbctools.madelung(self.cell, self.kpts)
                for ik in range(self.nk):
                    for n in range(self.nocc):
                        self.mo_energy[ik, n] += madelung
                if self.verbose >= logger.DEBUG:
                    logger.debug(self, '  Applied Madelung shift to occupied orbitals: %.6f Ha', madelung)
            
            # DEBUG: Check orbital properties after Davidson update
            if scf_iter == 1 and logger.DEBUG >= self.verbose:
                logger.debug(self, '\n=== DEBUG: After Davidson update ===')
                for n in range(self.nband):
                    psi_g_norm = np.sqrt(np.sum(np.abs(self.psi_g[0,n])**2))
                    psi_r_norm = np.sqrt(np.sum(np.abs(self.psi_r[0,n])**2) * self.grid_weight)
                    logger.debug(self, f'  Band {n}: psi_g norm = {psi_g_norm:.6e}, psi_r norm = {psi_r_norm:.6e}')
                
                # Check density
                rho_check = self.get_density_r()
                rho_integral = np.sum(rho_check) * self.grid_weight
                logger.debug(self, f'  Density integral = {rho_integral:.6f}')
                
                # Check individual components
                logger.debug(self, '  Computing energy components manually:')
                E_kin_check = 0.0
                for n in range(self.nband):  # Check ALL bands, not just occupied
                    psi_r_n = self.psi_r[0, n]
                    psi_g_n = self.psi_g[0, n]
                    # Kinetic in G-space
                    k_energy = np.sum(self._kin_diag[0] * np.abs(psi_g_n)**2)
                    eigenvalue = self.mo_energy[0, n]
                    logger.debug(self, f'    Band {n}: KE = {k_energy:.6f}, eigenvalue = {eigenvalue:.6f}')
                    if n < self.nocc:
                        E_kin_check += k_energy * (2.0 / self.nk)
                logger.debug(self, f'  Total E_kin (manual, from occupied) = {E_kin_check:.6f}')
            
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


