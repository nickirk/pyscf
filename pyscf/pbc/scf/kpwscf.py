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
        self.verbose = logger.INFO if verbose is None else verbose
        self.max_memory = cell.max_memory

        # K-points
        if kpts is None:
            kpts = np.zeros((1, 3))
        if isinstance(kpts, KPoints):
            kpts = kpts.kpts
        self.kpts = np.asarray(kpts, dtype=float)
        self.nk = len(self.kpts)
        self.exxdiv = 'ewald'  # Default exchange divergence treatment for PBC

        # Mesh / grids
        # Use _select_mesh to determine mesh with proper priority: explicit > cell.mesh > ke_cutoff
        self.mesh = self._select_mesh(mesh)
        logger.debug(self, 'Using mesh %s for FFT grids', str(self.mesh))
        
        # Setup grids with the selected mesh
        self.grids = pbc_gen_grid.UniformGrids(self.cell)
        self.grids.mesh = np.asarray(self.mesh, dtype=int)
        self.ngrids = int(np.prod(self.grids.mesh))
        self.vol = float(self.cell.vol)
        self.grid_weight = self.grids.weights[0] if self.ngrids > 0 else 0.0  # ∫f ≈ weight * sum(f)

        # Number of bands
        nelec = sum(self.cell.nelec) if hasattr(self.cell, 'nelec') else self.cell.nelectron
        if nband is None:
            # Heuristic: filled + 20% extra, at least 1
            nocc = max(1, int(np.ceil(nelec/2)))
            nband = max(nocc, int(np.ceil(nocc * 1.2)))
        elif nband < nelec // 2:
            logger.warn(self, f'Number of bands nband={nband} is less than number of occupied orbitals {nelec/2}')
            nband = int(np.ceil(nelec // 2 * 1.2))

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

        Priority: explicit mesh > cell.mesh > cell.ke_cutoff
        
        Args:
            mesh: explicit mesh specification (overrides all defaults)
            
        Returns:
            tuple of mesh dimensions
        """
        if mesh is not None:
            return tuple(np.asarray(mesh, dtype=int))
        
        # Try cell.mesh first (highest priority among defaults)
        if hasattr(self.cell, 'mesh') and self.cell.mesh is not None:
            return tuple(np.asarray(self.cell.mesh, dtype=int))
        
        # Fall back to ke_cutoff if cell.mesh not specified
        if getattr(self.cell, 'ke_cutoff', None) is not None:
            Ecut = float(np.min(np.atleast_1d(self.cell.ke_cutoff)))
            m = self.cell.cutoff_to_mesh(Ecut)
            return tuple(np.asarray(m, dtype=int))
        
        # Last resort: try UniformGrids default
        raise ValueError('Cannot determine FFT mesh: provide mesh argument, '
                        'set cell.mesh, or set cell.ke_cutoff')

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
        """Build nuclear-electron potential V_ne in real space.
        
        This includes:
        1. Bare nucleus potential (Coulomb)
        2. Local pseudopotential term (if pseudopotential is specified in cell)
        
        Non-local pseudopotential terms are handled separately in _apply_pp_nonlocal().
        """
        from pyscf.pbc.gto.pseudo import pp as pseudo_module
        
        log = logger.new_logger(self, self.verbose)
        mesh = self.mesh
        cell = self.cell
        
        # Get G-vectors
        Gv = self.cell.get_Gv(mesh=mesh)  # Use the KPWSCF mesh, not cell.mesh!
        
        # Get structure factors and atomic charges
        SI = cell.get_SI(mesh=mesh)  # (natm, ngrids) complex
        charge = cell.atom_charges()  # Nuclear charges (positive)
        
        # Check if pseudopotential is specified
        has_pseudo = hasattr(cell, '_pseudo') and cell._pseudo
        
        if has_pseudo:
            # With pseudopotential: use local PP which replaces bare Coulomb
            log.debug1('Using local pseudopotential (replaces bare Coulomb)')
            vpplocG = pseudo_module.get_vlocG(cell, Gv)  # (natm, ngrids)
            vneG = -np.einsum('ij,ij->j', SI, vpplocG)  # Contract with structure factors
            log.debug2('Local PP V_loc(G): min/max %.6g / %.6g', 
                      vneG.min(), vneG.max())
        else:
            # Without pseudopotential: use bare Coulomb potential
            rhoG = charge @ SI  # (ngrids,) nuclear charge density in G-space
            # Get Coulomb kernel and compute V_ne in G-space
            # V_ne(G) = -4π*Z(G)/|G|^2 (negative for attractive potential)
            vneG = -rhoG * pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)  # negative sign for attractive potential
        
        # Cache Coulomb kernel for use in Hartree energy calculation
        self._coulG0 = pbctools.get_coulG(cell, mesh=mesh, Gv=Gv)
        
        # Transform to real space
        # pbctools.ifft uses 1/N normalization, but we need 1/Ω for potentials
        # V(r) = (1/Ω) Σ_G V(G) e^(iG·r) = (N/Ω) × IFFT[V(G)]
        self._vne_R = pbctools.ifft(vneG, mesh).real * (self.ngrids / self.vol)
        log.debug1('Built V_ne+Vloc on grid; min/max %.6g / %.6g', 
                   self._vne_R.min(), self._vne_R.max())
        return self._vne_R


    # ----------------------------- initialization ----------------------------- #
    def _init_wavefunctions_storage(self):
        """Initialize psi_r and psi_g arrays after ensuring grids are built."""
        self._ensure_grids_built()
        self.psi_r = np.zeros((self.nk, self.nband, self.ngrids), dtype=complex)
        self.psi_g = np.zeros_like(self.psi_r)

    def _normalize_and_store_orbital_r(self, psi_r_n, ik, n):
        """Normalize orbital in real space and store in psi_r and psi_g.
        
        Args:
            psi_r_n: orbital in real space (ngrids,)
            ik: k-point index
            n: band index
            log: logger object
            
        Returns:
            True if successful, False if orbital has near-zero norm
        """
        norm_r_integral = np.sqrt(np.sum(np.abs(psi_r_n)**2) * self.grid_weight)
        if norm_r_integral > 1e-10:
            psi_r_n /= norm_r_integral
            self.psi_r[ik, n] = psi_r_n
            self.psi_g[ik, n] = self._fft_r2g(psi_r_n)
            return True
        else:
            logger.warn(self, 'Orbital %d at k-point %d has near-zero norm, skipping', n, ik)
            return False

    def _fill_random_virtual_orbitals(self, n_start, rng):
        """Fill virtual bands with random values in G-space.
        
        Args:
            n_start: starting band index for random orbitals
            rng: numpy random number generator
            log: logger object
        """
        for ik in range(self.nk):
            for n in range(n_start, self.nband):
                psi_g_n = (rng.randn(self.ngrids) + 1j * rng.randn(self.ngrids))
                norm_g = np.sqrt(np.sum(np.abs(psi_g_n)**2))
                if norm_g > 1e-10:
                    psi_g_n /= norm_g
                self.psi_g[ik, n] = psi_g_n
                self.psi_r[ik, n] = self._ifft_g2r(psi_g_n)

    def _log_normalization_check(self):
        """Log normalization check for debugging."""
        if self.verbose >= logger.DEBUG:
            logger.debug(self, '  Checking normalization:')
            for ik in range(min(2, self.nk)):
                for n in range(min(4, self.nband)):
                    norm_r = np.sqrt(np.sum(np.abs(self.psi_r[ik, n])**2) * self.grid_weight)
                    norm_g = np.sqrt(np.sum(np.abs(self.psi_g[ik, n])**2))
                    logger.debug(self, '    k=%d n=%d: ∫|ψ_r|²dr=%.6e, Σ|ψ_g|²=%.6e', 
                             ik, n, norm_r, norm_g)

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
                 min(nocc_from_ao, self.nband), self.nband - 1)
        
        self._init_wavefunctions_storage()
        coords = self.grids.coords
        rng = np.random.RandomState(seed)
        
        n_fill = min(nocc_from_ao, self.nband)
        for ik, kpt in enumerate(self.kpts):
            log.debug('  Processing k-point %d/%d', ik + 1, self.nk)
            
            # Evaluate AOs at k-point on grid and transform to MOs
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            mo_on_grid = np.dot(ao_value, mo_coeff_occ)
            
            # Fill bands from projected AO orbitals
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()
                self._normalize_and_store_orbital_r(psi_r_n, ik, n)
        
        # Fill remaining virtual bands with random orbitals
        self._fill_random_virtual_orbitals(n_fill, rng)
        
        logger.info(self, '  Initialization complete')
        self._log_normalization_check()
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
                 min(nocc_from_ao, self.nband), self.nband - 1)
        
        self._init_wavefunctions_storage()
        coords = self.grids.coords
        rng = np.random.RandomState(seed)
        
        n_fill = min(nocc_from_ao, self.nband)
        for ik, kpt in enumerate(self.kpts):
            log.debug('  Processing k-point %d/%d', ik + 1, self.nk)
            
            # Evaluate AOs at k-point on grid and transform to MOs
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            mo_on_grid = np.dot(ao_value, mo_coeff_occ)
            
            # Fill bands from projected AO orbitals
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()
                self._normalize_and_store_orbital_r(psi_r_n, ik, n)
        
        # Fill remaining virtual bands with random orbitals
        self._fill_random_virtual_orbitals(n_fill, rng)
        
        log.info('  Initialization complete')
        self._log_normalization_check()
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
        
        # Handle single k-point case
        if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
            mo_coeff = [mo_coeff]
            if mo_occ is not None:
                mo_occ = [mo_occ]
        
        if len(mo_coeff) != self.nk:
            raise ValueError(f'mo_coeff has {len(mo_coeff)} k-points but KPWSCF has {self.nk}')
        
        self._init_wavefunctions_storage()
        coords = self.grids.coords
        rng = np.random.RandomState(seed)
        
        # Determine minimum n_fill across all k-points
        n_fill_min = self.nband
        for ik, kpt in enumerate(self.kpts):
            logger.debug(self, '  Processing k-point %d/%d', ik + 1, self.nk)
            
            mo_k = mo_coeff[ik]  # (nao, nmo)
            nmo_available = mo_k.shape[1]
            
            # Determine which MOs to use
            if mo_occ is not None:
                # Use occupation to select orbitals (prioritize occupied)
                occ_k = mo_occ[ik]
                idx_sorted = np.argsort(-occ_k)
                n_fill = min(nmo_available, self.nband)
                mo_to_use = mo_k[:, idx_sorted[:n_fill]]
            else:
                # Just use first nband orbitals
                n_fill = min(nmo_available, self.nband)
                mo_to_use = mo_k[:, :n_fill]
            
            n_fill_min = min(n_fill_min, n_fill)
            logger.debug(self, '  K-point %d: using %d MOs from provided coefficients', ik, n_fill)
            
            # Evaluate AOs at k-point on grid and transform to MOs
            ao_value = numint.eval_ao(self.cell, coords, kpt=kpt, deriv=0)
            mo_on_grid = np.dot(ao_value, mo_to_use)
            
            # Fill bands from MO projections
            for n in range(n_fill):
                psi_r_n = mo_on_grid[:, n].copy()
                self._normalize_and_store_orbital_r(psi_r_n, ik, n)
        
        # Fill remaining virtual bands with random orbitals
        self._fill_random_virtual_orbitals(n_fill_min, rng)
        
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

    def _apply_nuc(self, ik, psi_nk_g):
        """Apply nuclear (and pseudopotential) potential to a wavefunction.
        
        This function applies both:
        1. Local potential: V_ne(r) - includes bare nucleus and local pseudopotential
        2. Non-local potential: projector-based term from pseudopotential (if present)
        
        Formula:
        V_nuc|ψ_n⟩ = V_loc(r)|ψ_n⟩ + Σ_{ia,lm,n'} <p_{ia,l,n'}|ψ_n⟩ h_{l,n'n''} |p_{ia,l,n''}⟩
        
        where:
        - V_loc includes the local PP term (added to self._vne_R during build())
        - p_{ia,l,n'} are projector functions (if pseudopotential is present)
        - h_{l,n'n''} are matrix elements between projectors
        
        Args:
            ik: k-point index
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
            
        Returns:
            V_nuc|ψ_n⟩ in G-space (ngrids,)
        """
        cell = self.cell
        has_pseudo = hasattr(cell, '_pseudo') and cell._pseudo
        
        # 1. Apply local potential (nucleus + local PP if present)
        psi_nk_r = self._ifft_g2r(psi_nk_g)
        vnuc_psi_r = self._vne_R * psi_nk_r  # self._vne_R includes both nuclear and local PP
        vnuc_psi_g = self._fft_r2g(vnuc_psi_r)
        
        # 2. Apply non-local pseudopotential (if present)
        if has_pseudo:
            vnl_psi_g = self._apply_pp_nonlocal(ik, psi_nk_g)
            vnuc_psi_g += vnl_psi_g
        
        return vnuc_psi_g

    def _apply_pp_nonlocal(self, ik, psi_nk_g):
        """Apply non-local pseudopotential projector operator.
        
        Computes: V_nl|ψ_n⟩ = Σ_{ia,lm,n'} <p_{ia,l,n'}|ψ_n⟩ h_{l,n'n''} |p_{ia,l,n''}⟩
        
        Args:
            ik: k-point index
            psi_nk_g: (ngrids,) wavefunction in G-space
            
        Returns:
            V_nl|ψ_n⟩ in G-space (ngrids,)
        """
        from pyscf.pbc.gto.pseudo import pp
        from pyscf import gto
        
        cell = self.cell
        Gv = self._Gv
        kpt = self.kpts[ik]
        G_rad = lib.norm(Gv, axis=1)
        
        # Transform wavefunction to G-space (already in G-space, but need components)
        # Note: psi_nk_g is already in G-space
        
        vnl_g = np.zeros_like(psi_nk_g)
        
        # Setup fake molecule for evaluating projectors
        fakemol = gto.Mole()
        fakemol._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
        fakemol._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
        ptr = gto.PTR_ENV_START
        fakemol._env = np.zeros(ptr + 10)
        fakemol._bas[0, gto.NPRIM_OF] = 1
        fakemol._bas[0, gto.NCTR_OF] = 1
        fakemol._bas[0, gto.PTR_EXP] = ptr + 3
        fakemol._bas[0, gto.PTR_COEFF] = ptr + 4
        
        # Get structure factors for the selected k-point
        SI = cell.get_SI(mesh=self.mesh)  # (natm, ngrids)
        
        # Buffer for projectors (handle up to l=0..3, nl<=3)
        buf = np.empty((48, self.ngrids), dtype=np.complex128)
        
        # Loop over atoms
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            
            pp_data = cell._pseudo[symb]
            # pp_data structure: [Zeff, rloc, nexp, cexp, nproj, [l, rl, nl, hl], ...]
            # pp_data[5:] contains projector blocks: (l, rl, nl, hl)
            
            p1 = 0
            # Evaluate projector functions in G+k space
            Gk = Gv + kpt
            
            for l, proj in enumerate(pp_data[5:]):
                rl, nl, hl = proj
                
                if nl > 0:
                    # Setup fake molecule for angular momentum l
                    fakemol._bas[0, gto.ANG_OF] = l
                    fakemol._env[ptr + 3] = 0.5 * rl ** 2
                    fakemol._env[ptr + 4] = rl ** (l + 1.5) * np.pi ** 1.25
                    
                    # Evaluate Gaussian at |G+k|
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)  # (ngrids, ncomp)
                    
                    p0, p1 = p1, p1 + nl * (l * 2 + 1)
                    
                    # Compute radial part q_kl(|G+k|*rl) and multiply by Ylm
                    pYlm = np.ndarray((nl, l * 2 + 1, self.ngrids), 
                                     dtype=np.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pp._qli(G_rad * rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
            
            # Contract projectors with wavefunction and apply h matrix
            if p1 > 0:
                # SPG_lmi = structure_factor * projectors (complex), shape: (p1, ngrids)
                SPG_lmi = buf[:p1].copy()
                SPG_lmi *= SI[ia].conj()  # Apply structure factor for atom ia
                
                # Contract projectors with wavefunction in G-space
                # SPG_lm_aoGs = Σ_G SPG_lm(G) * ψ(G) 
                # This is a simple dot product: (p1, ngrids) @ (ngrids,) -> (p1,)
                SPG_lm_aoGs = np.dot(SPG_lmi, psi_nk_g)  # Direct matrix-vector product
                
                # Apply h matrix and accumulate
                p1_reset = 0
                for l, proj in enumerate(pp_data[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1_reset = p1_reset, p1_reset + nl * (l * 2 + 1)
                        hl = np.asarray(hl)
                        
                        # Reshape: (nl*(2l+1),) -> (nl, 2l+1)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1_reset].reshape(nl, l * 2 + 1)
                        
                        # Apply h matrix: (nl, nl) @ (nl, 2l+1) -> (nl, 2l+1)
                        tmp = np.einsum('ij,jm->im', hl, SPG_lm_aoG)
                        
                        # Contract back: buf[p0:p1_reset] has shape (nl*(2l+1), ngrids)
                        # Reshape tmp back to (nl*(2l+1),) for contraction
                        tmp_flat = tmp.ravel()
                        proj_back = np.dot(buf[p0:p1_reset].T.conj(), tmp_flat)
                        proj_back *= SI[ia]  # Apply structure factor to shift to atom position
                        vnl_g += proj_back
        
        # Normalize by volume (from pseudopotential convention)
        vnl_g *= (1.0 / cell.vol)
        
        return vnl_g

    def apply_fock(self, ik,  psi_nk_g, rho_r=None, with_j=True, with_k=True):
        """Apply Fock operator to a single wavefunction.
        
        F = h + J - K/2 (for closed-shell RHF)
        
        where:
        - h = T + V_ne (one-electron Hamiltonian)
        - J = Hartree (Coulomb) operator
        - K = exchange operator (note: _apply_k returns negative values, so we add +0.5*K)
        
        The Fock operator gives orbital energies as eigenvalues: F|ψ⟩ = ε|ψ⟩
        This is different from the full many-electron Hamiltonian H = h + J + K.
        
        Args:
            ik: k-point index
            psi_nk_g: (ngrids,) wavefunction in G-space for band n at k-point ik
            rho_r: (ngrids,) total electron density in real space (if None, computed from self.psi_r)
            with_j: whether to include Hartree term
            with_k: whether to include exchange term
        Returns:
            F|ψ⟩ in G-space (ngrids,)
        """
        F_psi_g = np.zeros_like(psi_nk_g)
        
        # 1. Apply Kinetic Energy (Diagonal in G-space)
        # T = 0.5 * |k + G|^2
        t_g = self._kin_diag[ik]  # (ngrids,)
        F_psi_g += t_g * psi_nk_g
        
        # 2. Apply Nuclear and Pseudopotential Potential (local + non-local)
        vnuc_psi_g = self._apply_nuc(ik, psi_nk_g)
        F_psi_g += vnuc_psi_g
        
        # Get density (use provided or compute from current state)
        if rho_r is None:
            rho_r = self.get_density_r()
        
        # 3. Apply Hartree Potential J (electron-electron repulsion, direct term)
        if with_j:
            # V_H(r) from total density
            rho_g = self._fft_density_r2g(rho_r)
            coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
            vH_g = coulG * rho_g
            vH_r = self._ifft_potential_g2r(vH_g).real
            psi_nk_r = self._ifft_g2r(psi_nk_g)
            vH_psi_r = vH_r * psi_nk_r
            F_psi_g += self._fft_r2g(vH_psi_r)
        
        # 4. Apply Exchange term: -K/2 (Fock operator has opposite sign and factor 1/2)
        # Note: _apply_k returns K|ψ⟩ with negative values (attractive)
        # For Fock: F = h + J - K_positive/2 = h + J + K_negative/2
        if with_k:
            K_psi_g = self._apply_k(ik, psi_nk_g)
            F_psi_g += K_psi_g  # Factor of 0.5 for Fock operator
        
        return F_psi_g

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
        
        
        # Precompute Hartree potential and energy from total density
        rho_r = self.get_density_r()  
        rho_g = self.get_density_G()
        coulG = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        vH_g = coulG * rho_g
        vH_r = self._ifft_g2r(vH_g).real
        
        E_hartree_r = 0.5 * np.sum(rho_r * vH_r).real * self.grid_weight
        
        E_hartree = E_hartree_r
        
        # Initialize other energy components
        E_kin = 0.0
        E_ne = 0.0
        E_exchange = 0.0
        E_exchange_g = 0.0  # G-space version for comparison
        
        # Debug: check if we have occupied orbitals
        if self.nocc == 0:
            raise RuntimeError("No occupied orbitals (self.nocc = 0)!")
        
        # Loop over k-points and occupied orbitals
        for ik in range(self.nk):
            for n in range(self.nocc):
                logger.debug(self, f"Computing energy contributions for k-point {ik}, band {n}")
                psi_g = self.psi_g[ik, n].copy()
                psi_r = self._ifft_g2r(psi_g)
                
                # Kinetic energy: <ψ|T|ψ> = Σ_G T(G) |ψ(G)|²
                # With L2 normalization (Σ|ψ_g|²=1), we compute directly without grid_weight
                t_diag = self._kin_diag[ik]
                E_kin_contrib = occ_weight * np.sum((t_diag * np.abs(psi_g)**2).real)
                E_kin += E_kin_contrib
                
                # Nuclear-electron energy: <ψ|V_nuc|ψ> includes both local and non-local PP
                # Use _apply_nuc to get the full nuclear potential operator applied to ψ
                Vnuc_psi_g = self._apply_nuc(ik, psi_g)
                E_ne_contrib = occ_weight * np.vdot(psi_g, Vnuc_psi_g).real
                E_ne += E_ne_contrib
                
                
                if with_k:
                    # Exchange in R-space
                    K_psi_g = self._apply_k(ik, psi_g)
                    K_psi_r = self._ifft_g2r(K_psi_g)
                    K_psi_integral_r = np.sum(psi_r.conj() * K_psi_r).real * self.grid_weight
                    E_x_contrib_r = 0.5 * occ_weight * K_psi_integral_r  # 0.5 factor, NO minus sign
                    E_exchange += E_x_contrib_r
                    
                    # Exchange in G-space: <ψ|K|ψ> = Σ_G ψ*(G) K_psi(G)
                    # Since both psi_g and K_psi_g are in G-space with L2 norm, we use:
                    # <ψ|K|ψ> = Σ_G ψ*(G) K_psi(G) (no grid_weight needed)
                    K_psi_integral_g = np.vdot(psi_g, K_psi_g).real
                    E_x_contrib_g = 0.5 * occ_weight * K_psi_integral_g
                    E_exchange_g += E_x_contrib_g
                    
        
        
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
            #logger.debug(self, f"Ewald exchange correction: madelung={madelung:.6e}, "
            #            f"N_elec={self.cell.nelectron}, correction={E_ewald_correction:.6e}")
            E_exchange += E_ewald_correction
            E_exchange_g += E_ewald_correction
            
            #logger.debug(self, f"Total Exchange (R-space): {E_exchange:.10f}")
            #logger.debug(self, f"Total Exchange (G-space): {E_exchange_g:.10f}")
            #logger.debug(self, f"Total Exchange difference (R-G): {abs(E_exchange - E_exchange_g):.2e}")
        
        # Total energy
        E_tot = E_kin + E_ne + E_hartree + E_exchange + E_nuc
        
        return {
            'E_kin': E_kin,
            'E_ne': E_ne,
            'E_hartree': E_hartree,
            'E_exchange': E_exchange,
            'E_exchange_g': E_exchange_g,  # G-space version for debugging
            'E_nuc': E_nuc,
            'E_tot': E_tot
        }

    def _apply_k(self, ik, psi_nk_g):
        """Apply exchange operator K to wavefunction for closed-shell system.
        
        In Fourier space:
        1. Compute pair density ρ_{nm}(r) = ψ_m*(r) * ψ_n(r)
        2. Solve Poisson in G-space: V_{nm}(G) = (4π/|G|^2) FFT[ρ_{nm}(r)]
        3. Transform back: V_{nm}(r) = IFFT[V_{nm}(G)]
        
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
            k_psi_r = -V_ij_r * psi_occ_r  # (ngrids,)
            k_psi_g += self._fft_r2g(k_psi_r)  # (ngrids,)
        
        return k_psi_g

    # ----------------------------- Solvers ----------------------------- #
    def _block_davidson(self, ik, with_k=True, max_cycle=5, tol=1e-6, 
                        scf_iter=1, rho_r=None):
        """Block Davidson diagonalization for all nband orbitals at once.
        
        Solves the eigenvalue problem for the Fock operator in a subspace of
        trial vectors. Uses QR orthonormalization and diagonal preconditioning.
        
        Args:
            ik: k-point index
            with_k: whether to include exchange operator
            max_cycle: maximum Davidson iterations
            scf_iter: current SCF iteration (for logging)
            rho_r: electron density in real space (optional, computed if not provided)
            
        Returns:
            (psi_new, e_sorted) where:
            - psi_new: (nband, ngrids) converged eigenstates in G-space
            - e_sorted: (nband,) corresponding eigenvalues
        """
        log = logger.new_logger(self, self.verbose)
        
        # Parameters
        max_subspace_size = 3 * self.nband
        
        # Initialize subspace with current orbitals
        subspace = [self.psi_g[ik, n].copy() for n in range(self.nband)]
        
        for davidson_iter in range(max_cycle):
            nv = len(subspace)
            
            # 1. Orthonormalize subspace using QR
            subspace_matrix = np.column_stack([v.reshape(-1) for v in subspace])
            Q, _ = np.linalg.qr(subspace_matrix)
            subspace = [Q[:, i] for i in range(Q.shape[1])]
            nv = len(subspace)
            
            # 2. Build subspace Hamiltonian: H_ij = ⟨v_i|F|v_j⟩
            H_subspace = np.zeros((nv, nv), dtype=complex)
            for i in range(nv):
                F_vi = self.apply_fock(ik, subspace[i], rho_r=rho_r, with_j=True, with_k=with_k)
                for j in range(nv):
                    H_subspace[j, i] = np.vdot(subspace[j], F_vi)
            
            # 3. Diagonalize subspace Hamiltonian
            e, c = np.linalg.eigh(H_subspace)
            idx = np.argsort(e.real)[:self.nband]
            e_sorted = e.real[idx]
            
            # 4. Construct new wavefunctions from subspace
            psi_new = np.zeros((self.nband, self.ngrids), dtype=complex)
            for n in range(self.nband):
                for i in range(nv):
                    psi_new[n] += c[i, idx[n]] * subspace[i]
                # Normalize in G-space
                norm = np.sqrt(np.sum(np.abs(psi_new[n])**2))
                if norm > 1e-10:
                    psi_new[n] /= norm
            
            # 5. Compute residuals and check convergence
            residuals = []
            max_res_norm = 0.0
            for n in range(self.nband):
                F_psi_n = self.apply_fock(ik, psi_new[n], rho_r=rho_r, with_j=True, with_k=with_k)
                R_n = F_psi_n - e_sorted[n] * psi_new[n]
                res_norm = np.sqrt(np.sum(np.abs(R_n)**2))
                max_res_norm = max(max_res_norm, res_norm)
                residuals.append(R_n)
            
            logger.debug1(self, '    Davidson iter %d: max_res = %.3e', davidson_iter + 1, max_res_norm)
            
            if max_res_norm < tol:
                logger.debug(self, ' Converged in %d iterations, max_res = %.3e', 
                        davidson_iter + 1, max_res_norm)
                break
            
            # 6. Precondition residuals and expand subspace
            hdiag = self._kin_diag[ik]
            for n in range(self.nband):
                shift = 0.001
                precond_denom = hdiag - (e_sorted[n] + shift)
                precond_denom[np.abs(precond_denom) < 1e-8] = 1e-8
                P_n = residuals[n] / precond_denom
                
                # Normalize and add to subspace
                norm_p = np.sqrt(np.sum(np.abs(P_n)**2))
                if norm_p > 1e-10:
                    P_n /= norm_p
                    subspace.append(P_n)
            
            # 7. Restart if subspace too large
            if len(subspace) > max_subspace_size:
                logger.debug(self, '    Subspace size %d > %d, restarting', 
                         len(subspace), max_subspace_size)
                subspace = [psi_new[n].copy() for n in range(self.nband)]
        
        return psi_new, e_sorted
    

    def kernel(self, init='minao', max_cycle=50, conv_tol=1e-7, conv_tol_rho=1e-6,
               with_k=True, davidson_tol=1e-6, davidson_max_cycle=5, mo_coeff=None, mo_occ=None,
               alpha=0.5
               ):
        """Self-consistent HF loop in G-space (J and optional K); no XC.
        
        Uses block Davidson diagonalization to solve for eigenstates at each SCF iteration.

        Args:
            init: initialization method ('random', 'minao', 'atom')
            max_cycle: maximum number of SCF iterations
            conv_tol: energy convergence tolerance
            conv_tol_rho: density convergence tolerance
            with_k: include exchange operator
            davidson_tol: tolerance for Davidson diagonalization (residual norm)
            davidson_max_cycle: max Davidson iterations per SCF cycle
            mo_coeff: MO coefficients for initialization (optional)
            mo_occ: MO occupations for initialization (optional)
            alpha: density mixing parameter (0 < alpha <= 1). Smaller = more stable, larger = faster.
                   rho_new = alpha * rho_computed + (1-alpha) * rho_old
            
        Returns:
            (E_tot, converged) tuple
        """
        # Build operators if not already built
        if not hasattr(self, '_vne_R') or self._vne_R is None:
            self.build()
        
        # Initialize if requested
        if init is not None:
            self.init_guess(kind=init, mo_coeff=mo_coeff, mo_occ=mo_occ)
        
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
            energy_dict_init = self.compute_energy_components(with_k=with_k)
            e_init = energy_dict_init['E_tot']
            logger.info(self, '  Initial E_tot = %.10f Ha', e_init)
            logger.debug(self, '    E_kin     = %.10f', energy_dict_init['E_kin'])
            logger.debug(self, '    E_ne      = %.10f', energy_dict_init['E_ne'])
            logger.debug(self, '    E_hartree = %.10f', energy_dict_init['E_hartree'])
            logger.debug(self, '    E_exchange= %.10f', energy_dict_init['E_exchange'])
            logger.debug(self, '    E_nuc     = %.10f', energy_dict_init['E_nuc'])
            
            # Compute and print initial orbital energies <ψ|F|ψ>
            logger.debug(self, '  Initial orbital energies (before Davidson):')
            for ik in range(self.nk):
                for n in range(self.nband):
                    psi_g_n = self.psi_g[ik, n]
                    # Apply Fock operator
                    F_psi_g = self.apply_fock(ik, psi_g_n, with_j=True, with_k=with_k)
                    # Compute <ψ|F|ψ>
                    e_orbital = np.vdot(psi_g_n.conj(), F_psi_g).real
                    self.mo_energy[ik, n] = e_orbital
                    if n < 4:  # Print first few
                        logger.debug(self, '    k=%d, band %d: e = %.6f Ha', ik, n, e_orbital)
            
        
        # SCF loop
        e_tot_prev = 0.0
        rho_r_prev = None
        rho_r_mixed = None  # For density mixing
        converged = False
        
        for scf_iter in range(1, max_cycle + 1):
            # Get density from current orbitals
            rho_r_computed = self.get_density_r()
            
            # Apply density mixing (except first iteration)
            if rho_r_prev is not None and alpha < 1.0:
                rho_r = alpha * rho_r_computed + (1.0 - alpha) * rho_r_prev
            else:
                rho_r = rho_r_computed
            
            # Compute density change for convergence check
            if rho_r_prev is not None:
                drho = rho_r_computed - rho_r_prev
                rho_norm = np.linalg.norm(drho) * np.sqrt(self.grid_weight)
            else:
                rho_norm = 1.0
            
            # Solve orbitals using block Davidson
            for ik in range(self.nk):
                logger.debug(self, '  k-point %d/%d: Block Davidson diagonalization...', ik + 1, self.nk)
                
                psi_new, e_sorted = self._block_davidson(
                    ik, 
                    with_k=with_k,
                    max_cycle=davidson_max_cycle,
                    tol=davidson_tol,
                    scf_iter=scf_iter,
                    rho_r=rho_r
                )
                
                # Update wavefunctions and energies
                for n in range(self.nband):
                    self.psi_g[ik, n] = psi_new[n]
                    self.psi_r[ik, n] = self._ifft_g2r(psi_new[n])
                    self.mo_energy[ik, n] = e_sorted[n]
                
                logger.debug(self, '    mo_energy: %s', 
                            self.mo_energy[ik, :self.nband])
            
            
            energy_dict = self.compute_energy_components(with_k=with_k)
            e_tot = energy_dict['E_tot']
            E_kin = energy_dict['E_kin']
            E_ne = energy_dict['E_ne']
            E_hartree = energy_dict['E_hartree']
            E_exchange = energy_dict['E_exchange']
            E_nuc = energy_dict['E_nuc']
            
            # 6. Check energy convergence
            de = e_tot - e_tot_prev
            logger.info(self, 'Cycle %3d: E = %8.10f  dE = %+.6e  |dρ| = %.6e',
                        scf_iter, e_tot, de, rho_norm)
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
            rho_r_prev = rho_r_computed.copy()  # Save the computed density (before mixing)
        
        if not converged:
            logger.warn(self, '\n*** SCF NOT converged after %d iterations ***', max_cycle)
        
        # Print final summary
        logger.info(self, '\n' + '=' * 60)
        logger.info(self, 'Final Results:')
        logger.info(self, '  Total Energy: %.10f Ha', e_tot)
        logger.info(self, '  Converged: %s', converged)
        for ik in range(self.nk):
            logger.info(self, '  k-point %d orbital energies :', ik)
            logger.info(self, '    %s', self.mo_energy[ik, :self.nband])
        logger.info(self, '=' * 60)
        
        return e_tot, converged


__all__ = ['KPWSCF']


