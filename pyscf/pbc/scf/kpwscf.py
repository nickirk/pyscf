#!/usr/bin/env python
# Copyright 2025 The PySCF Developers.
#

import numpy as np
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
        self.mesh = self._select_mesh(mesh)
        # Use UniformGrids for consistent grid weights
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
        self.nband = int(nband)
        self.nocc = min(int(np.ceil(nelec/2)), self.nband)

        # Wavefunctions on grid (complex)
        self.psi_r = None  # shape (nk, nband, ngrids)
        self.psi_g = None  # shape (nk, nband, ngrids)
        self.mo_energy = None  # shape (nk, nband)

        # Cached operators
        self._Gv = None             # (ngrids, 3)
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

    # ----------------------------- utilities ----------------------------- #
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
        self._precompute_Gv()
        self._precompute_kinetic()
        self._build_vne_R()
        return self

    def _precompute_Gv(self):
        if self._Gv is None:
            self._Gv = self.cell.get_Gv(self.mesh)

    def _precompute_kinetic(self):
        """Diagonal kinetic operator in G for each k: 0.5*|G+k|^2."""
        Gv = self._Gv
        self._kin_diag = np.empty((self.nk, self.ngrids), dtype=float)
        for ik, kpt in enumerate(self.kpts):
            Gk = Gv + kpt  # broadcast
            self._kin_diag[ik] = 0.5 * np.einsum('ij,ij->i', Gk, Gk)

    def _build_vne_R(self):
        """Ewald nuclear potential on the uniform grid (real)."""
        from pyscf.pbc.df.aft import _check_kpts  # for shape norm; no heavy import
        log = logger.new_logger(self, self.verbose)
        mesh = self.mesh
        cell = self.cell

        # Structure factors SI (natm, ngrids), atomic charges
        SI = cell.get_SI(mesh=mesh)  # complex
        charge = -cell.atom_charges()
        rhoG = charge @ SI  # (ngrids,)

        coulG = pbctools.get_coulG(cell, mesh=mesh)  # (ngrids,)
        vneG = rhoG * coulG
        vneR = pbctools.ifft(vneG, mesh).real  # (ngrids,)
        self._vne_R = vneR
        # Cache Coulomb kernel for later Hartree / exchange
        self._coulG0 = coulG
        log.debug1('Built V_ne on grid; min/max %.6g / %.6g', vneR.min(), vneR.max())

    # FFT wrappers (batch over bands)
    def _fft_r2g(self, psi_r):
        # psi_r: (nband, ngrids)
        return pbctools.fft(psi_r, self.mesh)

    def _ifft_g2r(self, psi_g):
        # psi_g: (nband, ngrids)
        return pbctools.ifft(psi_g, self.mesh)

    def _kerker(self, drho_R, k_screen=1.0):
        """Kerker preconditioner in G-space for density residual.

        Scales long-wavelength components to mitigate charge sloshing:
        f(G) = |G|^2 / (|G|^2 + k_screen^2).
        """
        drhoG = pbctools.fft(drho_R.reshape(1, -1), self.mesh)[0]
        G2 = np.einsum('ij,ij->i', self._Gv, self._Gv)
        fac = G2 / np.maximum(1e-12, G2 + k_screen * k_screen)
        drhoG *= fac
        out = pbctools.ifft(drhoG, self.mesh).real
        return out

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
        self.psi_g = np.empty_like(self.psi_r)
        for ik in range(nk):
            self.psi_g[ik] = self._fft_r2g(self.psi_r[ik])
        return self

    def _orthonormalize_block(self, psi_r_block):
        """Orthonormalize a (nband, ngrids) block with uniform-grid inner product.

        We use standard QR on Euclidean inner product; the uniform weight is a
        constant factor and does not affect orthogonality across vectors.
        Final normalization is adjusted so that ∫|ψ|^2 dR ≈ 1.
        """
        # QR factorization (columns are grid points; we need orthonormal rows),
        # so operate on transposed and transpose back.
        Q, _ = np.linalg.qr(psi_r_block.T)  # (ngrids, nband)
        psi = Q.T.astype(np.complex128, copy=False)
        # Normalize to integral 1 with grid weight
        norms = np.sqrt(np.maximum(1e-300, (np.abs(psi)**2).sum(axis=1) * self.grid_weight))
        psi /= norms[:, None]
        return psi

    # ----------------------------- operators ----------------------------- #
    def _apply_hcore(self, ik, psi_r_k):
        """Apply H_core = T + V_ne to a (nband, ngrids) block at k.

        Returns Hψ in real-space grid representation.
        """
        # T in G-space
        psi_g = self._fft_r2g(psi_r_k)
        t_g = self._kin_diag[ik]  # (ngrids,)
        tpsi_g = psi_g * t_g[None, :]
        tpsi_r = self._ifft_g2r(tpsi_g)
        # V_ne in R-space
        vpsi_r = psi_r_k * self._vne_R[None, :]
        return tpsi_r + vpsi_r

    def _apply_hcore_eff(self, ik, psi_r_k, v_eff_R):
        """Apply H_eff = T + V_eff(R) on (nband, ngrids) at k.

        v_eff_R: (ngrids,) real array (e.g., V_ne + V_H + V_xc)
        Returns (nband, ngrids) in real space.
        """
        psi_g = self._fft_r2g(psi_r_k)
        tpsi_r = self._ifft_g2r(psi_g * self._kin_diag[ik][None, :])
        return tpsi_r + psi_r_k * v_eff_R[None, :]

    def get_density_R(self):
        """Total electron density on the grid from current ψ (closed shell).

        ρ(R) = Σ_{k,n≤nocc} f_kn |ψ_kn(R)|^2 with f_kn = 2/nk.
        Returns (ngrids,) real array.
        """
        if self.psi_r is None:
            raise RuntimeError('Call init_guess() first')
        rho = np.zeros(self.ngrids, dtype=float)
        occ_weight = 2.0 / self.nk
        for ik in range(self.nk):
            psi = self.psi_r[ik, :self.nocc]  # (nocc, ngr)
            rho += occ_weight * (np.abs(psi)**2).sum(axis=0)
        return rho

    def get_vhartree_R(self, rho_R):
        """Hartree potential on grid from density.

        vH(G) = coulG(G) * ρ(G); vH(R) = ifft(vH(G)).real
        Use k=0 Coulomb kernel for Hartree.
        """
        rhoG = pbctools.fft(rho_R.reshape(1, -1), self.mesh)[0]
        coulG0 = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        vHG = rhoG * coulG0
        vHR = pbctools.ifft(vHG, self.mesh).real
        return vHR

    # ----------------------------- energies ----------------------------- #
    def _expect_from_apply(self, psi_r_k, op_psi_r_k):
        """Return <psi|op|psi> for a block (nband, ngrids) with grid weights.

        Assumes ψ are orthonormal per band; returns array (nband,).
        """
        # Weighted inner product over grid
        val = np.einsum('in,in->i', psi_r_k.conj(), op_psi_r_k)
        return self.grid_weight * val.real

    def energy_terms(self, v_eff_R=None, rho_R=None, vH_R=None, with_k=False, debug=False):
        """Compute energy components for current ψ and potentials (HF: J and optional K).

        Returns dict with keys: E_T, E_ne, E_H, E_x, E_one_e, E_tot, E_nn.
        When debug=True, also returns E_ne_rho (from density*V_ne) and basic v_ne stats.
        vH_R, rho_R can be provided to avoid recomputation.
        """
        if v_eff_R is None:
            v_eff_R = self._vne_R
        if rho_R is None:
            rho_R = self.get_density_R()
        if vH_R is None:
            vH_R = self.get_vhartree_R(rho_R)

        E_T = 0.0
        E_ne = 0.0
        # Sum occupied expectations
        occ_weight = 2.0 / self.nk
        zv = np.zeros_like(v_eff_R)
        for ik in range(self.nk):
            psi = self.psi_r[ik]
            # T part
            tpsi = self._apply_hcore_eff(ik, psi, zv)
            e_t_bands = self._expect_from_apply(psi, tpsi)
            # V_ne part
            vnepsi = psi * self._vne_R[None, :]
            e_ne_bands = self._expect_from_apply(psi, vnepsi)
            E_T += occ_weight * e_t_bands[:self.nocc].sum()
            E_ne += occ_weight * e_ne_bands[:self.nocc].sum()

        # Hartree energy: 0.5 ∫ ρ vH
        E_H = self.grid_weight * float(np.dot(rho_R, vH_R))

        # Exchange energy (gamma-only implementation)
        E_x = 0.0
        if with_k:
            if self.nk != 1:
                raise NotImplementedError('Exchange (K) currently implemented for gamma-only (nk=1)')
            # Build K|phi_i> for occupied orbitals and compute -0.5 * sum_i <phi_i|K|phi_i>
            occ_weight = 2.0 / self.nk
            phi_occ = self.psi_r[0, :self.nocc]
            Kphi = self._apply_exchange_gamma(phi_occ)
            for i in range(self.nocc):
                e_i = self.grid_weight * float(np.vdot(phi_occ[i], Kphi[i]).real)
                E_x += -0.5 * occ_weight * e_i

        E_one_e = E_T + E_ne
        E_tot = E_T + E_ne + E_H + E_x
        # Nuclear-nuclear Ewald (repulsion)
        try:
            E_nn = float(self.cell.energy_nuc())
        except Exception:
            E_nn = 0.0
        out = dict(E_T=E_T, E_ne=E_ne, E_H=E_H, E_x=E_x,
                   E_one_e=E_one_e, E_tot=E_tot, E_nn=E_nn,
                   E_tot_plus_nuc=E_tot+E_nn)
        if debug:
            E_ne_rho = self.grid_weight * float(np.dot(rho_R, self._vne_R))
            vne = self._vne_R
            out.update(E_ne_rho=E_ne_rho,
                       vne_mean=float(vne.mean()), vne_min=float(vne.min()), vne_max=float(vne.max()))
        return out

    def _apply_exchange_gamma(self, psi_r_block, occ_orbs=None):
        """Apply exchange operator K (gamma-only) to a block of vectors.

        psi_r_block: (nb, ngrids)
        occ_orbs: optional (nocc, ngrids); defaults to current occupied at k=Gamma.
        Returns Kpsi with shape (nb, ngrids).
        """
        if self.nk != 1:
            raise NotImplementedError('Exchange (K) is only available for gamma-only (nk=1)')
        if occ_orbs is None:
            occ_orbs = self.psi_r[0, :self.nocc]
        coulG0 = self._coulG0 if self._coulG0 is not None else pbctools.get_coulG(self.cell, mesh=self.mesh)
        nb, ngr = psi_r_block.shape
        Kpsi = np.zeros((nb, ngr), dtype=np.complex128)
        for phi in occ_orbs:
            # chi_beta(r) = phi*(r) * psi_beta(r) for all beta in block
            chi = psi_r_block * phi.conj()[None, :]
            chiG = pbctools.fft(chi, self.mesh)
            uG = chiG * coulG0[None, :]
            uR = pbctools.ifft(uG, self.mesh)
            Kpsi += phi[None, :] * uR
        return Kpsi

    def _apply_fock(self, ik, psi_r_k, v_eff_R, with_k=False):
        """Apply Fock operator F = T + V_eff - K (if with_k and supported)."""
        # One-body part
        Fpsi = self._apply_hcore_eff(ik, psi_r_k, v_eff_R)
        # Exchange
        if with_k and self.nk == 1 and ik == 0:
            Fpsi -= self._apply_exchange_gamma(psi_r_k)
        return Fpsi

    def _precondition_residual(self, ik, R_r, kappa=1.0):
        """Simple diagonal preconditioner in G-space: divide by T+κ."""
        Rg = self._fft_r2g(R_r)
        den = self._kin_diag[ik][None, :] + float(kappa)
        Rg /= den
        return self._ifft_g2r(Rg)

    def _iterate_bands(self, ik, v_eff_R, with_k, nsweeps=2, step=0.5, prec_kappa=1.0,
                      occ_only=True, step_clip=0.1):
        """Few sweeps of preconditioned steepest descent + subspace rotation.

        occ_only: update only occupied bands in SD step (stabilizes density)
        step_clip: clip factor for max SD move per band (in L2, weighted)
        """
        psi = self.psi_r[ik]
        for _ in range(max(1, int(nsweeps))):
            # Build subspace and rotate to (approx) diagonal form
            Hpsi = self._apply_fock(ik, psi, v_eff_R, with_k)
            Hsub = (psi.conj() @ Hpsi.T) * self.grid_weight
            #Hsub = 0.5 * (Hsub + Hsub.conj().T)
            #print(Hsub)
            eig, U = np.linalg.eigh(Hsub)
            idx = np.argsort(eig)
            eig, U = eig[idx], U[:, idx]
            #print("eigenvalues:", eig)
            psi = (U.conj().T @ psi)
            #psi = self._orthonormalize_block(psi)
            # Residual for occupied + a few virtuals (use all bands available)
            Hpsi = self._apply_fock(ik, psi, v_eff_R, with_k)
            res = Hpsi - psi * eig[:, None]
            # Optionally only update occupied subspace to avoid large density swings
            if occ_only:
                res[self.nocc:] = 0.0
            # Precondition and move
            res_p = self._precondition_residual(ik, res, kappa=prec_kappa)
            # Step-size clipping based on residual norms (weighted L2)
            psi = psi - float(step) * res_p
            #psi = self._orthonormalize_block(psi)
        # Final subspace diag to update eigenvalues
        Hpsi = self._apply_fock(ik, psi, v_eff_R, with_k)
        Hsub = (psi.conj() @ Hpsi.T) * self.grid_weight
        Hsub = 0.5 * (Hsub + Hsub.conj().T)
        eig, U = np.linalg.eigh(Hsub)
        idx = np.argsort(eig)
        eig, U = eig[idx], U[:, idx]
        psi = (U.conj().T @ psi)
        psi = self._orthonormalize_block(psi)
        self.psi_r[ik] = psi
        self.psi_g[ik] = self._fft_r2g(psi)
        if self.mo_energy is None:
            self.mo_energy = np.zeros((self.nk, self.nband))
        self.mo_energy[ik] = eig
        return eig

    def _ortho_metrics(self):
        """Compute orthonormality diagnostics in R and G spaces for all k.

        Returns a dict with max deviations across k-points:
        - max_diag_dev_r/g: max |S_ii - 1| in real/G spaces
        - max_offdiag_r/g: max |S_ij| for i!=j in real/G spaces
        """
        if self.psi_r is None or self.psi_g is None:
            return dict(max_diag_dev_r=np.nan, max_offdiag_r=np.nan,
                        max_diag_dev_g=np.nan, max_offdiag_g=np.nan)
        gw_r = self.grid_weight
        # Parseval scaling for our FFT convention (fft:1, ifft:1/N)
        N = float(self.ngrids)
        gw_g = self.vol / (N * N)
        max_diag_dev_r = 0.0
        max_offdiag_r = 0.0
        max_diag_dev_g = 0.0
        max_offdiag_g = 0.0
        I = None
        for ik in range(self.nk):
            psi_r = self.psi_r[ik]  # (nb, ngr)
            psi_g = self.psi_g[ik]  # (nb, ngr)
            # Real-space Gram matrix
            S_r = (psi_r.conj() @ psi_r.T) * gw_r
            # G-space Gram matrix (scaled so that <psi|psi> matches R-space)
            S_g = (psi_g.conj() @ psi_g.T) * gw_g
            if I is None or I.shape != S_r.shape:
                I = np.eye(S_r.shape[0])
            # Diagonal deviations from 1
            dr = np.max(np.abs(np.diag(S_r) - 1.0))
            dg = np.max(np.abs(np.diag(S_g) - 1.0))
            # Off-diagonal magnitudes
            S_r_off = S_r - np.diag(np.diag(S_r))
            S_g_off = S_g - np.diag(np.diag(S_g))
            or_max = np.max(np.abs(S_r_off)) if S_r_off.size else 0.0
            og_max = np.max(np.abs(S_g_off)) if S_g_off.size else 0.0
            max_diag_dev_r = max(max_diag_dev_r, float(dr))
            max_diag_dev_g = max(max_diag_dev_g, float(dg))
            max_offdiag_r = max(max_offdiag_r, float(or_max))
            max_offdiag_g = max(max_offdiag_g, float(og_max))
        return dict(max_diag_dev_r=max_diag_dev_r,
                    max_offdiag_r=max_offdiag_r,
                    max_diag_dev_g=max_diag_dev_g,
                    max_offdiag_g=max_offdiag_g)

    class _SimpleDIIS:
        """Minimal Pulay DIIS for scalar fields on uniform grids.

        Stores pairs (vec_i, err_i) and extrapolates vec.
        Inner products are weighted by grid_weight.
        """
        def __init__(self, space=6, weight=1.0):
            self.space = int(space)
            self.weight = float(weight)
            self._vecs = []
            self._errs = []

        def push(self, vec, err):
            self._vecs.append(np.array(vec, copy=True))
            self._errs.append(np.array(err, copy=True))
            if len(self._vecs) > self.space:
                self._vecs.pop(0)
                self._errs.pop(0)

        def extrapolate(self, grid_weight):
            m = len(self._errs)
            if m < 2:
                return self._vecs[-1]
            # Build B matrix
            B = np.empty((m+1, m+1), dtype=float)
            B[0, 0] = 0.0
            B[0, 1:] = -1.0
            B[1:, 0] = -1.0
            for i in range(m):
                for j in range(m):
                    B[i+1, j+1] = grid_weight * float(np.dot(self._errs[i], self._errs[j]))
            rhs = np.zeros(m+1, dtype=float)
            rhs[0] = -1.0
            try:
                coef = np.linalg.solve(B, rhs)[1:]
            except np.linalg.LinAlgError:
                return self._vecs[-1]
            v = np.zeros_like(self._vecs[0])
            for c, vi in zip(coef, self._vecs):
                v += c * vi
            return v

    # ----------------------------- SCF loop (J only) ----------------------------- #
    def kernel_scf(self, init='random', max_cycle=50, conv_tol=1e-7, conv_tol_rho=1e-6,
                      alpha=0.3, with_k=True, diis_space=6, diis_start_cycle=2, kerker_k=1.0,
                      band_nsweeps=2, band_step=0.6, band_prec_kappa=1.0, trace=False):
        """Self-consistent HF loop (J and optional K); no XC.

        alpha: linear mixing parameter for density (0<alpha<=1)
        with_k: include exchange operator (gamma-only for now)
        Returns (E_tot, converged)
        """
        log = logger.new_logger(self, self.verbose)
        log_iter = log.note if trace else log.info
        self.build()
        self.init_guess(kind=init)

        # Initial diagonalization with V_ne only
        self.diagonalize_hcore()
        rho = self.get_density_R()
        vH = self.get_vhartree_R(rho)
        v_eff = self._vne_R + vH
        e_prev = None
        diis = self._SimpleDIIS(space=diis_space) if (diis_space and diis_space > 1) else None

        for ic in range(1, max_cycle+1):
            # Solve bands for current V_eff
            for ik in range(self.nk):
                self._iterate_bands(ik, v_eff, with_k, nsweeps=band_nsweeps, step=band_step, prec_kappa=band_prec_kappa)

            # New density and potential
            rho_new = self.get_density_R()
            vH_new = self.get_vhartree_R(rho_new)
            v_eff_new = self._vne_R + vH_new

            # Energies
            eterms = self.energy_terms(v_eff_R=v_eff_new, rho_R=rho_new, vH_R=vH_new, with_k=with_k)
            e_tot = eterms['E_tot']

            # Convergence metrics
            drho = rho_new - rho
            norm_drho = np.sqrt(self.grid_weight * float(np.dot(drho, drho)))
            de = np.inf if e_prev is None else abs(e_tot - e_prev)
            msg1 = f" KPWSCF iter {ic:3d}  E_tot = {e_tot:.12f}  dE = {de:.3e}  ||dρ|| = {norm_drho:.3e}"
            msg2 = (
                f"   components: E_T={eterms['E_T']:.8f}  E_ne={eterms['E_ne']:.8f}  E_H={eterms['E_H']:.8f}" +
                (('' if not with_k else f"  E_x={eterms['E_x']:.8f}"))
            )
            log_iter(msg1)
            log_iter(msg2)
            # Orthonormality diagnostics
            ortho = self._ortho_metrics()
            msg_ortho = (f"   ortho: R diag-dev={ortho['max_diag_dev_r']:.2e} offdiag={ortho['max_offdiag_r']:.2e}; "
                      f"G diag-dev={ortho['max_diag_dev_g']:.2e} offdiag={ortho['max_offdiag_g']:.2e}")
            log_iter(msg_ortho)
            if trace:
                print(msg1)
                print(msg2)
                print(msg_ortho)

            if (de < conv_tol) and (norm_drho < conv_tol_rho):
                return e_tot, True

            # Preconditioned residual (Kerker)
            res_p = self._kerker(drho, k_screen=kerker_k)
            norm_res_p = np.sqrt(self.grid_weight * float(np.dot(res_p, res_p)))
            msg3 = f"   residual: ||dρ||_precond = {norm_res_p:.3e} (alpha={alpha:.2f}, kerker_k={kerker_k:.2f})"
            log_iter(msg3)
            if trace:
                print(msg3)
            # Update density with DIIS or linear mixing
            if (diis is not None) and (ic >= diis_start_cycle):
                diis.push(rho, res_p)
                try:
                    rho = diis.extrapolate(self.grid_weight)
                except Exception:
                    # Fallback to linear mixing if DIIS fails
                    rho = rho + alpha * res_p
            else:
                rho = rho + alpha * res_p
            # Rebuild potentials
            vH = self.get_vhartree_R(rho)
            v_eff = self._vne_R + vH
            e_prev = e_tot

        return e_tot, False

    # ----------------------------- solve (one-shot) ----------------------------- #
    def diagonalize_hcore(self):
        """Diagonalize H_core per k using Rayleigh–Ritz in the ψ subspace.

        On return, updates self.psi_r, self.psi_g, and self.mo_energy.
        """
        nk, nb = self.nk, self.nband
        mo_energy = np.zeros((nk, nb), dtype=float)
        for ik in range(nk):
            psi = self.psi_r[ik]               # (nb, ngr)
            Hpsi = self._apply_hcore(ik, psi)  # (nb, ngr)
            # Subspace Hamiltonian: H_sub = <ψ|H|ψ>
            # Use weighted inner product with grid_weight
            Hsub = (psi.conj() @ Hpsi.T) * self.grid_weight  # (nb, nb)
            Hsub = (Hsub + Hsub.conj().T) * 0.5  # Hermitize
            eig, U = np.linalg.eigh(Hsub)
            idx = np.argsort(eig)
            eig = eig[idx]
            U = U[:, idx]
            # Rotate ψ within subspace; re-orthonormalize for numerical safety
            psi_new = (U.conj().T @ psi)  # (nb, ngr)
            psi_new = self._orthonormalize_block(psi_new)
            self.psi_r[ik] = psi_new
            self.psi_g[ik] = self._fft_r2g(psi_new)
            mo_energy[ik] = eig
        self.mo_energy = mo_energy
        return mo_energy

    def band_energy(self):
        """Return current band energies (nk, nband)."""
        return np.array(self.mo_energy, copy=True)

    def get_e_tot_one_electron(self):
        """Total one-electron energy sum_k Σ_occ ε_{k,n} (no double counting)."""
        if self.mo_energy is None:
            self.diagonalize_hcore()
        nk = self.nk
        occ = np.zeros((nk, self.nband))
        occ[:, :self.nocc] = 2.0 / nk  # equal k-point weights for closed shell
        return float((self.mo_energy * occ).sum())

    # Convenience driver for this step
    def kernel(self, init='random', **kwargs):
        """Build operators, initialize ψ, diagonalize H_core, return E (one-electron)."""
        self.build()
        self.init_guess(kind=init)
        self.diagonalize_hcore()
        etot = self.get_e_tot_one_electron()
        logger.note(self, 'KPWSCF (T+V_ne) one-electron energy = %.12f', etot)
        return etot


__all__ = ['KPWSCF']

