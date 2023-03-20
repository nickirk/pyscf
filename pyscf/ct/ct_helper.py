import numpy as np
from pyscf.lib import logger
from pyscf import lib


def fill_dms(dm1, dm2, mf, cas):
    """This function fills rdm1 and rdm2 in the whole space
    with those from active space

    Args:
        dm1 (np 2d array): rdm1 in active space
        dm2 (np 4d array): rdm2 in active space 
        mf (pyscf mf object): mean-field object provides rdms in the whole space 
        cas (tuple/list): (num orb, num elec)

    Returns:
        f_dm1, f_dm2: filled rdm1 and rdm2
    """
    mf_dm1 = np.diag(mf.mo_occ)
    mf_dm2 = (np.einsum('ij, kl -> ikjl', mf_dm1, mf_dm1) - np.einsum(
        'il, kj -> ikjl', mf_dm1, mf_dm1) / 2.)
    f_dm1 = mf_dm1.copy()
    f_dm2 = mf_dm2.copy()
    nocc = int(np.sum(mf.mo_occ))
    nelec = nocc
    ncasorb, ncaselec = cas
    f_l= (nelec - ncaselec)//2
    f_h = f_l + ncasorb
    f_dm1[f_l:f_h, f_l:f_h] = dm1.copy()
    f_dm2[f_l:f_h, f_l:f_h, f_l:f_h, f_l:f_h] = dm2.copy()
    # the core-active blocks are not correct either.
    dm2_ = np.einsum('ij, kl -> ikjl', f_dm1, f_dm1)
    dm2_ -= np.einsum('il, kj -> ikjl', f_dm1, f_dm1)/2.
    
    f_dm2[f_l:, :f_l, f_l:, :f_l] = dm2_[f_l:, :f_l, f_l:, :f_l]
    f_dm2[:f_l, f_l:, :f_l, f_l:] = dm2_[:f_l, f_l:, :f_l, f_l:]
    f_dm2[f_l:, :f_l, :f_l, f_l:] = dm2_[f_l:, :f_l, :f_l, f_l:]
    f_dm2[:f_l, f_l:, f_l:, :f_l] = dm2_[:f_l, f_l:, f_l:, :f_l]
    if mf.verbose > logger.DEBUG1:
        logger.info(mf, "Diagonal of filled dm1 = %s", f_dm1.diagonal())

    return f_dm1, f_dm2