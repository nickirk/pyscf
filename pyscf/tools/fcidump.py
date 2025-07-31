#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

'''
FCIDUMP functions (write, read) for real Hamiltonian
'''

import re
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf import __config__

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)

MOLPRO_ORBSYM = getattr(__config__, 'fcidump_molpro_orbsym', False)

# Mapping Pyscf symmetry numbering to Molpro symmetry numbering for each irrep.
# See also pyscf.symm.param.IRREP_ID_TABLE
# https://www.molpro.net/info/current/doc/manual/node36.html
ORBSYM_MAP = {
    'D2h': (1,         # Ag
            4,         # B1g
            6,         # B2g
            7,         # B3g
            8,         # Au
            5,         # B1u
            3,         # B2u
            2),        # B3u
    'C2v': (1,         # A1
            4,         # A2
            2,         # B1
            3),        # B2
    'C2h': (1,         # Ag
            4,         # Bg
            2,         # Au
            3),        # Bu
    'D2' : (1,         # A
            4,         # B1
            3,         # B2
            2),        # B3
    'Cs' : (1,         # A'
            2),        # A"
    'C2' : (1,         # A
            2),        # B
    'Ci' : (1,         # Ag
            2),        # Au
    'C1' : (1,)
}

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


def write_eri(fout, eri, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    npair = nmo*(nmo+1)//2
    output_format = float_format + ' %4d %4d %4d %4d\n'
    if eri.size == nmo**4:
        eri = ao2mo.restore(8, eri, nmo)

    if eri.ndim == 2: # 4-fold symmetry
        assert (eri.size == npair**2)
        ij = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if abs(eri[ij,kl]) > tol:
                            fout.write(output_format % (eri[ij,kl], i+1, j+1, k+1, l+1))
                        kl += 1
                ij += 1
    else:  # 8-fold symmetry
        assert (eri.size == npair*(npair+1)//2)
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ijkl]) > tol:
                                fout.write(output_format % (eri[ijkl], i+1, j+1, k+1, l+1))
                            ijkl += 1
                        kl += 1
                ij += 1

def write_hcore(fout, h, nmo, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(nmo,nmo)
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h[i,j]) > tol:
                fout.write(output_format % (h[i,j], i+1, j+1))


def from_chkfile(filename, chkfile, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT,
                 molpro_orbsym=MOLPRO_ORBSYM, orbsym=None):
    '''Read SCF results from PySCF chkfile and transform 1-electron,
    2-electron integrals using the SCF orbitals.  The transformed integrals is
    written to FCIDUMP

    Kwargs:
        molpro_orbsym (bool): Whether to dump the orbsym in Molpro orbsym
            convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
    '''
    mol, scf_rec = scf.chkfile.load_scf(chkfile)
    mo_coeff = numpy.array(scf_rec['mo_coeff'])
    nmo = mo_coeff.shape[1]

    s = reduce(numpy.dot, (mo_coeff.conj().T, mol.intor_symmetric('int1e_ovlp'), mo_coeff))
    if abs(s - numpy.eye(nmo)).max() > 1e-6:
        # Not support the chkfile from pbc calculation
        raise RuntimeError('Non-orthogonal orbitals found in chkfile')

    if mol.symmetry:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                     mol.symm_orb, mo_coeff, check=False)
    from_mo(mol, filename, mo_coeff, orbsym=orbsym, tol=tol,
            float_format=float_format, molpro_orbsym=molpro_orbsym,
            ms=mol.spin)

def from_integrals(filename, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
    with open(filename, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri(fout, h2e, nmo, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, nmo, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

def from_mo(mol, filename, mo_coeff, orbsym=None,
            tol=TOL, float_format=DEFAULT_FLOAT_FORMAT,
            molpro_orbsym=MOLPRO_ORBSYM, ms=0):
    '''Use the given MOs to transform the 1-electron and 2-electron integrals
    then dump them to FCIDUMP.

    Kwargs:
        molpro_orbsym (bool): Whether to dump the orbsym in Molpro orbsym
            convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
    '''
    if getattr(mol, '_mesh', None):
        raise NotImplementedError('PBC system')

    if orbsym is None:
        orbsym = getattr(mo_coeff, 'orbsym', None)
        if molpro_orbsym and orbsym is not None:
            orbsym = [ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    h1ao = scf.hf.get_hcore(mol)
    h1e = reduce(numpy.dot, (mo_coeff.T, h1ao, mo_coeff))
    eri = ao2mo.full(mol, mo_coeff, verbose=0)
    nuc = mol.energy_nuc()
    from_integrals(filename, h1e, eri, h1e.shape[0], mol.nelec, nuc, ms, orbsym,
                   tol, float_format)

def from_scf(mf, filename, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT,
             molpro_orbsym=MOLPRO_ORBSYM):
    '''Use the given SCF object to transform the 1-electron and 2-electron
    integrals then dump them to FCIDUMP.

    Kwargs:
        molpro_orbsym (bool): Whether to dump the orbsym in Molpro orbsym
            convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == numpy.double

    h1e = reduce(numpy.dot, (mo_coeff.T, mf.get_hcore(), mo_coeff))
    if mf._eri is None:
        if getattr(mf, 'exxdiv', None):  # PBC system
            eri = mf.with_df.ao2mo(mo_coeff)
        else:
            eri = ao2mo.full(mf.mol, mo_coeff)
    else:  # Handle cached integrals or customized systems
        eri = ao2mo.full(mf._eri, mo_coeff)
    orbsym = getattr(mo_coeff, 'orbsym', None)
    if molpro_orbsym and orbsym is not None:
        orbsym = [ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    nuc = mf.energy_nuc()
    from_integrals(filename, h1e, eri, h1e.shape[0], mf.mol.nelec, nuc, 0, orbsym,
                   tol, float_format)

def from_mcscf(mc, filename, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT,
               molpro_orbsym=MOLPRO_ORBSYM):
    '''Use the given MCSCF object to obtain the CAS 1-electron and 2-electron
    integrals and dump them to FCIDUMP.

    Kwargs:
        tol (float): Threshold for writing elements to FCIDUMP
        float_format (str): Float format for writing elements to FCIDUMP
        molpro_orbsym (bool): Whether to dump the orbsym in Molpro orbsym
            convention as documented in
            https://www.molpro.net/manual/doku.php?id=general_program_structure#symmetry
    '''
    mol = mc.mol
    mo_coeff = mc.mo_coeff
    assert mo_coeff.dtype == numpy.double

    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    orbsym = getattr(mo_coeff, 'orbsym', None)
    if orbsym is not None:
        orbsym = orbsym[mc.ncore:mc.ncore + mc.ncas]
    if molpro_orbsym and orbsym is not None:
        orbsym = [ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    nelecas = mc.nelecas[0] + mc.nelecas[1]
    ms = abs(mc.nelecas[0] - mc.nelecas[1])
    from_integrals(filename, h1eff, h2eff, mc.ncas, nelecas, nuc=ecore, ms=ms,
                   orbsym=orbsym, tol=tol, float_format=float_format)

def read(filename, molpro_orbsym=MOLPRO_ORBSYM, verbose=True):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM

    Kwargs:
        molpro_orbsym (bool): Whether the orbsym in the FCIDUMP file is in
            Molpro orbsym convention as documented in::

                https://www.molpro.net/info/current/doc/manual/node36.html

            In return, orbsym is converted to pyscf symmetry convention
        verbose (bool): Whether to print debugging information

    '''
    if verbose:
        print('Parsing %s' % filename)
    finp = open(filename, 'r')

    data = []
    while True:
        line = finp.readline().upper()
        if not line:
            raise RuntimeError('Problematic FCIDUMP header: &END not found')
        data.append(line)
        if '&END' in line or '/' in line:
            break

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '').replace('/', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    # Convert to Molpro orbsym convert_orbsym
    if 'ORBSYM' in result:
        if molpro_orbsym:
            # Guess which point group the orbsym belongs to. FCIDUMP does not
            # save the point group information, the guess might be wrong if
            # the high symmetry numbering of orbitals are not presented.
            orbsym = result['ORBSYM']
            if max(orbsym) > 4:
                result['ORBSYM'] = [ORBSYM_MAP['D2h'].index(i) for i in orbsym]
            elif max(orbsym) > 2:
                # Fortunately, without molecular orientation, B2 and B3 in D2
                # are not distinguishable
                result['ORBSYM'] = [ORBSYM_MAP['C2v'].index(i) for i in orbsym]
            elif max(orbsym) == 2:
                result['ORBSYM'] = [i-1 for i in orbsym]
            elif max(orbsym) == 1:
                result['ORBSYM'] = [0] * len(orbsym)
            else:
                raise RuntimeError('Unknown orbsym')
        elif min(result['ORBSYM']) < 0:
            raise RuntimeError('Unknown orbsym convention')

    norb = result['NORB']
    norb_pair = norb * (norb+1) // 2
    h1e = numpy.zeros((norb,norb))
    h2e = numpy.zeros(norb_pair*(norb_pair+1)//2)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[1:5]]
        if k != 0:
            if i >= j:
                ij = i * (i-1) // 2 + j-1
            else:
                ij = j * (j-1) // 2 + i-1
            if k >= l:
                kl = k * (k-1) // 2 + l-1
            else:
                kl = l * (l-1) // 2 + k-1
            if ij >= kl:
                h2e[ij*(ij+1)//2+kl] = float(dat[0])
            else:
                h2e[kl*(kl+1)//2+ij] = float(dat[0])
        elif k == 0:
            if j != 0:
                h1e[i-1,j-1] = float(dat[0])
            else:
                result['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = numpy.tril_indices(norb, -1)
    if numpy.linalg.norm(h1e[idy,idx]) == 0:
        h1e[idy,idx] = h1e[idx,idy]
    elif numpy.linalg.norm(h1e[idx,idy]) == 0:
        h1e[idx,idy] = h1e[idy,idx]
    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result

def to_scf(filename, molpro_orbsym=MOLPRO_ORBSYM, mf=None, **kwargs):
    '''Use the Hamiltonians defined by FCIDUMP to build an SCF object'''
    ctx = read(filename, molpro_orbsym)
    mol = gto.M()
    mol.nelectron = ctx['NELEC']
    mol.spin = ctx['MS2']
    norb = mol.nao = ctx['NORB']
    if 'ECORE' in ctx:
        mol.energy_nuc = lambda *args: ctx['ECORE']
    mol.incore_anyway = True

    if 'ORBSYM' in ctx:
        mol.symmetry = True
        mol.groupname = 'N/A'
        orbsym = numpy.asarray(ctx['ORBSYM'])
        mol.irrep_id = list(set(orbsym))
        mol.irrep_name = [('IR%d' % ir) for ir in mol.irrep_id]
        so = numpy.eye(norb)
        mol.symm_orb = []
        for ir in mol.irrep_id:
            mol.symm_orb.append(so[:,orbsym==ir])

    if mf is None:
        mf = mol.RHF(**kwargs)
    else:
        mf.mol = mol
    h1 = ctx['H1']
    idx, idy = numpy.tril_indices(norb, -1)
    if h1[idx,idy].max() == 0:
        h1[idx,idy] = h1[idy,idx]
    else:
        h1[idy,idx] = h1[idx,idy]
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(norb)
    mf._eri = ctx['H2']

    return mf

def scf_from_fcidump(mf, filename, molpro_orbsym=MOLPRO_ORBSYM):
    '''Update the SCF object with the quantities defined in FCIDUMP file'''
    return to_scf(filename, molpro_orbsym, mf)

scf.hf.SCF.from_fcidump = scf_from_fcidump

def read_tc(filename):
    '''Parse FCIDUMP configuration file in TC format.
    This format has a header like:
    &FCI
     NORB=25,
     NELEC=10,
     MS2=0,
     ISYM=1,
     ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
     ST=1,
     III=0,
     OCC=5,
     CLOSED=5,
     IUHF=0,
     NPY1=int1.npy,
     NPY2=int2.npy,
     ENUC=    59.022315384270307     ,
     /

    Args:
        filename : a FCIDUMP file/ or just the header part

    Returns:
        h1e : (nmo,nmo) array
        h2e : (nmo,nmo,nmo,nmo) array
        ecore : float
        norb : int
        nelec : int
        ms2 : int
        orbsym : list
    '''
    with open(filename, 'r') as f:
        data = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError('Incomplete FCIDUMP header')
            data.append(line.strip())
            if line.strip().endswith('/') or line.strip().endswith('&END'):
                break
        
        # Join all header lines and remove &FCI and /
        header = ' '.join(data).replace('&FCI', '').replace('/', '')
        
        # Split by comma and clean up
        tokens = [x.strip() for x in header.split(',') if x.strip()]
        
        # Parse key-value pairs
        params = {}
        for token in tokens:
            if '=' not in token:
                continue
            key, val = token.split('=', 1)
            key = key.strip().upper()
            val = val.strip()
            
            if key in ('NORB', 'NELEC', 'MS2', 'ISYM', 'ST', 'III', 'OCC', 'CLOSED'):
                params[key] = int(val)
            elif key == 'ORBSYM':
                # Handle ORBSYM specially as it can have multiple values
                orbsym_vals = [x.strip() for x in val.split(',') if x.strip()]
                params[key] = [int(x) for x in orbsym_vals]
            elif key == 'ENUC':
                params[key] = float(val)
            else:
                params[key] = val.strip('"\'')  # Remove any quotes from string values

        # Get required parameters with defaults
        norb = params.get('NORB')
        if norb is None:
            raise ValueError('NORB not found in FCIDUMP header')
        nelec = params.get('NELEC', 0)
        ms2 = params.get('MS2', 0)
        orbsym = params.get('ORBSYM', [1] * norb)
        ecore = params.get('ENUC', 0.0)

        # Check if NPY files are specified
        if 'NPY1' in params and 'NPY2' in params:
            # Read from NPY files
            import os
            dirname = os.path.dirname(filename)
            h1e = numpy.load(os.path.join(dirname, params['NPY1']))
            int_2elec = numpy.load(os.path.join(dirname, params['NPY2']))
            
            # Check if NPY2 is in triangular storage format
            if len(int_2elec.shape) == 3:  # Triangular storage
                norb = len(int_2elec)
                h2e = numpy.zeros((norb, norb, norb, norb))
                
                # Create indices for the lower triangular part
                l_idx, k_idx = numpy.tril_indices(norb, -1)
                # Create all combinations of i,j with k,l indices
                i_idx = numpy.arange(norb)[:,None,None]
                j_idx = numpy.arange(norb)[None,:,None]
                
                # Assign values for k < l
                kl_idx = numpy.arange(len(k_idx))
                h2e[i_idx, k_idx, j_idx, l_idx] = int_2elec[:,:,kl_idx]
                h2e[j_idx, l_idx, i_idx, k_idx] = int_2elec[:,:,kl_idx]  # (ik,jl) permutation symmetry
                
                # Assign values for k == l
                l_diag = numpy.arange(norb)
                kl_diag = kl_idx[-1] + 1 + numpy.arange(norb)
                h2e[i_idx, l_diag, j_idx, l_diag] = int_2elec[:,:,kl_diag]
            else:  # Full storage
                h2e = int_2elec
            
            return h1e, h2e, ecore, norb, nelec, ms2, orbsym

        # If no NPY files, read integrals from the FCIDUMP file
        h1e = numpy.zeros((norb, norb))
        h2e = numpy.zeros((norb, norb, norb, norb))
        
        while True:
            line = f.readline()
            if not line:
                break
            if len(line.split()) == 0:
                continue
            a, i, j, k, l = line.split()
            i, j, k, l = [int(x) - 1 for x in [i, j, k, l]]
        
            if i + j + k + l == -4:
                ecore = float(a)
            elif k + l == -2:
                h1e[i, j] = float(a)
                h1e[j, i] = float(a)
            else:
                h2e[i, j, k, l] = float(a)
                h2e[k, l, i, j] = float(a)

    return h1e, h2e, ecore, norb, nelec, ms2, orbsym

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert chkfile to FCIDUMP')
    parser.add_argument('chkfile', help='pyscf chkfile')
    parser.add_argument('fcidump', help='FCIDUMP file')
    args = parser.parse_args()

    # fcidump.py chkfile output
    from_chkfile(args.fcidump, args.chkfile)
