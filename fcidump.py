#!/usr/bin/env python

def fcidump(mf, filename='FCIDUMP'):
    from pyscf.tools.fcidump import from_integrals
    from pyscf import ao2mo
    from numpy import dot

    h1 = reduce(dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eri = ao2mo.kernel(mf.mol, mf.mo_coeff)

    from_integrals(filename, h1, eri, mf.mo_coeff.shape[1], mf.mol.nelectron, ms=0)
