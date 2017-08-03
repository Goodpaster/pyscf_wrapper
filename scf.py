#! /usr/bin/env python
from __future__ import print_function, division

def do_scf(inp):
    '''Do the requested SCF.'''

    from pyscf import gto, scf, dft, cc
    from pyscf.cc import ccsd_t

    # sort out the method
    mol = inp.mol
    method = inp.scf.method.lower()

    # UHF
    if method == 'uhf':
        inp.timer.start('uhf')
        mSCF = scf.UHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.kernel()
        inp.timer.end('uhf')

    # RHF
    elif method in ('rhf', 'hf'):
        inp.timer.start('hf')
        if mol.nelectron%2 == 0:
            mSCF = scf.RHF(mol)
        else:
            mSCF = scf.ROHF(mol)
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.kernel()
        inp.timer.end('hf')

    # CCSD
    elif method in ('ccsd', 'ccsd(t)'):
        inp.timer.start('hf')
        tSCF = scf.RHF(mol)
        tSCF.conv_tol = inp.scf.conv
        tSCF.conv_tol_grad = inp.scf.grad
        tSCF.max_cycle = inp.scf.maxiter
        tSCF.init_guess = inp.scf.guess
        ehf = tSCF.kernel()
        inp.timer.end('hf')
        inp.timer.start('ccsd')
        mSCF = cc.RCCSD(tSCF)
        mSCF.max_cycle = inp.scf.maxiter
        eccsd, t1, t2 = mSCF.kernel()
        inp.timer.end('ccsd')

        if method in ('ccsd(t)'):
            inp.timer.start('ccsd(t)')
            eris = mSCF.ao2mo()
            e3 = ccsd_t.kernel(mSCF, eris)
            print ('Total CCSD(T) = {0:20.15f}'.format(ehf + eccsd + e3))
            inp.timer.end('ccsd(t)')

    # UKS
    elif method == 'uks' or method ==  'udft':
        inp.timer.start('uks')
        inp.timer.start('grids')
        grids = dft.gen_grid.Grids(mol)
        grids.level = inp.scf.grid
        grids.build()
        inp.timer.end('grids')
        mSCF = dft.UKS(mol)
        mSCF.grids = grids
        mSCF.xc = inp.scf.xc
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.small_rho_cutoff = 1e-20
        mSCF.kernel()
        inp.timer.end('uks')

    # RKS
    elif method in ('rks', 'ks', 'rdft', 'dft'):
        inp.timer.start('ks')
        inp.timer.start('grids')
        grids = dft.gen_grid.Grids(mol)
        grids.level = inp.scf.grid
        grids.build()
        inp.timer.end('grids')
        if mol.nelectron%2 == 0:
            mSCF = dft.RKS(mol)
        else:
            mSCF = dft.ROKS(mol)
        mSCF.grids = grids
        mSCF.xc = inp.scf.xc
        mSCF.conv_tol = inp.scf.conv
        mSCF.conv_tol_grad = inp.scf.grad
        mSCF.max_cycle = inp.scf.maxiter
        mSCF.init_guess = inp.scf.guess
        mSCF.small_rho_cutoff = 1e-20
        mSCF.kernel()
        inp.timer.end('ks')

    else:
        print ('ERROR: Unrecognized SCF method!')

    # save and return
    inp.mf = mSCF
    return inp

