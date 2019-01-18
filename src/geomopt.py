#!/usr/bin/env python
from __future__ import print_function

def update_mole(mol, dx=None):

    import numpy as np
    import pyscf
    from .read_input import pstr

    # update the mole geometry
    q = mol.atom_charges()
    if dx is None:
        c = mol.atom_coords()
        newc = np.copy(c)
    else:
        c = mol.atom_coords()
        newc = c + dx

    # shift to center of charge
#    cc = np.dot(q,newc) / q.sum()
    cc = newc[0]
    newc.transpose()[0] -= cc[0]
    newc.transpose()[1] -= cc[1]
    newc.transpose()[2] -= cc[2]

    # get change in coordinates
    rmax = np.abs(newc - c).max()

    # change to relevant unit
    conv = 1.
    if mol.unit in ('angstrom', 'a'): conv = 0.5291772
    newc *= conv

    # update atom
    atom = []
    for i in range(mol.natm):
        atom.append([mol.atom_symbol(i), (newc[i][0], newc[i][1], newc[i][2])])
    mol.atom = atom
    mol.build()

    # print new coords
    if dx is not None:
        pstr ('Geometry (Angstrom)', delim='-', addline=False)
        for i in range(mol.natm):
            print ('{0:>2}   {1:11.6f}   {2:11.6f}   {3:11.6f}'.format(mol.atom_symbol(i),
                   newc[i][0], newc[i][1], newc[i][2]))
        pstr ('', delim='-', addline=False)

    return mol, rmax

def symmetrize(M):
    n = M.shape[0]
    for i in range(n):
        for j in range(i,n):
            temp = 0.5 * ( M[i][j] + M[j][i] )
            M[i][j] = M[j][i] = temp
    return M

def do_geomopt(inp):
    '''Geometry optimization.'''

    import pyscf
    from pyscf import grad
    from pyscf.future import hessian
    import numpy as np
    import scipy as sp
    from .read_input import pstr
    from .scf import do_scf

    # get convergence criteria
    errconv = inp.geomopt.conv
    rconv   = inp.geomopt.rconv
    gradtol = inp.geomopt.gradtol
    maxiter = inp.geomopt.maxiter

    # get the molecule and scf object
    mol, null = update_mole(inp.mol)
    mf = inp.mf
    n3 = mol.natm * 3

    lconv = False
    oldE = 0.
    icyc = 0
    rmax = 1.
    while not lconv and icyc <= maxiter:
        icyc += 1

        pstr("Geomtry cycle {0}".format(icyc), delim="=")

        # do SCF
        if icyc > 1:
            inp.timer.start('scf')
            inp.mol = mol
            inp = do_scf(inp)
            mf = inp.mf
            inp.timer.end('scf')
        enew = mf.e_tot

        # get gradient
        inp.timer.start('gradients')
        if inp.scf.method in ('uhf', 'rhf', 'hf'):
            g = grad.RHF(mf).kernel()
        else:
            g = grad.RKS(mf).kernel()
        g = g.reshape(n3)

        # remove coordinates with tiny gradients
        a = np.where(np.abs(g) >= gradtol)
        if len(a[0]) == 0:
            a = tuple([np.arange(mol.natm*3),])
        g = g[a]
        inp.timer.end('gradients')

        # get hessian
        inp.timer.start('hessian')
        if inp.scf.method in ('uhf', 'rhf', 'hf'):
            h = hessian.RHF(mf).kernel()
        else:
            h = hessian.RKS(mf).kernel()
        h = h.transpose(0,2,1,3).reshape(n3,n3)
        inp.hessian = np.copy(h)
        h = h[np.ix_(a[0],a[0])]
        h = symmetrize(h)
        inp.timer.end('hessian')

        # invert hessian
        inp.timer.start('inverse hessian')
        if rmax < np.sqrt(rconv):
            hinv = sp.linalg.inv(h)
        else:
            hinv = np.copy(h)
        inp.timer.end('inverse hessian')

        # get new coordinate step
        dx = np.zeros((n3))
        dx[a] = -1. * np.dot(hinv, g)
        dx = dx.reshape(mol.natm, 3)

        # update mole object
        mol, rmax = update_mole(mol, dx)

        # update error
        err = abs(enew - oldE)
        print ('Cycle {0:>2}   delta E: {1:15.10f}   rmax: {2:9.6f}'.format(icyc,
               enew-oldE, rmax))
        oldE = np.copy(enew)
        lconv = (err < errconv) and (rmax < rconv)

    # print geometry convergence message
    if err <= errconv and rmax <= rconv:
        pstr ("Geometry Converged", delim="!")
    else:
        pstr ("Geometry NOT Converged", delim="!")

    return inp
