#!/usr/bin/env python

def plot_MO(molid, mol, C, coords):

    from pyscf import dft
    import numpy as np

    Ci = C.transpose()[molid]
    AOr = dft.numint.eval_ao(mol, coords, deriv=0)
    vals = np.dot(AOr, Ci)
    vals = vals

    return vals

def save_MOs(inp, mf, Cmat, spacing=0.1, x=None, y=None, z=None):

    import numpy as np
    import .gaussian_cube import gaussian_cube

    # filename
    filename = inp.filename[:-4] + '_orbitals'

    # get coords
    if x is None or y is None or z is None:
        atc = mf.mol.atom_coords()
        atcx = atc.transpose()[0]
        atcy = atc.transpose()[1]
        atcz = atc.transpose()[2]
        x = np.arange(atcx.min()-2.5,atcx.max()+2.5+spacing,spacing)
        y = np.arange(atcy.min()-2.5,atcy.max()+2.5+spacing,spacing)
        z = np.arange(atcz.min()-2.5,atcz.max()+2.5+spacing,spacing)
    else:
        spacing = x[1] - x[0]

    nx = len(x)
    ny = len(y)
    nz = len(z)

    coords = np.zeros((nx*ny*nz,3))

    j = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                coords[j] = [x[ix], y[iy], z[iz]]
                j += 1


    # convert to bohr
    # cube files assume that coordinates are in bohr
    # by default
    ang2bohr = 1.8897261328856432
    coordsb = coords * ang2bohr
    spaceb = spacing * ang2bohr
    origin = np.array([x[0], y[0], z[0]]) * ang2bohr

    cub = gaussian_cube()

    cub.title = 'ORBITALS'
    cub.nx = nx
    cub.vx = [spaceb, 0.0, 0.0]
    cub.ny = ny
    cub.vy = [0.0, spaceb, 0.0]
    cub.nz = nz
    cub.vz = [0.0, 0.0, spaceb]
    cub.origin = origin

    cub.natoms = mf.mol.natm
    cub.atype = mf.mol.atom_charges()
    cub.atvar = [0.0 for i in range(cub.natoms)]
    cub.atcoord = mf.mol.atom_coords()

    for i in range(Cmat.shape[1]):
        cub.subtitle = "ORBITAL NUMBER {0} - ".format(i+1)
        st = ["occ" if mf.mo_occ[i] > 0 else "vir"][0]
        cub.subtitle = cub.subtitle + st

        cub.filename = filename+"_{0}".format(i+1)+"_"+st+".cub"

        vals = plot_MO(i, mf.mol, Cmat, coordsb)
        cub.values = vals[:]

        cub.write()
