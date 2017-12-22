#! /usr/bin/env python
from __future__ import print_function, division

def read_input(filename):
    '''Reads a formatted pySCF input file, and
    generates a relevant pySCF Mole object from it.

    Input: the filename of the pySCF input file
    Output: a pySCF Mole object'''

    import input_reader
    from pyscf import gto, scf, dft, cc
    import numpy as np
    from pyscf.cc import ccsd_t


    # initialize reader for a pySCF input
    reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
             case=False, ignoreunknown=True)

    # add atoms block
    atoms = reader.add_block_key('atoms', required=True)
    atoms.add_regex_line('atom',
        '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)', repeat=True)

    # add simple line keys
    reader.add_line_key('memory', type=(int, float))            # max memory in MB
    reader.add_line_key('unit', default='angstrom')             # coord unit
    reader.add_line_key('basis', default='sto-3g')              # basis
    reader.add_line_key('charge', type=int)                     # molecular charge
    reader.add_line_key('spin', type=int)                       # molecular spin
    reader.add_line_key('symmetry', type=int, default=None)     # mol symmetry (False, 0, 1)
    reader.add_line_key('verbose', type=(0,1,2,3,4,5,6,7,8,9), default=4) # verbose level

    # add boolean key
    reader.add_boolean_key('vfreq', default=False)          # vibrational frequencies
    reader.add_boolean_key('fcidump', default=False)        # create FCIDUMP file

    # add scf block keys
    scf = reader.add_block_key('scf', required=True)
    scf.add_line_key('method', default='hf')                 # SCF method
    scf.add_line_key('xc', default='lda,vwn')                # XC functional
    scf.add_line_key('conv', type=float, default=1e-8)       # conv_tol
    scf.add_line_key('grad', type=float, default=1e-6)       # conv_tol_grad
    scf.add_line_key('maxiter', type=int, default=50)        # max iterations
    scf.add_line_key('guess', type=('minao', 'atom', '1e'), default='atom') # intial density guess
    scf.add_line_key('grid', type=(1,2,3,4,5,6,7,8,9), default=2) # dft numint grid
    scf.add_line_key('freeze', type=int, default=None)      # frozen core orbitals
    scf.add_line_key('cas', type=[int, int], default=None)  # CAS space for CASCI or CASSCF

    # add geomopt block key
    geomopt = reader.add_block_key('geomopt', required=False)
    geomopt.add_line_key('conv', type=float, default=1e-6)      # energy convergence
    geomopt.add_line_key('rconv', type=float, default=1e-2)     # distance convergence
    geomopt.add_line_key('maxiter', type=int, default=25)       # geometry iterations
    geomopt.add_line_key('gradtol', type=float, default=1e-4)   # gradient tolerance

    # read the input filename
    inp  = reader.read_input(filename)
    inp.filename = filename

    # print input file to screen
    pstr("Input File")
    [print (i[:-1]) for i in open(filename).readlines() if ((i[0] not in ['#', '!'])
        and (i[0:2] not in ['::', '//']))]
    pstr("End Input", addline=False)

    # initialze pySCF molecule object
    mol = gto.Mole()

    # collect atoms in pyscf format
    mol.atom = []
    ghbasis = []
    for r in inp.atoms.atom:
        if 'ghost.' in r.group(1).lower() or 'gh.' in r.group(1).lower():
            ghbasis.append(r.group(1).split('.')[1])
            rgrp1 = 'ghost:{0}'.format(len(ghbasis))
            mol.atom.append([rgrp1, (float(r.group(2)), float(r.group(3)), float(r.group(4)))])
        else:
            mol.atom.append([r.group(1), (float(r.group(2)), float(r.group(3)), float(r.group(4)))])

    # build dict of basis for each atom
    mol.basis = {}
    nghost = 0
    for i in range(len(mol.atom)):
        if 'ghost' in mol.atom[i][0]:
            mol.basis.update({mol.atom[i][0]: gto.basis.load(inp.basis, ghbasis[nghost])})
            nghost += 1
        else:
            mol.basis.update({mol.atom[i][0]: gto.basis.load(inp.basis, mol.atom[i][0])})

    # build molecule object
    if inp.memory is not None: mol.max_memory = inp.memory
    mol.unit = inp.unit
    if inp.charge is not None: mol.charge = inp.charge
    if inp.spin is not None: mol.spin = inp.spin
    mol.verbose = inp.verbose
    if inp.symmetry is not None: mol.symmetry = inp.symmetry
    mol.build()#dump_input=False)
    inp.mol = mol

    # return inp object
    return inp

def pstr(st, delim="=", l=80, fill=True, addline=True):
    '''Print formatted string <st> to output'''
    if addline: print ("")
    if len(st) == 0:
        print (delim*l)
    elif len(st) >= l:
        print (st)
    else:
        l1 = int((l-len(st)-2)/2)
        l2 = int((l-len(st)-2)/2 + (l-len(st)-2)%2)
        if fill:
            print (delim*l1+" "+st+" "+delim*l2)
        else:
            print (delim+" "*l1+st+" "*l2+delim)
