from __future__ import print_function
import numpy as np

class gaussian_cube():
    '''A class to store the Gaussian cube file information.'''

    def __init__(self):
        self.filename   = None
        self.title      = None
        self.subtitle   = None
        self.natoms     = None
        self.origin     = None
        self.nx         = None
        self.vx         = None
        self.ny         = None
        self.vy         = None
        self.nz         = None
        self.vz         = None
        self.atype      = None
        self.atvar      = None
        self.atcoord    = None
        self.values     = None

    def read(self, filename):
        self.filename   = filename
        f               = open(self.filename, 'r')
        lines           = f.readlines()
        self.title      = lines[0]
        if self.title[-1:]=='\n':
            self.title = self.title[:-1]
        self.subtitle   = lines[1]
        if self.subtitle[-1:]=='\n':
            self.subtitle = self.subtitle[:-1]
        self.natoms     = int(lines[2].split()[0])
        self.origin     = np.array(lines[2].split()[1:4], dtype=float)
        self.nx         = int(lines[3].split()[0])
        self.vx         = np.array(lines[3].split()[1:4], dtype=float)
        self.ny         = int(lines[4].split()[0])
        self.vy         = np.array(lines[4].split()[1:4], dtype=float)
        self.nz         = int(lines[5].split()[0])
        self.vz         = np.array(lines[5].split()[1:4], dtype=float)
        self.atype      = np.zeros((self.natoms), dtype=int)
        self.atvar      = np.zeros((self.natoms), dtype=float)
        self.atcoord    = np.zeros((self.natoms,3), dtype=float)
        for i in range(self.natoms):
            self.atype[i]   = lines[6+i].split()[0]
            self.atvar[i]   = lines[6+i].split()[1]
            self.atcoord[i] = lines[6+i].split()[2:5]
        templist        = []
        for i in range(6+self.natoms,len(lines)):
            templist.append(lines[i].split())
        self.values     = []
        for i in range(len(templist)):
            for j in range(len(templist[i])):
                self.values.append(templist[i][j])
        self.values     = np.array(self.values, dtype=float)

    def write(self, outfile=None):
        if not outfile: outfile = self.filename
        f   = open(outfile, 'w')
        print (self.title, file=f)
        print (self.subtitle, file=f)
        tempstr = '{0:5d} {1:11.6f} {2:11.6f} {3:11.6f}'
        print (tempstr.format(self.natoms, self.origin[0], self.origin[1], self.origin[2]), file=f)
        print (tempstr.format(self.nx, self.vx[0], self.vx[1], self.vx[2]), file=f)
        print (tempstr.format(self.ny, self.vy[0], self.vy[1], self.vy[2]), file=f)
        print (tempstr.format(self.nz, self.vz[0], self.vz[1], self.vz[2]), file=f)
        tempstr = '{0:5d} {1:11.6f} {2:11.6f} {3:11.6f} {4:11.6f}'
        for i in range(self.natoms):
            print (tempstr.format(self.atype[i], self.atvar[i], self.atcoord[i][0], self.atcoord[i][1], self.atcoord[i][2]), file=f)
        tempstr = ''
        for i in range(len(self.values)):
            tempstr += ' {0:12.5e}'.format(self.values[i])
            if ((i+1)%self.nz)==0:
                tempstr += '\n'
            elif (((i+1)%self.nz)%6)==0:
                tempstr += '\n'
        print (tempstr, file=f)
        f.close()

    def save(self, filename):
        '''saves the cubefile to filename'''

        fout = open(filename, 'w')

        print (self.title, file=fout)
        print (self.subtitle, file=fout)
        print ("{0:4d} {1:12.6f} {2:12.6f} {3:12.6f}".format(self.natoms,
            self.origin[0], self.origin[1], self.origin[2]), file=fout)
        print ("{0:4d} {1:12.6f} {2:12.6f} {3:12.6f}".format(self.nx,
            self.vx[0], self.vx[1], self.vx[2]), file=fout)
        print ("{0:4d} {1:12.6f} {2:12.6f} {3:12.6f}".format(self.ny,
            self.vy[0], self.vy[1], self.vy[2]), file=fout)
        print ("{0:4d} {1:12.6f} {2:12.6f} {3:12.6f}".format(self.nz,
            self.vz[0], self.vz[1], self.vz[2]), file=fout)

        for i in range(self.natoms):
            print ("{0:4d} {1:12.6f} {2:12.6f} {3:12.6f} {4:12.6f}".format(self.atype[i],
                self.atvar[i], self.atcoord[i][0], self.atcoord[i][1], self.atcoord[i][2]),
                file=fout)

        li = len(self.values)/6
        lj = len(self.values)%6

        k = 0
        for i in range(li):
            st = ""
            for j in range(6):
                st = st + " {0:12.6f}".format(self.values[k])
                k += 1
            print (st, file=fout)

        st = ""
        for j in range(lj):
            st = st + " {0:12.6f}".format(self.values[k])
            k += 1
        print (st, file=fout)

        fout.close()

    @property
    def coords(self):
        coords = np.zeros((self.nx*self.ny*self.nz,3))
        l = 0
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    coords[l] = self.origin + i*self.vx + j*self.vy + k*self.vz
                    l += 1
        return coords
