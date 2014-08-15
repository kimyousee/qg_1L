import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import qg_1L_fxs as fx
import matplotlib.pyplot as plt
import os

Print = PETSc.Sys.Print # For printing with only 1 processor

OutpDir = "storage"
if not os.path.exists(OutpDir):
    os.mkdir(OutpDir)
data = open('storage/InputData','wb')
kkFile=open('storage/kk','wb')
evalsFile = open('storage/evals','wb')

rank = PETSc.COMM_WORLD.Get_rank()
opts = PETSc.Options()
nEV = opts.getInt('nev', 5)

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = opts.getInt('Ny',400)

y = fx.vec(0, float(Ly)/float(Ny), Ny+1)
hy = float(Ly)/float(Ny)

Dy = fx.Dy(hy, Ny)
Dy2 = fx.Dy(hy, Ny, Dy2=True)

f0 = 1.e-4
bet = 0
g0 = 9.81

Phi = fx.Phi(y, Ly, Lj, Ny)
U = fx.U(Dy, Phi, Ny)
etaB = fx.etaB(y)

F0 = f0**2/(g0*Hm)
dkk = 2e-2

dataArr = np.array([Ly,Lj,Hm,hy,f0,bet,g0,F0,dkk,Ny,hy,nEV])

kk = np.arange(dkk,2+dkk,dkk)/Lj
nk = len(kk)

# Temporary vector to store Dy2*Phi
temp = PETSc.Vec().createMPI(Ny-1,comm=PETSc.COMM_WORLD)
Dy2.mult(Phi,temp)
temp.assemble()

temp2 = PETSc.Vec().createMPI(Ny+1, comm=PETSc.COMM_WORLD)
ts,te = temp2.getOwnershipRange()
if ts == 0: temp2[0] = temp[0]; ts+=1
if te == Ny+1: temp2[Ny] = temp[Ny-2]; te -= 1
for i in xrange(ts,te):
    temp2[i] = temp[i-1]
temp2.assemble()

Q = temp2 - F0*Phi + bet*y + f0/Hm*etaB # size Ny+1

evalsArr = np.zeros(nk)
grow = np.zeros([nEV,nk])
freq = np.zeros([nEV,nk])
mode = np.zeros([Ny+1,nEV,nk], dtype=np.complex128)
grOut = open('storage/grow', 'wb')
frOut = open('storage/freq', 'wb')
mdOut = open('storage/mode', 'wb')
cnt = 0

for kx in kk[0:nk]:

    kx2=kx**2

    Lap = fx.Lap(Dy2, kx2, Ny)
    B = fx.B(Lap, F0, Ny)
    A = fx.A(U,Lap, F0,Dy,Q,Ny)

    # Set up slepc, generalized eig problem
    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)

    E.setOperators(A,B); E.setDimensions(nEV, PETSc.DECIDE)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP); E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    E.setTolerances(1e-8, max_it=50)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    evalsArr[cnt] = evals
    for i in xrange(evals):
        eigVal = E.getEigenvalue(i)
        grow[i,cnt] = eigVal.imag*kx
        freq[i,cnt] = eigVal.real*kx

        eigVec=E.getEigenvector(i,vr,vi)

        start,end = vi.getOwnershipRange()
        if start == 0: mode[0,i,cnt] = 0; start+=1
        if end == Ny: mode[Ny,i,cnt] = 0; end -=1

        for j in xrange(start,end):
            mode[j,i,cnt] = vr[j].real + 1j*vr[j].imag

    cnt = cnt+1

grow.tofile(grOut); freq.tofile(frOut); mode.tofile(mdOut)
evalsArr.tofile(evalsFile)
kk.tofile(kkFile)
dataArr.tofile(data)
grOut.close(); frOut.close(); mdOut.close()
kkFile.close(); data.close(); evalsFile.close()

