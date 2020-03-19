#############################################################################################################
#																											#
#	Title			:	Eigen energies and modes as the gap between the wires is varied						#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_lobpcg/4qw_gap.py													#
#	Figure ref.		:	Figures 2.22 and 2.23																#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

import scipy
import numpy as np
from scipy.sparse.linalg import lobpcg
from scipy import sparse
import matplotlib.pyplot as plt
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import poisson
import pylab

N = 100																# Size of the wire
K = 9																# Number of eigen modes
pot = 10.0/57 														# Substrate crystal potential
L = np.arange(-(N/5),(N/10),1)										# varying gaps between the wires

VV = np.zeros((len(L),N*N))
WW = np.zeros((len(L),K))

#############################################################################################################
# Simulating eigen modes and calculating eigen energies for varying gaps between wires						#
#############################################################################################################
for m in range(len(L)):
	row = []
	col = []
	data = []

	# Setting potentials such that 4 wires are placed on a substrate
	for i in range(N):
		for j in range(N):
			ij = i*N+j
			d = 4.0
			row.append(ij)
			col.append(ij)
			if (not((((i>=(N/5)+L[m])&(i<(2*N/5)+L[m]))|((i>=(3*N/5)-L[m])&(i<(4*N/5)-L[m])))&(((j>=(N/5)+L[m])&(j<(2*N/5)+L[m]))|((j>=(3*N/5)-L[m])&(j<(4*N/5)-L[m]))))):
				d+=pot
			data.append(d)
			if(i!=0):
				ij1 = (i-1)*N+j
				row.append(ij)
				col.append(ij1)
				data.append(-1.0)
			if(i!=N-1):
				ij2 = (i+1)*N+j
				row.append(ij)
				col.append(ij2)
				data.append(-1.0)
			if(j!=0):
				ij3 = i*N+j-1
				row.append(ij)
				col.append(ij3)
				data.append(-1.0)
			if(j!=N-1):
				ij4 = i*N+j+1
				row.append(ij)
				col.append(ij4)
				data.append(-1.0)
				
	# Eigen value solver using LOBPCG method
	A = sparse.csr_matrix((data, (row, col)), shape=(N*N, N*N)) 	# for 2D matrix
	ml = smoothed_aggregation_solver(A)								# create the AMG hierarchy
	X = scipy.rand(A.shape[0], K) 									# initial approximation to the K eigenvectors
	M = ml.aspreconditioner()										# preconditioner based on ml
	W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)				# compute eigenvalues and eigenvectors with LOBPCG
	WW[m] = W*57													# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm
	VV[m,:] = V[:,0]

#############################################################################################################
# Plotting eigen energies																					#
#############################################################################################################
plt.title('$E$ (energy) vs gap (gap between the wires) for different energy levels')
pltmr0, = plt.plot(0.2*abs(L-10),WW[:,0])
pltmr1, = plt.plot(0.2*abs(L-10),WW[:,2])
pltmr2, = plt.plot(0.2*abs(L-10),WW[:,3])
pltmr3, = plt.plot(0.2*abs(L-10),WW[:,5])
pltmr4, = plt.plot(0.2*abs(L-10),WW[:,7])
pltmr5, = plt.plot(0.2*abs(L-10),WW[:,8])
plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5],["L0", "L1 &L2", "L3", "L4 & L5", "L6 & L7", "L8"],loc=5)
plt.xlabel('gap (in nm)')
plt.ylabel('$E$ (in eV)')
plt.show()

#############################################################################################################
# Plotting ground state for varying wire sizes																#
#############################################################################################################
pylab.figure(figsize=(9,9))
for i in range(len(L)/4+1):
    pylab.subplot(3, 3, i+1)
    pylab.title('gap=%0.1lf nm' % (0.2*abs(L[i*4]-10)))
    pylab.pcolor(VV[i*4,:].reshape(N,N),cmap="RdGy")
    pylab.axis('equal')
    pylab.axis('off')

pylab.show()       
