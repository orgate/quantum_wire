#############################################################################################################
#																											#
#	Title			:	First 10 eigen energies for increasing wire sizes 									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_lobpcg/1qw_E_vs_lw.py												#
#	Figure ref.		:	Figures 2.17																		#
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

K = 10
NN = np.arange(10,200,10)											# increasing wire sizes
WW = np.zeros((len(NN),K))											# number of eigen modes with eigen energies below EF

for i in range(len(NN)):
	n = NN[i]
	A = poisson((n,n), format='csr')
	ml = smoothed_aggregation_solver(A)								# create the AMG hierarchy
	X = scipy.rand(A.shape[0], K) 									# initial approximation to the K eigenvectors
	M = ml.aspreconditioner()										# preconditioner based on ml
	W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)				# compute eigenvalues and eigenvectors with LOBPCG
	WW[i] = W*57													# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm

NN/=10

#############################################################################################################
# Plotting E vs lw																							#
#############################################################################################################
plt.title('$E$ (energy) vs $l_{w}$ (wire size) for first $10$ eigen levels')
pltmr0, = plt.plot(NN,WW[:,0])
pltmr1, = plt.plot(NN,WW[:,2])
pltmr2, = plt.plot(NN,WW[:,3])
pltmr3, = plt.plot(NN,WW[:,5])
pltmr4, = plt.plot(NN,WW[:,7])
pltmr5, = plt.plot(NN,WW[:,8])
pltmr6, = plt.plot(NN,WW[:,9])
plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5, pltmr6],["L0", "L1 &L2", "L3", "L4 & L5", "L6 & L7", "L8", "L9"],loc=1)

plt.xlabel('$l_{w}$ (in $nm$)')
plt.ylabel('$E$ (in eV)')
plt.show()
