#############################################################################################################
#																											#
#	Title			:	First 9 eigen energies and modes for a single square quantum wire					#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_lobpcg/1qw_psi_E.py													#
#	Figure ref.		:	Figures 2.15 and 2.16																#
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
A = poisson((N,N), format='csr')
ml = smoothed_aggregation_solver(A)									# create the AMG hierarchy
X = scipy.rand(A.shape[0], K) 										# initial approximation to the K eigenvectors
M = ml.aspreconditioner()											# preconditioner based on ml
W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)					# compute eigenvalues and eigenvectors with LOBPCG
W*=57																# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm

#############################################################################################################
# Plotting eigen energies																					#
#############################################################################################################
plt.plot(W)
plt.show()

#############################################################################################################
# Plotting eigen modes																						#
#############################################################################################################
pylab.figure(figsize=(9,9))
for i in range(K):
    pylab.subplot(3, 3, i+1)
    pylab.title('Eigenvector %d' % i)
    pylab.pcolor(V[:,i].reshape(N,N),cmap="RdGy")
    pylab.axis('equal')
    pylab.axis('off')

pylab.show()    
