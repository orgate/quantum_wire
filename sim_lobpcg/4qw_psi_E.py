#############################################################################################################
#																											#
#	Title			:	First 9 eigen energies and modes when a 4 quantum wires	are placed on a substrate	#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_lobpcg/4qw_psi_E.py													#
#	Figure ref.		:	Figures 2.18 and 2.19																#
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

row = []
col = []
data = []

#############################################################################################################
# Setting potentials such that 4 wires are placed on a substrate											#
#############################################################################################################
for i in range(N):
	for j in range(N):
		ij = i*N+j
		d = 4.0
		row.append(ij)
		col.append(ij)
		if (not((((i>=(N/5))&(i<(2*N/5)))|((i>=(3*N/5))&(i<(4*N/5))))&(((j>=(N/5))&(j<(2*N/5)))|((j>=(3*N/5))&(j<(4*N/5)))))):
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
A = sparse.csr_matrix((data, (row, col)), shape=(N*N, N*N)) 		# for 2D matrix
ml = smoothed_aggregation_solver(A)									# create the AMG hierarchy
X = scipy.rand(A.shape[0], K) 										# initial approximation to the K eigenvectors
M = ml.aspreconditioner()											# preconditioner based on ml
W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)					# compute eigenvalues and eigenvectors with LOBPCG
W*=57																# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm
#############################################################################################################
# Plotting eigen energies																					#
#############################################################################################################
plt.plot(W)
plt.ylabel('$E$ (in eV)')
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
