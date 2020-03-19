#############################################################################################################
#																											#
#	Title			:	Functions used for STT when distribution of electrons are considered				#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/stt/dist/defs.py															#
#	Figure ref.		:	NA																					#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import scipy
import cmath as c
from scipy.sparse.linalg import lobpcg
from scipy import sparse
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import poisson


#############################################################################################################
# Solves eigen value equation using LOBPCG method															#
#############################################################################################################
def solve(N,K):
	A = poisson((N,N), format='csr')
	ml = smoothed_aggregation_solver(A)								# create the AMG hierarchy
	X = scipy.rand(A.shape[0], K) 									# initial approximation to the K eigenvectors
	M = ml.aspreconditioner()										# preconditioner based on ml
	W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)				# compute eigenvalues and eigenvectors with LOBPCG
	W*=57															# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm
	return W

#############################################################################################################
# Reflection and transmission related parameters															#
#############################################################################################################
def RsTs(q):
	kx = np.sqrt(1-q*q)												# wave vector of electrons in non-magnet
	kxu = np.sqrt(1.5*1.5-q*q)										# wave vector of up-spin electrons
	kxd = np.sqrt(0.5*0.5-q*q)										# wave vector of down-spin electrons

	Ru = (kx-kxu)/(kx+kxu)											# Reflection coeffcient of up-spin electrons
	Rd = (kx-kxd)/(kx+kxd)											# Reflection coeffcient of down-spin electrons
	Tu = 2*kx/(kx+kxu)												# Transmission coeffcient of up-spin electrons
	Td = 2*kx/(kx+kxd)												# Transmission coeffcient of down-spin electrons

	RU = abs(Ru)**2													# Reflection probability of up-spin electrons
	RD = abs(Rd)**2													# Reflection probability of down-spin electrons
	TU = kxu.real*(abs(Tu)**2)/kx									# Transmission probability of up-spin electrons
	TD = kxd.real*(abs(Td)**2)/kx									# Transmission probability of down-spin electrons
	Rud = abs(Ru*Rd.conjugate())									# Reflected spin current density
	Tud = (Tu*Td*((kxd+kxu)/(2*kx))*np.exp((kxd*100-kxu*100)*1j))	# Transmitted spin current density
	return kx,kxu,kxd,Ru,Rd,Tu,Td

