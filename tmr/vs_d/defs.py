#############################################################################################################
#																											#
#	Title			:	Functions to calculate wave vectors, transformation matrices at FM-barrier interfac-#
#						es and for eigen value solver														#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/tmr/vs_d/defs.py															#
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

CONST = 1.756e+18													# 2me/(hbar*hbar) - relating energy to wave vector squared

#############################################################################################################
# Calculating wave vector																					#
#############################################################################################################
def k(E,V,k2):
	return np.sqrt(CONST*(E-V)-k2)

def add(k,K):
	return (k+K*1j)

#############################################################################################################
# Calculating transformation matrices for ferromagnet metal-insulator barrier interfaces					#
#############################################################################################################
def R(k,K,z,sgn):
	R = np.zeros((2,2,len(k)))*1j
	R[0,0,:] = add(k,-K)*np.exp(sgn*z*add(K,-k))
	R[0,1,:] = sgn*add(k,K)*np.exp(z*add(-K,-k))
	R[1,0,:] = sgn*add(k,K)*np.exp(z*add(K,k))
	R[1,1,:] = add(k,-K)*np.exp(z*sgn*add(-K,k))
	return R

#############################################################################################################
# Eigen value solver based on LOBPCG																		#
#############################################################################################################
def solve(N,K):
	A = poisson((N,N), format='csr')
	ml = smoothed_aggregation_solver(A)								# create the AMG hierarchy
	X = scipy.rand(A.shape[0], K) 									# initial approximation to the K eigenvectors
	M = ml.aspreconditioner()										# preconditioner based on ml
	W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)				# compute eigenvalues and eigenvectors with LOBPCG
	W*=57															# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm
	return W

