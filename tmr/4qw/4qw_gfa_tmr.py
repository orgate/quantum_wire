#############################################################################################################
#																											#
#	Title			:	Tunneling conductances and TMR for the case with four nanowires placed apart		#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/tmr/4qw/4qw_gfa_tmr.py													#
#	Figure ref.		:	Figures 3.14 and 3.15																#
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
# Initializing the constants and simulation parameters														#
#############################################################################################################
CONST = 1.756e+18													# 2me/(hbar*hbar) - relating energy to wave vector squared
N = 100																# Wire size in terms of number of grids (one grid is assumed to be 0.1nm)
K = 9																# Number of eigen modes below EF (Fermi energy)
E = 6.0																# EF (Fermi energy in eV)
VB = 12.0															# Barrier potential
h0 = 0.2															# Exchange potential
z1 = 50e-9															# left layer is 50nm long
d = 10e-9															# spacer layer is 10nm thick
z2 = z1 + d	
pot = 10.0/57 														# Substrate crystal potential
L = np.arange(-(N/5),(N/10),4)										# varying gaps between the wires


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


GFM = L*0
GFm = L*0
GAMm = L*0
TMR = L*0

#############################################################################################################
# Finding transmission probabilities, tunneling conductances and TMR for varying gaps between wires			#
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
				
	A = sparse.csr_matrix((data, (row, col)), shape=(N*N, N*N)) 	# 2D matrix for 4 wires placed on a substrate
	ml = smoothed_aggregation_solver(A)								# create the AMG hierarchy
	X = scipy.rand(A.shape[0], K) 									# initial approximation to the K eigenvectors
	M = ml.aspreconditioner()										# preconditioner based on ml
	W,V = lobpcg(A, X, M=M, tol=1e-8, largest=False)				# compute eigenvalues and eigenvectors with LOBPCG
	W*=57															# factor used to relate it to the actual eigen energy in eV, when the 																		  dimensions of the grids of the wire is about 0.1nm

	# Finds transmission probabilities (TF and TA), tunneling conductances (GF and GA) and TMR for varying gaps between wires
	k2pp = CONST*W													# relating energy to wave vector squared

	kB = k(-E,-VB,-k2pp)											# wave vector in barrier spacer layer
	kM = k(E,-h0,k2pp)												# wave vector for majority spin channel
	km = (k(E,h0,k2pp)).real+1										# wave vector for minority spin channel

	R1M = R(kM,kB,z1,1)/(2*kM)										# Transformation matrix at the left FM-barrier interface for majority FM
	R2M = R(kM,kB,z2,-1)*1j/(2*kB)									# Transformation matrix at the right FM-barrier interface for majority FM

	R1m = R(km,kB,z1,1)/(2*km)										# Transformation matrix at the left FM-barrier interface for minority FM
	R2m = R(km,kB,z2,-1)*1j/(2*kB)									# Transformation matrix at the right FM-barrier interface for minority FM

	TMM = (kM/kM)/((abs(R1M[0,0,:]*R2M[0,0,:]+R1M[0,1,:]*R2M[1,0,:]))**2)	# transmission probability
	Tmm = (km/km)/((abs(R1m[0,0,:]*R2m[0,0,:]+R1m[0,1,:]*R2m[1,0,:]))**2)	# transmission probability
	TMm = (km/kM)/((abs(R1M[0,0,:]*R2m[0,0,:]+R1M[0,1,:]*R2m[1,0,:]))**2)	# transmission probability

	dk2 = np.roll(k2pp,-1) - k2pp
	dk2[-1] = CONST - k2pp[-1]

	GFM[m] = np.cumsum((1.962635e-6)*dk2*np.roll(TMM,1))[-2]		# conductance for ferromagnetic spin majority channel
	GFm[m] = np.cumsum((1.962635e-6)*dk2*np.roll(Tmm,1))[-2]		# conductance for ferromagnetic spin minority channel
	GAMm[m] = np.cumsum((1.962635e-6)*dk2*np.roll(TMm,1))[-2]		# conductance for anti-ferromagnetic channel

	GF = GFM + GFm
	GA = 2*GAMm

	TMR[m] = (GF[m]-GA[m])*100/GA[m]

# Plotting tunneling conductances for the case with 4 nanowires
gfM, = plt.plot(0.2*abs(L-10),GFM)
gfm, = plt.plot(0.2*abs(L-10),GFm)
gaMm, = plt.plot(0.2*abs(L-10),GAMm)
plt.legend([gfM, gfm, gaMm],["$G_{F}^{\downarrow}$", "$G_{F}^{\uparrow}$", "$0.5G_{A}$"],loc=3)
plt.title('Tunneling conductance vs Gap (in nm)')
plt.ylabel("Conductance per unit area")
plt.xlabel("Gap between wires $(nm)$")
plt.show()


# Plotting TMR for the case with 4 nanowires
plt.plot(0.2*abs(L-10),TMR)
plt.title('Tunneling magnetoresistance (TMR) ratio vs Gap (in $nm$)')
plt.ylabel("TMR ratio (percentage)")
plt.xlabel("Gap between wires $(nm)$")
plt.show()

