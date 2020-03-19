#############################################################################################################
#																											#
#	Title			:	Transmission probabilities, tunneling conductances and TMR for planar and nanowire  #
#						tunnel junctions as a function of barrier potential									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/tmr/vs_vb/tfa_gfa_tmr_vs_vb.py											#
#	Figure ref.		:	Figures 3.9, 3.10, 3.11, 3.12 and 3.13												#
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
import defs															# importing custom library

#############################################################################################################
# Initializing the constants and simulation parameters														#
#############################################################################################################
TFA_GFA_TMR = 0														# "0" means 'Transmission probability', "1" means 'Tunneling conductance',
																	# "2" means 'Relative tunneling conductance' and "3" means 'Tunneling
																	# Magneto Resistance'

traprob = 0															# "0" means 'Transmission probability of majority spin electrons in (F)
																	# alignment', "1" means 'Transmission probability of 	
																	# minority spin electrons in (F) alignment' and "2" means 'Transmission
																	# probability of electrons in (A) alignment'

CONST = 1.756e+18													# 2me/(hbar*hbar) - relating energy to wave vector squared

N = 100
K = 11
Narray = np.array([53, 67, 74, 85, 97, 100])						# Sizes of the wire
Karray = np.array([2, 3, 5, 7, 9, 10])								# Number of modes considered for each wire size (below EF)
E = 0.8+0j															# EF (Fermi energy in eV)
VB = np.arange(0.5,5.0,0.01)+0j										# Barrier potential
h0 = 0.2+0j															# Exchange potential
z1 = 50e-9															# left layer is 50nm thick
d = 10e-9															# Barrier thickness
z2 = z1 + d															# location of the right layer interface


#############################################################################################################
# Function to plot transmission probabilities for different eigen modes										#
#############################################################################################################
def plotTraProb(VB,Tplt,traprob):
	pltmr0, = plt.plot(VB,Tplt[:,0])
	pltmr1, = plt.plot(VB,Tplt[:,2])
	pltmr2, = plt.plot(VB,Tplt[:,3])
	pltmr3, = plt.plot(VB,Tplt[:,5])
	pltmr4, = plt.plot(VB,Tplt[:,7])
	pltmr5, = plt.plot(VB,Tplt[:,9])
	pltmr6, = plt.plot(VB,Tplt[:,10])

	plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5, pltmr6],["L0", "L1 &L2", "L3", "L4 & L5", "L6 & L7", "L8 & L9", "L10"],loc=1)
	plt.yscale('log')
	if(traprob==0):
		plt.ylabel("Transmission probability $T_F^{\downarrow}(k_{\Vert})$", fontsize=20)
	elif(traprob==1):
		plt.ylabel("Transmission probability $T_F^{\uparrow}(k_{\Vert})$", fontsize=20)
	elif(traprob==2):
		plt.ylabel("Transmission probability $T_A(k_{\Vert})$", fontsize=20)

#############################################################################################################
# Function to find transmission probabilities, tunneling conductances and TMR								#
#############################################################################################################
def TMR_G(W,E,VB,h0,z1,z2,Tplt):
	k2pp = CONST*W													# relating energy to wave vector squared
	kM = (defs.k(E,-h0,k2pp))										# wave vector for majority spin channel
	km = (defs.k(E,h0,k2pp)).real+1									# wave vector for minority spin channel
	gfM = VB*0+0j
	gfm = VB*0+0j
	gaMm = VB*0+0j
	tmr = VB*0+0j
	dk2 = np.roll(k2pp,-1) - k2pp									# differential used while integrating or summing over all eigen modes
	dk2[-1] = 1.0*CONST-k2pp[-1]									# differential for the last eigen mode

	for i in range(len(VB)):
		kB = (defs.k(-E,-VB[i],-k2pp))								# wave vector in barrier spacer layer (could be just real too)
		R1M = defs.R(kM,kB,z1,1)/(2*kM)
		R2M = defs.R(kM,kB,z2,-1)*1j/(2*kB)
		R1m = defs.R(km,kB,z1,1)/(2*km)
		R2m = defs.R(km,kB,z2,-1)*1j/(2*kB)
		TMM = (kM/kM)/((abs(R1M[0,0,:]*R2M[0,0,:]+R1M[0,1,:]*R2M[1,0,:]))**2)	# transmission probability
		Tmm = (km/km)/((abs(R1m[0,0,:]*R2m[0,0,:]+R1m[0,1,:]*R2m[1,0,:]))**2)	# transmission probability
		TMm = (km/kM)/((abs(R1M[0,0,:]*R2m[0,0,:]+R1M[0,1,:]*R2m[1,0,:]))**2)	# transmission probability
	
		if(traprob==0):
			Tplt[i,:] = TMM
		elif(traprob==1):
			Tplt[i,:] = Tmm
		elif(traprob==2):
			Tplt[i,:] = TMm

		gfM[i] = np.cumsum((1.962635e-6)*dk2*TMM)[-1]				# conductance for ferromagnetic spin majority channel
		gfm[i] = np.cumsum((1.962635e-6)*dk2*Tmm)[-1]				# conductance for ferromagnetic spin minority channel
		gaMm[i] = np.cumsum((1.962635e-6)*dk2*TMm)[-1]				# conductance for anti-ferromagnetic channel

		gf = gfM + gfm
		ga = 2*gaMm

		tmr[i] = (gf[i]-ga[i])*100/ga[i]
	return Tplt,gfM,gfm,gaMm,tmr


#############################################################################################################
# Calulating TF and TA for different eigen modes, GF and GA for nanowire case								#
#############################################################################################################
W = defs.solve(N,K)
Tplt = np.zeros((len(VB),len(W)))
tmr_g = TMR_G(W,E,VB,h0,z1,z2,Tplt)

#############################################################################################################
# Calculating TMR																							#
#############################################################################################################
TMR = np.zeros((len(Narray)+1,len(VB)))
for i in range(len(Narray)):
#	if i>0:
	W1 = defs.solve(Narray[i],Karray[i]+1)
	Tplt = np.zeros((len(VB),len(W1)))
	TMRG = TMR_G(W1,E,VB,h0,z1,z2,Tplt)
	TMR[i,:] = TMRG[4]

# Planar case
W1 = np.arange(0,1,0.01)
Tplt = np.zeros((len(VB),len(W1)))
TMRG = TMR_G(W1,E,VB,h0,z1,z2,Tplt)
TMR[-1,:] = TMRG[4]

d*=1e9

if(TFA_GFA_TMR==0):

	# Plotting transmission probabilities
	plotTraProb(VB,tmr_g[0],traprob)
elif(TFA_GFA_TMR==1):

	# Plotting tunneling conductances for nanowire magnetic tunnel junction
	gfM, = plt.plot(VB,tmr_g[1])
	gfm, = plt.plot(VB,tmr_g[2])
	gaMm, = plt.plot(VB,tmr_g[3])

	W = np.arange(0.01,1.0,0.01)+0j
	Tplt = np.zeros((len(VB),len(W)))
	tmr_g1 = TMR_G(W,E,VB,h0,z1,z2,Tplt)

	# Plotting tunneling conductances for planar magnetic tunnel junction
	gfM1, = plt.plot(VB,tmr_g1[1])
	gfm1, = plt.plot(VB,tmr_g1[2])
	gaMm1, = plt.plot(VB,tmr_g1[3])
	plt.legend([gfM, gfm, gaMm, gfM1, gfm1, gaMm1],["$G_{F}^{\downarrow}$ (nanowire)", "$G_{F}^{\uparrow}$ (nanowire)", "$0.5G_{A}$ (nanowire)", "$G_{F}^{\downarrow}$ (planar)", "$G_{F}^{\uparrow}$ (planar)", "$0.5G_{A}$ (planar)"],loc=1, fontsize=15)
	plt.yscale('log')
	plt.ylabel("Tunneling conductance per unit area", fontsize=20)
elif(TFA_GFA_TMR==2):

	# Plotting relative tunneling conductances for nanowire tunnel junction
	gfM, = plt.plot(VB,tmr_g[1]/tmr_g[2])
	gfm, = plt.plot(VB,tmr_g[2]/tmr_g[2])
	gaMm, = plt.plot(VB,tmr_g[3]/tmr_g[2])
	plt.legend([gfM, gfm, gaMm],["$G_{F}^{\downarrow}$ (nanowire)", "$G_{F}^{\uparrow}$ (nanowire)", "$0.5G_{A}$ (nanowire)"],loc=1, fontsize=20)
	plt.ylabel("Relative tunneling conductance", fontsize=20)
	plt.xlabel("Barrier potential $V_{B}$ (V)", fontsize=15)
	plt.show()

	W = np.arange(0.01,1.0,0.01)+0j
	Tplt = np.zeros((len(VB),len(W)))
	tmr_g = TMR_G(W,E,VB,h0,z1,z2,Tplt)

	# Plotting relative tunneling conductances for planar tunnel junction
	gfM, = plt.plot(VB,tmr_g[1]/tmr_g[2])
	gfm, = plt.plot(VB,tmr_g[2]/tmr_g[2])
	gaMm, = plt.plot(VB,tmr_g[3]/tmr_g[2])
	plt.legend([gfM, gfm, gaMm],["$G_{F}^{\downarrow}$ (planar)", "$G_{F}^{\uparrow}$ (planar)", "$0.5G_{A}$ (planar)"],loc=1, fontsize=20)
	plt.ylabel("Relative tunneling conductance", fontsize=20)
elif(TFA_GFA_TMR==3):

	# Plotting TMR for both nanowire and planar cases
	pltmr0, = plt.plot(VB,TMR[0,:])
	pltmr1, = plt.plot(VB,TMR[1,:])
	pltmr2, = plt.plot(VB,TMR[2,:])
	pltmr3, = plt.plot(VB,TMR[3,:])
	pltmr4, = plt.plot(VB,TMR[4,:])
	pltmr5, = plt.plot(VB,TMR[5,:])
	pltmr6, = plt.plot(VB,TMR[6,:],'+')
	plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5, pltmr6],["5.3nm", "6.7nm", "7.4nm", "8.5nm", "9.7nm", "10.0nm", "planar"],loc=2)
	plt.ylabel("TMR ratio (percentage)")

plt.xlabel("Barrier potential $V_{B}$ (V)", fontsize=15)
plt.show()

