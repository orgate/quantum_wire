#############################################################################################################
#																											#
#	Title			:	Spin transfer torque and transverse transmitted spin current						#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/stt/dist/stt,qtra_qtra_vs_kfx.py											#
#	Figure ref.		:	Figure 4.5 and 4.6																	#
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
import defs as d								# Importing custom library

stt = 0											# "0" means 'transverse transmitted spin current' and "1" means 'spin transfer torque'

#############################################################################################################
# Sizes at which EF (Fermi level) = eigen energy level (making sure one extra level gets below EF)			#
#############################################################################################################
Narray = np.array([53, 67, 74, 85, 97, 100]) 	# wire sizes 
Karray = np.array([2, 3, 5, 7, 9, 10])			# number of modes considered
x = np.arange(0,10,0.1)+0j

#############################################################################################################
# Incident, reflected and transmitted transverse spin currents initalized and calculated for different wire #
# sizes																										#
#############################################################################################################
Qin = np.zeros((len(x),len(Narray)+1))
Qref = np.zeros((len(x),len(Narray)+1))
Qtra = np.zeros((len(x),len(Narray)+1))
for i in range(len(Narray)):
	q = np.sqrt(d.solve(Narray[i],Karray[i]+1)+0j)
	dq = np.roll(q,-1) - q
	dq[-1] = 1-q[-1]
	rsts = d.RsTs(q)

	Qin[:,i] = np.ones(len(x))*np.cumsum(q*dq)[-1]
	Qref[:,i] = np.cumsum(q*dq*abs(rsts[3].conjugate()*rsts[4]))[-1]/Qin[:,i]
	for j in range(len(x)):
		Qtra[j,i] = np.cumsum(q*dq*abs((rsts[1]+rsts[2])*rsts[5].conjugate()*rsts[6]/(2*rsts[0])*np.exp(-1j*(rsts[1]-rsts[2])*x[j])))[-1]/Qin[j,i]

q = np.arange(0,1,0.001)+0j
dq = np.roll(q,-1) - q
dq[-1] = 1-q[-1]								# differential used for integrating over all eigen modes
rsts = d.RsTs(q)								# gets reflection and transmission related parameters

Qin[:,-1] = np.ones(len(x))*np.cumsum(q*dq)[-1]
Qref[:,-1] = np.cumsum(q*dq*abs(rsts[3].conjugate()*rsts[4]))[-1]/Qin[:,-1]
for j in range(len(x)):
	Qtra[j,-1] = np.cumsum(q*dq*abs((rsts[1]+rsts[2])*rsts[5].conjugate()*rsts[6]/(2*rsts[0])*np.exp(-1j*(rsts[1]-rsts[2])*x[j])))[-1]/Qin[j,-1]

if(stt==1):
	# Spin transfer torque per unit area
	Qtra = Qin - Qtra + Qref
	plt.ylabel('$STT$ per unit area')
else:
	plt.ylabel('$Q_{tra}$')

pltmr0, = plt.plot(x,Qtra[:,0])
pltmr1, = plt.plot(x,Qtra[:,1])
pltmr2, = plt.plot(x,Qtra[:,2])
pltmr3, = plt.plot(x,Qtra[:,3])
pltmr4, = plt.plot(x,Qtra[:,4])
pltmr5, = plt.plot(x,Qtra[:,5])
pltmr11, = plt.plot(x,Qtra[:,6],'+')

#############################################################################################################
# Plotting of the Spin transfer torque and transverse transmitted spin current for several wire sizes		#
#############################################################################################################
if(stt==1):
	plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5, pltmr11],["5.3nm", "6.7nm", "7.4nm", "8.5nm", "9.7nm", "10.0nm", "planar"],loc=4)
else:
	plt.legend([pltmr0, pltmr1, pltmr2, pltmr3, pltmr4, pltmr5, pltmr11],["5.3nm", "6.7nm", "7.4nm", "8.5nm", "9.7nm", "10.0nm", "planar"],loc=1)

plt.xlabel('distance (in $K_Fx$)')
plt.show()
