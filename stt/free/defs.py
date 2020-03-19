#############################################################################################################
#																											#
#	Title			:	Function for several Reflection and transmission related parameters					#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/stt/free/defs.py															#
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
import cmath as c

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
	return Ru,Rd,Tu,Td,RU,RD,TU,TD,Rud,Tud


