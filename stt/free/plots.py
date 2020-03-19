#############################################################################################################
#																											#
#	Title			:	Fucntions for several kinds of plots												#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/stt/free/plots.py														#
#	Figure ref.		:	NA																					#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import cmath as c

#############################################################################################################
# Plots spin filtering - Reflection probability & current													#
#############################################################################################################
def plotR(q,RU,RD,Rud,nwpl):

	# nanowire tunnel junction
	if(nwpl==0):
		R_up, = plt.plot(q,RU)
		R_down, = plt.plot(q,RD)
		R_ud, = plt.plot(q,Rud)
		plt.title('Spin filtering - Reflection probability & current (planar)')

	# planar tunnel junction
	else:
		R_up, = plt.plot(q,RU,'+')
		R_down, = plt.plot(q,RD,'.')
		R_ud, = plt.plot(q,Rud,'o')
		plt.title('Spin filtering - Reflection probability & current (nanowire)')
	plt.legend([R_up, R_down, R_ud], ["$R^{\uparrow}$ (probability)", "$R^{\downarrow}$ (probability)", "$|R_{\uparrow}R_{\downarrow}^{*}|}$ reflected spin current density"],loc=2)
	return None

#############################################################################################################
# Plots spin filtering - Transmission probability & current													#
#############################################################################################################
def plotT(q,TU,TD,Tud,nwpl):

	# nanowire tunnel junction
	if(nwpl==0):
		T_up, = plt.plot(q,TU)
		T_down, = plt.plot(q,TD)
		T_ud, = plt.plot(q,abs(Tud))
		plt.title('Spin filtering - Transmission probability & current (planar)')

	# planar tunnel junction
	else:
		T_up, = plt.plot(q,TU,'+')
		T_down, = plt.plot(q,TD,'.')
		T_ud, = plt.plot(q,abs(Tud),'o')
		plt.title('Spin filtering - Transmission probability & current (nanowire)')
	plt.legend([T_up, T_down, T_ud], ["$T^{\uparrow}$ (probability)", "$T^{\downarrow}$ (probability)", "$|T_{\uparrow}T_{\downarrow}^{*}\Phi|_{x\longrightarrow100}$ transmitted spin current density"],loc=3)
	return None

#############################################################################################################
# Plots spin filtering - Transverse spin current															#
#############################################################################################################
def plotTra(q,Rud,Tud,nwpl):
	T_spin = abs(Rud) + abs(Tud)

	# nanowire tunnel junction
	if(nwpl==0):
		Tra_spin, = plt.plot(q,T_spin)
		plt.title('Spin filtering - Transverse spin current: $|R^{\uparrow}R^{\downarrow}|+|T_{\uparrow}T_{\downarrow}^{*}\Phi|_{x\longrightarrow100}$ (planar)')

	# planar tunnel junction
	else:
		Tra_spin, = plt.plot(q,T_spin,'o')
		plt.title('Spin filtering - Transverse spin current: $|R^{\uparrow}R^{\downarrow}|+|T_{\uparrow}T_{\downarrow}^{*}\Phi|_{x\longrightarrow100}$ (nanowire)')
	plt.legend([Tra_spin], ["$|R^{\uparrow}R^{\downarrow}|+|T_{\uparrow}T_{\downarrow}^{*}\Phi|_{x\longrightarrow100}$"])


#############################################################################################################
# Plots spin rotation																						#
#############################################################################################################
def plotSR(q,Ru,Rd,nwpl):
	Spin_Rot = (np.log(Ru*Rd)).imag

	# nanowire tunnel junction
	if(nwpl==0):
		Spin_R = plt.plot(q,Spin_Rot)
		plt.title('Spin rotation: $phase(R_{\uparrow}R_{\downarrow} ) $ (planar)')

	# planar tunnel junction
	else:
		Spin_R = plt.plot(q,Spin_Rot,'o')
		plt.title('Spin rotation: $phase(R_{\uparrow}R_{\downarrow} ) $ (nanowire)')
	plt.ylabel('radians')


#############################################################################################################
# Plots spatial precession																					#
#############################################################################################################
def plotSP(q,Tu,Td):

	# nanowire and planar tunnel junction
	kx = np.sqrt(1-q*q)
	kxu = np.sqrt(1.5*1.5-q*q)
	kxd = np.sqrt(0.5*0.5-q*q)
	x = np.arange(0,101,1)
	x, y = np.meshgrid(x, q)
	Phase = (np.log((Tu*Td*((kxd+kxu)/(2*kx)))*(np.exp((kxd*x.T-kxu*x.T)*1j)))).imag
	fig, ax = plt.subplots()
	p = ax.pcolor(y, x, Phase.T, cmap=cm.RdBu, vmin=abs(Phase).min(), vmax=abs(Phase).max())
	cb = fig.colorbar(p, ax=ax)
	plt.title('Spatial precession $phase(T_{\uparrow}T_{\downarrow}^{*}\Phi)$ (nanowire)')
	plt.ylabel('x')

