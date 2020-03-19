#############################################################################################################
#																											#
#	Title			:	Simulation of wave function using finite difference method for circular quantum wir-#
#						e and the error compared to the analytical solution									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_fdm/cqw/fdm_cqw_sim_err_iter.py										#
#	Figure ref.		:	Figures 2.11, 2.12 and 2.13															#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

#	This code solves time-independent schrodinger equation in polar coordinate system:
#	Zrr + Zr/r + Z00/(r*r) = K*Z
#	where Zrr means 2nd partial derivative of Z along radial-axis and similarly Z00
#	means, 2nd partial derivative of Z along azimuhal axis.
#	K is some real number, representing the total energy of the system
# 	Here the initial conditions and boundary conditions are all zero except for a delta input 
#	at certain points depending upon the energy levels along the different axis.
#	For the following code Z(r,0) is assumed to be the electron wavefunction 
#	Z(r,0) = 0 at the boundaries and depending on the values of nr and n0 the initial conditions are given.
#	
#	This particular code has options to plot the actual electron wavefuncion, or the error w.r.t actual
#	solution, or the maximum error for different number of iterations or the mean error for different
#	number of iterations.

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma, airy, jn, jn_zeros
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

sim_err_iter=0														# "0" plots the wave functions, "1" plots the error in the wavefunction
																	# and "2" plots the variation of the errors for increasing number of
																	# iterations

times	=	1000													# No. of times the iteration happens
delr	=	0.01													# The radial-axis length scale
del0	=	1.0														# The angular-axis length scale

Lr		=	1.0														# Length along radial-axis
nr		=	2														# radial-axis energy level
n0		=	1														# angular-axis energy level


n0*=2
K=5*(nr*nr + n0*n0)													# The parameters defining the energy

theta = np.radians(np.linspace(0, 360, 360.0/del0))					#angular-axis input values
R = np.arange(0, Lr, delr)											#radial-axis input values

cols = len(theta)													# No. of columns or no. of angular-axis length divisions 
rows = len(R)														# No. of rows or no. of radial-axis length divisions

theta, R = np.meshgrid(theta, R)									# mesh grid generated for the entire circular area

Z = theta*R*0
Z_act = theta*R*0

minmax_err = 1
minmean_err = 1
max_err_list = []
mean_err_list = []

del0*=(2*np.pi/360.0)												# The angular-axis length scale, expressed in radians

#############################################################################################################
# The actual analytical solution																			#
#############################################################################################################
for i in range(rows):
	for j in range(cols):
		Z_act[i,j] = jn(nr,0.04*nr*i)*np.cos(j*np.pi*n0/cols)		# psi(R)*psi(0)
integ_act = np.cumsum(Z_act**2)[-1]
Z_act = Z_act/np.sqrt(integ_act)									# Normalizing the actual solution


#############################################################################################################
# Boundary condition and initial condition are characterized in Z											#
#############################################################################################################
for i in range(rows):
	for j in range(cols):
		if ((i+1)%(rows/(2*nr))==0 or (i)%(rows/(2*nr))==0) and ((j+1)%(cols/(2*n0))==0 or (j)%(cols/(2*n0))==0) and i!=0 and i!=rows-1:
			if ((((i+1)/(rows/(2*nr)))%4==1 or ((i)/(rows/(2*nr)))%4==1) and (((j+1)/(cols/(2*n0)))%4==0 or ((j)/(cols/(2*n0)))%4==0)) or ((((i+1)/(rows/(2*nr)))%4==3 or ((i)/(rows/(2*nr)))%4==3) and (((j+1)/(cols/(2*n0)))%4==2 or ((j)/(cols/(2*n0)))%4==2)):
				Z[i,j] = 1
			elif ((((i+1)/(rows/(2*nr)))%4==1 or ((i)/(rows/(2*nr)))%4==1) and (((j+1)/(cols/(2*n0)))%4==2 or ((j)/(cols/(2*n0)))%4==2)) or ((((i+1)/(rows/(2*nr)))%4==3 or ((i)/(rows/(2*nr)))%4==3) and (((j+1)/(cols/(2*n0)))%4==0 or ((j)/(cols/(2*n0)))%4==0)):
				Z[i,j] = -1


#############################################################################################################
# Iteration begins																							#
#############################################################################################################
for expr in range(times):

	#	The following lines solve the PDE (partial differential equation) using finite difference method
	Z[1:-1,1:-1] = ( (Z[2:,1:-1]/(delr*delr)) + (Z[0:-2,1:-1]/(delr*delr)) + (Z[2:,1:-1]/(2*R[1:-1,1:-1]*delr)) - (Z[0:-2,1:-1]/(2*R[1:-1,1:-1]*delr)) + (Z[1:-1,2:]/(R[1:-1,1:-1]*R[1:-1,1:-1]*del0*del0)) + (Z[1:-1,0:-2]/(R[1:-1,1:-1]*R[1:-1,1:-1]*del0*del0)) )/( (2/(delr*delr)) + (2/(R[1:-1,1:-1]*R[1:-1,1:-1]*del0*del0)) - K )
	#	At angle 0 degree boundary
	Z[1:-1,0] = ( (Z[2:,0]/(delr*delr)) + (Z[0:-2,0]/(delr*delr)) + (Z[2:,0]/(2*R[1:-1,0]*delr)) - (Z[0:-2,0]/(2*R[1:-1,0]*delr)) + (Z[1:-1,1]/(R[1:-1,0]*R[1:-1,0]*del0*del0)) + (Z[1:-1,-1]/(R[1:-1,0]*R[1:-1,0]*del0*del0)) )/( (2/(delr*delr)) + (2/(R[1:-1,0]*R[1:-1,0]*del0*del0)) - K )
	#	At angle 360 degree boundary
	Z[1:-1,-1] = ( (Z[2:,-1]/(delr*delr)) + (Z[0:-2,-1]/(delr*delr)) + (Z[2:,-1]/(2*R[1:-1,-1]*delr)) - (Z[0:-2,-1]/(2*R[1:-1,-1]*delr)) + (Z[1:-1,0]/(R[1:-1,-1]*R[1:-1,-1]*del0*del0)) + (Z[1:-1,-2]/(R[1:-1,-1]*R[1:-1,-1]*del0*del0)) )/( (2/(delr*delr)) + (2/(R[1:-1,-1]*R[1:-1,-1]*del0*del0)) - K )

	integ=np.cumsum(Z*Z)[-1]
	Z/=np.sqrt(integ)												# Normalizing the calculated solution

	#	Error Analysis
	mean_err_i = np.cumsum(abs(Z-Z_act))[-1]/(cols*rows)
	max_err_i = (Z-Z_act).max()
	max_err_list.append(max_err_i)
	mean_err_list.append(mean_err_i)

	if minmax_err>max_err_i:
		minmax_err=max_err_i
		times_min = expr

	if minmean_err>mean_err_i:
		minmean_err=mean_err_i
		times_mean = expr

#############################################################################################################
# Plotting the outputs																						#
#############################################################################################################
if(sim_err_iter==0):

	#	Plots the numerical solution
	fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
	plt.title('Phi for 2D circular quantum wire. $n_{p}=1,n_{\phi}=1$')
	surf = ax.contourf(theta,R,Z,100)
	fig.colorbar(surf, shrink=0.5, aspect=10)

elif(sim_err_iter==1):

	#	Plots the error in numerical solution
	fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
	plt.title('Error in phi for 2D circular quantum wire. $n_{p}=1,n_{\phi}=1$')
	surf = ax.contourf(theta,R,abs(Z-Z_act),100)
	fig.colorbar(surf, shrink=0.5, aspect=10)

elif(sim_err_iter==2):

	#	Plots the maximum error for different iterations
	plt.plot(max_err_list)
	plt.title('Maximum error in phi for different number of iterations. $n_{p}=1,n_{\phi}=1$')
	plt.show()

	#	Plots the mean error for different iterations
	plt.plot(mean_err_list)
	plt.title('Mean error in phi for different number of iterations. $n_{p}=1,n_{\phi}=1$')

plt.show()
