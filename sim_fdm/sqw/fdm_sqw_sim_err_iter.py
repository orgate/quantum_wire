#############################################################################################################
#																											#
#	Title			:	Simulation of wave function using finite difference method for square quantum wire  #
#						and the error compared to the analytical solution									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_fdm/sqw/fdm_sqw_sim_err_iter.py										#
#	Figure ref.		:	Figures 2.8, 2.9 and 2.10															#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

#	This code solves time-independent schrodinger equation in rectangular coordinate system:
#	Zxx + Zyy = K*Z
#	where Zxx means 2nd partial derivative of Z along x-axis and similarly other notations
#	and K is some real number
# 	Here the initial conditions and boundary conditions are all zero except for a delta input 
#	at certain points depending upon the energy levels along the different axis.
#	For the following code Z(x,y) is assumed to be the electron wavefunction 
#	Z(X,Y) = 0 at the boundaries and depending on the values of nx and ny the initial conditions are given.
#	
#	This particular code has options to plot the actual electron wavefuncion, or the error w.r.t actual
#	solution, or the maximum error for different number of iterations or the mean error for different
#	number of iterations.

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

sim_err_iter=0														# "0" plots the wave functions, "1" plots the error in the wavefunction
																	# and "2" plots the variation of the errors for increasing number of
																	# iterations

times=1000															# No. of times the iteration happens
h = 0.025															# The x-axis length scale
k = 0.025															# The y-axis length scale

Lx = 2																# Length along x-axis
Ly = 2																# Length along y-axis
nx=1																# x-axis energy level or quantum number
ny=2																# y-axis energy level or quantum number

X = np.arange(0, Lx, h)												# X-axis input values
Y = np.arange(0, Ly, k)												# Y-axis input values

cols = len(X)														# No. of columns or no. of x-axis length divisions 
rows = len(Y)														# No. of rows or no. of y-axis length divisions

X, Y = np.meshgrid(X, Y)											# mesh grid generated for the entire area

#############################################################################################################
# The actual analytical solution																			#
#############################################################################################################
Z_act = ((np.sqrt(2/Lx))*(np.sin(np.pi*nx*X/Lx)))*((np.sqrt(2/Ly))*(np.sin(np.pi*ny*Y/Ly)))	# psi(X)*psi(Y)
integ_act = np.cumsum(Z_act**2)[-1]
Z_act = Z_act/np.sqrt(integ_act)									# Normalizing the actual solution

minmax_err = 1		
minmean_err = 1
max_err_list = []
mean_err_list = []

Z = X*Y*0

#############################################################################################################
# Boundary condition and initial condition are characterized in Z											#
#############################################################################################################
for i in range(rows):
	for j in range(cols):
		if ((i+1)%(rows/(2*ny))==0 or (i)%(rows/(2*ny))==0) and ((j+1)%(cols/(2*nx))==0 or (j)%(cols/(2*nx))==0) and i!=0 and j!=0 and i!=rows-1 and j!=cols-1:
			if ((((i+1)/(rows/(2*ny)))%4==1 or ((i)/(rows/(2*ny)))%4==1) and (((j+1)/(cols/(2*nx)))%4==1 or ((j)/(cols/(2*nx)))%4==1)) or ((((i+1)/(rows/(2*ny)))%4==3 or ((i)/(rows/(2*ny)))%4==3) and (((j+1)/(cols/(2*nx)))%4==3 or ((j)/(cols/(2*nx)))%4==3)):
				Z[i,j] = 1
			elif ((((i+1)/(rows/(2*ny)))%4==1 or ((i)/(rows/(2*ny)))%4==1) and (((j+1)/(cols/(2*nx)))%4==3 or ((j)/(cols/(2*nx)))%4==3)) or ((((i+1)/(rows/(2*ny)))%4==3 or ((i)/(rows/(2*ny)))%4==3) and (((j+1)/(cols/(2*nx)))%4==1 or ((j)/(cols/(2*nx)))%4==1)):
				Z[i,j] = -1

#############################################################################################################
# Iteration begins																						#
#############################################################################################################
for expr in range(times):

	#	The following lines solve the PDE (partial differential equation) using finite difference method
	Z[1:-1,1:-1] = ( (Z[2:,1:-1]/(h*h)) + (Z[1:-1,2:]/(k*k)) + (Z[0:-2,1:-1]/(h*h)) + (Z[1:-1,0:-2]/(k*k)) )/( 1 + (2/(h*h)) + (2/(k*k)) )
	integ = np.cumsum(Z**2)[-1]
	Z = Z/np.sqrt(integ)											# Normalizing the calculated solution

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
# Plotting the outputs																					#
#############################################################################################################
if(sim_err_iter==0):

	# Plots the numerical solution
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.title('Phi for 2D rectangular coordinate system. $n_{x}=1,n_{y}=2$')
	ax.set_zlim(-0.101, 0.101)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.ylabel('Y-axis length')
	plt.xlabel('X-axis length')

elif(sim_err_iter==1):

	# Plots the error in numerical solution
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, abs(Z-Z_act), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.title('Error in phi for 2D rectangular coordinate system. $n_{x}=1,n_{y}=2$')
	ax.set_zlim(-0.101, 0.101)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.ylabel('Y-axis length')
	plt.xlabel('X-axis length')

elif(sim_err_iter==2):

	# Plots the maximum error for different iterations
	plt.semilogy(np.arange(0,1000,1),max_err_list)	
	plt.title('Maximum error vs iterations for 2D rectangular case. $n_{x}=1,n_{y}=2$')
	plt.show()

	# Plots the mean error for different iterations
	plt.semilogy(np.arange(0,1000,1),mean_err_list)	
	plt.title('Mean error vs iterations for 2D rectangular case. $n_{x}=1,n_{y}=2$')

	# These lines are required for the both the above plots
	plt.ylabel('Error in phi')
	plt.xlabel('No. of iterations')

plt.show()

