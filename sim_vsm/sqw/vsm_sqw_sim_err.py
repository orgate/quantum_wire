#############################################################################################################
#																											#
#	Title			:	Simulation of wave function using variable separation method for square quantum wir-#
#						e and the error compared to the analytical solution									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_vsm/sqw/vsm_sqw_sim_err.py											#
#	Figure ref.		:	Figures 2.1 and 2.2																	#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

from scipy.integrate import odeint
from scipy.special import gamma, airy
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

sim_err = 0 														# "0" plots the wavefunction and "1" plots the error in the simulation of
																	# the wavefunction
nx = 2 																# x-axis quantum number
ny = 1 																# y-axis quantum number

fig = plt.figure()
ax = fig.gca(projection='3d')

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for x-axis											#
#############################################################################################################
y0_0 = 0															# psi is zero at the boundaries
y1_0 = 1															# derivative is maintained at one along the boundaries (doesn't affect the
																	# shape)
y0 = [y0_0, y1_0]
k0=-nx*nx*pi*pi/4													# assumed eigen value

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for y-axis											#
#############################################################################################################
y0_1 = 0															# psi is zero at the boundaries
y1_1 = 1															# derivative is maintained at one along the boundaries (doesn't affect the 
																	# shape)
y1 = [y0_1, y1_1]
k1=-ny*ny*pi*pi/4													# assumed eigen value	

#############################################################################################################
# function description of 1D-SE equation for x-axis															#
#############################################################################################################
def func0(y, t):
	return [k0*y[1],y[0]]

#############################################################################################################
# function description of 1D-SE equation for y-axis															#
#############################################################################################################
def func1(y, t):
	return [k1*y[1],y[0]]

l=400																# number of length-wise grids
x = arange(0, 4.0, 4.0/l)											# array of grids
size = len(x)
t = x
X = x
Y = x
X, Y = meshgrid(X[:(size/2)], Y[:(size/2)])							# generating mesh grids for the entire area
Z1 = X*Y*0
Z2 = X*Y*0

Z_act = ((sqrt(2/2))*(sin(pi*nx*X/2)))*((sqrt(2/2))*(sin(pi*ny*Y/2)))	# psi(X)*psi(Y) - analytical solution

y_0 = odeint(func0, y0, t)											# solving the 1D-SE ode equation for x-axis
y_1 = odeint(func1, y1, t)											# solving the 1D-SE ode equation for y-axis
rows = len(y_0[(size/4):(3*size/4),1])
cols = len(y_1[(size/4):(3*size/4),1])

for i in range(rows):
	for j in range(cols):
		if (i!=0)&(i<rows-1)&(j!=0)&(j<cols-1):
			Z1[i][j] = y_0[(j+(size/4)/nx),1]*y_1[(i+(size/4)/ny),1]
			Z2[i][j] = y_0[(j+((size/4)+1)/nx),1]*y_1[(i+((size/4)+1)/ny),1]

Z = (Z1+Z2)/2														# calculated wave function - mean of two wave functions that are shifted
Z_err = (Z-Z_act)/abs(Z_act).mean() 								# error in the calculated wave function when compared to the analytical one

# Just removing a strip of data at two edges of Z_err that is inconsistent
for i in range(rows):
	for j in range(1):
		Z_err[i][cols-j-1] = 0

for i in range(1):
	for j in range(cols):
		Z_err[rows-i-1][j] = 0

#############################################################################################################
# Plotting the wave function or the error when compared to the actual analytical solution 					#
#############################################################################################################
if(sim_err==0):
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_zlim(-1, 1)
	plt.title('Solution of schrodinger equation - Magnitude of phi on z-axis')
elif(sim_err==1):
	surf = ax.plot_surface(X, Y, Z_err, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_zlim(-0.1, 0.1)
	plt.title('Error in the solution of schrodinger equation')

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('y-axis length')
plt.ylabel('x-axis length')

plt.show()
