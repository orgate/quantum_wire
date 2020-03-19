#############################################################################################################
#																											#
#	Title			:	Error in the simulation of wave function using variable separation method for squar-#
#						e quantum wire as the number of blocks is varied									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_vsm/sqw/vsm_sqw_err_blocks.py										#
#	Figure ref.		:	Figures 2.3																			#
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

error = 1 															# "0" stands for max error and "1" stands for mean error
blocks = 400														# number of length-wise grids

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for x-axis											#
#############################################################################################################
y0_0 = 0															# psi is zero at the boundaries
y1_0 = 1															# derivative is maintained at one along the boundaries (doesn't affect the
																	# shape)
y0 = [y0_0, y1_0]

def k0gen(n_x):
	return (-n_x*nx*pi*pi/4)

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for y-axis											#
#############################################################################################################
y0_1 = 0															# psi is zero at the boundaries
y1_1 = 1															# derivative is maintained at one along the boundaries (doesn't affect the
																	# shape)
y1 = [y0_1, y1_1]

def k1gen(n_y):
	return (-n_y*ny*pi*pi/4)

#############################################################################################################
# function description of 1D-SE equation for x-axis															#
#############################################################################################################
def func0(y, t):
	return [k0gen(nx)*y[1],y[0]]

#############################################################################################################
# function description of 1D-SE equation for y-axis															#
#############################################################################################################
def func1(y, t):
	return [k1gen(ny)*y[1],y[0]]

#############################################################################################################
# function to calculate maximum error and mean error														#
#############################################################################################################
def Z_err(del_size):
	x = arange(0, 4.0, del_size)									# array of grids
	size = len(x)
	t = x
	X = x
	Y = x
	X1, Y1 = meshgrid(X[:(size/2)], Y[:(size/2)])					# generating mesh grids for the entire area - for actual solution
	X, Y = meshgrid(X[:(size/2)], Y[:(size/2)])						# generating mesh grids for the entire area - for calculated solution
	Z1 = X*Y*0
	Z2 = X*Y*0

	Z_act = ((sqrt(2/2))*(sin(pi*ny*X1/2)))*((sqrt(2/2))*(sin(pi*nx*Y1/2)))	#psi(X1)*psi(Y1) - analytical solution

	y_0 = odeint(func0, y0, t)										# solving the 1D-SE ode equation for x-axis
	y_1 = odeint(func1, y1, t)										# solving the 1D-SE ode equation for y-axis
	rows = len(y_0[:(size/2),1])
	cols = len(y_1[:(size/2),1])

	for i in range(rows):
		for j in range(cols):
			if (i!=0)&(i<rows-1)&(j!=0)&(j<cols-1):
				Z1[i][j] = y_0[(i+(size/4)/nx),1]*y_1[(j+(size/4)/ny),1]
				Z2[i][j] = y_0[(i+((size/4)+1)/nx),1]*y_1[(j+((size/4)+1)/ny),1]

	Z = (Z1+Z2)/2													# calculated wave function - mean of two wave functions that are shifted
	Z_err = Z*0														# error in the calculated wave function when compared to the analytical one
	for i in range(rows):
		for j in range(cols):
			Z_err[i][j] = (Z[i][j]-Z_act[i][j])*100 				# matrix containing error percentage

	# Just removing a strip of data at two edges of Z_err that is inconsistent
	for i in range(rows):
		for j in range(1):
			Z_err[i][cols-j-1] = 0

	for i in range(1):
		for j in range(cols):
			Z_err[rows-i-1][j] = 0

	max_err = 0
	mean_err = 0
	
	# calculating mean error and maximum error
	for i in range(rows):
		for j in range(cols):
			mean_err = mean_err + abs(Z_err[i][j])
			if max_err < abs(Z_err[i][j]):
				max_err = abs(Z_err[i][j])
	mean_err = mean_err/(rows*cols)
	return [max_err, mean_err]

l = 2
Err = [[],[],[]]

#############################################################################################################
# calculating mean or maximum error in the simulation for different number of length-wise grids				#
#############################################################################################################
while(l<=blocks):
	nx=1
	ny=1
	Err_temp = Z_err(4.0/l)
	if(error==0):
		Err[0].append(Err_temp[0])
	else:
		Err[0].append(Err_temp[1])
	nx=1
	ny=2
	Err_temp = Z_err(4.0/l)
	if(error==0):
		Err[1].append(Err_temp[0])
	else:
		Err[1].append(Err_temp[1])
	nx=2
	ny=2
	Err_temp = Z_err(4.0/l)
	if(error==0):
		Err[2].append(Err_temp[0])
	else:
		Err[2].append(Err_temp[1])
	l+=1

#############################################################################################################
# Plotting mean or maximum error in the simulation for different number of length-wise grids				#
#############################################################################################################
X_axis = arange(1,blocks,1)
fig, axes = plt.subplots(nrows=1, ncols=1)
axes.plot(X_axis,Err[0], label="nx=1; ny=1")
axes.plot(X_axis,Err[1], label="nx=1; ny=2")
axes.plot(X_axis,Err[2], label="nx=2; ny=2")
axes.legend(loc=1)
if (error==0):
	axes.set_title('"Maximum error" percentage as the number of blocks are increased')
else:
	axes.set_title('"Mean error" percentage as the number of blocks are increased')
axes.set_xlabel('Number of blocks')
axes.set_ylabel('Error percentage')
fig.tight_layout()
plt.show()
