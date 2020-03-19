#############################################################################################################
#																											#
#	Title			:	Simulation of wave function using variable separation method for circular quantum w-#
#						ire and error compared to the analytical solution 									#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/sim_vsm/cqw/vsm_cqw_sim_err.py											#
#	Figure ref.		:	Figures 2.4, 2.5, 2.6 and 2.7														#
#	Date			:	19th May, 2015																		#
#																											#
#############################################################################################################

from scipy.integrate import odeint
from scipy.special import gamma, airy, jn, jn_zeros
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy.linalg import inv

#############################################################################################################
# function description of 1D-SE equation for radial axis													#
#############################################################################################################
def funcR(y, t):
	return [y[1],( ( ((l*l)/(t*t)-(k*k)) )*y[0] - (y[1]/t) )]

#############################################################################################################
# function description of 1D-SE equation for angular axis													#
#############################################################################################################
def func0(y, t):
	return [y[1],(-1)*l*l*y[0]]

delr = 0.01 														# del r: radial grid size
del0 = 1.0															# del theta: angular grid size

m=1																	# radial quantum number
l=3																	# angular quantum number

Radius = 1.0
mtemp = m

X = radians(linspace(0, 360, 360.0/del0))							# array of angular grids
Y = arange(delr,Radius+delr,delr)									# array of radial grids
rows = len(X)														# angular grids
cols = len(Y)														# radial grids

M = len(Y)
N = len(X)

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for radial axis										#
#############################################################################################################
if(l==0):
	R0_0 = 1														# psi is one at the centre (doesn't affect the shape)
	R0_1 = 0														# derivative is maintained at zero at the centre
else:
	R0_0 = 0														# psi is zero at the centre
	R0_1 = 1														# derivative is maintained at one at the centre (doesn't affect the shape)

R0 = [R0_0, R0_1]

#############################################################################################################
# initial conditions for solving ode equation of 1D-SE for angular axis										#
#############################################################################################################
theta0_0 = 1
theta0_1 = 0
theta0 = [theta0_0, theta0_1]

k_array = linspace(0.1,20.0,1000)									# array of possible eigen energies
m_temp = 0
t = Y
ind = 0
sign = +1

#############################################################################################################
# this loop calculates the eigen energy by running through different values and checking if the mth zero cr-#
# ossing happens at the boundary																			#
#############################################################################################################
for i in range(len(k_array)):
	k = k_array[i]
	R = odeint(funcR, R0, t)

	# checks if zero has been crossed and if m_temp<m (i.e how many zeros have been crossed)
	if(R[-1][0]/abs(R[-1][0])!=sign and m_temp<m):	
		m_temp+=1
	elif(R[-1][0]/abs(R[-1][0])!=0 and m_temp==m):
		ind = i-2
		break
	sign = R[-1][0]/abs(R[-1][0])

k = k_array[ind]													# this seems to give more accurate k as m goes higher
R = odeint(funcR, R0, t)											# solving the 1D-SE ode equation for radial axis

t = X
theta = odeint(func0, theta0, t)									# solving the 1D-SE ode equation for angular axis

X, Y = meshgrid(X, Y)												# generating mesh grids for the entire circular area

psi = X*Y*0
psi_act = X*Y*0

#############################################################################################################
# calculated solution																						#
#############################################################################################################
for i in range(M):
	for j in range(N):
		psi[i][j] = R[i][0]*theta[j][0]

integ = cumsum(psi**2)[-1]
psi = psi/sqrt(integ)												# normalizing the wavefunction

#############################################################################################################
# actual analytical solution																				#
#############################################################################################################
for i in range(cols):
	for j in range(rows):
		psi_act[i,j] = jn(l,jn_zeros(l,m)[-1]*Y[i,0]/Radius)*cos(l*X[0,j])		# psi(R)*psi(0)
integ_act = cumsum(psi_act**2)[-1]
psi_act = psi_act/sqrt(integ_act)									# normalizing the wavefunction

#############################################################################################################
# Plotting the wave function															 					#
#############################################################################################################
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
surf = ax.contourf(X,Y,psi,100)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.title('$\psi$ for circular quantum wire $m=1,l=3$')
plt.show()

#############################################################################################################
# Plotting the error when compared to the actual analytical solution					 					#
#############################################################################################################
fig1, ax1 = plt.subplots(subplot_kw=dict(projection='polar'))
surf1 = ax1.contourf(X,Y,abs(psi-psi_act)*100/(abs(psi_act).mean()),100)
fig1.colorbar(surf1, shrink=0.5, aspect=10)
plt.title('Error in $\psi$ for circular quantum wire $m=1,l=3$')
plt.show()
