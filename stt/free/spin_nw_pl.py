#############################################################################################################
#																											#
#	Title			:	Transverse spin current behaviour in planar and nanowire tunnel junction			#
#	Author			:	Alfred Ajay Aureate R																#
#	Roll No			:	EE10B052																			#
#	Project guide	:	Prof. Anil Prabhakar, Dept.of Electrical Engineering, IITM							#
#	Code location	:	/DDP_codes/stt/free/spin_nw_pl.py													#
#	Figure ref.		:	Figure 4.1, 4.2 and 4.3																#
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
import defs as d											# importing custom library
import plots as p											# importing custom library

nwpl = 0													# "0" means 'planar' or "1" means 'nanowire'

l = 0														# azimuthal (angular) quantum number

if(nwpl==0):
	q = np.arange(0.001,1,0.001)+0j							# Eigen energies considered for analysis (planar)
else:
	q = (sp.jn_zeros(l,11)[:-1]/sp.jn_zeros(l,11)[-1])+0j	# Eigen energies considered for analysis (nanowire)


#############################################################################################################
# Reflection and transmission related parameters															#
#############################################################################################################
rsts = d.RsTs(q)											# Returns [Ru,Rd,Tu,Td,RU,RD,TU,TD,Rud,Tud]

p.plotR(q,rsts[4],rsts[5],rsts[8],nwpl)						# Plots spin filtering - reflection probability and current (Fig 4.1.a) and b))
plt.show()
p.plotT(q,rsts[6],rsts[7],rsts[9],nwpl)						# Plots spin filtering - transmission probability and current (Fig 4.1.c) and d))
plt.show()
p.plotTra(q,rsts[8],rsts[9],nwpl)							# Plots spin filtering - transverse spin current (Fig 4.1.e) and f))
plt.show()
p.plotSR(q,rsts[0],rsts[1],nwpl)							# Plots spin rotation (Fig 4.2)
plt.show()
p.plotSP(q,rsts[2],rsts[3])									# Plots spatial spin precession (Fig 4.3)
	
plt.xlabel('$q$')
plt.show()
