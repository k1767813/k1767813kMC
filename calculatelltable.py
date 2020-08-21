from numpy import(
    
    log,sqrt,arctan,zeros,array,arange,diag,append,copy,
    ceil,vstack,reshape,where,concatenate,dtype,sign,sin,cos,
    delete,insert,abs,hstack,dot,pi,sum,empty,argmin,argmax,
    exp,log,cumsum,random,mean,save,load,round,unique,argsort
)

from numpy_indexed import group_by

import numpy as np

from scipy.interpolate import interp1d

from time import time

import matplotlib
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import ion,figure,subplot,plot,pause,show,text,close
matplotlib.use("TkAgg")


from loopselfenergy import loopselfenergy

from loopenginteract import loopenginteract

from looploopenginteractall import looploopenginteractall

from segments import *

from createtable import *

from linestress import linestress

from lstresschange import lstresschange

from econst import econst

import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


## Parameters:
eV=dtype("float128")
Wm=dtype("float128")
a0=dtype("float128")
a=dtype("float128")
h=dtype("float128")
bx=dtype("float128")
by=dtype("float128")
bz=dtype("float128")
omega0=dtype("float128")
sigmaxz=dtype("float128")
sigmaxy=dtype("float128")
S=dtype("float128")
T=dtype("float128")
kB=dtype("float128")
L=dtype("float128")
rc=dtype("float128")
mu=dtype("float128")
nu=dtype("float128")
Ekp=dtype("float128")
B=dtype("float128")
xmig=dtype("float128")
tmig=dtype("float128")
tc=dtype("float128")

eV=1.602176634e-19      #1eV in Joules
Wm=0                    #Kink migration energy barrier (in eV)
a0=2.87e-10             #Lattice constant Fe (in m)
a=a0/2*sqrt(3)          #Lattice constant along dislocation line 
#h=a0*sqrt(2/3)          #Kink height
h=2.31e-10
bx=a0/2*sqrt(3)         #Screw component of Burgers vector for a horizontal segment
by=0                    #Edge component of Burgers vector for a horizontal segment
bz=0                    #Screw component of Burgers vector for a vertical segment

omega0=2.31e9           #Ivo
#omega0=1.383e12         #Stukowski

##sigmaxz=173.2050808e6            #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
##sigmaxy=-100e6            #Applied stress in Pa (force in the -z-direction leading to cross-slip.

sigmaxz=48.29629131e6            #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
sigmaxy=-12.94095227e6            #Applied stress in Pa (force in the -z-direction leading to cross-slip.

##sigmaxz=35.35533906e6            #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
##sigmaxy=-35.35533906e6            #Applied stress in Pa (force in the -z-direction leading to cross-slip.

##sigmaxz=200e6
##sigmaxy=0

##sigmaxz=0#200e6
##sigmaxy=0


T=300                     #Temperature (in K)
mu=econst(300,111)[0]  #Shear modulus Fe (in Pa) T=300K
nu=econst(300,111)[1]         #Poisson's ratio Fe
rc=0.305293687*bx           #Dislocation radius parameter Fe



kB=1.380649e-23         #Boltzmann's constant (in J/K)  
kBT=kB*T                #In J

N=1000                  #Dislocation line length in unit of a
N0=1000
L=N*a                   #Dislocation line length (in Angstrom)


n0=0                    #Embrionic kink pair width minus 1
n=30                    #Target kink pair width

B=(2.7+0.008*T)*1e-5
xmig=25


Niter=2000           #maximum number of iterations
plotfreq=2000         #frequency of plotting

ysave=zeros([int(Niter/plotfreq),4])     #average dislocation position and number of kinks

rate=zeros([Niter,1],dtype="float128")  #total rate saved after each iteration
dt=zeros([Niter,1],dtype="float128")     #time step
t=zeros([Niter,1],dtype="float128")      #cumulated time


b=array([bx,by,bz])
sigma=array([sigmaxz,sigmaxy])
loopselfeng=loopselfenergy(mu,nu,b,a,h,rc)      #Self and interaction energy of an incremental dislocation loop


print(econst(300,111))




#Obtain energy tables
while True:
        try:
            etablesegs=load('etablesegsN%sT%s.npy'%(N0,T))
            break
        except FileNotFoundError:
            etablesegs=zeros([2*N,24,24,4,4],dtype="float128")
            break

print(etablesegs[etablesegs!=0].size)


while True:
    try:
        etable=load('etable111screwN%sT%s.npy' %(N0,T))
        break
    except FileNotFoundError:
        #etable=empty([0,0,0,3,3])
        print('Calculating loop-loop energy table...')
        etable=createlltable(mu,nu,b,a,h,rc,N,2*N,18,18,etablesegs)
        save('etable111screwN%sT%s.npy' %(N0,T),etable)
        break
save('etablesegsN%sT%s.npy'%(N0,T),etablesegs)
print(etablesegs[etablesegs!=0].size)
print("Done!")
