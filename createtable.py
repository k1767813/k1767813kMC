from numpy import arange,array,empty,hstack,reshape,zeros,exp,pi,sqrt,sum,dtype,load,save
                    
from segsenginteract import segsenginteract


def createlltable(mu,nu,b,a,h,rc,N,Nx,Ny,Nz,etablesegs):

#For precomputing loop-loop interaction energies that update the energy differences between dislocation line configurations that differ by a single
#segment. These are stored in a .npy file to be used by 'looploopenginteractall'. Mu is the shear modulus, nu is Poisson's ratio, b is the burgers vector,
#a is the dislocation repeat distance, h is the kink height, rc is the dislocation radius parameter, N is the number of horizontal segments of the dislocation.
#Nx, Ny and Nz are the table dimensions.


    def loopabovey(i,j,k):  #An incremental loop on the (x,y) plane with its bottom left-hand corner at (i,j,k)
        return array([[i,j,k],[i,j+1,k],[i,j+1,k],[i+1,j+1,k], \
                      [i+1,j+1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopaboveyplusz(i,j,k): #An incremental loop on an intersecting {110} plane
        return array([[i,j,k],[i,j+1/2,k+sqrt(3)/2],[i,j+1/2,k+sqrt(3)/2],[i+1,j+1/2,k+sqrt(3)/2], \
                     [i+1,j+1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")


    def loopaboveyminusz(i,j,k):     #An incremental loop below on another {110} plane
        return array([[i,j,k],[i,j+1/2,k-sqrt(3)/2],[i,j+1/2,k-sqrt(3)/2],[i+1,j+1/2,k-sqrt(3)/2], \
                      [i+1,j+1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        
##    #Load or create a segment-segment energy interaction table
##    while True:
##        try:
##            etablesegs=load('etablesegsN%s.npy'%N)
##            break
##        except FileNotFoundError:
##            etablesegs=zeros([2*N,24,24,4,4],dtype="float128")
##            break
    
    etable=empty([Nx,Ny,Nz,3,3],dtype="float128")
    Eint00=dtype("float128")
    Eint01=dtype("float128")
    Eint02=dtype("float128")
    Eint10=dtype("float128")
    Eint11=dtype("float128")
    Eint12=dtype("float128")
    Eint20=dtype("float128")
    Eint21=dtype("float128")
    Eint22=dtype("float128")
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dx=i
                dy=j/2
                dz=k*sqrt(3)/2
                dx-=round(dx/Nx)*Nx
                dy-=round(dy/(Ny/2))*Ny/2
                dz-=round(dz/(Nz*sqrt(3)/2))*Nz*sqrt(3)/2

    #given dx,dy,dz the row and column etc of etable would be dx,2*dy,2*(dz/sqrt(3))
    #to convert from a loopbelow position subtract 1 from its j and the energy is the negative of the one in the table for its (j-1)
    
                Eint00=segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint01=segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint02=segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[2,3],:],loopabovey(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopabovey(0.0,0,0)[[7,6],:],loopabovey(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint10=segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint11=segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint12=segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[2,3],:],loopaboveyplusz(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyplusz(0.0,0,0)[[7,6],:],loopaboveyplusz(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint20=segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[2,3],:],loopabovey(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopabovey(dx,dy,dz)[[7,6],:],loopabovey(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint21=segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[2,3],:],loopaboveyplusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopaboveyplusz(dx,dy,dz)[[7,6],:],loopaboveyplusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)

                Eint22=segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[2,3],:],loopaboveyminusz(dx,dy,dz)[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[2,3],:],loopaboveyminusz(0.0,0,0)[[0,1],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopaboveyminusz(0.0,0,0)[[7,6],:],loopaboveyminusz(0.0,0,0)[[5,4],:],loopaboveyminusz(dx,dy,dz)[[7,6],:],loopaboveyminusz(dx,dy,dz)[[5,4],:],a,h,rc,N,etablesegs)


                etable[i,j,k,0,0]=Eint00
                etable[i,j,k,0,1]=Eint01
                etable[i,j,k,0,2]=Eint02
                etable[i,j,k,1,0]=Eint10
                etable[i,j,k,1,1]=Eint11
                etable[i,j,k,1,2]=Eint12
                etable[i,j,k,2,0]=Eint20
                etable[i,j,k,2,1]=Eint21
                etable[i,j,k,2,2]=Eint22
                
    #save('etablesegsN%s.npy'%N,etablesegs)
    return etable

 ####################################################################################################################################################################################################################################   

import numpy as np

def kinknucrate(mu,nu,b,a,h,n0,n,Ekp,sigma,omega0,Wm,T,S):

#For calculating the wide-kink-pair nucleation rate using the average first arrival time method
    
    #### Parameters:
    eV=1.602176634e-19      #1eV in Joules
    kB=1.380649e-23         #Boltzmann's constant (in J/K)  
    kBT=kB*T                #In J
    bx=b[0]                 #Screw component of Burgers vector
    by=b[1]                 #Edge component of Burgers vector
    bz=b[2]                 #Edge component of Burgers vector    
    w=arange(n0,n+1)        #Space of kink pair width

    
    nn0=n-n0
    

    #Kink pair energy (analytic expression from Hirth-Lothe):
    B=mu*h**2/(8*pi*a)*(bx**2*(1+nu)/(1-nu)+(by**2+bz**2)*(1-2*nu)/(1-nu))
    #Ekp=0.69*eV
    Edkp=empty([nn0,],dtype=np.float128)
    Edkp=Ekp-B/w[1:]
    
    #print("Edkp.size")
    #print(Edkp.size)


    #Energy landscape
    E=hstack([array([0],dtype=np.float128),Edkp-w[1:]*sigma*a*bx*h]) #17/06/2020 factor of 1/2 added to last term, then taken away the day after!

    N=E.size
    #print("N is")
    #print(N)

    rwp=hstack([omega0*exp(-((E[1:]-E[0:-1])/2+Wm)/kBT+S),array([0],dtype=np.float128)])  #Forwards rate from state with kink pair width w
    rwm=hstack([array([0],dtype=np.float128),omega0*exp(-((E[:-1]-E[1:])/2+Wm)/kBT+S)])   #Backwards rate from state with kink pair width w
    rwm[-1]=0.0                                                                           #Last backwards rate is zero due to the absorbing boundary at w=n

    #Correction (two kinks in a pair, H(w) function in book p.185)
    rwp[1:]*=2
    rwm[2:]*=2


    tw=zeros([nn0,],dtype=np.float128)
    tw[nn0-1]=1/rwp[nn0-1]
    i=nn0-2
    while i>-1:
        tw[i]=1/rwp[i]+rwm[i+1]/rwp[i]*tw[i+1]
        i-=1

    Tw=sum(tw)

    #print("Using the paper's average first passage time method:",1/Tw)

    return 1/Tw


def createjtable(mu,nu,b,a,h,n0,n,Ekp,omega0,smax,Wm,T,S):

#For precomputing wide kink-pair nucleation rates using the function above


    Sigma=arange(-smax,smax+1e4,1e4)
    Sigma=Sigma.reshape([len(Sigma),1])
    Jtable=hstack([Sigma,zeros([len(Sigma),1],dtype="float128")])
    for i in range(Sigma.shape[0]):
        Jtable[i,1]=kinknucrate(mu,nu,b,a,h,n0,n,Ekp,Sigma[i],omega0,Wm,T,S)

    return Jtable


          
