
from numpy import vstack,sum,sqrt,array,dtype

from enginteract import enginteract

from segsenginteractline import segsenginteract


def elasticenergy(Mu,Nu,b,sigma,rc,yz,coordh,coordv,N,a,h,etable):
#Computes the elastic energy of a dislocation line with N segments given an array yz containing the y- and
#z-coordinates of each horizontal segment. Mu is the elastic shear modulus, Nu is Poisson's ratio, b is the
#burgers vector (bx,by,bz) (3,), 'a' is the lattice constant and rc is the dislocation radius parameter.

#Obtain the coordinates of all segments:
                      
    allcoord=vstack([coordh,coordv[:,:3]])
##    allcoord[:,0]*=a
##    allcoord[:,1:3]*=h                      #All segment coordinates
    vcoord=coordv
##    vcoord[:,0]*=a
##    vcoord[:,1:3]*=h
    #print(allcoord.dtype,vcoord.dtype)
    Nv=coordv.shape[0]//2                   #Number of v.segs

#Calculate the interaction and self energy:

    Eselfh=dtype("float128")
    Eselfv=dtype("float128")
    Eint=dtype("float128")
    E=dtype("float128")
    Eselfh=0.0
    Eselfv=0.0
    Eint=0.0
    E=0.0

    
                                                                #Self-energy of h.segs, which are all the same
    Eselfh=N*enginteract(Mu,Nu,b,b,array([0,0,0]),\
                           array([a,0,0],dtype("float128")),array([0,0,0]),\
                           array([a,0,0],dtype("float128")),rc)/2
                                                                #Self-energy of v.segs

    #Eselfv+=Nv*enginteract(Mu,Nu,b,b,array([0,0,0]),\
                           #array([0,h,0]),array([0,0,0]),\
                           #array([0,h,0]),rc)/2                #If the v.segs are all of unit length
##
##    for i in range(Nv):
##        Eselfv+=enginteract(Mu,Nu,b,b,vcoord[2*i,:3],\
##                            vcoord[2*i+1,:3],vcoord[2*i,:3],\
##                            vcoord[2*i+1,:3],rc)/2              #If the v.segs are different sizes


    for i in range(Nv):
        Eselfv+=enginteract(Mu,Nu,b,b,array([a*vcoord[2*i,0],h*vcoord[2*i,1],h*vcoord[2*i,2]],dtype("float128")),\
                            array([a*vcoord[2*i+1,0],h*vcoord[2*i+1,1],h*vcoord[2*i+1,2]],dtype("float128")),\
                            array([a*vcoord[2*i,0],h*vcoord[2*i,1],h*vcoord[2*i,2]],dtype("float128")),\
                            array([a*vcoord[2*i+1,0],h*vcoord[2*i+1,1],h*vcoord[2*i+1,2]],dtype("float128")),rc)/2              #If the v.segs are different sizes



    #Interaction energy

    Eint=segsenginteract(Mu,Nu,b,coordh,coordv,coordh,coordv,a,h,rc,N,etable)
    
##    for i in range(allcoord.shape[0]//2):
##        for j in range(i+1,allcoord.shape[0]//2):
##            xi=allcoord[2*i,0]*a
##            xj=allcoord[2*j,0]*a
##            pbcshift = array([round((xi-xj)/N/a)*N*a, 0,0],dtype("float128"))     #Minimum image convention
##            Eint+=enginteract(Mu,Nu,b,b,array([a*allcoord[2*i,0],h*allcoord[2*i,1],h*allcoord[2*i,2]],dtype("float128")),\
##                           array([a*allcoord[2*i+1,0],h*allcoord[2*i+1,1],h*allcoord[2*i+1,2]],dtype("float128")),\
##                           array([a*allcoord[2*j,0],h*allcoord[2*j,1],h*allcoord[2*j,2]],dtype("float128"))+pbcshift,\
##                           array([a*allcoord[2*j+1,0],h*allcoord[2*j+1,1],h*allcoord[2*j+1,2]],dtype("float128"))+pbcshift,rc)
##
    

    E=Eint+Eselfh+Eselfv


#Area of glide planes enclosed:

    Axy=sum(yz[:,0])*a*h
    Axz=sum(yz[:,1])*a*h
    
#Account for the work done by the applied stress (sigmaxz and sigmaxy)

    E-=sigma[0]*b[0]*Axy-sigma[1]*b[0]*Axz

    return E
          
