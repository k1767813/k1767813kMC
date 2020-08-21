from numpy import zeros,array,sqrt,dtype

from segsenginteract import segsenginteract


def loopenginteract(mu,nu,b,a,h,rc,loopselfeng,looppos,yz,coordh,coordv,etable):

    
#For computing the energy difference between dislocation line configurations that differ by a single segment.
#This difference is equal to the interaction energy of an incremental loop (including its self-energy)
#with the dislocation's line segments in its original configuration. The loop is placed where the configurations differ.
#Returns the energy differences for when the loop is placed above and below a line segment on each type of {110} plane. mu is the elastic shear
#modulus, nu is Poisson's ratio, b is the burgers vector (bx,by,bz) (3,), loopselfeng is the self-energy of the loop,
#looppos gives the start coordinate(s) of the line segment where the loop is placed (i,j,k), yz is an array containing the y-
#and z-coordinates of each horizontal segment, rc is the dislocation radius parameter, 'a' is the lattice constant along 
#the dislocation line, h is the height of a vertical segment, sigma is the applied stress (scalar for now sigma(xz))
#To be consistent with lineelasticenergy.py and the minimum image convention, incremental loop vertical segments at x=0 are shifted to x=N


    N=yz.shape[0]

    Eabovey=dtype("float128")
    Ebelowy=dtype("float128")
    Eaboveyplusz=dtype("float128")
    Ebelowyminusz=dtype("float128")
    Ebelowyplusz=dtype("float128")
    Eaboveyminusz=dtype("float128")


#x-direction <111>, y-direction <112>, z-direction <110>
#Shear stresses for a <111> screw can be sigma(xz) on {110} planes and sigma(xy) on {112} planes


#Calculate the interaction energy (including self-energy) of the loop with all line segments:


    def loopabovey(i,j,k):  #An incremental loop on the (x,y) plane with its bottom left-hand corner at (i,j,k)
        if i==0:
            return array([[N,j,k],[N,j+1,k],[i,j+1,k],[i+1,j+1,k], \
                          [i+1,j+1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j+1,k],[i,j+1,k],[i+1,j+1,k], \
                          [i+1,j+1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopbelowy(i,j,k):  #An incremental loop on the same plane with its top left-hand corner at (i,j,k)
        if i==0:
            return array([[N,j,k],[N,j-1,k],[i,j-1,k],[i+1,j-1,k], \
                          [i+1,j-1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j-1,k],[i,j-1,k],[i+1,j-1,k], \
                          [i+1,j-1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopaboveyplusz(i,j,k): #An incremental loop on an intersecting {110} plane
        if i==0:
            return array([[N,j,k],[N,j+1/2,k+sqrt(3)/2],[i,j+1/2,k+sqrt(3)/2],[i+1,j+1/2,k+sqrt(3)/2], \
                          [i+1,j+1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j+1/2,k+sqrt(3)/2],[i,j+1/2,k+sqrt(3)/2],[i+1,j+1/2,k+sqrt(3)/2], \
                          [i+1,j+1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopbelowyminusz(i,j,k):    #An incremental loop below on the same {110} plane
        if i==0:
            return array([[N,j,k],[N,j-1/2,k-sqrt(3)/2],[i,j-1/2,k-sqrt(3)/2],[i+1,j-1/2,k-sqrt(3)/2], \
                          [i+1,j-1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j-1/2,k-sqrt(3)/2],[i,j-1/2,k-sqrt(3)/2],[i+1,j-1/2,k-sqrt(3)/2], \
                          [i+1,j-1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopaboveyminusz(i,j,k):     #An incremental loop below on another {110} plane
        if i==0:
            return array([[N,j,k],[N,j+1/2,k-sqrt(3)/2],[i,j+1/2,k-sqrt(3)/2],[i+1,j+1/2,k-sqrt(3)/2], \
                          [i+1,j+1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j+1/2,k-sqrt(3)/2],[i,j+1/2,k-sqrt(3)/2],[i+1,j+1/2,k-sqrt(3)/2], \
                          [i+1,j+1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        
    def loopbelowyplusz(i,j,k):    #An incremental loop above on the same {110} plane
        if i==0:
            return array([[N,j,k],[N,j-1/2,k+sqrt(3)/2],[i,j-1/2,k+sqrt(3)/2],[i+1,j-1/2,k+sqrt(3)/2], \
                          [i+1,j-1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        else:
            return array([[i,j,k],[i,j-1/2,k+sqrt(3)/2],[i,j-1/2,k+sqrt(3)/2],[i+1,j-1/2,k+sqrt(3)/2], \
                          [i+1,j-1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    if len(looppos)==3:

        Ediffs=zeros([1,6],dtype="float128")
        
        Eabovey=segsenginteract(mu,nu,b,loopabovey(*looppos)[[2,3],:],loopabovey(*looppos)[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                -segsenginteract(mu,nu,b,loopabovey(*looppos)[[7,6],:],loopabovey(*looppos)[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

        Ebelowy=segsenginteract(mu,nu,b,loopbelowy(*looppos)[[2,3],:],loopbelowy(*looppos)[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                -segsenginteract(mu,nu,b,loopbelowy(*looppos)[[7,6],:],loopbelowy(*looppos)[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)
    
        Eaboveyplusz=segsenginteract(mu,nu,b,loopaboveyplusz(*looppos)[[2,3],:],loopaboveyplusz(*looppos)[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                 -segsenginteract(mu,nu,b,loopaboveyplusz(*looppos)[[7,6],:],loopaboveyplusz(*looppos)[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

        Ebelowyminusz=segsenginteract(mu,nu,b,loopbelowyminusz(*looppos)[[2,3],:],loopbelowyminusz(*looppos)[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                 -segsenginteract(mu,nu,b,loopbelowyminusz(*looppos)[[7,6],:],loopbelowyminusz(*looppos)[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)

        Eaboveyminusz=segsenginteract(mu,nu,b,loopaboveyminusz(*looppos)[[2,3],:],loopaboveyminusz(*looppos)[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                 -segsenginteract(mu,nu,b,loopaboveyminusz(*looppos)[[7,6],:],loopaboveyminusz(*looppos)[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

        Ebelowyplusz=segsenginteract(mu,nu,b,loopbelowyplusz(*looppos)[[2,3],:],loopbelowyplusz(*looppos)[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                 -segsenginteract(mu,nu,b,loopbelowyplusz(*looppos)[[7,6],:],loopbelowyplusz(*looppos)[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)

        Ediffs[0,0]=Eabovey+loopselfeng[0]
        Ediffs[0,1]=Ebelowy+loopselfeng[0]
        Ediffs[0,2]=Eaboveyplusz+loopselfeng[1]
        Ediffs[0,3]=Ebelowyminusz+loopselfeng[1]
        Ediffs[0,4]=Eaboveyminusz+loopselfeng[2]
        Ediffs[0,5]=Ebelowyplusz+loopselfeng[2]

        return Ediffs
##        return array([Eabovey+loopselfeng[0],Ebelowy+loopselfeng[0],\
##                      Eaboveyplusz+loopselfeng[1],Ebelowyminusz+loopselfeng[1],\
##                      Eaboveyminusz+loopselfeng[2],Ebelowyplusz+loopselfeng[2]],dtype("float128"))    #Return an array of the energy changes on each {110} plane

    else:
        looppos=coordh[::2,:]
        Ediffs=zeros([N,6],dtype("float128"))
        for i in range(N):
            Eabovey=0.0
            Ebelowy=0.0
            Eaboveyplusz=0.0
            Ebelowyminusz=0.0
            Ebelowyplusz=0.0
            Eaboveyminusz=0.0
    
            Eabovey+=segsenginteract(mu,nu,b,loopabovey(*looppos[i,:])[[2,3],:],loopabovey(*looppos[i,:])[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                    -segsenginteract(mu,nu,b,loopabovey(*looppos[i,:])[[7,6],:],loopabovey(*looppos[i,:])[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

            Ebelowy+=segsenginteract(mu,nu,b,loopbelowy(*looppos[i,:])[[2,3],:],loopbelowy(*looppos[i,:])[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                     -segsenginteract(mu,nu,b,loopbelowy(*looppos[i,:])[[7,6],:],loopbelowy(*looppos[i,:])[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)
    
            Eaboveyplusz+=segsenginteract(mu,nu,b,loopaboveyplusz(*looppos[i,:])[[2,3],:],loopaboveyplusz(*looppos[i,:])[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                     -segsenginteract(mu,nu,b,loopaboveyplusz(*looppos[i,:])[[7,6],:],loopaboveyplusz(*looppos[i,:])[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

            Ebelowyminusz+=segsenginteract(mu,nu,b,loopbelowyminusz(*looppos[i,:])[[2,3],:],loopbelowyminusz(*looppos[i,:])[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                     -segsenginteract(mu,nu,b,loopbelowyminusz(*looppos[i,:])[[7,6],:],loopbelowyminusz(*looppos[i,:])[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)

            Eaboveyminusz+=segsenginteract(mu,nu,b,loopaboveyminusz(*looppos[i,:])[[2,3],:],loopaboveyminusz(*looppos[i,:])[[0,1],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                     -segsenginteract(mu,nu,b,loopaboveyminusz(*looppos[i,:])[[7,6],:],loopaboveyminusz(*looppos[i,:])[[5,4],:],coordh,coordv[:,:3],a,h,rc,N,etable)

            Ebelowyplusz+=segsenginteract(mu,nu,b,loopbelowyplusz(*looppos[i,:])[[2,3],:],loopbelowyplusz(*looppos[i,:])[[4,5],:],coordh,coordv[:,:3],a,h,rc,N,etable)\
                     -segsenginteract(mu,nu,b,loopbelowyplusz(*looppos[i,:])[[7,6],:],loopbelowyplusz(*looppos[i,:])[[1,0],:],coordh,coordv[:,:3],a,h,rc,N,etable)
    


            Ediffs[i,0]=Eabovey+loopselfeng[0]
            Ediffs[i,1]=Ebelowy+loopselfeng[0]
            Ediffs[i,2]=Eaboveyplusz+loopselfeng[1]
            Ediffs[i,3]=Ebelowyminusz+loopselfeng[1]
            Ediffs[i,4]=Eaboveyminusz+loopselfeng[2]
            Ediffs[i,5]=Ebelowyplusz+loopselfeng[2]
    

        return Ediffs    #Return an array of the energy changes on each {110} plane
          
