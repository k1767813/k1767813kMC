from numpy import zeros,array,sqrt,dtype

from linestress import linestress

def lstresschange(mu,nu,b,a,h,rc,change,yz,coordh,coordv,stable):

    
#For computing the local stress change between dislocation line configurations that differ by a single segment.
#This difference is given  by the stress field of an incremental loop evaluated at the mid-point of each horizontal segment
#The loop is placed where the configurations differ. mu is the elastic shear modulus, nu is Poisson's ratio, b is the burgers vector (bx,by,bz) (3,), loopselfeng is the self-energy of the loop,
#looppos gives the start coordinate(s) of the line segment where the loop is placed (i,j,k), yz is an array containing the y-
#and z-coordinates of each horizontal segment, rc is the dislocation radius parameter, 'a' is the lattice constant along 
#the dislocation line, h is the height of a vertical segment.
#PBC: incremental loop vertical segments at x=0 are shifted to x=N


    N=yz.shape[0]


#x-direction <111>, y-direction <112>, z-direction <110>
#Shear stresses for a <111> screw can be sigma(xz) on {110} planes and sigma(xy) on {112} planes



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

    def loop():
        if change[1]>0:
            if change[2]>0:
                return loopaboveyplusz(change[0],*yz[change[0],:])
            elif change[2]<0:
                return loopaboveyminusz(change[0],*yz[change[0],:])
            else:
                return loopabovey(change[0],*yz[change[0],:])
        
        else:
            if change[2]>0:
                return loopbelowyplusz(change[0],*yz[change[0],:])
            elif change[2]<0:
                return loopbelowyminusz(change[0],*yz[change[0],:])
            else:
                return loopbelowy(change[0],*yz[change[0],:])

    sdiffs=zeros([N,3,3])

    if change[1]>0:
        sdiffs+=linestress(mu,nu,b,coordh,[],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,stable)[0]\
               -linestress(mu,nu,b,coordh,[],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,stable)[0]
    elif change[1]<0:
        sdiffs+=linestress(mu,nu,b,coordh,[],loop()[[2,3],:],loop()[[4,5],:],a,h,rc,N,stable)[0]\
               -linestress(mu,nu,b,coordh,[],loop()[[7,6],:],loop()[[1,0],:],a,h,rc,N,stable)[0]

    return sdiffs
          
