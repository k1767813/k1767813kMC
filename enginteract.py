from numpy import pi,sqrt,cross,dot,zeros,linalg

from defs import *
##from numba import njit, prange
##
##@njit(parallel=True)


def engparallelb2(MU,NU,b1,b2,x1,x2,y1,y2,eta,a):

#For use in enginteract below
#HL p.154 Eq.(6-45)

    b1x=b1[0]
    b1y=b1[1]
    b1z=b1[2]

    b2x=b2[0]
    b2y=b2[1]
    b2z=b2[2]

    Rab=Rp(x2,y2,eta,a)-Rp(x2,y1,eta,a)-Rp(x1,y2,eta,a)+Rp(x1,y1,eta,a)

    #[b1',b2',x1,x2,y1,y2,eta,a,Rab]
    #b1

    ap=sqrt(eta**2+a**2)
    Iab=Ia(x2,y2,1,ap)-Ia(x2,y1,1,ap)-Ia(x1,y2,1,ap)+Ia(x1,y1,1,ap)
    Jab=Ja(x2,y2,1,ap)-Ja(x2,y1,1,ap)-Ja(x1,y2,1,ap)+Ja(x1,y1,1,ap)



    return MU/4/pi*(b1x*b2x+(b1z*b2z+b1y*b2y)/(1-NU))*Iab \
              + MU/4/pi*(b1x*b2x)*(a**2/2)*Jab \
              - MU/4/pi/(1-NU)*b1z*b2z*eta*eta*Jab



def engnonplanarb2(MU,NU,b1,b2,xi1,xi2,e3,costheta,x1,x2,y1,y2,z,a):

#For use in enginteract below

#
#          ^ y axis
#         /
#        -
#    y  /
#      / theta
#       ---------------|----------------> x axis
#           x
#
#       x>0, y>0    HL p152, Eq.(6-33)

    ap=sqrt(z*z+a*a)

    Iab = Ia(x2,y2,costheta,ap)-Ia(x2,y1,costheta,ap)-Ia(x1,y2,costheta,ap)+Ia(x1,y1,costheta,ap)
    Jab = Ja(x2,y2,costheta,ap)-Ja(x2,y1,costheta,ap)-Ja(x1,y2,costheta,ap)+Ja(x1,y1,costheta,ap)

    Tab = ( Tfa(b1,b2,xi1,xi2,e3,costheta,x2,y2,z,a)
            - Tfa(b1,b2,xi1,xi2,e3,costheta,x2,y1,z,a)
            - Tfa(b1,b2,xi1,xi2,e3,costheta,x1,y2,z,a)
            + Tfa(b1,b2,xi1,xi2,e3,costheta,x1,y1,z,a) )

    return ( MU/4/pi*(-2*dot(cross(b1,b2),cross(xi1,xi2))
            + dot(b1,xi1)*dot(b2,xi2) )*(Iab+a**2/2*Jab)
            + MU/4/pi/(1-NU)*Tab )

#When Iab incorporates Jab
#W = ( MU/4/pi* (-2*dot(cross(b1,b2),cross(xi1,xi2)) + dot(b1,xi1)*dot(b2,xi2) )*(Iab)
#    + MU/4/pi/(1-NU)* Tab )



def enginteract(MU,NU,b1,b2,r1,r2,r3,r4,a):


#Computes interaction energy between two straight dislocation segments
#r1-r2 (Burgers vector b1) and r3-r4 (Burgers vector b2)
#MU is shear modulus, NU is Poisson ratio, a is core spread radius


    r21=r2-r1
    r43=r4-r3
    r31=r3-r1


#Make sure that the segments are represented by column vectors

    #if r21.shape[0]==1:
        #r21=r21.T
    #if r43.shape[0]==1:
        #r43=r43.T
    #if r31.shape[0]==1:
        #r31=r31.T


#Segment line sense unit vectors 

    e1=r21/norm(r21)
    e2=r43/norm(r43)


#Catagorise line segments according to whether they are parallel or not

    e3=cross(e1,e2)
    subzero=1e-10

    if norm(e3)<subzero:
        e2a=schmidt(r31,e1)
        e3=cross(e1,e2a)
        e3=e3/norm(e3)
        eta=(dot(r3-r1,e2a)+dot(r4-r1,e2a))/2
        x1=0
        x2=dot(r2-r1,e1)
        y1=dot(r3-r1,e1)
        y2=dot(r4-r1,e1)
#engparallelb2 doesn't rotate b, it needs to be done here
        b1n=zeros([3,1])
        b2n=zeros([3,1])
        b1n[0],b2n[0]=dot(b1,e1),dot(b2,e1)
        b1n[1],b2n[1]=dot(b1,e2a),dot(b2,e2a)
        b1n[2],b2n[2]=dot(b1,e3),dot(b2,e3)
        return engparallelb2(MU,NU,b1n,b2n,x1,x2,y1,y2,eta,a)
    else:
        costheta=dot(e1,e2)
        e3=e3/norm(e3)
        e2a=cross(e3,e1)
        z=dot(r31,e3) 
        z=-z
        A=zeros([2,2])
        A[0,0],A[0,1]=dot(r21,e1),-dot(r43,e1)
        A[1,0],A[1,1]=dot(r21,e2a),-dot(r43,e2a)
        rhs=zeros([2,1])
        rhs[0],rhs[1]=dot(r31,e1),dot(r31,e2a)
        t=linalg.solve(A,rhs)
        r0=(1-t[0])*r1+t[0]*r2
        x1=dot(r1-r0,e1)
        x2=dot(r2-r0,e1)
        y1=dot(r3-r0,e2)
        y2=dot(r4-r0,e2)
        return engnonplanarb2(MU,NU,b1,b2,e1,e2,e3,costheta,x1,x2,y1,y2,z,a)
        
                 
    

