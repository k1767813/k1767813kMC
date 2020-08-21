from numpy import sqrt,dot,cross,arccos,arctan,tan,arcsinh,log

#from numba import njit, prange

#################################################################################

def econst(T):
    if T==150:
        c11=236.97
        c12=135.44
        c44=118.87
    elif T==300:
        c11=230.37
        c12=134.07
        c44=115.87
    elif T==400:
        c11=224.73
        c12=132.91
        c44=113.76
    else:
        print("Valid temperatures are 150K, 300K and 400K only")
        return

    #mu=c44*(c11-c12)/(4/3*c44+1/3*(c11-c12)) #Shear modulus in the <111> zone
    mu=1/5*(c11-c12+3*c44)                   #Voigt average shear modulus
    K  =1/3*(c11+2*c12)
    nu=(3*K-2*mu)/(6*K+2*mu)

    return mu,nu
    
################################################################################



################################################################################    

#@njit(parallel=True)
def norm(a):                        #Faster than numpy's linalg.norm
    return sqrt((a*a).sum(axis=0))

################################################################################

#@njit(parallel=True)
def schmidt(e2,e1):                 #Schmidt orthonormalise e2 vs e1
    e3=cross(e1,e2)
    l3=norm(e3)
    if l3==0:
        e2a=e2
        e2a[0]+=1
        e3=cross(e1,e2a)
        l3=norm(e3)
        if l3==0:
            e2a=e2
            e2a[1]+=1
            e3=cross(e1,e2a)
            l3=norm(e3)
    e3=e3/l3
    e2a=cross(e3,e1)
    e2a=e2a/norm(e2a)
    return e2a

################################################################################

#@njit(parallel=True)
def Rf(x,y,costheta, a):

#HL p.151 Eq.(6-25) with z=a
#For parallel segments a**2=eta**2+a**2
#Otherwise a**2=z**2+a**2

    return sqrt(x*x+y*y-2*x*y*costheta+a**2)

################################################################################

#@njit(parallel=True)
def Rp(x,y,eta, a):

#HL p.154 Eq.(6.47)

    return sqrt((x-y)**2+eta**2+a**2)

################################################################################

#@njit(parallel=True)
def Ja(x,y,costheta,a):

#H-L p.152 Eq.(6.37)

    if a==0:
        return 0

    R=Rf(x,y,costheta,a)
    u=x-y+R
    v=y-x+R
    w=x+y+R

    theta=arccos(costheta)
    ta2=tan(theta/2)

    if abs(ta2)<1e-10:              #Parallel segments
        return -sqrt((x-y)**2+a**2)/a**2
    else:
        return ( ta2/(2*a)*(arctan(u/a/ta2)
                 +arctan(v/a/ta2))
                 -1/a/ta2*arctan(w/a*ta2) )

################################################################################

#@njit(parallel=True)
def Ia(x,y,costheta, a):

#HL p.154 Eq.(6-48) for parallel segments, p.152 Eq.(6-36) otherwise
#For parallel segments a**2=eta**2+a**2
#Otherwise a**2=z**2+a**2

    if a==0:
        return 0
        
    R=Rf(x,y,costheta,a)
    u=x-y+R
    v=y-x+R
    w=x+y+R
    s=y*costheta-x+R
    t=x*costheta-y+R

    theta=arccos(costheta)
    ta2=tan(theta/2)

    if abs(ta2)<1e-10:
                 #Parallel segments                 
        return -(x-y)*arcsinh((x-y)/a)+sqrt((x-y)**2+a**2)   
    else:
        return ( x/2*log((a**2+v**2/ta2**2)/s/t)
            +y/2*log((a**2+u**2/ta2**2)/s/t)
            -a**2*Ja(x,y,costheta,a) )

#if Iab includes Jab automatically
#i=i+(a.^2/2)*Ja(x,y,costheta, a);

################################################################################

#@njit(parallel=True)
def s(x,y,costheta,a):
    return y*costheta-x+Rf(x,y,costheta,a)
def t(x,y,costheta,a):
    return x*costheta-y+Rf(x,y,costheta,a)

#HL p.152 Eq.(6-35)

################################################################################

#@njit(parallel=True)
def Tfa(b1, b2, xi1, xi2, e3, costheta, x, y, z, a):
    
#
#          ^ y axis
#         /
#        -
#    y  /
#      / theta
#       ---------------|----------------> x axis
#           x
#
#       x>0, y>0   HL p152, Eq.(6-34)

    sintheta=dot(cross(xi1,xi2),e3)

    bt1=dot(cross(b1,xi1),xi2)
    bt2=dot(cross(b2,xi2),xi1)
    bt3=dot(cross(b1,xi1),e3 )
    bt4=dot(cross(b2,xi2),e3 )

    costheta2 = costheta*costheta
    sintheta2 = 1-costheta2
    sintheta4 = sintheta2*sintheta2

    ap=sqrt(z**2+a**2)

    L1=-Rf(x,y,costheta,ap)*costheta-y*sintheta2*log(s(x,y,costheta,ap))
    L2=-Rf(x,y,costheta,ap)*costheta-x*sintheta2*log(t(x,y,costheta,ap))

    return ( -bt1*bt2*costheta/sintheta4*(L1+L2)
              -bt1*bt2*(1+costheta2)/sintheta4*Rf(x,y,costheta,ap)
              +(bt3*bt2-bt1*bt4*costheta)*z/sintheta2*(-log(t(x,y,costheta,ap)))
              +(bt3*bt2*costheta-bt1*bt4)*z/sintheta2*(-log(s(x,y,costheta,ap)))
              +bt3*bt4*(Ia(x,y,costheta,ap)-(z**2)*Ja(x,y,costheta,ap)) )



