from numpy import array,dot,cross,sqrt,pi,tensordot,reshape,zeros

from defs import norm,schmidt

###################################################################################################################################################################################################
#Ref W. Cai et al. / J. Mech. Phys. Solids 54 (2006) 561–587 "A non-singular continuum theory of dislocations" (p583-584)

#Slow! Takes fives times longer to calculate compared with the below

def T(mu,nu,b,R,t,a):      

    R=R.reshape([3,])
    t=t.reshape([3,])
    b=b.reshape([3,])
    I=array([[1,0,0],[0,1,0],[0,0,1]])

    Ra=sqrt(dot(R,R)+a**2)
    T0=mu/(4*pi*(1-nu))
    A1=-dot(R,t)*( 3*Ra**2-(dot(R,t))**2 )/( (Ra**2-(dot(R,t))**2)**2*Ra**3 )
    A2=1/Ra**3 - dot(R,t)*A1
    A6=-dot(R,t)/( (Ra**2-(dot(R,t))**2)*Ra )
    A3=-dot(R,t)/Ra**3+A6+(dot(R,t))**2*A1
    A4=A6+a**2*A1
    A5=(nu-1)*A6-a**2*(1-nu)/2*A1
    A7=nu/Ra-dot(R,t)*nu*A6-a**2*(1-nu)/2*A2

    #need shape (3, ) for tensordot

    TT=dot(cross(R,b),t)*( A1*tensordot(R,R,axes=0)+A2*(tensordot(t,R,axes=0)+tensordot(R,t,axes=0))+A3*tensordot(t,t,axes=0) +A4*I )\
       +A5*( tensordot(cross(R,b),t,axes=0)+tensordot(t,cross(R,b),axes=0) )+A6*( tensordot(cross(t,b),R,axes=0)+tensordot(R,cross(t,b),axes=0) )\
       +A7*( tensordot(cross(t,b),t,axes=0)+tensordot(t,cross(t,b),axes=0) )
    TT*=T0

    return TT

def stress(mu,nu,b,x,x1,x2,a):

    x=x.reshape([3,])
    x1=x1.reshape([3,])
    x2=x2.reshape([3,])
    t=(x2-x1)/norm((x2-x1))

    sigma=T(mu,nu,b,x-x2,t,a)-T(mu,nu,b,x-x1,t,a)

    return sigma











def T1(mu,nu,b,R,t,a):
    def tensordot1(a,b):
        c=array([[a[0]*b[0],a[0]*b[1],a[0]*b[2]],[a[1]*b[0],a[1]*b[1],a[1]*b[2]],[a[2]*b[0],a[2]*b[1],a[2]*b[2]]])
        return c

    def dot1(a,b):
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    def cross1(a,b):
        return array([a[1]*b[2]-a[2]*b[1],-(a[0]*b[2]-a[2]*b[0]),a[0]*b[1]-a[1]*b[0]])
        
        

    R=R.reshape([3,])
    t=t.reshape([3,])
    b=b.reshape([3,])
    I=array([[1,0,0],[0,1,0],[0,0,1]])

    Ra=sqrt(dot1(R,R)+a**2)
    T0=mu/(4*pi*(1-nu))
    A1=-dot1(R,t)*( 3*Ra**2-(dot1(R,t))**2 )/( (Ra**2-(dot1(R,t))**2)**2*Ra**3 )
    A2=1/Ra**3 - dot1(R,t)*A1
    A6=-dot1(R,t)/( (Ra**2-(dot1(R,t))**2)*Ra )
    A3=-dot1(R,t)/Ra**3+A6+(dot1(R,t))**2*A1
    A4=A6+a**2*A1
    A5=(nu-1)*A6-a**2*(1-nu)/2*A1
    A7=nu/Ra-dot1(R,t)*nu*A6-a**2*(1-nu)/2*A2

    #need shape (3, ) for tensordot

    TT=dot1(cross1(R,b),t)*( A1*tensordot1(R,R)+A2*(tensordot1(t,R)+tensordot1(R,t))+A3*tensordot1(t,t) +A4*I )\
       +A5*( tensordot1(cross1(R,b),t)+tensordot1(t,cross1(R,b)) )+A6*( tensordot1(cross1(t,b),R)+tensordot1(R,cross1(t,b)) )\
       +A7*( tensordot1(cross1(t,b),t)+tensordot1(t,cross1(t,b)) )
    TT*=T0

    return TT

def stress1(mu,nu,b,x,x1,x2,a):

    x=x.reshape([3,])
    x1=x1.reshape([3,])
    x2=x2.reshape([3,])
    t=(x2-x1)/norm((x2-x1))

    sigma=T1(mu,nu,b,x-x2,t,a)-T1(mu,nu,b,x-x1,t,a)

    return sigma



















###################################################################################################################################################################################################

#Cai et al online non-singular equations:

def stresszcai(mu,nu,b,r,z1,z2,a):

    bx=b[0]
    by=b[1]
    bz=b[2]

    x=r[0]
    z=r[2]


    l=z1-z
    lp=z2-z
    dx=x
    
    ra=sqrt(dx*dx+l*l+a*a)
    rap=sqrt(dx*dx+lp*lp+a*a)

    dx2=dx*dx
    a2=a*a

    rap2=rap*rap
    rap3=rap2*rap
    ra2=ra*ra
    ra3=ra2*ra

    sunit=mu*(1/4/pi/(1-nu))

    s=zeros([1,6])
    if l>0 and lp>0:
      form=1
    elif l<0 and lp<0:
      form=2
    else:
      form=3

    if form==1:
        invral=1/ra/(ra+l)
        invrapl=1/rap/(rap+lp)
        s[0,0] = by*dx*( invrapl*(1-(dx2+a2)/rap2-(dx2+a2)*invrapl)\
                 - invral *(1-(dx2+a2)/ra2 -(dx2+a2)*invral) )
        s[0,1] = -bx*dx*( invrapl - invral )
        s[0,2] = by*( (-nu/rap+dx2/rap3+(1-nu)*(a2/2)/rap3)\
                 -(-nu/ra +dx2/ra3 +(1-nu)*(a2/2)/ra3 ) )

        s[0,3] = -by*dx*( invrapl*(1+a2/rap2+a2*invrapl)\
                 - invral *(1+a2/ra2 +a2*invral ) )
        s[0,4] = ( bx*(nu/rap-(1-nu)*(a2/2)/rap3) - bz*dx*(1-nu)*invrapl*(1+(a2/2)/rap2+(a2/2)*invrapl) )\
                -( bx*(nu/ra -(1-nu)*(a2/2)/ra3 ) - bz*dx*(1-nu)*invral *(1+(a2/2)/ra2 +(a2/2)*invral ) )
        s[0,5] = by*dx*( (-2*nu*invrapl*(1+(a2/2)/rap2+(a2/2)*invrapl)-lp/rap3)\
                - (-2*nu*invral *(1+(a2/2)/ra2 +(a2/2)*invral )-l /ra3 ) )

        
    elif form==2:
        invral=1/ra/(ra-l)
        invrapl=1/rap/(rap-lp)
        s[0,0] = -by*dx*( invrapl*(1-(dx2+a2)/rap2-(dx2+a2)*invrapl)\
                 - invral *(1-(dx2+a2)/ra2 -(dx2+a2)*invral) )
        s[0,1] = +bx*dx*( invrapl - invral )
        s[0,2] = by*( (-nu/rap+dx2/rap3+(1-nu)*(a2/2)/rap3)\
                 -(-nu/ra +dx2/ra3 +(1-nu)*(a2/2)/ra3 ) )
        s[0,3] = by*dx*( invrapl*(1+a2/rap2+a2*invrapl)\
                 -invral *(1+a2/ra2 +a2*invral ) )
        s[0,4] = ( bx*(nu/rap-(1-nu)*(a2/2)/rap3) + bz*dx*(1-nu)*invrapl*(1+(a2/2)/rap2+(a2/2)*invrapl) )\
                -( bx*(nu/ra -(1-nu)*(a2/2)/ra3 ) + bz*dx*(1-nu)*invral *(1+(a2/2)/ra2 +(a2/2)*invral ) )
        s[0,5] = by*dx*( (2*nu*invrapl*(1+(a2/2)/rap2+(a2/2)*invrapl)-lp/rap3)\
                - (2*nu*invral *(1+(a2/2)/ra2 +(a2/2)*invral )-l /ra3 ) )
    else:
        rhoa2=dx*dx+a*a
        s[0,0] = by*dx/rhoa2 *( lp/rap*(-1+2*(dx2+a2)/rhoa2+(dx2+a2)/rap/rap)\
                -l /ra *(-1+2*(dx2+a2)/rhoa2+(dx2+a2)/ra /ra ) )
        s[0,1] = +bx*dx/rhoa2*( lp/rap - l/ra )
        s[0,2] = by*( (-nu/rap+dx2/rap3+(1-nu)*(a2/2)/rap3)\
                 -(-nu/ra +dx2/ra3 +(1-nu)*(a2/2)/ra3 ) ) 
        s[0,3] = by*dx/rhoa2*( lp/rap*(1+(a2*2)/rhoa2+(a2)/rap/rap)\
                 - l /ra *(1+(a2*2)/rhoa2+(a2)/ra /ra ) )
        s[0,4] = ( bx*(nu/rap -(1-nu)*(a2/2)/rap3) + bz*dx/rhoa2*(1-nu)*lp/rap*(1+(a2)/rhoa2+(a2/2)/rap2) )\
                 -( bx*(nu/ra  -(1-nu)*(a2/2)/ra3 ) + bz*dx/rhoa2*(1-nu)*l /ra *(1+(a2)/rhoa2+(a2/2)/ra2 ) )
        s[0,5] = by*dx*( (2*nu*lp/rhoa2/rap*(1+(a2)/rhoa2+(a2/2)/rap2) - lp/rap3)\
                 - (2*nu*l /rhoa2/ra *(1+(a2)/rhoa2+(a2/2)/ra2 ) - l /ra3 ) )
        
    return sunit*array([[s[0,0],s[0,1],s[0,2]],[s[0,1],s[0,3],s[0,4]],[s[0,2],s[0,4],s[0,5]]])


def stresszr(mu,nu,b,r,r1,r2,a):
    r21=r2-r1
    rt=r-r1

    if norm(r21)<1e-20:
        s=zeros([3,3])
        return s

    e1=r21/norm(r21)
    e2=schmidt(rt,e1)
    e3=cross(e1,e2)

    e3=e3/norm(e3)

    #print(e2,'\n',e3,'\n',e1,'\n\n')

    Mt=array([[e2],[e3],[e1]]).reshape([3,3])
    M=Mt.T

    #print("M is,'\n'",M)
    #print("Mt is,'\n'",Mt,'\n\n')

    rr21=dot(Mt,r21)
    #print("rr21 is")
    #print(rr21)
    rr=dot(Mt,rt)
    #print("rr is")
    #print(rr)
    br=dot(Mt,b)
    #print("br is")
    #print(br,'\n\n')
    
    sr=stresszcai(mu,nu,br,rr,0,rr21[2],a)
    #calsegstrhor3da(MU,NU,br(1),br(2),br(3),0,rr21(3),rr(1),rr(3),a);
    s=dot(M,dot(sr,Mt))

    return s
       

#H & L singular equations
def stresshl(mu,nu,b,r,z1,z2):
    T0=mu/(4*pi*(1-nu))
    R=sqrt(r[0]**2+r[1]**2+(r[2]-z2)**2)
    L=z2-r[2]

    sigmaxx2=b[0]*r[1]/(R*(R+L))*( 1+r[0]**2/R**2+r[0]**2/(R*(R+L)) ) + b[1]*r[0]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
    sigmaxx2*=T0

    sigmayy2=-b[0]*r[1]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) - b[1]*r[0]/(R*(R+L))*( 1+r[1]**2/R**2+r[1]**2/(R*(R+L)) )
    sigmayy2*=T0

    sigmazz2=b[0]*( 2*nu*r[1]/(R*(R+L))+r[1]*L/R**3 )+b[1]*( -2*nu*r[0]/(R*(R+L))-r[0]*L/R**3 )
    sigmazz2*=T0

    sigmaxz2=-b[0]*r[0]*r[1]/R**3 + b[1]*( -nu/R + r[0]**2/R**3 ) +b[2]*r[1]*(1-nu)/(R*(R+L))
    sigmaxz2*=T0

    sigmayz2=b[0]*( nu/R-r[1]**2/R**3 ) +b[1]*r[0]*r[1]/R**3-b[2]*r[0]*(1-nu)/(R*(R+L))
    sigmayz2*=T0

    sigmaxy2=-b[0]*r[0]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) + b[1]*r[1]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
    sigmaxy2*=T0

    R=sqrt(r[0]**2+r[1]**2+(r[2]-z1)**2)
    L=z1-r[2]

    sigmaxx1=b[0]*r[1]/(R*(R+L))*( 1+r[0]**2/R**2+r[0]**2/(R*(R+L)) ) + b[1]*r[0]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
    sigmaxx1*=T0

    sigmayy1=-b[0]*r[1]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) - b[1]*r[0]/(R*(R+L))*( 1+r[1]**2/R**2+r[1]**2/(R*(R+L)) )
    sigmayy1*=T0

    sigmazz1=b[0]*( 2*nu*r[1]/(R*(R+L))+r[1]*L/R**3 )+b[1]*( -2*nu*r[0]/(R*(R+L))-r[0]*L/R**3 )
    sigmazz1*=T0

    sigmaxz1=-b[0]*r[0]*r[1]/R**3 + b[1]*( -nu/R + r[0]**2/R**3 ) +b[2]*r[1]*(1-nu)/(R*(R+L))
    sigmaxz1*=T0

    sigmayz1=b[0]*( nu/R-r[1]**2/R**3 ) +b[1]*r[0]*r[1]/R**3-b[2]*r[0]*(1-nu)/(R*(R+L))
    sigmayz1*=T0

    sigmaxy1=-b[0]*r[0]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) + b[1]*r[1]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
    sigmaxy1*=T0

    sigmaxx=sigmaxx2-sigmaxx1
    sigmayy=sigmayy2-sigmayy1
    sigmazz=sigmazz2-sigmazz1
    sigmaxz=sigmaxz2-sigmaxz1
    sigmayz=sigmayz2-sigmayz1
    sigmaxy=sigmaxy2-sigmaxy1

    return array([[sigmaxx,sigmaxy,sigmaxz],[sigmaxy,sigmayy,sigmayz],[sigmaxz,sigmayz,sigmazz]])


def stresshlr(mu,nu,b,r,r1,r2):
    r21=r2-r1
    rt=r-r1

    if norm(r21)<1e-20:
        s=zeros([3,3])
        return s

    e1=r21/norm(r21)
    e2=schmidt(rt,e1)
    e3=cross(e1,e2)

    e3=e3/norm(e3)

    #print(e2,'\n',e3,'\n',e1,'\n\n')

    Mt=array([[e2],[e3],[e1]]).reshape([3,3])
    M=Mt.T

    #print("M is,'\n'",M)
    #print("Mt is,'\n'",Mt,'\n\n')

    rr21=dot(Mt,r21)
    #print("rr21 is")
    #print(rr21)
    rr=dot(Mt,rt)
    #print("rr is")
    #print(rr)
    br=dot(Mt,b)
    #print("br is")
    #print(br,'\n\n')

    sr=stresshl(mu,nu,br,rr,0,rr21[2])

    s=dot(M,dot(sr,Mt))

    return s
    
    
def stressxx(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):
    
        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmaxx=b[0]*r[1]/(R*(R+L))*( 1+r[0]**2/R**2+r[0]**2/(R*(R+L)) ) + b[1]*r[0]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
        sigmaxx*=T0

        return sigmaxx

    sigmaxx=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmaxx


def stressyy(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):
    
        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmayy=-b[0]*r[1]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) - b[1]*r[0]/(R*(R+L))*( 1+r[1]**2/R**2+r[1]**2/(R*(R+L)) )
        sigmayy*=T0

        return sigmayy

    sigmayy=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmayy


def stresszz(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):
    
        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmazz=b[0]*( 2*nu*r[1]/(R*(R+L))+r[1]*L/R**3 )+b[1]*( -2*nu*r[0]/(R*(R+L))-r[0]*L/R**3 )
        sigmazz*=T0

        return sigmazz

    sigmazz=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmazz




def stressxz(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):
    
        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmaxz=-b[0]*r[0]*r[1]/R**3 + b[1]*( -nu/R + r[0]**2/R**3 ) +b[2]*r[1]*(1-nu)/(R*(R+L))
        sigmaxz*=T0

        return sigmaxz

    sigmaxz=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmaxz

def stressyz(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):

        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmayz=b[0]*( nu/R-r[1]**2/R**3 ) +b[1]*r[0]*r[1]/R**3-b[2]*r[0]*(1-nu)/(R*(R+L))
        sigmayz*=T0

        return sigmayz

    sigmayz=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmayz


def stressxy(mu,nu,b,r,z1,z2):

    def stressz(mu,nu,b,r,z):

        T0=mu/(4*pi*(1-nu))
        R=sqrt(r[0]**2+r[1]**2+(r[2]-z)**2)
        L=z-r[2]

        sigmaxy=-b[0]*r[0]/(R*(R+L))*( 1-r[1]**2/R**2-r[1]**2/(R*(R+L)) ) + b[1]*r[1]/(R*(R+L))*( 1-r[0]**2/R**2-r[0]**2/(R*(R+L)) )
        sigmaxy*=T0

        return sigmaxy

    sigmaxy=stressz(mu,nu,b,r,z2)-stressz(mu,nu,b,r,z1)

    return sigmaxy



