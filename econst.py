from numpy import zeros,array,sqrt,roots,empty,dot

def econst(T,method):
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
    elif T==1000:
        c11=169
        c12=123
        c44=102
    else:
        print("Valid temperatures are 150K, 300K, 400K and 1000K only")
        return


    if method=="v" or method=="V":
        mu=1/5*(c11-c12+3*c44)              #Voigt average shear modulus
    elif method=="r" or method=="R":
        mu=5/(4/(c11-c12)+3/c44)            #Reuss average shear modulus
    elif method=="vrha" or method=="VRHA":
        mu=( 1/5*(c11-c12+3*c44) \
             +5/(4/(c11-c12)+3/c44) )/2     #Voigt-Reuss-Hill arithmetic average
    elif method=="vrhg" or method=="VRHG":
        mu=sqrt( 1/5*(c11-c12+3*c44) \
             *(5/(4/(c11-c12)+3/c44)) )     #Voigt-Reuss-Hill geometric average
    elif method=="hk" or method=="HK":
        p4=64
        p3=16*(4*c11+5*c12)
        p2=3*(c11+2*c12)*(5*c11+4*c12)-8*(7*c11-4*c12)*c44
        p1=-(29*c11-20*c12)*(c11+2*c12)*c44
        p0=-3*((c11+2*c12)**2)*(c11-c12)*c44
        mu=roots([p4,p3,p2,p1,p0])
        mu=mu[mu>0]                         
        mu=mu[0]                            #Hershey-Kroner average
    elif method==111:
        mu=1/3*(c11-c12+c44)                #<111> zone shear modulus
    else:
        print("Unknown calc. method")
        return
    

    #mu1=c44*(c11-c12)/(4/3*c44+1/3*(c11-c12)) #Taylor shear modulus in the <111> zone
    
    s11=(c11+c12)/((c11-c12)*(c11+2*c12))
    s12=-c12/((c11-c12)*(c11+2*c12))
    s44=1/c44
    Sa=s11-s12-s44/2

    #Function to find average Poisson ratio on a plane with normal [h,k,l]  (doi:10.1016/j.physb.2006.08.008)
    def v(h,k,l):
        a00=h*l/(sqrt(h**2+k**2)*sqrt(h**2+k**2+l**2))
        a01=k*l/(sqrt(h**2+k**2)*sqrt(h**2+k**2+l**2))
        a02=-sqrt(h**2+k**2)/sqrt(h**2+k**2+l**2)
        a10=-k/sqrt(h**2+k**2)
        a11=h/sqrt(h**2+k**2)
        a12=0
        a20=h/sqrt(h**2+k**2+l**2)
        a21=k/sqrt(h**2+k**2+l**2)
        a22=l/sqrt(h**2+k**2+l**2)
        a=array([[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]])
        a1133=0.0
        a2233=0.0
        a3333=0.0
        for i in range(3):
            a1133+=(a[0,i]**2)*(a[2,i]**2)
            a2233+=(a[1,i]**2)*(a[2,i]**2)
            a3333+=a[2,i]**4
        s13=s12+a1133*Sa
        s23=s12+a2233*Sa
        s33=s12+1/2*s44+a3333*Sa
        nu=-(s13+s23)/(2*s33)    
        return nu          

    K  =1/3*(c11+2*c12)                     #Bulk modulus
    if method==111:
        nu=v(1,1,0)                         #Average Poissons ratio for {110} planes
    else:
        nu=(3*K-2*mu)/(6*K+2*mu)            #Isotropic Poissons ratio

    return mu*1e9,nu
