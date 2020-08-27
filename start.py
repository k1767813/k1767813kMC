from numpy import(
    
    log,sqrt,arctan,zeros,array,arange,diag,append,copy,
    ceil,vstack,reshape,where,concatenate,dtype,sign,sin,cos,
    delete,insert,abs,hstack,dot,pi,sum,empty,argmin,argmax,
    exp,log,cumsum,random,mean,save,load,round,unique,argsort
)

##from numpy_indexed import group_by


from scipy.interpolate import interp1d

from time import time

##import matplotlib
###from mpl_toolkits import mplot3d
##from mpl_toolkits.mplot3d import Axes3D
##from matplotlib.pyplot import ion,figure,subplot,plot,pause,show,text,close
##matplotlib.use("TkAgg")


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
a0=2.87e-10             #Lattice constant Fe (in m)
a=a0/2*sqrt(3)          #Lattice constant along dislocation line 
h=2.31e-10              #Kink height 
bx=a0/2*sqrt(3)         #Screw component of Burgers vector for a horizontal segment
by=0                    #Edge component of Burgers vector for a horizontal segment
bz=0                    #Screw component of Burgers vector for a vertical segment

#omega0=2.31e11         #Ivo
#omega0=0.9489157e8     #0.69*eV 1D =10.65
#omega0=1.046926e8       #0.69*eV 3D ~10.65

##sigmaxz= 31.87555227e6   #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
##sigmaxy=-8.541028488e6  #Applied stress in Pa (force in the -z-direction leading to cross-slip.

sigmaxz= 48.29629131e6   #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
sigmaxy=-12.94095227e6  #Applied stress in Pa (force in the -z-direction leading to cross-slip.

##sigmaxz= 96.59258263e6   #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
##sigmaxy=-25.88190451e6  #Applied stress in Pa (force in the -z-direction leading to cross-slip.

##sigmaxz= 193.1851653e6   #Applied stress in Pa (force in the +y-direction causing glide on x-y slip plane)
##sigmaxy=-51.76380902e6  #Applied stress in Pa (force in the -z-direction leading to cross-slip.





Ts=400
T=400                   #Temperature (in K)
mu=econst(400,111)[0]  #Shear modulus Fe (in Pa) T=400K
nu=econst(400,111)[1]  #Poisson's ratio Fe
rc=0.29964115*bx       #Dislocation radius parameter Fe

##Ts=300
##T=300                       #Temperature (in K)
##mu=econst(300,111)[0]      #Shear modulus Fe (in Pa) T=300K
##nu=econst(300,111)[1]      #Poisson's ratio Fe
##rc=0.305293687*bx           #Dislocation radius parameter Fe

##Ts=150
##T=150                     #Temperature (in K)
##mu=econst(150,111)[0]    #Shear modulus Fe (in Pa) T=150K
##nu=econst(150,111)[1]    #Poisson's ratio Fe
##rc=0.311988037*bx         #Dislocation radius parameter Fe     
##


kB=1.380649e-23         #Boltzmann's constant (in J/K)  
kBT=kB*T                #In J

N=800                  #Dislocation line length in unit of a
N0=1000
L=N*a                   #Dislocation line length (in Angstrom)


n0=0                    #Embrionic kink pair width minus 1
n=30                    #Target kink pair width

B=(2.7+0.008*T)*1e-5
xmig=5


Niter=5000           #maximum number of iterations
plotfreq=5          #frequency of plotting

ysave=zeros([int(Niter/plotfreq),4])     #average dislocation position and number of kinks

rate=zeros([Niter,1],dtype="float128")  #total rate saved after each iteration
dt=zeros([Niter,1],dtype="float128")     #time step
t=zeros([Niter,1],dtype="float128")      #cumulated time


b=array([bx,by,bz])
sigma=array([sigmaxz,sigmaxy])
loopselfeng=loopselfenergy(mu,nu,b,a,h,rc)      #Self and interaction energy of an incremental dislocation loop




def frange(x):
    k=0
    l=[0]
    for i in range(x):
        k+=(i+1)*(-1)**(i+1)
        l.append(k)
    return l

def identify(array1,array2,i2):
    w=where( (array1[::2,0]==array2[i2,0])&(array1[::2,1]==array2[i2,1])&(abs(array1[::2,2]-array2[i2,2])<1e-5)& \
             (array1[1::2,0]==array2[i2+1,0])&(array1[1::2,1]==array2[i2+1,1])&(abs(array1[1::2,2]-array2[i2+1,2])<1e-5) )[0]
    return w





#Obtain energy tables
while True:
        try:
            etablesegs=load('etablesegsN%sT%s.npy'%(N0,T),allow_pickle=True)
            break
        except FileNotFoundError:
            etablesegs=zeros([2*N,24,24,4,4],dtype="float128")
            break

print(etablesegs[etablesegs!=0].size)


while True:
    try:
        etable=load('etable111screwN%sT%s.npy' %(N0,T),allow_pickle=True)
        break
    except FileNotFoundError:
        #etable=empty([0,0,0,3,3])
        print('Calculating loop-loop energy table...')
        etable=createlltable(mu,nu,b,a,h,rc,N0,2*N0,24,24,etablesegs)
        save('etable111screwN%sT%s.npy' %(N0,T),etable)
        break
save('etablesegsN%sT%s.npy'%(N0,T),etablesegs)
print(etablesegs[etablesegs!=0].size)




#Initialize dislocation line
yz=zeros([N,2],dtype="float128")                                                #N segments, y- and z-coordinates
ri=concatenate([arange(1,N),[0]])                                               #indices of i+1 segment; e.g. (1,2,3,4,0), if N=5
li=concatenate([[N-1],arange(N-1)])                                             #Indices of ith segment left neighbour

coordv=empty([0,5],dtype="float128")                                            #Coordinates of vertical segments
segmentplane=zeros([N,])                                                        #Plane of each segment: 0=a-type, 1=b-type, 2=c-type
change=[0,0,0,0]                                                                #Change to dislocation line: [segment, ydiff, zdiff, plane]
yz,coordh,coordv,segmentplane=segments(yz,coordv,segmentplane,change,N)         #Coordinates of horizontal and vertical segments

Changes=array([[1,0,0],[-1,0,0],[1/2,sqrt(3)/2,1],[-1/2,-sqrt(3)/2,1],\
                   [1/2,-sqrt(3)/2,2],[-1/2,sqrt(3)/2,2]],dtype="float128").reshape([6,3])

changeskm=array([[Changes[0,:]],[Changes[1,:]],[Changes[0,:]],[Changes[1,:]],\
                         [Changes[2,:]],[Changes[3,:]],[Changes[2,:]],[Changes[3,:]],\
                         [Changes[4,:]],[Changes[5,:]],[Changes[4,:]],[Changes[5,:]]],dtype="float128").reshape([12,3])




#Energy differences due to an incremental change to each segment (Nx6)
while True:
    try:
        Ediffs=load('Ediffs111screwN%sT%s.npy' %(N,T))
        break
    except FileNotFoundError:
        print("Calculating initial Ediffs...")
        Ediffs=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,[],yz,coordh,coordv,etablesegs)
        save('Ediffs111screwN%sT%s.npy' %(N,T),Ediffs)
        break

save('etablesegsN%sT%s.npy'%(N0,T),etablesegs)



#Load stress table
while True:
    try:
        stable=load('stableN%sT%s.npy' %(N0,T),allow_pickle=True)
        break
    except FileNotFoundError:
        stable=zeros([2*N0,36,36,4,4,3,3],dtype="float64")
        break
#Initial stress on each segment
while True:
    try:
        sh=load('lstressN%sT%s.npy' %(N,T))
        break
    except FileNotFoundError:
        print("Calculating initial stress...")
        sh=linestress(mu,nu,b,coordh,[],coordh,[],a,h,rc,N,stable)[0]
        save('lstressN%sT%s.npy' %(N,T),sh)
        break

save('stableN%sT%s.npy' %(N0,T),stable)
print(stable[stable!=0].size)
#print(sh,'\n\n')



#Add the applied stress
sh[:,0,1]+=sigma[1]
sh[:,1,0]+=sigma[1]
sh[:,0,2]+=sigma[0]
sh[:,2,0]+=sigma[0]
sv=empty([0,3,3],dtype="float128")
sv[:,0,1]+=sigma[1]
sv[:,1,0]+=sigma[1]
sv[:,0,2]+=sigma[0]
sv[:,2,0]+=sigma[0]


Ea1=zeros([N,],dtype="float128")
Ea2=zeros([N,],dtype="float128")
Eb1=zeros([N,],dtype="float128")
Eb2=zeros([N,],dtype="float128")
Ec1=zeros([N,],dtype="float128")
Ec2=zeros([N,],dtype="float128")
                                                                                #Applied RSS is cos(theta)*sigmaxz-sin(theta)*sigmaxy (theta = angle between +y-axis and +z-axis)
Ea1=Ediffs[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
Ea2=Ediffs[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
Eb1=Ediffs[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
Eb2=Ediffs[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
Ec1=Ediffs[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
Ec2=Ediffs[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

#print(Ea1.dtype,Ea1.shape)

effstra=(-Ea1+Ea2)/(2*a*b[0]*h)                                                 #Effective stress on a-planes
effstrb=(-Eb1+Eb2)/(2*a*b[0]*h)                                                 #Effective stress on b-planes
effstrc=(-Ec1+Ec2)/(2*a*b[0]*h)                                                 #Effective stress on c-planes


#Compute the relevant rates
ra1=zeros([N,1],dtype="float128")
ra2=zeros([N,1],dtype="float128")
rna1=zeros([N,1],dtype="float128")
rna2=zeros([N,1],dtype="float128")
rb1=zeros([N,1],dtype="float128")
rb2=zeros([N,1],dtype="float128")
rnb1=zeros([N,1],dtype="float128")
rnb2=zeros([N,1],dtype="float128")
rc1=zeros([N,1],dtype="float128")
rc2=zeros([N,1],dtype="float128")
rnc1=zeros([N,1],dtype="float128")
rnc2=zeros([N,1],dtype="float128")





H0=0.69*eV
tau0=258.2267284e6    #Corresponding to 0.69eV barrier
omega0=1.046926e8       #0.69*eV 3D ~10.65
#p=1
#q=1

p=0.2
q=2





def rnp(tau):
    rn=zeros([N,])
    w1=where( (tau<0 ) & (abs(tau)<tau0) )[0]
    w2=where( (tau>=0) & (abs(tau)<tau0) )[0]
    rn[w1]=omega0*exp(-(H0*(2-(1-(abs(tau[w1])/tau0)**p)**q))/kBT )
    rn[w2]=omega0*exp(-(H0*(1-(tau[w2]/tau0)**p)**q)/kBT )    
    return rn.reshape([N,1])
def rnm(tau):
    rn=zeros([N,])
    w1=where( (tau<0 ) & (abs(tau)<tau0) )[0]
    w2=where( (tau>=0) & (abs(tau)<tau0) )[0]
    rn[w1]=omega0*exp(-(H0*(1-(abs(tau[w1])/tau0)**p)**q)/kBT)
    rn[w2]=omega0*exp(-(H0*(2-(1-(tau[w2]/tau0)**p)**q))/kBT)
    return rn.reshape([N,1])

rna1=rnp(effstra)
rna2=rnm(effstra)
rnb1=rnp(effstrb)
rnb2=rnm(effstrb)
rnc1=rnp(effstrc)
rnc2=rnm(effstrc)




#Wide-kink-pair nucleation rates set according to whether or not the neighbourhood of the h.segment has enough room for a wide-kink-pair
n1=-round(n/2)+1
n2=n1+n-1
n1a=n1-1
n2a=n2+1
def rmvembnuc(N):
    which=zeros([N,])
    yz1=2*yz
    yz1[:,1]/=sqrt(3)
    yz1=round(yz1)
    ra1[:]=0.0
    ra2[:]=0.0
    rb1[:]=0.0
    rb2[:]=0.0
    rc1[:]=0.0
    rc2[:]=0.0
    for i in range(N):
        w1=where( (coordv[:,0]>(n1a+i)%N)&(coordv[:,0]<(n2a+i)%N) )[0]
        w2=where( (coordv[:,0]>(-n2a+i)%N)&(coordv[:,0]<(-n1a+i)%N) )[0]
        if n%2!=0:
            if sum((yz1[(arange(n1a,n2a+1,dtype=int)+i)%N,0]==yz1[i,0])&\
                   (yz1[(arange(n1a,n2a+1,dtype=int)+i)%N,1]==yz1[i,1]))==(n2a-n1a+1) and w1.size==0:    #If the neighbourhood of i has enough room for wide kink-pair
                ra1[i]=rna1[i]
                ra2[i]=rna2[i]
                rb1[i]=rnb1[i]
                rb2[i]=rnb2[i]
                rc1[i]=rnc1[i]
                rc2[i]=rnc2[i]
        else:
            if sum((yz1[(arange(n1a,n2a+1,dtype=int)+i)%N,0]==yz1[i,0])&\
                   (yz1[(arange(n1a,n2a+1,dtype=int)+i)%N,1]==yz1[i,1]))==(n2a-n1a+1) and w1.size==0:
                ra1[i]=rna1[i]
                ra2[i]=rna2[i]
                rb1[i]=rnb1[i]
                rb2[i]=rnb2[i]
                rc1[i]=rnc1[i]
                rc2[i]=rnc2[i]
            elif sum((yz1[(arange(-n2a,-n1a+1,dtype=int)+i)%N,0]==yz1[i,0])&\
                     (yz1[(arange(-n2a,-n1a+1,dtype=int)+i)%N,1]==yz1[i,1]))==(-n1a--n2a+1) and w2.size==0:
                which[i]=1
                ra1[i]=rna1[i]
                ra2[i]=rna2[i]
                rb1[i]=rnb1[i]
                rb2[i]=rnb2[i]
                rc1[i]=rnc1[i]
                rc2[i]=rnc2[i]
                    
    r=vstack([[0],vstack([ra1.T,ra2.T,rb1.T,rb2.T,rc1.T,rc2.T])\
              .reshape([6*N,1],order="F")])                                         #r=(0,ra1[0],ra2[0],rb1[0],rb2[0],rc1[0],rc2[0],...,rc1[N-1],rc2[N-1])

    return r,which

r,which=rmvembnuc(N) #Call the function just defined


#Plot dislocation line profile
coordl=empty([0,6])                               #Coordinates of any dislocation debris loops
##allcoord=coordh
##xvalues=allcoord[:,0]
##yvalues=allcoord[:,1]
##zvalues=allcoord[:,2]
##ax1=figure().add_subplot(111,projection='3d')                               #Create a figure "ax" with one plot 
##ax1.get_proj = lambda: dot(Axes3D.get_proj(ax1), diag([2, 0.75, 0.75, 1]))
##Ln1,=ax1.plot(xvalues,yvalues,zvalues,"b-")                                 #Plot line "Ln" on the figure "ax"
##ax1.set_xlabel(r"$\mathrm{i^{\,th}}$"" segment")
##ax1.set_ylabel("line displacement / h")
##ax1.set_zlabel("line displacement / h")
##ion()         


Ediffsdiffs=zeros([N,6],dtype="float128")
Ediffsx=zeros([1,6],dtype="float128")
sdiffs=zeros([N,3,3])
tic=time()
tottic=0
tc=0.0                                              #Current time
ksave=empty([0,5])                                  #For saving kink info. coords. (0:5), direction +1 or -1 (5) and velocity (6)
coordvxnw=empty([0,10])
nucrate=empty([0,1])
nucratefb=empty([0,1])
nokinkrate=empty([0,1])
ndkink=empty([0,1])
bplanes=empty([0,2])
cplanes=empty([0,2])
errors=empty([0,1])
vksave=empty([0,2])

#Starting kMC iterations
for it in range(Niter):
    if it==0:
        print("1st iteration starting now...")
    print("Iteration number is: ",it,'\n\n')
    cumrate=np.cumsum(r)                               #Cumulative sum of the elements of the 6Nx1 vector r 
    totrate=cumrate[-1]                             #Total rate is the last element of the cumulative sum
    dtit=-(1/totrate)*log(random.uniform(0,1))    #Random number mapped onto the exp. dist.
    #dtit=1/totrate
    if it%100==0:
        print("coordl is")
        print(coordl,'\n\n')
        print("total rate is")
        print(totrate,'\n\n')
    rate[it]=totrate
    tcold=copy(tc)
    print("N = ",N)

    
    

    
    #Kink migration constant velocity:
    if coordv.size!=0:                              #If there are existing kinks

        #Check for cross-kinks
        ksave=empty([0,5])
        xlist=unique(coordv[:,0],axis=0)
        for i in range(len(xlist)):
            w1=where(coordv[::2,0]==xlist[i])[0]       #indices of coordvspts of the kinks            
            if w1.size==1:
                continue
            else:
                for j in range(len(w1)):
                    m=w1[j]
                    ks=array([[*coordv[2*m,:]],[*coordv[2*m+1,:]]]).reshape([2,5])
                    ksave=append(ksave,ks,0)
        if ksave.size>0 and it%10==0:
            print("after appending ksave")
            print(ksave,'\n\n')
            
        #Calculate the local stress on each vseg crosskink
        if ksave.size>0:
            coordvv1=copy(coordv)
            meanyz=mean(sqrt(yz[:,0]**2+yz[:,1]**2))
            for i in range(len(coordl)//2):
                if abs(sqrt(coordl[2*i,1]**2+coordl[2*i,2]**2)-meanyz)<=2:
                    coordvv1=append(coordvv1,coordl[[2*i,2*i+1],:5].reshape([-1,5]),0) 
            sv=linestress(mu,nu,b,[],ksave,coordh,coordvv1,a,h,rc,N,stable)[1]
            #print("sv is")
            #print(sv,'\n\n')
            #Add the applied stress
            sv[:,0,1]+=sigma[1]
            sv[:,1,0]+=sigma[1]
            sv[:,0,2]+=sigma[0]
            sv[:,2,0]+=sigma[0]
            #print("after adding applied stress")
            #print(sv,'\n\n')
            #Find the resolved (glide) stress
            vk2=empty([ksave.shape[0]//2,1],dtype="float128")
            kmchange2=empty([ksave.shape[0]//2,2])
            vsegdiffs=ksave[1::2,:]-ksave[::2,:]
            for i in range(len(ksave)//2):
                if abs(vsegdiffs[i,2])<1e-5:
                    if vsegdiffs[i,1]>0:
                        theta=0
                    else:
                        theta=pi            
                elif abs(vsegdiffs[i,1]-vsegdiffs[i,2]/sqrt(3))<1e-5:
                    if vsegdiffs[i,1]>0:
                        theta=pi/3
                    else:
                        theta=4*pi/3                                
                elif abs(vsegdiffs[i,1]+vsegdiffs[i,2]/sqrt(3))<1e-5:
                    if vsegdiffs[i,1]>0:
                        theta=-pi/3
                    else:
                        theta=2*pi/3
                #kink velocity is prop. to its driving force (bx*sigma)(cross-product)line direction
                vk2[i,0]=(sin(theta)*sv[i,0,1]-cos(theta)*sv[i,0,2])*b[0]/B
                kmchange2[i,0]=sign(vk2[i,0])
                if sign(vk2[i,0])>0:
                    i1=int(ksave[2*i,3])        #We need to move the right h.seg
                else:
                    i1=int((ksave[2*i,3]-1)%N)  #We need to move the left h.seg
                kmchange2[i,1]=i1
                           
                                   

            vkvk2  =hstack([vk2,vk2]).reshape([2*len(vk2),1])
            #coordvx=copy(coordv)
            ksavex=copy(ksave)
            ksavex=hstack([ksavex,vkvk2])
    ##        if ksave.size>0:
    ##            print("coordvx before averaging xkink velocities")
    ##            print(coordvx)
            #Average any crosskink velocities
            xlist=unique(ksavex[:,0],axis=0)
            for i in range(len(xlist)):
                w4=where(ksavex[::2,0]==xlist[i])[0]
                if w4.size>1:
                    vsum=mean(ksavex[2*w4,5])
                    ksavex[[2*w4,2*w4+1],5]=vsum
            if ksave.size>0 and it%10==0:
                print("ksavex is after averaging any xkink velocities:")
                print(ksavex,'\n\n')
            vk2=ksavex[::2,5].reshape([len(ksave)//2,1])
            kmchange2[:,0]=sign(vk2[:,0])
            
           
                
        
        #coordvspts=coordvsptsrnd[(coordvsptsrnd[:,None]!=xkinks).any(-1).all(-1)]                
        vk=empty([coordv.shape[0]//2,1],dtype="float128")
        kmchange=empty([coordv.shape[0]//2,4],dtype=int)

        def kspeed(coordv):
            global vk
            global kmchange
            global ksave
            vk=zeros([coordv.shape[0]//2,1],dtype="float128")
            kmchange=zeros([coordv.shape[0]//2,4],dtype=int)

            #Load the details of xkink velocities from previous
            if len(ksave)>0:
                for i in range(len(ksave)//2):
                    w1=identify(coordv,ksave,2*i)
                    vk[w1,0]=ksavex[2*i,5]
                    kmchange[w1,0]=kmchange2[i,0]
                #print("vk not equal to zero")
                #print(vk2[vk[:,0]!=0])
                #print("kmchange[w1,0] ne to zero")
                #print(kmchange2[kmchange2[:,0]!=0])



            
            #Find velocities of all kinks        
            for i in range(coordv.shape[0]//2):
                #kmchange[i,0]=sign(vk[i,0])
                w2=identify(ksave,coordv,2*i)
                if w2.size>0:
                    continue
                
                #Indices of the h.segments whose motion up or down leads to glide of the v.segment
                i1=int((coordv[2*i,3]-1)%N)      #Left h.seg
                i2= int(coordv[2*i,3])           #Right h.seg
                
                #Energy costs of the L and R h.seg moves
                Ea=array([Ea1[i1],Ea2[i1],Ea1[i2],Ea2[i2]]).reshape([4,1]) #L h.seg up or down a-planes, R h.seg up or down a-planes
                Eb=array([Eb1[i1],Eb2[i1],Eb1[i2],Eb2[i2]]).reshape([4,1]) #As above b-planes
                Ec=array([Ec1[i1],Ec2[i1],Ec1[i2],Ec2[i2]]).reshape([4,1]) #c-planes
                Ediffskm=vstack([Ea,Eb,Ec])
                ind=array([i1,i1,i2,i2,i1,i1,i2,i2,i1,i1,i2,i2]).reshape([12,1])
                
                indkm1=argmin(Ediffskm)     #Find the direction  of kink migration - move that costs the least energy
##                indkm3=argsort(Ediffskm,axis=0)
##                indkm3=indkm3[1,0]          #Find the second lowest energy move also
                
                if indkm1%2==0:
                    indkm3=indkm1+1
                    if ind[indkm1]==i1:
                        indkm2=indkm1+3
                        kmchange[i,0]=-1
                        
                    else:
                        indkm2=indkm1-1
                        kmchange[i,0]=1
                else:
                    indkm3=indkm1-1
                    if ind[indkm1]==i1:
                        indkm2=indkm1+1
                        kmchange[i,0]=-1
                    else:
                        indkm2=indkm1-3
                        kmchange[i,0]=1
                        
                kmchange[i,1:]=[ind[indkm1],indkm1,indkm3]                  #The change giving the kink migration: [0] gives the direction,[1] gives the index 
                                                                            #of the 1st h.seg that moves, and [2] gives the index of Ediffskm and changeskm
                if ind[indkm1]==i1:                                         
                    lstrs=(-Ediffskm[indkm2]+Ediffskm[indkm1])/(2*a*b[0]*h) #The local stress on the v.segment
                else:
                    lstrs=(-Ediffskm[indkm1]+Ediffskm[indkm2])/(2*a*b[0]*h)
                    
                vk[i,0]=lstrs*b[0]/B                                        #Current kink velocity with the +x-direction as positive

            #print("vk2 is again")
            #print(vk2)
            #print("and kmchange2")
            #print(kmchange2[:,0],'\n\n')
            #print("vk is")
            #print(vk,'\n\n')


        #Call the function just defined to get the speeds etc.
        kspeed(coordv)
        vks=array([it,mean(abs(vk[:,0]))]).reshape([1,2])
        vksave=append(vksave,vks,0)

        tmig=abs(xmig*b[0]/vk)                              #Time for each kink to move a distance xmig*bx
        tmig=min(tmig)                                      #Lowest migration time selected
        xk=abs(vk*tmig/a)                                  
        xk=round(xk)                                        #Distance each kink travels

        if it%20==0:
            print("tmig is %s, dtit is %s"%(tmig,dtit),'\n\n')
        
        #Calculate inter-kink distances:
        dist=zeros([len(vk),len(vk),2])
        dist[:,:,1]=tmig
        for i in range(len(coordv)//2):
            for j in range(i+1,len(coordv)//2):
                x1=coordv[2*i,0]
                x2=coordv[2*j,0]
                pbcshift=round((x1-x2)/N)*N
                x2p=x2+pbcshift
                if kmchange[i,0]==kmchange[j,0]:
                    if kmchange[i,0]==-1:
                        if x1<x2p and abs(vk[i])>abs(vk[j]):
                            dist[i,j,0]=(x1-x2)%N
                            dist[j,i,0]=dist[i,j,0]
                        elif x2p<x1 and abs(vk[j])>abs(vk[i]):
                            dist[i,j,0]=(x2-x1)%N
                            dist[j,i,0]=dist[i,j,0]
                        elif x2p<x1 and abs(vk[i])>abs(vk[j]):
                            dist[i,j,0]=x1-x2p
                            dist[j,i,0]=dist[i,j,0]
                        elif x1<x2p and abs(vk[j])>abs(vk[i]):
                            dist[i,j,0]=x2p-x1
                            dist[j,i,0]=dist[i,j,0]
                    elif kmchange[i,0]==1:
                        if x1<x2p and abs(vk[i])>abs(vk[j]):
                            dist[i,j,0]=x2p-x1
                            dist[j,i,0]=dist[i,j,0]
                        elif x2p<x1 and abs(vk[j])>abs(vk[i]):
                            dist[i,j,0]=x1-x2p
                            dist[j,i,0]=dist[i,j,0]
                        elif x2p<x1 and abs(vk[i])>abs(vk[j]):
                            dist[i,j,0]=(x2p-x1)%N
                            dist[j,i,0]=dist[i,j,0]
                        elif x1<x2p and abs(vk[j])>abs(vk[i]):
                            dist[i,j,0]=(x1-x2p)%N
                            dist[j,i,0]=dist[i,j,0]                    

                elif kmchange[i,0]!=kmchange[j,0]:
                    if kmchange[i,0]==1 and x1>x2:
                        dist[i,j,0]=N-x1+x2
                        dist[j,i,0]=dist[i,j,0]
                    elif kmchange[i,0]==1 and x1<x2:
                        dist[i,j,0]=x2-x1
                        dist[j,i,0]=dist[i,j,0]
                    elif kmchange[i,0]==-1 and x1<x2:
                        dist[i,j,0]=N-x2+x1
                        dist[j,i,0]=dist[i,j,0]
                    elif kmchange[i,0]==-1 and x1>x2:
                        dist[i,j,0]=x1-x2
                        dist[j,i,0]=dist[i,j,0]
                if dist[i,j,0]<2*xmig and dist[i,j,0]>0:
                    if kmchange[i,0]==kmchange[j,0]:
                        tint=a*dist[i,j,0]/abs((abs(vk[i,0])-abs(vk[j,0])))      #Kink intercept time
                        if tint<tmig:
                            dist[i,j,1]=tint
                    else:
                        tint=a*dist[i,j,0]/(abs(vk[i,0])+abs(vk[j,0]))
                        if tint<tmig:
                            dist[i,j,1]=tint

        #Select the lowest time                                                                                          
        tmig2=np.min(dist[:,:,1])
        
        #Choose the new lower time if any
        if abs(tmig2-tmig)/tmig>1e-9:
            #print("dist[i,j] is",'\n',dist,'\n\n')
            #print(coordv,'\n\n')
            tmig=tmig2
            w1=where((abs(dist[:,:,1]-tmig))/tmig<1e-9) #where in dist (ith block,jth row), which kinks are intercepting
            #print(w1)
           #If the new lower migration time is less than the w.k.p. nucleation time
            if tmig<dtit:                 
                #print("tmig less than dtit*1e-6")
                print("tmig less than dtit")
                tc+=tmig                    #Increment time by the migration time
                dt[it]=tmig
                t[it]=tc
                xk=abs(vk*tmig/a)
                #print("xk is before rounding")
                print(xk)
                xk=round(xk)                #Update the distance each kink travels
                #print("xk is after rounding")
                print(xk)
                #print("xk is ",hstack([xk,kmchange[:,0].reshape([-1,1])]))
                
                for i in range(w1[0].shape[0]):
                    l=w1[0][i]
                    m=w1[1][i]
                    if (kmchange[l,0]==kmchange[m,0])*abs(xk[l]-xk[m])+(kmchange[l,0]!=kmchange[m,0])*(xk[l]+xk[m])>dist[l,m,0]: #Check for any rounding errors for the kinks that intercept
                        print("rounding error?",xk[w1[0][i],0],xk[w1[1][i],0])
                        print((kmchange[l,0]==kmchange[m,0])*abs(xk[l]-xk[m]))
                        print((kmchange[l,0]!=kmchange[m,0])*(xk[l]+xk[m]))
                        print(dist[w1[0][i],w1[1][i],0],'\n\n')
                        if xk[w1[0][i],0]>xk[w1[1][i],0]:
                            xk[w1[0][i],0]-=1
                            print("subtract 1 from xk")
                        else:
                            xk[w1[1][i],0]-=1
                            print("subtract 1 from xk")
                    elif (kmchange[l,0]==kmchange[m,0])*abs(xk[l]-xk[m])+(kmchange[l,0]!=kmchange[m,0])*(xk[l]+xk[m])!=dist[l,m,0]:
                        print("rounding error not equal")
                        print("xk[l] = %s, kmchange[l,0] = %s, xk[m] = %s, kmchange[m,0] = %s" %(xk[l],kmchange[l,0],xk[m],kmchange[m,0]) )
                        print("distance between them is")
                        print(dist[l,m,0])
                        xk[l]+=1
                        print("added 1 to xk[l]")
                        
                        
                                                       
            else:                           #If the w.k.p. nucleation time is less than the new lower migraton time
                tc+=dtit                  #Increment time by the w.k.p. time
                #print("dtit less than tmig*1e6")
                print("dtit less than tmig")
                dt[it]=dtit
                t[it]=tc
                #xk=abs((vk*dtit*1e-6)/a)
                xk=abs(vk*dtit/a)
                xk=round(xk)                #Update the distance each kink travels
                for i in range(w1[0].shape[0]):
                    l=w1[0][i]
                    m=w1[1][i]
                    if (kmchange[l,0]==kmchange[m,0])*abs(xk[l]-xk[m])+(kmchange[l,0]!=kmchange[m,0])*(xk[l]+xk[m])>=dist[l,m,0]: #Check for any rounding errors for the kinks that intercept
                        if (kmchange[l,0]==kmchange[m,0])*abs(xk[l]-xk[m])+(kmchange[l,0]!=kmchange[m,0])*(xk[l]+xk[m])>dist[l,m,0]:
                            print("rounding error?",xk[w1[0][i],0],xk[w1[1][i],0])
                            print(dist[w1[0][i],w1[1][i],0],'\n\n')
                            if xk[w1[0][i],0]>xk[w1[1][i],0]:
                                xk[w1[0][i],0]-=1

                                    
        #If there is no new lower time but the wkp nucleation time is less than the migration time                       
        elif dtit<tmig: 
            #print("dtit<tmig*1e6")
            print("dtit<tmig")
            tc+=dtit                  #Increment time by the w.k.p. time
            dt[it]=dtit
            t[it]=tc
            #xk=abs((vk*dt[it]*1e-6)/a)
            xk=abs(vk*dt[it]/a)         #Recalculate the migration distances
            xk=round(xk)
        
        #Increment time by the migration time only as nothing has changed from the 1st time that the migration distances were calculated
        else:
            tc+=tmig                    
            t[it]=tc
            dt[it]=tmig
                      
               
        #Arrange all of the kink details into one array
        kmckmc=hstack([kmchange,kmchange]).reshape([2*len(kmchange),-1])
        #kmckmc2=hstack([kmchange2,kmchange2]).reshape([2*len(kmchange2),-1])
        vkvk  =hstack([vk,vk]).reshape([2*len(vk),1])
        #vkvk2  =hstack([vk2,vk2]).reshape([2*len(vk2),1]) 
        xkxk  =hstack([xk,xk]).reshape([2*len(xk),1])        
        coordvx=hstack([coordv,vkvk,xkxk,kmckmc])
        coordvxx=hstack([coordv,vkvk,xkxk,kmckmc])

        
        #Expected new kink details after migration (annihilation not accounted for)
        coordvxnw=copy(coordvxx)
        coordvxnw[:,0]=coordvxnw[:,0]+kmckmc[:,0]*xkxk[:,0]
        coordvxnw[:,3]=coordvxnw[:,3]+kmckmc[:,0]*xkxk[:,0]
        #Take into account PBC 
        def order(coordvxnw,N):
            w2=where(coordvxnw[::2,0]<0)[0]
            #print("where coordvxnw<0")
            #print(w2)
            if w2.size>0:
                for i in range(len(w2)):
                    modx=coordvxnw[[2*w2[0],2*w2[0]+1],:]                    
                    modx[:,0]=modx[:,0]%N
                    modx[:,3]=modx[:,3]%N
                    #print("modx is")
                    #print(modx)
                    coordvxnw=delete(coordvxnw,[2*w2[0],2*w2[0]+1],0)
                    w3=where(coordvxnw[:,0]>modx[0,0])[0]
                    if w3.size!=0:
                        coordvxnw=insert(coordvxnw,w3[0],modx,0)
                    else:
                        coordvxnw=append(coordvxnw,modx,0)
                    w2=where(coordvxnw[::2,0]<0)[0]
            #Verticals at x=0 are shifted to x=N
            w2=where(coordvxnw[::2,0]==0)[0]
            #print("where coordvxnw=0")
            #print(w2)
            if w2.size>0:
                for i in range(len(w2)):
                    modx=coordvxnw[[2*w2[0],2*w2[0]+1],:]
                    modx[:,0]=N
                    modx[:,3]=0
                    #print("modx is")
                    #print(modx)
                    coordvxnw=delete(coordvxnw,[2*w2[0],2*w2[0]+1],0)
                    w3=where(coordvxnw[:,0]>modx[0,0])[0]
                    if w3.size!=0:
                        coordvxnw=insert(coordvxnw,w3[0],modx,0)
                    else:
                        coordvxnw=append(coordvxnw,modx,0)
                    w2=where(coordvxnw[::2,0]==0)[0]
            #Verticals at x>N become x-N
            w2=where(coordvxnw[::2,0]>N)[0]
            #print("where coordvxnw>N")
            #print(w2)
            if w2.size>0:
                for i in range(len(w2)):
                    modx=coordvxnw[[2*w2[0],2*w2[0]+1],:]
                    modx[:,0]=modx[:,0]%N
                    modx[:,3]=modx[:,3]%N
                    #print("modx is")
                    #print(modx)
                    coordvxnw=delete(coordvxnw,[2*w2[0],2*w2[0]+1],0)
                    w3=where(coordvxnw[:,0]>modx[0,0])[0]
                    if w3.size!=0:
                        coordvxnw=insert(coordvxnw,w3[0],modx,0)
                    else:
                        coordvxnw=append(coordvxnw,modx,0)
                    w2=where(coordvxnw[::2,0]>N)[0]
            #print("coordvxnw is:")
            #print(coordvxnw,'\n\n')
            wwf=where(coordvxnw[:,0]==N)[0]
            if wwf.size>0:
                print("vseg with N")
                print(coordvxnw[wwf,3])
                coordvxnw[wwf,3]=0
                print("After change to zero:")
                print(coordvxnw[wwf,3])
            return coordvxnw

        coordvxnw=order(coordvxnw,N)
        
        #Average the expected crosskink velocities (Even if some xkinks annihilate the remaining xkinks have the conserved momentum)
        xlist=unique(coordvxnw[:,0],axis=0)
        for i in range(len(xlist)):
            w4=where(coordvxnw[::2,0]==xlist[i])[0]
            if w4.size>1:
                vsum=mean(coordvxnw[2*w4,7]*abs(coordvxnw[2*w4,5]))
                vsign=sign(vsum)
                coordvxnw[[2*w4,2*w4+1],5]=vsum
                coordvxnw[[2*w4,2*w4+1],7]=vsign
        #print("coordvxnw is after averaging any xkink velocities:")
        #print(coordvxnw,'\n\n')




        ###
        #Loop through the existing cross kinks first to find the one with the lowest energy change consistent with the summed speed and direction execute that one first update the energy changes then do
        #the next one
        ###      
            
        coordvx1=copy(coordvx)
        def migxkinks(num):
            #global sh
            global coordvx1
            global yz
            global coordh
            global coordv
            global segmentplane
            global Ediffs
            global xk
            global kmchange
            global ksave
            global Ea1
            global Ea2
            global Eb1
            global Eb2
            global Ec1
            global Ec2
            if ksave.size!=0:
                xlist=unique(ksave[:,0],axis=0)      #x-positions of the cross-kinks
                #print(coordv,'\n\n')
                #print("xk is:",'\n',hstack([xk,kmchange[:,0].reshape([-1,1])]),'\n\n')
                #print("ksave is: ",'\n',ksave,'\n\n')
                #print("migrating crosskinks")
                #print("x values of crosskinks")
                #print(xlist)
            else:
                xlist=[]
            #sht=copy(sh)
            yzt=copy(yz)
            coordht=copy(coordh)
            coordvt=copy(coordv)
            segmentplanet=copy(segmentplane)
            Ediffst=copy(Ediffs)
            pppp=0
            kkkk=0
            while True:
                                                    
                for i in range(len(xlist)):
                    #shtt=copy(sht)
                    yztt=copy(yzt)
                    coordhtt=copy(coordht)
                    coordvtt=copy(coordvt)
                    segmentplanett=copy(segmentplanet)
                    Ediffstt=copy(Ediffst)
                    w1=where(coordvx1[::2,0]==xlist[i])[0]    #indices of coordvspts of the cross-kinks
                    #print("where coordvx1=xlist value")
                    #print(w1)
                    if w1.size==0:
                        continue
                    
                    #Check for any vsegs in the way
                    t1=int(xlist[i])
                    t2=int((coordvx1[2*w1[0],0]+coordvx1[2*w1[0],7]*coordvx1[2*w1[0],6])%N)
                    t3=[]
                    if num<2:
                        if coordvx1[2*w1[0],7]>0:
                            if t1<t2:
                                t3=where( (coordvx1[:,0]>t1)&(coordvx1[:,0]<=t2) )[0]
                            elif t1>t2:
                                t3=where( (coordvx1[:,0]>t1)|(coordvx1[:,0]<=t2) )[0]
                        elif coordvx1[2*w1[0],7]<0:
                            if t2<t1:
                                t3=where( (coordvx1[:,0]<t1)&(coordvx1[:,0]>=t2) )[0]
                            elif t2>t1:
                                t3=where( (coordvx1[:,0]>=t2)|(coordvx1[:,0]<t1) )[0]
                    else:
                        if coordvx1[2*w1[0],7]>0:
                            if t1<t2:
                                t3=where( (coordvx1[:,0]>t1)&(coordvx1[:,0]<t2) )[0]
                            elif t1>t2:
                                t3=where( (coordvx1[:,0]>t1)|(coordvx1[:,0]<t2) )[0]
                        elif coordvx1[2*w1[0],7]<0:
                            if t2<t1:
                                t3=where( (coordvx1[:,0]<t1)&(coordvx1[:,0]>t2) )[0]
                            elif t2>t1:
                                t3=where( (coordvx1[:,0]>t2)|(coordvx1[:,0]<t1) )[0]
                    if len(t3)>0:
                        print("vseg in between cross kink initial and final positions")
                        print("cross kink with x: ",xlist[i])
                        print("v seg with x: ",coordvx1[t3[0],0])
                        print("ksavex")
                        print(ksavex)
                        print("coordvx")
                        print(coordvx)
                        continue
                    if pppp==1:
                        print("pppp=1")
                        print("t1 = ",t1)
                        print("t2 = ",t2)
                        print("t1<t2:",t1<t2)
                        print("line in coordvx1")
                        print(coordvx1[2*w1[0],:])
                        print("coordvx1 is")
                        print(coordvx1)
                        print("coordvx is")
                        print(coordvx)
                        print("t3 is:",t3)
                        print("where(coordvx1[:,0]>t1)[0]")
                        print(where(coordvx1[:,0]>t1)[0])
                        print("where(coordvx1[:,0]<=t2)[0]")
                        print(where(coordvx1[:,0]<=t2)[0])
                        print("where(coordvx1[:,0]<t2)[0]")
                        print(where(coordvx1[:,0]<t2)[0])
                        
                    
                    w2=where(coordvx1[::2,0]==xlist[i])[0]    #indices of ksaveold for the x-position in question
                    #print("where coordvx1=xlist value")
                    #print(w2)
                    w2=w2[0]                                #select the first one (doesn't matter which one, the details are mostly the same for a particular cross-kink or pile-up)
                    if coordvx1[2*w1[0],6]==0:
                        coordvx1=delete(coordvx1,[2*w1,2*w1+1],0)
                        continue
                                    
                    if coordvx1[2*w2,7]>0:                    #if the cross-kink or pile-up is moving in the +x-direction
                        i1=int(coordvx1[2*w2,3])              #We need to move the right h.seg
                        #print(list(range(len(w1)-1,-1,-1)))
                                               
                        for j in range(len(w1)-1,-1,-1):
    ##                        if j==len(w1)-1:
    ##                            w11=identify(coordvx,coordvx1,2*w1[j])
    ##                            if where(coordvx[2*w11,0]+coordvx1[2*w1[j],6]==coordvx
                            #shtt1=copy(sht)
                            yztt1=copy(yzt)
                            coordhtt1=copy(coordht)
                            coordvtt1=copy(coordvt)
                            segmentplanett1=copy(segmentplanet)
                            Ediffstt1=copy(Ediffst)
                            ydiff=coordvx1[2*w1[j]+1,1]-coordvx1[2*w1[j],1]
                            zdiff=coordvx1[2*w1[j]+1,2]-coordvx1[2*w1[j],2]
                            kp=0
                            rowlist=[0,1,2]    
                            while True:
                                Ea1=Ediffst[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
                                Ea2=Ediffst[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
                                Eb1=Ediffst[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
                                Eb2=Ediffst[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
                                Ec1=Ediffst[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
                                Ec2=Ediffst[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

                                Ediffskm=array([[Ea1[i1],Ea2[i1]],[Eb1[i1],Eb2[i1]],[Ec1[i1],Ec2[i1]]]).reshape([3,2])
                                Ediffskm2=copy(Ediffskm)

                                if kp==0 or kp==1 or kp==4:
                                    if abs(zdiff)<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[0,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[0,:]),argmin(Ediffskm[0,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[0,:])
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[0,:])
                                            rowlist.remove(0)
                                    elif abs(ydiff-zdiff/sqrt(3))<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[1,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[1,:]),argmin(Ediffskm[1,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[1,:])+2
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[1,:])+2
                                            rowlist.remove(1)
                                    elif abs(ydiff+zdiff/sqrt(3))<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[2,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[2,:]),argmin(Ediffskm[2,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[2,:])+4
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[2,:])+4
                                            rowlist.remove(2)
                                if kp==2 or kp==3:
                                    Ediffskm2=Ediffskm2[rowlist,:]
                                    #print("Ediffskm2 is")
                                    #print(Ediffskm2)
                                    #print("Original Ediffskm is")
                                    #print(Ediffskm)
                                    pq=np.unravel_index(argmin(Ediffskm2, axis=None), Ediffskm2.shape)
                                    ww=where(abs(Ediffskm-Ediffskm2[pq])/Ediffskm2[pq]<1e-5)[0]
                                    #print("Row of Ediffskm which has minimum of Ediffskm2")
                                    #print(Ediffskm[ww,:])
                                    ww=ww[0]
                                    if kp==2:
                                        pq=argmin(Ediffskm[ww,:])+2*ww
                                        #print("kp=2, pq=%s"%pq)
                                    else:
                                        pq=argmax(Ediffskm[ww,:])+2*ww
                                        #print("kp=3, pq=%s"%pq)
                                   
                                #print("pq is",pq)
                                print("kink being moved:")
                                print(coordvx1[2*w1[j],:])
                                print(coordvx1[2*w1[j]+1,:])
                                x1=i1
                                x2=i1+coordvx1[2*w1[j],6]
                                indices=(arange(x1,x2,dtype=int))%N
                                #if kp==4:
                                if kp==0 or kp==1:
                                    print(", coordvt before migration")
                                    print(coordvt)
                                for k in range(len(indices)):
                                    changes=[indices[k],*Changes[pq,:]]
    ##                                coordvv1=copy(coordvt)
    ##                                coordvv1=append(coordvv1,coordl[:,:5].reshape([-1,5]),0) 
    ##                                sht+=lstresschange(mu,nu,b,a,h,rc,changes,yzt,coordht,coordvv1,stable)                          #Update the stress on h-segs
                                    Ediffsdiffs=looploopenginteractall(mu,nu,b,a,h,rc,changes,yzt,etable,etablesegs)
                                    Ediffst+=Ediffsdiffs                                                                           #Update the previous state's energy differences
                                    yzt,coordht,coordvt,segmentplanet=segments(yzt,coordvt,segmentplanet,changes,N)
                                    #if kp==4:
                                    if kp==0 or kp==1:
                                        print("moving segs in range xk[i]")
                                        print(coordvt)
                                    coordvv=copy(coordvt)
                                    coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0) 
                                    Ediffsx=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,\
                                                            [changes[0],*yzt[changes[0]]],yzt,coordht,coordvv,etablesegs)      #Calculate the energy differences for the changed segment separately

                                    Ediffst[changes[0],:]=Ediffsx
                                                                                  
                                #Update energy changes 
                                Ea1=Ediffst[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
                                Ea2=Ediffst[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
                                Eb1=Ediffst[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
                                Eb2=Ediffst[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
                                Ec1=Ediffst[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
                                Ec2=Ediffst[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

                  
                               #1st check
                                coordvt=makeunitvsegs(coordvt)
                                #if kp==4:
                                if kp==0 or kp==1:
                                    print("coordvt after making unit vsegs:")
                                    print(coordvt,'\n\n')
                                w8=identify(coordvx,coordvx1,2*w1[j])
                                coordvx2=copy(coordvx)
                                coordvx2[:,0]=(coordvx2[:,0]+coordvx2[:,7]*coordvx2[:,6])%N
                                ww=where(coordvx2[:,0]==0)[0]
                                coordvx2[ww,0]=N
                                w9=identify(coordvt,coordvx2,2*w8[0])
                                #if kp==4:
                                if kp==0 or kp==1:
                                    print("Check...")
                                    print("coordvx1[2*w1[j],:] is")
                                    print(coordvx1[[2*w1[j],2*w1[j]+1],:])
                                    print("identify in coordvx... coordvx[2*w8,:] is")
                                    print(coordvx[[2*w8,2*w8+1],:])
                                    print("will be the same row index in coordvx2:")      
                                    print(coordvx2[[2*w8,2*w8+1],:])
                                    print("and in the just updated coordvt:")
                                    print(coordvt[[2*w9,2*w9+1],:])
                                if w9.size==0:
                                    if len(coordvt)>=len(coordvtt1)-3: #Kink annihilation would be 4 less
                                        if kp==0:
                                            #print("Possible error in migrating cross kinks 1st check! Resetting")
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=1
                                            print("kp is: ",kp)
                                        elif kp==1:
                                            #print("Possible error in migrating cross kinks 1st check! Resetting back to argmin")
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=2
                                            print("kp is: ",kp)
                                        elif kp==2:
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=3
                                            print("kp is: ",kp)
                                        elif kp==3:
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=4
                                            print("kp is: ",kp)
                                        elif kp==4:
                                            kkkk=1
                                            break
                                    else:
                                        break
                                else:
                                    break
                                            
                                    
                    elif coordvx1[2*w2,7]<0:
                        i1=int((coordvx1[2*w2,3]-1)%N)   #We need to move the left h.seg
                        for j in range(len(w1)):
                            #shtt1=copy(sht)
                            yztt1=copy(yzt)
                            coordhtt1=copy(coordht)
                            coordvtt1=copy(coordvt)
                            segmentplanett1=copy(segmentplanet)
                            Ediffstt1=copy(Ediffst)
                            ydiff=coordvx1[2*w1[j]+1,1]-coordvx1[2*w1[j],1]
                            zdiff=coordvx1[2*w1[j]+1,2]-coordvx1[2*w1[j],2]
                            kp=0
                            rowlist=[0,1,2]
                            while True:
                                Ea1=Ediffst[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
                                Ea2=Ediffst[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
                                Eb1=Ediffst[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
                                Eb2=Ediffst[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
                                Ec1=Ediffst[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
                                Ec2=Ediffst[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

                                Ediffskm=array([[Ea1[i1],Ea2[i1]],[Eb1[i1],Eb2[i1]],[Ec1[i1],Ec2[i1]]]).reshape([3,2])
                                Ediffskm2=copy(Ediffskm)
                                
                                if kp==0 or kp==1 or kp==4:
                                    if abs(zdiff)<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[0,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[0,:]),argmin(Ediffskm[0,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[0,:])
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[0,:])
                                            rowlist.remove(0)
                                    elif abs(ydiff-zdiff/sqrt(3))<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[1,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[1,:]),argmin(Ediffskm[1,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[1,:])+2
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[1,:])+2
                                            rowlist.remove(1)
                                    elif abs(ydiff+zdiff/sqrt(3))<1e-5:
                                        #print("Ediffskm is: ")
                                        #print(Ediffskm[2,:])
                                        #print("argmax, argmin: ")
                                        #print(argmax(Ediffskm[2,:]),argmin(Ediffskm[2,:]))
                                        if kp==0 or kp==4:
                                            pq=argmin(Ediffskm[2,:])+4
                                        else:
                                            #print("else if kp=1")
                                            pq=argmax(Ediffskm[2,:])+4
                                            rowlist.remove(2)
                                if kp==2 or kp==3:
                                    Ediffskm2=Ediffskm2[rowlist,:]
                                    #print("Ediffskm2 is")
                                    #print(Ediffskm2)
                                    #print("Original Ediffskm is")
                                    #print(Ediffskm)
                                    pq=np.unravel_index(argmin(Ediffskm2, axis=None), Ediffskm2.shape)
                                    ww=where(abs(Ediffskm-Ediffskm2[pq])/Ediffskm2[pq]<1e-5)[0]
                                    #print("Row of Ediffskm which has minimum of Ediffskm2")
                                    #print(Ediffskm[ww,:])
                                    ww=ww[0]
                                    if kp==2:
                                        pq=argmin(Ediffskm[ww,:])+2*ww
                                        #print("kp=2, pq=%s"%pq)
                                    else:
                                        pq=argmax(Ediffskm[ww,:])+2*ww
                                        #print("kp=3, pq=%s"%pq)
                                #print("pq is",pq)
                                print("kink being moved:")
                                print(coordvx1[2*w1[j],:])
                                print(coordvx1[2*w1[j]+1,:])
                                x1=i1-coordvx1[2*w1[j],6]+1
                                x2=i1+1
                                indices=(arange(x1,x2,dtype=int))%N
                                for k in range(len(indices)):
                                    changes=[indices[k],*Changes[pq,:]]
    ##                                coordvv1=copy(coordvt)
    ##                                coordvv1=append(coordvv1,coordl[:,:5].reshape([-1,5]),0) 
    ##                                sht+=lstresschange(mu,nu,b,a,h,rc,changes,yzt,coordht,coordvv1,stable)
                                    Ediffsdiffs=looploopenginteractall(mu,nu,b,a,h,rc,changes,yzt,etable,etablesegs)
                                    Ediffst+=Ediffsdiffs                                                                    #Update the previous state's energy differences
                                    yzt,coordht,coordvt,segmentplanet=segments(yzt,coordvt,segmentplanet,changes,N)
                                    coordvv=copy(coordvt)
                                    coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)
                                    #if kp==4:
                                    if kp==0 or kp==1:
                                        print("moving segs in range xk[i]")
                                        print(coordvt)                        
                                    Ediffsx=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,\
                                                            [changes[0],*yzt[changes[0]]],yzt,coordht,coordvv,etablesegs)      #Calculate the energy differences for the changed segment separately

                                    Ediffst[changes[0],:]=Ediffsx
                                                    
                                #Update energy changes 
                                Ea1=Ediffst[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
                                Ea2=Ediffst[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
                                Eb1=Ediffst[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
                                Eb2=Ediffst[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
                                Ec1=Ediffst[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
                                Ec2=Ediffst[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment


                                #1st check
                                coordvt=makeunitvsegs(coordvt)
                                #if kp==4:
                                if kp==0 or kp==1:
                                    print("coordvt after making unit vsegs:")
                                    print(coordvt,'\n\n')
                                w8=identify(coordvx,coordvx1,2*w1[j])
                                coordvx2=copy(coordvx)
                                coordvx2[:,0]=(coordvx2[:,0]+coordvx2[:,7]*coordvx2[:,6])%N
                                ww=where(coordvx2[:,0]==0)[0]
                                coordvx2[ww,0]=N
                                w9=identify(coordvt,coordvx2,2*w8[0])
                                #if kp==4:
                                if kp==0 or kp==1:
                                    print("Check...")
                                    print("coordvx1[2*w1[j],:] is")
                                    print(coordvx1[[2*w1[j],2*w1[j]+1],:])
                                    print("identify in coordvx... coordvx[2*w8,:] is")
                                    print(coordvx[[2*w8,2*w8+1],:])
                                    print("will be the same row index in coordvx2:")      
                                    print(coordvx2[[2*w8,2*w8+1],:])
                                    print("and in the just updated coordvt:")
                                    print(coordvt[[2*w9,2*w9+1],:])
                                if w9.size==0:
                                    if len(coordvt)>=len(coordvtt1)-3: #Kink annihilation would be 4 less
                                        if kp==0:
                                            #print("Possible error in migrating cross kinks 1st check! Resetting")
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=1
                                            #print("kp is: ",kp)
                                        elif kp==1:
                                            #print("Possible error in migrating cross kinks 1st check! Resetting back to argmin")
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=2
                                            print("kp is: ",kp)
                                        elif kp==2:
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=3
                                            print("kp is: ",kp)
                                        elif kp==3:
                                            #sht=copy(shtt1)
                                            yzt=copy(yztt1)
                                            coordht=copy(coordhtt1)
                                            coordvt=copy(coordvtt1)
                                            segmentplanet=copy(segmentplanett1)
                                            Ediffst=copy(Ediffstt1)
                                            kp=4
                                            print("kp is: ",kp)
                                        elif kp==4:
                                            kkkk=1
                                            break
                                    else:
                                        break
                                else:
                                    break
                                                                                                                                                                      
                    #2nd check
                    if len(coordvt)>len(coordv):
                        print("Error in migrating cross kinks! Resetting")
                        #sht=copy(shtt)
                        yzt=copy(yztt)
                        coordht=copy(coordhtt)
                        coordvt=copy(coordvtt)
                        segmentplanet=copy(segmentplanett)
                        Ediffst=copy(Ediffstt)
                    else:
                        coordvx1=delete(coordvx1,[2*w1,2*w1+1],0)
                        print("coordvx1 after delete")
                        print(coordvx1,'\n\n')
                
                if kkkk==1 and pppp==0:
                    pppp=1
                else:
                    break

                                
            #sh=copy(sht)           
            yz=copy(yzt)
            coordh=copy(coordht)
            coordv=copy(coordvt)
            segmentplane=copy(segmentplanet)
            Ediffs=copy(Ediffst)
                       
##            if ksave.size>0:
##                print(coordv,'\n\n')
##                print("ksave is: ",'\n',ksave,'\n\n')
##
##            coordv=makeunitvsegs(coordv)
##
##            if ksave.size>0:
##                print("coordv after making unit vsegs")
##                print(coordv,'\n\n')





        #Migrate the kinks
        num=0
        while len(coordvx1)>0 and num<3:
            #print("Iteration No. is: ",it)
            #Migrate the xkinks first
            migxkinks(num)                            
            #Check for any kink annihilation
            alist=[]
            for i in range(len(coordvx1)//2):
                a1=identify(coordv,coordvx1,2*i)
                if a1.size==0:
                    alist.extend([2*i,2*i+1])
            if len(alist)>0:
                coordvx1=delete(coordvx1,alist,0)
            #Now migrate the other kinks
            #print(coordv)
            #print(hstack([xk,kmchange[:,0].reshape([-1,1])]))
            #shold=copy(sh)
            yzold=copy(yz)
            coordhold=copy(coordh)
            coordvold=copy(coordv)
            segmentplaneold=copy(segmentplane)
            Ediffsold=copy(Ediffs)
            indlist=[]
            pr=0
                
            for i in range(coordvx1.shape[0]//2):
                if len(ksave)>0:
                    w1=identify(ksave,coordvx1,2*i)
                    if w1.size>0:
                        continue
                #sholdt=copy(shold)
                yzoldt=copy(yzold)
                coordholdt=copy(coordhold)
                coordvoldt=copy(coordvold)
                segmentplaneoldt=copy(segmentplaneold)
                Ediffsoldt=copy(Ediffsold)

                #print("coordvx1 index and index lines at loop cycle start",'\n',i,'\n',coordvx1[[2*i,2*i+1],:],'\n\n')                   
                if coordvx1[2*i,7]==-1:
                    x1=coordvx1[2*i,8]-coordvx1[2*i,6]+1
                    x2=coordvx1[2*i,8]+1
                else:
                    x1=coordvx1[2*i,8]
                    x2=coordvx1[2*i,8]+coordvx1[2*i,6]
                indices=(arange(x1,x2,dtype=int))%N
                sp=0

                #Check for any vsegs in the way
                t1=int(coordvx1[2*i,0])
                t2=int((coordvx1[2*i,0]+coordvx1[2*i,7]*coordvx1[2*i,6])%N)
                t3=[]
                if coordvx1[2*i,7]>0:
                    if t1<t2:
                        t3=where( (coordvx1[:,0]>t1)&(coordvx1[:,0]<t2) )[0]
                    elif t1>t2:
                        t3=where( (coordvx1[:,0]>t1)|(coordvx1[:,0]<t2) )[0]
                elif coordvx1[2*i,7]<0:
                    if t2<t1:
                        t3=where( (coordvx1[:,0]<t1)&(coordvx1[:,0]>t2) )[0]
                    elif t2>t1:
                        t3=where( (coordvx1[:,0]>t2)|(coordvx1[:,0]<t1) )[0]
                if len(t3)>0:
                    print("vseg in between kink initial and final positions (not migrating crosskinks section)")
                    print("kink with x: ",coordvx1[2*i,0])
                    print("v seg with x: ",coordvx1[t3[0],0])
                    print("coordvxnw")
                    print(coordvxnw)
                    print("coordvx")
                    print(coordvx)
                    continue
 
                rowlist=[0,1,2]
                ydiff=coordvx1[2*i+1,1]-coordvx1[2*i,1]
                zdiff=coordvx1[2*i+1,2]-coordvx1[2*i,2]
                if coordvx1[2*i,7]==-1:
                    i1=int((coordvx1[2*i,3]-1)%N)
                else:
                    i1=int(coordvx1[2*i,3]) 
                while True:
##                    print("changeskm is")
##                    if sp==0 or sp==4:
##                        print(changeskm[int(coordvx1[2*i,9]),:])
##                    elif sp==1:
##                        print(changeskm[int(coordvx1[2*i,10]),:])
##                    
                    Ediffskm=array([[Ea1[i1],Ea2[i1]],[Eb1[i1],Eb2[i1]],[Ec1[i1],Ec2[i1]]]).reshape([3,2])
                    Ediffskm2=copy(Ediffskm)
                            
                    if sp==0 or sp==1 or sp==4:
                        if abs(zdiff)<1e-5:
                            #print("Ediffskm is: ")
                            #print(Ediffskm[0,:])
                            #print("argmax, argmin: ")
                            #print(argmax(Ediffskm[0,:]),argmin(Ediffskm[0,:]))
                            if sp==0 or sp==4:
                                pq=argmin(Ediffskm[0,:])
                            else:
                                #print("else if sp=1")
                                pq=argmax(Ediffskm[0,:])
                                rowlist.remove(0)
                        elif abs(ydiff-zdiff/sqrt(3))<1e-5:
                            #print("Ediffskm is: ")
                            #print(Ediffskm[1,:])
                            #print("argmax, argmin: ")
                            #print(argmax(Ediffskm[1,:]),argmin(Ediffskm[1,:]))
                            if sp==0 or sp==4:
                                pq=argmin(Ediffskm[1,:])+2
                            else:
                                #print("else if kp=1")
                                pq=argmax(Ediffskm[1,:])+2
                                rowlist.remove(1)
                        elif abs(ydiff+zdiff/sqrt(3))<1e-5:
                            #print("Ediffskm is: ")
                            #print(Ediffskm[2,:])
                            #print("argmax, argmin: ")
                            #print(argmax(Ediffskm[2,:]),argmin(Ediffskm[2,:]))
                            if sp==0 or sp==4:
                                pq=argmin(Ediffskm[2,:])+4
                            else:
                                #print("else if kp=1")
                                pq=argmax(Ediffskm[2,:])+4
                                rowlist.remove(2)
                    if sp==2 or sp==3:
                        Ediffskm2=Ediffskm2[rowlist,:]
                        #print("Ediffskm2 is")
                        #print(Ediffskm2)
                        #print("Original Ediffskm is")
                        #print(Ediffskm)
                        pq=np.unravel_index(argmin(Ediffskm2, axis=None), Ediffskm2.shape)
                        ww=where(abs((Ediffskm-Ediffskm2[pq])/Ediffskm2[pq])<1e-8)[0]
                        #print("Row of Ediffskm which has minimum of Ediffskm2")
                        #print(Ediffskm[ww,:])
                        ww=ww[0]
                        if sp==2:
                            pq=argmin(Ediffskm[ww,:])+2*ww
                            #print("sp=2, pq=%s"%pq)
                        else:
                            pq=argmax(Ediffskm[ww,:])+2*ww
                            #print("sp=3, pq=%s"%pq)
                    #print("pq is",pq)
                    for j in range(len(indices)):
                        changes=[indices[j],*Changes[pq,:]]
##                        coordvv1=copy(coordvold)
##                        coordvv1=append(coordvv1,coordl[:,:5].reshape([-1,5]),0) 
##                        shold+=lstresschange(mu,nu,b,a,h,rc,changes,yzold,coordhold,coordvv1,stable)                        #Update the stress on h-segs
                        Ediffsdiffs=looploopenginteractall(mu,nu,b,a,h,rc,changes,yzold,etable,etablesegs)
                        Ediffsold+=Ediffsdiffs                                                                              #Update the previous state's energy differences
                        yzold,coordhold,coordvold,segmentplaneold=segments(yzold,coordvold,segmentplaneold,changes,N)
                        coordvv=copy(coordvold)
                        coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)                     
                        Ediffsx=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,\
                                                [changes[0],*yzold[changes[0]]],yzold,coordhold,coordvv,etablesegs)      #Calculate the energy differences for the changed segment separately

                        Ediffsold[changes[0],:]=Ediffsx
                        if sp==0 or sp==1:
                        #if sp==4:
                            print("moving segment")
                            print(coordvx1[[2*i,2*i+1],:],'\n\n')
                            print(coordvold)
                    coordvold=makeunitvsegs(coordvold)
                    w8=identify(coordvx,coordvx1,2*i)
                    coordvx2=copy(coordvx)
                    coordvx2[:,0]=(coordvx2[:,0]+coordvx2[:,7]*coordvx2[:,6])%N
                    ww=where(coordvx2[:,0]==0)[0]
                    coordvx2[ww,0]=N
                    w9=identify(coordvold,coordvx2,2*w8[0])
                    print("Check...")
                    print("coordvx1[2*i,:] is")
                    print(coordvx1[[2*i,2*i+1],:])
                    print("identify in coordvx... coordvx[2*w8,:] is")
                    print(coordvx[[2*w8,2*w8+1],:])
                    print("will be the same row index in coordvx2:")      
                    print(coordvx2[[2*w8,2*w8+1],:])
                    print("and in the just updated coordvold:")
                    print(coordvold[[2*w9,2*w9+1],:])

                    
                    if w9.size==0:
                        if len(coordvold)>=len(coordvoldt)-3:
                            if sp==0:
                                #print(coordvold)
                                #print(coordvxnw)
                                #shold=copy(sholdt)
                                yzold=copy(yzoldt)
                                coordhold=copy(coordholdt)
                                coordvold=copy(coordvoldt)
                                segmentplaneold=copy(segmentplaneoldt)
                                Ediffsold=copy(Ediffsoldt)
                                sp=1
                                #print("sp is: ",sp)
                                #print(coordvold)
                            elif sp==1:
                                print("coordvold is")
                                print(coordvold)
                                print("coordvxnw is")
                                print(coordvxnw)
                                #shold=copy(sholdt)
                                yzold=copy(yzoldt)
                                coordhold=copy(coordholdt)
                                coordvold=copy(coordvoldt)
                                segmentplaneold=copy(segmentplaneoldt)
                                Ediffsold=copy(Ediffsoldt)
                                sp=2
                                print("sp is: ",sp)
                                #print(coordvold)
                            elif sp==2:
                                #print(coordvold)
                                #print(coordvxnw)
                                #shold=copy(sholdt)
                                yzold=copy(yzoldt)
                                coordhold=copy(coordholdt)
                                coordvold=copy(coordvoldt)
                                segmentplaneold=copy(segmentplaneoldt)
                                Ediffsold=copy(Ediffsoldt)
                                sp=3
                                print("sp is: ",sp)
                                #print(coordvold)
                            elif sp==3:
                                #print(coordvold)
                                #print(coordvxnw)
                                #shold=copy(sholdt)
                                yzold=copy(yzoldt)
                                coordhold=copy(coordholdt)
                                coordvold=copy(coordvoldt)
                                segmentplaneold=copy(segmentplaneoldt)
                                Ediffsold=copy(Ediffsoldt)
                                sp=4
                                print("sp is: ",sp)
                                #print(coordvold)
                            else:
                                break
                        else:
                            break
                    else:
                        break
                                    
                #check
                if len(coordvold)>len(coordv):
                    #Revert to copies as at the beginning of current i loop cycle
                    #shold=copy(sholdt)
                    yzold=copy(yzoldt)
                    coordhold=copy(coordholdt)
                    coordvold=copy(coordvoldt)
                    segmentplaneold=copy(segmentplaneoldt)
                    Ediffsold=copy(Ediffsoldt)
                else:
##                    if pr==1:
##                        i=m
                    indlist.append([2*i,2*i+1])

                print("coordv after a kink migration",'\n',coordvold,'\n\n')

            print("coordvx1 b4 delete")
            print(coordvx1)
            coordvx1=delete(coordvx1,indlist,0)
            print("coordvx1 after delete")
            print(coordvx1,'\n\n')
                    
            #sh=copy(shold)
            yz=copy(yzold)
            coordh=copy(coordhold)
            coordv=copy(coordvold)
            segmentplane=copy(segmentplaneold)
            Ediffs=copy(Ediffsold)
            Ea1=Ediffs[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
            Ea2=Ediffs[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
            Eb1=Ediffs[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
            Eb2=Ediffs[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
            Ec1=Ediffs[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
            Ec2=Ediffs[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

            num+=1
            #print("num is: ",num)



            
            
                                           
        if dtit<tmig:
            #Update event rates
            Ea1=Ediffs[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
            Ea2=Ediffs[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
            Eb1=Ediffs[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
            Eb2=Ediffs[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
            Ec1=Ediffs[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
            Ec2=Ediffs[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment


            #sh=linestress(mu,nu,b,coordh,[],coordh,coordv,a,h,rc,N,stable)[0]


            effstra=(-Ea1+Ea2)/(2*a*b[0]*h)                                                 #Effective stress on a-planes                
            effstrb=(-Eb1+Eb2)/(2*a*b[0]*h)                                                 #Effective stress on b-planes                
            effstrc=(-Ec1+Ec2)/(2*a*b[0]*h)                                                 #Effective stress on c-planes

##            rna1=omega0*exp(-GptT(effstra)/kBT)
##            rna2=omega0*exp(-GmtT(effstra)/kBT)
##            rnb1=omega0*exp(-GptT(effstrb)/kBT)
##            rnb2=omega0*exp(-GmtT(effstrb)/kBT)
##            rnc1=omega0*exp(-GptT(effstrc)/kBT)
##            rnc2=omega0*exp(-GmtT(effstrc)/kBT)

            rna1=rnp(effstra)
            rna2=rnm(effstra)
            rnb1=rnp(effstrb)
            rnb2=rnm(effstrb)
            rnc1=rnp(effstrc)
            rnc2=rnm(effstrc)

            
            #W.k.p. nucleation event k
            r,which=rmvembnuc(N)
            cumrate=cumsum(r)                               #Cumulative sum of the elements of the 6Nx1 vector r 
            totrate=cumrate[-1]                             #Total rate is the last element of the cumulative sum
            cumprob=cumrate/totrate                         #Compute the partial sum for each of the 6N transitions
            x=random.uniform(0,1)
            k=where(cumprob>=x)[0][0]                       #The selected event
            print("k is %s"%k)
            ndkink=append(ndkink,array([[it]]).reshape([1,1]),0)

            #Execute event k
            change[0]=int((k-1)//6)
            if k%6==1:
                change[1]= Changes[0,0]
                change[2]= Changes[0,1]
                change[3]= 0
            elif k%6==2:
                change[1]= Changes[1,0]
                change[2]= Changes[1,1]
                change[3]= 0
            elif k%6==3:
                change[1]= Changes[2,0]
                change[2]= Changes[2,1]
                change[3]= 1
            elif k%6==4:
                change[1]= Changes[3,0]
                change[2]= Changes[3,1]
                change[3]= 1
            elif k%6==5:
                change[1]= Changes[4,0]
                change[2]= Changes[4,1]
                change[3]= 2
            elif k%6==0:
                change[1]= Changes[5,0]
                change[2]= Changes[5,1]
                change[3]= 2
            
            print("change is: ",change)
            if change[3]==1:
                bplanes=append(bplanes,array([[it,sign(change[1])]]).reshape([1,2]),0)
            if change[3]==2:
                cplanes=append(cplanes,array([[it,sign(change[1])]]).reshape([1,2]),0)

            if which[change[0]]==0:
                indices=(arange(n1,n2+1)+change[0])%N
            else:
                indices=(arange(-n2,-n1+1)+change[0])%N
            for i in range(len(indices)):
                changes=[int(indices[i]),*change[1:]]
##                coordvv1=copy(coordv)
##                coordvv1=append(coordvv1,coordl[:,:5].reshape([-1,5]),0) 
##                sh+=lstresschange(mu,nu,b,a,h,rc,changes,yz,coordh,coordvv1,stable)                        #Update the stress on h-segs
                Ediffsdiffs=looploopenginteractall(mu,nu,b,a,h,rc,changes,yz,etable,etablesegs)
                Ediffs+=Ediffsdiffs                                                         #Update the previous state's energy differences
                yz,coordh,coordv,segmentplane=segments(yz,coordv,segmentplane,changes,N)
                coordvv=copy(coordv)
                coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)
                Ediffsx=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,\
                                        [changes[0],*yz[changes[0]]],yz,coordh,coordvv,etablesegs)      #Calculate the energy differences for the changed segment separately

                Ediffs[changes[0],:]=Ediffsx


            #Insert the new double kink into coordvxnw
            notnewdkink=[]
            for i in range(len(coordvxnw)//2):
                w6=identify(coordv,coordvxnw,2*i)
                if w6.size!=0:
                    notnewdkink.extend([2*w6,2*w6+1])

            coordv2=copy(coordv)
            coordv2=delete(coordv2,notnewdkink,0)

            #print("coordvxnw b4 inserting ndkink")
            #print(coordvxnw,'\n\n')

            for i in range(len(coordv2)//2):
                w7=where(coordvxnw[:,0]>coordv2[2*i,0])[0]
                ks=array([[*coordv2[2*i,:],0,0,0,0,0,0],[*coordv2[2*i+1,:],0,0,0,0,0,0]]).reshape([2,11])
                if w7.size!=0:
                    coordvxnw=insert(coordvxnw,w7[0],ks,0)
                else:
                    coordvxnw=append(coordvxnw,ks,0)

            #print("coordvxnw after inserting ndkink")
            #print(coordvxnw,'\n\n')

            #print(coordv,'\n\n')

##            close()
##            allcoord=coordh
##            x=where(coordv[:,0]==N)[0]
##            if x.size!=0:
##                allcoord=append(allcoord,coordv[x,:3],0)
##            for i in range(1,N):
##                x=where(coordv[:,0]==i)[0]
##                y=where(allcoord[:,0]==i)[0]
##                if x.size==0:
##                    continue
##                else:
##                    allcoord=insert(allcoord,y[-1],coordv[x,:3],0)
##            xvalues=allcoord[:,0]
##            yvalues=allcoord[:,1]
##            zvalues=allcoord[:,2]
##            ax1=figure().add_subplot(111,projection='3d')                               #Create a figure "ax" with one plot 
##            ax1.get_proj = lambda: dot(Axes3D.get_proj(ax1), diag([2, 0.75, 0.75, 1]))
##            Ln1,=ax1.plot(xvalues,yvalues,zvalues,"b-")                                 #Plot line "Ln" on the figure "ax"
##            if len(coordl)>0:
##                iterlist=unique(coordl[:,-1],axis=0)
##                for m in range(len(iterlist)):
##                    ww=where(coordl[:,-1]==iterlist[m])[0]
##                    #print("where(coordl[:,-1]==iterlist[m])[0]")
##                    #print(ww)
##                    xvaluesi=coordl[ww,0]
##                    yvaluesi=coordl[ww,1]
##                    zvaluesi=coordl[ww,2]
##                    #print("xvaluesi=coordl[ww,0].shape")
##                    #print(coordl[ww,0].shape)
##                    #print("xvalues=allcoord[:,0].shape")
##                    #print(allcoord[:,0].shape)
##                    #print(xvaluesi,yvaluesi,zvaluesi)
##                    Lni,=ax1.plot(xvaluesi,yvaluesi,zvaluesi,"r-")
##            ax1.set_xlabel(r"$\mathrm{i^{\,th}}$"" segment")
##            ax1.set_ylabel("line displacement / h")
##            ax1.set_zlabel("line displacement / h")
##            ion()
##            #plot dislocation line profile
##            Ln1.set_xdata(allcoord[:,0])
##            Ln1.set_ydata(allcoord[:,1])                                                        #update the yvalues of the dislocation line "Ln"
##            Ln1.set_3d_properties(allcoord[:,2])
##            if len(coordl)>0:
##                for m in range(len(iterlist)):
##                    ww=where(coordl[:,-1]==iterlist[m])[0]
##                    Lni.set_xdata(coordl[ww,0])
##                    Lni.set_ydata(coordl[ww,1])                                                    #update the yvalues of the dislocation line "Ln"
##                    Lni.set_3d_properties(coordl[ww,2])                       
##                ax1.set_ylim(min(min(coordl[:,1])-0.10,min(allcoord[:,1])-0.10),max(max(coordl[:,1])+0.1,max(allcoord[:,1])+0.1))   #update the y-axis limits
##                ax1.set_zlim(min(min(coordl[:,2])-0.10,min(allcoord[:,2])-0.10),max(max(coordl[:,2])+0.1,max(allcoord[:,2])+0.1))
##            else:
##                ax1.set_ylim(min(allcoord[:,1])-0.10,max(allcoord[:,1])+0.1)   #update the y-axis limits
##                ax1.set_zlim(min(allcoord[:,2])-0.10,max(allcoord[:,2])+0.1)
##            pause(10)                                                                      #pause before the line profile is updated
##                    
                


            
            

        #print("coordvxnw is ",'\n',coordvxnw,'\n\n')
        #print(coordv,'\n\n')
        coordv=makeunitvsegs(coordv)
        #print("coordv after splitting any long vsegs")
        #print(coordv)


         #Check for kink annihilation in expected new kink details coordvxnw
        if len(coordvxnw)>0:
            xlist=unique(coordvxnw[:,0],axis=0)
            for i in range(len(xlist)):
                w1=where(coordvxnw[::2,0]==xlist[i])[0]
                if w1.size>1:
                    while True:
                        ppp=0
                        for j in range(len(w1)-1):
                            l=w1[j]
                            m=w1[j]+1
                            if coordvxnw[2*l,1]==coordvxnw[2*m+1,1] and abs(coordvxnw[2*l,2]-coordvxnw[2*m+1,2])<1e-5 and \
                               coordvxnw[2*l+1,1]==coordvxnw[2*m,1] and abs(coordvxnw[2*l+1,2]-coordvxnw[2*m,2])<1e-5:
                                coordvxnw=delete(coordvxnw,[2*l,2*l+1,2*m,2*m+1],0)
                                ppp=1
                                break
                        w1=where(coordvxnw[::2,0]==xlist[i])[0]
                        if w1.size<2 or ppp==0:
                            break                        



        if len(coordv)>0 and coordv[0,1]!=coordv[-1,1] and abs(coordv[0,2]-coordv[-1,2])>1e-5:
            print("Error somewhere - line start doesn't match the end")
            break
                    
        if len(coordvxnw)!=len(coordv):
            print("Error in migrating kinks, coordvxnw is not the same length as coordv")
            print("coordvxnw is")
            print(coordvxnw)
            print("coordv is")
            print(coordv)
            errors=append(errors,array([it]).reshape([1,1]),0)
            #break



        
        
        #Check for kink annihilation in expected new kink details coordvxnw
        #Find the arithmetic mean of any crosskink velocities
        if len(coordvxnw)>0:
            xlist=unique(coordvxnw[:,0],axis=0)
            for i in range(len(xlist)):
                w1=where(coordvxnw[:,0]==xlist[i])[0]
                if w1.size>2:          
                    vsum=mean(coordvxnw[w1,7]*abs(coordvxnw[w1,5]))
                    coordvxnw[w1,7]=sign(vsum)
                    coordvxnw[w1,5]=vsum
                    #print("coordvxnw after averaging an xkink velocities")
                    #print(coordvxnw,'\n\n')
                    #Check for any dislocation loops that may have formed (start coordinate of cross-kinks will equal the end coordinate)
                    newloop=0
                    if w1.size>5:
                        for j in range(0,len(w1),2):                            
                            for k in range(j+5,len(w1),2): #Must be at least 3 segments in a loop
                                l=w1[j]
                                m=w1[k]
                                #print("comparing segments...")
                                #print("segments in coordv")
                                #print(coordv[[l,l+1],:])
                                #print(coordv[[m-1,m],:])
                                #print("segments in coordvxnw")
                                #print(coordvxnw[[l,l+1],:5])
                                #print(coordvxnw[[m-1,m],:5],'\n\n')
                                if coordv[w1[j],1]==coordv[w1[k],1] and abs(coordv[w1[j],2]-coordv[w1[k],2])<1e-5:
                                    #print("coordv[w1[j],:]")
                                    #print(coordv[w1[j],:])
                                    #print("coordv[w1[k],:]")
                                    #print(coordv[w1[k],:])
                                    nwcoordl=coordv[l:m+1,:].reshape([-1,5])
                                    nrows=len(list(range(l,m+1)))
                                    iteration=empty([nrows,1])
                                    iteration[:,0]=it
                                    nwcoordl=hstack([nwcoordl,iteration])
                                    chkl=[]
                                    for i in range(0,len(nwcoordl),2):
                                        for j in range(len(nwcoordl)-1,-1,-2):
                                            if nwcoordl[i,1]==nwcoordl[j,1] and abs(nwcoordl[i,2]-nwcoordl[j,2])<1e-3 and \
                                               nwcoordl[i+1,1]==nwcoordl[j-1,1] and abs(nwcoordl[i+1,2]-nwcoordl[j-1,2])<1e-3:
                                                chkl.extend([i,i+1,j,j-1])
                                    if len(chkl)>0:
                                        nwcoordl=delete(nwcoordl,chkl,0)
                                    plane=coordv[l,4]#Record the plane at the start of the loop
                                    while True:
                                        try:
                                            coordv[m+1,4]=plane#Make sure the planes are lined up correctly
                                            break
                                        except IndexError:
                                            coordv[0,4]=plane
                                            break                                    
                                    #print("w1=where(coordvxnw[:,0]==xlist[i])[0]")
                                    #print(w1)
                                    #print(coordv[w1,:],'\n\n')
                                    #print(coordv[l:m+1,:].reshape([-1,5]),'\n\n')
                                    #Save loop coordinates
                                    coordl=append(coordl,nwcoordl,0)
                                    print("New coordl is")
                                    print(coordl,'\n\n')
                                    #Delete them from the main coordinate arrays (coordv and coordvxnw)
                                    xx=coordv[w1[0],0] #x-coordinate of loop
                                    print("coordv before delete")
                                    print(coordv)
                                    coordv=delete(coordv,list(range(l,m+1)),0)
                                    coordvxnw=delete(coordvxnw,list(range(l,m+1)),0)
                                    print("coordv after delete")
                                    print(coordv,'\n\n')
                                    cut=round(len(list(range(l,m+1)))*h/a/2)
                                    newloop=1
                                    break
                            if newloop==1:
                                break
                        
                        if newloop==1:
                            #Remember to cut the segment bit of coordv index 3                            
                            #Shorten the dislocation line appropriately
                            ltcut=where(coordv[:,0]<cut)[0] #Problem where first segment was becoming 2nd to last after cut rather than last
                            gtecut=where(coordv[:,0]>=cut)[0]
                            if ltcut.size>0:
                                cutlt=ltcut[-1]             #Find the segment with x less than cut with largest x
                                cutlt=coordv[ltcut,0]

                            print("old N = ",N)
                            N=int(N-cut)
                            print("new N = ",N)

                            if ltcut.size>0:
                                coordv[ltcut,0]-=cutlt
                                coordv[gtecut,0]-=cut
                                coordv[ltcut,3]-=cutlt
                                coordv[gtecut,3]-=cut
                                coordvxnw[ltcut,0]-=cutlt
                                coordvxnw[ltcut,3]-=cutlt
                                coordvxnw[gtecut,0]-=cut
                                coordvxnw[gtecut,3]-=cut
                            else:
                                coordv[:,0]-=cut
                                coordv[:,3]-=cut
                                coordvxnw[:,0]-=cut
                                coordvxnw[:,3]-=cut
                                
                            coordv=order(coordv,N)
                            coordvxnw=order(coordvxnw,N)
##                            ww1=where(coordv[::2,0]<xx)[0]
##                            ww2=where(coordv[::2,0]>xx)[0]
##                            print("coordv b4 cut")
##                            print(coordv,'\n\n')
##                            if ww1.size>0 and ww2.size>0:                               
##                                if where(coordv[::2,0]==xx)[0].size>0:
##                                    print("here")
##                                    if coordv[2*ww2[0],0]-xx>cut:
##                                        print("here1")
##                                        coordv[coordv[:,0]>xx,0]-=cut
##                                        coordv[coordv[:,3]>xx,3]-=cut
##                                        coordvxnw[coordvxnw[:,0]>xx,0]-=cut
##                                        coordvxnw[coordvxnw[:,3]>xx,3]-=cut
##                                    elif xx-coordv[2*ww1[-1],0]>cut:
##                                        print("here2")
##                                        coordv[coordv[:,0]>=xx,0]-=cut
##                                        coordv[coordv[:,3]>=xx,3]-=cut
##                                        coordvxnw[coordvxnw[:,0]>=xx,0]-=cut
##                                        coordvxnw[coordvxnw[:,3]>=xx,3]-=cut
##                                    else:
##                                        print("here3")
##                                        if ww2.size>1:
##                                            if coordv[2*ww2[1],0]-coordv[2*ww2[0],0]>cut:
##                                                print("here4")
##                                                coordv[coordv[:,0]>coordv[2*ww2[0],0],0]-=cut
##                                                coordv[coordv[:,3]>coordv[2*ww2[0],0],3]-=cut
##                                                coordvxnw[coordvxnw[:,0]>coordvxnw[2*ww2[0],0],0]-=cut
##                                                coordvxnw[coordvxnw[:,3]>coordvxnw[2*ww2[0],0],3]-=cut
##                                            else:
##                                                print("here5")
##                                                coordv[coordv[:,0]>coordv[2*ww2[1],0],0]-=cut
##                                                coordv[coordv[:,3]>coordv[2*ww2[1],0],3]-=cut
##                                                coordvxnw[coordvxnw[:,0]>coordvxnw[2*ww2[1],0],0]-=cut
##                                                coordvxnw[coordvxnw[:,3]>coordvxnw[2*ww2[1],0],3]-=cut
##                                        else:
##                                            print("here6")
##                                            coordv[coordv[:,0]>coordv[2*ww2[0],0],0]-=cut
##                                            coordv[coordv[:,3]>coordv[2*ww2[0],0],3]-=cut
##                                            coordvxnw[coordvxnw[:,0]>coordvxnw[2*ww2[0],0],0]-=cut
##                                            coordvxnw[coordvxnw[:,3]>coordvxnw[2*ww2[0],0],3]-=cut
##                                else:
##                                    print("here7")
##                                    coordv[coordv[:,0]>coordv[2*ww2[0],0],0]-=cut
##                                    coordv[coordv[:,3]>coordv[2*ww2[0],0],3]-=cut
##                                    coordvxnw[coordvxnw[:,0]>coordvxnw[2*ww2[0],0],0]-=cut
##                                    coordvxnw[coordvxnw[:,3]>coordvxnw[2*ww2[0],0],3]-=cut
##                            elif ww1.size>0:
##                                print("here8")
##                                coordv[coordv[:,0]>xx,0]-=cut
##                                coordv[coordv[:,3]>xx,3]-=cut
##                                coordvxnw[coordvxnw[:,0]>xx,0]-=cut
##                                coordvxnw[coordvxnw[:,3]>xx,3]-=cut
##                            elif ww2.size>0:
##                                print("here9")
##                                if coordv[2*ww2[0],0]-xx>cut:
##                                    print("here10")
##                                    coordv[coordv[:,0]>xx,0]-=cut
##                                    coordv[coordv[:,3]>xx,3]-=cut
##                                    coordvxnw[coordvxnw[:,0]>xx,0]-=cut
##                                    coordvxnw[coordvxnw[:,3]>xx,3]-=cut
##                                else:
##                                    print("here11")
##                                    coordv[coordv[:,0]>coordv[2*ww2[0],0],0]-=cut
##                                    coordv[coordv[:,3]>coordv[2*ww2[0],0],3]-=cut
##                                    coordvxnw[coordvxnw[:,0]>coordvxnw[2*ww2[0],0],0]-=cut
##                                    coordvxnw[coordvxnw[:,3]>coordvxnw[2*ww2[0],0],3]-=cut
                            print("coordv after cut")
                            print(coordv,'\n\n')
                            #Re-initialize dislocation line and energies, etc.
                            ri=concatenate([arange(1,N),[0]])                                               #indices of i+1 segment; e.g. (1,2,3,4,0), if N=5
                            li=concatenate([[N-1],arange(N-1)])                                             #Indices of ith segment left neighbour
                            change=[0,0,0,0]                                                                #Change to dislocation line: [segment, ydiff, zdiff, plane]
                            yz,coordh,coordv,segmentplane=segments(yz,coordv,segmentplane,change,N)         #Coordinates of horizontal and vertical segments
                            #print("new yz is")
                            #print(yz)
                            #print("new coordh is")
                            #print(coordh,'\n\n')
                            coordvv=copy(coordv)
                            coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)
                            Ediffs=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,[],yz,coordh,coordvv,etablesegs)
##                            save('Ediffs111screwN%sT%s.npy' %(N,T),Ediffs)
##                            sh=linestress(mu,nu,b,coordh,[],coordh,coordvv,a,h,rc,N,stable)[0]
##                            #Add the applied stress
##                            sh[:,0,1]+=sigma[1]
##                            sh[:,1,0]+=sigma[1]
##                            sh[:,0,2]+=sigma[0]
##                            sh[:,2,0]+=sigma[0]

                            Ea1=zeros([N,],dtype="float128")
                            Ea2=zeros([N,],dtype="float128")
                            Eb1=zeros([N,],dtype="float128")
                            Eb2=zeros([N,],dtype="float128")
                            Ec1=zeros([N,],dtype="float128")
                            Ec2=zeros([N,],dtype="float128")
                                                                                                            #Applied RSS is cos(theta)*sigmaxz-sin(theta)*sigmaxy (theta = angle between +y-axis and +z-axis)
                            Ea1=Ediffs[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
                            Ea2=Ediffs[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
                            Eb1=Ediffs[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
                            Eb2=Ediffs[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
                            Ec1=Ediffs[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
                            Ec2=Ediffs[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

                            #print(Ea1.dtype,Ea1.shape)

                            effstra=(-Ea1+Ea2)/(2*a*b[0]*h)                                                 #Effective stress on a-planes
                            effstrb=(-Eb1+Eb2)/(2*a*b[0]*h)                                                 #Effective stress on b-planes
                            effstrc=(-Ec1+Ec2)/(2*a*b[0]*h)                                                 #Effective stress on c-planes

                            #print(effstra.dtype)

                            #Compute the relevant rates
                            ra1=zeros([N,1],dtype="float128")
                            ra2=zeros([N,1],dtype="float128")
                            rna1=zeros([N,1],dtype="float128")
                            rna2=zeros([N,1],dtype="float128")
                            rb1=zeros([N,1],dtype="float128")
                            rb2=zeros([N,1],dtype="float128")
                            rnb1=zeros([N,1],dtype="float128")
                            rnb2=zeros([N,1],dtype="float128")
                            rc1=zeros([N,1],dtype="float128")
                            rc2=zeros([N,1],dtype="float128")
                            rnc1=zeros([N,1],dtype="float128")
                            rnc2=zeros([N,1],dtype="float128")


##                            rna1=omega0*exp(-GptT(effstra)/kBT)
##                            rna2=omega0*exp(-GmtT(effstra)/kBT)
##                            rnb1=omega0*exp(-GptT(effstrb)/kBT)
##                            rnb2=omega0*exp(-GmtT(effstrb)/kBT)
##                            rnc1=omega0*exp(-GptT(effstrc)/kBT)
##                            rnc2=omega0*exp(-GmtT(effstrc)/kBT)

                            rna1=rnp(effstra)
                            rna2=rnm(effstra)
                            rnb1=rnp(effstrb)
                            rnb2=rnm(effstrb)
                            rnc1=rnp(effstrc)
                            rnc2=rnm(effstrc)
                                                        
                            r,which=rmvembnuc(N)

                            #Re-initialize the plot
##                            close()
##                            allcoord=coordh
##                            x=where(coordv[:,0]==N)[0]
##                            if x.size!=0:
##                                allcoord=append(allcoord,coordv[x,:3],0)
##                            for i in range(1,N):
##                                x=where(coordv[:,0]==i)[0]
##                                y=where(allcoord[:,0]==i)[0]
##                                if x.size==0:
##                                    continue
##                                else:
##                                    allcoord=insert(allcoord,y[-1],coordv[x,:3],0)
##                            xvalues=allcoord[:,0]
##                            yvalues=allcoord[:,1]
##                            zvalues=allcoord[:,2]
##                            ax1=figure().add_subplot(111,projection='3d')                               #Create a figure "ax" with one plot 
##                            ax1.get_proj = lambda: dot(Axes3D.get_proj(ax1), diag([2, 0.75, 0.75, 1]))
##                            Ln1,=ax1.plot(xvalues,yvalues,zvalues,"b-")                                 #Plot line "Ln" on the figure "ax"
##                            #xlist2=unique(coordl[:,0],axis=0)
##                            iterlist=unique(coordl[:,-1],axis=0)
##                            #print("xlist2 is:")
##                            #print(xlist2)
##                            for m in range(len(iterlist)):
##                                ww=where(coordl[:,-1]==iterlist[m])[0]
##                                #print("where(coordl[:,-1]==iterlist[m])[0]")
##                                #print(ww)
##                                xvaluesi=coordl[ww,0]
##                                yvaluesi=coordl[ww,1]
##                                zvaluesi=coordl[ww,2]
##                                #print("xvaluesi=coordl[ww,0].shape")
##                                #print(coordl[ww,0].shape)
##                                #print("xvalues=allcoord[:,0].shape")
##                                #print(allcoord[:,0].shape)
##                                #print(xvaluesi,yvaluesi,zvaluesi)
##                                Lni,=ax1.plot(xvaluesi,yvaluesi,zvaluesi,"r-")
##                            ax1.set_xlabel(r"$\mathrm{i^{\,th}}$"" segment")
##                            ax1.set_ylabel("line displacement / h")
##                            ax1.set_zlabel("line displacement / h")
##                            ion()
##                            #plot dislocation line profile
##                            Ln1.set_xdata(allcoord[:,0])
##                            Ln1.set_ydata(allcoord[:,1])                                                        #update the yvalues of the dislocation line "Ln"
##                            Ln1.set_3d_properties(allcoord[:,2])
##                            for m in range(len(iterlist)):
##                                ww=where(coordl[:,-1]==iterlist[m])[0]
##                                Lni.set_xdata(coordl[ww,0])
##                                Lni.set_ydata(coordl[ww,1])                                                    #update the yvalues of the dislocation line "Ln"
##                                Lni.set_3d_properties(coordl[ww,2])                           
##                            ax1.set_ylim(min(min(coordl[:,1])-0.10,min(allcoord[:,1])-0.10),max(max(coordl[:,1])+0.1,max(allcoord[:,1])+0.1))   #update the y-axis limits
##                            ax1.set_zlim(min(min(coordl[:,2])-0.10,min(allcoord[:,2])-0.10),max(max(coordl[:,2])+0.1,max(allcoord[:,2])+0.1))  
##                            pause(10)                                                                      #pause before the line profile is updated
                                                    
        
                
    #If there are no kinks
    else:
        #print(r)
        rf=vstack([[0],vstack([ra1.T,rb1.T,rc1.T])\
              .reshape([3*N,1],order="F")])         
        cumnucrate=cumsum(rf)                   
        #cumnucrate=np.cumsum(ra1)                               #Cumulative sum of the elements of the 6Nx1 vector r 
        totnucrate=cumnucrate[-1]
        print("Speed is ","{:e}".format(totnucrate*h*1e9),"nm/s",'\n\n')
        nucrate=append(nucrate,array([[totnucrate]]).reshape([1,1]),0)
        nucratefb=append(nucratefb,array([[totrate]]).reshape([1,1]),0)
        nokinkrate=append(nokinkrate,array([[dtit]]).reshape([1,1]),0)
        tc+=dtit                                      #Increment time by the w.k.p. nucleation time
        t[it]=tc
        dt[it]=dtit
        cumprob=cumrate/totrate                         #Compute the partial sum for each of the 6N transitions
        x=random.uniform(0,1)
        k=where(cumprob>=x)[0][0]                       #The selected event
        print("k is %s"%k)

        #Execute event k
        change[0]=int((k-1)//6)
        if k%6==1:
            change[1]= Changes[0,0]
            change[2]= Changes[0,1]
            change[3]= 0
        elif k%6==2:
            change[1]= Changes[1,0]
            change[2]= Changes[1,1]
            change[3]= 0
        elif k%6==3:
            change[1]= Changes[2,0]
            change[2]= Changes[2,1]
            change[3]= 1
        elif k%6==4:
            change[1]= Changes[3,0]
            change[2]= Changes[3,1]
            change[3]= 1
        elif k%6==5:
            change[1]= Changes[4,0]
            change[2]= Changes[4,1]
            change[3]= 2
        elif k%6==0:
            change[1]= Changes[5,0]
            change[2]= Changes[5,1]
            change[3]= 2
        
        print("change is: ",change)
        if change[3]==1:
            bplanes=append(bplanes,array([[it,sign(change[1])]]).reshape([1,2]),0)
        if change[3]==2:
            cplanes=append(cplanes,array([[it,sign(change[1])]]).reshape([1,2]),0)


        if which[change[0]]==0:
            indices=(arange(n1,n2+1)+change[0])%N
        else:
            indices=(arange(-n2,-n1+1)+change[0])%N
        for i in range(len(indices)):
            changes=[int(indices[i]),*change[1:]]
##            coordvv1=copy(coordv)
##            coordvv1=append(coordvv1,coordl[:,:5].reshape([-1,5]),0) 
##            sh+=lstresschange(mu,nu,b,a,h,rc,changes,yz,coordh,coordvv1,stable)                     #Update the stress on h-segs
            Ediffsdiffs=looploopenginteractall(mu,nu,b,a,h,rc,changes,yz,etable,etablesegs)
            Ediffs+=Ediffsdiffs                                                                     #Update the previous state's energy differences
            yz,coordh,coordv,segmentplane=segments(yz,coordv,segmentplane,changes,N)
            coordvv=copy(coordv)
            coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)
            Ediffsx=loopenginteract(mu,nu,b,a,h,rc,loopselfeng,\
                                    [changes[0],*yz[changes[0]]],yz,coordh,coordvv,etablesegs)      #Calculate the energy differences for the changed segment separately

            Ediffs[changes[0],:]=Ediffsx


##    if it%10==0:    
##        print(coordv,'\n\n')
    #print("coordv is ",'\n',coordv,'\n\n')
    #print("total time is ",tc)
    #print("t array is ",'\n',t,'\n\n')
##    if ksave.shape[0]!=0:
##        print("ksave is: ",'\n',ksave,'\n\n')
        

                                        
    #Update event rates 
    Ea1=Ediffs[:,0]-( sigma[0]*b[0]*a*h)                                            #Energy differences due to a step forwards on an a-type plane, for each segment
    Ea2=Ediffs[:,1]-(-sigma[0]*b[0]*a*h)                                            #Energy differences due to a step backwards on an a-type plane, for each segment
    Eb1=Ediffs[:,2]-( sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a b-type plane, for each segment
    Eb2=Ediffs[:,3]-(-sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a b-type plane, for each segment
    Ec1=Ediffs[:,4]-( sigma[0]*b[0]*a*h*1/2+sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step forwards on a c-type plane, for each segment
    Ec2=Ediffs[:,5]-(-sigma[0]*b[0]*a*h*1/2-sigma[1]*b[0]*a*h*sqrt(3)/2)            #Energy differences due to a step backwards on a c-type plane, for each segment

    effstra=(-Ea1+Ea2)/(2*a*b[0]*h)                                                 #Effective stress on a-planes        
    effstrb=(-Eb1+Eb2)/(2*a*b[0]*h)                                                 #Effective stress on b-planes        
    effstrc=(-Ec1+Ec2)/(2*a*b[0]*h)                                                 #Effective stress on c-planes

##    if it%25==0:
##        print("where effstra<0")
##        print(where(effstra<0)[0])
##        print("where(effstra>tau0)[0]")
##        print(where(effstra>tau0)[0])
##        print(effstra[effstra>tau0])

##    rna1=omega0*exp(-GptT(effstra)/kBT)
##    rna2=omega0*exp(-GmtT(effstra)/kBT)
##    rnb1=omega0*exp(-GptT(effstrb)/kBT)
##    rnb2=omega0*exp(-GmtT(effstrb)/kBT)
##    rnc1=omega0*exp(-GptT(effstrc)/kBT)
##    rnc2=omega0*exp(-GmtT(effstrc)/kBT)

    rna1=rnp(effstra)
    rna2=rnm(effstra)
    rnb1=rnp(effstrb)
    rnb2=rnm(effstrb)
    rnc1=rnp(effstrc)
    rnc2=rnm(effstrc)



    #Remove embryonic kink pair nucleation
    r,which=rmvembnuc(N)
                    
    
    if it%plotfreq==0:
        meanpos=np.mean(sqrt(yz[:,0]**2+yz[:,1]**2))
        v=meanpos/tc*h
        ydiffl=(yz-yz[li])                                                          #Differences in y-,z-values of ith segment with left neighbours
        nkinks=(where((ydiffl[:,0]!=0)|(ydiffl[:,1]!=0))[0]).size
        toc=time()
        tottic+=toc-tic
        telapse=toc-tic
        tic=toc
        ysave[int(it/plotfreq),:]=[meanpos,nkinks,tc,dt[it]]
        print("iter = %s, mean pos. = %s, total t = %s, v = %s, nkinks = %s, iter. t = %s, total iter. t = %s" %(it,meanpos,tc,v,nkinks,telapse,tottic),'\n',\
              "sigmaxz is %s Pa, sigmaxy is %s Pa and T is %s K"%(sigmaxz,sigmaxy,T),'\n\n')
        
        
##    if it%50==0:
##        coordvv=copy(coordv)
##        coordvv=append(coordvv,coordl[:,:5].reshape([-1,5]),0)
##        sh=linestress(mu,nu,b,coordh,[],coordh,coordvv,a,h,rc,N,stable)[0]
##        #Add the applied stress
##        sh[:,0,1]+=sigma[1]
##        sh[:,1,0]+=sigma[1]
##        sh[:,0,2]+=sigma[0]
##        sh[:,2,0]+=sigma[0]
##        print("Diff between Energy stress and calculated stress")
##        apdiff=abs(effstra-sh[:,0,2])
##        print("a-plane diffs >1e3")
##        print(apdiff[apdiff>1e3])
##        bpdiff=abs(effstrb-(sh[:,0,1]*(-sqrt(3)/2)+sh[:,0,2]/2))
##        print("b-plane diffs >1e3")
##        print(bpdiff[bpdiff>1e3])
##        cpdiff=abs(effstrc-(sh[:,0,1]*sqrt(3)/2+sh[:,0,2]/2))
##        print("c-plane diffs >1e3")
##        print(cpdiff[cpdiff>1e3])
##        print("effstra is")
##        print(effstra)
##        print("sh is")
##        print(sh[:,0,2],'\n\n')
        ##print(sh[:,0,1]*(-sqrt(3)/2)+sh[:,0,2]/2)


        

        #Combine all segments in order for plotting
        #print("processing allcoord...")
##        allcoord=coordh
##        #print(coordv,'\n\n')
##        x=where(coordv[:,0]==N)[0]
##        if x.size!=0:
##            allcoord=append(allcoord,coordv[x,:3],0)
##        for i in range(1,N):
##            x=where(coordv[:,0]==i)[0]
##            y=where(allcoord[:,0]==i)[0]
##            if x.size==0:
##                continue
##            else:
##                allcoord=insert(allcoord,y[-1],coordv[x,:3],0)
##                
##        #print(allcoord)
##        #plot dislocation line profile
##        Ln1.set_xdata(allcoord[:,0])
##        Ln1.set_ydata(allcoord[:,1])                                                        #update the yvalues of the dislocation line "Ln"
##        Ln1.set_3d_properties(allcoord[:,2])
##        if len(coordl)>0:
##            iterlist=unique(coordl[:,-1],axis=0)
##            for i in range(len(iterlist)):
##                ww=where(coordl[:,-1]==iterlist[i])[0]
##                Lni.set_xdata(coordl[ww,0])
##                Lni.set_ydata(coordl[ww,1])                                                    #update the yvalues of the dislocation line "Ln"
##                Lni.set_3d_properties(coordl[ww,2])
##            ax1.set_ylim(min(min(coordl[:,1])-0.10,min(allcoord[:,1])-0.10),max(max(coordl[:,1])+0.1,max(allcoord[:,1])+0.1))   #update the y-axis limits
##            ax1.set_zlim(min(min(coordl[:,2])-0.10,min(allcoord[:,2])-0.10),max(max(coordl[:,2])+0.1,max(allcoord[:,2])+0.1))
##        else:
##            ax1.set_ylim(min(allcoord[:,1])-0.10,max(allcoord[:,1])+0.1)   #update the y-axis limits
##            ax1.set_zlim(min(allcoord[:,2])-0.10,max(allcoord[:,2])+0.1)
##        pause(10)                                                                      #pause before the line profile is updated
##        #break
        
    if it%10==0:
        print("migration time is")
        print(tmig)
        print("time difference with previous iteration")
        print((tc-tcold)*1e9,'ns','\n\n')
        v=np.mean(sqrt(yz[:,0]**2+yz[:,1]**2))*h/tc
        vy=np.mean(yz[:,0])*h/tc
        print('measured velocity v =',v,'m/s')
        print('measured velocity component vy =',vy,'m/s')
    if it%10==0 and it!=0:
        print("coordv is:")
        print(coordv,'\n\n')
        print("Migration distances:")
        print(xk)
        print("Kink speeds:")
        print(vk)
        
##    if coordvxnw.size!=0:
##        print("coordvxnw is")
##        print(coordvxnw)

    if it%100==0:
        save('etablesegsN%sT%s.npy' %(N0,T),etablesegs)
        print(etablesegs[etablesegs!=0].size)
        save('stableN%sT%s.npy' %(N0,T),stable)
        print(stable[stable!=0].size)




   



###Figure 2
##ax2=figure().add_subplot(111)                                           #create a figure "ax2" with one plot 
##Ln2,=ax2.plot(list(range(Niter)),t*1e6,"r-")                            #plot line "Ln2" on the figure "ax2" 
##ax2.set_xlabel("number of kMC steps")
##ax2.set_ylabel("real time "r"$\mu\:\mathrm{s}$")
##
###Figure 3
##ax3=figure().add_subplot(111)
##Ln3,=ax3.plot(t[0:Niter:plotfreq,0]*1e6,ysave[:,0],"r-")
##ax3.set_xlabel("t "r"$\mu\:\mathrm{s}$")
##ax3.set_ylabel("av. pos. / h")

distance=np.mean(sqrt(yz[:,0]**2+yz[:,1]**2))
v=distance*h/t[-1]
vy=np.mean(yz[:,0])*h/t[-1]
avgnkink=sum(ysave[:,1]*ysave[:,3])/sum(ysave[:,3])
avgrate=np.mean(rate)
avgnucrate=np.mean(nucrate)
avgnucratefb=np.mean(nucratefb)

print("coordv is:")
print(coordv)
print("coordl is:")
print(coordl)
print("Errors at iterations:")
print(errors)
print('measured average number of kinks =',avgnkink)
print('average total nucleation rate =',avgrate,'/s')
print('average nucleation nucleation rate =',avgnucrate,'/s')
print('average nucleation nucleation rate fb =',avgnucratefb,'/s')
print('measured velocity v =',v,'m/s')
print('measured velocity component vy =',vy,'m/s')
print('h / avg no kink times (nokink speeds):')
print(h/(mean(nokinkrate)))
print("Distance/h =",distance)
print("average y:")
print(mean(yz[:,0]))
print("average z:")
print(mean(yz[:,1]))
print("bplane iterations and sign(y):")
print(bplanes)
print("cplane iterations and sign(y):")
print(cplanes)
print("New dkinks when there are existing kinks iterations:")
print(ndkink)
print("average kink speed:")
print(mean(vksave[:,1]),"m/s")
print('------')


N=N0
save('etablesegsN%sT%s.npy' %(N0,T),etablesegs)
print(etablesegs[etablesegs!=0].size)
save('stableN%sT%s.npy' %(N0,T),stable)
print(stable[stable!=0].size)

Sr=round(sqrt(sigma[0]**2+sigma[1]**2)*1e-6)
it_vs_t=arange(Niter)
it_vs_t=it_vs_t.reshape([len(it_vs_t),1])
it_vs_t=hstack([it_vs_t,t*1e6])
save('iter_vs_time_5000N%sT%sS%s'%(N,T,Sr),it_vs_t)
t_vs_y=hstack([t[0:Niter:plotfreq,0]*1e6,ysave[:,0]])
save('time_vs_distance_5000N%sT%sS%s'%(N,T,Sr),t_vs_y)


#Combine all segments in order for plotting
allcoord=coordh
x=where(coordv[:,0]==N)[0]
if x.size!=0:
    allcoord=append(allcoord,coordv[x,:3],0)
for i in range(1,N):
    x=where(coordv[:,0]==i)[0]
    y=where(allcoord[:,0]==i)[0]
    if x.size==0:
        continue
    else:
        allcoord=insert(allcoord,y[-1],coordv[x,:3],0)
        
save('allcoord5000',allcoord)
save('coordl5000',coordl)

