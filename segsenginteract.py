from numpy import array,sqrt,reshape,round,sign,dot,min,max,dtype,save,load,zeros

from enginteract import enginteract


def segsenginteract(mu,nu,b,hseg1,vseg1,hseg2,vseg2,a,h,rc,N,etable):

#For calculating the elastic interaction energy between incremental-loop segments
#(hseg1,vseg1) and the dislocation's horizontal and vertical segements (hseg2,vseg2)
#PBC applied through the minimum image convention

    if etable.shape[0]!=0:
        if etable[0,0,0,0,0]!=0:
            if abs(etable[0,0,0,0,0]/2-enginteract(mu,nu,b,b,array([0,0,0]),array([a,0,0]),array([0,0,0]),array([a,0,0]),rc)/2)>1e-32:
                etable=zeros([2*N,24,24,4,4])
                print("parameters of etablesegs differs from current ones")

    Nx=etable.shape[0]
    Ny=etable.shape[1]
    Nz=etable.shape[2]
    #print(Nx,Ny,Nz)


    r0=array([0,0,0])
    r1a=array([a,0,0])
    r1h=array([0,h,0])
    r1hz=array([0,h/2,h*sqrt(3)/2])
    r1hzm=array([0,h/2,-h*sqrt(3)/2])


    if len(vseg1)!=0:
        vseg1diffs=vseg1[1::2,:]-vseg1[::2,:]
    if len(vseg2)!=0:
        vseg2diffs=vseg2[1::2,:]-vseg2[::2,:]

    #Number of segments:
    
    N1 =len(hseg1)//2
    Nv1=len(vseg1)//2
    N2 =len(hseg2)//2
    Nv2=len(vseg2)//2


    #Elastic interaction energy:

    Eint=dtype("float128")
    Eint=0.0


    #Horizontal-Horizontal segment interactions:

    if len(hseg1)!=0:

        dxyzhh=(hseg2[::2,:].reshape([N2,1,3])-hseg1[::2,:])  #Distances between h.seg. startpoints
        dxyzhh[:,:,0]-=round(dxyzhh[:,:,0]/N)*N             #PBC shift (minimum image convention)
        
        for i in range(N2):
            for j in range(N1):
                dx=dxyzhh[i,j,0]
                #dx-=round(dx/N)*N
                dy=dxyzhh[i,j,1]                                                  
                dz=dxyzhh[i,j,2]
                dz=dz/(sqrt(3)/2)
                dz=round(dz)
                if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                    if dx<0:
                        dx+=Nx
                    if dy<0:
                        dy+=Ny/2
                    if dz<0:
                        dz+=Nz
                    if etable[int(dx),int(2*dy),int(dz),0,0]!=0:
                        Eint+=etable[int(dx),int(2*dy),int(dz),0,0]
                        #print("etable hsegs used!")
                    else:
                        dxhh=dxyzhh[i,j,0]*a
                        dyhh=dxyzhh[i,j,1]*h
                        dzhh=dxyzhh[i,j,2]*h
                        W=enginteract(mu,nu,b,b,r0,r1a,array([dxhh,dyhh,dzhh]),array([dxhh+a,dyhh,dzhh]),rc)
                        Eint+=W
                        etable[int(dx),int(2*dy),int(dz),0,0]=W                        
                else:
                    dxhh=dxyzhh[i,j,0]*a
                    dyhh=dxyzhh[i,j,1]*h
                    dzhh=dxyzhh[i,j,2]*h
                    Eint+=enginteract(mu,nu,b,b,r0,r1a,array([dxhh,dyhh,dzhh]),array([dxhh+a,dyhh,dzhh]),rc)
        
        


    #Vertical-Vertical segment interactions:

    for i in range(Nv2):
        for j in range(Nv1):

            sign1=sign(vseg1diffs[j,1])
            sign2=sign(vseg2diffs[i,1])
            
            if abs(vseg1diffs[j,2])<1e-5:
                typea=1
                r1=r1h
            elif abs(vseg1diffs[j,1]-vseg1diffs[j,2]/sqrt(3))<1e-5:
                typea=2
                r1=r1hz
            elif abs(vseg1diffs[j,1]+vseg1diffs[j,2]/sqrt(3))<1e-5:
                typea=3
                r1=r1hzm
            else:
                print("vseg1diffs: %s" %vseg1diffs[j,:])
                print("Lines in vseg1:")
                print(vseg1[[2*j,2*j+1],:])
                
            if abs(vseg2diffs[i,2])<1e-5:
                typeb=1
                zdiff=0
            elif abs(vseg2diffs[i,1]-vseg2diffs[i,2]/sqrt(3))<1e-5:
                typeb=2
                zdiff=abs(vseg2diffs[i,2])*h
            elif abs(vseg2diffs[i,1]+vseg2diffs[i,2]/sqrt(3))<1e-5:
                typeb=3
                zdiff=-abs(vseg2diffs[i,2])*h
            else:
                print("vseg2diffs: %s" %vseg2diffs[i,:])
                print("Lines in vseg2:")
                print(vseg2[[2*i,2*i+1],:])
                
            dx=vseg2[2*i,0]-vseg1[2*j,0]
            dx-=round(dx/N)*N
            dy=min(vseg2[[2*i,2*i+1],1])-min(vseg1[[2*j,2*j+1],1])
            ydiff=abs(vseg2diffs[i,1])*h
            
            if (typea==1 or typea==2) and (typeb==1 or typeb==2):
                dz=min(vseg2[[2*i,2*i+1],2])-min(vseg1[[2*j,2*j+1],2])
            elif (typea==1 and typeb==3) or (typea==3 and typeb==1) or (typea==3 and typeb==3):
                dz=max(vseg2[[2*i,2*i+1],2])-max(vseg1[[2*j,2*j+1],2])
            elif typea==2 and typeb==3:
                dz=max(vseg2[[2*i,2*i+1],2])-min(vseg1[[2*j,2*j+1],2])
            elif typea==3 and typeb==2:
                dz=min(vseg2[[2*i,2*i+1],2])-max(vseg1[[2*j,2*j+1],2])
            else:
                print("Error!")
                break

            dxvv=dx
            dyvv=dy
            dzvv=dz
            r2=array([dxvv*a,dyvv*h,dzvv*h])
            r3=array([dxvv*a,dyvv*h+ydiff,dzvv*h+zdiff])
            
            dz=dz/(sqrt(3)/2)
            dz=round(dz)

            if typea==1 and abs(vseg1diffs[j,1])-1>1e-5:
                typea=0
            elif typea==2 and abs(vseg1diffs[j,1])-1/2>1e-5:
                typea=0
            elif typea==3 and abs(vseg1diffs[j,1])-1/2>1e-5:
                typea=0
            if typeb==1 and abs(vseg2diffs[i,1])-1>1e-5:
                typeb=0
            elif typeb==2 and abs(vseg2diffs[i,1])-1/2>1e-5:
                typeb=0
            elif typeb==3 and abs(vseg2diffs[i,1])-1/2>1e-5:
                typeb=0
            
                        
            #Use of etable condition:
            #Nx=0
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2 and\
                typea!=0 and typeb!=0:
                #print(dx,dy,dz)
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz
                    
                if typea==1 and typeb==1:
                    if etable[int(dx),int(2*dy),int(dz),1,1]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),1,1]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),1,1]=W
                    
                elif typea==1 and typeb==2:
                    if etable[int(dx),int(2*dy),int(dz),1,2]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),1,2]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),1,2]=W
                        
                elif typea==1 and typeb==3:
                    if etable[int(dx),int(2*dy),int(dz),1,3]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),1,3]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),1,3]=W
                    
                elif typea==2 and typeb==1:
                    if etable[int(dx),int(2*dy),int(dz),2,1]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),2,1]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),2,1]=W

                elif typea==2 and typeb==2:
                    if etable[int(dx),int(2*dy),int(dz),2,2]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),2,2]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),2,2]=W

                elif typea==2 and typeb==3:
                    if etable[int(dx),int(2*dy),int(dz),2,3]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),2,3]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),2,3]=W

                elif typea==3 and typeb==1:
                    if etable[int(dx),int(2*dy),int(dz),3,1]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),3,1]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),3,1]=W

                elif typea==3 and typeb==2:
                    if etable[int(dx),int(2*dy),int(dz),3,2]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),3,2]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),3,2]=W

                elif typea==3 and typeb==3:
                    if etable[int(dx),int(2*dy),int(dz),3,3]!=0:
                        Eint+=sign1*sign2*etable[int(dx),int(2*dy),int(dz),3,3]
                    else:
                        W=enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                        Eint+=sign1*sign2*W
                        etable[int(dx),int(2*dy),int(dz),3,3]=W
                        
            else:
                
                Eint+=sign1*sign2*enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)
                


    #Horizontal-Vertical segment interactions

    if b[1]==0 and b[2]==0: #No interaction between edge and screw segments
        #save('etablesegsN%s'%N,etable)
        return Eint

    else:
        for i in range(Nv2):
            for j in range(N1):
                dxhv=(vseg2[2*i,0]-hseg1[2*j,0])*a
                dxhv-=round(dxhv/N/a)*N*a                       #PBC shift (minimum image convention)
                dyhv=(min(vseg2[[2*i,2*i+1],1])-hseg1[2*j,1])*h
                ydiff=abs(vseg2diffs[i,1])*h

                if round(vseg2diffs[i,2])==0:
                    dzhv=(min(vseg2[[2*i,2*i+1],2])-hseg1[2*j,2])*h
                    zdiff=0
                elif 2*vseg2diffs[i,1]== round(2*vseg2diffs[i,2]/sqrt(3)):
                    zdiff= abs(vseg2diffs[i,2])*h
                    dzhv=(min(vseg2[[2*i,2*i+1],2])-hseg1[2*j,2])*h
                elif 2*vseg2diffs[i,1]==-round(2*vseg2diffs[i,2]/sqrt(3)):
                    zdiff=-abs(vseg2diffs[i,2])*h
                    dzhv=(max(vseg2[[2*i,2*i+1],2])-hseg1[2*j,2])*h
                else:
                    print(vseg2diffs[i,:],'\n\n',vseg2[[2*i,2*i+1],:])
                    
                r1=r1a
                r2=array([dxhv,dyhv,dzhv])
                r3=array([dxhv,dyhv+ydiff,dzhv+zdiff])

                sign2=sign(vseg2diffs[i,1])
                                      
                Eint+=sign2*enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)

        for i in range(N2):
            for j in range(Nv1):
                dxvh=(vseg1[2*j,0]-hseg2[2*i,0])*a
                dxvh-=round(dxvh/N/a)*N*a                       #PBC shift (minimum image convention)
                dyvh=(min(vseg1[[2*j,2*j+1],1])-hseg2[2*i,1])*h

                if round(vseg1diffs[j,2])==0:
                    ydiff=h
                    dzvh=(min(vseg1[[2*j,2*j+1],2])-hseg2[2*i,2])*h
                    zdiff=0
                elif 2*vseg1diffs[j,1]== round(2*vseg1diffs[j,2]/sqrt(3)):
                    ydiff=h/2
                    dzvh=(min(vseg1[[2*j,2*j+1],2])-hseg2[2*i,2])*h
                    zdiff=h*sqrt(3)/2
                elif 2*vseg1diffs[j,1]==-round(2*vseg1diffs[j,2]/sqrt(3)):
                    ydiff=h/2
                    dzvh=(max(vseg1[[2*j,2*j+1],2])-hseg2[2*i,2])*h
                    zdiff=-h*sqrt(3)/2
                else:
                    print(vseg1diffs[j,:],'\n\n',vseg1[[2*j,2*j+1],:])
                    
                r1=r1a
                r2=array([dxvh,dyvh,dzvh])
                r3=array([dxvh,dyvh+ydiff,dzvh+zdiff])

                sign1=sign(vseg1diffs[j,1])
                                      
                Eint+=sign1*enginteract(mu,nu,b,b,r0,r1,r2,r3,rc)

    #save('etablesegsN%s'%N,etable)       
    return Eint
                
    
