from numpy import array,zeros,sqrt,reshape,round,sign,dot,min,max,dtype

from stress import stress1


def linestress(mu,nu,b,hseg1,vseg1,hseg2,vseg2,a,h,rc,N,stable):
#Computes the stress tensor at the midpoint of each segment


    if stable.shape[0]!=0:
        if stable[0,0,0,0,0,0,0]!=0:
            if abs(stable[0,0,0,0,0,0,0]-stress1(mu,nu,b,array([a/2,0,0]),array([0,0,0]),array([a,0,0]),rc)[0,0])>1e-32:
                stable=zeros([2*N,24,24,4,4,3,3])
                print("parameters of stable differs from current ones")

    Nx=stable.shape[0]
    Ny=stable.shape[1]
    Nz=stable.shape[2]
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
    
    sh=zeros([N1,3,3])
    sv=zeros([Nv1,3,3])

    #Horizontal segment stresses
    #Compute the stress tensor at the centre of each hseg1 due to the hseg2s
    for i in range(N1):
        for j in range(N2):
            dx=hseg1[2*i,0]+1/2-hseg2[2*j,0]
            dx-=round(dx/N)*N
            dy=hseg1[2*i,1]-hseg2[2*j,1]
            dz=hseg1[2*i,2]-hseg2[2*j,2]
            r=array([dx*a,dy*h,dz*h])

            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz
                if stable[int(dx),int(2*dy),int(dz),0,0,:,:].any()!=0:
                    sh[i,:,:]+=stable[int(dx),int(2*dy),int(dz),0,0,:,:]
                else:
                    s=stress1(mu,nu,b,r,r0,r1a,rc)
                    sh[i,:,:]+=s
                    stable[int(dx),int(2*dy),int(dz),0,0,:,:]=s                        
            else:
                sh[i,:,:]+=stress1(mu,nu,b,r,r0,r1a,rc)
                
    #Add the stress due to the vseg2s
    for i in range(N1):
        for j in range(Nv2):
            sign2=sign(vseg2diffs[j,1])
            dx=hseg1[2*i,0]+1/2-vseg2[2*j,0]
            dx-=round(dx/N)*N
            dy=hseg1[2*i,1]-min(vseg2[[2*j,2*j+1],1])
            #Determine type of vseg
            if abs(vseg2diffs[j,2])<1e-5:
                typeb=1
                r1=r1h
                dz=hseg1[2*i,2]-min(vseg2[[2*j,2*j+1],2])
            elif abs(vseg2diffs[j,1]-vseg2diffs[j,2]/sqrt(3))<1e-5:
                typeb=2
                r1=r1hz
                dz=hseg1[2*i,2]-min(vseg2[[2*j,2*j+1],2])
            elif abs(vseg2diffs[j,1]+vseg2diffs[j,2]/sqrt(3))<1e-5:
                typeb=3
                r1=r1hzm
                dz=hseg1[2*i,2]-max(vseg2[[2*j,2*j+1],2])
            else:
                print("linestress error! vseg2diffs: %s" %vseg2diffs)
            #Position of hseg[i] relative to vseg2[j]
            r=array([dx*a,dy*h,dz*h])
            #Prepare dz for use in stable
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #Check that the vseg2 is a unit vseg
            if typeb==1 and abs(vseg2diffs[j,1])-1>1e-5:
                typeb=0
            elif typeb==2 and abs(vseg2diffs[j,1])-1/2>1e-5:
                typeb=0
            elif typeb==3 and abs(vseg2diffs[j,1])-1/2>1e-5:
                typeb=0
            #Use of etable condition:
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2 and typeb!=0:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if stable[int(dx),int(2*dy),int(dz),0,int(typeb),:,:].any()!=0:
                    sh[i,:,:]+=sign2*stable[int(dx),int(2*dy),int(dz),0,int(typeb),:,:]
                else:
                    s=stress1(mu,nu,b,r,r0,r1,rc)
                    sh[i,:,:]+=sign2*s
                    stable[int(dx),int(2*dy),int(dz),0,int(typeb),:,:]=s
            else:
                sh[i,:,:]+=sign2*stress1(mu,nu,b,r,r0,r1,rc)





    #Vertical segment stresses
    #Compute the stress tensor at the centre of each vseg1 due to the hseg2s
    for i in range(Nv1):
        for j in range(N2):
            #sign1=sign(vseg1diffs[i,1])
            dx=vseg1[2*i,0]-hseg2[2*j,0]
            dx-=round(dx/N)*N
            #Determine type of vseg
            if abs(vseg1diffs[i,2])<1e-5:
                typea=1
                dy=min(vseg1[[2*i,2*i+1],1])+1/2-hseg2[2*j,1]
                dz=min(vseg1[[2*i,2*i+1],2])-hseg2[2*j,2]
            elif abs(vseg1diffs[i,1]-vseg1diffs[i,2]/sqrt(3))<1e-5:
                typea=2
                dy=min(vseg1[[2*i,2*i+1],1])+1/4-hseg2[2*j,1]
                dz=min(vseg1[[2*i,2*i+1],2])+sqrt(3)/4-hseg2[2*j,2]
            elif abs(vseg1diffs[i,1]+vseg1diffs[i,2]/sqrt(3))<1e-5:
                typea=3
                dy=min(vseg1[[2*i,2*i+1],1])+1/4-hseg2[2*j,1]
                dz=max(vseg1[[2*i,2*i+1],2])-sqrt(3)/4-hseg2[2*j,2]
            else:
                print("linestress error! vseg1diffs: %s" %vseg1diffs)
            #Position of vseg[i] relative to hseg2[j]
            r=array([dx*a,dy*h,dz*h])
            #Prepare dz for use in stable
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #Check that the vseg1 is a unit vseg
            if typea==1 and abs(vseg1diffs[i,1])-1>1e-5:
                typea=0
            elif typea==2 and abs(vseg1diffs[i,1])-1/2>1e-5:
                typea=0
            elif typea==3 and abs(vseg1diffs[i,1])-1/2>1e-5:
                typea=0
            #Use of etable condition:
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2 and typea!=0:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if stable[int(dx),int(2*dy),int(dz),int(typea),0,:,:].any()!=0:
                    sv[i,:,:]+=stable[int(dx),int(2*dy),int(dz),int(typea),0,:,:]
                else:
                    s=stress1(mu,nu,b,r,r0,r1a,rc)
                    sv[i,:,:]+=s
                    stable[int(dx),int(2*dy),int(dz),int(typea),0,:,:]=s
            else:
                sv[i,:,:]+=stress1(mu,nu,b,r,r0,r1a,rc)

    #Add the stress due to the vseg2s
    for i in range(Nv1):
        for j in range(Nv2):
            #sign1=sign(vseg1diffs[i,1])
            sign2=sign(vseg2diffs[j,1])                       
            dx=vseg1[2*i,0]-vseg2[2*j,0]
            dx-=round(dx/N)*N
            #Determine types of vsegs
            if abs(vseg1diffs[i,2])<1e-5:
                typea=1
                dy=min(vseg1[[2*i,2*i+1],1])+1/2-min(vseg2[[2*j,2*j+1],1])
            elif abs(vseg1diffs[i,1]-vseg1diffs[i,2]/sqrt(3))<1e-5:
                typea=2
                dy=min(vseg1[[2*i,2*i+1],1])+1/4-min(vseg2[[2*j,2*j+1],1])
            elif abs(vseg1diffs[i,1]+vseg1diffs[i,2]/sqrt(3))<1e-5:
                typea=3
                dy=min(vseg1[[2*i,2*i+1],1])+1/4-min(vseg2[[2*j,2*j+1],1])
            else:
                print("linestress error! vseg1diffs: %s" %vseg1diffs)
            if abs(vseg2diffs[j,2])<1e-5:
                typeb=1
                r1=r1h
            elif abs(vseg2diffs[j,1]-vseg2diffs[j,2]/sqrt(3))<1e-5:
                typeb=2
                r1=r1hz
            elif abs(vseg2diffs[j,1]+vseg2diffs[j,2]/sqrt(3))<1e-5:
                typeb=3
                r1=r1hzm
            else:
                print("linestress error! vseg2diffs: %s" %vseg2diffs)
            #Determine dz
            if typea==1 and (typeb==1 or typeb==2):
                dz=min(vseg1[[2*i,2*i+1],2])-min(vseg2[[2*j,2*j+1],2])
            elif typea==1 and typeb==3:
                dz=min(vseg1[[2*i,2*i+1],2])-max(vseg2[[2*j,2*j+1],2])
            elif typea==2 and (typeb==1 or typeb==2):
                dz=min(vseg1[[2*i,2*i+1],2])+sqrt(3)/4-min(vseg2[[2*j,2*j+1],2])
            elif typea==2 and typeb==3:
                dz=min(vseg1[[2*i,2*i+1],2])+sqrt(3)/4-max(vseg2[[2*j,2*j+1],2])
            elif typea==3 and (typeb==1 or typeb==3):
                dz=max(vseg1[[2*i,2*i+1],2])-sqrt(3)/4-max(vseg2[[2*j,2*j+1],2])
            elif typea==3 and typeb==2:
                dz=max(vseg1[[2*i,2*i+1],2])-sqrt(3)/4-min(vseg2[[2*j,2*j+1],2])
            else:
                print("Error!")
                break
            #Position of vseg1[i] relative to vseg2[j]
            r=array([dx*a,dy*h,dz*h])
            #Prepare dz for use in stable
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #Check that the vsegs are unit vsegs
            if typea==1 and abs(vseg1diffs[i,1])-1>1e-5:
                typea=0
            elif typea==2 and abs(vseg1diffs[i,1])-1/2>1e-5:
                typea=0
            elif typea==3 and abs(vseg1diffs[i,1])-1/2>1e-5:
                typea=0
            if typeb==1 and abs(vseg2diffs[j,1])-1>1e-5:
                typeb=0
            elif typeb==2 and abs(vseg2diffs[j,1])-1/2>1e-5:
                typeb=0
            elif typeb==3 and abs(vseg2diffs[j,1])-1/2>1e-5:
                typeb=0
            #Use of etable condition:
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2 and typea!=0 and typeb!=0:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if stable[int(dx),int(2*dy),int(dz),int(typea),int(typeb),:,:].any()!=0:
                    sv[i,:,:]+=sign2*stable[int(dx),int(2*dy),int(dz),int(typea),int(typeb),:,:]
                else:
                    #print("r is",r)
                    #print("sign2 is",sign2)
                    #print("r1 is",r1)
                    s=stress1(mu,nu,b,r,r0,r1,rc)
                    sv[i,:,:]+=sign2*s
                    stable[int(dx),int(2*dy),int(dz),int(typea),int(typeb),:,:]=s
            else:
                sv[i,:,:]+=sign2*stress1(mu,nu,b,r,r0,r1,rc)


    return sh,sv


