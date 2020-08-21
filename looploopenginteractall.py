from numpy import array,zeros,sqrt,empty,round,dtype

from segsenginteract import segsenginteract

from loopselfenergy import loopselfenergy

def looploopenginteractall(mu,nu,b,a,h,rc,change,yz,etable,etablesegs):

    if etable.shape[0]!=0:
        if abs(etable[0,0,0,0,0]/2-loopselfenergy(mu,nu,b,a,h,rc)[0])>1e-32:
            etable=empty([0,0,0,3,3])

    Nx=etable.shape[0]
    Ny=etable.shape[1]
    Nz=etable.shape[2]
    #print(Nx,Ny,Nz)

    
#For updating the energy differences between dislocation line configurations that differ by a single segment.
#This update is equal to the interaction energy of the incremental loop that causes the initial energy difference with
#another incremental loop, which accounts for a new configuration change. The new incremental loop is placed
#on the horizontal segment in the original configuration where the configurations differ. The new incremental loop cannot
#be placed at the position of the initial loop, which caused the original energy difference.
#Mu is the elastic shear modulus, Nu is Poisson's ratio, b is the burgers vector (bx,by,bz) (3,), coordh gives the start
#coordinates of the horizontal segments (Nx3) in the original configuation, changepos[0] gives the index of the
#configuration change position such that coordh[changepos[0],:] gives the start coordinate of the horizontal segment
#where the configuration changes and changepos[1]&[2] give the changes in y- & z-coordinates. 
#looppos is the start coordinate of the line segment where the loop is placed (i,j,k), yz is an array containing the y-
#rc is the dislocation radius parameter, 'a' is the lattice constant along the dislocation line, h is the kink height, 
#sigma is the applied stress (sigma(xz)) and N is the number of line segments.


    N=yz.shape[0]

    Eaboveydiff=dtype("float128")
    Ebelowydiff=dtype("float128")
    Eaboveypluszdiff=dtype("float128")
    Ebelowyminuszdiff=dtype("float128")
    Ebelowypluszdiff=dtype("float128")
    Eaboveyminuszdiff=dtype("float128")
    
    Eaboveydiff=0.0
    Ebelowydiff=0.0
    Eaboveypluszdiff=0.0
    Ebelowyminuszdiff=0.0
    Ebelowypluszdiff=0.0
    Eaboveyminuszdiff=0.0
    energydiffsdiffs=zeros([N,6],dtype="float128")


#Calculate the interaction energy (including self-energy) of the loop with all line segments:

    def loopabovey(i,j,k):  #An incremental loop on the (x,y) plane with its bottom left-hand corner at (i,j,k)
        return array([[i,j,k],[i,j+1,k],[i,j+1,k],[i+1,j+1,k], \
                      [i+1,j+1,k],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopbelowy(i,j,k):  #An incremental loop on the same plane with its top left-hand corner at (i,j,k)
        return array([[i+1,j-1,k],[i+1,j,k],[i,j-1,k],[i+1,j-1,k], \
                      [i,j,k],[i,j-1,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopaboveyplusz(i,j,k): #An incremental loop on an intersecting {110} 
        return array([[i,j,k],[i,j+1/2,k+sqrt(3)/2],[i,j+1/2,k+sqrt(3)/2],[i+1,j+1/2,k+sqrt(3)/2], \
                    [i+1,j+1/2,k+sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopbelowyminusz(i,j,k):    #An incremental loop below on the same {110} plane
        return array([[i+1,j-1/2,k-sqrt(3)/2],[i+1,j,k],[i,j-1/2,k-sqrt(3)/2],[i+1,j-1/2,k-sqrt(3)/2], \
                      [i,j,k],[i,j-1/2,k-sqrt(3)/2],[i+1,j,k],[i,j,k]],dtype="float128")

    def loopaboveyminusz(i,j,k):     #An incremental loop below on another {110} plane
        return array([[i,j,k],[i,j+1/2,k-sqrt(3)/2],[i,j+1/2,k-sqrt(3)/2],[i+1,j+1/2,k-sqrt(3)/2], \
                      [i+1,j+1/2,k-sqrt(3)/2],[i+1,j,k],[i+1,j,k],[i,j,k]],dtype="float128")
        
    def loopbelowyplusz(i,j,k):    #An incremental loop above on the same {110} plane
        return array([[i+1,j-1/2,k+sqrt(3)/2],[i+1,j,k],[i,j-1/2,k+sqrt(3)/2],[i+1,j-1/2,k+sqrt(3)/2], \
                      [i,j,k],[i,j-1/2,k+sqrt(3)/2],[i+1,j,k],[i,j,k]],dtype="float128")                        #Loops with below y are changed from those in loopenginteract.py with
                                                                                                                #the positive and negative v.segs swapping positions in the array.


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

        
    for i in range(N):    
        if i==change[0]:                                                                #energy diff. for the change pos. must be calculated separately
            continue
                                    
        if change[1]>0:                                                                 #positive loops in the change position

            #positive interacting loops
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]-yz[i,0]                                                  
            dz=yz[change[0],1]-yz[i,1]
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopabovey
                    Eaboveydiff=etable[int(dx),int(2*dy),int(dz),0,0]
                    Eaboveypluszdiff=etable[int(dx),int(2*dy),int(dz),1,0]
                    Eaboveyminuszdiff=etable[int(dx),int(2*dy),int(dz),2,0]
                elif change[2]>0:                                                       #loop() is a loopaboveyplusz
                    Eaboveydiff=etable[int(dx),int(2*dy),int(dz),0,1]
                    Eaboveypluszdiff=etable[int(dx),int(2*dy),int(dz),1,1]
                    Eaboveyminuszdiff=etable[int(dx),int(2*dy),int(dz),2,1]
                elif change[2]<0:                                                       #loop() is a loopaboveyminusz
                    Eaboveydiff=etable[int(dx),int(2*dy),int(dz),0,2]
                    Eaboveypluszdiff=etable[int(dx),int(2*dy),int(dz),1,2]
                    Eaboveyminuszdiff=etable[int(dx),int(2*dy),int(dz),2,2]                    

            else:
                #print("etable not used!")
                Eaboveydiff=segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[2,3],:],loopabovey(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[7,6],:],loopabovey(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[2,3],:],loopabovey(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[7,6],:],loopabovey(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                Eaboveypluszdiff=segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[2,3],:],loopaboveyplusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[7,6],:],loopaboveyplusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[2,3],:],loopaboveyplusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[7,6],:],loopaboveyplusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                Eaboveyminuszdiff=segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[2,3],:],loopaboveyminusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[7,6],:],loopaboveyminusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[2,3],:],loopaboveyminusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[7,6],:],loopaboveyminusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)  #Add the interaction energies when both sets
                                                                                                                                                                        #of segments are positive or negative and subtract
                                                                                                                                                                        #when their signs differ 
            #loopbelowy interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]-(yz[i,0]-1)
            dz=yz[change[0],1]-yz[i,1]
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopabovey
                    Ebelowydiff=-etable[int(dx),int(2*dy),int(dz),0,0]
                elif change[2]>0:                                                       #loop() is a loopaboveyplusz
                    Ebelowydiff=-etable[int(dx),int(2*dy),int(dz),0,1]
                elif change[2]<0:                                                       #loop() is a loopaboveyminusz
                    Ebelowydiff=-etable[int(dx),int(2*dy),int(dz),0,2]    
            else:
                #print("etable not used!")
                Ebelowydiff=segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[2,3],:],loopbelowy(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[7,6],:],loopbelowy(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[2,3],:],loopbelowy(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[7,6],:],loopbelowy(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)

            #loopbelowyminusz interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]-(yz[i,0]-1/2)
            dz=yz[change[0],1]-(yz[i,1]-sqrt(3)/2)
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz
                if change[2]==0:                                                        #loop() is a loopabovey
                    Ebelowyminuszdiff=-etable[int(dx),int(2*dy),int(dz),1,0]
                elif change[2]>0:                                                       #loop() is a loopaboveyplusz
                    Ebelowyminuszdiff=-etable[int(dx),int(2*dy),int(dz),1,1]
                elif change[2]<0:                                                       #loop() is a loopaboveyminusz
                    Ebelowyminuszdiff=-etable[int(dx),int(2*dy),int(dz),1,2]
            else:
                #print("etable not used!")
                Ebelowyminuszdiff=segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[2,3],:],loopbelowyminusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[7,6],:],loopbelowyminusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[2,3],:],loopbelowyminusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[7,6],:],loopbelowyminusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                
            #loopbelowyplusz interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]-(yz[i,0]-1/2)
            dz=yz[change[0],1]-(yz[i,1]+sqrt(3)/2)
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopabovey
                    Ebelowypluszdiff=-etable[int(dx),int(2*dy),int(dz),2,0]
                elif change[2]>0:                                                       #loop() is a loopaboveyplusz
                    Ebelowypluszdiff=-etable[int(dx),int(2*dy),int(dz),2,1]
                elif change[2]<0:                                                       #loop() is a loopaboveyminusz
                    Ebelowypluszdiff=-etable[int(dx),int(2*dy),int(dz),2,2]
            else:
                #print("etable not used!")
                Ebelowypluszdiff=segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[2,3],:],loopbelowyplusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[7,6],:],loopbelowyplusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[2,3],:],loopbelowyplusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[7,6],:],loopbelowyplusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                    


        elif change[1]<0:                                                               #negative loops in the change position

            #positive interacting loops
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]+change[1]-yz[i,0]
            dz=yz[change[0],1]+change[2]-yz[i,1]
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz
                #print("dx,dy,dz = ",dx,dy,dz)
                if change[2]==0:                                                        #loop() is a loopbelowy
                    Eaboveydiff=-etable[int(dx),int(2*dy),int(dz),0,0]
                    Eaboveypluszdiff=-etable[int(dx),int(2*dy),int(dz),1,0]
                    Eaboveyminuszdiff=-etable[int(dx),int(2*dy),int(dz),2,0]
                elif change[2]<0:                                                       #loop() is a loopbelowyminusz
                    Eaboveydiff=-etable[int(dx),int(2*dy),int(dz),0,1]
                    Eaboveypluszdiff=-etable[int(dx),int(2*dy),int(dz),1,1]
                    Eaboveyminuszdiff=-etable[int(dx),int(2*dy),int(dz),2,1]
                elif change[2]>0:                                                       #loop() is a loopbelowyplusz
                    Eaboveydiff=-etable[int(dx),int(2*dy),int(dz),0,2]
                    Eaboveypluszdiff=-etable[int(dx),int(2*dy),int(dz),1,2]
                    Eaboveyminuszdiff=-etable[int(dx),int(2*dy),int(dz),2,2]                    

            else:
                #print("etable not used!")
                Eaboveydiff=segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[2,3],:],loopabovey(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[7,6],:],loopabovey(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[2,3],:],loopabovey(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopabovey(i,*yz[i,:])[[7,6],:],loopabovey(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                Eaboveypluszdiff=segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[2,3],:],loopaboveyplusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[7,6],:],loopaboveyplusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[2,3],:],loopaboveyplusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopaboveyplusz(i,*yz[i,:])[[7,6],:],loopaboveyplusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)
                Eaboveyminuszdiff=segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[2,3],:],loopaboveyminusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[7,6],:],loopaboveyminusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[2,3],:],loopaboveyminusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopaboveyminusz(i,*yz[i,:])[[7,6],:],loopaboveyminusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)

            #loopbelowy interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]+change[1]-(yz[i,0]-1)
            dz=yz[change[0],1]+change[2]-yz[i,1]
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopbelowy
                    Ebelowydiff=etable[int(dx),int(2*dy),int(dz),0,0]
                elif change[2]<0:                                                       #loop() is a loopbelowyminusz
                    Ebelowydiff=etable[int(dx),int(2*dy),int(dz),0,1]
                elif change[2]>0:                                                       #loop() is a loopbelowyplusz
                    Ebelowydiff=etable[int(dx),int(2*dy),int(dz),0,2]    
            else:
                #print("etable not used!")
                Ebelowydiff=segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[2,3],:],loopbelowy(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[7,6],:],loopbelowy(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                        -segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[2,3],:],loopbelowy(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                        +segsenginteract(mu,nu,b,loopbelowy(i,*yz[i,:])[[7,6],:],loopbelowy(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)

            #loopbelowyminusz interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]+change[1]-(yz[i,0]-1/2)
            dz=yz[change[0],1]+change[2]-(yz[i,1]-sqrt(3)/2)
            dz=dz/(sqrt(3)/2)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopbelowy
                    Ebelowyminuszdiff=etable[int(dx),int(2*dy),int(dz),1,0]
                elif change[2]<0:                                                       #loop() is a loopbelowyminusz
                    Ebelowyminuszdiff=etable[int(dx),int(2*dy),int(dz),1,1]
                elif change[2]>0:                                                       #loop() is a loopbelowyplusz
                    Ebelowyminuszdiff=etable[int(dx),int(2*dy),int(dz),1,2]
            else:
                #print("etable not used!")
                Ebelowyminuszdiff=segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[2,3],:],loopbelowyminusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[7,6],:],loopbelowyminusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[2,3],:],loopbelowyminusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopbelowyminusz(i,*yz[i,:])[[7,6],:],loopbelowyminusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)

            #loopbelowyplusz interacting
            dx=change[0]-i
            dx-=round(dx/N)*N
            dy=yz[change[0],0]+change[1]-(yz[i,0]-1/2)
            dz=yz[change[0],1]+change[2]-(yz[i,1]+sqrt(3)/2)
            dz=2*dz/sqrt(3)
            dz=round(dz)
            #print("dx,dy,dz = ",dx,dy,dz)
            if abs(dx)<Nx/2 and abs(dy)<Ny/4 and abs(dz)<Nz/2:
                if dx<0:
                    dx+=Nx
                if dy<0:
                    dy+=Ny/2
                if dz<0:
                    dz+=Nz                    
                if change[2]==0:                                                        #loop() is a loopbelowy
                    Ebelowypluszdiff=etable[int(dx),int(2*dy),int(dz),2,0]
                elif change[2]<0:                                                       #loop() is a loopbelowyminusz
                    Ebelowypluszdiff=etable[int(dx),int(2*dy),int(dz),2,1]
                elif change[2]>0:                                                       #loop() is a loopbelowyplusz
                    Ebelowypluszdiff=etable[int(dx),int(2*dy),int(dz),2,2]
            else:
                #print("etable not used!")
                Ebelowypluszdiff=segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[2,3],:],loopbelowyplusz(i,*yz[i,:])[[0,1],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[7,6],:],loopbelowyplusz(i,*yz[i,:])[[5,4],:],loop()[[2,3],:],loop()[[0,1],:],a,h,rc,N,etablesegs)\
                            -segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[2,3],:],loopbelowyplusz(i,*yz[i,:])[[0,1],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)\
                            +segsenginteract(mu,nu,b,loopbelowyplusz(i,*yz[i,:])[[7,6],:],loopbelowyplusz(i,*yz[i,:])[[5,4],:],loop()[[7,6],:],loop()[[5,4],:],a,h,rc,N,etablesegs)

    
        energydiffsdiffs[i,0]=Eaboveydiff
        energydiffsdiffs[i,1]=Ebelowydiff
        energydiffsdiffs[i,2]=Eaboveypluszdiff
        energydiffsdiffs[i,3]=Ebelowyminuszdiff
        energydiffsdiffs[i,4]=Eaboveyminuszdiff
        energydiffsdiffs[i,5]=Ebelowypluszdiff



    return energydiffsdiffs

            

        
          
