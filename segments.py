from numpy import hstack,reshape,array,vstack,concatenate,where,ceil,arange,delete,insert,abs,sqrt,append,round,dtype,unique,zeros,copy,mean



def segments(yz,coordv,segmentplane,change,N):
        
    if yz.shape[0]!=N:
        yz11=copy(yz)
        yorig=mean(yz11[:,0])
        zorig=mean(yz11[:,1])
        N=int(N)
        yz=zeros([N,2])
        segmentplane=zeros([N,])
        kk=0
        if len(coordv)>0:
            xlist=unique(coordv[:,0],axis=0)
            for i in range(len(xlist)):
                ww=where(coordv[::2,0]==xlist[i])[0]
                ww=ww[0]
                l=int(coordv[2*ww,0])
                yl=coordv[2*ww,1]
                zl=coordv[2*ww,2]
                sp=coordv[2*ww,4]
                yz[kk:l,0]=yl
                yz[kk:l,1]=zl
                segmentplane[kk:l]=sp
                kk=l
                if i==len(xlist)-1 and xlist[-1]!=N:
                    l=int(N-1)
                    yl=yz[0,0]
                    zl=yz[0,1]
                    sp=segmentplane[0]
                    yz[kk:N,0]=yl
                    yz[kk:N,1]=zl
                    segmentplane[kk:N]=sp
        else:
            yz[:,0]=yorig
            yz[:,1]=zorig

        yzv=hstack([yz,yz]).reshape([2*N,2])        #Coordinate y-,z-values as a column vector
        yzh=yzv.T                                   #Coordinate y-,z-values as a row vector
        xh=array([ceil(arange(0,N,0.5))])           #Coordinate x-values as a row vector
        coordh=vstack([xh,yzh]).T                   #Coordinates of horizontal segments
        
        return yz,coordh,coordv,segmentplane

   
    yz[change[0],0]=yz[change[0],0]+change[1]   #Update y- and z-coordinates of horizontal segments
    yz[change[0],1]=yz[change[0],1]+change[2]
    if abs(yz[change[0],1])<1e-5:
        yz[change[0],1]=0

    segmentplaneold=copy(segmentplane[change[0]])
    segmentplane[change[0]]=change[3]                   #Update segmentplane 
    

    yzv=hstack([yz,yz]).reshape([2*N,2])        #Coordinate y-,z-values as a column vector
    yzh=yzv.T                                   #Coordinate y-,z-values as a row vector
    xh=array([ceil(arange(0,N,0.5))])           #Coordinate x-values as a row vector

    coordh=vstack([xh,yzh]).T                   #Coordinates of horizontal segments
    coordhspts=coordh[::2,:]                    #Start coordinates, only, of horizontal segments##########################################
    
    ri=concatenate([arange(1,N),[0]])           #Indices of ith segment right neighbour
    li=concatenate([[N-1],arange(N-1)])         #Indices of ith segment left neighbour
    ydiffr=(yz-yz[ri])                          #Difference in y-,z-values of ith segment with right neighbours                                 
    ydiffl=(yz-yz[li])                          #Difference in y-,z-values of ith segment with left neighbours
    

    

    def ndkink():
        if change[0]==0:
            return      array([[coordhspts[N-1,0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0],segmentplaneold],\
                               [coordhspts[N-1,0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0]+1,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0]+1,segmentplaneold]])
        elif change[0]==N-1:
            return      array([[coordhspts[change[0],0],coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0],segmentplaneold],\
                               [coordhspts[change[0],0],coordhspts[change[0],1],coordhspts[change[0],2],change[0],change[3]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],0,change[3]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],0,segmentplaneold]])
        else:
            return      array([[coordhspts[change[0],0],coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0],segmentplaneold],\
                               [coordhspts[change[0],0],coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0]+1,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0]+1,segmentplaneold]])

    def nlkink():
        if change[0]==0:
            return      array([[coordhspts[N-1,0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0],segmentplaneold],\
                               [coordhspts[N-1,0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]]])
        else:
            return      array([[coordhspts[change[0],0],coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0],segmentplaneold],\
                               [coordhspts[change[0],0],coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]]])
    

    
    def nlkinkrn():
        if change[0]==N-1:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],0,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0]+1,1],coordhspts[change[0]+1,2],0,segmentplane[change[0]]]])
        else:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0]+1,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0]+1,1],coordhspts[change[0]+1,2],change[0]+1,segmentplane[change[0]]]])
        
    
    def nrkink():
        if change[0]==N-1:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],0,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],0,segmentplaneold]])
        else:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0]+1,segmentplane[change[0]]],\
                               [coordhspts[change[0],0]+1,coordhspts[change[0],1]-change[1],coordhspts[change[0],2]-change[2],change[0]+1,segmentplaneold]])

    def nendposlhs():
        if change[0]==0:
            return      array([[coordhspts[N-1,0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]]])
        else:
            return      array([[coordhspts[change[0],0],coordhspts[change[0],1],coordhspts[change[0],2],change[0],segmentplane[change[0]]]])

    def nstartposrhs():
        if change[0]==N-1:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],0,segmentplane[change[0]]]])
        else:
            return      array([[coordhspts[change[0],0]+1,coordhspts[change[0],1],coordhspts[change[0],2],change[0]+1,segmentplane[change[0]]]])


    def dkink():
        return          hstack([x1[y1]-1,x1[y1],x2[y2],x2[y2]+1])       #double kink indices

    nseg=change[0]+1

    w=where(coordv[:,0]>coordhspts[change[0],0])[0]                
  
    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                   #Vertical coordinates at the x-position of the current segment's beginning (LHS).    
    y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        #Of these, the one at the seg's initial pos. (The end point of a v.seg. (or v.segs) coming from the LHS neighbour h.seg.)
    if y1.size>1:
        y1=y1[-1]
    y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]                 #Of these, the one at the seg's new pos.
    if y1a.size>1:
        y1a=y1a[-1]
    
    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]#+a              #Vertical coordinates at the x-position of the current seg's end (RHS)    
    y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]        #Of these, the one at the seg's initial pos. (The start point of a v.seg. going to the RHS neighbour h.seg.)
    if y2.size>1:
        y2=y2[0]
    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]                 #Of these, the one at the seg's new pos.
    if y2a.size>1:
        y2a=y2a[0]

    #print("x1 is: ",x1)
    #print("y1 is: ",y1,'\n\n')
    #print("x2 is: ",x2) 
    #print("y2 is: ",y2,'\n\n')


    def p0(): 
        return abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])              #If 0, the h.seg. has moved to the same 'a' plane as its LHS v.seg's start pos; i.e. it has gone backwards to its previous plane
    
    def p1a():
        return 2*coordv[x1[y1]-1,1]-round(2*coordv[x1[y1]-1,2]/sqrt(3))     #If p1a=p1b, then as above, but with the h.seg. moving to the same 'b' plane

    def p1b():
        return 2*coordhspts[change[0],1]-round(2*coordhspts[change[0],2]/sqrt(3))
    
    def p2a(): 
        return 2*coordv[x1[y1]-1,1]+round(2*coordv[x1[y1]-1,2]/sqrt(3))      #If p2a=p2b, then as above, but with the h.seg. moving to the same 'c' plane

    def p2b():
        return 2*coordhspts[change[0],1]+round(2*coordhspts[change[0],2]/sqrt(3))
 
    def q0():
        return abs(coordv[x2[y2]+1,2]-coordhspts[change[0],2])              #If 0, the h.seg. has moved to the same 'a' plane as its RHS v.seg's end pos.

    def q1a():
        return 2*coordv[x2[y2]+1,1]-round(2*coordv[x2[y2]+1,2]/sqrt(3))     #If q1a=q1b, then as above, but with the h.seg. moving to the same 'b' plane

    def q1b():
        return 2*coordhspts[change[0],1]-round(2*coordhspts[change[0],2]/sqrt(3))

    def q2a():
        return 2*coordv[x2[y2]+1,1]+round(2*coordv[x2[y2]+1,2]/sqrt(3))     #If q2a=q2b, then as above, but with the h.seg. moving to the same 'c' plane

    def q2b():
        return 2*coordhspts[change[0],1]+round(2*coordhspts[change[0],2]/sqrt(3))

    def ydifflhs():
        return coordv[x1[y1],1]-coordv[x1[y1]-1,1]

    def zdifflhs():
        return coordv[x1[y1],2]-coordv[x1[y1]-1,2]

    def wherewhat(coords):
        if w.size!=0:                                                   #<If there are v.coords at a greater x-value>:
            vcoord=insert(coordv,w[0],coords,0)                         #Insert the coords before the 1st one
        else:                                                           #<If there aren't v.coords at a greater x-value>:
            vcoord=append(coordv,coords,0)                              #Insert the coords at the end of the array
        return vcoord
        

    
    if change[1]==0:
        return yz,coordh,coordv,segmentplane


##################################################################################################################################################################################################
###Combine this section with the next to avoid repetition

    if change[0]==0:                                                #If the change is to the first h.seg, LHS v.segs are the RHS v.segs of the last h.seg
        
        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
        if y1.size>1:
            y1=y1[-1]
        y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]         
        if y1a.size>1:
            y1a=y1a[-1]
        

        if coordv.size==0:
            coordv=insert(coordv,0,nrkink(),0)
            coordv=append(coordv,nlkink(),0)
            return yz,coordh,coordv,segmentplane



###<If the change causes kink annihilation>:
        
        if (abs(ydiffl[change[0]])<1e-5).all() and (abs(ydiffr[change[0]])<1e-5).all() and\
           x1.size<3 and x2.size<3:
           #change[3]==segmentplane[N-1] and change[3]==segmentplane[1]:#change 21/02
            coordv=delete(coordv,hstack([x1,x2]),0)
            segmentplane[change[0]]=segmentplane[nseg]
            return yz,coordh,coordv,segmentplane    


###<If the seg. doesn't change plane>:
            
        if change[3]==segmentplaneold:                      
            if x1.size==0:                                                  #<If there are no existing LHS v.coords>:
            
                if x2.size==0:                                              #<If there are also no existing RHS v.coords>:
                    coordv=insert(coordv,0,nrkink(),0)                      #Insert a new right kink
                    coordv=append(coordv,nlkink(),0)                        #Add a new left kink
                
                elif (abs(ydiffr[change[0]])<1e-5).all() and\
                    x2.size<3:
                    segmentplane[change[0]]=segmentplane[change[0]+1]       #<If there is kink migration left (RHS v.coords, but no LHS ones)>: change 21/01
                    coordv=delete(coordv,x2,0)                              #Delete RHS vertical coordinates 
                    coordv=append(coordv,nlkink(),0)                        #Add a new left kink

                elif coordv[x2[y2]+1,1]==coordhspts[change[0],1] and\
                     abs(coordv[x2[y2]+1,2]-coordhspts[change[0],2])<1e-5:  #<If the seg. moves to the end of a RHS vertical (meaning that a new vertical starts next on a new plane)>:
                    #print('hello')
                    coordv=delete(coordv,[x2[y2],x2[y2]+1],0)               #Delete the RHS vertical
                    coordv=append(coordv,nlkink(),0)                          #Add a new left kink
                    x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                    y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                    y2a=y2a[0]
                    segmentplane[change[0]]=coordv[x2[y2a],4]
                    coordv[x1[y1a],4]=coordv[x2[y2a],4]
                   
                elif coordv[x2[y2]+1,4]!=change[3]:                         #<If the end point of the existing RHS v.coord is on another plane>:
                    coordv=insert(coordv,x2[y2],nrkink(),0)                   #Insert a new right kink before the existing RHS v.coord start point
                    coordv=append(coordv,nlkink(),0)                          #Add a new left kink
                    
                else:
                    coordv=delete(coordv,x2[y2],0)                          #Otherwise, delete the old RHS v.seg start point
                    coordv=insert(coordv,x2[y2],nstartposrhs(),0)             #Insert a new one in its place
                    coordv=append(coordv,nlkink(),0)                          #Add a new left kink
                return yz,coordh,coordv,segmentplane
                    

                
            else:                                                           #<If LHS v.coords exist>:
           
                if x2.size==0:                                              #<But no RHS v.coords exist>:
                
                    if (abs(ydiffl[change[0]])<1e-5).all() and\
                       x1.size<3:
                        #print('here')
                        segmentplane[change[0]]=segmentplane[N-1]           #<If there is kink migration right>: 21/02
                        coordv=delete(coordv,x1,0)                          #Delete LHS v.coords
                        coordv=insert(coordv,0,nrkink(),0)                  #Insert a new right kink

                    
                    elif coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                         abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:  #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                        coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)               #Delete the LHS vertical
                        coordv=insert(coordv,0,nrkink(),0)                      #Insert a new right kink
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                        y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                        y1a=y1a[-1]
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                        y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]                    
                        segmentplane[change[0]]=coordv[x1[y1a],4]
                        coordv[x2[y2a],4]=coordv[x1[y1a],4]
                
                    elif coordv[x1[y1]-1,4]!=change[3]:                     #<If the existing LHS v.seg. comes from another plane>:
                        if (abs(zdifflhs())<1e-5 and segmentplane[change[0]]==0) or\
                           (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and segmentplane[change[0]]==1) or\
                           (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and segmentplane[change[0]]==2):
                            #print("new here!")
                            coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                            coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place    
                            coordv=insert(coordv,0,nrkink(),0)                  #Insert a new right kink
                        else:                        
                            coordv=insert(coordv,0,nrkink(),0)                    #Insert a new right kink
                            coordv=append(coordv,nlkink(),0)                      #Add a new left kink
                    
                    else:                                                   #<The existing LHS v.seg. doesn't come from another plane or is after another (no RHS v.segs)
                        coordv=delete(coordv,x1[y1],0)                      #Delete the old LHS v.seg. end point
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)           #Insert a new one in its place
                        coordv=insert(coordv,0,nrkink(),0)                    #Insert a new right kink
                    return yz,coordh,coordv,segmentplane

                
                else:                                                       #<LHS and RHS v.segs exist>:
                    #print("L and R vsegs")

                
                    if (abs(ydiffr[change[0]])<1e-5).all() and\
                       x2.size<3:
                       #change[3]==segmentplane[1]:                          #<If there is kink migration left>: 21/02
                        segmentplane[change[0]]=segmentplane[change[0]+1]
                        coordv=delete(coordv,x2,0)                          #Delete RHS vertical coordinates
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                        if y1.size>1:
                            y1=y1[-1]
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]            
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]

                    elif coordv[x2[y2]+1,1]==coordhspts[change[0],1] and\
                     abs(coordv[x2[y2]+1,2]-coordhspts[change[0],2])<1e-5:          #<If the seg. moves to the end of a RHS vertical (meaning that a new vertical starts next on a new plane)>:
                        #print('here')
                        if coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                           abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:    #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                            #print('here')
                            coordv=delete(coordv,[x1[y1]-1,x1[y1],x2[y2],x2[y2]+1],0)
                            #x1=where(coordv[:,0]==coordhspts[change[0],0])[0]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                            y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                            y2a=y2a[0]
                            segmentplane[change[0]]=coordv[x2[y2a],4]
                            return yz,coordh,coordv,segmentplane
                        else:
                            #print('here')
                            coordv=delete(coordv,[x2[y2],x2[y2]+1],0)                   #Delete the RHS vertical
                            w=where(coordv[:,0]>coordhspts[change[0],0])[0]             #Reset w, x1, y1, x2, y2
                            x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]                    
                            y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                            if y1.size>1:
                                y1=y1[-1]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                            y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                            if y2.size>1:
                                y2=y2[0]                  
                            y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                            y2a=y2a[0]
                            segmentplane[change[0]]=coordv[x2[y2a],4]

                    elif coordv[x1[y1]-1,1]==coordv[x2[y2]+1,1] and\
                         abs(coordv[x1[y1]-1,2]-coordv[x2[y2]+1,2])<1e-5:   #<If the seg. is upon a double kink>:
                        coordv=delete(coordv,x2[y2],0)                      #Delete the existing RHS v.seg. start pos.
                        coordv=insert(coordv,x2[y2],nstartposrhs(),0)         #Insert a new one in its place
                        coordv=delete(coordv,x1[y1],0)                     #Delete the existing LHS v.seg. start pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)           #Insert a new one in its place                    
                        return yz,coordh,coordv,segmentplane

                    elif coordv[x2[y2]+1,4]!=change[3]:                     #<If the existing RHS v.seg. goes to another plane>:
##                        if coordv[x2[y2],4]==change[3]:
##                            print("here")
##                            coordv=delete(coordv,x2[y2],0)
##                            coordv=insert(coordv,x2[y2],nstartposrhs(),0)
##                        else:                           
                        coordv=insert(coordv,x2[y2],nrkink(),0)               #Insert a new right kink before it
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                        if y1.size>1:
                            y1=y1[-1]
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]             
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]
                        y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]                 #Of these, the one at the seg's new pos.
                        if y2a.size>1:
                            y2a=y2a[0]

                    else:
                        #print('here')
                        coordv=delete(coordv,x2[y2],0)                      #Otherwise delete the existing RHS vertical start point
                        coordv=insert(coordv,x2[y2],nstartposrhs(),0)         #Insert a new one in its place
                        w=where(coordv[:,0]>coordhspts[change[0],0])[0]     #Reset w, x1, y1, x2, y2
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                        if y1.size>1:
                            y1=y1[-1]
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]              
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]
                        y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                        y2a=y2a[0]
                        segmentplane[change[0]]=coordv[x2[y2a],4]

                    if (abs(ydiffl[change[0]])<1e-5).all() and\
                        x1.size<3:
                       #change[3]==segmentplane[N-1]:                        #<If there is kink migration right>: 21/02
                        segmentplane[change[0]]=segmentplane[N-1]
                        coordv=delete(coordv,x1,0)                          #Delete LHS v.coords
                        coordv[x2[y2a],4]=segmentplane[N-1]
                                  
                    elif coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                         abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:      #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                        #print('here')
                        coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)                   #Delete the LHS vertical
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]                    
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        if x1.size!=0 and x2.size!=0 and coordv[x1[-1],4]==coordv[x2[0],4]:
                            #print('there')
                            segmentplane[change[0]]=coordv[x1[-1],4]
                        elif (ydiffl[change[0]]==0).all() and\
                             (ydiffr[change[0]]==0).all() and\
                             segmentplane[N-1]==segmentplane[1]:
                            segmentplane[change[0]]=segmentplane[1]
                        elif yz[change[0],0]==0 and yz[change[0],1]==0:
                            segmentplane[change[0]]=0

##                    elif coordv[x1[y1]-1,1]==coordv[x2[y2]+1,1] and\
##                         coordv[x1[y1]-1,2]==coordv[x2[y2]+1,2]:            #<If the seg. is upon a double kink>:
##                        #print("upon a double kink after deleting the R vertical")
##                        coordv=delete(coordv,x1[y1],0)                      #Delete the existing LHS v.seg. end pos.
##                        coordv=insert(coordv,x1[y1],nendposlhs,0)           #Insert a new one in its place

                    elif coordv[x1[y1]-1,4]!=change[3]:         #<If the existing LHS v.seg. comes from another plane>:
                        if (abs(zdifflhs())<1e-5 and change[3]==0) or\
                           (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and change[3]==1) or\
                           (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and change[3]==2):
                            #print("new here!")
                            coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                            coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place    
                        else:
                            coordv=append(coordv,nlkink(),0)                    #Insert a new left kink after it
                   
                    else:
                        #print('here too')
                        #print(x1,y1)
                        coordv=delete(coordv,x1[y1],0)                      #Delete the existing LHS vertical endpoint
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)           #Insert a new on in its place
                        x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                        y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                        if y1a.size>1:
                            y1a=y1a[-1]
                        coordv[x1[y1a],4]=segmentplane[change[0]]

                    return yz,coordh,coordv,segmentplane
                
               
###<If the segment changes plane>:
                
        else:
            #print('hello')
            if x1.size==0:                                                      #<No LHS v.segs>:

                if x2.size==0:                                                  #<No RHS v.segs>:
                    coordv=insert(coordv,0,nrkink(),0)                            #Insert a new right kink
                    coordv=append(coordv,nlkink(),0)                              #Add a new left kink

                elif (abs(ydiffr[change[0]])<1e-5).all() and\
                    x2.size<3:
                       #change[3]==segmentplane[1]:                              #<If there is kink migration left>: 21/02
                    coordv=delete(coordv,x2,0)                                  #Delete RHS v.segs
                    coordv=append(coordv,nlkink(),0)                              #Insert a new left kink

            #Consider making it so that when a segment changes to the same plane as one of its neighbours,
            #delete all verticals and replace with a single one joining them, rather than only deleting the verticals when ydiffl or ydiffr are zero:
            #elif change[3]==segmentplane[change[0]+1] and\
                #(change[3]==0 and s0==0) or\
                #(change[3]==1 and s1a==s1b) or\
                #(change[3]==2 and s2a==s2b):
                #coordv=delete(coordv,x2,0)
                #w=where(coordv[:,0]>coordhspts[change[0],0])[0]             #Reset w
                #coordv=wherewhat(nlkink)                                    #Insert a new left kink
                #coordv=wherewhat(nlkinkrn)
            #Repeat this where needed
                 
            

                elif coordhspts[change[0],1]==coordv[x2[y2]+1,1] and\
                     abs(coordhspts[change[0],2]-coordv[x2[y2]+1,2])<1e-5:               #<If the seg. moves to the endpoint of its RHS vertical>:
                    coordv=delete(coordv,[x2[y2],x2[y2]+1],0)           #Delete the RHS vertical (another vertical will be 'above')
                    coordv=append(coordv,nlkink(),0)                              #Insert a new left kink

                elif coordv[x2[y2]+1,4]!=change[3]:                             #<The existing RHS v.seg goes to a different plane>:
                    coordv=insert(coordv,0,nrkink(),0)                            #Insert a new right kink
                    coordv=append(coordv,nlkink(),0)                              #Add a new left kink
                
                elif change[3]==coordv[x2[y2]+1,4] and\
                    (change[3]==0 and q0()<1e-5) or\
                    (change[3]==1 and q1a()==q1b()) or\
                    (change[3]==2 and q2a()==q2b()):                                #<If the RHS v.seg. end point is on the same plane as the seg's new pos>:
                    coordv=delete(coordv,x2[y2],0)                              #Delete the old RHS v.seg. start pos.
                    coordv=insert(coordv,x2[y2],nstartposrhs(),0)                 #Insert a new one
                    coordv=append(coordv,nlkink(),0)                              #Insert a new left kink

                else:
                    coordv=insert(coordv,0,nrkink(),0)                             #Otherwise, insert a new right kink
                    coordv=append(coordv,nlkink(),0)                              #Add a new left kink
                
                return yz,coordh,coordv,segmentplane

            else:                                                               #<LHS v.segs exist>:

                if x2.size==0:                                                  #<But no RHS v.segs>:
                    #print("seg changes plane L vsegs but no R ones")
                    if (abs(ydiffl[change[0]])<1e-5).all() and\
                       x1.size<3:
                       #change[3]==segmentplane[N-1]:                            #<If there is kink migration right>: !!!Added requirement for the adjacent seg to be on the same plane 21/02/20!!!
                        coordv=delete(coordv,x1,0)                              #Delete LHS v.coords
                        coordv=insert(coordv,0,nrkink(),0)                        #Insert a new right kink

                    elif coordhspts[change[0],1]==coordv[x1[y1]-1,1] and\
                         abs(coordhspts[change[0],2]-coordv[x1[y1]-1,2])<1e-5:  #<If the seg. moves to the startpoint of its LHS vertical>:
                        coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)               #Delete the LHS vertical (another vertical will be 'above')
                        coordv=insert(coordv,0,nrkink(),0)                        #Insert a new right kink

                    elif coordv[x1[y1]-1,4]!=change[3]:                         #<If the LHS v.seg. comes from a plane different to the h.seg's new pos.>:
                        coordv=insert(coordv,0,nrkink(),0)                        #Insert a new right kink
                        coordv=append(coordv,nlkink(),0)                          #Add a new left kink

                    elif change[3]==coordv[x1[y1]-1,4] and\
                        (change[3]==0 and p0()<1e-5) or\
                        (change[3]==1 and p1a()==p1b()) or\
                        (change[3]==2 and p2a()==p2b()):                            #<If the LHS v.seg. startpoint is on the same plane as the seg's new pos>:
                        coordv=delete(coordv,x1[y1],0)                          #Delete the old LHS v.seg. end pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)               #Insert a new one
                        coordv=insert(coordv,0,nrkink(),0)                        #Insert a new right kink

                    else:
                        coordv=insert(coordv,0,nrkink(),0)                        #Otherwise, insert a new right kink
                        coordv=append(coordv,nlkink(),0)                          #Add a new left kink
                
                    return yz,coordh,coordv,segmentplane

                else:                                                               #<LHS and RHS v.segs exist>:

                    if (abs(coordhspts[change[0],1:]-coordv[x2[y2]+1,1:3])<1e-5).all() and\
                       (abs(coordhspts[change[0],1:]-coordv[x1[y1]-1,1:3])<1e-5).all():     #<If the end pos. of the existing RHS v.seg. and the start pos. of the LHS v.seg are the seg's new pos. (If both true, then the h.seg. has gone backwards, annihilating a kink pair on the previous plane.)>:
                        coordv=delete(coordv,dkink(),0)                                     #Delete the double kink.

                        return yz,coordh,coordv,segmentplane

                    else:

                        if (abs(ydiffr[change[0]])<1e-5).all() and\
                           x2.size<3:
                           #change[3]==segmentplane[1]:                              #<Kink migration left>: 21/02
                            coordv=delete(coordv,x2,0)                              #Delete RHS v.segs
                            w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                            x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                            y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                            if y1.size>1:
                                y1=y1[-1]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]              
                            y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                            if y2.size>1:
                                y2=y2[0]

                        elif coordhspts[change[0],1]==coordv[x2[y2]+1,1] and\
                             abs(coordhspts[change[0],2]-coordv[x2[y2]+1,2])<1e-5:           #<If the seg. moves to the endpoint of its RHS vertical>:
                            coordv=delete(coordv,[x2[y2],x2[y2]+1],0)       #Delete the RHS vertical (another vertical will be 'above')
                            w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                            x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                            y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                            if y1.size>1:
                                y1=y1[-1]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]              
                            y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                            if y2.size>1:
                                y2=y2[0]

                        elif coordv[x2[y2]+1,4]!=change[3]:                         #<If the RHS v.seg. goes to a different plane than the h.seg.>:
                            coordv=insert(coordv,x2[y2],nrkink(),0)                   #Insert a new right kink before it
                            x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                            y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                            if y1.size>1:
                                y1=y1[-1]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]              
                            y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                            if y2.size>1:
                                y2=y2[0]
                               
                        elif change[3]==coordv[x2[y2]+1,4] and\
                            (change[3]==0 and q0()<1e-5) or\
                            (change[3]==1 and q1a()==q1b()) or\
                            (change[3]==2 and q2a()==q2b()):                            #<If the RHS v.seg. end point is on the same plane as the seg's new pos>:
                            coordv=delete(coordv,x2[y2],0)                          #Delete the old RHS v.seg. start pos.
                            coordv=insert(coordv,x2[y2],nstartposrhs(),0)             #Insert a new one

                        else:
                            coordv=insert(coordv,x2[y2],nrkink(),0)                   #Insert a new right kink
                            x1=where(coordv[:,0]==coordhspts[N-1,0]+1)[0]
                            y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]
                            if y1.size>1:
                                y1=y1[-1]
                            x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]              
                            y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                            if y2.size>1:
                                y2=y2[0]

                    
                        if (abs(ydiffl[change[0]])<1e-5).all() and\
                           x1.size<3:
                           #change[3]==segmentplane[N-1]:                            #<Kink migration right>: 21/02
                            coordv=delete(coordv,x1,0)                              #Delete LHS v.segs

                        elif coordhspts[change[0],1]==coordv[x1[y1]-1,1] and\
                             abs(coordhspts[change[0],2]-coordv[x1[y1]-1,2])<1e-5:      #<If the seg. moves to the startpoint of its LHS vertical>:
                            coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)                   #Delete the LHS vertical (another vertical will be 'above')

                        elif coordv[x1[y1]-1,4]!=change[3]:                         #<If the LHS v.seg. comes from a different plane than the h.seg.>:
                            coordv=append(coordv,nlkink(),0)                          #Insert a new left kink after it
                               
                        elif change[3]==coordv[x1[y1]-1,4] and\
                            (change[3]==0 and p0()<1e-5) or\
                            (change[3]==1 and p1a()==p1b()) or\
                            (change[3]==2 and p2a()==p2b()):                            #<If the LHS v.seg. start point is on the same plane as the seg's new pos>:
                            coordv=delete(coordv,x1[y1],0)                          #Delete the old LHS v.seg. end pos.
                            coordv=insert(coordv,x1[y1],nendposlhs(),0)               #Insert a new one

                        else:
                            coordv=append(coordv,nlkink(),0)                   #Insert a new left kink

                        return yz,coordh,coordv,segmentplane


###################################################################################################################################################################################################
        

    if change[0]==N-1:


        nseg=0


        if (abs(ydiffl[change[0]])<1e-5).all() and (abs(ydiffr[change[0]])<1e-5).all() and\
           x1.size<3 and x2.size<3:
           #change[3]==segmentplane[N-2] and change[3]==segmentplane[0]:
            coordv=delete(coordv,hstack([x1,x2]),0)
            segmentplane[change[0]]=segmentplane[nseg]
            return yz,coordh,coordv,segmentplane
        

        
        
        




    if coordv.size==0:                                                  #Are there existing vertical coordinates?
        coordv=insert(coordv,0,ndkink(),0)                                #Insert a new double kink at index 0
        return yz,coordh,coordv,segmentplane






    #r0=coordhspts[change[0]-1,2]-coordhspts[change[0],2]                #If 0, the h.seg. has moved to the same 'a' plane as its LHS neighbour h.seg

    #r1a=coordhspts[change[0]-1,1]-coordhspts[change[0]-1,2]#/sqrt(3)    #If r1a=r1b, then as above, but with the h.seg. moving to the same 'b' plane
    #r1b=coordhspts[change[0],1]-coordhspts[change[0],2]#/sqrt(3)

    #r2a=coordhspts[change[0]-1,1]+coordhspts[change[0]-1,2]#/sqrt(3)    #If r2a=r2b, then as above, but with the h.seg. moving to the same 'c' plane
    #r2b=coordhspts[change[0],1]+coordhspts[change[0],2]#/sqrt(3)


    #s0=coordhspts[change[0]+1,2]-coordhspts[change[0],2]                #If 0, the h.seg. has moved to the same 'a' plane as its RHS neighbour

    #s1a=coordhspts[change[0]+1,1]-coordhspts[change[0]+1,2]#/sqrt(3)    #If q1a=q1b, then as above, but with the h.seg. moving to the same 'b' plane
    #s1b=coordhspts[change[0],1]-coordhspts[change[0],2]#/sqrt(3)

    #s2a=coordhspts[change[0]+1,1]+coordhspts[change[0]+1,2]/sqrt(3)     #If q2a=q2b, then as above, but with the h.seg. moving to the same 'c' plane
    #s2b=coordhspts[change[0],1]+coordhspts[change[0],2]#/sqrt(3)









###<If the change causes kink annihilation>:
        
    if (abs(ydiffl[change[0]])<1e-5).all() and (abs(ydiffr[change[0]])<1e-5).all() and\
       x1.size<3 and x2.size<3:
       #change[3]==segmentplane[change[0]-1] and change[3]==segmentplane[nseg]:
        segmentplane[change[0]]=segmentplane[nseg]
        coordv=delete(coordv,hstack([x1,x2]),0)
        return yz,coordh,coordv,segmentplane



###<If the seg. doesn't change plane>:
    
    if change[3]==segmentplaneold:                      
        if x1.size==0:                                                      #<If there are no existing LHS v.coords>:
            
            if x2.size==0:                                                  #<If there are also no existing RHS v.coords>:
                coordv=wherewhat(ndkink())                                  #Insert a new double kink
                
            elif (abs(ydiffr[change[0]])<1e-5).all() and\
                 x2.size<3:                                                 #<If there is kink migration left (RHS v.coords, but no LHS ones)>:
                segmentplane[change[0]]=segmentplane[nseg]                                                        
                coordv=delete(coordv,x2,0)                                  #Delete RHS vertical coordinates 
                coordv=insert(coordv,x2[0],nlkink(),0)                      #Insert a new left kink before the first one's place

            elif coordv[x2[y2]+1,1]==coordhspts[change[0],1] and\
                 abs(coordv[x2[y2]+1,2]-coordhspts[change[0],2])<1e-5:  #<If the seg. moves to the end of a RHS vertical (meaning that a new vertical starts next on a new plane)>:
                coordv=delete(coordv,[x2[y2],x2[y2]+1],0)               #Delete the RHS vertical
                coordv=insert(coordv,x2[y2],nlkink(),0)                   #Insert a new left kink before its place
                x1=where(coordv[:,0]==coordhspts[change[0],0])[0]
                y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                y2a=y2a[0]
                segmentplane[change[0]]=coordv[x2[y2a],4]
                coordv[x1[y1a],4]=coordv[x2[y2a],4]
                   
            elif coordv[x2[y2]+1,4]!=change[3]:                         #<If the end point of the existing RHS v.coord is on another plane>:
                coordv=insert(coordv,x2[y2],ndkink(),0)                   #Insert a new double kink before the existing RHS v.coord start point
                    
            else:
                coordv=delete(coordv,x2[y2],0)                          #Otherwise, delete the old RHS v.seg start point
                coordv=insert(coordv,x2[y2],nstartposrhs(),0)             #Insert a new one in its place
                coordv=insert(coordv,x2[y2],nlkink(),0)                   #Insert a new left kink before the new RHS start point
            return yz,coordh,coordv,segmentplane
                    

                
        else:                                                           #<If LHS v.coords exist>:

            
            if x2.size==0:                                              #<But no RHS v.coords exist>:
                
                if (abs(ydiffl[change[0]])<1e-5).all() and\
                   x1.size<3:                                           #<If there is kink migration right>:
                    segmentplane[change[0]]=segmentplane[change[0]-1]                
                    coordv=delete(coordv,x1,0)                          #Delete LHS v.coords
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]     #Reset w
                    coordv=wherewhat(nrkink())                          #Insert a new right kink after the places of the LHS v.coords.

                    
                elif coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                     abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:  #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                    coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)               #Delete the LHS vertical
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                    coordv=wherewhat(nrkink())                              #Insert a new right kink after its place
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]
                    y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                    y1a=y1a[-1]
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]                    
                    segmentplane[change[0]]=coordv[x1[y1a],4]
                    coordv[x2[y2a],4]=coordv[x1[y1a],4]
                
                elif coordv[x1[y1]-1,4]!=change[3]:                         #<If the existing LHS v.seg. comes from another plane>:
                    if (abs(zdifflhs())<1e-5 and segmentplane[change[0]]==0) or\
                       (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and segmentplane[change[0]]==1) or\
                       (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and segmentplane[change[0]]==2):
                        #print("new here!")
                        coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place    
                        coordv=wherewhat(nrkink())
                    else:
                        coordv=wherewhat(ndkink())                          #Insert a new double kink after it
                    
                else:                                                       #<The existing LHS v.seg. doesn't come from another plane or is after another (no RHS v.segs)
                    coordv=delete(coordv,x1[y1],0)                          #Delete the old LHS v.seg. end point
                    coordv=insert(coordv,x1[y1],nendposlhs(),0)             #Insert a new one in its place
                    coordv=wherewhat(nrkink())                              #Insert a new right kink after it
                return yz,coordh,coordv,segmentplane

                
            else:                                                       #<LHS and RHS v.segs exist>:

                
                if (abs(ydiffr[change[0]])<1e-5).all() and\
                   x2.size<3:
                   #change[3]==segmentplane[nseg]:                       #<If there is kink migration left>:
                    #print('here')
                    coordv=delete(coordv,x2,0)                          #Delete RHS vertical coordinates
                    segmentplane[change[0]]=segmentplane[nseg]
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]     #Reset w,x1, y1, x2, y2
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                    y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                    if y1.size>1:
                        y1=y1[-1]    
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                    y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                    if y2.size>1:
                        y2=y2[0]

                elif coordv[x2[y2]+1,1]==coordhspts[change[0],1] and\
                     abs(coordv[x2[y2]+1,2]-coordhspts[change[0],2])<1e-5:       #<If the seg. moves to the end of a RHS vertical (meaning that a new vertical starts next on a new plane)>:
                    if coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                       abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:    #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                        #print("Potential problem?")
                        #print("yz is")
                        #print(yz)
                        #print("coordh is")
                        #print(coordh)
                        #print("coordv is before")
                        #print(coordv)
                        #print('here')
                        coordv=delete(coordv,[x1[y1]-1,x1[y1],x2[y2],x2[y2]+1],0)
                        #print("coordv after delete segments module")
                        #print(coordv,'\n\n')
                        #x1=where(coordv[:,0]==coordhspts[change[0],0])[0]
                        x2=where(coordv[:,0]>=coordhspts[change[0],0]+1)[0]
                        if x2.size!=0:
                            y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                            y2a=y2a[0]
                            segmentplane[change[0]]=coordv[x2[y2a],4]
                            return yz,coordh,coordv,segmentplane
                        else:
                            x2=where(coordv[:,0]>0)[0]
                            y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                            y2a=y2a[0]
                            segmentplane[change[0]]=coordv[x2[y2a],4]
                            return yz,coordh,coordv,segmentplane
                    else:
                        #print("here")
                        coordv=delete(coordv,[x2[y2],x2[y2]+1],0)                   #Delete the RHS vertical
                        w=where(coordv[:,0]>coordhspts[change[0],0])[0]             #Reset w, x1, y1, x2, y2
                        x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                        if y1.size>1:
                            y1=y1[-1]
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]                  
                        y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                        if y2a.size!=0:
                            y2a=y2a[0]
                            segmentplane[change[0]]=coordv[x2[y2a],4]


                elif coordv[x1[y1]-1,1]==coordv[x2[y2]+1,1] and\
                     abs(coordv[x1[y1]-1,2]-coordv[x2[y2]+1,2])<1e-5:   #<If the seg. is upon a double kink>:
                    #print('here')
                    coordv=delete(coordv,x2[y2],0)                     #Delete the existing RHS v.seg. start pos.
                    coordv=insert(coordv,x2[y2],nstartposrhs(),0)         #Insert a new one in its place
                    coordv=delete(coordv,x1[y1],0)                     #Delete the existing LHS v.seg. start pos.
                    coordv=insert(coordv,x1[y1],nendposlhs(),0)           #Insert a new one in its place                    
                    return yz,coordh,coordv,segmentplane

                elif coordv[x2[y2]+1,4]!=change[3]:                     #<If the existing RHS v.seg. goes to another plane>:
                    #print('here')
                    coordv=insert(coordv,x2[y2],nrkink(),0)             #Insert a new right kink before it
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]     #Reset w, x1, y1, x2, y2
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                    y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                    if y1.size>1:
                        y1=y1[-1]
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                    y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                    if y2.size>1:
                        y2=y2[0]                  
                    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                    y2a=y2a[0]
                    segmentplane[change[0]]=coordv[x2[y2a],4]

                else:
                    #print('here')
                    coordv=delete(coordv,x2[y2],0)                      #Otherwise delete the existing RHS vertical start point
                    coordv=insert(coordv,x2[y2],nstartposrhs(),0)         #Insert a new one in its place
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]     #Reset w, x1, y1, x2, y2
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                    y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                    if y1.size>1:
                        y1=y1[-1]
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                    y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                    if y2.size>1:
                        y2=y2[0]                  
                    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]
                    if y2a.size>0:
                        y2a=y2a[0]
                        segmentplane[change[0]]=coordv[x2[y2a],4]

                if (abs(ydiffl[change[0]])<1e-5).all() and\
                   x1.size<3:
                    #print("here")
                    segmentplane[change[0]]=segmentplane[change[0]-1]   #<If there is kink migration right>:
                    coordv=delete(coordv,x1,0)                          #Delete LHS v.coords
                    
                elif coordv[x1[y1]-1,1]==coordhspts[change[0],1] and\
                     abs(coordv[x1[y1]-1,2]-coordhspts[change[0],2])<1e-5:          #<If the seg. moves to the start of a LHS vertical (meaning that another vertical ends before it on another plane)>:
                    #print('there')
                    coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)                       #Delete the LHS vertical
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                    x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]
                    y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]                 
                    if y1a.size>1:
                        y1a=y1a[-1]
                    y2a=where(coordv[x2,1]==coordhspts[change[0],1])[0]                 
                    if y2a.size>1:
                        y2a=y2a[0]
                    segmentplane[change[0]]=coordv[x1[y1a],4]
                    if y2a.size!=0:
                        coordv[x2[y2a],4]=coordv[x1[y1a],4]
                    if yz[change[0],0]==0 and yz[change[0],1]==0:
                        segmentplane[change[0]]=0
                        
##                elif coordv[x1[y1]-1,1]==coordv[x2[y2]+1,1] and\
##                     coordv[x1[y1]-1,2]==coordv[x2[y2]+1,2]:            #<If the seg. is upon a double kink>:
##                    coordv=delete(coordv,x1[y1],0)                      #Delete the existing LHS v.seg. end pos.
##                    coordv=insert(coordv,x1[y1],nendposlhs,0)           #Insert a new one in its place
        #The above is redundant -action is carried out by the else statement

#The extra below IS needed as the segment doesn't change plane?
                elif coordv[x1[y1]-1,4]!=change[3]:           #<If the existing LHS v.seg. comes from another plane>:
                    if (abs(zdifflhs())<1e-5 and change[3]==0) or\
                       (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and change[3]==1) or\
                       (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and change[3]==2):
                        #print("new here!")
                        coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place
                    else:
                        coordv=wherewhat(nlkink())                          #Insert a new left kink after it
    
                   
                else:
                    #print('or there')
                    coordv=delete(coordv,x1[y1],0)                      #Delete the existing LHS vertical endpoint
                    coordv=insert(coordv,x1[y1],nendposlhs(),0)           #Insert a new on in its place
                    x1=where(coordv[:,0]==coordhspts[change[0],0])[0]
                    y1a=where(coordv[x1,1]==coordhspts[change[0],1])[0]
                    y1a=y1a[-1]
                    coordv[x1[y1a],4]=segmentplane[change[0]]
                return yz,coordh,coordv,segmentplane
                
               
###<If the segment changes plane>:
                
    else:
        #print('hello')
        if x1.size==0:                                                      #<No LHS v.segs>:

            if x2.size==0:                                                  #<No RHS v.segs>:
                coordv=wherewhat(ndkink())                                    #Insert a new double kink

            elif (abs(ydiffr[change[0]])<1e-5).all() and\
                 x2.size<3:
                 #change[3]==segmentplane[nseg]:                             #Kink migration left
                coordv=delete(coordv,x2,0)                                  #Delete RHS v.segs
                w=where(coordv[:,0]>coordhspts[change[0],0])[0]             #Reset w
                coordv=wherewhat(nlkink())                                    #Insert a new left kink


            #Change kink annihilation? Problem is that having the rule so that change[3]==segmentplane[nseg] means that the energy will have to be
            #recalculated for the whole line using elastic energy (which will be very slow) whenever kink migration/annihilation happens as updating the
            #energy using loop interactions is not consistent with this.

            #(Update: maybe not with regard to the above comment) Consider making it so that when a segment changes to the same plane as one of its neighbours,
            #delete all verticals and replace with a single one joining them, rather than only deleting the verticals when ydiffl or ydiffr are zero:
            #elif change[3]==segmentplane[change[0]+1] and\
                #(change[3]==0 and s0==0) or\
                #(change[3]==1 and s1a==s1b) or\
                #(change[3]==2 and s2a==s2b):
                #coordv=delete(coordv,x2,0)
                #w=where(coordv[:,0]>coordhspts[change[0],0])[0]             #Reset w
                #coordv=wherewhat(nlkink)                                    #Insert a new left kink
                #coordv=wherewhat(nlkinkrn)
            #Repeat this where needed
                 
            

            elif coordhspts[change[0],1]==coordv[x2[y2]+1,1] and\
                 abs(coordhspts[change[0],2]-coordv[x2[y2]+1,2])<1e-5:          #<If the seg. moves to the endpoint of its RHS vertical>:
                coordv=delete(coordv,[x2[y2],x2[y2]+1],0)                       #Delete the RHS vertical (another vertical will be 'above')
                coordv=insert(coordv,x2[y2],nlkink(),0)                         #Insert a new left kink

            elif coordv[x2[y2]+1,4]!=change[3]:                             #<The existing RHS v.seg goes to a different plane>:
                coordv=wherewhat(ndkink())                                  #Insert a new double kink after it
                
            elif change[3]==coordv[x2[y2]+1,4] and\
                (change[3]==0 and q0()<1e-5) or\
                (change[3]==1 and q1a()==q1b()) or\
                (change[3]==2 and q2a()==q2b()):                                #<If the RHS v.seg. end point is on the same plane as the seg's new pos>:
                coordv=delete(coordv,x2[y2],0)                              #Delete the old RHS v.seg. start pos.
                coordv=insert(coordv,x2[y2],nstartposrhs(),0)                 #Insert a new one
                coordv=insert(coordv,x2[y2],nlkink(),0)                       #Insert a new left kink

            else:
                coordv=insert(coordv,x2[y2],ndkink(),0)                       #Otherwise, insert a new double kink
                
            return yz,coordh,coordv,segmentplane

        else:                                                               #<LHS v.segs exist>:

            if x2.size==0:                                                  #<But no RHS v.segs>:

                if (abs(ydiffl[change[0]])<1e-5).all() and\
                   x1.size<3:
                   #change[3]==segmentplane[change[0]-1]:                    #<If there is kink migration right>:
                    coordv=delete(coordv,x1,0)                              #Delete LHS v.coords
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                    coordv=wherewhat(nrkink())                                #Insert a new right kink after the places of the LHS v.coords.

                elif coordhspts[change[0],1]==coordv[x1[y1]-1,1] and\
                     abs(coordhspts[change[0],2]-coordv[x1[y1]-1,2])<1e-5:           #<If the seg. moves to the startpoint of its LHS vertical>:
                    coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)       #Delete the LHS vertical (another vertical will be 'above')
                    w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                    coordv=wherewhat(nrkink())                                #Insert a new right kink

#Extra below not needed as the segment changes plane
                elif coordv[x1[y1]-1,4]!=change[3]:                         #<If the LHS v.seg. comes from a plane different to the h.seg's new pos.>:
                    if (abs(zdifflhs())<1e-5 and change[3]==0) or\
                       (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and change[3]==1) or\
                       (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and change[3]==2):
                        #print("new here!")
                        coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place
                        coordv=wherewhat(nrkink())
                    else:
                        coordv=wherewhat(ndkink())                          #Insert a new double kink after it

                elif change[3]==coordv[x1[y1]-1,4] and\
                    (change[3]==0 and p0()<1e-5) or\
                    (change[3]==1 and p1a()==p1b()) or\
                    (change[3]==2 and p2a()==p2b()):                            #<If the LHS v.seg. startpoint is on the same plane as the seg's new pos>:
                    coordv=delete(coordv,x1[y1],0)                          #Delete the old LHS v.seg. end pos.
                    coordv=insert(coordv,x1[y1],nendposlhs(),0)               #Insert a new one
                    coordv=wherewhat(nrkink())                                #Insert a new right kink

                else:
                    coordv=wherewhat(ndkink())                                #Otherwise, insert a new double kink
                
                return yz,coordh,coordv,segmentplane

            else:                                                                       #<LHS and RHS v.segs exist>:

                if (abs(coordhspts[change[0],1:]-coordv[x2[y2]+1,1:3])<1e-5).all() and\
                   (abs(coordhspts[change[0],1:]-coordv[x1[y1]-1,1:3])<1e-5).all():     #<If the end pos. of the existing RHS v.seg. and the start pos. of the LHS v.seg are the seg's new pos. (If both true, then the h.seg. has gone backwards, annihilating a kink pair on the previous plane.)>:
                    coordv=delete(coordv,dkink(),0)                                     #Delete the double kink.

                    return yz,coordh,coordv,segmentplane

                else:

                    if (abs(ydiffr[change[0]])<1e-5).all() and\
                       x2.size<3:
                       #change[3]==segmentplane[nseg]:                           #<Kink migration left>:
                        coordv=delete(coordv,x2,0)                              #Delete RHS v.segs
                        w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                        x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                        if y1.size>1:
                            y1=y1[-1]    
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]

                    elif coordhspts[change[0],1]==coordv[x2[y2]+1,1] and\
                         abs(coordhspts[change[0],2]-coordv[x2[y2]+1,2])<1e-5:  #<If the seg. moves to the endpoint of its RHS vertical>:
                        coordv=delete(coordv,[x2[y2],x2[y2]+1],0)               #Delete the RHS vertical (another vertical will be 'above')
                        w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                        x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                        if y1.size>1:
                            y1=y1[-1]    
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]


                    elif coordv[x2[y2]+1,4]!=change[3]:                         #<If the RHS v.seg. goes to a different plane than the h.seg.>:
                        coordv=insert(coordv,x2[y2],nrkink(),0)                   #Insert a new right kink before it
                        w=where(coordv[:,0]>coordhspts[change[0],0])[0]         #Reset w
                        x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                        if y1.size>1:
                            y1=y1[-1]    
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]
                               
                    elif change[3]==coordv[x2[y2]+1,4] and\
                        (change[3]==0 and q0()<1e-5) or\
                        (change[3]==1 and q1a()==q1b()) or\
                        (change[3]==2 and q2a()==q2b()):                            #<If the RHS v.seg. end point is on the same plane as the seg's new pos>:
                        coordv=delete(coordv,x2[y2],0)                          #Delete the old RHS v.seg. start pos.
                        coordv=insert(coordv,x2[y2],nstartposrhs(),0)             #Insert a new one

                    else:
                        coordv=insert(coordv,x2[y2],nrkink(),0)                   #Insert a new right kink
                        x1=where(coordv[:,0]==coordhspts[change[0],0])[0]                    
                        y1=where(coordv[x1,1]==coordhspts[change[0],1]-change[1])[0]        
                        if y1.size>1:
                            y1=y1[-1]    
                        x2=where(coordv[:,0]==coordhspts[change[0],0]+1)[0]  
                        y2=where(coordv[x2,1]==coordhspts[change[0],1]-change[1])[0]
                        if y2.size>1:
                            y2=y2[0]


                    
                    if (abs(ydiffl[change[0]])<1e-5).all() and\
                       x1.size<3:
                       #change[3]==segmentplane[change[0]-1]:                       #<Kink migration right>:
                        coordv=delete(coordv,x1,0)                                  #Delete LHS v.segs

                    elif coordhspts[change[0],1]==coordv[x1[y1]-1,1] and\
                         abs(coordhspts[change[0],2]-coordv[x1[y1]-1,2])<1e-5:      #<If the seg. moves to the startpoint of its LHS vertical>:
                        coordv=delete(coordv,[x1[y1]-1,x1[y1]],0)                   #Delete the LHS vertical (another vertical will be 'above')

#Extra below not needed as the segment changes plane?
                    elif coordv[x1[y1]-1,4]!=change[3]:                               #<If the LHS v.seg. comes from a different plane than the h.seg.>:
##                        if coordv[x1[y1],4]==segmentplane[change[0]]:               #<If the LHS v.seg. ends on the same plane as the h.seg.>:
                        if (abs(zdifflhs())<1e-5 and change[3]==0) or\
                           (abs(ydifflhs()-zdifflhs()/sqrt(3))<1e-5 and change[3]==1) or\
                           (abs(ydifflhs()+zdifflhs()/sqrt(3))<1e-5 and change[3]==2):
                            #print("new here!")
                            coordv=delete(coordv,x1[y1],0)                      #Delete the existing end pos.
                            coordv=insert(coordv,x1[y1],nendposlhs(),0)         #insert a new one in its place
                        else:
                            coordv=wherewhat(nlkink())                              #Insert a new left kink after it
                               
                    elif change[3]==coordv[x1[y1]-1,4] and\
                        (change[3]==0 and p0()<1e-5) or\
                        (change[3]==1 and p1a()==p1b()) or\
                        (change[3]==2 and p2a()==p2b()):                            #<If the LHS v.seg. start point is on the same plane as the seg's new pos>:
                        coordv=delete(coordv,x1[y1],0)                          #Delete the old LHS v.seg. end pos.
                        coordv=insert(coordv,x1[y1],nendposlhs(),0)               #Insert a new one

                    else:
                        coordv=wherewhat(nlkink())                                #Insert a new left kink

                    return yz,coordh,coordv,segmentplane

#Breakdown long vertical segments (if any) into unit segments
def makeunitvsegs(coordv):
    
    ind0=where((abs(coordv[::2,1]-coordv[1::2,1])>1) & \
               (abs(coordv[::2,2]-coordv[1::2,2])<1e-5))[0]
    while ind0.size>0:
        k=2*ind0[0]
        dy=coordv[k+1,1]-coordv[k,1]
        coordv[k+1,1]=coordv[k,1]+dy/abs(dy)
        endplane=copy(coordv[k+1,4])       
        coordv[k+1,4]=0
        for j in range(int(abs(dy)-1)):
            if k+2+2*j<coordv.shape[0]:
                coordv=insert(coordv,k+2+2*j,array([[coordv[k+1,0],coordv[k+1,1]+j*dy/abs(dy),\
                                                     coordv[k+1,2],coordv[k+1,3],coordv[k+1,4]],\
                                                    [coordv[k+1,0],coordv[k+1,1]+(j+1)*dy/abs(dy),\
                                                     coordv[k+1,2],coordv[k+1,3],0]]),0)
            else:
                coordv=append(coordv,array([[coordv[k+1,0],coordv[k+1,1]+j*dy/abs(dy),\
                                             coordv[k+1,2],coordv[k+1,3],coordv[k+1,4]],\
                                            [coordv[k+1,0],coordv[k+1,1]+(j+1)*dy/abs(dy),\
                                             coordv[k+1,2],coordv[k+1,3],0]]),0)
        o=int(abs(dy)-1)
        coordv[k+2+2*o-1,4]=endplane
        ind0=where((abs(coordv[::2,1]-coordv[1::2,1])>1) & \
                   (abs(coordv[::2,2]-coordv[1::2,2])<1e-5))[0]
    #endplane above?

    ind12=where((abs(coordv[::2,1]-coordv[1::2,1])>1/2) & \
                (abs(round(2*coordv[::2,2]/sqrt(3))-round(2*coordv[1::2,2]/sqrt(3)))>1) & \
               ((abs(coordv[::2,2]-coordv[1::2,2]))%(sqrt(3)/2)-sqrt(3)/2<1e-5) )[0]
    while ind12.size>0:
        k=2*ind12[0]
        dy=coordv[k+1,1]-coordv[k,1]
        dz=coordv[k+1,2]-coordv[k,2]
        endplane=coordv[k+1,4]
        if abs(dy-dz/sqrt(3))<1e-5:
            p=1
        else:
            p=2        
        #print(dy,'\n',dz,'\n',p,'\n\n')
        coordv[k+1,1]=coordv[k,1]+dy/abs(dy)/2
        coordv[k+1,2]=coordv[k,2]+dz/abs(dz)*sqrt(3)/2
        coordv[k+1,4]=p
        q=int((abs(dy)-1/2)*2)
        for j in range(int((abs(dy)-1/2)*2)):
            if k+2+2*j<coordv.shape[0]:
                coordv=insert(coordv,k+2+2*j,array([[coordv[k+1,0],coordv[k+1,1]+j*dy/abs(dy)/2,\
                                                     coordv[k+1,2]+j*dz/abs(dz)*sqrt(3)/2,coordv[k+1,3],p],\
                                                    [coordv[k+1,0],coordv[k+1,1]+(j+1)*dy/abs(dy)/2,\
                                                     coordv[k+1,2]+(j+1)*dz/abs(dz)*sqrt(3)/2,coordv[k+1,3],p]]),0)
            else:
                coordv=append(coordv,array([[coordv[k+1,0],coordv[k+1,1]+j*dy/abs(dy)/2,\
                                             coordv[k+1,2]+j*dz/abs(dz)*sqrt(3)/2,coordv[k+1,3],p],\
                                            [coordv[k+1,0],coordv[k+1,1]+(j+1)*dy/abs(dy)/2,\
                                             coordv[k+1,2]+(j+1)*dz/abs(dz)*sqrt(3)/2,coordv[k+1,3],p]]),0)
        coordv[k+2+2*q-1,4]=endplane
        ind12=where((abs(coordv[::2,1]-coordv[1::2,1])>1/2) & \
                    (abs(round(2*coordv[::2,2]/sqrt(3))-round(2*coordv[1::2,2]/sqrt(3)))>1) & \
                   ((abs(coordv[::2,2]-coordv[1::2,2]))%(sqrt(3)/2)-sqrt(3)/2<1e-5) )[0]

    return coordv


