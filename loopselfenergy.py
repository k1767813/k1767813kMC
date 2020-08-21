from enginteract import enginteract

from numpy import array,sqrt,dtype

def loopselfenergy(mu,nu,b,a,h,rc):


#For calculating the self-energy (including self-interactions) of an incremental dislocation loop.

    Eloopself1=dtype("float128")
    Eloopself2=dtype("float128")
    Eloopself3=dtype("float128")

    def loop1():                                                    #An incremental loop with its bottom left-hand corner at (0,0,0) (a-type planes)
        return array([[0,0,0],[0,h,0],[0,h,0],[a,h,0], \
                      [a,h,0],[a,0,0],[a,0,0],[0,0,0]],dtype="float128")

    def loop2():                                                    #An incremental loop with its bottom left-hand corner at (0,0,0) (b-type planes)            
        return array([[0,0,0],[0,h/2,h*sqrt(3)/2],[0,h/2,h*sqrt(3)/2],[a,h/2,h*sqrt(3)/2], \
                      [a,h/2,h*sqrt(3)/2],[a,0,0],[a,0,0],[0,0,0]],dtype="float128")

    def loop3():                                                    #An incremental loop with its bottom left-hand corner at (0,0,0) (c-type planes)            
        return array([[0,0,0],[0,h/2,-h*sqrt(3)/2],[0,h/2,-h*sqrt(3)/2],[a,h/2,-h*sqrt(3)/2], \
                      [a,h/2,-h*sqrt(3)/2],[a,0,0],[a,0,0],[0,0,0]],dtype="float128")

                                             
    Eloopself1=0.0                                                    #Self energy of loop1 (loop self energy will change with the Burgers vector!)
    for i in range(4):
        for j in range(4):
            Eloopself1+=enginteract(mu,nu,b,b,loop1()[2*i,:],loop1()[2*i+1,:],loop1()[2*j,:],loop1()[2*j+1,:],rc)/2


    Eloopself2=0.0                                                    #Self energy of loop2 (loop self energy will change with the Burgers vector!)
    for i in range(4):
        for j in range(4):
            Eloopself2+=enginteract(mu,nu,b,b,loop2()[2*i,:],loop2()[2*i+1,:],loop2()[2*j,:],loop2()[2*j+1,:],rc)/2

    Eloopself3=0.0                                                    #Self energy of loop3 (loop self energy will change with the Burgers vector!)
    for i in range(4):
        for j in range(4):
            Eloopself3+=enginteract(mu,nu,b,b,loop3()[2*i,:],loop3()[2*i+1,:],loop3()[2*j,:],loop3()[2*j+1,:],rc)/2


    loopselfeng=array([Eloopself1,Eloopself2,Eloopself3],dtype="float128")

    return loopselfeng
