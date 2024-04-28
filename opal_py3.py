import scipy.linalg
import scipy.optimize
import numpy
from math import sqrt
import sys
import pylab
import pickle
import time
#numpy.random.seed(1928)
#RandomArray.seed(1928,1277)

def Mini(a):
    #Finds a minimum value in the list and returns the value and the index in the list.
    for l,v in enumerate(a):
       if l==0:
          r=v
          index=l
       else:
          if r>v:
             r=v
             index=l
    return [r, index]

def Maxi(a):
    for l,v in enumerate(a):
       if l==0:
          r=v
          index=l
       else:
          if r<v:
             r=v
             index=l
    return [r, index]

def func(X): #calls external program that computes the objective function value
        import os
        Npar=len(X)
        F=0
        ofile=open('p.in','w')
        for m in range(0,Npar):
            ofile.write('%23.16e \n'%(X[m]))#Write out the values of the parameters to the input file
        ofile.close()
        cmd='python myfunc.py p.in r.out' #Calculate the objective function
        failure=os.system(cmd)
        if failure: #If the objective funtion fails, return to console
            print ('%s: running %s failed'%(sys.argv[0],cmd))
            successFile=open('done','w')
            successFile.write('failure \n')
            successFile.close()            
            sys.exit()
        ifile=open('r.out','r')
        lines=ifile.readlines()
        F=float(lines.pop(0))#Read the response from the output file
        ifile.close()
        return F

def wrap_function(function, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper
    
def resloader(filename):
    ifile=open(filename,'rb')
    X = pickle.load(ifile)
    while 1:
        try:
            X=numpy.append(X,pickle.load(ifile),axis=0)
        except EOFError:
            break
    ifile.close()
    return X

def feasible_history(filename, Trunc):
    Nf=0;Ns=0
    Nf_history=[];Ns_history=[]
    ifile=open(filename,'r')
    while 1:
        try:
            X=pickle.load(ifile)
            ind = pylab.find(X[:,-1]<=Trunc)
            Nf+=len(X[:,-1])
            Ns+=len(ind)
            Nf_history.append(Nf);Ns_history.append(Ns)
        except EOFError:
            break
    ifile.close()
    return Nf_history, Ns_history
    
def resplotter(filename, bounds = [(-10,10)] * 2, plotevery = 5):
    ifile=open(filename,'r')
    j=0
    while 1:
        try:
            X = pickle.load(ifile)
            if numpy.mod(j,plotevery)==0:
                pylab.figure()
                pylab.title("Iteration %d"%j)
                pylab.plot(X[:,0],X[:,1],'+',ms=10)
                pylab.axis([bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]])
                pylab.grid(True)
                pylab.hold
            j+=1
        except EOFError:
            pylab.figure()
            pylab.title("Iteration %d"%(j-1))
            pylab.plot(X[:,0],X[:,1],'+',ms=10)
            pylab.axis([bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]])
            pylab.grid(True)            
            break
        
    ifile.close()
    pylab.show()
   
#x0=numpy.random.uniform(-10,10,50)
##x0=[-0.77471374,  0.61162612,  0.38027611,  0.14460913]
#epsilon=1.0e-6
#epsilon1=0.001
#epsilon2=0.00001
    ##alpha=1.0
def sd_bfgs(x0,func,fprime=None,bounds=None,boundary_stop=False,epsilon=1.4e-8,epsilon1=0.001,epsilon2=0.00001,gtol=1.0e-6,IniNfeval=0,maxNfeval=5000000,maxiter=50000,*args):
    #INPUT
    #x0: starting input vector (as python list)
    #func: python function
    #fprime: python function generating analytical gradient of func (if available)
    #bounds: [(lower_bound_1, upper_bound_1),(lower_bound_2, upper_bound_2),...,(lower_bound_n, upper_bound_n)], n = # of dim. If None, it is automatically set to [(-10,10)]*n
    #boundary_stop: if false it will continue to search with gradient of parameters at boundary set to zero. 
    #epsilon: perturbation to calculate finite difference gradient (not used if fprime is provided)
    #epsilon1: factor multiplied to gradient*(stepsize in the line search direction) controlling whether to change the step size to a smaller one or not.
    #epsilon2: threshold below which to consider that local gradient and the search direction is nearly perpendicular (cos (theta))
    #gtol: threshold below which the gradient is considered zero and therefore converged
    #IniNfeval: parameter for accounting for number of function evaluation done before reaching this optimization function.  This optimization function will perform total of maxNfeval - IniNfeval function evaluations.
    #maxNfeval: total number of function evaluation
    #maxiter: total number of iteration to be performed.  The optimization terminates when maxNfeval or maxiter is reached
    
    #OUTPUT (x, y, history)
    #x: optimized input vector
    #y: objective value at x, i.e. y=func(x)
    #history: history of objective value with respect to number of function evaluation
    Npar=len(x0)
    Nfeval=IniNfeval
    history=[]
    nanfail=0
    
    def line(alpha,s,X):
        y=func(X+alpha*s,*args)
        return y
    
    def Mini(a):
        #Finds a minimum value in the list and returns the value and the index in the list.
        for l,v in enumerate(a):
            if l==0:
                r=v
                index=l
            else:
                if r>v:
                    r=v
                    index=l
        return [r, index]

    y=func(x0,*args)
    Nfeval=Nfeval+1
    print ('Initial objective value = ',y); history.append(list([Nfeval,y])) #to keep track of convergence history
    
    Xmax=list()
    Xmin=list()    
    if bounds is None:
        bounds = [(-10.0,10.0)] * Npar
        print ('No bounds specified. Default:(-10,10).')
    if len(bounds) != Npar:
        raise ValueError('Number of parameters Npar != length of bounds')
    for m in range(0,Npar):
        Xmin.append(bounds[m][0])
        Xmax.append(bounds[m][1])

    if fprime==None:
        g=scipy.optimize.approx_fprime(x0,func,epsilon,*args)
        Nfeval=Nfeval+len(x0)+1
    else:
        g=fprime(x0,*args);Nfeval=Nfeval+1
        
    #g=2*scipy.optimize.approx_fprime(x0,func,epsilon,*args)\
    #-scipy.optimize.approx_fprime(x0,func,2*epsilon,*args)
    #Nfeval=Nfeval+2*len(x0)+2
    g_i=g.copy()
    g_i_s=sqrt(scipy.dot(g_i,g_i))
#     print 'Initial gradient = ',g
    x=scipy.array(x0)
    B=scipy.eye(len(x))
    #R=scipy.zeros((len(x),len(x)))
    #T=scipy.zeros((len(x),len(x)))
    for m in range(0,maxiter):
    
        #print 'Iteration Number = ',m+1
#        print 'B = \n',B
        s=-scipy.dot(B,g) #s(k)
#        print 's = ',s
        x0=x.copy()
        y0=y
        alpha_array=scipy.array([0.0])
        y_array=scipy.array([y0])
        
        alpha=1.0; alpha_array=scipy.append(alpha_array,alpha)
        y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
        Nfeval=Nfeval+1
        dx=x+alpha*s-x0
        gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
        gTs=scipy.dot(g.reshape(1,len(g)),s.reshape(len(s),1))
        alpha=-0.5*float(gTs/(y-y0-gTs)); alpha_array=scipy.append(alpha_array,alpha)
        y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
        Nfeval=Nfeval+1
#         print 'y_array = ',y_array
#         print 'diff(y_array) [-1]= ',scipy.diff(y_array)[-1]
#         print 'alpha_array = ',alpha_array
        [r,index]=Mini(y_array)
        y=r; alpha=alpha_array[index]
#         print 'y = ',y
#         print 'alpha = ',alpha
        if (alpha < 1.0e-7):
            print ('Hessian is not representing the objective function adequately.')
            print ('Switching to Steepest Descent Method for this iteration.')
            B=scipy.eye(len(x))
            s=-g
#            print 's = ',s
            alpha_array=scipy.array([0.0])
            y_array=scipy.array([y0])
            alpha=1.0; alpha_array=scipy.append(alpha_array,alpha)
            y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
            Nfeval=Nfeval+1
            dx=x+alpha*s-x0
            gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
            gTs=scipy.dot(g.reshape(1,len(g)),s.reshape(len(s),1))
            alpha=-0.5*float(gTs/(y-y0-gTs)); alpha_array=scipy.append(alpha_array,alpha)
            y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
            Nfeval=Nfeval+1
#             print 'y_array = ',y_array
#             print 'diff(y_array) [-1]= ',scipy.diff(y_array)[-1]
#             print 'alpha_array = ',alpha_array
            [r,index]=Mini(y_array)
            y=r; alpha=alpha_array[index]
#             print 'y = ',y
#             print 'alpha = ',alpha
    
    #     if scipy.diff(y_array)[-1]>0.0:
    #         y=y_array[-2]; alpha=alpha_array[-2]
        #alpha=scipy.optimize.brent(line,args=(s,x))    
        for n in range(0,20): #Line search
            if (y0-y)>=-epsilon1*gTdx:
                if (y0-y)<=-(1.0-epsilon1)*gTdx:
                    if -gTs>=epsilon2*sqrt(scipy.inner(g,g)*scipy.inner(s,s)):
                        #print 'stepsize accepted: alpha = ',alpha
                        #y=func(x+alpha*s)
                        #Nfeval=Nfeval+1
                        break
                    else:
                        print ('gradient and search direction are nearly perpendicular')
                        #y=func(x+alpha*s)
                        #Nfeval=Nfeval+1
                        break
                #elif abs(func(x+(alpha+epsilon)*s)-y)/epsilon<=-(1.0-epsilon1)*gTs:
                    #y=func(x+alpha*s)
                    #break
                else:
                    #print 'doubling step size'
                    alpha=2.0**(n+1)*alpha; alpha_array=scipy.append(alpha_array,alpha)
                    y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
                    Nfeval=Nfeval+1
                    dx=x+alpha*s-x0
                    gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
                    #continue
            else:
                #print 'halving step size'
                alpha=0.5**(n+1)*alpha; alpha_array=scipy.append(alpha_array,alpha)
                y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
                Nfeval=Nfeval+1
                dx=x+alpha*s-x0
                gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
                #continue
            if len(y_array)>=3:
                #print 'interpolating'
                xx=scipy.array([alpha_array[-3],alpha_array[-2],alpha_array[-1]])
                yy=scipy.array([y_array[-3],y_array[-2],y_array[-1]])
                X=scipy.zeros([3,3])
                X[:,0]=xx*xx
                X[:,1]=xx
                X[:,2]=scipy.ones(3)
                try:
                    coeff=scipy.dot(scipy.linalg.pinv2(X),yy)
                    #coeff=scipy.polyfit(xx,yy,2)
                    alphat=-coeff[1]/(2*coeff[0])
                except:
                    alphat=1.0e-8
                
                if alphat<0.0:
                    #print 'extrapolated to negative alpha. '
                    alphat=1.0e-8
                elif coeff[0]<0.0:
                    #print 'local maxima in line search. setting alpha to twice the value for maxima.'
                    alphat=2.0*abs(alphat)
            y=func(x+alphat*s,*args)
            Nfeval=Nfeval+1
            #print 'alpha interpolated = ',alphat
            #print 'y interpolated = ',y
            [r,index]=Mini(y_array)
            #print 
            if y>r: #if the interploated alpha is worse than the bisect ones
                #print 'bisect alpha is better'
                y=r
                alpha=alpha_array[index]
                #print 'alpha bisect = ',alpha
                #print 'y bisect = ',y
            else: #if the interpolated alpha is better,
                #print 'interpolated alpha is better'
                alpha=alphat
                                        
        x=x+alpha*s
#         print 'decided to alpha = ',alpha
#         print 'x = ',x
        print ('Objective value = ',y); history.append(list([Nfeval,y])) #to keep track of convergence history
        flag=0;brec=[]
        for m in range(0,Npar):
            if x[m]<Xmin[m]:
                x[m]=Xmin[m]
                g[m]=0.0 ##
                brec.append(m) ##
                flag+=1
            if x[m]>Xmax[m]:
                x[m]=Xmax[m]
                g[m]=0.0 ##
                brec.append(m) ##
                flag+=1
            
        if flag > 0:
            y=func(x,*args);Nfeval=Nfeval+1
            print ('bounds exceeded!')
            print ('objective value at boundary: ',y); history.append(list([Nfeval,y])) 
            if boundary_stop:
                break ##


        if Nfeval >= maxNfeval:
            print ('maximum number of function evaluation reached!')
            break
        
        dx=x-x0 #dx(k)=x(k+1)-x(k)
#         print 'dx = ',dx
        g0=g.copy()
        if fprime==None:
            g=scipy.optimize.approx_fprime(x,func,epsilon,*args)
            Nfeval=Nfeval+len(x)+1
            if flag>0: ##
                for m in brec: ##
                    g[m]=0.0 ##
        else:
            g=fprime(x,*args);Nfeval=Nfeval+1
            if flag>0: ##
                for m in brec: ##
                    g[m]=0.0 ##

        #if sqrt(scipy.dot(g,g)) < g_i_s*1.0e-8:
            #print "Using second order gradient."
            #g=2*g-scipy.optimize.approx_fprime(x,func,2*epsilon,*args)
            #Nfeval=Nfeval+len(x)+1
#         print 'g = ',g
        dg=g-g0 #dg(k)=g(k+1)-g(k)
#         print 'dg = ',dg
        #if min(sqrt(scipy.inner(g,g)),sqrt(scipy.inner(dx,dx))) < 1.0e-8:
        #if min(abs(g))<gtol:
        if sqrt(scipy.inner(g,g)) < gtol:
            print ('converged!')
            break
        dxTdg=scipy.dot(dx.reshape(1,len(dx)),dg.reshape(len(dg),1))
        if B.all()==scipy.eye(len(x)).all():
            #B=(dxTdg/scipy.dot(dg,dg))*B
            B=alpha*B
            #R=scipy.zeros((len(x),len(x)))
            #T=scipy.zeros((len(x),len(x)))
#             print 'scaled B = \n',B
        
        
        dgTB=scipy.dot(dg.reshape(1,len(dg)),B)
        dxdgTB=scipy.dot(dx.reshape(len(dx),1),dgTB)
    #    print 'dxdgTB = \n',dxdgTB
        BdgdxT=dxdgTB.T
    #    print 'BdgdxT = \n',BdgdxT
        Bh1=1.0+scipy.dot(dgTB,dg.reshape(len(dg),1))/dxTdg
    #    print 'Bh1 = \n',Bh1
        Bh2=scipy.dot(dx.reshape(len(dx),1),dx.reshape(1,len(dx)))/dxTdg
    #    print 'Bh2 = \n',Bh2
        Bh3=(dxdgTB+BdgdxT)/dxTdg
    #    print 'Bh3 = \n',Bh3
        Bhat=Bh1*Bh2-Bh3
    #    print 'Bhat = \n',Bhat
        B=B+Bhat #B(k+1)=B(k)+Bhat(k)

        #5step high accuracy adition--------
        #R=R+Bhat
        #T=B
        #B=B+R
        #T=B-T
        #R=R-T
        #-----------------------------------
        nantest=numpy.isnan(B)
        if nantest.any():
            nanfail=nanfail+1
            if nanfail > 2:
                print ('Stopping...')
                break
            print ('Nan detected in the inverse of Hessian B. Letting B=I.')
            B=scipy.eye(len(x))
    print ('Number of function evaluation = ', Nfeval)
    return (x,y,history)

def l_sd_bfgs(x0,func,epsilon=1.4e-8,epsilon1=0.001,epsilon2=0.001,gtol=1.0e-5,mm=10,maxiter=500,*args):
    Nfeval=0
    nanfail=0
    def line(alpha,s,X):
        y=func(X+alpha*s,*args)
        return y
    
    def Mini(a):
        #Finds a minimum value in the list and returns the value and the index in the list.
        for l,v in enumerate(a):
            if l==0:
                r=v
                index=l
            else:
                if r>v:
                    r=v
                    index=l
        return [r, index]

    y=func(x0,*args)
    Nfeval=Nfeval+1
    print ('Initial objective value = ',y)
    g=scipy.optimize.approx_fprime(x0,func,epsilon,*args)
    Nfeval=Nfeval+len(x0)+1
    #g=2*scipy.optimize.approx_fprime(x0,func,epsilon,*args)\
    #-scipy.optimize.approx_fprime(x0,func,2*epsilon,*args)
    #Nfeval=Nfeval+2*len(x0)+2
    g_i=g.copy()
    g_i_s=sqrt(scipy.dot(g_i,g_i))
#     print 'Initial gradient = ',g
    x=scipy.array(x0)
    B=scipy.eye(len(x))
    #R=scipy.zeros((len(x),len(x)))
    #T=scipy.zeros((len(x),len(x)))
    rho=[0 for i in range(mm)]
    dxmem=[scipy.zeros(len(x)) for i in range(mm)]
    dgmem=[scipy.zeros(len(x)) for i in range(mm)]
    pflag=0     
    for m in range(0,maxiter):
    
        #print 'Iteration Number = ',m+1
#        print 'B = \n',B
        if (m==0) or (pflag==1):
            s=-scipy.dot(B,g) #s(k)
        else:
            s=-rd.reshape(len(s))
        #print 's = ',s
        x0=x.copy()
        y0=y
        alpha_array=scipy.array([0.0])
        y_array=scipy.array([y0])
        
        alpha=1.0; alpha_array=scipy.append(alpha_array,alpha)
        y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
        Nfeval=Nfeval+1
        dx=x+alpha*s-x0
        gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
        gTs=scipy.dot(g.reshape(1,len(g)),s.reshape(len(s),1))
        alpha=-0.5*float(gTs/(y-y0-gTs)); alpha_array=scipy.append(alpha_array,alpha)
        y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
        Nfeval=Nfeval+1
#         print 'y_array = ',y_array
#         print 'diff(y_array) [-1]= ',scipy.diff(y_array)[-1]
#         print 'alpha_array = ',alpha_array
        [r,index]=Mini(y_array)
        y=r; alpha=alpha_array[index]
#         print 'y = ',y
         #print 'alpha = ',alpha
        if (alpha < 1.0e-5):
            print ('Hessian is not representing the objective function adequately.')
            print ('Switching to Steepest Descent Method for this iteration.')
            B=scipy.eye(len(x))
            s=-g
            #print 's = ',s
            alpha_array=scipy.array([0.0])
            y_array=scipy.array([y0])
            alpha=1.0; alpha_array=scipy.append(alpha_array,alpha)
            y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
            Nfeval=Nfeval+1
            dx=x+alpha*s-x0
            gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
            gTs=scipy.dot(g.reshape(1,len(g)),s.reshape(len(s),1))
            alpha=-0.5*float(gTs/(y-y0-gTs)); alpha_array=scipy.append(alpha_array,alpha)
            y=func(x+alpha*s,*args); y_array=scipy.append(y_array,y)
            Nfeval=Nfeval+1
#             print 'y_array = ',y_array
#             print 'diff(y_array) [-1]= ',scipy.diff(y_array)[-1]
#             print 'alpha_array = ',alpha_array
            [r,index]=Mini(y_array)
            y=r; alpha=alpha_array[index]
#             print 'y = ',y
#             print 'alpha = ',alpha
            #rho=[0 for i in range(mm)]
            #dxmem=[scipy.zeros(len(x)) for i in range(mm)]
            #dgmem=[scipy.zeros(len(x)) for i in range(mm)]

    
    #     if scipy.diff(y_array)[-1]>0.0:
    #         y=y_array[-2]; alpha=alpha_array[-2]
        #alpha=scipy.optimize.brent(line,args=(s,x))
        pflag=0    
        for n in range(0,20): #Line search
            if (y0-y)>=-epsilon1*gTdx:
                if (y0-y)<=-(1.0-epsilon1)*gTdx:
                    if -gTs>=epsilon2*sqrt(scipy.inner(g,g)*scipy.inner(s,s)):
                        #print 'stepsize accepted: alpha = ',alpha
                        y=func(x+alpha*s)
                        Nfeval=Nfeval+1
                        break
                    else:
                        print ('gradient and search direction are nearly perpendicular')
                        y=func(x+alpha*s)
                        Nfeval=Nfeval+1
                        pflag=1
                        break
                #elif abs(func(x+(alpha+epsilon)*s)-y)/epsilon<=-(1.0-epsilon1)*gTs:
                    #y=func(x+alpha*s)
                    #break
                else:
                    #print 'doubling step size'
                    alpha=2.0**(n+1)*alpha; alpha_array=scipy.append(alpha_array,alpha)
                    y=func(x+alpha*s); y_array=scipy.append(y_array,y)
                    Nfeval=Nfeval+1
                    dx=x+alpha*s-x0
                    gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
                    #continue
            else:
                #print 'halving step size'
                alpha=0.5**(n+1)*alpha; alpha_array=scipy.append(alpha_array,alpha)
                y=func(x+alpha*s); y_array=scipy.append(y_array,y)
                Nfeval=Nfeval+1
                dx=x+alpha*s-x0
                gTdx=scipy.dot(g.reshape(1,len(g)),dx.reshape(len(dx),1))
                #continue
            if len(y_array)>=3:
                #print 'interpolating'
                xx=scipy.array([alpha_array[-3],alpha_array[-2],alpha_array[-1]])
                yy=scipy.array([y_array[-3],y_array[-2],y_array[-1]])
                coeff=scipy.polyfit(xx,yy,2)
                alphat=-coeff[1]/(2*coeff[0])
                if alphat<0.0:
                    print ('extrapolated to negative alpha. ')
                    alphat=1.0e-8
                elif coeff[0]<0.0:
                    print ('local maxima in line search. setting alpha to twice the value for maxima.')
                    alphat=2.0*abs(alphat)
            y=func(x+alphat*s)
            Nfeval=Nfeval+1
            #print 'alpha interpolated = ',alphat
            #print 'y interpolated = ',y
            [r,index]=Mini(y_array)
            #print 
            if y>r: #if the interploated alpha is worse than the bisect ones
                #print 'bisect alpha is better'
                y=r
                alpha=alpha_array[index]
                #print 'alpha bisect = ',alpha
                #print 'y bisect = ',y
            else: #if the interpolated alpha is better,
                #print 'interpolated alpha is better'
                alpha=alphat
                                        
        x=x+alpha*s
         #print 'decided to alpha = ',alpha
#         print 'x = ',x
        print ('Objective value = ',y)
        dx=x-x0 #dx(k)=x(k+1)-x(k)
#         print 'dx = ',dx
        g0=g.copy()
        g=scipy.optimize.approx_fprime(x,func,epsilon,*args)
        Nfeval=Nfeval+len(x)+1
        if sqrt(scipy.dot(g,g)) < g_i_s*1.0e-8:
            print ("Using second order gradient.")
            g=2*g-scipy.optimize.approx_fprime(x,func,2*epsilon,*args)
            Nfeval=Nfeval+len(x)+1
#         print 'g = ',g
        dg=g-g0 #dg(k)=g(k+1)-g(k)
#         print 'dg = ',dg
        #if min(sqrt(scipy.inner(g,g)),sqrt(scipy.inner(dx,dx))) < 1.0e-8:
        #print "magnitude of g = ",sqrt(scipy.inner(g,g))
        #if max(abs(g)) < gtol:
        if sqrt(scipy.inner(g,g)) < gtol:
            print ('converged!')
            break
                
        dxTdg=scipy.dot(dx.reshape(1,len(dx)),dg.reshape(len(dg),1))
        dgTdg=scipy.inner(dg,dg)
        B=alpha*B
        #B=float(dxTdg/dgTdg)*B
        #if (abs(dxTdg) < epsilon) or (dgTdg < epsilon):
            #B=alpha*B
            #dxTdg=epsilon
        #else:
            #B=(dxTdg/dgTdg)*B
        #print B

        #if B.all()==scipy.eye(len(x)).all():
            #B=alpha*B
            #R=scipy.zeros((len(x),len(x)))
            #T=scipy.zeros((len(x),len(x)))
#             print 'scaled B = \n',B
        #dgTB=scipy.dot(dg.reshape(1,len(dg)),B)
        #dxdgTB=scipy.dot(dx.reshape(len(dx),1),dgTB)
        del rho[mm-1]
        try:
            rho.insert(0,1.0/float(dxTdg))
        except:
            print ("Solution stalled.  Stopping...")
            break
        #print 'rho = ',rho
        if abs(dxTdg) > epsilon*dgTdg:
            del dxmem[mm-1]
            dxmem.insert(0,dx)
            del dgmem[mm-1]
            dgmem.insert(0,dg)
        else:
            print ("Correction vectors discarded to maintain positive definiteness.")
        memrange=[i for i in range(mm)]
        al=[0 for i in range(mm)]
        qq=g.copy()
        for i in memrange:
            al[i]=float(rho[i])*scipy.inner(dxmem[i],qq)
            qq=qq-float(al[i])*dgmem[i]
        rd=scipy.dot(B,qq)
        
        for i in reversed(memrange):
            be=rho[i]*scipy.inner(dgmem[i],rd)
            rd=rd+dxmem[i]*(float(al[i])-float(be))
            
        #print rd
    ##    print 'dxdgTB = \n',dxdgTB
        #BdgdxT=dxdgTB.T
    ##    print 'BdgdxT = \n',BdgdxT
        #Bh1=1.0+scipy.dot(dgTB,dg.reshape(len(dg),1))/dxTdg
    ##    print 'Bh1 = \n',Bh1
        #Bh2=scipy.dot(dx.reshape(len(dx),1),dx.reshape(1,len(dx)))/dxTdg
    ##    print 'Bh2 = \n',Bh2
        #Bh3=(dxdgTB+BdgdxT)/dxTdg
    ##    print 'Bh3 = \n',Bh3
        #Bhat=Bh1*Bh2-Bh3
    ##    print 'Bhat = \n',Bhat
        #B=B+Bhat #B(k+1)=B(k)+Bhat(k)

        #5step high accuracy adition--------
        #R=R+Bhat
        #T=B
        #B=B+R
        #T=B-T
        #R=R-T
        #-----------------------------------
        nantest=numpy.isnan(B)
        if nantest.any():
            nanfail=nanfail+1
            if nanfail > 2:
                print ('Stopping...')
                break
            print ('Nan detected in the inverse of Hessian B. Letting B=I.')
            B=scipy.eye(len(x))
    print ('Number of function evaluation = ', Nfeval)
    return (x,y)

def pso(func2, Npar, bounds=None, gaptol=1.0e-04, pop=50, maxiter=5000, maxNfeval=200000,*args):
    #INPUT
    #func2: python function that returns input parameters as well as objective value y, xh,y =func2(x). xh may not be equal to x if func2 itself has some optimization routine inside
    #Npar: number of input parameters
    #bounds: list of tuples defining upper and lower bound of each input parameters [(lo_1,up_1),(lo_2,up_2),...,(lo_Npar,up_Npar)]
    #pop: number of population
    #maxNfeval: total number of function evaluation
    #maxiter: total number of iteration to be performed.  The optimization terminates when maxNfeval or maxiter is reached
    
    #OUTPUT (x,y,history)
    #x: optimized input vector
    #y: objective value at x, i.e. y=func(x)
    #history: history of objective value with respect to number of function evaluation

    
    import os, sys
    import scipy
    import scipy.stats
    from math import sqrt
    #scipy.random.seed(2198)
    Nfeval=0
    history=[]
    #pop=50 #number of population of the swarm
    #Npar=20 #number of parameters to be optimized
    Niter=maxiter #number of swarm optimization iteration
    #Vmax=20
    Xmax=list()
    Xmin=list()

    if bounds is None:
        bounds = [(-10.0,10.0)] * Npar
    if len(bounds) != Npar:
        raise ValueError('Number of parameters Npar != length of bounds')
    for m in range(0,Npar):
        Xmin.append(bounds[m][0])
        Xmax.append(bounds[m][1])
        
    c1=2.0
    c2=2.0
    def Mini(a):
        #Finds a minimum value in the list and returns the value and the index in the list.
        for l,v in enumerate(a):
            if l==0:
                r=v
                index=l
            else:
                if r>v:
                    r=v
                    index=l
        return [r, index]
    
    # Create initial population
    X=list()
    for l in range(0,pop):
        xl=list()
        for m in range(0,Npar):#Each population consists of Npar parameters
            xl.append(numpy.random.uniform(Xmin[m],Xmax[m]))
        X.append(xl)
    
    print ('\n')
    
    
    Ypbest=list()
    Xpbest=list()
    
    V=list()
    for l in range(0,pop):
        v=list()
        for m in range(0, Npar):
            v.append(0)
        V.append(v)
    
    #Initialization strategy 1.
    ##rbest=1e10
    ##Xgbest=list()
    ##for l in range(0,pop):
    ##    Ypbest.append(1e10)
    ##    Xpbest.append(0)
    
    #Initialization strategy 2.
    Y=list()
    for l in range(0,pop): #For every population, calculate the response Y for each population
        #ofile=open('p.in','w')
        #for m in range(0,Npar):
                #ofile.write('%f \n'%(X[l][m]))#Write out the values of the parameters to the input file
        #ofile.close()
        #cmd='python myfunc2.py p.in r.out' #Calculate the objective function
        #failure=os.system(cmd)
        #if failure: #If the objective funtion fails, return to console
                #print '%s: running %s failed'%(sys.argv[0],cmd)
                #successFile=open('done','w')
                #successFile.write('failure \n')
                #successFile.close()
                
                #sys.exit()
        
        #ifile=open('r.out','r')
        #lines=ifile.readlines()
        #Y.append(float(lines.pop(0)))#Read the response from the output file
        #for m in range(0,Npar): #Read the modified X
                #X[l][m]=float(lines[m])
        #ifile.close()
        xh,y=func2(X[l],*args); Nfeval=Nfeval+1
        Y.append(y)
        X[l]=xh
    Ypbest=Y
    Xpbest=X
    [rbest,index]=Mini(Y)
    Xgbest=list(Xpbest[index]);history.append(list([Nfeval,rbest]))
    #print 'Initial X = ', X
    #print 'Initial Y = ', Y
    print ('Initial Xgbest = ', Xgbest)
    print ('Initial Ygbest = ', rbest)
    gap=max(Y)-rbest
    #Start iteration
    for j in range(0,Niter):
        if Nfeval >= maxNfeval:
            print ('maximum number of function evaluation reached')
            break

        print ('Iteration Number = ', j)
        Y=list()
        for l in range(0,pop): #For every population, calculate the response Y for each population
            #ofile=open('p.in','w')
            #for m in range(0,Npar):
                #ofile.write('%f \n'%(X[l][m]))#Write out the values of the parameters to the input file
            #ofile.close()
            #cmd='python myfunc2.py p.in r.out' #Calculate the objective function
            #failure=os.system(cmd)
            #if failure: #If the objective funtion fails, return to console
                #print '%s: running %s failed'%(sys.argv[0],cmd)
                #successFile=open('done','w')
                #successFile.write('failure \n')
                #successFile.close()
                
                #sys.exit()
            
            #ifile=open('r.out','r')
            #lines=ifile.readlines()
            #Y.append(float(lines.pop(0)))#Read the response from the output file
            #for m in range(0,Npar): #Read the modified X
                #X[l][m]=float(lines[m])
            #ifile.close()
            xh,y=func2(X[l],*args); Nfeval=Nfeval+1
            Y.append(y)
            X[l]=xh
            if Ypbest[l] > Y[l]:
                Ypbest[l]=Y[l]
                Xpbest[l]=list(X[l])
        #print 'X = ', X
        #print 'Y = ', Y
        #print 'Xpbest = ', Xpbest
        print ('Ypbest = ', Ypbest)
        #print 'Gbest1 = ',Gbest
    
        [r,index]=Mini(Ypbest) #Find the best (minimum) response
        #print 'r = ', r
        #print 'index = ',index
        if r < rbest:
            rbest=r 
            Xgbest=list(Xpbest[index]) #Determine which  population gave the best (minimum)  response.  

        #index=int(scipy.floor(scipy.random.uniform(0,pop+1)))
        #rr=Ypbest[index]
        #Xr=list(Xpbest[index])
        #if scipy.random.uniform(0,1)< .25:
            #print 'perturbation accepted.'
            #rbest=rr
            #Xgbest=list(Xr)
        print ('rbest = ', rbest); history.append(list([Nfeval,rbest]))
        print ('Xgbest = ', Xgbest)
        w=(float(Niter)-float(j))/float(Niter)
        for l in range(0,pop):
            ru1=numpy.random.uniform(0,1);
            ru2=numpy.random.uniform(0,1);
            
            for m in range(0,Npar):
                V[l][m]=w*V[l][m]+c1*ru1*(Xpbest[l][m]-X[l][m])+c2*ru2*(Xgbest[m]-X[l][m])
    #             V[l][m]=V[l][m]+c1*ru1*(Xpbest[l][m]-X[l][m])+c2*ru2*(Xgbest[m]-X[l][m])
    #             if V[l][m]>Vmax:
    #                 V[l][m]=Vmax
    #             elif V[l][m]<-Vmax:
    #                 V[l][m]=-Vmax
            
            for m in range(0,Npar):
                X[l][m]=X[l][m]+V[l][m]
                if X[l][m]>Xmax[m]:
                    X[l][m]=Xmax[m]
                elif X[l][m]<Xmin[m]:
                    X[l][m]=Xmin[m]
    
                #print 'Gbest3  = ', Gbest
            #print 'Gbest4  = ', Gbest
        #print 'Gbest5 = ', Gbest
        #print 'V=',V 
        if max(Ypbest)-rbest<gaptol*gap:
            print ('converged!')
            break
    print ('number of function evaluation:',Nfeval)
    #ofile=open('p.in','w')
    #for m in range(0,Npar):
        #ofile.write('%17.10e \n'%(Xgbest[m]))#Write out the values of the parameters to the input file
    #ofile.close()
    
    #ofile=open('r.out','w')
    #ofile.write('%17.10e \n'%(r))
    #ofile.close()
    return (Xgbest,rbest,history)

def de(func2, Npar, X=None, bounds=None, gaptol=1.0e-06, pop=50, Cr=0.65, F=0.75, dither=True, IniNfeval=0, maxiter=5000, endMinf=1e-6, endMaxf=1e-6, maxNfeval=10000, args=()):
    #INPUT
    #func2: python function that returns input parameters as well as objective value y, xh,y =func2(x). xh may not be equal to x if func2 itself has some optimization routine inside
    #Npar: number of input parameters
    #X: matrix containing specific starting point to be included in the population. One row corresponds to input parameters of one individual
    #bounds: list of tuples defining upper and lower bound of each input parameters [(lo_1,up_1),(lo_2,up_2),...,(lo_Npar,up_Npar)]. If None, it is automatically set to [(-10,10)]*Npar
    #pop: number of population. If X is not empty, the number of newly (and randomly) generated individual is pop - len(X)
    #Cr: cross over constant, set high value for epistatic problems, if problem is separable lower value is appropriate
    #F: contraction factor of perturbation vector (difference of two individual's input parameter). Not active when dither is True
    #dither: randomly chooses F between 0.5 and 1.0 if True. Usually enhances convergence.
    #maxiter: total number of iteration to be performed.  
    
    #OUTPUT (x,y)
    #x: optimized input vector
    #y: objective value at x, i.e. y=func(x)
    e0=time.time()
    def Mini(a):
        #Finds a minimum value in the list and returns the value and the index in the list.
        for l,v in enumerate(a):
            if l==0:
                r=v
                index=l
            else:
                if r>v:
                    r=v
                    index=l
        return [r, index]
    try:
        Npar = len(X[0])
    except:
        X=[]
    Niter=maxiter
    Xmax=list()
    Xmin=list()
    history=list()
    if bounds is None:
        bounds = [(-10.0,10.0)] * Npar
    if len(bounds) != Npar:
        raise ValueError('Number of parameters Npar != length of bounds')
    for m in range(0,Npar):
        Xmin.append(bounds[m][0])
        Xmax.append(bounds[m][1])
    Nfeval = IniNfeval
    # Create initial population
    #X=list()
    for l in range(0,max([0,pop-len(X)])):
        xl=list()
        for m in range(0,Npar):#Each population consists of Npar parameters
            xl.append(numpy.random.uniform(Xmin[m],Xmax[m]))
        X.append(xl)
    if max([0,pop-len(X)]) == 0: #20240426
        xl=list(X[0]) #20240426
    Y=list()
    pop=len(X)
    for l in range(0,pop): #For every population, calculate the response Y for each population
            xh,y=func2(X[l],*args); Nfeval += 1
            Y.append(y)
            X[l]=list(xh)
    [rbest,ibest]=Mini(Y)
    Xgbest=list(X[ibest])
    gap=max(Y)-rbest
    Xstore=numpy.array(X);Ystore=numpy.array(Y)
    Xstore=numpy.append(Xstore,Ystore.reshape(-1,1),axis=1)
    ofile = open('DEexperiments.p','wb')
    pickle.dump(Xstore,ofile)

#    print 'initial Y = ',Y
#    print 'initial X = ',X
#    print 'initial best individual id = ',ibest
#    print 'initial best response = ',rbest
#    print 'initial best solution = ',Xgbest
    for j in range(0,Niter):
        if dither == True:
            F=numpy.random.uniform(0.5,1.0)
        UX=list()
        UY=list()
        #print 'Iteration Number = ', j
        for l in range(0,pop):
            #print 'Individual Number = ', l
            r=[0,0,0]
            r[0]=int(numpy.random.uniform(0,pop))
            while 1:
                if r[0]==l:
                    r[0]=int(numpy.random.uniform(0,pop))
                else:
                    break
            r[1]=int(numpy.random.uniform(0,pop))
            while 1:
                if (r[1]==r[0]) or (r[1]==l):
                    r[1]=int(numpy.random.uniform(0,pop))
                else:
                    break
            r[2]=int(numpy.random.uniform(0,pop))
            while 1:
                if (r[2]==r[1]) or (r[2]==r[0]) or (r[2]==l):
                    r[2]=int(numpy.random.uniform(0,pop))
                else:
                    break
            #print 'parents selected: ',r
            Rnd=int(numpy.random.uniform(0,Npar))
            for m in range(0,Npar):
                if (numpy.random.uniform(0,1)<Cr) or (Rnd == m):
                    xl[m]=X[r[2]][m]+F*(X[r[0]][m]-X[r[1]][m])
                else:
                    xl[m]=X[l][m]
            for m in range(0,Npar):
                if (xl[m]<Xmin[m]) or (xl[m]>Xmax[m]):
                    xl[m]=numpy.random.uniform(Xmin[m],Xmax[m])
            xh,y=func2(xl,*args); Nfeval += 1
            #print 'xl = ',xl
            #print 'xh = ',xh
            #print 'y = ',y
            UX.append(list(xh));UY.append(y)
        #print 'UY = ',UY
        #print 'UX = ',UX
        Xstore=numpy.array(UX);Ystore=numpy.array(UY)
        Xstore=numpy.append(Xstore,Ystore.reshape(-1,1),axis=1)
        pickle.dump(Xstore,ofile)

        for l in range(0,pop):
            if UY[l]<=Y[l]:
                for m in range(0,Npar):
                    X[l][m]=UX[l][m]
                Y[l]=UY[l]
                #print 'offspring accepted: ',Y[l]
                if UY[l]<=rbest:
                    ibest=l
                    Xgbest=list(UX[l])
                    rbest=UY[l]
        
        if max(Y)-rbest<gaptol:
            print ('min(Y) and max(Y) below tolerance. Converged!')
            print ('Number of function evaluations: ',Nfeval)
            break
        if max(Y)<endMaxf:
            print ('max(Y) < endMaxf.')
            print ('Number of function evaluations: ',Nfeval)       
            break
        if Nfeval > maxNfeval:
            print ('Maximum allowable number of function evaluations reached!')
            break
        if min(Y) < endMinf:
            print ('Required Minimum value reached!')
            break
        history.append(list([Nfeval,rbest]))
        #print 'Y = ',Y
        #print 'X = ',X
        #print 'best individual id = ', ibest
        #print 'population responses = ', Y
        #print 'best response = ',rbest
        #print 'best solution = ', Xgbest
    #print 'number of function evaluations:',Nfeval
    
    ofile.close()
    e1=time.time()
    print ('elapsed time: ',e1-e0)
    return (Xgbest,rbest, X,Y,Nfeval,history)
def spsa(x0,func,bounds=None,alpha=0.602,gamma=0.101,deltax_0=0.1,a=None,a_min=1.0e-6,c=1.0e-6,stepredf=0.5,gtol=1.0e-5,graditer=1,memsize=100,IniNfeval=0,maxiter=5000,adaptive_step=True,relaxation=True,dynamic_system=False,*args):
    #INPUT
    #x0: starting input vector (as python list), if dynamic_system=True, append 0 in the list
    #func: python function
    #bounds: [(lower_bound_1, upper_bound_1),(lower_bound_2, upper_bound_2),...,(lower_bound_n, upper_bound_n)], n = # of dim. If None, it is automatically set to [(-10,10)]*n
    #alpha: exponential controlling the reduction of step size
    #gamma: exponential controlling the finite differece gradient perturbation magnitude
    #deltax_0: desiried minimum initial perturbation of x0
    #stepredf: factor controlling the recduction of stepsize along stochastic gradient descent if no improvement in objective function was observed in the generated perturbations to compute stochastic gradient
    #gtol: threshold value below which gradient is considered to be zero and therefore converged
    #graditer: number of times gradients are computed to obtain an averaged stochastic gradient
    #IniNfeval: parameter for accounting for number of function evaluation done before reaching this optimization function.  This optimization function will perform total of maxNfeval - IniNfeval function evaluations.
    #maxiter: total number of iteration to be performed.  The optimization terminates when maxNfeval or maxiter is reached
    #adaptive_step: Initial stepsize is automatically reduced if set to True to provide reliable objective value descent
    redcounter=0
    if dynamic_system == False:
        Npar=len(x0)
    else:
        Npar=len(x0)-1
        
    def g_sa(x,func,ck,niter,*args):#stochastic gradient calculation
        p=len(x)
        gsum=0.0
        for m in range(niter):
            delta=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
                
            # print "delta = ",delta
            xp=x+ck*delta
            xm=x-ck*delta
            if dynamic_system == True:
                xp[-1]=xm[-1]=x[-1]
            yp=func(xp,*args)
            ym=func(xm,*args)
            gsum=gsum+(yp-ym)/(2*ck*delta)
        ghat=gsum/niter;# print 'ghat = ',ghat
        if dynamic_system == True:
            ghat[-1]=0
        return (ghat,yp,ym,xp,xm,delta)
    
    Xmax=list()
    Xmin=list()
    if bounds is None:
        bounds = [(-10.0,10.0)] * Npar
        print ('No bounds specified. Default:(-10,10).')
    if len(bounds) != Npar:
        raise ValueError('Number of parameters Npar != length of bounds')
    for m in range(0,Npar):
        Xmin.append(bounds[m][0])
        Xmax.append(bounds[m][1])

    Nfeval=IniNfeval
    x0=numpy.array(x0)
    history=[]
    historyx=[]
    p=len(x0)
    A=int(0.1*maxiter)
    y0=func(x0,*args); Nfeval=Nfeval+1
    mem=numpy.ones(memsize)*y0
    x=x0.copy()
    print ('initial objective value = ',y0)
    x_best=x0.copy();y_best=y0; #y_ave=y0; y_max=y0
    for k in range(0,maxiter):
        if dynamic_system == True:
            x[-1]=k
        ck=c/(k+1)**gamma
        ghat,yp,ym,xp,xm,delta=g_sa(x,func,ck,graditer,*args);Nfeval=Nfeval+graditer*2
        if (k==0):
            if a == None:
                a=deltax_0*(A+1)**alpha/(min(abs(ghat[:Npar])))
            a_ini=a
            print ('ghat0 = ',ghat[:])
        ak=a/(k+1+A)**alpha
        #y_ave_old=y_ave
        #y_ave=(k+1)*y_ave/(k+2)+max(yp,ym)/(k+2)

        #delta=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
        #xp=x+ck*delta
        #xm=x-ck*delta
        #yp=func(xp,*args); Nfeval=Nfeval+1
        #print 'yp = ',yp
        #ym=func(xm,*args); Nfeval=Nfeval+1
        #print 'ym = ',ym
        #ghat=(yp-ym)/(2*ck*delta); #print 'ghat = ',ghat
        print ('k: %d, ym = %f, yp = %f, a = %f'%(k,ym,yp,a))
        xold=x.copy()
        x=x-ak*ghat
        for m in range(0,Npar):
            if x[m]<Xmin[m]:
                x[m]=Xmin[m]
            elif x[m]>Xmax[m]:
                x[m]=Xmax[m]
        y=func(x,*args); history.append(list([Nfeval,y])); historyx.append(list(x))#to keep track of convergence history
        mem=numpy.append(mem,numpy.min([ym,yp]))
        mem=numpy.delete(mem,0)
        #if sqrt(scipy.inner(ghat,ghat)) < gtol:
            #print 'converged!'
            #break
        if ym<y_best:
            x_best=xm; #print 'x_best = ',xm
            y_best=ym
            #a=a/stepredf
        if yp<y_best:
            x_best=xp; #print 'x_best = ',xp
            y_best=yp
            #a=a/stepredf
        if adaptive_step == True:
            #if ((yp-y0)>abs(y0)) or ((ym-y0)>abs(y0)):
            if ((y0-min(yp,ym))<0):
            #if (y0-y_ave)<0:
            #if (mem.mean()>y0+c) and (numpy.mod(k,memsize)==0):
                print ('divergence detected. reinitializing.')
                redcounter+=1
                #x=x0.copy()
                x=x_best.copy()
                a=stepredf*a
                if (redcounter > int(0.05*maxiter)) and relaxation:
                #if (a < a_min) and relaxation:
                    print ("Too many divergence. Resetting a and relaxing threshold!")
                    a=a_ini
                    #dim=numpy.random.randint(0,Npar)
                    #x[dim]=numpy.random.uniform(bounds[dim][0],bounds[dim][1])
                    y0=min(yp,ym)
                    redcounter=0
                #y_max=max(yp,ym)
                #a=numpy.max([stepredf*a,c])
                #a=(stepredf+c*numpy.random.randn())*a
                 
            # if y_ave < y0:
                # a=a_ini
    y=func(x,*args); Nfeval=Nfeval+1
    history.append(list([Nfeval,y]))
    historyx.append(list(x))
    print ('number of function evaluation: ',Nfeval)
    return (x,y,history,historyx,Nfeval)
        
def spsa2(x0,func,alpha=0.602,gamma=0.101,a=1.0,deltax_0=0.1,c=1.0e-6,ct=1.0e-6,stepredf=0.7,divf=1.5,gtol=1.0e-5,maxiter=5000,*args):
    epsilon=scipy.finfo(float).eps
    Nfeval=0
    history=[]
    p=len(x0)
    A=0.05*maxiter
    y0=func(x0,*args); Nfeval=Nfeval+1
    x=x0.copy()
    print ('initial objective value = ',y0)
    #Hbk=numpy.zeros([len(x),len(x)])
    Hbk=numpy.eye(len(x))
    for k in range(0,maxiter):
        print ('iteration: ',k)
        ak=a/(k+1+A)**alpha
        ck=c/(k+1)**gamma
        
        delta=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
        #print 'x = ',x
        xp=x+ck*delta; #print 'xp = ',xp
        xm=x-ck*delta; #print 'xm = ',xm
        yp=func(xp,*args); Nfeval=Nfeval+1
        print ('yp = ',yp)
        ym=func(xm,*args); Nfeval=Nfeval+1
        print ('ym = ',ym)
        num1=2*ck*delta
        ghat=(yp-ym)/(num1); #print 'ghat = ',ghat
        
        ckt=ct/(k+1)**gamma
        deltat=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
        num2=ckt*deltat
        G1p=(func(xp+ckt*deltat,*args)-yp)/num2; Nfeval=Nfeval+1
        G1m=(func(xm+ckt*deltat,*args)-ym)/num2; Nfeval=Nfeval+1
        dGk=G1p-G1m
        Htemp=dGk/(num1.reshape(len(num1),1))
        Hhk=0.5*(Htemp+Htemp.T)
        Hbk=(float(k)/(k+1))*Hbk+(1.0/(1+k))*Hhk
        #print 'Hbk',Hbk
        u,s,vh=scipy.linalg.svd(Hbk)
        #r,index=Mini(abs(s))
        print ('s = ',s)
        #print 'Minimum s and index',r,index
        for itemp,vtemp in enumerate(s):
            if vtemp<=1.0e-8:
                print ('small or negative eigenvalue at element',itemp)
                break
        if (itemp<=(len(s)-1)) and (vtemp<=1.0e-8):
            ef=max((s[itemp-1]/s[0])**(itemp-2),epsilon)
            #ef=0.5
            print ('s[0] = ',s[0])
            print ('s[i-1] = ',s[itemp-1])
            print ('ef = ',ef)
            for l in range(itemp,len(s)):
                s[l]=s[l-1]*ef
            print ('modified s = ',s)
        lambda_kb=1.0
        for m in s:
            lambda_kb=m*lambda_kb
        if scipy.isnan(lambda_kb):
            lambda_kb=0.0
        print ('lambda before normalization: ',lambda_kb)
        print ('dimension: ',p)
        lambda_kb=max(lambda_kb,epsilon)**(1.0/p)
        si=1.0/s
        Hbbk=scipy.dot(scipy.dot(u,scipy.diag(s)),vh)
        Bbbk=scipy.dot(scipy.dot(vh.T,scipy.diag(si)),u.T)
        print ('lambda_kb = ',lambda_kb)
        #print 'Bbbk*Hbk',scipy.dot(Bbbk,Hbk)
        xold=x.copy()
        if k<500:
            #x=x-(ak/lambda_kb)*ghat
            x,info=scipy.linalg.bicgstab(scipy.diag(scipy.diag(Hbbk)),scipy.dot(scipy.diag(scipy.diag(Hbbk)),x)-ak*ghat/lambda_kb,x0=None,tol=1.0e-16)
        else:
            x=x-(ak)*scipy.dot(Bbbk,ghat)
            #x,info=scipy.linalg.bicgstab(Hbbk,scipy.dot(Hbbk,x)-ak*ghat,x0=x,tol=1.0e-16)
        y=func(x,*args); history.append(list([Nfeval,y])) #to keep track of convergence history

        #if k==0:
            #a=deltax_0*(A+1)**alpha/(min(abs(ghat)))
        #if sqrt(scipy.inner(ghat,ghat)) < gtol:
            #print 'converged!'
            #break
        if (yp>divf*y0) or (ym>divf*y0):
            x=x0.copy()
            a=stepredf*a
            #Hbk=scipy.eye(p)
            print ('divergence detected. reinitializing.' )
        # else:
            # a=1.0
    y=func(x,*args); Nfeval=Nfeval+1
    
    print ('number of function evaluation: ',Nfeval)
    return (x,y,history)

def sqsd(x0,func,epsilon=1.4e-10,epsilon_g=1.0e-8,epsilon_x=1.0e-8,step_limit=1.0,IniNfeval=0,maxiter=5000,*args):
#    Npar=len(x0)
    Nfeval=IniNfeval
    history=[]
    historyx=[]
    y=func(x0,*args)
    Nfeval=Nfeval+1
    historyx.append(list(x0))
    print ('Initial objective value = ',y); history.append(list([Nfeval,y]))
    
    g=scipy.optimize.approx_fprime(x0,func,epsilon,*args)
    Nfeval=Nfeval+len(x0)+1
    
    g_norm=numpy.sqrt(numpy.dot(g,g))
    c=g_norm/step_limit
    x=numpy.asarray(x0)
    for i in range(maxiter):
        #print "c = ",c
        #print "g = ",g
        if g_norm < epsilon_g:
            print ("Gradient norm less than epsilon_g. Converged!")
            break
        else:
            x_old=x.copy()
            x=x-g/c
            
        x_diff=x-x_old
        x_diff_norm=numpy.sqrt(numpy.dot(x_diff,x_diff))
        
        if x_diff_norm > step_limit:
            print ("Change in solution exceeds step_limit: ",x_diff_norm)
            x=x_old-step_limit*g/g_norm
            x_diff=x-x_old
            x_diff_norm=numpy.sqrt(numpy.dot(x_diff,x_diff))
            print ("Reduced step: ",x_diff_norm)
        if x_diff_norm < epsilon_x:
            print ("Solution change norm is less than epsilon_x. Converged!")
            break
        g_old=g.copy()
        g=scipy.optimize.approx_fprime(x,func,epsilon,*args)        
        Nfeval=Nfeval+len(x0)+1
        g_norm=numpy.sqrt(numpy.dot(g,g));g_old_norm=numpy.sqrt(numpy.dot(g_old,g_old))

        check_1=numpy.dot(g,g_old)/(g_norm*g_old_norm)
        check_2=g_norm/g_old_norm
        print ("cos(theta) = ",check_1)
        print ("gradient norm ratio = ",check_2)

        
        y_old=y.copy()
        y=func(x,*args);Nfeval=Nfeval+1;historyx.append(list(x))
        print ("New objective value = ",y); history.append(list([Nfeval,y])) 
        
        c=2.0*(y_old-y-numpy.dot(g,x_old-x))/(x_diff_norm**2)
#        if (check_1 < -0.9) and (check_2 > 0.9):
#            #c=2*c; 
#                        step_limit/=2.0
#            print "Entering into zig-zag. Reducing step length."
        
        if c < 0.0:
            c=1.0e-10
        if i == maxiter-1:
            print ("Maximum number of iteration reached: ", maxiter)
    print ('Number of function evaluation = ', Nfeval)
    #if len(x0)==2:
        #pylab.plot(numpy.asarray(historyx)[:,0],numpy.asarray(historyx)[:,1],'+')
        #pylab.show()
    return (x,y,history,historyx)
 
def ce(x0,func,samp=1000,top=100,sigma=None,epsilon=0.01,alpha=0.8,beta=0.7,q=5,*args):
    """
    cf. Dirk P. Kroese, Sergey Porotsky, and Reuven Y. Rubinstein, "The Cross-
    Entropy Method for Continuous Multi-Extremal Optimization"
    contopt_fin.pdf
    """
    n = len(x0); N=samp; Nel = top; history=[]
    #alpha = 0.8; beta = 0.7;
    #mu=-2 + 4*numpy.random.uniform(size=n)
    mu = numpy.array(x0)
    mu_last = mu
    if sigma == None:
        sigma = 100.0*numpy.ones(n)
    sigma_last = sigma
    X_best_overall = numpy.zeros(n)
    S_best_overall = 1E10
    t = 0; Nfeval = 0
    while max(sigma) > epsilon:
        t = t+1
        mu = alpha*mu + (1.0-alpha)*mu_last
        B_mod = beta - beta*(1.0-1.0/t)**q
        sigma = B_mod*sigma + (1-B_mod)*sigma_last  # dynamic smoothing
        X = mu + numpy.dot(numpy.random.randn(N,n),numpy.diag(sigma)); # generate samples
        SA = numpy.array(map(lambda x: func(x,*args), X)); Nfeval += N # compute responses
                #print SA
        I_sort = SA.argsort()
                #print I_sort
        S_sort = SA[I_sort]
        #gam = S_sort[Nel-1]
        S_best = S_sort[0]; history.append(S_best)
        if (S_best < S_best_overall):
            S_best_overall = S_best
            X_best_overall = X[I_sort[0],:]
        mu_last = mu
        sigma_last = sigma
        Xel = X[I_sort[1:Nel],:];
        mu = numpy.mean(Xel, axis = 0);
        sigma = numpy.std(Xel, axis = 0);
        if numpy.mod(t,100)==0: # print each 100 iterations
            print ('N. of func. eval. = %d, f = %5.4f, max_sigma = %9.7f'%(Nfeval,S_best,max(sigma)))
#            print mu
#            print '\n'
            
    print ('number of function evaluation: ',Nfeval)
        
    return (X_best_overall, S_best_overall, sigma, history)
    
def lsrs(func, npar, bounds=None, nsamp=100, niter=10, maxiter=50, epsilon=1.4e-8, *args):
    def line_search(func, niter, Xin, *args):
        nvar=len(Xin[0])
        nsamp=len(Xin)
        Y=numpy.array(map(lambda x: func(x,*args), Xin))
        for k in range(niter):
            for i in range(nsamp):
                x=numpy.array(Xin[i,:])
                p=-1#numpy.random.rand()
                alpha=2.0+3.0/(2**(k**2+1))
                x=Xin[i,:]+p*alpha
                y=func(x,*args)
                if y < Y[i]:
                    Y[i]=y
                    Xin[i,:]=numpy.array(x)
        return [Xin, Y]
        
    def re_start(Xin, Y, func, bounds, *args):
        r,index = Mini(Y)
        x=numpy.array(Xin[index,:])
        bounds_list=[]
        for lb,ub in bounds:
            bounds_list.append([lb,ub])
        g=scipy.optimize.approx_fprime(x,func,epsilon,*args)
        for i,r in enumerate(g):
            if r > 0:
                bounds_list[i][1]=x[i]
            if r < 0:
                bounds_list[i][0]=x[i]
        return bounds_list
        
    if bounds == None:
        bounds=[(-10,10)]*npar
    for m in range(maxiter):
        print ('iteration %d'%(m))
        Xin=numpy.random.rand(nsamp,npar)
        for i, (lb, ub) in enumerate (bounds):
            Xin[:,i] = lb + (ub -lb)*Xin[:,i]
    
    
        Xin, Y = line_search(func, niter, Xin, *args)
        bounds = re_start(Xin, Y, func, bounds, *args)
        
    r,index = Mini(Y)
    x=Xin[index,:]
    y=Y[index]
    return [x, y, Xin, Y, bounds]
        
def hj(x0, func, npar, bounds=None, maxiter=50, epsilon=1.4e-8, annealing=0.2,*args):
    #x0 is a python list
    y0=func(x0)
    ybest=y0
    T=abs(y0)
    x=list(x0)
    xbest=list(x0)
    move=[0]*npar
    r=[u-l for l,u in bounds]
    for k in range(maxiter):
        print ('iteration ',k)
        print (xbest,ybest)
        for i in range(npar):
            xold=x[i]
            if numpy.random.rand() < 0.5:
                x[i]=min(x[i]+0.5*r[i],bounds[i][1])
                move[i]=0.5*r[i]
            else:
                x[i]=max(x[i]-0.5*r[i],bounds[i][0])
                move[i]=-0.5*r[i]
            yu=func(x)
            #print x,r,yu
            print (yu)
            if yu < ybest:
                xbest[i]=x[i]
                ybest = yu
                r[i]*=1.25                
            # elif numpy.random.rand() < numpy.exp((ybest-yu)/(T+epsilon)):
                # xbest[i]=x[i]
                # ybest = yu
                # r[i]*=0.8
            else:
                x[i]=xold
                r[i]*=0.8
                move[i]=0
            
        #vector move
        xold=list(x)
        x=[a+b for a,b in zip(x,move)]
        x=[min(bounds[i][1],a) for i,a in enumerate(x)]
        x=[max(bounds[i][0],a) for i,a in enumerate(x)]
        
        yu=func(x)
        if yu < ybest:
            print ("vector move succeded!")
            xbest=list(x)
            ybest = yu
        else:
            print ("vector move failed.")
            x=list(xold)

        T*=annealing
                
    return [xbest, ybest]
            
            
        
            
        
    
    
     
                    
