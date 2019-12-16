#------------------------------------------------------------------------------#
# Numerical simulation of sea-surface elevation of linear Gaussian wave packet #
#------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------------#
# Important parameters:
# k_0L_f=10
# simulation from t=-|\tau| to t=0 where 0 is the linear focusing time
# phase=uniform, constant, can be random or 0
#---------------#

random_phase=1 # set "1" for random phase, or "0" for phi=0

## Imports
import numpy as np

## Define the wavenumber and angular frequency axis 
k0=0.02; kxmin=0.005; kxmax=0.037; Nkx=150
kx=np.linspace(kxmin,kxmax,Nkx)
g=9.81 #acceleration of gravity
omega=np.sqrt(g*kx)
dkx=np.ones(Nkx,float)*(kx[1]-kx[0]) #wavenumber increment

## Define the group size at focus L_f
L=(10)*(k0)**-1

## Calculate the group spectrum and associated amplitudes
G=L*((2*np.pi)**-0.5)*np.exp(-(L**2)*0.5*(kx-k0)**2)
A0=5
a=A0*G*dkx/np.sum(G*dkx)

## Define the phase
if random_phase:
    phi0=(1-2*np.random.rand(1))*np.pi # random draw between -Pi and Pi
else:
    phi0=0
phi=phi0*np.ones(Nkx) # uniform and constant

## Compute some quantities related to derivatives of omega at k0
cg0=(np.sqrt(g/k0))*(0.5) #group velocity (first derivative of the phase velocity c_0)
par2om=(np.sqrt(g*k0)/(k0**2))*(-0.25) #second derivative of c_0 with respect to the wavenumber k
tau=L**2/par2om #linear group contraction time-scale (negative)

## Define the temporal and spatial axis
dt=0.13*2*np.pi/np.max(omega)
Nt=1*int(np.abs(tau)/dt)
t=np.linspace(-dt*Nt,0,Nt)

Lx=4*(2*np.pi/k0)
dx=0.000015*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)

## Calculate the elevation
eta1=np.real( 
    np.tensordot( np.tensordot(a*np.exp(1j*phi),np.ones(Nx),axes=0)*
        np.exp( 1j*np.tensordot(kx,x,axes=0) ), 
    np.exp( -1j*np.tensordot(omega-cg0*kx,t,axes=0) ),
            axes=[0,0] )
                    )

#---------------------------------------------------#
# Second part : apply the method and store the data #
#---------------------------------------------------#

Spread_x=np.sqrt(L**2+(L**(-2))*(par2om*t)**2)
c0=(np.sqrt(g/k0))
ccr_1ord=cg0+0.5*k0*par2om-c0
ccr_grad_1ord=(-c0+cg0)*k0*t/tau
ccr_2ord=c0-cg0-0.5*k0*par2om
ccr_grad_2ord=k0*(1.5*par2om*k0+4*c0-4*cg0)*t/tau
ccr_lapl_2ord=2*((c0-cg0+0.5*k0*par2om)*(t/tau)**2-0.5*k0*par2om)*(k0)**2
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)

#---------------#

extrema=[]
for n in range(len(t)):
    for i in range(1,len(x)-1):
        if (eta1[i,n]<eta1[i-1,n] and eta1[i,n]<eta1[i+1,n]) or (eta1[i,n]>eta1[i-1,n] and eta1[i,n]>eta1[i+1,n]) :
            extrema.append([i,n])

#---------------#

extr_speed1=np.zeros(len(extrema))
for m in range(len(extrema)):
    i=extrema[m][0]; n=extrema[m][1]
    if n<len(t)-1:
        dist=len(x)
        for m2 in range(len(extrema)):
            if extrema[m2][1]==n+1 :
                if np.abs(extrema[m2][0]-i)<dist:
                    i2=extrema[m2][0]
                    dist=np.abs(extrema[m2][0]-i)
        
        extr_speed1[m]=(x[i2]-x[i])/dt
        
#---------------#

c_grad_mes=np.zeros(Nt-2)
dist0=np.max(x)*10**2
LM=linear_model.LinearRegression()
for n in range(Nt-2):
    m0=0
    dist=dist0
    for m in range(len(extrema)):
        if (extrema[m][1]==n
            and np.abs(x[extrema[m][0]])<dist):
            dist=np.abs(x[extrema[m][0]])
            m0=m
            
    y=np.array([  
                extr_speed1[m0-1] , 
                extr_speed1[m0] ,
                extr_speed1[m0+1]  
               ])
    
    X=np.array([ 
                [1,x[extrema[m0-1][0]],x[extrema[m0-1][0]]**2] , 
                [1,x[extrema[m0][0]],x[extrema[m0][0]]**2] ,
                [1,x[extrema[m0+1][0]],x[extrema[m0+1][0]]**2] 
               ])
    
    c_grad_mes[n]=LM.fit(X,y).coef_[1]
    del X, y
    
#---------------#

def gradient_fit22( time , t0 , Tau ):

    return 2*((time-t0)/(Tau**2+(time-t0)**2))*( 1-3.25/((k0*c0*Tau/4)*(1+((time-t0)/Tau)**2)) )

#---------------#

STD_FIT=np.std(c_grad_mes-c_grad_expect)

nstart1=0
nstart2=578

# tfoc, A_ratio and L are predicted values based on the best fit
tfoc1=np.zeros(Nt-3)
tau_abs1=np.zeros(Nt-3)
A_ratio1=np.zeros(Nt-3)
L1=np.zeros(Nt-3)
cov1=np.full((Nt-3,2,2),np.inf)
tfoc2=np.zeros(Nt-3)
tau_abs2=np.zeros(Nt-3)
A_ratio2=np.zeros(Nt-3)
L2=np.zeros(Nt-3)
cov2=np.full((Nt-3,2,2),np.inf)

nstart=nstart1
for nend in range(nstart+3,Nt-3):
    popt, pcov = curve_fit(gradient_fit22, t[nstart:nend], c_grad_mes[nstart:nend],p0=[0,10*2*np.pi/(k0*c0)]
                       ,sigma=STD_FIT*np.ones(nend-nstart), bounds=([-np.inf,0],np.inf))
    tfoc1[nend]=popt[0]
    tau_abs1[nend]=popt[1]
    L1[nend]=(c0*popt[1]/(4*k0))**0.5
    A_ratio1[nend]=(1+((t[nend]+popt[0])/popt[1])**2)**0.25
    cov1[nend,:,:]=pcov

nstart=nstart2
for nend in range(nstart+3,Nt-3):
    popt, pcov = curve_fit(gradient_fit22, t[nstart:nend], c_grad_mes[nstart:nend],p0=[0,10*2*np.pi/(k0*c0)]
                       ,sigma=STD_FIT*np.ones(nend-nstart), bounds=([-np.inf,0],np.inf))
    tfoc2[nend]=popt[0]
    tau_abs2[nend]=popt[1]
    L2[nend]=(c0*popt[1]/(4*k0))**0.5
    A_ratio2[nend]=(1+((t[nend]+popt[0])/popt[1])**2)**0.25
    cov2[nend,:,:]=pcov
    

# Store the data
np.savez('GWP_2D_Lk10_phi%.5f' % phi0,tfoc1=tfoc1,tau_abs1=tau_abs1
         ,A_ratio1=A_ratio1,L1=L1,tfoc2=tfoc2,tau_abs2=tau_abs2,A_ratio2=A_ratio2,L2=L2,
         cov1_00=cov1[:,0,0],cov1_11=cov1[:,1,1],cov1_10=cov1[:,1,0],cov1_01=cov1[:,0,1],
         cov2_00=cov2[:,0,0],cov2_11=cov2[:,1,1],cov2_10=cov2[:,1,0],cov2_01=cov2[:,0,1])

# Computation time
computation_time=time.time()-start