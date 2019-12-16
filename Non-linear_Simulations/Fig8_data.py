#------------------------------------------------------------------------------#
# Testing our crest-velocity-based method on a non-linear Gaussian wave packet #
#------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop
# This script uses up to approximately 11Go of RAM.

#---------------#
# The NLSE is the Non-Linear Schrödinger Equation
# This code first executes "NonLin_Init_Integr.py"
# which processes the initialization and integration.
# The value of amp_choice=1.1 must be set inside
# the "NonLin_Init_Integr.py" script before executing.
# The value of "N8" is set line 210 and greatly influences
# the computation time of this script.
# This script processes all the steps to generate the data for Figure 8.
# This includes interpolation in space of the envelope, measurement of
# crest/trough velocities and specular velocity gradient,
# and finally evaluation of our method. 
#---------------#

# Go to the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Executing "NonLin_Init_Integr.py" (where amp_choice must be set before executing)

from NonLin_Init_Integr import *

## Imports (other imports are made in "Init_Integr_linear.py")

from scipy import interpolate
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from Gauss import gauss

## Interpolation (can take a few minutes)

#new temporal axis
if amp_choice==1.1:
    nt2=5078; nt3=5078 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=1648 # and this is thus -2.5|tau| before non-linear focusing
else :
    print('ERROR: You need to set amp_choice=1.1')
t_axis=(T_axis[nt1:nt2+1:dnt]-T_axis[nt3])/w0
#new spatial axis
scale=300
dx_new=dx/scale; dX_new=dX/scale
X_axis_new=np.arange(-75,75,dX_new); x_axis=X_axis_new/(2*np.sqrt(2)*k0)
#new envelope
U_new=np.zeros((nt2-nt1+1,len(X_axis_new)),np.complex64)
for n in range(nt1,nt2+1):
    tck_re=interpolate.splrep(X_axis,np.real(np.fft.ifft(C[n,:])),s=0)
    tck_im=interpolate.splrep(X_axis,np.imag(np.fft.ifft(C[n,:])),s=0)
    U_new[n-nt1,:]=interpolate.splev(X_axis_new,tck_re,der=0)+1j*interpolate.splev(X_axis_new,tck_im,der=0)
#associated sea-surface elevation
eta1=np.real( (( (np.sqrt(2)/k0) * U_new * np.exp(1j*k0*x_axis) ).T)
             *np.exp(1j*(-0.5*T_axis[nt1:nt2+1:dnt])) )

## Recomputing some important parameters

par2om=-w0/(4*k0**2)
L=sig/(2*np.sqrt(2)*k0)
tau=L**2/par2om
c0=w0/k0
cg0=w0/(2*k0)
ccr_1ord=cg0+0.5*k0*par2om-c0
ccr_grad_1ord=(-c0+cg0)*k0*t_axis/tau
ccr_2ord=c0-cg0-0.5*k0*par2om
ccr_grad_2ord=k0*(1.5*par2om*k0+4*c0-4*cg0)*t_axis/tau
ccr_lapl_2ord=2*((c0-cg0+0.5*k0*par2om)*(t_axis/tau)**2-0.5*k0*par2om)*(k0)**2
Spread_x=np.sqrt(L**2+(L**(-2))*(par2om*t_axis)**2)
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)
c_grad_expect1ord=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)

## Measuring crest/trough velocities (can take a few minutes)

#finding the extrema
extrema2=argrelextrema(np.abs(eta1.T), np.greater, axis=1)
extrema=[]
for k in range(len(extrema2[0])):
    extrema.append([extrema2[1][k],extrema2[0][k]])
#computing the crest/trough velocities
nmoy=15 # number of time-steps over which the mean velicity is computed
extr_speed1=np.zeros(len(extrema))
for m in range(len(extrema)):
    i=extrema[m][0]; n=extrema[m][1]
    signe=np.sign(eta1[i,n])
    if n<=nmoy:
        i2=False
        for m2 in range(m,len(extrema)):
            if ( extrema[m2][1]==n+nmoy and extrema[m2][0]-i>0 
                    and np.sign(eta1[extrema[m2][0],extrema[m2][1]])==signe ):
                i2=extrema[m2][0]
                break
        if i2:
            extr_speed1[m]=(x_axis[i2]-x_axis[i])/(nmoy*dt*dnt)
        else:
            extr_speed1[m]=np.nan
    else:
        for m2 in range(len(extrema)):
            if ( extrema[m-m2][1]==n-nmoy and extrema[m-m2][0]-i<=0
                    and np.sign(eta1[extrema[m-m2][0],extrema[m-m2][1]])==signe ):
                i2=extrema[m-m2][0]
                break
        if i2:
            extr_speed1[m]=(x_axis[i]-x_axis[i2])/(nmoy*dt*dnt)
        else:
            extr_speed1[m]=np.nan

## Measure specular velocity gradient (only when the central crest/trough is close enough to the center)

# position of the largest crest/trough in space, over time
eta_abs_argmax=np.argmax(np.abs(eta1),axis=0)
# criterion for having the largest crest/trough close enough to the center of the WP
thresh_center= 4 * ((2*np.pi)/k0) * dt/(2*tp) 

# computing the velocity gradient
c_grad_mes=[]
t_grad_mes=[] # storing the time at which the gradient is measured
dist0=np.max(x_axis)*10**2
LM=linear_model.LinearRegression()
for n in range(len(t_axis)):
    if (-thresh_center/2<x_axis[eta_abs_argmax[n]]
        and x_axis[eta_abs_argmax[n]]<thresh_center/2):
        
        t_grad_mes.append(t_axis[n])
        
        m0=0
        dist=dist0
        for m in range(len(extrema)):
            if (extrema[m][1]==n
                and np.abs(x_axis[extrema[m][0]])<dist):
                dist=np.abs(x_axis[extrema[m][0]])
                m0=m
            
        y=np.array([  
                extr_speed1[m0-1] , 
                extr_speed1[m0] ,
                extr_speed1[m0+1]  
                   ])
    
        X=np.array([ 
                [1,x_axis[extrema[m0-1][0]],x_axis[extrema[m0-1][0]]**2] , 
                [1,x_axis[extrema[m0][0]],x_axis[extrema[m0][0]]**2] ,
                [1,x_axis[extrema[m0+1][0]],x_axis[extrema[m0+1][0]]**2] 
                   ])
    
        c_grad_mes.append(LM.fit(X,y).coef_[1])
        del X, y
# aggregating gradients that correspond to the same crest/trough through a mean
c_grad_agg=[] # "agg" stands for "aggregate"
t_grad_agg=[]
c_=[c_grad_mes[0]]
t_=[t_grad_mes[0]]
for n in range(len(t_grad_mes)-1):
    if t_grad_mes[n+1]-t_grad_mes[n]<2*dt:
        c_.append(c_grad_mes[n+1])
        t_.append(t_grad_mes[n+1])
    else:
        c_grad_agg.append(np.mean(np.array(c_)))
        t_grad_agg.append(np.mean(np.array(t_)))
        c_=[c_grad_mes[n+1]]
        t_=[t_grad_mes[n+1]]
    if n+1==len(t_grad_mes)-1:
        c_grad_agg.append(np.mean(np.array(c_)))
        t_grad_agg.append(np.mean(np.array(t_)))

## Compute some extra elements useful for the plots and the fit

#measuring the amplitude over time
A_adim=np.amax(np.abs(U_new),axis=1)
#measuring the non-linear focusing width of the Gaussian wave packet
fit=curve_fit(gauss,x_axis,np.abs(U_new[-1,:]),p0=[1,0,1])
L_foc=fit[0][2]
#to initialize the fit
STD_FIT=np.std(np.array(c_grad_agg))
# fuction used for the fit
def gradient_fit( time , t0 , Tau ):
    return 2*((time-t0)/(Tau**2+(time-t0)**2))*( 1-3.25/((k0*c0*Tau/4)*(1+((time-t0)/Tau)**2)) )

## Apply the forecasting method

#Initialize      
TFOC=[]; TAU_ABS=[]; A_RATIO=[]; L_ESTIM=[]
T_GRAD=[]; C_GRAD=[]

#Apply the method
nstart=9
Nts=len(t_grad_agg)
# tfoc, A_ratio and L are predicted values based on the best fit
tfoc=np.zeros(Nts); tau_abs=np.zeros(Nts)
A_ratio=np.zeros(Nts); L_estim=np.zeros(Nts)

for nend in range(nstart+3,Nts):
    n=nend-nstart-3
    popt, pcov = curve_fit(gradient_fit, np.array(t_grad_agg)[nstart:nend], np.array(c_grad_agg)[nstart:nend],p0=[0,10*2*np.pi/(k0*c0)]
                   ,sigma=STD_FIT*np.ones(nend-nstart),bounds=([-np.inf,1/w0],[np.inf,40000/w0]))
    tfoc[nend]=popt[0]
    tau_abs[nend]=np.abs(popt[1])
    L_estim[nend]=(c0*np.abs(popt[1])/(4*k0))**0.5
    A_ratio[nend]=(1+((np.array(t_grad_agg)[nend]+popt[0])/popt[1])**2)**0.25
    
TFOC.append(tfoc); TAU_ABS.append(tau_abs)
A_RATIO.append(A_ratio); L_ESTIM.append(L_estim)
T_GRAD.append(np.array(t_grad_agg)); C_GRAD.append(np.array(c_grad_agg))

## Repeating this operation N8 times with different phases to get more points

N8=20 # to generate the data for figure 8 of the article we set N8=4*44

DPHI=np.linspace(np.pi/(N8+1),np.pi,num=N8,endpoint=0)
avancement=0
for l in range(N8):
    #interpolated envelope
    U_new=np.zeros((nt2-nt1+1,len(X_axis_new)),np.complex64) 
    for n in range(nt1,nt2+1):
        tck_re=interpolate.splrep(X_axis,np.real(np.fft.ifft(C[n,:]*np.exp(1j*DPHI[l]))),s=0)
        tck_im=interpolate.splrep(X_axis,np.imag(np.fft.ifft(C[n,:]*np.exp(1j*DPHI[l]))),s=0)
        U_new[n-nt1,:]=interpolate.splev(X_axis_new,tck_re,der=0)+1j*interpolate.splev(X_axis_new,tck_im,der=0)

    # Recalculating the associated sea-surface elevation
    eta1=np.real( (( (np.sqrt(2)/k0) * U_new * np.exp(1j*k0*x_axis) ).T)
                 *np.exp(1j*(-0.5*T_axis[nt1:nt2+1:dnt])) )

    extrema2=argrelextrema(np.abs(eta1.T), np.greater, axis=1)
    extrema=[]
    for k in range(len(extrema2[0])):
        extrema.append([extrema2[1][k],extrema2[0][k]])

    # compute velocity
    nmoy=15 # number of time-steps over which the mean velicity is computed
    extr_speed1=np.zeros(len(extrema))
    for m in range(len(extrema)):
        i=extrema[m][0]; n=extrema[m][1]
        signe=np.sign(eta1[i,n])
        if n<=nmoy:
            i2=False
            for m2 in range(m,len(extrema)):
                if ( extrema[m2][1]==n+nmoy and extrema[m2][0]-i>0 
                        and np.sign(eta1[extrema[m2][0],extrema[m2][1]])==signe ):
                    i2=extrema[m2][0]
                    break
            if i2:
                extr_speed1[m]=(x_axis[i2]-x_axis[i])/(nmoy*dt*dnt)
            else:
                extr_speed1[m]=np.nan
        else:
            for m2 in range(len(extrema)):
                if ( extrema[m-m2][1]==n-nmoy and extrema[m-m2][0]-i<=0
                        and np.sign(eta1[extrema[m-m2][0],extrema[m-m2][1]])==signe ):
                    i2=extrema[m-m2][0]
                    break
            if i2:
                extr_speed1[m]=(x_axis[i]-x_axis[i2])/(nmoy*dt*dnt)
            else:
                extr_speed1[m]=np.nan

    # position of the largest crest/trough in space, over time
    eta_abs_argmax=np.argmax(np.abs(eta1),axis=0)
    # criterion for having the largest crest/trough close enough to the center of the WP
    thresh_center= 4 * ((2*np.pi)/k0) * dt/(2*tp) 
    
    # measuring the center gradient
    c_grad_mes=[]
    t_grad_mes=[]
    dist0=np.max(x_axis)*10**2
    LM=linear_model.LinearRegression()

    for n in range(len(t_axis)):
        if (-thresh_center/2<x_axis[eta_abs_argmax[n]]
            and x_axis[eta_abs_argmax[n]]<thresh_center/2):
        
            t_grad_mes.append(t_axis[n])
        
            m0=0
            dist=dist0
            for m in range(len(extrema)):
                if (extrema[m][1]==n
                    and np.abs(x_axis[extrema[m][0]])<dist):
                    dist=np.abs(x_axis[extrema[m][0]])
                    m0=m
            
            y=np.array([  
                extr_speed1[m0-1] , 
                extr_speed1[m0] ,
                extr_speed1[m0+1]  
                   ])
    
            X=np.array([ 
                [1,x_axis[extrema[m0-1][0]],x_axis[extrema[m0-1][0]]**2] , 
                [1,x_axis[extrema[m0][0]],x_axis[extrema[m0][0]]**2] ,
                [1,x_axis[extrema[m0+1][0]],x_axis[extrema[m0+1][0]]**2] 
                   ])
    
            c_grad_mes.append(LM.fit(X,y).coef_[1])
            del X, y

    c_grad_agg=[]
    t_grad_agg=[]
    c_=[c_grad_mes[0]]
    t_=[t_grad_mes[0]]
    for n in range(len(t_grad_mes)-1):
        if t_grad_mes[n+1]-t_grad_mes[n]<2*dt:
            c_.append(c_grad_mes[n+1])
            t_.append(t_grad_mes[n+1])
        else:
            c_grad_agg.append(np.mean(np.array(c_)))
            t_grad_agg.append(np.mean(np.array(t_)))
            c_=[c_grad_mes[n+1]]
            t_=[t_grad_mes[n+1]]
        if n+1==len(t_grad_mes)-1:
            c_grad_agg.append(np.mean(np.array(c_)))
            t_grad_agg.append(np.mean(np.array(t_)))

    nstart=9
    Nts=len(t_grad_agg)
    # tfoc, A_ratio and L are predicted values based on the best fit
    tfoc=np.zeros(Nts)
    tau_abs=np.zeros(Nts)
    A_ratio=np.zeros(Nts)
    L_estim=np.zeros(Nts)

    for nend in range(nstart+3,Nts):
        n=nend-nstart-3
        popt, pcov = curve_fit(gradient_fit, np.array(t_grad_agg)[nstart:nend], np.array(c_grad_agg)[nstart:nend],p0=[0,10*2*np.pi/(k0*c0)]
                       ,sigma=STD_FIT*np.ones(nend-nstart),bounds=([-np.inf,1/w0],[np.inf,40000/w0]))
        tfoc[nend]=popt[0]
        tau_abs[nend]=np.abs(popt[1])
        L_estim[nend]=(c0*np.abs(popt[1])/(4*k0))**0.5
        A_ratio[nend]=(1+((np.array(t_grad_agg)[nend]+popt[0])/popt[1])**2)**0.25
    
    TFOC.append(tfoc)
    TAU_ABS.append(tau_abs)
    A_RATIO.append(A_ratio)
    L_ESTIM.append(L_estim)
    T_GRAD.append(np.array(t_grad_agg))
    C_GRAD.append(np.array(c_grad_agg))
    
## Saving

np.savez('Fig8_out.npz',TFOC=TFOC,TAU_ABS=TAU_ABS,A_RATIO=A_RATIO,
         L_ESTIM=L_ESTIM,T_GRAD=T_GRAD,C_GRAD=C_GRAD,tau=tau,L=L,
         t_axis=t_axis,A_adim=A_adim,L_foc=L_foc)
