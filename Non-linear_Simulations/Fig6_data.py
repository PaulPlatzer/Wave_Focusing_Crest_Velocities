#-------------------------------------------------------------------------------#
# Evaluation of specular velocity gradient of a non-linear Gaussian wave packet #
#-------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop
# This script uses up to approximately 11Go of RAM.

#---------------#
# The NLSE is the Non-Linear Schrödinger Equation
# This code first executes "NonLin_Init_Integr.py"
# which processes the initialization and integration.
# The chosen value of amp_choice must be set inside
# the "NonLin_Init_Integr.py" script before executing.
# This script processes all the steps to generate the data for Figure 6.
# This includes interpolation in space of the envelope, measurement of
# crest and trough velocities, and specular velocity gradient. 
#---------------#

# Go to the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Executing "NonLin_Init_Integr.py" (where amp_choice must be set before executing)

from NonLin_Init_Integr import *

## Imports

from scipy import interpolate
from scipy.signal import argrelextrema

## Interpolation (can take a few minutes)

#new temporal axis
if amp_choice==0.2:
    nt2=5449; nt3=5449 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=2019 # and this is thus -2.5|tau| before non-linear focusing
elif amp_choice==0.8:
    nt2=5300; nt3=5300 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=1870 # and this is thus -2.5|tau| before non-linear focusing
elif amp_choice==1.1:
    nt2=5078; nt3=5078 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=1648 # and this is thus -2.5|tau| before non-linear focusing
elif amp_choice==np.sqrt(2):
    nt2=4670; nt3=4670 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=1240 # and this is thus -2.5|tau| before non-linear focusing
elif amp_choice==1.6:
    nt2=4366; nt3=4366 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=936 # and this is thus -2.5|tau| before non-linear focusing
elif amp_choice==1.8:
    nt2=4031; nt3=4031 # one can check that these correspond to non-linear focusing time
    dnt=1; nt1=601 # and this is thus -2.5|tau| before non-linear focusing
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
        
## REPEATING this procedure to get more points (this takes between 10min and 20min)

DPHI=np.array([0.25,0.75,1.63])*np.pi
for dphi in DPHI:
    #interpolated envelope
    U_new=np.zeros((nt2-nt1+1,len(X_axis_new)),np.complex64) 
    for n in range(nt1,nt2+1):
        tck_re=interpolate.splrep(X_axis,np.real(np.fft.ifft(C[n,:]*np.exp(1j*dphi))),s=0)
        tck_im=interpolate.splrep(X_axis,np.imag(np.fft.ifft(C[n,:]*np.exp(1j*dphi))),s=0)
        U_new[n-nt1,:]=interpolate.splev(X_axis_new,tck_re,der=0)+1j*interpolate.splev(X_axis_new,tck_im,der=0)

    # Recalculating the associated sea-surface elevation
    eta1=np.real( (( (np.sqrt(2)/k0) * U_new * np.exp(1j*k0*x_axis) ).T)
                 *np.exp(1j*(-0.5*T_axis[nt1:nt2+1:dnt])) )

    # Finding crests/troughs
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
    
## Putting all the results together and saving

t_grad_agg=np.array(t_grad_agg); c_grad_agg=np.array(c_grad_agg)
index=np.argsort(t_grad_agg); t_grad_agg=t_grad_agg[index]; c_grad_agg=c_grad_agg[index]

if amp_choice==0.2:
    np.savez('Fig6_out_linear.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==0.8:
    np.savez('Fig6_out_0p11.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.1:
    np.savez('Fig6_out_0p16.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==np.sqrt(2):
    np.savez('Fig6_out_0p23.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.6:
    np.savez('Fig6_out_0p28.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.8:
    np.savez('Fig6_out_0p33.npz',t_grad_agg=t_grad_agg,c_grad_agg=c_grad_agg,tau=tau,L=L,alkp=A*np.sqrt(2))
