#------------------------------------------------------------------------------#
# Evaluation of specular velocity profile of a non-linear Gaussian wave packet #
#------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop

#---------------#
# The NLSE is the Non-Linear Schrödinger Equation
# This code first executes "NonLin_Init_Integr.py"
# which processes the initialization and integration.
# The chosen value of amp_choice must be set inside
# the "NonLin_Init_Integr.py" script before executing.
# This script processes all the steps to generate the data for Figure 7.
# This includes interpolation of the envelope,
# and measurement of crest/trough velocities.
#---------------#

# Go to the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Executing "NonLin_Init_Integr.py" (where amp_choice must be set before executing)

from NonLin_Init_Integr import *

## Imports (other imports are made in "Init_Integr_linear.py")

from scipy import interpolate
from scipy.signal import argrelextrema

## Interpolation (should take less than 10 seconds)

#new temporal axis (here we focus around the time of minimum specular velocity gradient)
if amp_choice==0.2:
    nt2=2019+1977+20; nt3=5449 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=2019+1977-20 # and that 2019+1977 is the time of minimum specular velocity gradient (using the data from Fig6) 
elif amp_choice==0.8:
    nt2=1870+2049+20; nt3=5300 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=1870+2049-20 # and that 1870+2049 is the time of minimum specular velocity gradient (using the data from Fig6)
elif amp_choice==1.1:
    nt2=1648+2242+20; nt3=5078 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=1648+2242-20 # and that 1648+2242 is the time of minimum specular velocity gradient (using the data from Fig6)
elif amp_choice==np.sqrt(2):
    nt2=1240+2480+20; nt3=4670 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=1240+2480-20 # and that 1240+2480 is the time of minimum specular velocity gradient (using the data from Fig6)
elif amp_choice==1.6:
    nt2=936+2644+20; nt3=4366 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=936+2644-20 # and that 936+2644 is the time of minimum specular velocity gradient (using the data from Fig6)
elif amp_choice==1.8:
    nt2=601+2838+20; nt3=4031 # one can check that nt3 correspond to non-linear focusing time
    dnt=1; nt1=601+2838-20 # and that 601+2838 is the time of minimum specular velocity gradient (using the data from Fig6)
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

## Measuring crest/trough velocities (should take less than 10 seconds)

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

## Repeating the operation to have more points on the specular velocity profile (should take less then 3 minutes)
            
nn=20;
x_ccr=[]; ccr_adim=[]; t_adim=t_axis[nn]*k0*c0/(2*np.pi)
DPHI=np.linspace(0,2*np.pi,num=100)
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

    # store the results
    for m in range(len(extrema)):
        if extrema[m][1]==nn:
            x_ccr.append(x_axis[extrema[m][0]]*k0/(2*np.pi))
            ccr_adim.append((extr_speed1[m]-c0/2)/c0)


## Putting all the results together and saving

x_ccr=np.array(x_ccr); ccr_adim=np.array(ccr_adim)
index=np.argsort(x_ccr); x_ccr=x_ccr[index]; ccr_adim=ccr_adim[index]

if amp_choice==0.2:
    np.savez('Fig7_out_linear.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==0.8:
    np.savez('Fig7_out_0p11.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.1:
    np.savez('Fig7_out_0p16.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==np.sqrt(2):
    np.savez('Fig7_out_0p23.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.6:
    np.savez('Fig7_out_0p28.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
elif amp_choice==1.8:
    np.savez('Fig7_out_0p33.npz',x_ccr=np.array(x_ccr),ccr_adim=np.array(ccr_adim),t_adim=t_adim,tau=tau,L=L,alkp=A*np.sqrt(2))
