#-------------------------------------------------------------------#
# Plotting crest/trough velocities of a linear Gaussian wave packet #
#-------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop
# This script uses up to approximately 7Go of RAM.

#---------------#
# We set k_0L_f=10
# We plot the crest velocities of a wave pacekt of Gaussian spectrum under
# linear evolution at different times (before focusing, near focusing, after focusing).
# The phase is uniform, constant, and equal to zero.
#---------------#

## Choose to plot 'prior to focusing', 'nearly focused' or 'after focusing'
plot_time='after' # either 'prior' or 'focus' or 'after'

## Imports
import numpy as np
from scipy.signal import argrelextrema
from sklearn import linear_model
import matplotlib.pyplot as plt

## Define the wavenumber axis and associated pulse
k0=0.02; kxmin=0.005; kxmax=0.037
Nkx=150; kx=np.linspace(kxmin,kxmax,Nkx)
g=9.81 #acceleration of gravity
omega=np.sqrt(g*kx)
dkx=np.ones(Nkx,float)*(kx[1]-kx[0]) #wavenumber increment
Tp=2*np.pi/np.sqrt(k0*g) #peak period

## Define the group size at focus L_f
L=10*(k0)**-1

## Calculate the group spectrum and associated amplitudes
G=L*((2*np.pi)**-0.5)*np.exp(-(L**2)*0.5*(kx-k0)**2)
Af=5
a=Af*G*dkx/np.sum(G*dkx)

## Define the phase
phi0=0
phi=phi0*np.ones(Nkx) # uniform and constant

## Compute some quantities related to derivatives of omega at k0
cg0=(np.sqrt(g/k0))*(0.5) #group velocity (first derivative of the phase velocity c_0)
par2om=(np.sqrt(g*k0)/(k0**2))*(-0.25) #second derivative of c_0 with respect to the wavenumber k
tau=L**2/par2om #linear group contraction time-scale (negative)

## Define the temporal and spatial axis
dt=0.13*2*np.pi/np.max(omega)
if plot_time=='prior':
    tmin=-63*Tp; tmax=-61*Tp
elif plot_time=='after':
    tmin=61*Tp; tmax=63*Tp
elif plot_time=='focus':
    tmin=-1*Tp; tmax=1*Tp
Nt=int((tmax-tmin)/dt)+1
t=np.linspace(tmin,tmax,Nt)
dt=t[1]-t[0]

## Define the spatial axis
Lx=8*(2*np.pi/k0)
dx=0.000015*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)

## Calculate the elevation (should take less than one minute)
eta1=np.real( 
    np.tensordot( np.tensordot(a*np.exp(1j*phi),np.ones(Nx),axes=0)*
        np.exp( 1j*np.tensordot(kx,x,axes=0) ), 
    np.exp( -1j*np.tensordot(omega-cg0*kx,t,axes=0) ),
            axes=[0,0] )
                    )

## Compute some quantities to be used later on
#wave packet's spatial width as a function of time (analytical)
Spread_x=np.sqrt(L**2+(L**(-2))*(par2om*t)**2)
#phase velocity
c0=(np.sqrt(g/k0))
#specular velocity (1st order term, analytical)
ccr_1ord=cg0+0.5*k0*par2om-c0 
#specular velocity gradient at the center of the packet (1st order term, analytical) 
ccr_grad_1ord=(-c0+cg0)*k0*t/tau
#specular velocity (2nd order term, analytical)
ccr_2ord=c0-cg0-0.5*k0*par2om
#specular velocity gradient at the center of the packet (2nd order term, analytical) 
ccr_grad_2ord=k0*(1.5*par2om*k0+4*c0-4*cg0)*t/tau
#specular velocity laplacian at the center of the packet (2nd order term, analytical) 
ccr_lapl_2ord=2*((c0-cg0+0.5*k0*par2om)*(t/tau)**2-0.5*k0*par2om)*(k0)**2
#specular velocity gradient at the center of the packet as a function of time (analytical)
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)

## Store the position in space and time of crests and troughs (this should take less than 10 seconds)
extrema2=argrelextrema(np.abs(eta1.T), np.greater_equal, axis=1)
extrema=[]
for k in range(len(extrema2[0])):
    extrema.append([extrema2[1][k],extrema2[0][k]])
del extrema2

## Compute the velocity of crests and troughs (this should take less than 10 seconds)
extr_speed1=np.zeros(len(extrema))
for m in range(len(extrema)):
    i=extrema[m][0]; n=extrema[m][1]
    if n<len(t)-1:
        dist=len(x)
        for m2 in range(len(extrema)):
            if extrema[m2][1]==n+1 :
                if (extrema[m2][0]>=i) and (extrema[m2][0]-i<dist):
                    i2=extrema[m2][0]
                    dist=np.abs(extrema[m2][0]-i)
        extr_speed1[m]=(x[i2]-x[i])/dt
        
## Plot (a,b,c)
n=10
if plot_time=='prior':
    title_phrase='$T_p$ (prior to focusing)'
elif plot_time=='focus':
    title_phrase='$T_p$ (nearly focused)'
elif plot_time=='after':
    title_phrase='$T_p$ (after focusing)'
plt.clf(); fig=plt.figure(); fig.add_axes()
ax=fig.add_subplot(111)
ax.plot(x*k0/(2*np.pi),eta1[:,n]/Af,'-k',label="Group elevation")
for m in range(len(extrema)):
    if extrema[m][1]==n:
        if (np.abs(x[extrema[m][0]]-x[0])+c0*dt)>np.pi/k0 and (np.abs(x[extrema[m][0]]-x[-1])+c0*dt)>np.pi/k0: # to prevent boundary effects
            if extr_speed1[m]>c0/2:
                arr_col='r'
            else:
                arr_col='b'
            arrow=ax.arrow(x[extrema[m][0]]*k0/(2*np.pi),eta1[extrema[m][0],n]/Af,170*(extr_speed1[m]-c0/2)*k0/(2*np.pi),0,
                     head_width=0.3/Af, head_length=15*k0/(2*np.pi), fc=arr_col, ec=arr_col)
ax.plot([0,0],[0,0],'-k',label="$c_{cr}-c_0$ (numerical)")
ax.set_xlabel('$(x-x_c)/\lambda_0$',fontsize=16);
ax.set_ylabel('$\eta/A$',fontsize=16);
if int(np.log(np.abs(t[n]/Tp))/np.log(10))<-4:
    num1=(t[n]/Tp)*10**(-int(np.log(np.abs(t[n]/Tp))/np.log(10))+1)
    num2=int(np.log(np.abs(t[n]/Tp))/np.log(10))-1
    plt.title('$t=$'+str(num1)[0:4]+'e'+str(num2)+' * '+title_phrase,fontsize=16)
else:
    plt.title('$t=$'+str(t[n]/Tp)[0:5]+title_phrase,fontsize=16)
ax.set_xlim(-1500*k0/(2*np.pi),1500*k0/(2*np.pi))
ax.set_ylim(-1.1,1.1)
fig.set_size_inches(5,4)
ax.tick_params(length=7,labelsize=16,direction='out',pad=10)
plt.show()

##Plot (d,e,f)
fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(x*k0/(2*np.pi),((ccr_1ord+ccr_grad_1ord[n]*x)/((k0*Spread_x[n])**2)
           +(ccr_grad_2ord[n]*x+0.5*ccr_lapl_2ord[n]*x**2)/((k0*Spread_x[n])**4))/c0,'k')
ax.plot(x*k0/(2*np.pi),((ccr_1ord+ccr_grad_1ord[n]*x)/((k0*Spread_x[n])**2))/c0,'--k')
for m in range(len(extrema)):
    if extrema[m][1]==n:
        if (np.abs(x[extrema[m][0]]-x[0])+c0*dt)>np.pi/k0 and (np.abs(x[extrema[m][0]]-x[-1])+c0*dt)>np.pi/k0: # to prevent boundary effects
            ax.plot(x[extrema[m][0]]*k0/(2*np.pi),(extr_speed1[m]-c0/2)/c0,'ok',markersize=2.8,label='c1='+str(extr_speed1[m]))
ax.set_xlabel('$(x-x_c)/\lambda_0$',fontsize=16);
ax.set_ylabel("$(c_{cr}-c_0)/c_0$",fontsize=16);
ax.set_xlim(-1500*k0/(2*np.pi),1500*k0/(2*np.pi))
ax.set_ylim(-0.06,0.06)
ax.tick_params(length=7,labelsize=16,direction='out',pad=10)
fig.set_size_inches(5,4)
plt.show()