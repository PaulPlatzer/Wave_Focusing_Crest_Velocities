#-----------------------------------------------------------------------------#
#
# Plotting estimations of focusing parameters of linear Gaussain wave packets #
#
#-----------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------#
# This piece of code is to be used alongside with "Linear_Init_Integr.py"
# (setting width_choice=2.36) that produces "eta1",
# the sea-surface elevation of a linear Gaussian wave packet,
# and with "Fig5_data.py" that estimates focusing parameters based
# on crest and trough velocity measurements.
# Here we simply load the files that were created by "Fig5_data.py",
# and plot the results.
#---------#

## Put yourself in the directory where the "GWP_k0Lf*.npz" files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Linear_Simulations/

## Imports
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#### RECOMPUTING SOME QUANTITIES (already computed in the previous codes)

## Define the wavenumber and angular frequency axis 
k0=0.02; kxmin=0.001; kxmax=0.05; Nkx=150
kx=np.linspace(kxmin,kxmax,Nkx)
g=9.81 #acceleration of gravity
omega=np.sqrt(g*kx)
dkx=np.ones(Nkx,float)*(kx[1]-kx[0]) #wavenumber increment
## Define the group size at focus L_f
L=(2.36)*(k0)**-1
## Calculate the group spectrum and associated amplitudes
G=L*((2*np.pi)**-0.5)*np.exp(-(L**2)*0.5*(kx-k0)**2)
Af=5
a=Af*G*dkx/np.sum(G*dkx)
## Compute some quantities related to derivatives of omega at k0
cg0=(np.sqrt(g/k0))*(0.5) #group velocity (first derivative of the phase velocity c_0)
par2om=(np.sqrt(g*k0)/(k0**2))*(-0.25) #second derivative of c_0 with respect to the wavenumber k
tau=L**2/par2om #linear group contraction time-scale (negative)
## Define the temporal and spatial axis
dt=0.13*2*np.pi/np.max(omega)
Nt=4*int(np.abs(tau)/dt)
t=np.linspace(-dt*Nt,0,Nt)
Lx=4*(2*np.pi/k0)
dx=0.000025*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)
## Compute some quantities to be used later on
Spread_x=np.sqrt(L**2+(L**(-2))*(par2om*t)**2)
c0=(np.sqrt(g/k0))
ccr_1ord=cg0+0.5*k0*par2om-c0
ccr_grad_1ord=(-c0+cg0)*k0*t/tau
ccr_2ord=c0-cg0-0.5*k0*par2om
ccr_grad_2ord=k0*(1.5*par2om*k0+4*c0-4*cg0)*t/tau
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)
## Define the starting and ending points of measurements of crest velocities
nstart1=0
nstart2=85
nstart3=85+42
nend=Nt-3


#### LOADING THE DATA

## Call all the files
fnames = glob('Fig5_data/GWP_k0Lf2_phi*.npz') # to open the files created by the user
#fnames = glob('Fig5_article_data/GWP_2D_Lk02_phi*.npz') # to open the files used in the article
fnames.sort()
Nfi=len(fnames)

## Concatenate  output results
# initialize
TFOC1=np.zeros((Nfi,Nt-3)); TAUABS1=np.zeros((Nfi,Nt-3))
ARATIO1=np.zeros((Nfi,Nt-3)); LL1=np.zeros((Nfi,Nt-3))
TFOC2=np.zeros((Nfi,Nt-3)); TAUABS2=np.zeros((Nfi,Nt-3))
ARATIO2=np.zeros((Nfi,Nt-3)); LL2=np.zeros((Nfi,Nt-3))
TFOC3=np.zeros((Nfi,Nt-3)); TAUABS3=np.zeros((Nfi,Nt-3))
ARATIO3=np.zeros((Nfi,Nt-3)); LL3=np.zeros((Nfi,Nt-3))
# read
for n in range(Nfi):
    data=np.load(fnames[n])
    TFOC1[n,:]=data['tfoc1']; TAUABS1[n,:]=data['tau_abs1']
    ARATIO1[n,:]=data['A_ratio1']; LL1[n,:]=data['L1']
    TFOC2[n,:]=data['tfoc2']; TAUABS2[n,:]=data['tau_abs2']
    ARATIO2[n,:]=data['A_ratio2']; LL2[n,:]=data['L2']
    TFOC3[n,:]=data['tfoc3']; TAUABS3[n,:]=data['tau_abs3']
    ARATIO3[n,:]=data['A_ratio3']; LL3[n,:]=data['L3']
    
    
#### PLOT

## Estimation of focusing time

plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
# Objective
ax.plot(t[nstart1:nend]*c0*k0/(2*np.pi),np.zeros(len(t[nstart1:nend])),'k',label=r'$t_{f}\, /T_p=0$')
# Median
ax.plot(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.median(TFOC1,axis=0)[nstart1+2:nend]*c0*k0/(2*np.pi), '-r',
        linewidth=2,label=r'Median of estimations, $t_{start}=-4|\tau|$')
ax.plot(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.median(TFOC2,axis=0)[nstart2+2:nend]*c0*k0/(2*np.pi), '-b',
        linewidth=2,label=r'Median of estimations, $t_{start}=-2|\tau|$')
ax.plot(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.median(TFOC3,axis=0)[nstart3+2:nend]*c0*k0/(2*np.pi), '-g',
        linewidth=2,label=r'Median of estimations, $t_{start}=-|\tau|$')
# Quantiles : 2.5% and 97.5%
ax.fill_between(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.percentile(TFOC1,axis=0,q=2.5)[nstart1+2:nend]*c0*k0/(2*np.pi),
                np.percentile(TFOC1,axis=0,q=97.5)[nstart1+2:nend]*c0*k0/(2*np.pi),
                color='r', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-4|\tau|$')
ax.fill_between(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.percentile(TFOC2,axis=0,q=2.5)[nstart2+2:nend]*c0*k0/(2*np.pi),
                np.percentile(TFOC2,axis=0,q=97.5)[nstart2+2:nend]*c0*k0/(2*np.pi),
                color='b', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-2|\tau|$')
ax.fill_between(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.percentile(TFOC3,axis=0,q=2.5)[nstart3+2:nend]*c0*k0/(2*np.pi),
                np.percentile(TFOC3,axis=0,q=97.5)[nstart3+2:nend]*c0*k0/(2*np.pi),
                color='g', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-|\tau|$')
ftsz=26#fontsize
ax.set_title(r'Focusing time, $k_0L_f=2.36$',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel('$t_{f}\, /T_p$',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
ax.set_ylim([-3,3])
ax.set_xlim([-4*np.abs(tau)*c0*k0/(2*np.pi),0])
fig.set_size_inches([8,6])
plt.show()


## Estimation of focusing amplitude

plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
# Objective
ax.plot(t[nstart1:nend]*c0*k0/(2*np.pi),(1+(t[nstart1:nend]/tau)**2)**0.25,'k',
        label=r'$A_{f}\, /A(t_{end})=\left(1+\left(t_{end}/\tau\right)^2\right)^{1/4}$')
# Median
ax.plot(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.median(ARATIO1,axis=0)[nstart1+2:nend], '-r',
        linewidth=2,label=r'Median of estimations, $t_{start}=-4|\tau|$')
ax.plot(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.median(ARATIO2,axis=0)[nstart2+2:nend], '-b',
        linewidth=2,label=r'Median of estimations, $t_{start}=-2|\tau|$')
ax.plot(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.median(ARATIO3,axis=0)[nstart3+2:nend], '-g',
        linewidth=2,label=r'Median of estimations, $t_{start}=-|\tau|$')
# Quantiles : 2.5% and 97.5%)
ax.fill_between(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.percentile(ARATIO1,axis=0,q=2.5)[nstart1+2:nend],
                np.percentile(ARATIO1,axis=0,q=97.5)[nstart1+2:nend],
                color='r', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-4|\tau|$')
ax.fill_between(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.percentile(ARATIO2,axis=0,q=2.5)[nstart2+2:nend],
                np.percentile(ARATIO2,axis=0,q=97.5)[nstart2+2:nend],
                color='b', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-2|\tau|$')
ax.fill_between(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.percentile(ARATIO3,axis=0,q=2.5)[nstart3+2:nend],
                np.percentile(ARATIO2,axis=0,q=97.5)[nstart3+2:nend],
                color='g', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-|\tau|$')
ftsz=26 #fontsize
ax.set_title(r'Amplitude, $k_0L_f=2.36$',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel(r'$A_{f}\, /A(t_{end})$',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
ax.set_ylim([1,2.5])
ax.set_xlim([-4*np.abs(tau)*c0*k0/(2*np.pi),0])
fig.set_size_inches([8,6])
plt.show()


## Estimation of focusing spatial width

plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
# Objective
ax.plot(t[nstart1:nend]*c0*k0/(2*np.pi),k0*L*np.ones(len(t[nstart1:nend])),'k',label=r'$L_{f}k_0=2.36$')
# Median
ax.plot(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.median(LL1,axis=0)[nstart1+2:nend]*k0, '-r',
        linewidth=2,label=r'Median of estimations, $t_{start}=-4|\tau|$')
ax.plot(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.median(LL2,axis=0)[nstart2+2:nend]*k0, '-b',
        linewidth=2,label=r'Median of estimations, $t_{start}=-2|\tau|$')
ax.plot(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.median(LL3,axis=0)[nstart3+2:nend]*k0, '-g',
        linewidth=2,label=r'Median of estimations, $t_{start}=-|\tau|$')
# Quantiles : 2.5% and 97.5%
ax.fill_between(t[nstart1+2:nend]*c0*k0/(2*np.pi), np.percentile(LL1,axis=0,q=2.5)[nstart1+2:nend]*k0,
                np.percentile(LL1,axis=0,q=97.5)[nstart1+2:nend]*k0,
                color='r', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-4|\tau|$')
ax.fill_between(t[nstart2+2:nend]*c0*k0/(2*np.pi), np.percentile(LL2,axis=0,q=2.5)[nstart2+2:nend]*k0,
                np.percentile(LL2,axis=0,q=97.5)[nstart2+2:nend]*k0,
                color='b', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-2|\tau|$')
ax.fill_between(t[nstart3+2:nend]*c0*k0/(2*np.pi), np.percentile(LL3,axis=0,q=2.5)[nstart3+2:nend]*k0,
                np.percentile(LL3,axis=0,q=97.5)[nstart3+2:nend]*k0,
                color='g', alpha=.2, label=r'2.5% to 97.5% percentiles, $t_{start}=-|\tau|$')
ftsz=26 #fontsize
ax.set_ylim([2.27,2.45])
ax.set_xlim([-4*np.abs(tau)*c0*k0/(2*np.pi),0])
ax.set_title(r'Group size, $k_0L_f=2.36$',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel(r'$L_{f}k_0 $',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
fig.set_size_inches([8,6])
plt.show()
