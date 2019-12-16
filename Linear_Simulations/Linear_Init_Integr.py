#------------------------------------------------------------------------------#
# Numerical simulation of sea-surface elevation of linear Gaussian wave packet #
#------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop
# /!\ this piece of code is not to be run independently but inside other codes such as "Fig5_data.py"

#---------------#
# Important parameters:
# k_0L_f=2.36 or 5 or 10 (set the value of "width_choice")
# simulation from t=-4|\tau| or -2|\tau| or -|\tau| (depending on the value of "width_choice")
# to t=0 where 0 is the linear focusing time
# The phase is uniform and constant,
# it can be random (set "random_phase=1") or 0 (set "random_phase=0")
#---------------#

random_phase=1 # set "1" for random phase, or "0" for phi=0
width_choice=2.36 # choose the value of k_0L_f

## Imports
import numpy as np

## Define the wavenumber and angular frequency axis 
if width_choice==2.36:
    k0=0.02; kxmin=0.001; kxmax=0.05
elif width_choice==5:
    k0=0.02; kxmin=0.005; kxmax=0.037
elif width_choice==10:
    k0=0.02; kxmin=0.005; kxmax=0.037
else:
    print('ERROR: width_choice must be set to either 2.36 or 5 or 10')
Nkx=150; kx=np.linspace(kxmin,kxmax,Nkx)
g=9.81 #acceleration of gravity
omega=np.sqrt(g*kx)
dkx=np.ones(Nkx,float)*(kx[1]-kx[0]) #wavenumber increment

## Define the group size at focus L_f
L=(width_choice)*(k0)**-1

## Calculate the group spectrum and associated amplitudes
G=L*((2*np.pi)**-0.5)*np.exp(-(L**2)*0.5*(kx-k0)**2)
Af=5
a=Af*G*dkx/np.sum(G*dkx)

## Define the phase
phi0=random_phase*(1-2*np.random.rand(1))*np.pi # random draw between -Pi and Pi if random_phase=1
phi=phi0*np.ones(Nkx) # uniform and constant

## Compute some quantities related to derivatives of omega at k0
cg0=(np.sqrt(g/k0))*(0.5) #group velocity (first derivative of the phase velocity c_0)
par2om=(np.sqrt(g*k0)/(k0**2))*(-0.25) #second derivative of c_0 with respect to the wavenumber k
tau=L**2/par2om #linear group contraction time-scale (negative)

## Define the temporal and spatial axis
dt=0.13*2*np.pi/np.max(omega)
if width_choice==2.36:
    Nt=4*int(np.abs(tau)/dt)
elif width_choice==5:
    Nt=2*int(np.abs(tau)/dt)
elif width_choice==10:
    Nt=1*int(np.abs(tau)/dt)
t=np.linspace(-dt*Nt,0,Nt)

Lx=4*(2*np.pi/k0)
if width_choice==2.36:
    dx=0.000025*2*np.pi/kxmax
elif width_choice==5 or width_choice==10:
    dx=0.000015*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)

## Calculate the elevation (this should take less than 1 minute)
eta1=np.real( 
    np.tensordot( np.tensordot(a*np.exp(1j*phi),np.ones(Nx),axes=0)*
        np.exp( 1j*np.tensordot(kx,x,axes=0) ), 
    np.exp( -1j*np.tensordot(omega-cg0*kx,t,axes=0) ),
            axes=[0,0] )
                    )
