#------------------------------------------------------------------------------#
# Numerical simulation of sea-surface elevation of linear Gaussian wave packet #
#------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------------#
# Important parameters:
# k_0L_f=2.36
# simulation from t=-4|\tau| to t=0 where 0 is the linear focusing time
# phase=uniform, constant, can be random or 0
#---------------#

random_phase=1 # set "1" for random phase, or "0" for phi=0

# Imports
import numpy as np

# Define the wavenumber and angular frequency axis 
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
Nt=4*int(np.abs(tau)/dt)
t=np.linspace(-dt*Nt,0,Nt)

Lx=4*(2*np.pi/k0)
dx=0.000025*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)

## Calculate the elevation
eta1=np.real( 
    np.tensordot( np.tensordot(a*np.exp(1j*phi),np.ones(Nx),axes=0)*
        np.exp( 1j*np.tensordot(kx,x,axes=0) ), 
    np.exp( -1j*np.tensordot(omega-cg0*kx,t,axes=0) ),
            axes=[0,0] )
                    )   