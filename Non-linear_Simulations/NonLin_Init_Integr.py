#--------------------------------------------------------------------------------------#
# Numerical simulation of sea-surface elevation of Gaussian wave packet using the NLSE #
#--------------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr
# The indicated approximate computation times were evaluated on a laptop

#---------------#
# The NLSE is the Non-Linear Schrödinger Equation
# The Gaussian wave packet has a spectral width such that k_0L_f^L=5
# The wave packet's amplitude can have 6 different values:
#   - nearly linear with k_0A_f^L=0.0238 and k_0A_f^{NL}=0.0240 (amp_choice=0.2)
#   - weakly non-linear with k_0A_f^L=0.095 and k_0A_f^{NL}=0.11 (amp_choice=0.8)
#   - weakly non-linear with k_0A_f^L=0.13 and k_0A_f^{NL}=0.16 (amp_choice=1.1) -> used for figure 8
#   - weakly non-linear with k_0A_f^L=0.17 and k_0A_f^{NL}=0.23 (amp_choice=sqrt(2))
#   - weakly non-linear with k_0A_f^L=0.19 and k_0A_f^{NL}=0.28 (amp_choice=1.6)
#   - weakly non-linear with k_0A_f^L=0.21 and k_0A_f^{NL}=0.33 (amp_choice=1.8)
#---------------#

## Put yourself in the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Imports
import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
from RK4 import RK4 # Runge-Kutta solver
from NLSE import NLS_spec

## Choose the amplitude of the wave-packet (see introduction of this code)
amp_choice=1.1

## Physical parameters
tp=12 #(s) peak period
grav=9.81 # (m/s^2) gravity acceleration at the earth's surface
w0=2*np.pi/tp # (s^-1) peak pulse
k0=(w0**2)/grav # (m^-1) peak wavenumber
lbda0=2*np.pi/k0 # (m) peak wavelength

## Domain
# Spatial
dx=10; dX=2*np.sqrt(2)*k0*dx # Spatial discretization
Nx=2**12 # Number of points in space (=number of Fourier modes)
def NLS(c_t):
    return NLS_spec(Nx,dX,c_t) # the iteration of the spectral NLS depends on (Nx,dX)
Nlbda=Nx*dx/lbda0 # Number of peak wavelength in the spatial domain
X_axis=np.arange(-dX*Nx/2,dX*Nx/2,dX)
# Temporal
#Nper=2*1500 # Number of peak period in the temporal domain
Nper=65 # Number of peak period in the temporal domain
dt=0.14; dT=w0*dt # Temporal discretization
Nt=round((Nper+1)*tp/dt)
T_axis=np.arange(0,Nt*dT,dT)

## Initialization
C=np.zeros((Nt,Nx),np.complex64) # Fourier modes for U
# Back-propagated Gaussian packet
sig=2*np.sqrt(2)*5  # sig = 2sqrt(2)*LfkO where Lf is the spatial width at linear focus
A=amp_choice*(2**0.25)/sig  # adimensional amplitude
phi0=0 # uniform phase
U00=A*np.exp(-X_axis**2/(2*sig**2))*np.exp(1j*phi0)
C00=np.fft.fft(U00)
Delta_T=4*(-0.5*sig**2) # (-0.5*sig**2) corresponds to om0*tau for the article on GWP
                        # and crest velocities
OmNxdX=-(2*np.pi*np.fft.fftfreq(Nx,dX))**2    
                                    # -> This formula for the dispertion relation
                                    # for the envelope is true in the narrow-band limit
                                    # that is used for the NLSE
                                    # and in the reference frame
                                    # moving with the group-velocity
C0=C00*np.exp(-1j*OmNxdX*Delta_T)
U0=np.fft.ifft(C0)
C[0,:]=C0

## Integration of the NLSE ## (should take less than one minute)
for i in range(Nt-1):
    C[i+1,:]=RK4(C[i,:],dT,NLS)
