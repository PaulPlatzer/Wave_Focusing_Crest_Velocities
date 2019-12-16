## Function for integration of the NLSE in Fourier space ##

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

# This function solves the Non-Linear Schrödinger Equation (NLSE) in Fourier space
# Where the NLSE for the field u is: i*d(u)/dt = d^2(u)/dx^2 + u*|u|^2
# It takes the Fourier coefficients c_t of the filed u_t at time t
# and returns the time derivative of c_t.
# Nx: number of points in space (/!\ must be specified before using the function)
# dX: spatial discretization (/!\ must be specified before using the function)
# /!\ numpy must have been imported as "np"

def NLS_spec(Nx,dX,c_t): # "spec" stands for "spectral"
    import numpy as np
    u_t=np.fft.ifft(c_t)
    c_prime_t=np.fft.fft((np.abs(u_t)**2)*u_t)
    return 1j*( ((2*np.pi*np.fft.fftfreq(Nx,dX))**2)*c_t - c_prime_t )