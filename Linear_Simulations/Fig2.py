#-------------------------------------------------------------------------------#
# Evaluating & plotting an approximation for a wave packet of Gaussian spectrum #
#-------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------------#
# We set k_0L_f=5 (see related paper)
# The phase is uniform, constant, equal to zero.
# We assume a linear evolution of the ocean surface wave packet.
# The packet is plotted ~20 periods before focusing.
#---------------#


## Imports
import numpy as np
import matplotlib.pyplot as plt


## Defining the wavenumber axis
kxmin=0.001; kxmax=0.05; Nkx=150; k0=0.02
kx=np.linspace(kxmin,kxmax,Nkx)
dkx=np.ones(Nkx,float)*(kx[1]-kx[0]) #wavenumber increment

## Width and spectumr of the wave-packet
L=(5)*(k0)**-1
G=L*((2*np.pi)**-0.5)*np.exp(-(L**2)*0.5*(kx-k0)**2)

## Gravity and pulse
g=9.81
omega=np.sqrt(g*kx)

## Phase
phi=np.zeros(Nkx)

## AMplitude
A0=5 # "Amplitude at focus"
a=A0*G*dkx/np.sum(G*dkx)

## Time axis
dt=2*np.pi/(3*np.max(omega)); Nt=2
t=np.linspace(-dt*(100),0,Nt)
Tp=2*np.pi/np.sqrt(k0*g) #peak period

## Space axis (unidimensional)
Lx=7*(2*np.pi/k0)
dx=0.02*2*np.pi/kxmax
Nx=int(Lx/dx+2)
x=np.linspace(-Lx/2,Lx/2,Nx)

## Some parameters related to the phase speed
c0=np.sqrt(9.81*k0)/k0
cg0=(np.sqrt(9.81*k0)/k0)*(0.5)
cg0sk0=(np.sqrt(9.81*k0)/(k0**2))*(0.5) #cg(k0)/k0
par2om=(np.sqrt(9.81*k0)/(k0**2))*(-0.25)
Spread_x=np.sqrt(L**2+(L**(-2))*(par2om*t)**2) # theoretical spatial width ("spread") of the wave packet

## Analyticaly derived phase calculation
phas_an=np.zeros((Nx,Nt),float)                
for i in range(Nx):
    for l in range(Nt):
        phas_an[i,l]= (-np.arctan((par2om*t[l])/(2*L**2))+(
                        ((x[i]+cg0*t[l]-cg0*t[l])/(np.sqrt(2)*L*Spread_x[l]))**2)*(par2om*t[l])+
                        +k0*(x[i]+cg0*t[l])-(c0*k0*t[l]))%(2*np.pi)

## Analyticaly derived envelope calculation
enve_an=np.zeros((Nx,Nt),float)                
for i in range(Nx):
    for l in range(Nt):
        enve_an[i,l]= A0*(  ((1+((par2om*t[l])**2)/(L**4))**(-0.25)) 
                                * np.exp(-((x[i]+cg0*t[l]-cg0*t[l])/(np.sqrt(2)*Spread_x[l]))**2 ) )   


## Analyticaly derived surface elevation
eta1_an=enve_an*np.cos(phas_an)
            
## Real surface elevation (approximated numerically)
eta1=np.real( 
    np.tensordot( np.tensordot(a*np.exp(1j*phi),np.ones(Nx),axes=0)*
        np.exp( 1j*np.tensordot(kx,x,axes=0) ), 
    np.exp( -1j*np.tensordot(omega-cg0*kx,t,axes=0) ),
            axes=[0,0] )
                    )

## Plot
plt.clf()
n=0
fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(x*k0/(2*np.pi),eta1[:,n],'r',label=r'Re($\Psi_G$)')
ax.plot(x*k0/(2*np.pi),eta1_an[:,n],'k',label=r'Re($\Psi_G^a$)',linewidth=1)
ax.plot(x*k0/(2*np.pi),enve_an[:,n],'--k',label=r'$\pm |\Psi_G^a|$')
ax.plot(x*k0/(2*np.pi),-enve_an[:,n],'--k')
plt.title('t='+str(t[n]/Tp)[0:6]+r'$T_p$ before focusing',fontsize=16)
ax.set_xlabel(r'$(x-x_c)/\lambda_0$ (m)',fontsize=16);
ax.set_ylabel(r'$\eta_G$',fontsize=16);
fig.set_size_inches(5,4)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=16)
ax.tick_params(length=7,labelsize=16,direction='out',pad=10)
plt.show()

