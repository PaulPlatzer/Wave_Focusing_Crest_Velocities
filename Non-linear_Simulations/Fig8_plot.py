#---------------------------------------------------------------------------------#
# Plotting the results of our method applied to a non-linear Gaussian wave packet #
#---------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------------#
# This code is to be used after having launched "NonLin_Init_Integr.py" and "Fig8_data.py"
# with amp_choice=1.1. These codes produce the data to be plotted here.
# Here we plot the 8th figure of the paper on crest velocities and contraction
# of Gaussian wave packets.
#---------------#

# Go to the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Imports
import numpy as np
import matplotlib.pyplot as plt

## Load the data
d=np.load('Fig8_out.npz')

## Recompute some physical parameters
tp=12 #(s) peak period
grav=9.81 # (m/s^2) gravity acceleration at the earth's surface
w0=2*np.pi/tp # (s^-1) peak pulse
k0=(w0**2)/grav # (m^-1) peak wavenumber
lbda0=2*np.pi/k0 # (m) peak wavelength
c0=w0/k0

## Extract information from the data
tau=d['tau'];L=d['L']
t_axis=d['t_axis'];A_adim=d['A_adim'];L_foc=d['L_foc']
nstart=9
nend=len(d['T_GRAD'][0]); N4=len(d['T_GRAD'])
ntot=0
for l in range(N4):
    ntot+=len(d['T_GRAD'][l])

## Concatenate the data to be plotted
t_grad=np.zeros( int(ntot-N4*(nstart+2)) );tfoc=np.zeros( int(ntot-N4*(nstart+2)) )
A_ratio=np.zeros( int(ntot-N4*(nstart+2)) );L_estim=np.zeros( int(ntot-N4*(nstart+2)) )
tau_abs=np.zeros( int(ntot-N4*(nstart+2)) );c_grad=np.zeros( int(ntot-N4*(nstart+2)) )
i0=0; i1=len(d['T_GRAD'][0])-nstart-2
t_grad[i0:i1]=d['T_GRAD'][0][nstart+2:] ; tfoc[i0:i1]=d['TFOC'][0][nstart+2:] 
A_ratio[i0:i1]=d['A_RATIO'][0][nstart+2:] ; L_estim[i0:i1]=d['L_ESTIM'][0][nstart+2:] 
tau_abs[i0:i1]=d['TAU_ABS'][0][nstart+2:] ; c_grad[i0:i1]=d['C_GRAD'][0][nstart+2:] 
for l in range(1,N4):
    i0+=len(d['T_GRAD'][l-1])-nstart-2 ; i1+=len(d['T_GRAD'][l])-nstart-2
    t_grad[i0:i1]=d['T_GRAD'][l][nstart+2:]; tfoc[i0:i1]=d['TFOC'][l][nstart+2:];
    A_ratio[i0:i1]=d['A_RATIO'][l][nstart+2:]; L_estim[i0:i1]=d['L_ESTIM'][l][nstart+2:];
    tau_abs[i0:i1]=d['TAU_ABS'][l][nstart+2:]; c_grad[i0:i1]=d['C_GRAD'][l][nstart+2:];
ind=np.argsort(t_grad)
t_grad=t_grad[ind]; tfoc=tfoc[ind]; A_ratio=A_ratio[ind]; L_estim=L_estim[ind];
tau_abs=tau_abs[ind]; c_grad=c_grad[ind]; 

## Plot the figure
ns=0
#focusing time
plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(np.array([-60,10]),np.zeros(2),'k',label=r'$t_{f}^{NL}\,/T_p=0$')
#ns=174
ftsz=19
ax.plot(t_grad[ns:]*w0/(2*np.pi),tfoc[ns:]*c0*k0/(2*np.pi),'.k',markersize=.5,label='Estimation')
ax.set_title('Non-linear focusing time estimation',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel('$t_{f}^{NL}\, /T_p$',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
ax.set_ylim([-5,5])
ax.set_xlim([-30,0])
fig.set_size_inches([7,5])
plt.show()

#focusing amplitude ratio
plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(t_axis*w0/(2*np.pi),(A_adim/A_adim[-1])**(-1),'k',
        label='Truth')
#ns=174
ax.plot(t_grad[ns:]*w0/(2*np.pi),A_ratio[ns:],'.k',markersize=.5,label='Estimation')
ax.set_ylim([1,14])
ax.set_xlim([-30,0])
ax.set_title('Non-linear amplitude ratio estimation',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel(r'$A_{f}^{NL}\, /A(t_{end})$',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
fig.set_size_inches([7,5])
plt.show()

#focusing width
plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(np.array([-60,10]),k0*L_foc*np.ones(2),'k',label=r'$k_0L_{f}^{NL}\approx 3.2$')
#ns=174
ax.plot(t_grad[ns:]*w0/(2*np.pi),k0*L_estim[ns:],'.k',markersize=.5,label='Estimation')
ax.set_ylim([3.8,4.1])
ax.set_ylim([0,5])
ax.set_xlim([-30,0])
ax.set_title('Non-linear group size estimation',fontsize=ftsz)
ax.set_xlabel(r'$t_{end}/T_p$',fontsize=ftsz)
ax.set_ylabel(r'$k_0L_{f}^{NL} $',fontsize=ftsz)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
fig.set_size_inches([7,5])
plt.show()