#---------------------------------------------------------------------------------#
# Plotting the the gradient of non-linear specular velocity as a function of time #
#---------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------------#
# This code is to be used after having launched "NonLin_Init_Integr.py" and "Fig7_data.py"
# for all the proposed values of "amp_choice". These codes produce the data to be plotted here.
# Here we plot the 7th figure of the paper on crest velocities and contraction
# of Gaussian wave packets.
#---------------#

# Go to the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Non-linear_Simulations/

## Imports
import numpy as np
import matplotlib.pyplot as plt

## Load the data
d1=np.load('Fig7_out_linear.npz')
d2=np.load('Fig7_out_0p11.npz')
d3=np.load('Fig7_out_0p16.npz')
d4=np.load('Fig7_out_0p23.npz')
d5=np.load('Fig7_out_0p28.npz')
d6=np.load('Fig7_out_0p33.npz')

## Recompute some physical parameters
tp=12 #(s) peak period
grav=9.81 # (m/s^2) gravity acceleration at the earth's surface
w0=2*np.pi/tp # (s^-1) peak pulse
k0=(w0**2)/grav # (m^-1) peak wavenumber
lbda0=2*np.pi/k0 # (m) peak wavelength
c0=w0/k0

## Plot the figure
plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
ax.plot(d1['x_ccr'],d1['ccr_adim'],color='0.8',label=r'Linear')
ax.plot(d2['x_ccr'],d2['ccr_adim'],color='0.65',label=r'$k_0A_f^{NL}=0.11$')
ax.plot(d3['x_ccr'],d3['ccr_adim'],color='0.5',label=r'$k_0A_f^{NL}=0.16$')
ax.plot(d4['x_ccr'],d4['ccr_adim'],color='0.3',label=r'$k_0A_f^{NL}=0.23$')
ax.plot(d5['x_ccr'],d5['ccr_adim'],color='0.1',label=r'$k_0A_f^{NL}=0.28$')
ax.plot(d6['x_ccr'],d6['ccr_adim'],color='0',label=r'$k_0A_f^{NL}=0.33$')
ax.set_ylim(-0.2,1)
ax.set_xlim(-4,4)
plt.grid(alpha=.5)
ftsz=19
plt.legend(bbox_to_anchor=(.97, .97), loc=1, borderaxespad=0.,fontsize=ftsz-1,framealpha=1)
ax.set_xlabel('$(x-x_c)/\lambda_0$',fontsize=ftsz);
ax.set_ylabel("$(c_{sp}-c_0)/c_0$",fontsize=ftsz);
plt.title(r'Non-linear specular velocity profile, $k_0L_f^L=5$',fontsize=ftsz)
ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
fig.set_size_inches(8,6)
plt.show()