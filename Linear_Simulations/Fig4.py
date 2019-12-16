#----------------------------------------------------------------------#
#
# Plotting specular velocity gradients of linear Gaussain wave packets #
#
#----------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------#
# This piece of code launches "Linear_Init_Integr.py"
# (choosing width_choice=... and setting random_phase=0) that produces "eta1",
# the sea-surface elevation of a linear Gaussian wave packet.
#---------#

## Put yourself in the directory where the python files are located
#cd /home/administrateur/Documents/Thèse/Gaussian_Wave_Packet/ARTICLE1/CODES_Finaux/Linear_Simulations/

## Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn import linear_model

from Linear_Init_Integr import *

## Compute some quantities useful for the plot
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
#specular velocity gradient at the center of the packet as a function of time (analytical)
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)

## Store the position in space and time of crests and troughs (should take less than 1 minute)
extrema2=argrelextrema(np.abs(eta1.T), np.greater, axis=1)
extrema=[]
for k in range(len(extrema2[0])):
    extrema.append([extrema2[1][k],extrema2[0][k]])
del extrema2

## Compute the velocity of crests and troughs (should take less than 10 seconds)
extr_speed1=np.zeros(len(extrema))
for m in range(len(extrema)):
    i=extrema[m][0]; n=extrema[m][1]
    if n<len(t)-1:
        dist=len(x)
        for m2 in range(len(extrema)):
            if extrema[m2][1]==n+1 :
                if np.abs(extrema[m2][0]-i)<dist:
                    i2=extrema[m2][0]
                    dist=np.abs(extrema[m2][0]-i)
        extr_speed1[m]=(x[i2]-x[i])/dt
        
## Estimate the central gradient of specular velocity
c_grad_mes=np.zeros(Nt-2)
dist0=np.max(x)*10**2
LM=linear_model.LinearRegression()
for n in range(Nt-2):
    m0=0
    dist=dist0
    for m in range(len(extrema)):
        if (extrema[m][1]==n
            and np.abs(x[extrema[m][0]])<dist):
            dist=np.abs(x[extrema[m][0]])
            m0=m
            
    y=np.array([  
                extr_speed1[m0-1] , 
                extr_speed1[m0] ,
                extr_speed1[m0+1]  
               ])
    
    X=np.array([ 
                [1,x[extrema[m0-1][0]],x[extrema[m0-1][0]]**2] , 
                [1,x[extrema[m0][0]],x[extrema[m0][0]]**2] ,
                [1,x[extrema[m0+1][0]],x[extrema[m0+1][0]]**2] 
               ])
    
    c_grad_mes[n]=LM.fit(X,y).coef_[1]
    del X, y
    
## Plot
if width_choice==2.36:
    plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
    ax.fill_between([-1,0], [-0.3,-0.3],[0.05,0.05],color='g', alpha=.1)
    ax.fill_between([-2,-1], [-0.3,-0.3],[0.05,0.05],color='b', alpha=.1)
    ax.fill_between([-4,-2], [-0.3,-0.3],[0.05,0.05],color='r', alpha=.1)
    ax.plot(t[:-2]/np.abs(tau),c_grad_expect*L*L*k0*k0/(c0*k0),'-k')
    ax.plot(t[:-2]/np.abs(tau),c_grad_mes*L*L*k0*k0/(c0*k0),'.k',markersize=1.5)
    ax.set_xlim([-4,0])
    ax.set_ylim([-0.26,0.01])
    ftsz=22
    ax.set_xlabel(r'$t/ |\tau|$',fontsize=ftsz);
    ax.set_ylabel(r"$\dfrac{(L_f\, k_0)^2}{ \omega_0} \; \partial_x c_{sp}|_{x=x_c}$",fontsize=ftsz)
    plt.title(r'Gradient estimation, $L_f\, k_0=2.36$',fontsize=ftsz)
    ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
    fig.set_size_inches([6,6])
    plt.show()
elif width_choice==5:
    plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
    ax.fill_between([-1,0], [-0.3,-0.3],[0.05,0.05],color='g', alpha=.1)
    ax.fill_between([-2,-1], [-0.3,-0.3],[0.05,0.05],color='b', alpha=.1)
    ax.plot(t[:-2]/np.abs(tau),c_grad_expect*L*L*k0*k0/(c0*k0),'-k')
    ax.plot(t[:-2]/np.abs(tau),c_grad_mes*L*L*k0*k0/(c0*k0),'.k',markersize=1)
    ax.set_xlim([-2,0])
    ax.set_ylim([-0.26,0.01])
    ftsz=22
    ax.set_xlabel(r'$t/ |\tau|$',fontsize=ftsz);
    ax.set_ylabel(r"$\dfrac{(L_f\, k_0)^2}{ \omega_0} \; \partial_x c_{sp}|_{x=x_c}$",fontsize=ftsz)
    plt.title(r'Gradient estimation, $L_f\, k_0=5$',fontsize=ftsz)
    ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
    fig.set_size_inches([6,6])
    plt.show()
elif width_choice==10:
    plt.clf(); fig=plt.figure(); fig.add_axes(); ax=fig.add_subplot(111)
    ax.fill_between([-1,0], [-0.3,-0.3],[0.05,0.05],color='g', alpha=.1)
    ax.plot(t[:-2]/np.abs(tau),c_grad_expect*L*L*k0*k0/(c0*k0),'-k')
    ax.plot(t[:-2]/np.abs(tau),c_grad_mes*L*L*k0*k0/(c0*k0),'.k',markersize=0.7)
    ax.set_xlim([-1,0])
    ax.set_ylim([-0.26,0.01])
    ftsz=22
    ax.set_xlabel(r'$t/ |\tau|$',fontsize=ftsz);
    ax.set_ylabel(r"$\dfrac{(L_f\, k_0)^2}{ \omega_0} \; \partial_x c_{sp}|_{x=x_c}$",fontsize=ftsz)
    plt.title(r'Gradient estimation, $L_f\, k_0=10$',fontsize=ftsz)
    ax.tick_params(length=7,labelsize=ftsz,direction='out',pad=10)
    fig.set_size_inches([6,6])
    plt.show()