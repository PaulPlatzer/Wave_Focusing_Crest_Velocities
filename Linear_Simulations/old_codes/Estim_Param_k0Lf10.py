#------------------------------------------------------------------------------------#
# Estimate focusing parameters of linear Gaussian wave packet using crest velocities #
#------------------------------------------------------------------------------------#

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

#---------#
# This piece of code is to be used alongside with "Linear_GWP_k0Lf10.py" that produces "eta1",
# the sea-surface elevation of a linear Gaussian wave packet.
# Here we compute the position and velocity of crests and troughs, and we apply our method
# for the estimation of focusing parameters (focusing time, width, amplitude)

## Imports
import numpy as np
from scipy.signal import argrelextrema
from sklearn import linear_model
from scipy.optimize import curve_fit

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
#specular velocity gradient at the center of the packet as a function of time (analytical)
c_grad_expect=ccr_grad_1ord[:-2]/((k0*Spread_x[:-2])**2)+0.5*ccr_grad_2ord[:-2]/((k0*Spread_x[:-2])**4)

## Store the position in space and time of crests and troughs
extrema2=argrelextrema(np.abs(eta1.T), np.greater, axis=1)
extrema=[]
for k in range(len(extrema2[0])):
    extrema.append([extrema2[1][k],extrema2[0][k]])
del extrema2

## Compute the velocity of crests and troughs
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
    
## Define the function used for the fit
def gradient_fit_2( time , t0 , Tau ): # '2' stands for 'second-order' because of the term '-3.25/((...'
    return 2*((time-t0)/(Tau**2+(time-t0)**2))*( 1-3.25/((k0*c0*Tau/4)*(1+((time-t0)/Tau)**2)) )

## Apply the estimation method
STD_FIT=np.std(c_grad_mes-c_grad_expect)# this is used to feed the fit with an initial value of standard deviation
nstart1=0 #this starts the observation/estimation at t=-|\tau|
#'tfoc', 'A_ratio' and 'L' are estimated values based on the best fit
tfoc1=np.zeros(Nt-3); tau_abs1=np.zeros(Nt-3); A_ratio1=np.zeros(Nt-3); L1=np.zeros(Nt-3)
#'cov' are covariances of the estimation, this can be used as a check for the fit's quality
cov1=np.full((Nt-3,2,2),np.inf)

#apply the method starting at t=-|\tau|
nstart=nstart1
for nend in range(nstart+3,Nt-3):
    popt, pcov = curve_fit(gradient_fit2, t[nstart:nend], c_grad_mes[nstart:nend],p0=[0,10*2*np.pi/(k0*c0)]
                       ,sigma=STD_FIT*np.ones(nend-nstart), bounds=([-np.inf,0],np.inf))
    tfoc1[nend]=popt[0]
    tau_abs1[nend]=popt[1]
    L1[nend]=(c0*popt[1]/(4*k0))**0.5
    A_ratio1[nend]=(1+((t[nend]+popt[0])/popt[1])**2)**0.25
    cov1[nend,:,:]=pcov    

## Store the data
np.savez('GWP_k0Lf10_phi%.5f'%phi0,tfoc1=tfoc1,tau_abs1=tau_abs1,A_ratio1=A_ratio1,L1=L1,cov1_00=cov1[:,0,0],cov1_11=cov1[:,1,1],cov1_10=cov1[:,1,0],cov1_01=cov1[:,0,1])