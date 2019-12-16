## 4th-order explicit Runge-Kutta solver for time-independant ODE ##

# Written in 2019 by PhD student Paul Platzer
# Contact: paul.platzer@imt-atlantique.fr

# Using standard notations as in https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method

def RK4(yt,h,f):
    k1=h*f(yt)
    k2=h*f(yt+0.5*k1)
    k3=h*f(yt+0.5*k2)
    k4=h*f(yt+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return yt+dy # such that RK4(y(t),dt,f)=y(t+dt)