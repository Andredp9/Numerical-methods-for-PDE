import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def der_sec_x(n,ht,a,h):
    vn = np.ones(n)
    return a*ht*(np.diag(-2*vn) + np.diag(vn[:-1],-1) + np.diag(vn[:-1],1))/(h*h)

def vect_c(n,uf,ui,h,ht,a):
    c = np.zeros(n)
    c[0] = a*ht*ui/(h*h)
    c[-1] = a*ht*uf/(h*h)
    return c

def solve_exp(a,t0,tf,ht,h,u0,uf,ui,A):
    t = np.arange(t0,tf+ht,ht)
    u = np.zeros((len(u0),len(t)))
    C = vect_c(n,uf,ui,h,ht,a)
    u[:,0] = u0
    for i in range(1,len(t)):
        u[:,i] = np.dot(A,u[:,i-1]) + C
    return u

def solve_imp(a,t0,tf,ht,h,u0,uf,ui,A):
    t = np.arange(t0,tf+ht,ht)
    u = np.zeros((len(u0),len(t)))
    C = vect_c(n,uf,ui,h,ht,a)
    u[:,0] = u0
    A = np.linalg.inv(A)
    for i in range(1,len(t)):
        u[:,i] = np.dot(A,(u[:,i-1] + C))
    return u

# DATI

t0 = 0
tf = 200
n  = 40
L  = 0.008 
h  = L/(n+1)
ht = 0.01
a  = 7.05*1e-8
u0 = np.ones(n)*310
uf = 355
ui = 310

# RISOLUZIONE SCHEMA ESPLICITO

A = np.diag(np.ones(n)) + der_sec_x(n,ht,a,h) 
u_exp = solve_exp(a,t0,tf,ht,h,u0,uf,ui,A)
sp = np.arange(0+h,L,h)

# RISOLUZIONE SCHEMA IMPLICITO

A = np.diag(np.ones(n)) - der_sec_x(n,ht,a,h)
u_imp = solve_imp(a,t0,tf,ht,h,u0,uf,ui,A)
sp = np.arange(0+h,L,h)


# PLOT

# IMPLICITO

X = np.linspace(0,L,len(u_imp[:,0]))

plt.figure()
my_plot, =plt.plot([],[])
plt.title("TEMPERATURA SU PARETE\n (condizioni di dirichlet alla frontiera)")
plt.xlim((0,L))
plt.ylim((ui,uf))

t   = 0 
tmp = 0
U   = np.zeros(len(u_imp[:,0]))
i   = 0
while t<tf:
    U = u_imp[:,i]
    t = t + ht
    i = i + 1
    tmp = tmp +1
    
    if tmp==30 :  
        tmp=0     
        my_plot.set_data(X,U)
        plt.pause(0.001)         
        #1e-20
