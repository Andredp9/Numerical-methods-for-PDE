"""
@author: Andrea Dal Prete
"""

# Si risolvono le Equazioni di Navier-Stokes in regime non stazionario, per un flusso laminare che investe un profilo
# alare, a velocità costante. Considerando il corpo alare infinitamente lungo si trascura la dimensione Z, e dal momento
# che il flusso è laminare le velocità lungo Y e Z sono considerate nulle. Si prosegue con tutte le ipotesi del caso.
# Considerando che div(V) = 0, si ottiene quindi che la soluzione del problema coincide con la soluzione di un'unica EDP
# tale che Vt - c Vyy = -f. Dove f varia sinusoidalmente, dal momento che il distacco di vortici a valle genera una
# variazione sinusoidale di pressione. Si considera che al momento del distacco del primo vortice il profilo di
# velocità lungo Y segue un andamento lineare (questa coincide con la condizione iniziale u0). 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def der_x(n):
    vn     = np.ones(n)
    vd     = -2*vn
    A      = np.diag(vn[0:-1],1) + np.diag(vn[0:-1],-1) + np.diag(vd)
    
    return A

# BOUNDARY CONDITIONS

def vect_C(n,a,ht,h,uf,t):
    C     = np.zeros(n)
    C[0]  = 0*a*(ht)/(h*h)
    C[-1] = uf*a*(ht)/(h*h)
    return C

# FORCE APPLIED ON THE SYSTEM  

def vect_g(n,t,ht,f,f0):
    #g = ht*fun_ui(t,f,f0)*np.ones(n)            # se il gradiente di pressione risulta costante lungo la direzione Y
    #g = ht*fun_ui(t,f,f0)*np.linspace(0,L,n)    # se il gradiente di pressione varia linearmente lungo Y (nullo a contatto con l'ala, meno realistico)
    g = ht*fun_ui(t,f,f0)*np.linspace(L,0,n)     # se il gradiente di pressione varia linearmente lungo Y (max a contatto con l'ala, più realistico)
    return g    

def fun_ui(t,f,f0): 
    return np.cos(2*np.pi*f*t)*f0

def solve(t0,tf,ht,h,A,u0,n,a,u_inf,f,f0):
    t = np.arange(t0+ht,tf+ht,ht)
    u = np.zeros((len(u0),len(t)))
    u[:,0] = u0
    for i in range(1,len(t)):
        u[:,i] = np.dot(A,u[:,i-1]) + vect_C(n,a,ht,h,u_inf,t[i-1]) + vect_g(n,t[i-1],ht,f,f0)

    return u

# DATA


mu    = 1.8                               # viscosità dinamica del fluido incomprimibile
rho   = 1.5                               # densità del fluido incomprimibile
p     = 2*1e2                             # gradiente di pressione lungo x
n     = 35
L     = 2
h     = L/(n+1)
u_inf = 100
u0    = np.linspace(0,u_inf,n)
c     = mu/rho
f0    = p/rho 
t0    = 0
tf    = 40
ht    = 0.001
f     = 0.5                               # frequenza distacco vortici, n di vortici al secondo

# RESOLUTION

A = (np.diag(np.ones(n)) + (ht*c)*der_x(n)/(h**2))
u = solve(t0,tf,ht,h,A,u0,n,c,u_inf,f,f0)

# PLOT

X = np.linspace(0,L,len(u[:,0]))

plt.figure(figsize=(10,7))
my_plot, =plt.plot([],[])
plt.title("NAVIER-STOKES\n laminar flow on wing profile")
plt.xlim((-u_inf/5,u_inf + n))
plt.ylim((0,L))

t= 0 
tmp = 0
U= np.zeros(len(u[:,0]))
i = 0
while t<tf:
    U = u[:,i]
    t = t + ht
    i = i + 1
    tmp = tmp +1
    
    if tmp==30 :  
        tmp=0     
        my_plot.set_data(U,X)
        plt.pause(0.01)         
        #1e-20





