"""

@ author:   Andrea Dal Prete
@ copyright 

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def der_x(n):
    vn     = np.ones(n)
    vd     = -2*vn
    A      = np.diag(vn[0:-1],1) + np.diag(vn[0:-1],-1) + np.diag(vd)
    return A

# CONDITIONS A LA FRONTIERE

def vect_C(n,a,ht,h,uf,ui,t,coc):
    C     = np.zeros(n)
    C[0]  = ui*a*(ht**2)/(h*h) #fun_ui_V(t,coc)*a*(ht**2)/(h*h) 
    C[-1] = uf*a*(ht**2)/(h*h)
    return C

# EFFORT SUR LE SYSTEME 

def vect_g_D(n,t,coc):                                 # carico distribuito su tutta la fune
    co = fun_ui_V(t,coc)
    return co*1e-5*np.ones(n) #0

def vect_g_C(n,t,coc):                                 # carico distribuito su un certo dominio
    g = np.zeros(n)
    co = fun_ui_V(t,coc)
    for i in range(int(2*n/5),int(3*n/5)+1):
        g[i] = co  
    return g*1e-5 #0

def fun_ui_V(t,coc):                                   # sistema forzato per ogni istante di tempo 
    return np.sin(2*1.41*np.pi*t)*2 
    # first natural frequency 1.41


def fun_ui_S(t,coc):                                   # forza pulsante fino ad un certo istante di tempo in cui smette 
    if t<2.15:
        coc = np.sin(1.41*np.pi*t)*2
        costante = coc
        return coc
    else:
        return coc
    # first natural frequency 1.41

def solve(t0,tf,ht,h,A,u0,u1,n,a,uf,coc):
    t = np.arange(t0+ht,tf+ht,ht)
    u = np.zeros((len(u0),len(t)))
    u[:,0] = u0
    u[:,1] = u1
    for i in range(2,len(t)):
        u[:,i] = np.dot(A,u[:,i-1]) - u[:,i-2] + vect_C(n,a,ht,h,uf,ui,t[i-1],coc) + vect_g_D(n,t[i-1],coc)

    return u

# DATA

n   = 70
L   = 1
h   = L/(n+1)
u0  = np.zeros(n)
u1  = np.zeros(n)
ui  = 0
uf  = 0
a   = 2
t0  = 0
tf  = 20
ht  = 0.001
coc = np.sin(1.41*np.pi*2.15)*2
#ht = h/(np.sqrt(2))

# RESOLUTION

A = a*(np.diag(np.ones(n)) + (ht**2)*der_x(n)/(h**2))
u = solve(t0,tf,ht,h,A,u0,u1,n,a,uf,coc)

# PLOT

X = np.linspace(0,L,len(u[:,0]))

plt.figure()
my_plot, =plt.plot([],[])
plt.title("WAVE")
plt.xlim((0,L))
plt.ylim((-12,12))

t= 0 
tmp = 0
U= np.zeros(len(u[:,0]))
i = 0
while t < tf:
    U = u[:,i]
    t = t + ht
    i = i + 1
    tmp = tmp +1
    
    if tmp == 30:  
        tmp=0     
        my_plot.set_data(X,U)
        plt.pause(0.01)         
        #1e-20

# COMMENTS
# La matrice A est égal à la matrice identitè plus la matrice que approche
# l'operateur derivèe secunde de la fonction sur x, qui est une matrice
# symmetrique avec -2 sur la diagonale, 1 et 1 sur les sub e t sur diagonales.
# Cette derniere matrice est multiplié pour (ht**2)/(h**2), et la matrice finale
# est multiplié pour le coefficent 2

# forme matricelle:
# u([A]





