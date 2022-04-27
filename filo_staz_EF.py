# Metodo agli elementi finiti, funzioni di forma lineari
# Soluzione del problema del filo statico caricato da una forzante
#distribuita costante

import numpy as np
import matplotlib.pyplot as plt

# DATI

d   = 0.03
Su  = (np.pi/4)*d**2
rho = 7860
L   = 30
mu  = 70000
N   = 100
f   = -rho*Su
bet = 0
sig = 0
u_i = 0
u_f = -2

# COSTRUZIONE DEL SISTEMA

h = (L-0)/(N+1)

# matrice di rigidezza

K = (mu/h)*(np.diag(np.ones(N-1)*2) + np.diag(-1*np.ones(N-2),1) + np.diag(-1*np.ones(N-2),-1))

# matrice di trasporto

B = (np.diag(np.zeros(N-1)) + np.diag(-1*np.ones(N-2),-1) + np.diag(np.ones(N-2),1))
B[0,0]   = -1
B[-1,-1] = 1
B = B*(bet/2)

# matrice di massa

M = (np.diag((2/3)*np.ones(N-1)) + np.diag((1/6)*np.ones(N-2),-1) + np.diag((1/6)*np.ones(N-2),1))
M[0,0]   = 1/3
M[-1,-1] = 1/3
M = M*(sig*h)

# condizioni al contorno

C      = np.zeros(N-1)
C[0]   = u_i
C[N-2] = u_f
C      = (mu/h)*C 

# matrice globale

A = K + B + M

# vettore forzante

f1                 = np.zeros(N-1)
f1[10] = 2*600/(L/N)
f1[30] = 2*600/(L/N)
f1[50] = 2*600/(L/N)
f1[70] = 2*600/(L/N)
f1[90] = 2*600/(L/N)

f1[int(N/2)] = -50000 

f = h*f*np.ones(N-1) - h*f1

# SOLUZIONE DEL SISTEMA E CREAZIONE DEI VETTORI

uh = np.dot(np.linalg.inv(A),f + C) 
u  = np.zeros(N+1)

u[0] = u_i
u[N] = u_f

for i in range(1,N):
    u[i] = uh[i-1]
    
    

x  = np.linspace(0,L,N+1)
print("Punto d'imbarco massimo:" + str(round(min(u),3)))
print("Forza necessaria:" + str(f1[int(N/2)]*(L/N)))

# PLOT

plt.figure(1)
plt.plot(x, u, color = 'green')
plt.ylim((-5,5))
plt.title("FILO SOLLECITATO DA UNA FORZANTE DISTRIBUITA COSTANTE")
plt.show()
