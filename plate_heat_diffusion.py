import numpy as np
import matplotlib.pyplot as plt
import os
import time

def grad(n):
    vn     = np.ones(n)
    vn[-1] = 0
    vn     = np.tile(vn,n)
    vn = vn[:-1]
    return (-4*np.diag(np.ones(n*n)) + np.diag(vn,1) + np.diag(vn,-1) +
            np.diag(np.ones(n*n-(n)), n)+ np.diag(np.ones(n*n-(n)), -n))

def solve(t_0,t_f,ht,n,T_0,a):
    start_time = time.time()
    t      = np.arange(t_0+ht,t_f,ht)
    T      = np.zeros((len(t), n*n))
    T[0,:] = T_0
    for i in range(1,len(t)):
        q = np.zeros(n*n)
        if t[i] < 0.67:   # best 1.2
            q[(i-1)+int(n*n/2)] = 10**6 # best 10^7
        T[i,:] = np.matmul((np.eye(n*n) + (ht/a)*grad(n)/(h**2)),T[i-1,:]) + ht*q
    tim = time.time() - start_time
    print("Needed time for the calculation [s]:" + str(tim))
    return T,t,tim


# DATA

t_0 = 0
t_f = 1.1                     # time [s] (1.4 best) 
ht  = 0.015                   # delta t  (0.010 best)
L   = 10                      # plate length [mm] 
n   = 42                      # number of elements (70 best)
h   = L/n
a   = 18.8                    # thermal diffusivity [mm^2/s]
Ti  = 20 + 273.15             # initial temperature
T_0 = Ti*np.ones(n*n)
T_m = 1400 + 273.15           # melting temperature
T_e = 2700 + 273.15           # evaporation temperature
# SOURCE


# SYSTEM SOLUTION

st_time = time.time()

u,t,tempo = solve(t_0,t_f,ht,n,T_0,a)

# NOTICE

# PLOT

import easygui
easygui.msgbox("Nella seguente simulazione è rappresentata una piastra di acciaio: una fonte di calore (laser) rilascia su di essa energia sottoforma di potenza termica"
               " incrementandone la temperatura. Nei punti in cui la piastra è grigia significa che la temperatura non cambia considervolmente, dove è verde e poi arancione"
               " c'è un lieve incremento di temperatura, dove è rossa il materiale ha raggiunto la temperatura di fusione e dove è bianca quella di evaporazione, e qui ha"
               " luogo la conseguente asportazione di materiale." "\n\n\n\n" "Premi 'OK' per visualizzare la simulazione.", title="Instructions to understand the simulation")

for j in range(0,len(u[:,0])):    
    plt.figure(j, figsize=(9,8))
    plt.title('Laser Beam on a Steel plate, ' + 'time = ' + str(round(t[j],2)) + " s")
    u1 = u[j,:]
    x = np.linspace(0,L,n)
    y = np.linspace(0,L,n)

    for k in range(0,n):
        u_s = u1[k*n:(k+1)*n]
        for i in range(0,n):
            if u_s[i]<Ti+0.8:
                col = 'gray'
            elif u_s[i]>Ti+0.8 and u_s[i]<Ti+1.5:
                col = 'limegreen'
            elif u_s[i]>Ti+1.5 and u_s[i]<Ti+50:
                col = 'coral'
            elif u_s[i]>Ti+50 and u_s[i]<T_m:
                col = 'orangered'
            elif u_s[i]>T_m and u_s[i]<T_e:
                col = 'red'
            else:
                col = 'white'
            plt.plot(x[i],y[k], marker='s', markersize=9.8, color=col) # best size 6 
    plt.show(block=False)
    plt.pause(ht/(10**20))
    if j<len(u[:,0])-1:
        plt.close()
    else:
        os.system("say 'Andrea, ho completato la simulazione, la visualizzazione è pronta'")
        tt1 = "Tempo di calcolo [s]: " + str(round(tempo,3))
        tt2 = "\n\nTempo totale simulazione [s]: " + str(round(time.time() - st_time,3))
        easygui.msgbox(tt1 + tt2, title="Computation performances")
        
        
