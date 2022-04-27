import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def der_x(n):
    vn     = np.ones(n)
    vn[-1] = 0
    vn     = np.tile(vn,n)
    vn     = vn[:-1]
    X = np.diag(vn,1) + np.diag(-vn,-1)
    return X

def der_y(n):
    vn = np.ones(n*n - n)
    return np.diag(vn,n) + np.diag(-vn,-n)

def laplacian(n):
    return np.diag(-4*np.ones(n*n)) + np.abs(der_x(n)) + np.abs(der_y(n))

def u_ex(n,t0,tf,ht):
    T = np.arange(t0,tf,ht)
    X = np.linspace(0,1,n+2)[1:-1]
    Y = np.linspace(0,1,n+2)[1:-1]
    u = np.zeros((n*n,len(T)))
    for k in range(0,len(T)):
        t = T[k]
        cont = 0
        for j in range(0,len(Y)):
            y = Y[j]
            for i in range(0,len(X)):
                x = X[i]
                u[cont,k] = np.exp(np.pi*t)*np.sin(np.pi*x)*np.sin(np.pi*y)
                cont = cont + 1
    return u

def vect_g(n,a,b,c,la,t):
    X = np.linspace(0,1,n+2)[1:-1]
    Y = np.linspace(0,1,n+2)[1:-1]
    g = np.zeros(n*n)
    k = 0
    for y in Y:
        for x in X:
            g[k] = np.pi * np.exp(np.pi * t) * (
            (1 + 2 * np.pi * la) * np.sin(np.pi * x) * np.sin(np.pi * y) +
            a * np.cos(np.pi * x) * np.sin(np.pi * y) +
            b * np.sin(np.pi * x) * np.cos(np.pi * y))
            k = k + 1
            
    return -g 

def solve_a(n,a,b,la,t0,tf,ht,A,B,u0):
    t = np.arange(t0+ht,tf+ht,ht)
    u = np.zeros((len(A[:,1]),len(t)))
    u[:,0] = u0
    for i in range(1,len(t)):
        F = vect_g(n,a,b,c,la,t[i-1])
        u[:,i] = np.dot(np.linalg.inv(B),(np.dot(A,u[:,i-1]) + ht*F))

    return u

def cond_in(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)

# DATI

n = 60
L = 1
h = L/(n+1)
a = 0
b = 0
c = 0
la = 1
t0 = 0
tf = 0.5
#ht = 0.1*(h**2)/la
ht = 0.1
theta = 1/2
X = np.linspace(0,1,n)
Y = np.linspace(0,1,n)
u0 = np.zeros(n*n)
k = 0
for j in range(0,len(Y)):
    y = Y[j]
    for i in range(0,len(X)):
        x = X[i]
        u0[k] = cond_in(x,y)
        k = k + 1

# RESOLUTION
start = time.time()
B = (np.diag(np.ones(n*n)) - theta*ht*(la*laplacian(n)/(h**2) - a*der_x(n)/(2*h) - b*der_y(n)/(2*h)))
A = (np.diag(np.ones(n*n)) + (1 - theta)*ht*(la*laplacian(n)/(h**2) - a*der_x(n)/(2*h) - b*der_y(n)/(2*h)))
u = solve_a(n,a,b,la,t0,tf,ht,A,B,u0)
u_e = u_ex(n,t0,tf,ht)
print(np.linalg.norm((u-u_e)/u_e))

end = time.time()
print(end-start)


# PLOT
for j in range(0,len(u[0,:])):    
    fig = plt.figure(j)
    ax = fig.add_subplot(111, projection="3d")
    u1 = u[:,j]

    X = np.linspace(0,L,n)
    y = np.linspace(0,L,n)

    for i in range(0,n):
        Y = y[i]
        Z = u1[i*n:(i+1)*n]
        ax.scatter(X,Z,Y, s = 0.1, color='green')
        ax.set_xlim3d(0, 1)     
        ax.set_ylim3d(-10, 10)                
        ax.set_zlim3d(0, 1)
    plt.show(block=False)
    plt.pause(1)
    plt.close()



