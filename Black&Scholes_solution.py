# -----------------------------------------------------------------------------
# Copyright (c) 2025 Andrea Dal Prete - Politecnico di Milano
# All rights reserved.
#
# This script is part of the research published in:
# [Your Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Andrea Dal Prete (andrea.dalprete@polimi.it)
# -----------------------------------------------------------------------------

# This script proposes a numerical solution of the Black and Scholes Partial Differential Equation (PDE)
# governing the price of an option given the stochastic distribution of a general stock price.

#%% Initialization
print('\n\n---------------- INITIALIZATION ----------------')
print('Black and Scholes PDE solver. Engine is ready to start.\n\n')
print('-------------------- SOLVER --------------------\n')

# Import useful libraries
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch as t
device = t.device('mps') if t.backends.mps.is_available() else 'cpu'  # Choose the GPU (MPS) to perform matrix multiplications and accelerate computations
print('MACOX GPU accelerator ("mps") available.') if t.backends.mps.is_available() else print('MACOX GPU accelerator ("mps") is not available. Fallback to CPU.')
print('The PDE solver will use the following device: ', device)

# Define useful functions 
def D_s(n):  # Function defines the finite-differences approximation for the first partial derivative of a function 
    v = t.zeros(n, device=device)  # Define the diagonal of the matrix  
    v_1 = t.ones(n-1, device=device) 
    v_0 = -t.ones(n-1, device=device) 
    return t.diag(v) + t.diag(v_1, 1) + t.diag(v_0, -1)

def D_ss(n):  # Function defines the finite-differences approximation for the second partial derivative of a function 
    v = -2 * t.ones(n, device=device)  # Define the diagonal of the matrix  
    v_1 = t.ones(n-1, device=device)  
    return t.diag(v) + t.diag(v_1, 1) + t.diag(v_1, -1)

def euler_forward_solver(Vt0, r, sig, S, Ds, Dss, t0, tf, ht, h): # euler forward, differential equations solver
    # Note that S and Vt[:,0] must be vertical vectors
    t_values = t.arange(t0, tf + ht, ht, device=device)
    Vt = t.zeros((len(S), len(t_values)), device=device)
    # define boundaty conditions effects if needed (won't use for now)
    C = t.zeros(len(S), device=device)
    C[0]  =  ht * (r * S[0] * Vt0[0] / (2 * h) - S[0]**2 * sig**2 * Vt0[0] / (h ** 2)) 
    C[-1] = -ht * (r * S[-1] * Vt0[-1] / (2 * h) + S[-1]**2 * sig**2 * Vt0[-1] / (h ** 2))
    # Initialisation
    Vt[:, 0] = Vt0
    I = t.eye(len(S), device=device)
    # iterative solver 
    print('\n\nSolver is solving the PDE...')
    for i in tqdm(range(len(t_values) - 1)):
        Vt[:, i + 1] = t.matmul((1 + ht * r) * I - ht * (r * t.matmul(t.diag(S),Ds) - 0.5 * t.matmul(t.diag(S ** 2),Dss) * sig ** 2), Vt[:, i])
        # force boundary conditions for the european call 
        Vt[0, i+1] = 0
        Vt[-1, i+1] = S[-1] - K * t.exp(-r * (tf - t_values[i]))
    print('PDE succesfully solved. The solution is now available.\n\n')
    return Vt, t_values

#%% Main script 

# Numerical solution parameters 
h = 0.05
S_max = 3  # Choose a reasonable max stock price
S = t.arange(h, S_max, h, device=device)  # Discretization of the stochastic stock price
K = 1.5  # Strike price
Vt0 = t.maximum(S - K, t.tensor(0.0, device=device))  # Payoff at expiry
t0 = 0
tf = 1
ht = 0.0005 # euler forward method condition for numerical stability in this case is: ht < 0.5 * h**2 / (sigma**2 * S_max**2)
n = len(S)

# Physical/statistical parameters
sigma = 0.2  # Constant expressing the volatility of the underlying asset 
r = 0.02  # Constant expressing the risk-free interest rate 

#%% Solutions

# Equation solution
Ds = D_s(n) / (2 * h)  # Matrix approximating the first partial derivative 
Dss = D_ss(n) / (h ** 2)  # Matrix approximating the second partial derivative 

Vt, t_values = euler_forward_solver(Vt0, r, sigma, S, Ds, Dss, t0, tf, ht, h)

#%% Plot section

# Import packages for 3D plot  
from mpl_toolkits.mplot3d import Axes3D

# Create a meshgrid for plotting
S_grid, T_grid = t.meshgrid(S, t_values, indexing="ij")

# Convert tensors back to numpy for plotting
S_grid = S_grid.cpu().numpy()
T_grid = T_grid.cpu().numpy()
Vt = Vt.cpu().numpy()

# Plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot with color mapping
ax.plot_surface(S_grid, T_grid, Vt, cmap="viridis")

# Labels
ax.set_xlabel("Stock Price (S)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("Option Value (V(S,t))")

plt.show()

