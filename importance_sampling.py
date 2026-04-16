import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

def g(x, r, T, sigma, K, S0):
    return np.exp(-r*T) * np.maximum(0, S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*x) - K)

def u(N, theta, X, r, T, sigma, K, S0):
    return np.mean((theta - X) * g(X, r, T, sigma, K, S0)**2 * np.exp(-theta*X + 0.5*theta**2))

def grad_u(N, theta, X, r, T, sigma, K, S0):
    return np.mean((1 + (theta - X)**2) * g(X, r, T, sigma, K, S0)**2 * np.exp(-theta*X + 0.5*theta**2))

def theta_newton_algo(theta_0, N, X, r, T, sigma, K, S0, epsilon=1e-6, max_iter=1000):
    theta = theta_0
    theta_list = [theta]
    i = 0
    while (i < max_iter and abs(u(N, theta, X, r, T, sigma, K, S0)) > epsilon):
        theta = theta - u(N, theta, X, r, T, sigma, K, S0) / grad_u(N, theta, X, r, T, sigma, K, S0)
        theta_list.append(theta)
        i += 1
    return theta, theta_list

# def theta_newton_algo(theta_0, N, X, r, T, sigma, K, S0, epsilon=1e-6, max_iter=1000):
#     theta = theta_0
#     theta_list = [theta]
#     i = 0
    
#     while i < max_iter:
#         val_u = u(N, theta, X, r, T, sigma, K, S0)
        
#         # Condition d'arrêt
#         if abs(val_u) <= epsilon:
#             break
            
#         # Direction de Newton classique
#         step = - val_u / grad_u(N, theta, X, r, T, sigma, K, S0)
        
#         # Algorithme à pas décroissant (Damped Newton)
#         alpha = 1.0
#         while abs(u(N, theta + alpha * step, X, r, T, sigma, K, S0)) >= abs(val_u) and alpha > 1e-4:
#             alpha /= 2.0
            
#         theta = theta + alpha * step
#         theta_list.append(theta)
#         i += 1
        
#     return theta, theta_list

def normal_Abramowitz_Stegun(x):
    if x >= 0:
        k = 1.0 / (1.0 + 0.2316419 * x)
        return 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2) * (0.319381530 * k - 0.356563782 * k**2 + 1.781477937 * k**3 - 1.821255978 * k**4 + 1.330274429 * k**5)
    else:
        return 1.0 - normal_Abramowitz_Stegun(-x)

import numpy.random as npr

def price_option_MC(Z, S0, K, T, r, sigma, N):
    
    payoff = g(Z, r, T, sigma, K, S0)
    MC_price = np.mean(payoff)
    
    std = np.std(payoff)
    error = 1.96 * std / np.sqrt(N)
    CI_up = MC_price + error
    CI_down = MC_price - error

    d1 = 1.0 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r + (sigma**2) / 2) * T)
    d2 = 1.0 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r - (sigma**2) / 2) * T)
    True_price = S0 * normal_Abramowitz_Stegun(d1) - K * np.exp(-r * T) * normal_Abramowitz_Stegun(d2)

    return MC_price, True_price, CI_up, CI_down, error

def find_optimal_N(Z, S0, K, T, r, sigma, epsilon=1e-2):
    N_est = 10000
    MC_price, True_price, CI_up, CI_down, error_est = price_option_MC(Z, S0, K, T, r, sigma, N_est)
    std_est = error_est * np.sqrt(N_est) / 1.96
    
    N_opt = int((1.96 * std_est / epsilon)**2) # N_opt is the optimal number of samples to achieve the desired error epsilon with 95% confidence
    if N_opt % 2 != 0:
        N_opt += 1 # to make sure that N_opt is even for Box-Muller
    
    return max(N_opt, 10000)

def plot_theta(Z, theta_0, N, r, T, sigma, K, S0, epsilon=1e-6, max_iter=1000): 
    
    theta_list = theta_newton_algo(theta_0, N, Z, r, T, sigma, K, S0, epsilon, max_iter)[1]
    
    plt.plot(range(len(theta_list)), theta_list, marker='o', markersize=4, label='K={}'.format(K))
    plt.xlabel('iteration')
    plt.ylabel('theta')
    plt.title("Theta in Newton's Algorithm")
    plt.grid(True)

    print(f"Value of u at optimal theta: {u(N, theta_list[-1], Z, r, T, sigma, K, S0)}")

    return theta_list

def f(theta, x, r, T, sigma, K, S0):
    return g(x+theta, r, T, sigma, K, S0) * np.exp(-theta*x - 0.5*theta**2)

S0 = 1
sigma = 0.3
r = 0.01
K = 1
T = 2 

Z = npr.normal(0, 1, 1000000)
N_opt = find_optimal_N(Z, S0, K, T, r, sigma, epsilon=0.001)
print(f"\nOptimal N: {N_opt}")

U = npr.random(N_opt // 2) 
V = npr.random(N_opt // 2)
X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V) 
Y = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V) 
Z = np.concatenate((X, Y))

Ks = [0.35, 0.54, 0.7, 1.24, 1.6, 2.5]
for k in Ks:
    MC_price, True_price, CI_up, CI_down, error = price_option_MC(Z, S0, k, T, r, sigma, N_opt)
    print(f"MC price: {MC_price:.4f}")
    print(f"True price: {True_price:.4f}")
    print(f"95% confidence interval: [{CI_down:.4f}, {CI_up:.4f}]")
    print(f"Error: {error:.4f}")
    
    theta_list = plot_theta(Z, 0, N_opt, r, T, sigma, k, S0)
plt.legend()
plt.show()

for k in Ks:
    theta_list = theta_newton_algo(0, N_opt, Z, r, T, sigma, k, S0)[1]
    v_list = np.zeros(len(theta_list))
    for i, theta in enumerate(theta_list):
        f_theta = f(theta, Z, r, T, sigma, k, S0)
        variance = (N_opt)/(N_opt-1) * (np.mean(f_theta**2) - (np.mean(f_theta))**2)
        v = np.sqrt(variance / N_opt)
        v_list[i] = v
    
    plt.plot(range(len(v_list)), v_list, label='Variance of f(theta, Z) for K={}'.format(k))
    plt.xlabel('iteration')
    plt.ylabel('Variance')
    plt.title("Variance of f(theta, Z) during Newton's Algorithm")
    plt.grid(True)
plt.legend()
plt.show()
