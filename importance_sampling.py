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
    
    plt.plot(range(len(v_list)), v_list, label='K={}'.format(k))
    plt.xlabel('iteration')
    plt.ylabel('Variance')
    plt.title("Variance of f(theta, Z) during Newton's Algorithm")
    plt.grid(True)
plt.legend()
plt.show()


def european_call_importance_sampling_MC(G, s0, K, r, sigma, T, N, mu):
    MC_price_list = np.zeros(len(mu))
    CI_lower_list = np.zeros(len(mu))
    CI_upper_list = np.zeros(len(mu))

    for (i,mu_values) in enumerate(mu):
        ST_IS = s0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*(G+mu_values))
        payoff_IS = np.exp(-r*T)*np.maximum(ST_IS-K,0)*np.exp(-mu_values**2/2-mu_values*G)
        MC_price_IS = np.mean(payoff_IS)

        #confidence interval
        std_IS = np.std(payoff_IS)
        error_IS = 1.664*std_IS/np.sqrt(N)
        CI_upper = MC_price_IS + error_IS
        CI_lower = MC_price_IS - error_IS

        MC_price_list[i] = MC_price_IS
        CI_lower_list[i] = CI_lower
        CI_upper_list[i] = CI_upper

    d1 = 1.0 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r + (sigma**2) / 2) * T)
    d2 = 1.0 / (sigma * np.sqrt(T)) * (np.log(S0 / K) + (r - (sigma**2) / 2) * T)
    True_price = S0 * normal_Abramowitz_Stegun(d1) - K * np.exp(-r * T) * normal_Abramowitz_Stegun(d2)

    return MC_price_list, CI_lower_list, CI_upper_list, True_price

K = 2.5
N_values = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
MC_price_theta0 = []
MC_price_theta_N = []
CI_lower_list_theta0 = []
CI_upper_list_theta0 = []
CI_lower_list_theta_N = []
CI_upper_list_theta_N = []

npr.seed(95566)  # for reproducibility
for N in N_values:
    G = npr.normal(0, 1, N)
    theta_N = theta_newton_algo(0, N, G, r, T, sigma, K, S0)[0]
    theta_list = [0, theta_N]
    MC_price_list, CI_lower_list, CI_upper_list, True_price = european_call_importance_sampling_MC(G, S0, K, r, sigma, T, N, theta_list)
    
    MC_price_theta0.append(MC_price_list[0])
    MC_price_theta_N.append(MC_price_list[1])
    CI_lower_list_theta0.append(CI_lower_list[0])
    CI_upper_list_theta0.append(CI_upper_list[0])
    CI_lower_list_theta_N.append(CI_lower_list[1])
    CI_upper_list_theta_N.append(CI_upper_list[1])

plt.plot(N_values, MC_price_theta0, label='MC price with theta=0')
plt.plot(N_values, MC_price_theta_N, label='MC price with theta=theta_N')
plt.fill_between(N_values, CI_lower_list_theta0, CI_upper_list_theta0, color='blue', alpha=0.2, label='CI 90% (theta=0)')
plt.fill_between(N_values, CI_lower_list_theta_N, CI_upper_list_theta_N, color='orange', alpha=0.2, label='CI 90% (theta=theta_N)')
plt.axhline(y=True_price, color='red', linestyle='--', label='True price')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Option Price')
plt.title('Option Price vs N for K=2.5')
plt.legend()
plt.grid(True)
plt.show()

#####

def h(X, lamb, K):
    return np.maximum(0, lamb[0]*X[0] + lamb[1]*X[1] + lamb[2]*X[2] - K)

def g_basket(X, lamb, K, r, T):
    return np.exp(-r*T)*h(X, lamb, K)

def u_basket(theta, X, lamb, K, r, T):
    return np.mean((theta - X) * g_basket(X, lamb, K, r, T)**2 * np.exp(-theta*X + 0.5*theta**2))

def grad_u_basket(theta, X, lamb, K, r, T):
    return np.mean((1 + (theta - X)**2) * g_basket(X, lamb, K, r, T)**2 * np.exp(-theta*X + 0.5*theta**2))

def theta_newton_algo_basket(theta_0, N, X, lamb, K, epsilon=1e-6, max_iter=1000):
    theta = theta_0
    theta_list = [theta]
    i = 0
    while (i < max_iter and abs(u_basket(theta, X, lamb, K, r, T)) > epsilon):
        theta = theta - u_basket(theta, X, lamb, K, r, T) / grad_u_basket(theta, X, lamb, K, r, T)
        theta_list.append(theta)
        i += 1
    return theta, theta_list

def price_option_basket_MC(X, K, lamb, r, T, N):
    
    payoff = g_basket(X, lamb, K, r, T)
    MC_price = np.mean(payoff)
    
    std = np.std(payoff)
    error = 1.96 * std / np.sqrt(N)
    CI_up = MC_price + error
    CI_down = MC_price - error

    return MC_price, CI_up, CI_down, error

def importance_sampling_basket_MC(X, K, lamb, r, T, N, theta, S0, sigma):
    MC_price_list = np.zeros(len(theta))
    CI_lower_list = np.zeros(len(theta))
    CI_upper_list = np.zeros(len(theta))

    for (i,theta_value) in enumerate(theta):
        ST_IS = [S0[i]*]
        payoff_IS = g_basket(X + theta_value, lamb, K, r, T) * np.exp(- theta_value**2/2 - theta_value*X)
        MC_price_IS = np.mean(payoff_IS)

        # confidence interval
        std_IS = np.std(payoff_IS)
        error_IS = 1.664*std_IS/np.sqrt(N)
        CI_upper = MC_price_IS + error_IS
        CI_lower = MC_price_IS - error_IS

        MC_price_list[i] = MC_price_IS
        CI_lower_list[i] = CI_lower
        CI_upper_list[i] = CI_upper

    return MC_price_list, CI_lower_list, CI_upper_list

gamma = np.array([1, 0.5, 0.5],
                 [0.5, 1, 0.5],
                 [0.5, 0.5, 1])

L = np.linalg.cholesky(gamma)

K = 1.25
r = 0.01
T = 1
lamb = [1/3, 1/3, 1/3]
S0 = [1, 1, 1]

N_values = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
MC_price_theta0 = []
MC_price_theta_N = []
CI_lower_list_theta0 = []
CI_upper_list_theta0 = []
CI_lower_list_theta_N = []
CI_upper_list_theta_N = []

npr.seed(95566)  # for reproducibility

for N in N_values:
    G = npr.normal(0, 1, (3, N))
    X = np.sqrt(T) * L@G
    theta_N = theta_newton_algo_basket(0, N, X, lamb, K, r, T)[0]
    theta_list = [0, theta_N]
    MC_price_list, CI_lower_list, CI_upper_list = importance_sampling_basket_MC(X, K, lamb, r, T, N, theta_list)
    
    MC_price_theta0.append(MC_price_list[0])
    MC_price_theta_N.append(MC_price_list[1])
    CI_lower_list_theta0.append(CI_lower_list[0])
    CI_upper_list_theta0.append(CI_upper_list[0])
    CI_lower_list_theta_N.append(CI_lower_list[1])
    CI_upper_list_theta_N.append(CI_upper_list[1])

plt.plot(N_values, MC_price_theta0, label='MC price with theta=0')
plt.plot(N_values, MC_price_theta_N, label='MC price with theta=theta_N')
plt.fill_between(N_values, CI_lower_list_theta0, CI_upper_list_theta0, color='blue', alpha=0.2, label='CI 90% (theta=0)')
plt.fill_between(N_values, CI_lower_list_theta_N, CI_upper_list_theta_N, color='orange', alpha=0.2, label='CI 90% (theta=theta_N)')
plt.xscale('log')
plt.xlabel('N (log scale)')
plt.ylabel('Option Price')
plt.title('Option Price vs N for K=1.25')
plt.legend()
plt.grid(True)
plt.show()


