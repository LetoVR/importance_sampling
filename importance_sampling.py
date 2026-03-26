import numpy as np
import matplotlib.pyplot as plt


def g(x, r, T, sigma, K, S0):
    return np.exp(-r*T) * np.maximum(0, S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*x) - K)

def u(N, theta, X, r, T, sigma, K, S0):
    u = 0
    for i in range(N):
        u += (1/N)*(theta - X[i])*g(X[i], r, T, sigma, K, S0)**2*np.exp(-theta*X[i]+0.5*theta**2)
    return u

def grad_u(N, theta, X, r, T, sigma, K, S0):
    grad_u = 0
    for i in range(N):
        grad_u += (1/N)*(-2*X[i] + theta)*g(X[i], r, T, sigma, K, S0)**2*np.exp(-theta*X[i]+0.5*theta**2)
    return grad_u

def theta_newton_algo(theta_0, N, X, r, T, sigma, K, S0, epsilon=1e-6, max_iter=1000):
    theta = theta_0
    theta_list = [theta]
    i = 0
    while (i<max_iter and abs(grad_u(N, theta, X, r, T, sigma, K, S0)) > epsilon):
        theta = theta - u(N, theta, X, r, T, sigma, K, S0)/grad_u(N, theta, X, r, T, sigma, K, S0)
        theta_list.append(theta)
        i += 1
    return theta, theta_list

def normal_Abramowitz_Stegun(x):
    # Abramowitz and Stegun approximation for the normal distribution function
    if x >= 0:
        k = 1.0/(1.0 + 0.2316419*x)
        return 1.0 - (1.0/np.sqrt(2*np.pi))*np.exp(-x**2/2)*k*(0.319381530*k - 0.356563782*k**2 + 1.781477937*k**3 - 1.821255978*k**4 + 1.330274429*k**5)
    else:
        return 1 - normal_Abramowitz_Stegun(-x)

import numpy.random as npr

def price_option_MC(S0, K, T, r, sigma, N):
    # S0: initial stock price
    # K: strike price
    # T: time to maturity
    # r: risk-free interest rate
    # sigma: volatility of the stock
    # N: number of Monte Carlo simulations (sample size)

    # Box-Muller
    U = npr.random(N//2) # careful: N must be pair 
    V = npr.random(N//2)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V) 
    Y = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V) 
    Z = np.concatenate((X,Y)) # combine the two arrays to get N standard normal random variables
    
    payoff = g(Z, r, T, sigma, K, S0)
    MC_price = np.mean(payoff)
    
    # 95% confidence interval
    std = np.std(payoff)
    error = 1.96*std/np.sqrt(N)

    CI_up = MC_price + error
    CI_down = MC_price - error

    # true price using the Black-Scholes formula
    d1 = 1./(sigma*np.sqrt(T))*(np.log(K/S0) + (r + (sigma**2)/2)*T)
    d2 = 1./(sigma*np.sqrt(T))*(np.log(K/S0) + (r - (sigma**2)/2)*T)
    True_price = - K*np.exp(-r*T)*normal_Abramowitz_Stegun(d2) + S0*normal_Abramowitz_Stegun(d1)

    return MC_price, True_price, CI_up, CI_down, error

# our criteria for finding the optimal N is the difference between the MC price and the true price, which should be less than epsilon
def find_optimal_N(S0, K, T, r, sigma, epsilon=1e-1):
    N = int(10e6)
    price = price_option_MC(S0, K, T, r, sigma, N)
    while (abs(price[0] - price[1]) > epsilon):
        N *= 10
        price = price_option_MC(S0, K, T, r, sigma, N)
    return N

def plot_theta(theta_0, N, r, T, sigma, K, S0, epsilon=1e-6, max_iter=1000):
     # Box-Muller
    U = npr.random(N//2) # careful: N must be pair 
    V = npr.random(N//2)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V) 
    Y = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V) 
    Z = np.concatenate((X,Y)) # combine the two arrays to get N standard normal random variables
    
    theta_list = theta_newton_algo(theta_0, N, Z, r, T, sigma, K, S0, epsilon, max_iter)[1]
    index = []
    for i in range(len(theta_list)):
        index.append(i)
    
    plt.plot(index, theta_list)
    plt.xlabel('iteration')
    plt.ylabel('theta')
    plt.title('Convergence of theta in Newton\'s algorithm')
    plt.show()

S0 = 1
sigma = 0.3
r = 0.01
K = 1.2
T = 2 

N_opt = find_optimal_N(S0, K, T, r, sigma)
print("Optimal N: ", N_opt)
MC_price, True_price, CI_up, CI_down, error = price_option_MC(S0, K, T, r, sigma, N_opt)
print("MC price: ", MC_price)
print("True price: ", True_price)
print("95% confidence interval: [", CI_down, ", ", CI_up, "]")
print("Error: ", error)

plot_theta(0, N_opt, r, T, sigma, K, S0)
