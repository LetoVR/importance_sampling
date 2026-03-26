import numpy as np
import matplotlib.pyplot as plt


def g(x, r, T, sigma, K, S0):
    return np.exp(-r*T) * np.max(0, S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*x) - K)

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
    i = 0
    while (i<max_iter and abs(grad_u(N, theta, X, r, T, sigma, K, S0)) > epsilon):
        theta = theta - u(N, theta, X, r, T, sigma, K, S0)/grad_u(N, theta, X, r, T, sigma, K, S0)
        i += 1
    return theta

def normal_Abramowitz_Stegun(x):
    # Abramowitz and Stegun approximation for the normal distribution function
    if x >= 0:
        k = 1.0/(1.0 + 0.2316419*x)
        return 1.0 - (1.0/np.sqrt(2*np.pi))*np.exp(-x**2/2)*k*(0.319381530*k - 0.356563782*k**2 + 1.781477937*k**3 - 1.821255978*k**4 + 1.330274429*k**5)
    else:
        return 1 - normal_Abramowitz_Stegun(-x)

def price_option_MC(S0, K, T, r, sigma, N):
    # S0: initial stock price
    # K: strike price
    # T: time to maturity
    # r: risk-free interest rate
    # sigma: volatility of the stock
    # N: number of Monte Carlo simulations (sample size)

    # model (always have such a section in your MC code)
    X = npr.normal(0, 1, N)
    
    payoff = np.exp(-r*T)*np.maximum(S-K, 0) # call function

    MC_price = np.mean(payoff)

    # 95% confidence interval
    std = np.std(payoff) # standard deviation estimator
    error = 1.96*std/np.sqrt(N)

    CI_up = MC_price + error
    CI_down = MC_price - error

    # true price using the Black-Scholes formula
    d1 = 1./(sigma*np.sqrt(T))*(np.log(S0/K) + (r + (sigma**2)/2)*T)
    d2 = 1./(sigma*np.sqrt(T))*(np.log(S0/K) + (r - (sigma**2)/2)*T)
    True_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)