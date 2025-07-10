"""Rewritten Gavish and Donoho code for optimal SVHT coefficients in Python."""

import numpy as np
import scipy.special as sp
from typing import Callable

def linspace(start: float, end: float, num_points: int) -> np.ndarray:
    """Equivalent to numpy.linspace."""
    if num_points == 1:
        return np.array([start])    
    return np.linspace(start, end, num_points)

def lobatto_quadrature(func:Callable[[float], float], a: float, b: float, n: int) -> float:
    if a >= b:
        raise ValueError("Upper limit must be greater than lower limit.")
    h = (b - a) / n
    result = 0.0
    for i in range(n):
        x0 = a + i * h
        x2 = x0 + h
        x1 = (x0 + x2) / 2.0
        result += (h / 6) * (func(x0) + 4 * func(x1) + func(x2))
    return result


def MarcenkoPasturIntegral(x: float, beta: float) -> float:
    if not (0 < beta <= 1):
        raise ValueError("Beta must be in the range (0, 1].")
    
    lobnd = (1 - np.sqrt(beta)) ** 2
    hibnd = (1 + np.sqrt(beta)) ** 2

    if not (lobnd <= x <= hibnd):
        raise ValueError("x must be in the range [lobnd, hibnd].")
    
    def dens(t):
        return np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * beta * t)
    
    n = 1000
    dx = (x - lobnd) / n
    xs = np.linspace(lobnd, x, n)
    return np.sum([dens(xi)* dx for xi in xs])

def incMarPass(x0: float, beta:float, gamma: float) -> float:
    if beta > 1:
        raise ValueError("Beta beyond!!!")
    
    topSpec = (1 + np.sqrt(beta)) ** 2
    botSpec = (1 - np.sqrt(beta)) ** 2

    def MarPass(x):
        return (np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * np.pi)
                if (topSpec - x) * (x - botSpec) > 0 else 0.0)
    
    def fun(x):
        return (x ** gamma) * MarPass(x)
    
    return lobatto_quadrature(fun, botSpec, x0, 12000)

def MedianMarcenkoPastur(beta: float) -> float:
    lobnd = (1 - np.sqrt(beta)) ** 2
    hibnd = (1 + np.sqrt(beta)) ** 2

    for _ in range(20000):
        mid = 0.5 * (lobnd + hibnd)
        cdf = incMarPass(mid, beta, 0)
        if abs(cdf - 0.5) < 1e-13:
            return mid
        if cdf < 0.5:
            lobnd = mid
        else:
            hibnd = mid
    return 0.5 * (lobnd + hibnd)

def optimal_SVHT_coef_sigma_known(beta:float) -> float:
    w = (8 * beta) / (beta + 1 + np.sqrt(beta **2 + 14 * beta + 1))
    lambda_star = np.sqrt(2 * (beta + 1) + w)
    return lambda_star
    

def optimal_SVHT_coef_sigma_unknown(beta:float) -> float:
    coef = optimal_SVHT_coef_sigma_known(beta)
    NPmedian = MedianMarcenkoPastur(beta)
    return coef / np.sqrt(NPmedian)

def optimal_SVHT_coef_sigma(beta: float, sigma_known: int) -> float:
    """Calculate the optimal SVHT coefficient based on whether sigma is known or not."""
    if sigma_known == 1:
        return optimal_SVHT_coef_sigma_known(beta)
    else:
        return optimal_SVHT_coef_sigma_unknown(beta)
