"""
Finite Difference Method for Option Pricing

This module implements finite difference methods for option pricing,
including explicit, implicit, and Crank-Nicolson schemes.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class FiniteDifferenceModel:
    """
    Finite Difference Model for option pricing using various numerical schemes.
    
    Supports both European and American options with different boundary conditions.
    """
    
    def __init__(self, scheme: str = 'crank_nicolson'):
        """
        Initialize the finite difference model.
        
        Args:
            scheme: Numerical scheme to use ('explicit', 'implicit', 'crank_nicolson')
        """
        self.scheme = scheme.lower()
        if self.scheme not in ['explicit', 'implicit', 'crank_nicolson']:
            raise ValueError("Scheme must be 'explicit', 'implicit', or 'crank_nicolson'")
    
    def price_option(self, 
                    S: float,
                    K: float, 
                    T: float,
                    r: float,
                    sigma: float,
                    option_type: str = 'call',
                    american: bool = False,
                    N: int = 100,
                    M: int = 100) -> float:
        """
        Price an option using finite difference methods.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            american: Whether option is American style
            N: Number of time steps
            M: Number of price steps
            
        Returns:
            Option price
        """
        if self.scheme == 'explicit':
            return self._explicit_fd(S, K, T, r, sigma, option_type, american, N, M)
        elif self.scheme == 'implicit':
            return self._implicit_fd(S, K, T, r, sigma, option_type, american, N, M)
        else:  # crank_nicolson
            return self._crank_nicolson_fd(S, K, T, r, sigma, option_type, american, N, M)
    
    def _explicit_fd(self, S, K, T, r, sigma, option_type, american, N, M):
        """Explicit finite difference scheme."""
        dt = T / N
        dS = 2 * S / M
        S_max = M * dS
        
        # Create grid
        S_grid = np.linspace(0, S_max, M + 1)
        t_grid = np.linspace(0, T, N + 1)
        
        # Initialize option value matrix
        V = np.zeros((N + 1, M + 1))
        
        # Terminal condition
        if option_type.lower() == 'call':
            V[N, :] = np.maximum(S_grid - K, 0)
        else:
            V[N, :] = np.maximum(K - S_grid, 0)
        
        # Boundary conditions
        if option_type.lower() == 'call':
            V[:, 0] = 0  # V(0, t) = 0
            V[:, M] = S_max - K * np.exp(-r * (T - t_grid))  # V(S_max, t)
        else:
            V[:, 0] = K * np.exp(-r * (T - t_grid))  # V(0, t)
            V[:, M] = 0  # V(S_max, t) = 0
        
        # Explicit scheme coefficients
        alpha = 0.5 * sigma**2 * S_grid**2 * dt / (dS**2)
        beta = r * S_grid * dt / (2 * dS)
        gamma = r * dt
        
        # Time stepping
        for i in range(N - 1, -1, -1):
            for j in range(1, M):
                V[i, j] = (alpha[j] * V[i + 1, j + 1] + 
                          (1 - 2 * alpha[j] - gamma) * V[i + 1, j] +
                          alpha[j] * V[i + 1, j - 1] +
                          beta[j] * (V[i + 1, j + 1] - V[i + 1, j - 1]))
                
                if american:
                    # Early exercise condition
                    if option_type.lower() == 'call':
                        V[i, j] = max(V[i, j], S_grid[j] - K)
                    else:
                        V[i, j] = max(V[i, j], K - S_grid[j])
        
        # Interpolate to get price at current S
        j = int(S / dS)
        if j >= M:
            j = M - 1
        
        return V[0, j]
    
    def _implicit_fd(self, S, K, T, r, sigma, option_type, american, N, M):
        """Implicit finite difference scheme."""
        dt = T / N
        dS = 2 * S / M
        S_max = M * dS
        
        # Create grid
        S_grid = np.linspace(0, S_max, M + 1)
        t_grid = np.linspace(0, T, N + 1)
        
        # Initialize option value matrix
        V = np.zeros((N + 1, M + 1))
        
        # Terminal condition
        if option_type.lower() == 'call':
            V[N, :] = np.maximum(S_grid - K, 0)
        else:
            V[N, :] = np.maximum(K - S_grid, 0)
        
        # Boundary conditions
        if option_type.lower() == 'call':
            V[:, 0] = 0
            V[:, M] = S_max - K * np.exp(-r * (T - t_grid))
        else:
            V[:, 0] = K * np.exp(-r * (T - t_grid))
            V[:, M] = 0
        
        # Implicit scheme coefficients
        alpha = 0.5 * sigma**2 * S_grid**2 * dt / (dS**2)
        beta = r * S_grid * dt / (2 * dS)
        gamma = r * dt
        
        # Time stepping with tridiagonal solver
        for i in range(N - 1, -1, -1):
            # Build tridiagonal matrix
            A = np.zeros((M + 1, M + 1))
            b = np.zeros(M + 1)
            
            for j in range(1, M):
                A[j, j-1] = -alpha[j] + beta[j]
                A[j, j] = 1 + 2 * alpha[j] + gamma
                A[j, j+1] = -alpha[j] - beta[j]
                b[j] = V[i + 1, j]
            
            # Boundary conditions
            A[0, 0] = 1
            A[M, M] = 1
            b[0] = V[i, 0]
            b[M] = V[i, M]
            
            # Solve tridiagonal system
            V[i, :] = np.linalg.solve(A, b)
            
            if american:
                # Early exercise condition
                if option_type.lower() == 'call':
                    V[i, :] = np.maximum(V[i, :], S_grid - K)
                else:
                    V[i, :] = np.maximum(V[i, :], K - S_grid)
        
        # Interpolate to get price at current S
        j = int(S / dS)
        if j >= M:
            j = M - 1
        
        return V[0, j]
    
    def _crank_nicolson_fd(self, S, K, T, r, sigma, option_type, american, N, M):
        """Crank-Nicolson finite difference scheme."""
        dt = T / N
        dS = 2 * S / M
        S_max = M * dS
        
        # Create grid
        S_grid = np.linspace(0, S_max, M + 1)
        t_grid = np.linspace(0, T, N + 1)
        
        # Initialize option value matrix
        V = np.zeros((N + 1, M + 1))
        
        # Terminal condition
        if option_type.lower() == 'call':
            V[N, :] = np.maximum(S_grid - K, 0)
        else:
            V[N, :] = np.maximum(K - S_grid, 0)
        
        # Boundary conditions
        if option_type.lower() == 'call':
            V[:, 0] = 0
            V[:, M] = S_max - K * np.exp(-r * (T - t_grid))
        else:
            V[:, 0] = K * np.exp(-r * (T - t_grid))
            V[:, M] = 0
        
        # Crank-Nicolson coefficients
        alpha = 0.5 * sigma**2 * S_grid**2 * dt / (2 * dS**2)
        beta = r * S_grid * dt / (4 * dS)
        gamma = r * dt / 2
        
        # Time stepping
        for i in range(N - 1, -1, -1):
            # Build matrices for Crank-Nicolson
            A = np.zeros((M + 1, M + 1))
            B = np.zeros((M + 1, M + 1))
            b = np.zeros(M + 1)
            
            for j in range(1, M):
                # Matrix A (implicit part)
                A[j, j-1] = -alpha[j] + beta[j]
                A[j, j] = 1 + 2 * alpha[j] + gamma
                A[j, j+1] = -alpha[j] - beta[j]
                
                # Matrix B (explicit part)
                B[j, j-1] = alpha[j] - beta[j]
                B[j, j] = 1 - 2 * alpha[j] - gamma
                B[j, j+1] = alpha[j] + beta[j]
            
            # Boundary conditions
            A[0, 0] = 1
            A[M, M] = 1
            B[0, 0] = 1
            B[M, M] = 1
            
            # Right-hand side
            b = B @ V[i + 1, :]
            b[0] = V[i, 0]
            b[M] = V[i, M]
            
            # Solve system
            V[i, :] = np.linalg.solve(A, b)
            
            if american:
                # Early exercise condition
                if option_type.lower() == 'call':
                    V[i, :] = np.maximum(V[i, :], S_grid - K)
                else:
                    V[i, :] = np.maximum(V[i, :], K - S_grid)
        
        # Interpolate to get price at current S
        j = int(S / dS)
        if j >= M:
            j = M - 1
        
        return V[0, j]


def finite_difference_example():
    """Example usage of finite difference methods."""
    print("=== Finite Difference Option Pricing Example ===\n")
    
    # Model parameters
    S = 100.0  # Current stock price
    K = 100.0  # Strike price
    T = 1.0    # Time to maturity
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Create models
    explicit_model = FiniteDifferenceModel('explicit')
    implicit_model = FiniteDifferenceModel('implicit')
    cn_model = FiniteDifferenceModel('crank_nicolson')
    
    # Price European call option
    print(f"Pricing European call option:")
    print(f"S = {S}, K = {K}, T = {T}, r = {r}, sigma = {sigma}\n")
    
    price_explicit = explicit_model.price_option(S, K, T, r, sigma, 'call')
    price_implicit = implicit_model.price_option(S, K, T, r, sigma, 'call')
    price_cn = cn_model.price_option(S, K, T, r, sigma, 'call')
    
    print(f"Explicit FD:     {price_explicit:.6f}")
    print(f"Implicit FD:     {price_implicit:.6f}")
    print(f"Crank-Nicolson:  {price_cn:.6f}")
    
    # Price American put option
    print(f"\nPricing American put option:")
    price_american = cn_model.price_option(S, K, T, r, sigma, 'put', american=True)
    print(f"American Put:    {price_american:.6f}")


if __name__ == "__main__":
    finite_difference_example() 