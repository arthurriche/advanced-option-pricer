import numpy as np

class SimpleGreeksCalculator:
    def __init__(self, method="analytical"):
        self.method = method.lower()
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        
        def norm_cdf(x):
            return 0.5 * (1 + np.math.erf(x/np.sqrt(2)))
        
        if option_type.lower() == "call":
            delta = norm_cdf(d1)
        else:
            delta = norm_cdf(d1) - 1
        
        return {"delta": delta}

calculator = SimpleGreeksCalculator("analytical")
greeks = calculator.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
print("Simple Greeks:", greeks)
print("Test successful!")

