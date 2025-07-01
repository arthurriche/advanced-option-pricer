# API Reference

## Models

### BlackScholesPricer

The main class for Black-Scholes option pricing.

#### Methods

- `price_call(S, K, T, r, sigma)` - Price a European call option
- `price_put(S, K, T, r, sigma)` - Price a European put option
- `all_greeks(S, K, T, r, sigma, option_type)` - Calculate all Greeks
- `implied_volatility(price, S, K, T, r, option_type)` - Calculate implied volatility

### Binomial Tree

- `binomial_tree_price(S, K, T, r, sigma, N, option_type, american)` - Price options using binomial tree

### Monte Carlo

- `monte_carlo_option_price(S, K, T, r, sigma, n_paths, option_type)` - Price options using Monte Carlo

## Risk Management

### Greeks

- `delta(S, K, T, r, sigma, option_type)` - Calculate delta
- `gamma(S, K, T, r, sigma)` - Calculate gamma
- `theta(S, K, T, r, sigma, option_type)` - Calculate theta
- `vega(S, K, T, r, sigma)` - Calculate vega
- `rho(S, K, T, r, sigma, option_type)` - Calculate rho

### VaR

- `historical_var(returns, confidence_level)` - Calculate historical VaR
- `parametric_var(mean, std, confidence_level)` - Calculate parametric VaR

## Utilities

### Market Data

- `get_sample_data(symbol, start, end)` - Get sample market data

### Visualization

- `plot_option_prices(strikes, prices, option_type)` - Plot option prices
- `plot_vol_surface(grid_K, grid_T, grid_vol)` - Plot volatility surface 