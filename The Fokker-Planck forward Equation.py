import numpy as np
import matplotlib.pyplot as plt

def lognormal_forward_pdf(S0, S_prime, t_prime, mu, sigma, t=0):
    """
    Calculates the probability density function (PDF) for a future stock price S_prime,
    based on the analytical solution of the Forward Kolmogorov Equation (lognormal distribution).

    Args:
        S0 (float): Current stock price (S at time t).
        S_prime (float or array): Future stock price(s) (S' at time t').
        t_prime (float): Future time t'.
        mu (float): Real-world drift rate (expected return).
        sigma (float): Volatility.
        t (float): Current time (default is 0).

    Returns:
        float or array: The probability density at S_prime.
    """
    tau = t_prime - t  # Time remaining
    if tau <= 0:
        return np.inf if S_prime == S0 else 0.0 # Delta function at t'=t

    # Calculate mean and variance of the log price
    mu_log = np.log(S0) + (mu - 0.5 * sigma**2) * tau
    sigma_log_sq = sigma**2 * tau

    # Ensure S_prime is not zero to avoid division by zero in log
    S_prime = np.where(S_prime <= 0, 1e-10, S_prime)
    
    # Calculate the PDF according to the lognormal formula
    exponent_term = - (np.log(S_prime) - mu_log)**2 / (2 * sigma_log_sq)
    prefactor = 1 / (S_prime * sigma * np.sqrt(2 * np.pi * tau))
    pdf = prefactor * np.exp(exponent_term)
    
    return pdf

# --- Example Usage and Plotting ---

S_initial = 100.0        # Current stock price
future_time = 1.0        # Time horizon (1 year)
drift_mu = 0.10          # Real-world drift (10% annualized)
volatility_sigma = 0.20  # Volatility (20% annualized)

# 1. Generate the range of future prices (x-axis)
S_future_values = np.linspace(50, 200, 500) 

# 2. Calculate the probability densities (y-axis)
probability_densities = lognormal_forward_pdf(
    S_initial, S_future_values, future_time, drift_mu, volatility_sigma
)

# 3. Calculate the Expected Price E[S_T] for annotation
# E[S_T] = S0 * exp(mu * tau)
expected_price = S_initial * np.exp(drift_mu * future_time)

# 4. Find the Peak Price (Mode) for annotation
# Mode = S0 * exp((mu - sigma^2) * tau)
mode_price = S_initial * np.exp((drift_mu - volatility_sigma**2) * future_time)


# --- Plotting the PDF ---
plt.figure(figsize=(10, 6))
plt.plot(S_future_values, probability_densities, label=f'PDF at T={future_time} yr', color='darkblue')

# Annotations
plt.axvline(S_initial, color='grey', linestyle='--', linewidth=1, label=f'Initial Price S₀ = {S_initial:.0f}')
plt.axvline(expected_price, color='green', linestyle='-', linewidth=1, label=f'Expected Price E[Sᵀ] = {expected_price:.2f}')
plt.axvline(mode_price, color='red', linestyle=':', linewidth=1, label=f'Mode (Most Likely) = {mode_price:.2f}')

plt.title(f'Lognormal Probability Density Function (Forward Equation Solution)\n$S_0={S_initial}, \mu={drift_mu}, \sigma={volatility_sigma}, T={future_time}$ year')
plt.xlabel("Future Stock Price $S'$")
plt.ylabel("Probability Density $p(S', t')$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()