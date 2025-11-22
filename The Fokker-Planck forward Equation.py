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

def lognormal_risk_neutral_pdf(S0, S_prime, t_prime, r, sigma, t=0):
    """
    Calculates the Risk-Neutral Probability Density Function (p_RN) for a future stock price S_prime.
    This uses the risk-free rate (r) in place of the real-world drift (mu).

    Args:
        S0 (float): Current stock price.
        S_prime (float or array): Future stock price(s).
        t_prime (float): Future time in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility (annualized).
        t (float): Current time (default is 0).

    Returns:
        float or array: The risk-neutral probability density at S_prime.
    """
    tau = t_prime - t  # Time remaining
    if tau <= 0:
        return np.zeros_like(S_prime) if not np.isscalar(S_prime) else 0.0

    # ******* KEY CHANGE: Use 'r' instead of 'mu' *******
    mu_risk_neutral = r
    
    # Calculate mean and variance of the log price
    mu_log = np.log(S0) + (mu_risk_neutral - 0.5 * sigma**2) * tau
    sigma_log_sq = sigma**2 * tau

    # Ensure S_prime is positive
    S_prime = np.where(S_prime <= 0, 1e-10, S_prime)
    
    # Calculate the PDF according to the lognormal formula
    exponent_term = - (np.log(S_prime) - mu_log)**2 / (2 * sigma_log_sq)
    prefactor = 1 / (S_prime * sigma * np.sqrt(2 * np.pi * tau))
    pdf = prefactor * np.exp(exponent_term)
    
    return pdf

# --- Example Usage and Comparison ---
S_initial = 100.0
future_time = 1.0
drift_mu = 0.10          # Real-world drift (10%)
risk_free_r = 0.05       # Risk-free rate (5%)
volatility_sigma = 0.20

S_future_values = np.linspace(50, 200, 500)

# 1. Real-World PDF (Your previous calculation)
pdf_real = lognormal_forward_pdf(S_initial, S_future_values, future_time, drift_mu, volatility_sigma)

# 2. Risk-Neutral PDF (The new calculation)
pdf_rn = lognormal_risk_neutral_pdf(S_initial, S_future_values, future_time, risk_free_r, volatility_sigma)

# --- Plotting the Comparison ---
plt.figure(figsize=(10, 6))
plt.plot(S_future_values, pdf_real, label=f'Real-World (μ={drift_mu*100:.0f}%)', color='darkgreen')
plt.plot(S_future_values, pdf_rn, label=f'Risk-Neutral (r={risk_free_r*100:.0f}%)', color='red', linestyle='--')

plt.axvline(S_initial * np.exp(drift_mu * future_time), color='darkgreen', linestyle=':', alpha=0.6)
plt.axvline(S_initial * np.exp(risk_free_r * future_time), color='red', linestyle=':', alpha=0.6)

plt.title('Comparison of Real-World vs. Risk-Neutral Probability Distributions')
plt.xlabel("Future Stock Price $S'$")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

def calculate_discounted_pricing_kernel(S0, S_prime, t_prime, mu, r, sigma, t=0):
    """
    Calculates the Discounted Pricing Kernel (SDF) as a function of the future stock price S'.
    M_tilde(S') = [Risk-Neutral PDF] / [Real-World PDF]
    
    Args:
        S0 (float): Current stock price.
        S_prime (float or array): Future stock price(s) (S').
        t_prime (float): Future time in years.
        mu (float): Real-world drift rate.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        t (float): Current time (default is 0).

    Returns:
        float or array: The discounted pricing kernel M_tilde(S') at the future time t_prime.
    """
    # 1. Calculate the Risk-Neutral PDF (p_RN) using 'r'

    tau = t_prime - t
    mu_log_rn = np.log(S0) + (r - 0.5 * sigma**2) * tau
    sigma_log_sq = sigma**2 * tau
    
    S_prime = np.where(S_prime <= 0, 1e-10, S_prime)
    
    exponent_rn = - (np.log(S_prime) - mu_log_rn)**2 / (2 * sigma_log_sq)
    prefactor = 1 / (S_prime * sigma * np.sqrt(2 * np.pi * tau))
    pdf_rn = prefactor * np.exp(exponent_rn)

    # 2. Calculate the Real-World PDF (p_Real) using 'mu'
    mu_log_real = np.log(S0) + (mu - 0.5 * sigma**2) * tau
    exponent_real = - (np.log(S_prime) - mu_log_real)**2 / (2 * sigma_log_sq)
    pdf_real = prefactor * np.exp(exponent_real)

    # 3. Calculate the ratio: M_tilde = p_RN / p_Rea
    # Simpler, less prone to floating point errors, is using the ratio of the full PDFs:
    kernel_ratio = pdf_rn / np.where(pdf_real == 0, 1e-10, pdf_real)
    
    return kernel_ratio

# --- Example Usage and Plotting ---
S_initial = 100.0
future_time = 1.0
drift_mu = 0.10          # Real-world drift (10%)
risk_free_r = 0.05       # Risk-free rate (5%)
volatility_sigma = 0.20

S_future_values = np.linspace(50, 200, 500)

kernel_values = calculate_discounted_pricing_kernel(
    S_initial, S_future_values, future_time, drift_mu, risk_free_r, volatility_sigma
)

# --- Plotting the Pricing Kernel ---
plt.figure(figsize=(10, 6))
plt.plot(S_future_values, kernel_values, label='Discounted Pricing Kernel $M(S\')$', color='purple')
plt.axhline(np.exp(-risk_free_r * future_time), color='gray', linestyle='--', linewidth=1, 
            label=f'Discount Factor $e^{{-r\\tau}}={np.exp(-risk_free_r*future_time):.3f}$')
plt.axvline(S_initial * np.exp((drift_mu + risk_free_r - volatility_sigma**2) * future_time / 2), 
            color='darkred', linestyle=':', alpha=0.6, label='Point of steepest decline')

plt.title(f'Discounted Pricing Kernel (SDF) as a function of Future Stock Price $S\'$')
plt.xlabel("Future Stock Price $S'$")
plt.ylabel("Discounted Pricing Kernel $e^{-r\u03c4}M(T)$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
