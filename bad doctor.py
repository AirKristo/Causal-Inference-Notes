
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set up the simulation
np.random.seed(20200403)
n = 100_000

person = np.arange(1, n + 1)

# Potential outcomes: lifespan without and with a ventilator
y0 = np.random.normal(loc=9.4, scale=4, size=n)
y0 = np.maximum(y0, 0)

y1 = np.random.normal(loc=10, scale=4, size=n)
y1 = np.maximum(y1, 0)

# Individual treatment effects
delta = y1 - y0

# Bad doctor: treat the first 50,000 people
vents = np.zeros(n, dtype=int)
vents[:50_000] = 1

# Create the dataset
df = pd.DataFrame({
    "person": person,
    "y0": y0,
    "y1": y1,
    "delta": delta,
    "vents": vents,
})

# Aggregate causal parameters
ate = df["delta"].mean()
att = df.loc[df["vents"] == 1, "delta"].mean()
atu = df.loc[df["vents"] == 0, "delta"].mean()

# Realized outcome
df["y"] = df["vents"] * df["y1"] + (1 - df["vents"]) * df["y0"]

# Selection bias
ey01 = df.loc[df["vents"] == 1, "y0"].mean()
ey00 = df.loc[df["vents"] == 0, "y0"].mean()
selection_bias = ey01 - ey00

# Proportion treated
pi = df["vents"].mean()

# OLS regression with robust standard errors
X = sm.add_constant(df[["vents"]])
model = sm.OLS(df["y"], X).fit(cov_type="HC1")

# Decomposition of the observed difference in means
sdo = ate + selection_bias + (1 - pi) * (att - atu)

# Display results
print(f"ATE: {ate:.4f}")
print(f"ATT: {att:.4f}")
print(f"ATU: {atu:.4f}")
print(f"Selection bias: {selection_bias:.4f}")
print(f"Share treated: {pi:.4f}")

print(f"\nRegression coefficient: {model.params['vents']:.4f}")
print(f"Robust standard error: {model.bse['vents']:.4f}")

print(f"\nDecomposed difference in means: {sdo:.4f}")
print(f"Actual difference in means: "
      f"{df.loc[df['vents'] == 1, 'y'].mean() - df.loc[df['vents'] == 0, 'y'].mean():.4f}")
