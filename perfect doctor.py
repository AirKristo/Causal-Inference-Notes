
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reproducible simulation (Python draws will differ from Stata)
np.random.seed(20200403)
n = 100_000

# Identify each person
person = np.arange(1, n + 1)

# Potential outcomes: lifespan without and with a ventilator
y0 = np.random.normal(loc=9.4, scale=4, size=n)
y0 = np.maximum(y0, 0)

y1 = np.random.normal(loc=10, scale=4, size=n)
y1 = np.maximum(y1, 0)

# Individual treatment effect
delta = y1 - y0

# Perfect doctor treats only people with positive treatment effects
vents = (delta > 0).astype(int)

# Put the data into a DataFrame
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

# Realized outcome: observe y1 if treated, otherwise y0
df["y"] = df["vents"] * df["y1"] + (1 - df["vents"]) * df["y0"]

# Selection bias: difference in untreated potential outcomes
ey01 = df.loc[df["vents"] == 1, "y0"].mean()
ey00 = df.loc[df["vents"] == 0, "y0"].mean()
selection_bias = ey01 - ey00

# Share treated
pi = df["vents"].mean()

# Regression: observed outcome on treatment, with robust SEs
X = sm.add_constant(df[["vents"]])
model = sm.OLS(df["y"], X).fit(cov_type="HC1")

print("Causal parameters")
print(f"ATE: {ate:.4f}")
print(f"ATT: {att:.4f}")
print(f"ATU: {atu:.4f}")
print(f"Selection bias: {selection_bias:.4f}")
print(f"Share treated: {pi:.4f}")

print("\nObserved difference in means")
print(f"OLS coefficient on vents: {model.params['vents']:.4f}")
print(f"Robust standard error: {model.bse['vents']:.4f}")

# Decompose the observed difference in means
sdo = ate + selection_bias + (1 - pi) * (att - atu)

print(f"\nDecomposed difference in means: {sdo:.4f}")
print(f"Direct difference in means: "
      f"{df.loc[df['vents'] == 1, 'y'].mean() - df.loc[df['vents'] == 0, 'y'].mean():.4f}")
