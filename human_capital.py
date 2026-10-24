
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set random seed and number of observations
np.random.seed(1000)
n = 10_000

# Generate the data
background = np.random.normal(size=n)

pe = 1 + background + np.random.normal(size=n)

income = 1 + 5 * pe + np.random.normal(size=n)

college = (
    1 + background + pe + income
    + np.random.normal(size=n)
)

earnings = (
    1 + 2 * college + income
    + np.random.normal(size=n)
)

# Combine variables into a DataFrame
df = pd.DataFrame({
    "background": background,
    "pe": pe,
    "income": income,
    "college": college,
    "earnings": earnings,
})

# Naive regression: earnings on college
X1 = sm.add_constant(df[["college"]])
model1 = sm.OLS(df["earnings"], X1).fit(cov_type="HC1")

print("Naive model")
print(model1.summary())

# Adjusted regression: earnings on college and income
X2 = sm.add_constant(df[["college", "income"]])
model2 = sm.OLS(df["earnings"], X2).fit(cov_type="HC1")

print("\nAdjusted model")
print(model2.summary())
