
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Create 10,000 observations
np.random.seed(1000)
n = 10_000

# Generate gender, ability, and discrimination
female = (np.random.uniform(size=n) >= 0.5).astype(int)
ability = np.random.normal(size=n)
discrimination = female.copy()

# Generate occupation and wages
occupation = (
    1 + 2 * ability - 2 * discrimination
    + np.random.normal(size=n)
)

wage = (
    1 - discrimination + occupation + 2 * ability
    + np.random.normal(size=n)
)

# Combine variables into a DataFrame
df = pd.DataFrame({
    "female": female,
    "ability": ability,
    "discrimination": discrimination,
    "occupation": occupation,
    "wage": wage,
})

# Regression 1: Wage on discrimination
X1 = sm.add_constant(df[["discrimination"]])
model1 = sm.OLS(df["wage"], X1).fit()

# Regression 2: Add occupation
X2 = sm.add_constant(df[["discrimination", "occupation"]])
model2 = sm.OLS(df["wage"], X2).fit()

# Regression 3: Add ability as well
X3 = sm.add_constant(
    df[["discrimination", "occupation", "ability"]]
)
model3 = sm.OLS(df["wage"], X3).fit()

# Display the discrimination coefficient from each model
for name, model in [
    ("Model 1: Discrimination only", model1),
    ("Model 2: Add occupation", model2),
    ("Model 3: Add occupation and ability", model3),
]:
    print(f"\n{name}")
    print(f"Discrimination coefficient: "
          f"{model.params['discrimination']:.3f}")
    print(f"Standard error: "
          f"{model.bse['discrimination']:.3f}")
    print(f"R-squared: {model.rsquared:.3f}")
