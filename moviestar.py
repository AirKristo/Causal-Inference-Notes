
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Reset random seed and create 2,500 observations
np.random.seed(3444)
n = 2500

# Generate independent standard normal variables
beauty = np.random.normal(size=n)
talent = np.random.normal(size=n)

# Create the collider: total score depends on beauty and talent
score = beauty + talent

# Find the 85th percentile cutoff
c85 = np.percentile(score, 85)

# Classify the top 15% as movie stars
star = (score >= c85).astype(int)

# Combine variables into a DataFrame
df = pd.DataFrame({
    "beauty": beauty,
    "talent": talent,
    "score": score,
    "star": star,
})

# Plot beauty against talent: non-stars, stars, and everyone
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

groups = [
    (df[df["star"] == 0], "Not a star"),
    (df[df["star"] == 1], "Movie star"),
    (df, "Everyone"),
]

for ax, (group, title) in zip(axes, groups):
    ax.scatter(
        group["talent"],
        group["beauty"],
        color="black",
        s=12,
        alpha=0.65,
    )
    ax.set_title(title)
    ax.set_xlabel("Talent")
    ax.set_ylabel("Beauty")

fig.suptitle("Aspiring actors and actresses")
fig.tight_layout()
plt.show()

# Regression 1: beauty on talent
X1 = sm.add_constant(df[["talent"]])
model1 = sm.OLS(df["beauty"], X1).fit()

print("Regression 1: beauty on talent")
print(model1.summary())

# Regression 2: beauty on talent and star status
X2 = sm.add_constant(df[["talent", "star"]])
model2 = sm.OLS(df["beauty"], X2).fit()

print("\nRegression 2: beauty on talent and star status")
print(model2.summary())
