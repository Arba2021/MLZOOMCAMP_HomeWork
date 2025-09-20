import pandas as pd
import numpy as np
print(pd.__version__)
df = pd.read_csv("car_fuel_efficiency.csv")

len(df)
df['fuel_type'].unique()
(df.isna().sum(axis=0) > 0).sum()
df.loc[df.groupby("origin")["fuel_efficiency_mpg"].idxmax()]
horsepower = df['horsepower'].median()
print(horsepower)
new = df['horsepower'].value_counts().idxmax()
print(new)
df['horsepower'].fillna(new)
df['horsepower'].median()
asia = df[df['origin'] == 'Asia']
asia_subset = asia[['vehicle_weight', 'model_year']]
value7 = asia_subset.head(7)

x = value7.to_numpy()
print(x)
xtx = x.T @ x
print(xtx)
XTX_inv = np.linalg.inv(xtx)
print(XTX_inv)
y = [1100, 1300, 800, 900, 1000, 1100, 1200]
xt = x.T
middle = XTX_inv @ xt
w = middle @ y
print(w)
total = np.sum(w)
print(total)
