# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from tqdm import tqdm
# import polars as pl
#
# # Read the CSV file
# df = pd.read_csv('train.csv')
#
# # Convert and save as a Parquet file
# df.to_parquet('train.parquet')
#
# data = pl.read_parquet("train.parquet")
#
# num_stocks = data["stock_id"].n_unique()
# num_dates = data["date_id"].n_unique()
# num_updates = data["seconds_in_bucket"].n_unique()
#
# print(f"# stocks         : {num_stocks}")
# print(f"# dates          : {num_dates}")
# print(f"# updates per day: {num_updates}")
#
# data = data.sort(by=["stock_id", "date_id", "seconds_in_bucket"])
# data = data.with_columns( [
#     (((pl.col('wap').shift(-6).over(["stock_id", "date_id"]) /
#        pl.col('wap').over(["stock_id", "date_id"])) - 1)).alias('stock_return')] )
#
# data = data.with_columns([(pl.col("stock_return") - pl.col("target") / 10_000).alias("index_return")])
#
# stock_returns = data.pivot(values="stock_return", index=["date_id", "seconds_in_bucket"], columns="stock_id", maintain_order=True)
# stock_returns = stock_returns.drop(columns=["date_id", "seconds_in_bucket"]).to_numpy().T.reshape(num_stocks, num_dates, num_updates)
#
# index_returns = data.pivot(values="index_return", index=["date_id", "seconds_in_bucket"], columns="stock_id", maintain_order=True)
# index_returns_np = index_returns.drop(columns=["date_id", "seconds_in_bucket"]).to_numpy().T.reshape(num_stocks, num_dates, num_updates)
# index_return = np.mean(index_returns_np, axis=0)


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm


# Read the CSV file
df = pd.read_csv('train.csv')

# Convert and save as a Parquet file
df.to_parquet('train.parquet')

data = pd.read_parquet("train.parquet")

num_stocks = data["stock_id"].nunique()
num_dates = data["date_id"].nunique()
num_updates = data["seconds_in_bucket"].nunique()

print(f"# stocks         : {num_stocks}")
print(f"# dates          : {num_dates}")
print(f"# updates per day: {num_updates}")

stock_returns = np.zeros((num_stocks, num_dates, num_updates))
index_returns = np.zeros((num_stocks, num_dates, num_updates))

for (stock_id, date_id), frame in tqdm(data.groupby(["stock_id", "date_id"])):
    frame["stock_return"] = ((frame["wap"] / frame["wap"].shift(6)).shift(-6) - 1) * 10_000
    frame["index_return"] = frame["stock_return"] - frame["target"]

    stock_returns[stock_id, date_id] = frame["stock_return"].values
    index_returns[stock_id, date_id] = frame["index_return"].values

index_return = np.mean(index_returns, axis=0)

lr = LinearRegression()
y = index_return.reshape(-1)
X = stock_returns.reshape((num_stocks, -1)).T

mask = ~((np.isnan(y) | np.isnan(X).any(axis=1)))
X, y = X[mask], y[mask]

lr.fit(X, y)

print(" Fit ".center(80, ">"))
print("Coef:", lr.coef_)
print("Intercept:", lr.intercept_)
print("R2:", r2_score(y, lr.predict(X)))

lr.coef_ = lr.coef_.round(3)
lr.intercept_ = 0.0
print(" Round with 3 digits ".center(80, ">"))
print("Coef:", lr.coef_)
print("Sum of Coef:", lr.coef_.sum())
print("R2:", r2_score(y, lr.predict(X)))


# stocks         : 200
# dates          : 481
# updates per day: 55
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Coef: [0.004      0.00099987 0.00200041 0.00599891 0.00400068 0.00399949
#        0.00200014 0.0059992  0.00600029 0.00200012 0.00200072 0.00800012
#        0.00600039 0.00200007 0.0080014  0.00600051 0.00199993 0.0059999
#        0.00400013 0.00199974 0.00399933 0.00099993 0.00599955 0.00399971
#        0.00199876 0.00200008 0.00400097 0.0020008  0.00400021 0.00400005
#        0.00100001 0.00100016 0.00199924 0.00199993 0.00599995 0.00399917
#        0.00400017 0.00399906 0.00599977 0.00200015 0.0020007  0.03999955
#        0.0020001  0.00199981 0.00399968 0.03999983 0.00200033 0.00099959
#        0.00600002 0.00399944 0.00399979 0.00600017 0.00099921 0.00399993
#        0.00399936 0.0019985  0.00599995 0.00400028 0.00599981 0.00400045
#        0.00600029 0.00399988 0.00200012 0.00100003 0.00200041 0.00400043
#        0.00200011 0.00799989 0.00400056 0.00400011 0.00199956 0.00399921
#        0.00599994 0.00199992 0.00399979 0.00400047 0.00200035 0.00399996
#        0.00400003 0.00399989 0.00100029 0.00199997 0.00200019 0.00799969
#        0.02000066 0.00400029 0.00600019 0.00200024 0.02       0.00199965
#        0.00199968 0.00599962 0.00399987 0.00199997 0.00099953 0.02000033
#        0.00600029 0.00099946 0.00200062 0.00399946 0.00100021 0.00199986
#        0.00600047 0.00600071 0.00399889 0.00600028 0.0010003  0.00199986
#        0.00400008 0.00600071 0.00600016 0.00100013 0.04000011 0.00599947
#        0.00200012 0.00400061 0.00199956 0.00199961 0.00599985 0.00200055
#        0.00200083 0.00399993 0.00599961 0.00600028 0.00200005 0.00199999
#        0.0080003  0.00600021 0.00400044 0.00199994 0.00599988 0.00200002
#        0.00399908 0.00600099 0.00200021 0.00399981 0.00100042 0.00399994
#        0.00199993 0.0039995  0.00800063 0.00600055 0.00799953 0.00200005
#        0.00400064 0.00199988 0.00099981 0.00399912 0.00399992 0.00399974
#        0.00599995 0.00800022 0.00400055 0.00099968 0.00099956 0.00200017
#        0.00599973 0.00400059 0.00100004 0.00199961 0.00599971 0.00400006
#        0.00599987 0.00800007 0.00200054 0.00200102 0.00400114 0.00199923
#        0.0399987  0.00199969 0.00199982 0.00399976 0.00199995 0.00200001
#        0.00599982 0.02000055 0.0039996  0.00200022 0.00600002 0.01999984
#        0.00100015 0.00200021 0.0059996  0.00400055 0.00599988 0.00400022
#        0.00399931 0.0040002  0.0039998  0.002      0.00399975 0.04000041
#        0.00200037 0.00799998 0.00199977 0.00400093 0.00099996 0.00400022
#        0.00599947 0.00400014]
# Intercept: 3.6043735647806052e-06
# R2: 0.99999999573038
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Round with 3 digits >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Coef: [0.004 0.001 0.002 0.006 0.004 0.004 0.002 0.006 0.006 0.002 0.002 0.008
#        0.006 0.002 0.008 0.006 0.002 0.006 0.004 0.002 0.004 0.001 0.006 0.004
#        0.002 0.002 0.004 0.002 0.004 0.004 0.001 0.001 0.002 0.002 0.006 0.004
#        0.004 0.004 0.006 0.002 0.002 0.04  0.002 0.002 0.004 0.04  0.002 0.001
#        0.006 0.004 0.004 0.006 0.001 0.004 0.004 0.002 0.006 0.004 0.006 0.004
#        0.006 0.004 0.002 0.001 0.002 0.004 0.002 0.008 0.004 0.004 0.002 0.004
#        0.006 0.002 0.004 0.004 0.002 0.004 0.004 0.004 0.001 0.002 0.002 0.008
#        0.02  0.004 0.006 0.002 0.02  0.002 0.002 0.006 0.004 0.002 0.001 0.02
#        0.006 0.001 0.002 0.004 0.001 0.002 0.006 0.006 0.004 0.006 0.001 0.002
#        0.004 0.006 0.006 0.001 0.04  0.006 0.002 0.004 0.002 0.002 0.006 0.002
#        0.002 0.004 0.006 0.006 0.002 0.002 0.008 0.006 0.004 0.002 0.006 0.002
#        0.004 0.006 0.002 0.004 0.001 0.004 0.002 0.004 0.008 0.006 0.008 0.002
#        0.004 0.002 0.001 0.004 0.004 0.004 0.006 0.008 0.004 0.001 0.001 0.002
#        0.006 0.004 0.001 0.002 0.006 0.004 0.006 0.008 0.002 0.002 0.004 0.002
#        0.04  0.002 0.002 0.004 0.002 0.002 0.006 0.02  0.004 0.002 0.006 0.02
#        0.001 0.002 0.006 0.004 0.006 0.004 0.004 0.004 0.004 0.002 0.004 0.04
#        0.002 0.008 0.002 0.004 0.001 0.004 0.006 0.004]
# Sum of Coef: 1.0000000000000002
# R2: 0.999999995685508