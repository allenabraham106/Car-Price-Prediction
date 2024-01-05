import tensorflow as tf #models
import pandas as pd #reading and processing Data
import seaborn as sns #visualization
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv", ",")
print(data.head)
print(data.shape)

sns.pairplot(data[["v.id",	"on road old",	"on road now",	"years",	"km",	"rating",	"condition",	"economy",	"top speed",	"hp",	"torque","current price"]], diag_kind='kde')
plt.show()