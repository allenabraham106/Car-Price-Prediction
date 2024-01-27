import tensorflow as tf #models
import pandas as pd#reading and processing Data
import seaborn as sns #visualization
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import pydot
import graphviz

data = pd.read_csv("Car-Price-Prediction/train.csv")

print(data.head)
print(data.shape)

sns.pairplot(data[["v.id",	"on road old",	"on road now",	"years",	"km",	"rating",	"condition",	"economy",	"top speed",	"hp",	"torque","current price"]], diag_kind='kde')
#plt.show()

tensor_data = tf.cast(data, dtype=tf.float16)
tensor_data_1 = tf.random.shuffle(tensor_data)
print(tensor_data[0:5])
print(tensor_data_1[:5])

X = tensor_data[:,3:-1]
print(X.shape)
Y = tensor_data[:-1]
y_1 = tf.expand_dims(Y, axis=-1)
print(Y.shape)

normalizer = Normalization()
normalizer.adapt(X)
print(normalizer(X))

model = tf.keras.Sequential([
    normalizer,
    Dense(1)
])
model.summary()

tf.keras.utils.plot_model(model, to_file = "model.png", show_shapes=True)

