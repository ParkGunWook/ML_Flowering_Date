# ML_Flowering_Date
Using Azure Notebook_microsoft

Code Refer by https://www.tensorflow.org/tutorials/keras/basic_regression

## 1)	Import Library For Machine Leaning
```py
#Pandas library import for Excel Using
import pandas as pd 
#Numpy module import
import numpy as np 
import keras 
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense
```

## 2)	Data(Excel) Import with pandas, print and check Data(Excel) 
```py
Excelfile = pd.ExcelFile('dataset_for_Flowering_3.xlsx')
df = pd.read_excel(Excelfile)
df
```

## 3)	Set Train data and test data and separate ‘result data’ to another space
Print and Check train and test Data(2-Dimensional array)
```py
train_data = df.loc[1:104]
test_data = df.loc[105:112]

train_y = train_data['365percent']
train_data = train_data.drop('365percent', axis=1)

test_y = test_data['365percent']
test_data = test_data.drop('365percent', axis=1)

## 112 examples, 5 features
print("Training set: {}".format(train_data.shape))  
print("Testing set:  {}".format(test_data.shape))   

# Display sample features, notice the different scales
print(train_data.head(2)) 
```
## 4)	Naming for data
```py
column_names = ['Tem2', 'Tem3', 'Rain2', 'Rain3', 'Sun3']
df = pd.DataFrame(train_data, columns=column_names)
df.head(8)
```
## 5)	Build Model(2-layer-64-32-1)
```py
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer= 'adam',
                metrics=['mae'])
  return model
  
model = build_model()
model.summary()
```
## 6)	Display training progress with dot, Store data for progress working
```py
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

# Store training stats
history = model.fit(train_data, train_y, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
```
## 7)	Draw val loss with progress
```py
def plot_history(history):
  f, ax = plt.subplots(figsize=(24,8))
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([10, 30])

plot_history(history)
```
## 8)	Draw and print test result
```py
def plot_result():
  f, ax = plt.subplots(figsize=(24,8))
  plt.xlabel('Index')
  plt.ylabel('365Percent')
  plt.plot(test_data.index, model.predict(test_data),
           label='test_data predict')
  plt.plot(test_data.index, test_y,
           label = 'real test_data y')
  plt.legend()
  plt.ylim([70, 130])

plot_result()

print('365percent')
(test_data.index, model.predict(test_data))
```
