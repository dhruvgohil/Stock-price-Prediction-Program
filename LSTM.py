import logging, sys
# import tensorflow as tf
# if type(tf.contrib) != type(tf): tf.contrib._warning = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

f = open('/dev/null', 'w')
sys.stderr = f

# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# importing libraries
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
# %matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# reading the data
df = pd.read_csv('RELIANCE.csv')

# looking at the first five rows of the data
print(df.head())
print('\n Shape of the data:')
print(df.shape)

# setting the index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.xlabel("Years")
plt.ylabel("Close Price")
plt.plot(df['Close'], label='Close Price history')
plt.show()

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)

#plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(labels = ('train','valid','Predictions'),loc='upper left')
# plt.show()

#setting index as date values
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Open', 'Close'])

for i in range(0,len(data)):
    new_data['Open'][i] = data['Open'][i]
    new_data['Close'][i] = data['Close'][i]

#create features
# from fastai.structured import  add_datepart
# add_datepart(new_data, 'Date')
# new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

# new_data['mon_fri'] = 0
# for i in range(0,len(new_data)):
#     if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#         new_data['mon_fri'][i] = 1
#     else:
#         new_data['mon_fri'][i] = 0

#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.title("Moving Average")
plt.xlabel("Years/ Rows  of dataset")
plt.ylabel("Close Price")
plt.legend(labels = ('Training data','Valid data','Moving Average'),loc='upper left')
plt.show()

#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.title("Linear Regression")
plt.xlabel("Years/ Rows  of dataset")
plt.ylabel("Close Price")
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.legend(labels = ('Valid data','Predictions','Training data'),loc='upper left')
plt.show()

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

#for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.title("LSTM Prediction")
plt.xlabel("Years/ Rows  of dataset")
plt.ylabel("Close Price")
plt.legend(labels = ('Training data','Valid data','Predictions'),loc='upper left')
plt.show()





# # importing libraries
# import pandas as pd
# import numpy as np
# import fastbook
# fastbook.setup_book()

# #to plot within notebook
# import matplotlib.pyplot as plt
# # %matplotlib inline

# #setting figure size
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 20,10

# #for normalizing data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))


# # reading the data
# df = pd.read_csv('NSE-TATAGLOBAL11.csv')

# # looking at the first five rows of the data
# print(df.head())
# plt.show()
# print('\n Shape of the data:')
# print(df.shape)

# # setting the index as date
# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']

# #plot
# plt.figure(figsize=(16,8))
# plt.plot(df['Close'], label='Close Price history')
# plt.show()

# #creating dataframe with date and the target variable
# data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

# for i in range(0,len(data)):
#      new_data['Date'][i] = data['Date'][i]
#      new_data['Close'][i] = data['Close'][i]

# # NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

# # splitting into train and validation
# train = new_data[:987]
# valid = new_data[987:]

# # shapes of training set
# print('\n Shape of training set:')
# print(train.shape)

# # shapes of validation set
# print('\n Shape of validation set:')
# print(valid.shape)

# # In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# # making predictions
# preds = []
# for i in range(0,valid.shape[0]):
#     a = train['Close'][len(train)-248+i:].sum() + sum(preds)
#     b = a/248
#     preds.append(b)

# # checking the results (RMSE value)
# rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
# print('\n RMSE value on validation set:')
# print(rms)

# #plot
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.show()
# #setting index as date values
# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']


# #create features
# from fastai.structured import  add_datepart
# add_datepart(new_data, 'Date')
# new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

# new_data['mon_fri'] = 0
# for i in range(0,len(new_data)):
#     if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#         new_data['mon_fri'][i] = 1
#     else:
#         new_data['mon_fri'][i] = 0

# #split into train and validation
# train = new_data[:987]
# valid = new_data[987:]

# x_train = train.drop('Close', axis=1)
# y_train = train['Close']
# x_valid = valid.drop('Close', axis=1)
# y_valid = valid['Close']

# print("Output : {} {}".format(x_train,y_train))
# #implement linear regression
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train,y_train)

# #make predictions and find the rmse
# preds = model.predict(x_valid)
# rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
# print('\n RMSE value on validation set:')
# print(rms)

# # #plot
# # valid['Predictions'] = 0
# # valid['Predictions'] = preds

# valid.index = new_data[987:].index
# train.index = new_data[:987].index

# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])

# plt.show()