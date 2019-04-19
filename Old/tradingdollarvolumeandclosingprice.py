import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

inputDim = 3
features = 3
steps = 5
epoch = 20

##################################################################################################

def processData(data,inputDim):
    X,Y = [],[]
    for i in range(len(data)-inputDim):
        X.append(data[i:(i+inputDim),:])
        Y.append(data[(i+inputDim),:])
    return np.array(X),np.array(Y)

# architecture of model
def makeModel():
	# model = Sequential()
	# model.add(LSTM(10,return_sequences = True,input_shape=(inputDim,features)))
	# model.add(LSTM(units = 20,return_sequences = True))
	# model.add(Dropout(0.2))
	# model.add(LSTM(units = 10))
	# model.add(Dropout(0.2))
	# model.add(Dense(features, activation='relu'))

	# model = Sequential()
	# model.add(LSTM(128, input_shape=(inputDim, features), return_sequences = True))
	# model.add(LSTM(64, return_sequences = True))
	# model.add(LSTM(16))
	# model.add(Dense(features, activation='linear'))

	model = Sequential()
	model.add(LSTM(128,input_shape=(inputDim,features)))
	model.add(Dense(8))
	model.add(Dense(features))

	# regressor = Sequential()
	# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (inputDim, features)))
	# regressor.add(LSTM(units = 50, return_sequences = True))
	# regressor.add(Dropout(0.2))
	# regressor.add(LSTM(units = 50, return_sequences = True))
	# regressor.add(Dropout(0.2))
	# regressor.add(LSTM(units = 50))
	# regressor.add(Dropout(0.2))
	# regressor.add(Dense(units = features))
	# return regressor

	return model	

# train the model data is split into train and val and thin fitted on the lstm model
def trainModel(dataX,dataY,model):
	# split data into train and validation
	
	#way1
	train_idx = []
	test_idx = []
	for i in range(0,dataX.shape[0]):
		if(i%5 == 4):
			test_idx = test_idx + [i]
		else:
			train_idx =train_idx + [i]
	trainX = dataX[train_idx]
	trainY = dataY[train_idx]
	valX = dataX[test_idx]
	valY = dataY[test_idx]

	#way2
	# trainX = dataX[:int(dataX.shape[0]*.80)]
	# trainY = dataY[:int(dataY.shape[0]*.80)]
	# valX = dataX[int(dataX.shape[0]*.80):]
	# valY = dataY[int(dataY.shape[0]*.80):]

	#Fit model with history to check for overfitting
	history = model.fit(trainX,trainY,epochs=epoch,validation_data=(valX,valY),shuffle=False)
	return history

##################################################################################################

# read data 
# data = pd.read_csv('./input/all_stocks_5yr.csv')
data = pd.read_csv('./niftydata/SBIN.NS.csv')
print data.head()
# data = data[data['Name']=='URI']
print data.head()
data = data[['Close','Volume', 'USD']]
# data = data[['Volume','Close']]
data = data.fillna(data.mean())

print data.head()

#Scale the data
scl = MinMaxScaler(feature_range = (0, 1))
data = data.values.reshape(data.shape[0],features)
data = scl.fit_transform(data)
print data.shape
# print data[:5]
# data = scl.inverse_transform(data)
# print data[:5]

# Shape data
X,y = processData(data,inputDim)

#Split into test and train
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

#printing shape for reference
print('data Shape: ', data.shape)
print('XTrain Shape: ', X_train.shape)
print('XTest Shape: ', X_test.shape)
print('YTrain Shape: ', y_train.shape)
print('YTest Shape: ', y_test.shape)

# reshaping data to make predictions
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],features))
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],features))


#Build the model
model = makeModel()

#compile the model
model.compile(optimizer='adam',loss='mse')

# train the model
history = trainModel(X_train,y_train,model)

# plot of loss and val loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc = 'upper right')
plt.show()


#predicting one step in future 
test_predict = model.predict(X_test)
train_predict = model.predict(X_train)

# TRAINING RMSE
train_score = math.sqrt(mean_squared_error(y_train, train_predict))
print('Train RMSE: %.4f' % (train_score))

# TEST RMSE
test_score = math.sqrt(mean_squared_error(y_test, test_predict))
print('Test RMSE: %.4f' % (test_score))

#plot results on train data
plt.plot(scl.inverse_transform(y_train)[:,0], label='Original Stock Price(Train)')
plt.plot(scl.inverse_transform(train_predict)[:,0], label='Predicted Stock Price')
plt.legend(loc = 'upper right')
plt.show()

#plot results on test data
plt.plot(scl.inverse_transform(y_test)[:,0], label='Original Stock Price(Test)')
plt.plot(scl.inverse_transform(test_predict)[:,0], label='Predicted Stock Price')
plt.legend(loc = 'upper right')
plt.show()


#predicting multiple steps in future
plt.plot(scl.inverse_transform(y_test)[:,0], label='Original Stock Price')
plt.plot(scl.inverse_transform(test_predict)[:,0], label='Predicted Stock Price')
plt.legend(loc = 'upper right')


stocks=1 
money =0

inv_X_test = X_test[:,-1,:]
inv_X_test = scl.inverse_transform(inv_X_test)

inv_X_train = X_train[:,-1,:]
inv_X_train = scl.inverse_transform(inv_X_train)

inv_y_test = scl.inverse_transform(y_test)

for i in range(0,X_test.shape[0],1):
	temp = X_test[i]
	prediction = []
	for j in range (0,steps):
		temp = temp.reshape(1,inputDim,features)
		pred = model.predict(temp)
		pred = pred.reshape(1,1,features)
		temp = np.append(temp[:,1:],pred,axis=1)
		prediction.append(pred.reshape(1,features))
	prediction = np.array(prediction)
	prediction = prediction.reshape(prediction.shape[0],features)
	X = range(i,i+steps)
	Y = scl.inverse_transform(np.array(prediction))[:,0]
	plt.plot(X,Y)

	predicted_price = Y[-1]
	today_price = inv_X_test[i][0]

	print stocks, money, predicted_price > today_price

	if(predicted_price > today_price):
		#buy
		stocks = stocks + money/today_price
		money = 0
	else:
		#sell
		money = stocks*today_price + money
		stocks = 0 

	#data on which prediction is made can be assumed to be now available
	#and hence can be added to the training data
	X_train = np.append(X_train,X_test[i:i+steps],axis=0)
	y_train = np.append(y_train,y_test[i:i+steps],axis=0)
	# trainModel(X_train[-100:],y_train[-100:],model)
	if((i)%(50) == 49):
		trainModel(X_train,y_train,model)

print 'Profit - ',
print (stocks*inv_y_test[-1][0] + money)/inv_X_train[-1][0] , '%'
plt.show()


# for i in range(0,X_test.shape[0],steps):
# 	temp = X_test[i]
# 	prediction = []
# 	for j in range (0,steps):
# 		temp = temp.reshape(1,inputDim,features)
# 		pred = model.predict(temp)
# 		pred = pred.reshape(1,1,features)
# 		temp = np.append(temp[:,1:],pred,axis=1)
# 		prediction.append(pred.reshape(1,features))
# 	prediction = np.array(prediction)
# 	prediction = prediction.reshape(prediction.shape[0],features)
# 	X = range(i,i+steps)
# 	Y = scl.inverse_transform(np.array(prediction))[:,0]
# 	plt.plot(X,Y)
# 	#data on which prediction is made can be assumed to be now available
# 	#and hence can be added to the training data
# 	X_train = np.append(X_train,X_test[i:i+steps],axis=0)
# 	y_train = np.append(y_train,y_test[i:i+steps],axis=0)
# 	print X_train.shape, y_train.shape
# 	# trainModel(X_train[-100:],y_train[-100:],model)
# 	if((i/steps)%(5) == 4):
# 		trainModel(X_train,y_train,model)

# plt.show()
