import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

inputDim = 3
epoch = 10
steps = 5

# read data 
# data = pd.read_csv('./input/all_stocks_5yr.csv')
data = pd.read_csv('./niftydata/Reliance_industries_news.csv')
# data = data[data['Name']=='AAL']
cl = data.Close

scl = MinMaxScaler()
#Scale the data
cl = cl.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)

def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)

def makeModel():
	# model = Sequential()
	# model.add(LSTM(10,return_sequences = True,input_shape=(inputDim,1)))
	# model.add(Dropout(0.2))
	# model.add(LSTM(units = 20,return_sequences = True))
	# model.add(Dropout(0.2))
	# model.add(LSTM(units = 10))
	# model.add(Dropout(0.2))
	# model.add(Dense(1, activation='relu'))

	model = Sequential()
	model.add(LSTM(256,input_shape=(inputDim,1)))
	# model.add(Dense(4))
	model.add(Dense(1))

	return model	

def trainModel(dataX,dataY,model):
	# split data into train and validation
	trainX = dataX[int(dataX.shape[0]*.80):]
	trainY = dataY[int(dataY.shape[0]*.80):]
	valX = dataX[:int(dataX.shape[0]*.80)]
	valY = dataY[:int(dataY.shape[0]*.80)]
	

X,y = processData(cl,inputDim)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print('XTrain Shape: ', X_train.shape)
print('XTest Shape: ', X_test.shape)
print('YTrain Shape: ', y_train.shape)
print('YTest Shape: ', y_test.shape)


#Build the model
model = makeModel()
model.compile(optimizer='adam',loss='mse')

#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=epoch,validation_data=(X_test,y_test),shuffle=False)

# plot of loss and val loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc = 'upper right')
plt.show()

#predicting one step in future 
test_predict = model.predict(X_test)
print 'test_predict',test_predict
train_predict = model.predict(X_train)

# TRAINING RMSE
train_score = math.sqrt(mean_squared_error(y_train, train_predict))
print('Train RMSE: %.2f' % (train_score))

# TEST RMSE
test_score = math.sqrt(mean_squared_error(y_test, test_predict))
print('Test RMSE: %.2f' % (test_score))


#plot results on test data
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Original Stock Price')
plt.plot(scl.inverse_transform(test_predict), label='Predicted Stock Price')
plt.legend(loc = 'upper right')
plt.show()

#predicting multiple steps in future

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))

predictions = []

# for i in range(0,X_test.shape[0],steps):
# 	print i
# 	temp = X_test[i]
# 	prediction = []
# 	for j in range (0,steps):
# 		temp = temp.reshape(1,inputDim,1)
# 		pred = model.predict(temp)
# 		temp = temp.flatten()
# 		temp = np.append(temp[1:],pred.flatten())
# 		prediction.append(pred[0][0])
# 	X = range(i,i+steps)
# 	Y = scl.inverse_transform(np.array(prediction).reshape(-1,1))
# 	plt.plot(X,Y)


stocks=1 
money =0

for i in range(0,X_test.shape[0],1):	
	print i , stocks , money, 
	temp = X_test[i]
	prediction = []
	for j in range (0,steps):
		temp = temp.reshape(1,inputDim,1)
		pred = model.predict(temp)
		temp = temp.flatten()
		temp = np.append(temp[1:],pred.flatten())
		prediction.append(pred[0][0])
	print prediction[-1], X_test[i][-1][0], prediction[-1] < X_test[i][-1][0]
	if (i+steps < len(y_test)):
		if(prediction[-1]>X_test[i][-1][0]):
			#buy
			stocks = stocks + money/X_test[i][-1][0]
			money=0
		else:
			#sell
			money = money + stocks*X_test[i][-1][0]
			stocks=0
	X = range(i,i+steps)
	Y = scl.inverse_transform(np.array(prediction).reshape(-1,1))
	plt.plot(X,Y)
print 'Profit - ',
print (stocks*y_test[-1] + money)/X_test[-1][-1][0] , '%'
plt.show()
