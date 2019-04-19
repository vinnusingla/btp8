import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

imageName  = 'dynamicReliance.png'
inputDim = 3
steps = 5
epoch = 20

##################################################################################################

def processData(data,inputDim):
    X,Y = [],[]
    for i in range(len(data)-inputDim-1):
        X.append(data[i:(i+inputDim),0])
        Y.append(data[(i+inputDim),0])
    return np.array(X),np.array(Y)

# architecture of model
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
	model.add(Dense(1))

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
data = pd.read_csv('./niftydata/Reliance_industries_news.csv')
# data = data[data['Name']=='AAL']
cl = data.Close
cl = cl.fillna(cl.mean())

#Scale the data
scl = MinMaxScaler()
cl = cl.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)

# Shape data
X,y = processData(cl,inputDim)

#Split into test and train
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

#printing shape for reference
print('cl Shape: ', cl.shape)
print('XTrain Shape: ', X_train.shape)
print('XTest Shape: ', X_test.shape)
print('YTrain Shape: ', y_train.shape)
print('YTest Shape: ', y_test.shape)

# reshaping data to make predictions
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))

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

#plot results on test data
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Original Stock Price')
plt.plot(scl.inverse_transform(test_predict), label='Predicted Stock Price')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend(loc = 'upper right')
plt.savefig(imageName)
plt.show()

#predicting multiple steps in future
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Original Stock Price')

def give_prediction(init):
	prediction = []
	for j in range (0,steps):
		init = init.reshape(1,inputDim,1)
		pred = model.predict(init)
		init = init.flatten()
		init = np.append(init[1:],pred.flatten())
		prediction.append(pred[0][0])
	return prediction

predictions = []
repitition = 1

for i in range(0,X_test.shape[0],repitition):
	temp = X_test[i]
	prediction = give_prediction(temp)
	
	predictions.append(prediction[0])
	X = range(i,i+steps)
	Y = scl.inverse_transform(np.array(prediction).reshape(-1,1))
	plt.plot(X,Y)
	#data on which prediction is made can be assumed to be now available
	#and hence can be added to the training data
	X_train = np.append(X_train,X_test[i:i+repitition],axis=0)
	y_train = np.append(y_train,y_test[i:i+repitition])
	print X_train.shape, y_train.shape
	if((i)%(50) == 49):
		trainModel(X_train,y_train,model)

# TEST RMSE
print y_test.shape,np.array(predictions).shape
test_score = math.sqrt(mean_squared_error(y_test, np.array(predictions)))
print('Test RMSE: %.4f' % (test_score))


plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.savefig('multiple '+imageName)
plt.show()
