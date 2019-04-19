import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

inputDim = 3
features = 6
steps = 5
epoch = 20	

##################################################################################################

def processData(data,inputDim):
    X,Y = [],[]
    for i in range(len(data)-inputDim-1):
        X.append(data[i:(i+inputDim),0])
        Y.append(data[(i+inputDim),0])
    return np.array(X),np.array(Y)

def processData2(data,inputDim):
    X,Y = [],[]
    for i in range(len(data)-inputDim):
        X.append(data[i:(i+inputDim),:])
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
# architecture of model
def makeModel2():
	# model = Sequential()
	# model.add(LSTM(10,return_sequences = True,input_shape=(inputDim,1)))
	# model.add(Dropout(0.2))
	# model.add(LSTM(units = 20,return_sequences = True))
	# model.add(Dropout(0.2))
	# model.add(LSTM(units = 10))
	# model.add(Dropout(0.2))
	# model.add(Dense(1, activation='relu'))

	model = Sequential()
	model.add(LSTM(128,input_shape=(inputDim,features),return_sequences=True))
	model.add(LSTM(64,input_shape=(inputDim,features)))
	model.add(Dense(4))
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
data = pd.read_csv('./niftydata/ALL.SBIN.NS.csv')
columns = data.columns
data = data.fillna(data.mean())
data = data.values

#Scale the data
scl = {}
for i in range(0,features):
	scl[i] = MinMaxScaler()
	print data.shape , data[:,i].shape
	data[:,i:i+1] = scl[i].fit_transform(data[:,i].reshape(-1,1))

print data[:5]

# Shape data
X = {}
y = {}
for i in range(0,features):
	X[i],y[i] = processData(data[:,i].reshape(data.shape[0],1),inputDim)

#Split into test and train
X_train = {}
X_test = {}
y_train = {}
y_test = {}
for i in range(0,features):
	X_train[i],X_test[i] = X[i][:int(X[i].shape[0]*0.80)],X[i][int(X[i].shape[0]*0.80):]
	y_train[i],y_test[i] = y[i][:int(y[i].shape[0]*0.80)],y[i][int(y[i].shape[0]*0.80):]

#printing shape for reference
print('data Shape: ', data.shape)
print('XTrain Shape: ', X_train[0].shape)
print('XTest Shape: ', X_test[0].shape)
print('YTrain Shape: ', y_train[0].shape)
print('YTest Shape: ', y_test[0].shape)

# reshaping data to make predictions
for i in range(0,features):
	X_test[i] = X_test[i].reshape((X_test[i].shape[0],X_test[i].shape[1],1))
	X_train[i] = X_train[i].reshape((X_train[i].shape[0],X_train[i].shape[1],1))

#Build the model
model = {}
for i in range(1,features):
	model[i] = makeModel()
	#compile the model
	model[i].compile(optimizer='adam',loss='mse')
	model[i].load_weights('./models/model{}.h5'.format(str(i)))

for i in range(4,4):
	model[i] = makeModel()
	#compile the model
	model[i].compile(optimizer='adam',loss='mse')
	# train the model
	history = trainModel(X_train[i],y_train[i],model[i])
	print 'Results of column - ' + columns[i]
	
	# # plot of loss and val loss
	# plt.plot(history.history['loss'], label='loss')
	# plt.plot(history.history['val_loss'], label='val_loss')
	# plt.legend(loc = 'upper right')
	# plt.show()
	
	#predicting one step in future 
	test_predict = model[i].predict(X_test[i])
	train_predict = model[i].predict(X_train[i])

	# TRAINING RMSE
	train_score = math.sqrt(mean_squared_error(y_train[i], train_predict))
	print('Train RMSE: %.4f' % (train_score))

	# TEST RMSE
	test_score = math.sqrt(mean_squared_error(y_test[i], test_predict))
	print('Test RMSE: %.4f' % (test_score))

	# #plot results on test data
	# plt.plot(scl[i].inverse_transform(y_test[i].reshape(-1,1)), label='Original')
	# plt.plot(scl[i].inverse_transform(test_predict), label='Predicted')
	# plt.legend(loc = 'upper right')
	# plt.show()

	model[i].save_weights("models/model{}.h5".format(str(i)))
	print("Saved model to disk")

#main model
model[0] = makeModel2()
model[0].compile(optimizer='adam',loss='mse')
# Shape data
X,y = processData2(data,inputDim)
#Split into test and train
X_main_train,X_main_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_main_train,y_main_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
#printing shape for reference
print('data Shape: ', data.shape)
print('XTrain Shape: ', X_main_train.shape)
print('XTest Shape: ', X_main_test.shape)
print('YTrain Shape: ', y_main_train.shape)
print('YTest Shape: ', y_main_test.shape)
# reshaping data to make predictions
X_main_test = X_main_test.reshape((X_main_test.shape[0],X_main_test.shape[1],features))
X_main_train = X_main_train.reshape((X_main_train.shape[0],X_main_train.shape[1],features))

# train the model
history = trainModel(X_main_train,y_main_train,model[0])

# plot of loss and val loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc = 'upper right')
plt.show()


#predicting one step in future 
test_predict = model[0].predict(X_main_test)
train_predict = model[0].predict(X_main_train)

# TRAINING RMSE
train_score = math.sqrt(mean_squared_error(y_main_train, train_predict))
print('Train RMSE: %.4f' % (train_score))

# TEST RMSE
test_score = math.sqrt(mean_squared_error(y_main_test, test_predict))
print('Test RMSE: %.4f' % (test_score))

#plot results on train data
plt.plot(scl[0].inverse_transform(y_main_train.reshape(-1, 1)), label='Original Stock Price(Train)')
plt.plot(scl[0].inverse_transform(train_predict.reshape(-1, 1)), label='Predicted Stock Price')
plt.legend(loc = 'upper right')
plt.show()

#plot results on test data
plt.plot(scl[0].inverse_transform(y_main_test.reshape(-1, 1)), label='Original Stock Price(Test)')
plt.plot(scl[0].inverse_transform(test_predict.reshape(-1, 1)), label='Predicted Stock Price')
plt.legend(loc = 'upper right')
plt.show()



#predicting multiple steps in future
plt.plot(scl[0].inverse_transform(y_main_test.reshape(-1, 1)), label='Original Stock Price')

def give_prediction(init,idx):
	prediction = []
	for j in range (0,steps):
		init = init.reshape(1,inputDim,1)
		pred = model[idx].predict(init)
		init = init.flatten()
		init = np.append(init[1:],pred.flatten())
		prediction.append(pred[0][0])
	return prediction

predictions = []

for i in range(0,X_main_test.shape[0],steps):
	temp = X_main_test[i]
	prediction = []
	singelton_prediction = {}
	for k in range (1,features):
		singelton_prediction[k] = give_prediction(X_test[k][i],k)

	for j in range (0,steps):
		temp = temp.reshape(1,inputDim,features)
		pred = []
		pred.append(model[0].predict(temp)[0][0])
		for k in range (1,features):
			pred.append(singelton_prediction[k][j])
		pred = np.array(pred)
		pred = pred.reshape(1,1,features)
		temp = np.append(temp[:,1:],pred,axis=1)
		prediction.append(pred.reshape(1,features))
	prediction = np.array(prediction)
	prediction = prediction.reshape(prediction.shape[0],features)
	X = range(i,i+steps)
	Y = scl[0].inverse_transform(np.array(prediction))[:,0]
	plt.plot(X,Y)
	#data on which prediction is made can be assumed to be now available
	#and hence can be added to the training data
	X_main_train = np.append(X_main_train,X_main_test[i:i+steps],axis=0)
	y_main_train = np.append(y_main_train,y_main_test[i:i+steps],axis=0)
	# trainModel(X_train[-100:],y_train[-100:],model)
	if((i/steps)%(5) == 4):
		trainModel(X_main_train,y_main_train,model[0])

plt.show()
