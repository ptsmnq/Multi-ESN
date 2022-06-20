import numpy as np, pandas as pd, matplotlib.pyplot as plt
from model import EchoStateNetwork as ESN, functions
plt.rcParams['font.size'] = 15

dataframe = pd.read_csv('Chaos.csv', encoding='utf_8', usecols=[0], nrows=3000)
dataset = np.array(dataframe.astype('float'))

func = functions()
test_size, sample, sparse = 0.5, 3, 5
look_back = (sample-1) * sparse + 1

train = dataset[:int(len(dataset)*test_size), :]
test = dataset[int(len(dataset)*test_size)-look_back:, :]
trainX, trainY = func.create_dataset(train, look_back, sparse, sample)
testX, testY = func.create_dataset(test, look_back, sparse, sample)
print("\n")
print("*"*30)
print("Data Information")
print(f"train data : {train.shape}, test data : {test.shape}")
print(f"input shape : {trainX.shape[2]}, input dimension : {trainX.shape[1]}")
print(f"output shape : {trainY.shape[1]}")
print("*"*30)
print("\n")

model = ESN(units=300,
            SR=0.99,
            input_shape=trainX.shape[2],
            input_dims=trainX.shape[1],
            output_shape=1,
            W_in_scale=0.1,
            W_res_scale=1.0,
            W_res_density=0.05,
            W_fb_scale=0.05,
            leak_rate=1.0,
            alpha=1.0e-4,
            seed=0,
            feedback=False)

print("\n")
print("*"*30)
print("ESN Information")
print(f"Neuron : {model.units}")
print(f"Spectral Radius : {model.SR}")
print(f"W_in Scale : {model.W_in_scale}")
print(f"W_res Scale : {model.W_res_scale}")
print(f"W_res density : {model.W_res_density*100:.1f}%")
if model.feedback==True : print(f"W_fb Scale : {model.W_fb_scale}")
else : print(f"Leaking Rate : {model.leak_rate}")
print(f"L2 norm : {model.alpha}")
print("*"*30)
print("\n")

model.fit(trainX, trainY)
train_pred = model.predict(trainX)

# 1step prediction
#test_pred = model.predict(testX)

#freerun prediction
pred_range = len(testY)
freerun_data = test[:look_back, :]
model.reset_reservoir()
test_pred = model.freerun(freerun_data, sparse, pred_range=pred_range)

train_score = func.rmse(trainY[:,0], train_pred[:,0])
test_score = func.rmse(testY[:pred_range,0], test_pred[:,0])
print('training score : %.4f RMSE' %train_score)
print('test score : %.4f RMSE' %test_score)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.plot(dataframe.iloc[:,0], label='real data')
ax.plot(range(look_back, len(train)), train_pred[:,0], label='train pred')
ax.plot(range(len(train), len(train)+pred_range), test_pred[:,0], label='test pred')
ax.set_xlabel('timestep')
ax.set_ylabel('X')
ax.legend()
plt.show()
