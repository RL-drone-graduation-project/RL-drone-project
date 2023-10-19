import json
import torch
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class neural_Network(torch.nn.Module):
  def __init__(self):
    super(neural_Network, self).__init__()
    #self.lstm = torch.nn.LSTM(input_size=29, hidden_size=64, num_layers=2, batch_first=True)
    self.MLP1 = torch.nn.Linear(150, 300)
    self.MLP2 = torch.nn.Linear(300, 300)
    self.MLP3 = torch.nn.Linear(300, 300)
    self.MLP4 = torch.nn.Linear(300, 1)

  def forward(self, x):
    #x, _ = self.lstm(x)
    #x = x[:, -1, :]
    y = self.MLP1(x)
    y = torch.nn.functional.relu(y)
    y = self.MLP2(y)
    y = torch.nn.functional.relu(y)
    y = self.MLP3(y)
    y = torch.nn.functional.relu(y)
    y = self.MLP4(y)
    y = torch.nn.functional.sigmoid(y)
    return y


def my_accuracy(model, X, y, error):
    real_y = model(X)
    val_acc = (abs(real_y - y) <= error / 100).sum()
    val_total = y.shape[0]
    return val_acc / val_total * 100


with open('stack_10_px4.json') as f:
    json_data = json.load(f)

X = np.array(json_data['0']).reshape(-1, 290)
del_columns = []
for i in range(0,10):      #IMU가 아닌 부분은 빼주는 코드
    start = i * 29
    del_columns += list(range(start+6,start+20))
X = np.delete(X,del_columns,axis=1)
print(X.shape)

y = np.zeros((X.shape[0], 1))
temp = np.hstack((X, y))
np.random.shuffle(temp)
test = temp[:10000]
train = temp[10000:]
#, 60, 70, 80, 90, 100, 110, 120
for d in [10, 20, 30, 40, 50]:
    #print(np.array(json_data[str(d)]).shape)
    X = np.array(json_data[str(d)]).reshape(-1, 290)

    del_columns = []
    for i in range(0,10):      #IMU가 아닌 부분은 빼주는 코드
        start = i * 29
        del_columns += list(range(start+6,start+20))
    X = np.delete(X,del_columns,axis=1)

    y = d / 100 * np.ones((X.shape[0], 1))
    temp = np.hstack((X, y))
    np.random.shuffle(temp)
    test = np.vstack((test, temp[:10000]))
    train = np.vstack((train, temp[10000:]))

train_data = train.copy()
test_data = test.copy()
ori_test_X = torch.FloatTensor(test[:, :-1]).reshape(-1, 150)
ori_test_Y = torch.FloatTensor(test[:, -1]).reshape(-1, 1)

np.random.shuffle(train_data)
np.random.shuffle(test_data)
train_X = torch.FloatTensor(train_data[:, :-1]).reshape(-1, 150)
train_Y = torch.FloatTensor(train_data[:, -1]).reshape(-1, 1)
test_X = torch.FloatTensor(test_data[:, :-1]).reshape(-1, 150)
test_Y = torch.FloatTensor(test_data[:, -1]).reshape(-1, 1)
print('train_data')
print(f'x = {train_X.shape}, y = {train_Y.shape}')
print('test_data')
print(f'x = {test_X.shape}, y = {test_Y.shape}')

optimal_lr = 0.002

train_dataset = TensorDataset(train_X, train_Y)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(test_X, test_Y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = neural_Network()
loss_func = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=optimal_lr)

patience = 20
best_test_loss = 100
best_model = None
early_stopping_counter = 0

for epoch in range(1, 201):
    train_loss = 0
    train_total = 0
    test_loss = 0
    test_total = 0

    for i, data in enumerate(train_dataloader):
        x, target = data
        model.train()
        y = model(x)
        loss = loss_func(y, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss
        train_total += 1

    for i, data in enumerate(test_dataloader):
        x, target = data
        model.eval()
        with torch.no_grad():
            y = model(x)
            loss = loss_func(y, target)
            test_loss += loss
            test_total += 1

    print(f'epoch {epoch} - train_loss {train_loss / train_total}, test_loss {test_loss / test_total}')

    if test_loss / test_total <= best_test_loss:
        best_test_loss = test_loss / test_total
        best_model = model
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping after {epoch} epochs")
            break 

print(f'+-5% Accuracy of training dataset: {my_accuracy(best_model, train_X, train_Y, 5): .2f}%')
print(f'+-5% Accuracy of test dataset: {my_accuracy(best_model, test_X, test_Y, 5): .2f}%')

print(f'+-4% Accuracy of training dataset: {my_accuracy(best_model, train_X, train_Y, 4): .2f}%')
print(f'+-4% Accuracy of test dataset: {my_accuracy(best_model, test_X, test_Y, 4): .2f}%')

print(f'+-3% Accuracy of training dataset: {my_accuracy(best_model, train_X, train_Y, 3): .2f}%')
print(f'+-3% Accuracy of test dataset: {my_accuracy(best_model, test_X, test_Y, 3): .2f}%')

print(f'+-2% Accuracy of training dataset: {my_accuracy(best_model, train_X, train_Y, 2): .2f}%')
print(f'+-2% Accuracy of test dataset: {my_accuracy(best_model, test_X, test_Y, 2): .2f}%')

print(f'+-1% Accuracy of training dataset: {my_accuracy(best_model, train_X, train_Y, 1): .2f}%')
print(f'+-1% Accuracy of test dataset: {my_accuracy(best_model, test_X, test_Y, 1): .2f}%')

print("-"*20)


for i in range(6):
    print(f'Accuracy of test dataset in distortion {i*10}%: {my_accuracy(best_model, ori_test_X[i*10000:(i+1)*10000], ori_test_Y[i*10000:(i+1)*10000], 3): .2f}%')


'''
print(f'Accuracy of test dataset in distortion 0%: {my_accuracy(best_model, ori_test_X[:20000], ori_test_Y[:20000], 3): .2f}%')
print(f'Accuracy of test dataset in distortion 10%: {my_accuracy(best_model, ori_test_X[20000:40000], ori_test_Y[20000:40000], 3): .2f}%')
print(f'Accuracy of test dataset in distortion 20%: {my_accuracy(best_model, ori_test_X[40000:60000], ori_test_Y[40000:60000], 3): .2f}%')
print(f'Accuracy of test dataset in distortion 30%: {my_accuracy(best_model, ori_test_X[60000:80000], ori_test_Y[60000:80000], 3): .2f}%')
print(f'Accuracy of test dataset in distortion 40%: {my_accuracy(best_model, ori_test_X[80000:100000], ori_test_Y[80000:100000], 3): .2f}%')
print(f'Accuracy of test dataset in distortion 50%: {my_accuracy(best_model, ori_test_X[100000:120000], ori_test_Y[100000:120000], 3): .2f}%')
'''