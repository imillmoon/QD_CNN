import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
hdf = h5py.File('/disk/hyunwook/skku/labeled.h5', 'r')
data = hdf["data"]

x = data['feature']  # x : (10032, 224, 224) matrix
y = []               # y : (10032,1) matrix
for i in range(np.shape(data['label'])[0]):
    if data['label'][i] == True:
        y.append([1])
    if data['label'][i] == False:
        y.append([0])

# list to torch tensor
x_data = torch.Tensor(10032,1,224,224)
x_data[:,0] = torch.Tensor(data['feature'][:])
y_data = torch.Tensor(y)


# divide data & shuffle
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # input = (n, 1, 224, 224)  # 첫번째 n은 training data 갯수에 따라 달라짐. 위의 test_size를 변경시킴으로 조절할 수 있음
        # Conv -> (n, 4, 224, 224)  # 두번째 숫자 16는 Conv2d의 두번째 숫자인 16에 의해 결정됨
        # Pool -> (n, 4, 112, 112)  # MaxPool2d의 kernel_size=2,stride=2여서 224 -> 112로 줄어듬
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # input = (n, 4, 112, 112)
        # Conv -> (n, 8, 112, 112)
        # Pool -> (n, 8, 56, 56)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 세번째층
        # input = (n, 8, 56, 56)
        # Conv -> (n, 16, 56, 56)
        # Pool -> (n, 16, 28, 28)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 네번째층
        # input = (n, 16, 28, 28)
        # Conv -> (n, 32, 28, 28)
        # Pool -> (n, 32, 14, 14)
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 32x14x14 inputs -> 1 outputs
        self.fc = nn.Linear(32*14*14, 1, bias=True)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

        # 전결합층 한정으로 가중치 초기화
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

learning_rate = 0.001
training_epochs = 500
batch_size = 1000

model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
# criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

kfold = KFold(n_splits=6, shuffle = True)

Accuracy=[]
AUC=[]

for train_index, test_index in kfold.split(x_data):
    random.shuffle(train_index)
    random.shuffle(test_index)

    x_train = x_data[train_index]
    x_test = x_data[test_index]
    y_train = y_data[train_index]
    y_test = y_data[test_index]

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    total_batch = len(dataloader)

    for epoch in range(training_epochs):
        avg_loss = 0
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_predicted = model(X)

            Y_predicted = torch.reshape(Y_predicted, (1,Y_predicted.shape[0]))    # CrossEntropyLoss를 적용하기 위한 reshape. CrossEntropyLoss는 (1,n) matrix만 계산 가능.
            Y = torch.reshape(Y, (1,Y.shape[0]))

            loss = criterion(Y_predicted, Y)
            # print(loss)
            loss.backward()
            optimizer.step()
            avg_loss += loss/total_batch
        # print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))

    with torch.no_grad():
        X_test = x_test.to(device)
        Y_test = y_test.to(device)

        prediction = model(X_test)
        correct_prediction=[]

        for i in range(prediction.shape[0]):
            if prediction[i, 0] > 0.9 and Y_test[i] == 1:
                correct_prediction.append(1)
            elif prediction[i, 0] < 0.1 and Y_test[i] == 0:
                correct_prediction.append(1)
            else:
                correct_prediction.append(0)
        accuracy = sum(correct_prediction)/len(correct_prediction)
        # for i in range(prediction.shape[0]):
            # print('Y_prediction: {:.5f}, Y = {}'.format(float(prediction[i]), int(y_test[i])))
    auroc = torchmetrics.AUROC(task="binary")
    Accuracy.append(accuracy)
    AUC.append(float(auroc(prediction,Y_test)))
print('accuracy : ', Accuracy)
print('AUC      : ', AUC)

mean_Accuracy = sum(Accuracy)/len(Accuracy)
mean_AUC = sum(AUC)/len(AUC)

print('average_accuracy : ', mean_Accuracy)
print('average_AUC      : ', mean_AUC)