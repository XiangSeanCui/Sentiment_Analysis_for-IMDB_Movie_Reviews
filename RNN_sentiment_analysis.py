#only fine-tune the last lstm layer and the fully-connected layer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import pandas as pd
import torch.backends.cudnn as cudnn
import argparse
import time
import os
import sys
import io
from RNN_model_3c import RNN_model

parser = argparse.ArgumentParser(description='PyTorch hw8 Training')
parser.add_argument('--no_of_hidden_units', '-a', default = 500, type=int)
parser.add_argument('--no_of_epochs', '-e', default=20, type=int, help='total epoch')
parser.add_argument('--sequence_length', '-l', default=100, type=int, help='sequence of length for train')
parser.add_argument('--sequence_length_test', '-t', default=400, type=int, help='sequence of length for test')
args = parser.parse_args()

print('no_of_hidden_units: %d | no_of_epochs: %d | sequence_length_train: %d | sequence_length_test: %d' %
        (args.no_of_hidden_units, args.no_of_epochs, args.sequence_length, args.sequence_length_test))

vocab_size = 8000

x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)
    line[line>vocab_size] = 0
    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1
x_test = []

with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = RNN_model(vocab_size,args.no_of_hidden_units)

language_model = torch.load('language.model')
model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())
model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())
model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())
model.cuda()

params = []
# for param in model.embedding.parameters():
#     params.append(param)
# for param in model.lstm1.parameters():
#     params.append(param)
# for param in model.bn_lstm1.parameters():
#     params.append(param)
# for param in model.lstm2.parameters():
#     params.append(param)
# for param in model.bn_lstm2.parameters():
#     params.append(param)
for param in model.lstm3.parameters():
    params.append(param)
for param in model.bn_lstm3.parameters():
    params.append(param)
for param in model.fc_output.parameters():
    params.append(param)

opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(params, lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(params, lr=LR, momentum=0.9)

batch_size = 200
L_Y_train = len(y_train)
L_Y_test = len(y_test)
train_loss = []
train_accu = []
test_accu = []

for epoch in range(args.no_of_epochs):

    # training
    model.train()
    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0
    time1 = time.time()
    I_permutation = np.random.permutation(L_Y_train)
    for i in range(0, L_Y_train, batch_size):
        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size,args.sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < args.sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-args.sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+args.sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]
        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()
        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        loss.backward()
        optimizer.step()   # update weights

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))
    if((epoch+2)%3)==0:
        # do testing loop
        # ## test
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,args.sequence_length_test),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if(sl < args.sequence_length_test):
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-args.sequence_length_test+1)
                    x_input[j,:] = x[start_index:(start_index+args.sequence_length_test)]
            y_input = y_test[I_permutation[i:i+batch_size]]
            data = Variable(torch.LongTensor(x_input)).cuda()
            target = Variable(torch.FloatTensor(y_input)).cuda()
            with torch.no_grad():
                loss, pred = model(data,target,train=False)
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_acc += acc
            epoch_loss += loss.data.item()
            epoch_counter += batch_size

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)


torch.save(model,'rnn_{}_{}.model'.format(args.no_of_epochs,args.sequence_length))
#data = [train_loss,train_accu,test_accu]
#data = np.asarray(data)
#np.save('data.npy',data)
a = np.array(train_accu)
b = np.array(test_accu)
c = np.array(train_loss)

df_train = pd.DataFrame({"train_accu" : a, "train_loss":c})
df_test = pd.DataFrame({"test_accu" : b})
df_train.to_csv('result_train_default_{}_{}.csv'.format(args.no_of_epochs,args.sequence_length), index=False)
df_test.to_csv('result_test_default_{}_{}.csv'.format(args.no_of_epochs,args.sequence_length), index=False)
