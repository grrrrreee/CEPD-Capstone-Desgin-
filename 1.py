#!/usr/bin/env python
# coding: utf-8
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from random import randint
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', action='store_true', help="training")
parser.add_argument('--show', action='store_true', help="showing")
parser.add_argument('--epochs', metavar='E', type=int,
                    default=20000, help="number of epochs")
parser.add_argument('--infer', metavar='I', type=int,
                    default=1000, help="number of inferences")
parser.add_argument('--pati', metavar='P', type=int,
                    default=20, help="number of patience")
parser.add_argument('--nodes', metavar='N', type=int,
                    default=80, help="LSTM width")
parser.add_argument('--adam', action='store_true', help="Adam or RMSprop")

args = parser.parse_args()
print(args)


# 설정 변경 가능
# conda activate base
# python 1.py --train --epochs=2 --show
training = args.train
showing = args.show
NODES = args.nodes
N = args.infer
EPOCHS = args.epochs
PATIENCE = args.pati
OPTIMIZER = Adam() if args.adam else RMSprop()

data = pd.read_excel('final_version.xlsx', header=None)
data = data.to_numpy()[1:]

idx = 0
X, Y = [], []
while True:
    if idx+15 >= len(data):
        break
    X.append(data[idx:idx+15])
    Y.append(data[idx+15])
    idx += 1
X = np.array(X)
Y = np.array(Y)

# X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2)
# X_test, X_val, Y_test, Y_val = train_test_split(X_tmp, Y_tmp, test_size=0.5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = np.array(X_train)[:, :, [1, 3, 4]]
Y_train = np.array(Y_train)[:, -1].reshape(-1, 1)
X_test = np.array(X_test)[:, :, [1, 3, 4]]
Y_test = np.array(Y_test)[:, -1].reshape(-1, 1)

"""extend"""
X_nonzero = []
Y_nonzero = []
for i, y in enumerate(Y_train):
    if y[-1] != 0.0:
        X_nonzero.append(X_train[i])
        Y_nonzero.append(y)

X_train = np.concatenate((X_train, X_nonzero, X_nonzero, X_nonzero))
Y_train = np.concatenate((Y_train, Y_nonzero, Y_nonzero, Y_nonzero))


def build_model(input_shape):
    x = Input(shape=input_shape)

    # h = LSTM(20, return_sequences=True, activation='tanh', kernel_initializer='he_normal')(x)
    # h = LSTM(20, return_sequences=True, activation='tanh', kernel_initializer='he_normal')(h)
    # h = LSTM(20, activation='tanh', kernel_initializer='he_normal')(h)
    h = LSTM(NODES, activation='tanh',
             kernel_initializer='he_normal')(x)  # 128

    y = Dense(1, activation='relu')(h)
    return Model(inputs=x, outputs=y)


model = build_model((15, 3))
model.summary()

model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['mae'])

if training == True:

    es = EarlyStopping(monitor='val_mae', mode='min',
                       verbose=1, patience=PATIENCE)
    mc = ModelCheckpoint('best_model_'+str(EPOCHS)+'_' +
                         str(NODES)+'_'+('Adam' if args.adam else 'RMSprop')+'.h5', monitor='val_mae',
                         mode='min', save_best_only=True)

    hist = model.fit(
        X_train, Y_train,
        batch_size=128,
        validation_split=0.2,
        epochs=EPOCHS,  # 20000
        verbose=2,
        callbacks=[es, mc])

    model.save_weights('./trained_'+str(EPOCHS)+'_' +
                       str(NODES)+'_'+('Adam' if args.adam else 'RMSprop')+'.h5')

else:
    model.load_weights('./best_model_'+str(EPOCHS)+'_' +
                       str(NODES)+'_'+('Adam' if args.adam else 'RMSprop')+'.h5')


if training == True and showing == True:
    plt.plot(hist.history['loss'], 'y', label='train_loss')
    plt.plot(hist.history['val_loss'], 'r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper left')
    plt.show()
else:
    pass

loss, mae = model.evaluate(X_test, Y_test)
print(loss, mae)

samples = [randint(0, len(X_test)-1) for _ in range(N)]
pred = model.predict([X_test[samples]])
pred = pred.reshape(-1)
real = Y_test[samples]
real = real.reshape(-1)

res = np.abs(pred - real)
print(np.mean(res), np.max(res), np.min(res))