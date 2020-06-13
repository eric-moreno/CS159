import numpy as np
import argparse
import operator
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser =argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--outdir', type=str, default='training_dir', metavar='N', help='directory where outputs are saved ')
args = parser.parse_args()

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


loss_epoch_optimizer_RNN = np.load('training_RNN_1layer_final_2/loss_epoch_optimizer_val.npy')
loss_epoch_val_RNN = np.load('training_RNN_1layer_final_2/loss_epoch_val.npy')
accuracy_epoch_val_RNN = np.load('training_RNN_1layer_final_2/accuracy_epoch_val.npy')
loss_epoch_optimizer_GRU = np.load('training_GRU_1layer_final_2/loss_epoch_optimizer_val.npy')
loss_epoch_val_GRU = np.load('training_GRU_1layer_final_2/loss_epoch_val.npy')
accuracy_epoch_val_GRU = np.load('training_GRU_1layer_final_2/accuracy_epoch_val.npy')
loss_epoch_optimizer_LSTM = np.load('training_LSTM_1layer_final_2/loss_epoch_optimizer_val.npy')
loss_epoch_val_LSTM = np.load('training_LSTM_1layer_final_2/loss_epoch_val.npy')
accuracy_epoch_val_LSTM= np.load('training_LSTM_1layer_final_2/accuracy_epoch_val.npy')

fig, ax = plt.subplots()
ax.plot(loss_epoch_optimizer_RNN[:150] , label = 'RNN')
ax.plot(loss_epoch_optimizer_GRU[:150] , label = 'GRU')
ax.plot(loss_epoch_optimizer_LSTM[:150] , label = 'LSTM')
ax.set_xlabel('Epoch')
ax.set_ylabel('Optimizer Loss (NLL)')
ax.legend()

fig.savefig(args.outdir + '/lossoptimizer_fig.jpg')

fig, ax = plt.subplots()
#ax.plot(loss_epoch_Optimizer_Train[3:], label='Optimizer Loss Training')
#ax.plot(loss_epoch_OptimizerTrain_val, label = 'Model Loss Validation')
ax.plot(loss_epoch_val_RNN[:150] , label = 'RNN')
ax.plot(loss_epoch_val_GRU[:150] , label = 'GRU')
ax.plot(loss_epoch_val_LSTM[:150] , label = 'LSTM')
ax.set_xlabel('Epoch')
ax.set_ylabel('Model Loss (NLL)')
ax.legend()

fig.savefig(args.outdir + '/lossmodel_fig.jpg')

fig, ax = plt.subplots()
#ax.plot(loss_epoch_Optimizer_Train[3:], label='Optimizer Loss Training')
#ax.plot(loss_epoch_OptimizerTrain_val, label = 'Model Loss Validation')
ax.plot(accuracy_epoch_val_RNN[:150] , label = 'RNN')
ax.plot(accuracy_epoch_val_GRU[:150] , label = 'GRU')
ax.plot(accuracy_epoch_val_LSTM[:150] , label = 'LSTM')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()

fig.savefig(args.outdir + '/accuracy_fig.jpg')
