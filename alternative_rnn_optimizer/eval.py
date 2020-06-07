import argparse
import operator
import sys
import os
import setGPU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import get_batch
from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer, MetaOptimizerLSTM, MetaOptimizerGRU, MetaOptimizerRNN
from model import Model, ConvModel
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--outdir', type=str, default='training_dir', metavar='N', 
                    help='directory where outputs are saved ')
parser.add_argument('--RNN', type=str, default='LSTM', metavar='N', 
                    help='type of RNN you want to run with')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--train_split', type=float, default=0.8, metavar='N',
                    help='fraction of data going to training')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
os.system('mkdir -p %s'%args.outdir)

def main():
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = Model()
    if args.cuda:
        meta_model.cuda()
    
    if args.RNN == 'Fast':
        meta_optimizer = FastMetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    elif args.RNN == 'LSTM':
        meta_optimizer = MetaOptimizerLSTM(MetaModel(meta_model), args.num_layers, args.hidden_size)
    elif args.RNN == 'GRU':
        meta_optimizer = MetaOptimizerGRU(MetaModel(meta_model), args.num_layers, args.hidden_size)
    elif args.RNN == 'RNN':
        meta_optimizer = MetaOptimizerRNN(MetaModel(meta_model), args.num_layers, args.hidden_size)
    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)    
    if args.cuda:
        meta_optimizer.cuda()
    meta_optimizer.load_state_dict(torch.load('%s/%s_best.pth'%(args.outdir,'meta_optimizer')))
    
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

    l_val_model_best = 99999
    l_val_meta_model_best = 99999
    loss_epoch = []
    accuracy_epoch = []
    
    for epoch in range(args.max_epoch):
        print("Epoch %s\n" % epoch)
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)

        loss_train_model = []
        loss_train_meta = []
        loss_val_model = []
        loss_val_meta = []
        correct = 0 
        incorrect = 0 
        
        updates = args.updates_per_epoch
        for i in tqdm(range(updates)):
            
            # Sample a new model
            model = Model()
            if args.cuda:
                model.cuda()
        
            x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            
            
            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)

            
            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(args.truncated_bptt_step):
                    x, y = next(train_iter)
                    if args.cuda:
                        x, y = x.cuda(), y.cuda()
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = F.nll_loss(f_x, y)
                    loss_train_model.append(loss.item())
                    model.zero_grad()
                    loss.backward()

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)
                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data
                    
                # Update the parameters of the meta optimizer

                loss_train_meta.append(loss_sum.item())
                loss_sum.backward()



        #for i in tqdm(range(int((1-args.train_split) * len(train_loader)))):
        for i in tqdm(range(int(len(train_iter) - args.updates_per_epoch*(1+args.optimizer_steps)))):
            x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            
            
            #meta_optimizer.reset_lstm(
            #        keep_states=k > 0, model=model, use_cuda=args.cuda)
            
            # Compute initial loss of the model
            f_x = meta_model(x)
        
            for output, index in zip(f_x.cpu().detach().numpy(), range(len(f_x.cpu().detach().numpy()))):
                if y[index] == output.argmax():
                    correct += 1
                else: 
                    incorrect += 1
                    
            loss_model = F.nll_loss(f_x, y)
            loss_val_model.append(loss_model.item())
            
            
            #meta_model = meta_optimizer.meta_update(model, loss.data)

            # Compute a loss for a step the meta optimizer
            #f_x = meta_model(x)
            #loss_meta = F.nll_loss(f_x, y)
            
            #loss_val_meta.append(loss_meta.item())
        
        l_val_model = np.mean(loss_val_model)
        #l_val_meta_model = np.mean(loss_val_meta)
        loss_epoch.append(l_val_model)
        accuracy_epoch.append(float(correct) / (correct + incorrect))
        torch.save(meta_model.state_dict(), '%s/%s_last.pth'%(args.outdir,'meta_model_test'))

        if l_val_model < l_val_model_best:
            print("new best model")
            l_val_model_best = l_val_model
            torch.save(model.state_dict(), '%s/%s_best.pth'%(args.outdir,'meta_model_test'))
            
        print '\nValidation Loss Model: '+ str(l_val_model)     
        print '\nValidation Accuracy: ' + str(float(correct) / (correct + incorrect))
    np.save('%s/loss_epoch.npy'%(args.outdir), np.array(loss_epoch))
    np.save('%s/accuracy_epoch.npy'%(args.outdir), np.array(accuracy_epoch))
if __name__ == "__main__":
    main()