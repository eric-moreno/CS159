from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np
from utils import preprocess_gradients
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D

class MetaOptimizerLSTM(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizerLSTM, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(2, hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(nn.LSTMCell(hidden_size, hidden_size))

            self.lstms[-1].bias_ih.data.fill_(0)
            self.lstms[-1].bias_hh.data.fill_(0)
            self.lstms[-1].bias_hh.data[10:20].fill_(1)


        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizerLSTM, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, keep_states=True, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, inputs):
        initial_size = inputs.size()
        x = inputs.view(-1, 1)

        # Gradients preprocessing
        p = 10
        eps = 1e-6
        indicator = (x.abs() > math.exp(-p)).float()
        x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
        x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

        x = torch.cat((x1, x2), 1)

        x = F.tanh(self.linear1(x))

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.linear2(x)
        x = x.view(*initial_size)
        return x

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = Variable(torch.cat(grads))

        # Meta update itself
        flat_params = flat_params + self(flat_grads)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.

class MetaOptimizerGRU(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizerGRU, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(2, hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(nn.GRUCell(hidden_size, hidden_size))

            self.lstms[-1].bias_ih.data.fill_(0)
            self.lstms[-1].bias_hh.data.fill_(0)
            self.lstms[-1].bias_hh.data[10:20].fill_(1)


        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizerGRU, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, keep_states=True, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, inputs):
        initial_size = inputs.size()
        x = inputs.view(-1, 1)

        # Gradients preprocessing
        p = 10
        eps = 1e-6
        indicator = (x.abs() > math.exp(-p)).float()
        x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
        x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

        x = torch.cat((x1, x2), 1)

        x = F.tanh(self.linear1(x))

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i] = self.lstms[i](x, self.hx[i])
            x = self.hx[i]

        x = self.linear2(x)
        x = x.view(*initial_size)
        return x
    
    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = Variable(torch.cat(grads))

        # Meta update itself
        flat_params = flat_params + self(flat_grads)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

class MetaOptimizerRNN(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizerRNN, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(2, hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(nn.RNNCell(hidden_size, hidden_size))

            self.lstms[-1].bias_ih.data.fill_(0)
            self.lstms[-1].bias_hh.data.fill_(0)
            self.lstms[-1].bias_hh.data[10:20].fill_(1)


        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizerRNN, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, keep_states=True, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, inputs):
        initial_size = inputs.size()
        x = inputs.view(-1, 1)

        # Gradients preprocessing
        p = 10
        eps = 1e-6
        indicator = (x.abs() > math.exp(-p)).float()
        x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
        x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

        x = torch.cat((x1, x2), 1)

        x = F.tanh(self.linear1(x))

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i] = self.lstms[i](x, self.hx[i])
            x = self.hx[i]

        x = self.linear2(x)
        x = x.view(*initial_size)
        return x
    
    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = Variable(torch.cat(grads))

        # Meta update itself
        flat_params = flat_params + self(flat_grads)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model
    
class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(3, hidden_size)
        self.ln1 = LayerNorm1D(hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(hidden_size, hidden_size))

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizer, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, x):
        # Gradients preprocessing
        print(np.shape(np.array(x.cpu())))
        x = F.tanh(self.ln1(self.linear1(x)))

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.linear2(x)
        return x.squeeze()
    
    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
        

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = preprocess_gradients(torch.cat(grads))
        
        del grads
        inputs = Variable(torch.cat((flat_grads, flat_params.data), 1))
        inputs = torch.cat((inputs, self.cx, self.hx), 1)
        self.cx, self.hx = self(inputs)
        
        # Meta update itself
        #flat_params = flat_params + self(inputs)
        flat_params = self.cx * flat_params - self.hx * Variable(flat_grads)
        flat_params = flat_params.view(-1)
        self.meta_model.set_flat_params(flat_params)
        del flat_params
        
        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model
    
    '''
    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

                                            
        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)

        self.hx = self.hx.expand(flat_params.size(0), 1)
        self.cx = self.cx.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss), 1))
        inputs = torch.cat((inputs, self.cx, self.hx), 1)
        self.cx, self.hx = self(inputs)

        # Meta update itself
        flat_params = self.cx * flat_params - self.hx * Variable(flat_grads)
        flat_params = flat_params.view(-1)
        
        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        
        del flat_params, inputs, loss, flat_grads
        
        return self.meta_model.model
    '''
class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        #self.meta_model.apply(weight_reset)
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda()
                self.i = self.i.cuda()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

                                            
        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)

        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss), 1))
        inputs = torch.cat((inputs, self.f, self.i), 1)
        self.f, self.i = self(inputs)

        # Meta update itself
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)
        flat_params = flat_params.view(-1)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        
        del flat_params, inputs, loss, flat_grads
        
        return self.meta_model.model

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            module._parameters['weight'] = Variable(
                module._parameters['weight'].data)
            module._parameters['bias'] = Variable(
                module._parameters['bias'].data)

    def get_flat_params(self):
        params = []

        for module in self.model.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.model.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'] = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape)
            module._parameters['bias'] = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

            offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
