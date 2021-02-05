#!/usr/bin/env python
# coding: utf-8

# In[9]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[10]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from pyDOE import lhs
import shutil

import argparse


# ## 1、基础网络结构

# 网络结构分为普通的和带残差的全连接前馈神经网络(Fully-Connected Feedforward Network)，选择其中一种来实现。

# ### DNN

# In[11]:


def activation(name):
    if name in ['tanh','TANH']:
        return nn.Tanh()
    elif name in ['relu', 'RELU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'SIGMOID']:
        return nn.Sigmoid()
    elif name in ['softplus', 'SOFTPLUS']:
        return nn.Softplus()
    else: 
        raise ValueError(f'unknown activation function: {name}')


# In[12]:


class DNN(nn.Module):
    """Deep Neural Network"""
    
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers, 
                 act_name='tanh', init_name=None):
        super().__init__()
        model = nn.Sequential()
        
        model.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=True))
        model.add_module('act0', activation(act_name))
        
        for i in range(1, hidden_layers):
            model.add_module(f'fc{i}', nn.Linear(dim_hidden, dim_hidden, bias=True))
            model.add_module(f'act{i}', activation(act_name))
            
        model.add_module(f'fc{hidden_layers}', nn.Linear(dim_hidden, dim_out, bias=True))
        
        self.model = model
        
        if init_name is not None:
            self.init_weight(init_name)
            
    def init_weight(self, name):
        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init =  nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')
    
        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)
                
    def forward(self, x):
        return self.model(x)

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params                


# ### ResDNN

# In[13]:


class ResBlock(nn.Module):
    """Residual Block """
    
    def __init__(self, dim_in, dim_out, dim_hidden, act_name='tanh'):
        super().__init__()
        
        assert(dim_in == dim_out)
        
        block = nn.Sequential()
        block.add_module('act0', activation(act_name))
        block.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=True))
        block.add_module('act1', activation(act_name))
        block.add_module('fc1', nn.Linear(dim_hidden, dim_out, bias=True))
        self.block = block
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        return identity + out


# In[ ]:


class ResDNN(nn.Module):
    """Residual Deep Neural Network """
    
    def __init__(self, dim_in, dim_out, dim_hidden, res_blocks, act_name='tanh', init_name='kaiming_normal'):
        super().__init__()
        
        model = nn.Sequential()
        model.add_module('fc_first', nn.Linear(dim_in, dim_hidden, bias=True))
        
        for i in range(res_blocks):
            res_block = ResBlock(dim_hidden, dim_hidden, dim_hidden, act_name=act_name)
            model.add_module(f'res_block{i+1}', res_block)
            
        model.add_module('act_last', activation(act_name))
        model.add_module('fc_last', nn.Linear(dim_hidden, dim_out, bias=True))
        
        self.model = model
        
        if init_name is not None:
            self.init_weight(init_name)
        
    def init_weight(self, name):
        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init =  nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')
    
        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def forward(self, x):
        return self.model(x)         
    
    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params


# ## 2、Burgers方程

# 考虑一维Burgers方程：
# $$
# \left\{
# \begin{array}{rl}
# u_t + uu_x - \frac{0.01}\pi u_{xx} = 0, & x \in[-1, 1], ~~ t \in [0, 1]\\
# u(0, x) = - \sin(\pi x), & \\
# u(t,-1) = u(t, 1) = 0.
# \end{array}
# \right.
# $$

# ### 2.1、问题描述

# In[6]:


class Problem_Burgers(object):
    """ Description of Burgers Equation"""
    def __init__(self,domain=(0,1,-1,1)):
        self.domain=domain
        
    def __repr__(self):
        return f'{self.__doc__}'
    
    def iv(self,x):
        iv=-np.sin(np.pi*x[:,[1]])
        return iv
    
    def bv(self,x):
        bv=np.zeros_like(x[:,[0]])
        return bv
    
    def epilson(self):
        epilson=0.01/np.pi
        return epilson


# In[7]:


problem=Problem_Burgers()


# ### 2.2、数据集生成

# In[8]:


class Trainset_Burgers(object):
    def __init__(self,problem,*args,**kwargs):
        self.problem=problem
        self.domain=problem.domain
        self.args=args
        self.method=kwargs['method']
    
    def __call__(self,plot=False,verbose=None):
        if self.method=='uniform':
            n_t,n_x=self.args[0],self.args[1]
            p,p_bc,p_ic=self._uniform_sample(n_t,n_x)
        elif self.method=='lhs':
            n,n_bc,n_ic=self.args[0],self.args[1],self.args[2]
            p,p_bc,p_ic=self._lhs_sample(n,n_bc,n_ic)
            
        bv=self.problem.bv(p_bc)
        iv=self.problem.iv(p_ic)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(p[:, 0], p[:, 1], facecolor='r', s=10)
            ax.scatter(p_bc[:, 0], p_bc[:, 1], facecolor='b', s=10)
            ax.scatter(p_ic[:, 0], p_ic[:, 1], facecolor='g', s=10)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-1.01, 1.01)
            ax.set_aspect('equal')
            plt.show()
        
        if verbose=='tensor':
            p=torch.from_numpy(p).float()
            p_bc=torch.from_numpy(p_bc).float()
            p_ic=torch.from_numpy(p_ic).float()
            bv=torch.from_numpy(bv).float()
            iv=torch.from_numpy(iv).float()
            return p,p_bc,p_ic,bv,iv
        return p,p_bc,p_ic,bv,iv
    def _uniform_sample(self,n_t,n_x):
        t_min,t_max,x_min,x_max=self.domain
        t = np.linspace(t_min, t_max, n_t)
        x = np.linspace(x_min, x_max, n_x)
        t,x = np.meshgrid(t,x)
        tx=np.hstack((t.reshape(t.size,-1),x.reshape(x.size,-1)))
        
        mask_ic=(tx[:,0]-t_min)==0
        mask_bc=(tx[:,1]-x_min)*(x_max-tx[:,1])==0
        p_ic=tx[mask_ic]
        p_bc=tx[mask_bc]
        p=tx[np.logical_not(mask_ic,mask_bc)]
        return p,p_bc,p_ic
    
    def _lhs_sample(self,n,n_bc,n_ic):
        t_min,t_max,x_min,x_max=self.domain
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_max])
        p = lb + (ub - lb) * lhs(2, n)
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_min])
        p_bc = lb + (ub - lb) * lhs(2, n_bc//2)
        
        lb = np.array([t_min, x_max])
        ub = np.array([t_max, x_max])
        temp = lb + (ub - lb) * lhs(2, n_bc//2)
        p_bc = np.vstack((p_bc, temp))
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_min, x_max])
        p_ic = lb + (ub - lb) * lhs(2, n_ic)

        
        return p,p_bc,p_ic


# In[9]:


class Testset_Burgers(object):
    """Dataset on a square domain"""
    def __init__(self,problem,*args):
        self.problem = problem
        self.domain = problem.domain
        self.args = args
        
    def __repr__(self):
        return f'{self.__doc__}'
    
    def __call__(self,verbose=None):
        n_t,n_x=self.args[0],self.args[1]
        p,t,x=self._uniform_sample(n_t,n_x)
        if verbose=='tensor':
            p=torch.from_numpy(p).float()
            t=torch.from_numpy(t).float()
            x=torch.from_numpy(x).float()
            return p,t,x
        return p,t,x
        
    def _uniform_sample(self,n_t,n_x):
        t_min,t_max,x_min,x_max=self.domain
        t = np.linspace(t_min, t_max, n_t)
        x = np.linspace(x_min, x_max, n_x)
        t,x = np.meshgrid(t,x)
        p=np.hstack((t.reshape(t.size,-1),x.reshape(x.size,-1)))
        return p,t,x


# In[10]:


trainset = Trainset_Burgers(problem, 40, 40, method='uniform')
p,p_bc,p_ic,bv,iv=trainset(plot=True)
print(p.shape,p_bc.shape,p_ic.shape)


# ### 2.3、网络结构

# In[11]:


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

class PINN_Burgers(DNN):
    def __init__(self,dim_in,dim_out,dim_hidden, hidden_layers,
                act_name='sigmoid',init_name='xavier_normal'):
        super().__init__(dim_in,dim_out,dim_hidden,hidden_layers,
                        act_name=act_name, init_name=init_name)
    def forward(self,problem,p,p_bc=None,p_ic=None):
        p.requires_grad_(True)
        
        u=super().forward(p)
        if p_bc is None:
            return u
        grad_u = grad(u, p)[0]        
        u_t = grad_u[:, [0]]
        u_x = grad_u[:, [1]]
        u_xx = grad(u_x, p)[0][:, [1]]
        
        p.detach_()
        
        epilson=problem.epilson()
        f=u_t+u*u_x-epilson*u_xx
        
        bv_bar=super().forward(p_bc)
        iv_bar=super().forward(p_ic)
        return f,bv_bar,iv_bar


# In[12]:


class ResPINN_Burgers(ResDNN):
    def __init__(self,dim_in,dim_out,dim_hidden, hidden_layers,
                act_name='sigmoid',init_name='xavier_normal'):
        super().__init__(dim_in,dim_out,dim_hidden,res_blocks,
                        act_name=act_name, init_name=init_name)
    def forward(self,problem,p,p_bc=None,p_ic=None):
        p.requires_grad_(True)
        
        u=super().forward(p)
        if p_bc is None:
            return u
        grad_u = grad(u, p)[0]        
        u_t = grad_u[:, [0]]
        u_x = grad_u[:, [1]]
        u_xx = grad(u_x, p)[0][:, [1]]
        
        p.detach_()
        epilson=problem.epilson()
        f=u_t+ u*u_x-epilson*u_xx
        
        bv_bar=super().forward(p_bc)
        iv_bar=super().forward(p_ic)
        return f,bv_bar,iv_bar


# In[13]:


model = PINN_Burgers(2, 1, 10, 8)
print(model.model_size())


# ### 2.4、Options

# In[14]:


class Options_Burgers(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA or not')
        parser.add_argument('--dim_hidden', type=int, default=10, help='neurons in hidden layers')
        parser.add_argument('--hidden_layers', type=int, default=4, help='number of hidden layers')
        parser.add_argument('--res_blocks', type=int, default=4, help='number of residual blocks')
        parser.add_argument('--lam', type=float, default=1, help='weight in loss function')
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--epochs_Adam', type=int, default=1000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=200, help='epochs for LBFGS optimizer')
        parser.add_argument('--step_size', type=int, default=2000, help='step size in lr_scheduler for Adam optimizer')
        parser.add_argument('--gamma', type=float, default=0.7, help='gamma in lr_scheduler for Adam optimizer')
        parser.add_argument('--resume', type=bool, default=False, help='resume or not')
        parser.add_argument('--sample_method', type=str, default='lhs', help='sample method')
        parser.add_argument('--n_t', type=int, default=100, help='sample points in x-direction for uniform sample')
        parser.add_argument('--n_x', type=int, default=100, help='sample points in y-direction for uniform sample')
        parser.add_argument('--n', type=int, default=10000, help='sample points in domain for lhs sample')
        parser.add_argument('--n_bc', type=int, default=400, help='sample points on the boundary for lhs sample')
        parser.add_argument('--n_ic', type=int, default=400, help='sample points on the initial for lhs sample')
        
        self.parser = parser
    def parse(self):
        arg = self.parser.parse_args(args=[])
        arg.cuda = not arg.no_cuda and torch.cuda.is_available()
        arg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return arg


# In[15]:


args=Options_Burgers().parse()
print(args)


# ### 2.5、训练过程

# In[16]:


def save_model(state,is_best=None,save_dir=None):
    last_model = os.path.join(save_dir, 'last_model.pth.tar')
    torch.save(state, last_model)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(last_model, best_model)
        
class Trainer_Burgers(object):
    def __init__(self,args):
        self.device=args.device
        self.problem=args.problem
        
        self.lam=args.lam
        self.criterion=nn.MSELoss()
        
        self.model = args.model
        self.model_name = self.model.__class__.__name__
        self.model_path = self._model_path()
        
        self.epochs_Adam = args.epochs_Adam
        self.epochs_LBFGS = args.epochs_LBFGS
        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer_LBFGS = optim.LBFGS(self.model.parameters(), 
                                           max_iter=20, 
                                           tolerance_grad=1.e-8,
                                           tolerance_change=1.e-12)
        self.lr_scheduler = StepLR(self.optimizer_Adam, 
                                   step_size=args.step_size, 
                                   gamma=args.gamma)
        
        self.model.to(self.device)
        self.model.zero_grad()
        
        self.p,     self.p_bc,    self.p_ic, self.bv,     self.iv     = args.trainset(verbose='tensor')
        self.f=torch.from_numpy(np.zeros_like(self.p[:,0])).float()
        self.g=torch.cat((self.bv,self.iv))
        
    def _model_path(self):
        """Path to save the model"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        path = os.path.join('checkpoints', self.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
    
        return path
    
    def train(self):
        best_loss=1.e10
        for epoch in range(self.epochs_Adam):
            loss, loss1, loss2 = self.train_Adam()
            if (epoch + 1) % 100 == 0:
                self.infos_Adam(epoch+1, loss, loss1, loss2)
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                best_loss = valid_loss if is_best else best_loss                
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
            
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            loss, loss1, loss2 = self.train_LBFGS()
            if (epoch + 1) % 20 == 0:
                self.infos_LBFGS(epoch+1, loss, loss1, loss2)
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                best_loss = valid_loss if is_best else best_loss                
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
    def train_Adam(self):
        self.optimizer_Adam.zero_grad()
        
        f_pred,bv_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        g_pred=torch.cat((bv_pred,iv_pred))
        loss1=self.criterion(f_pred,self.f)
        loss2=self.criterion(g_pred,self.g)
        loss=loss1+self.lam*loss2
        
        loss.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()
        
        return loss.item(),loss1.item(),loss2.item()
    
    def infos_Adam(self,epoch,loss,loss1,loss2):
        infos = 'Adam  ' +             f'Epoch #{epoch:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.4e} = {loss1:.4e} + {self.lam} * {loss2:.4e} ' +             f'lr: {self.lr_scheduler.get_lr()[0]:.2e} '
        print(infos)
        
    def train_LBFGS(self):
        
        f_pred,bv_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        g_pred=torch.cat((bv_pred,iv_pred))
        loss1=self.criterion(f_pred,self.f)
        loss2=self.criterion(g_pred,self.g)
        
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()
            f_pred,bv_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
            g_pred=torch.cat((bv_pred,iv_pred))
            loss1=self.criterion(f_pred,self.f)
            loss2=self.criterion(g_pred,self.g)
            loss = loss1 + self.lam * loss2
            
            if loss.requires_grad:
                loss.backward()
            return loss
        self.optimizer_LBFGS.step(closure)
        loss=closure()
        
        return loss.item(), loss1.item(), loss2.item()
    
    def infos_LBFGS(self, epoch, loss, loss1, loss2):
        infos = 'LBFGS ' +             f'Epoch #{epoch:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.2e} = {loss1:.2e} + {self.lam:d} * {loss2:.2e} '
        print(infos)
        
    def validate(self, epoch):
        self.model.eval()
        f_pred,bv_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        g_pred=torch.cat((bv_pred,iv_pred))
        loss1=self.criterion(f_pred,self.f)
        loss2=self.criterion(g_pred,self.g)
        loss = loss1 + self.lam * loss2
        infos = 'Valid ' +             f'Epoch #{epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.4e} '
        print(infos)
        self.model.train()
        return loss.item()


# In[17]:


Problem=Problem_Burgers()
args=Options_Burgers().parse()
args.problem=Problem_Burgers()
args.model = PINN_Burgers(dim_in=2,
                  dim_out=1,
                  dim_hidden=args.dim_hidden,
                  hidden_layers=args.hidden_layers,
                  act_name='sigmoid')
if args.sample_method == 'uniform':
    args.trainset = Trainset_Burgers(args.problem, args.n_t, args.n_x, method='uniform')
elif args.sample_method == 'lhs':
    args.trainset = Trainset_Burgers(args.problem, args.n, args.n_bc,args.n_ic, method='lhs')
    
trainer_Burgers = Trainer_Burgers(args)
trainer_Burgers.train()


# ### 2.6、测试过程

# In[40]:


class Tester_Burgers(object):
    def __init__(self,args):
        self.device  = args.device
        self.problem = args.problem
        self.criterion = nn.MSELoss()
        self.model = args.model
        model_name = self.model.__class__.__name__
        model_path = os.path.join('checkpoints',
                                  model_name,
                                  'best_model.pth.tar')
        best_model = torch.load(model_path)
        self.model.load_state_dict(best_model['state_dict'])        
        self.model.to(self.device)
        
        self.p,self.t,self.x=args.testset(verbose='tensor')
    def predict(self):
        self.model.eval()
        u_pred=self.model(self.problem,self.p)
        u_pred=u_pred.detach().cpu().numpy()
        u_pred=u_pred.reshape(self.t.shape)
        
        plt.figure(figsize=(10,3),frameon=False)
        plt.contourf(self.t,self.x,u_pred,levels=1000,cmap='rainbow')
        plt.show


# In[41]:


args=Options_Burgers().parse()
args.problem=Problem_Burgers()

args.model = PINN_Burgers(dim_in=2,
                  dim_out=1,
                  dim_hidden=args.dim_hidden,
                  hidden_layers=args.hidden_layers,
                  act_name='sigmoid')
args.testset = Testset_Burgers(args.problem, 100, 100)  
tester = Tester_Burgers(args)
tester.predict()


# ## 3、Allen-Cahn方程

# 考虑带周期边界条件的Allen-Cahn方程：
# $$
# \left\{
# \begin{array}{rl}
# u_t -  0.0001 u_{xx} + 5u^3 - 5 u = 0, & x \in[-1, 1], ~~ t \in [0, 1]\\
# u(0, x) = x^2\cos(\pi x), & \\
# u(t,-1) = u(t, 1), & \\
# u_x(t, -1) = u_x(t, 1).
# \end{array}
# \right.
# $$
# 

# ### 3.1、问题描述

# In[14]:


class Problem_AC(object):
    def __init__(self,domain=(0,1,-1,1)):
        self.domain=domain
        
    def __repr__(self):
        return f'{self.__doc__}'
    
    def iv(self,p):
        x=p[:,[1]]
        return x**2*np.cos(np.pi*x)
    
    def bv(self,p):
        bv=np.zeros_like(p[:,[0]])
        return bv
    


# ### 3.2、数据集生成

# In[15]:


class Trainset_AC(object):
    def __init__(self,problem,*args,**kwargs):
        self.problem=problem
        self.domain=problem.domain
        self.args=args
        self.method=kwargs['method']
    
    def __call__(self,plot=False,verbose=None):
        if self.method=='uniform':
            n_t,n_x=self.args[0],self.args[1]
            p,p_bc,p_ic=self._uniform_sample(n_t,n_x)
        elif self.method=='lhs':
            n,n_bc,n_ic=self.args[0],self.args[1],self.args[2]
            p,p_bc,p_ic=self._lhs_sample(n,n_bc,n_ic)
            
        bv=self.problem.bv(p_bc)
        iv=self.problem.iv(p_ic)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(p[:, 0], p[:, 1], facecolor='r', s=10)
            ax.scatter(p_bc[:, 0], p_bc[:, 1], facecolor='b', s=10)
            ax.scatter(p_ic[:, 0], p_ic[:, 1], facecolor='g', s=10)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-1.01, 1.01)
            ax.set_aspect('equal')
            plt.show()
        
        if verbose=='tensor':
            p=torch.from_numpy(p).float()
            p_bc=torch.from_numpy(p_bc).float()
            p_ic=torch.from_numpy(p_ic).float()
            bv=torch.from_numpy(bv).float()
            iv=torch.from_numpy(iv).float()
            return p,p_bc,p_ic,bv,iv
        return p,p_bc,p_ic,bv,iv
    def _uniform_sample(self,n_t,n_x):
        t_min,t_max,x_min,x_max=self.domain
        t = np.linspace(t_min, t_max, n_t)
        x = np.linspace(x_min, x_max, n_x)
        t,x = np.meshgrid(t,x)
        tx=np.hstack((t.reshape(t.size,-1),x.reshape(x.size,-1)))
        
        mask_ic=(tx[:,0]-t_min)==0
        mask_bc=(tx[:,1]-x_min)*(x_max-tx[:,1])==0
        p_ic=tx[mask_ic]
        p_bc=tx[mask_bc]
        p=tx[np.logical_not(mask_ic,mask_bc)]
        return p,p_bc,p_ic
    
    def _lhs_sample(self,n,n_bc,n_ic):
        t_min,t_max,x_min,x_max=self.domain
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_max])
        p = lb + (ub - lb) * lhs(2, n)
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_max, x_min])
        p_bc = lb + (ub - lb) * lhs(2, n_bc//2)
        
        lb = np.array([t_min, x_max])
        ub = np.array([t_max, x_max])
        temp = lb + (ub - lb) * lhs(2, n_bc//2)
        p_bc = np.vstack((p_bc, temp))
        
        lb = np.array([t_min, x_min])
        ub = np.array([t_min, x_max])
        p_ic = lb + (ub - lb) * lhs(2, n_ic)

        
        return p,p_bc,p_ic


# In[16]:


class Testset_AC(object):
    """Dataset on a square domain"""
    def __init__(self,problem,*args):
        self.problem = problem
        self.domain = problem.domain
        self.args = args
        
    def __repr__(self):
        return f'{self.__doc__}'
    
    def __call__(self,verbose=None):
        n_t,n_x=self.args[0],self.args[1]
        p,t,x=self._uniform_sample(n_t,n_x)
        if verbose=='tensor':
            p=torch.from_numpy(p).float()
            t=torch.from_numpy(t).float()
            x=torch.from_numpy(x).float()
            return p,t,x
        return p,t,x
        
    def _uniform_sample(self,n_t,n_x):
        t_min,t_max,x_min,x_max=self.domain
        t = np.linspace(t_min, t_max, n_t)
        x = np.linspace(x_min, x_max, n_x)
        t,x = np.meshgrid(t,x)
        p=np.hstack((t.reshape(t.size,-1),x.reshape(x.size,-1)))
        return p,t,x


# ### 3.3、网络结构

# In[17]:


def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

class PINN_AC(DNN):
    def __init__(self,dim_in,dim_out,dim_hidden, hidden_layers,
                act_name='sigmoid',init_name='xavier_normal'):
        super().__init__(dim_in,dim_out,dim_hidden,hidden_layers,
                        act_name=act_name, init_name=init_name)
    def forward(self,problem,p,p_bc=None,p_ic=None):
        p.requires_grad_(True)
        
        u=super().forward(p)
        if p_bc is None:
            return u
        grad_u = grad(u, p)[0]        
        u_t = grad_u[:, [0]]
        u_x = grad_u[:, [1]]
        u_xx = grad(u_x, p)[0][:, [1]]
        
        p.detach_()
        
        f=u_t-0.0001*u_xx+5*u**3-5*u
        
        p_bc.requires_grad_(True)
        p_bcop=p_bc
        p_bcop[:,1]=-p_bc[:,1]
        p_bcop.requires_grad_(True)
        u_b=super().forward(p_bc)
        u_bop=super().forward(p_bcop)
        u_b_x=grad(u_b,p_bc)[0][:,1]
        u_bop_x=grad(u_bop,p_bcop)[0][:,1]
        p_bc.detach_()
        p_bcop.detach_()
        
        bv_bar=u_b-u_bop
        bv_x_bar=u_b_x-u_bop_x
        iv_bar=super().forward(p_ic)
        return f,bv_bar,bv_x_bar,iv_bar


# In[25]:


class ResPINN_AC(ResDNN):
    def __init__(self,dim_in,dim_out,dim_hidden, hidden_layers,
                act_name='sigmoid',init_name='xavier_normal'):
        super().__init__(dim_in,dim_out,dim_hidden,res_blocks,
                        act_name=act_name, init_name=init_name)
    def forward(self,problem,p,p_bc=None,p_ic=None):
        p.requires_grad_(True)
        
        u=super().forward(p)
        if p_bc is None:
            return u
        grad_u = grad(u, p)[0]        
        u_t = grad_u[:, [0]]
        u_x = grad_u[:, [1]]
        u_xx = grad(u_x, p)[0][:, [1]]
        
        p.detach_()
        f=u_t-0.0001*u_xx+5*u**3-5*u
        
        p_bc.requires_grad_(True)
        p_bcop=p_bc
        p_bcop[:,1]=-p_bc[:,1]
        p_bcop.requires_grad_(True)
        u_b=super().forward(p_bc)
        u_bop=super().forward(p_bcop)
        u_b_x=grad(u_b,p_bc)[0][:,[1]]
        u_bop_x=grad(u_bop,p_bcop)[0][:,[1]]
        p_bc.detach_()
        p_bcop.detach_()
        
        bv_bar=u_b-u_bop
        bv_x_bar=u_b_x-u_bop_x
        iv_bar=super().forward(p_ic)
        return f,bv_bar,bv_x_bar,iv_bar


# ### 3.4、Options

# In[35]:


class Options_AC(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA or not')
        parser.add_argument('--dim_hidden', type=int, default=10, help='neurons in hidden layers')
        parser.add_argument('--hidden_layers', type=int, default=4, help='number of hidden layers')
        parser.add_argument('--res_blocks', type=int, default=4, help='number of residual blocks')
        parser.add_argument('--lam', type=float, default=1, help='weight in loss function')
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--epochs_Adam', type=int, default=1000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=200, help='epochs for LBFGS optimizer')
        parser.add_argument('--step_size', type=int, default=2000, help='step size in lr_scheduler for Adam optimizer')
        parser.add_argument('--gamma', type=float, default=0.7, help='gamma in lr_scheduler for Adam optimizer')
        parser.add_argument('--resume', type=bool, default=False, help='resume or not')
        parser.add_argument('--sample_method', type=str, default='lhs', help='sample method')
        parser.add_argument('--n_t', type=int, default=100, help='sample points in x-direction for uniform sample')
        parser.add_argument('--n_x', type=int, default=100, help='sample points in y-direction for uniform sample')
        parser.add_argument('--n', type=int, default=10000, help='sample points in domain for lhs sample')
        parser.add_argument('--n_bc', type=int, default=400, help='sample points on the boundary for lhs sample')
        parser.add_argument('--n_ic', type=int, default=800, help='sample points on the initial for lhs sample')
        
        self.parser = parser
    def parse(self):
        arg = self.parser.parse_args(args=[])
        arg.cuda = not arg.no_cuda and torch.cuda.is_available()
        arg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return arg


# ### 3.5、训练过程

# In[36]:


def save_modelAC(state,is_best=None,save_dir=None):
    last_model = os.path.join(save_dir, 'last_modelAC.pth.tar')
    torch.save(state, last_model)
    if is_best:
        best_model = os.path.join(save_dir, 'best_modelAC.pth.tar')
        shutil.copyfile(last_model, best_model)
        
class Trainer_AC(object):
    def __init__(self,args):
        self.device=args.device
        self.problem=args.problem
        
        self.lam=args.lam
        self.criterion=nn.MSELoss()
        
        self.model = args.model
        self.model_name = self.model.__class__.__name__
        self.model_path = self._model_path()
        
        self.epochs_Adam = args.epochs_Adam
        self.epochs_LBFGS = args.epochs_LBFGS
        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer_LBFGS = optim.LBFGS(self.model.parameters(), 
                                           max_iter=20, 
                                           tolerance_grad=1.e-8,
                                           tolerance_change=1.e-12)
        self.lr_scheduler = StepLR(self.optimizer_Adam, 
                                   step_size=args.step_size, 
                                   gamma=args.gamma)
        
        self.model.to(self.device)
        self.model.zero_grad()
        
        self.p,     self.p_bc,    self.p_ic, self.bv,     self.iv     = args.trainset(verbose='tensor')
        self.f=torch.from_numpy(np.zeros_like(self.p[:,[0]])).float()
        
    def _model_path(self):
        """Path to save the model"""
        if not os.path.exists('checkpointsAC'):
            os.mkdir('checkpointsAC')

        path = os.path.join('checkpointsAC', self.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
    
        return path
    
    def train(self):
        best_loss=1.e10
        for epoch in range(self.epochs_Adam):
            loss, loss1, loss2 = self.train_Adam()
            if (epoch + 1) % 100 == 0:
                self.infos_Adam(epoch+1, loss, loss1, loss2)
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                best_loss = valid_loss if is_best else best_loss                
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_modelAC(state, is_best, save_dir=self.model_path)
            
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            loss, loss1, loss2 = self.train_LBFGS()
            if (epoch + 1) % 20 == 0:
                self.infos_LBFGS(epoch+1, loss, loss1, loss2)
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                best_loss = valid_loss if is_best else best_loss                
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_modelAC(state, is_best, save_dir=self.model_path)
    def train_Adam(self):
        self.optimizer_Adam.zero_grad()
        
        f_pred,bv_pred,bv_x_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        loss1=self.criterion(f_pred,self.f)
        loss2=10*self.criterion(iv_pred,self.iv)+self.criterion(bv_pred,self.bv)+self.criterion(bv_x_pred,self.bv)
        loss=loss1+self.lam*loss2
        
        loss.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()
        
        return loss.item(),loss1.item(),loss2.item()
    
    def infos_Adam(self,epoch,loss,loss1,loss2):
        infos = 'Adam  ' +             f'Epoch #{epoch:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.4e} = {loss1:.4e} + {self.lam} * {loss2:.4e} ' +             f'lr: {self.lr_scheduler.get_lr()[0]:.2e} '
        print(infos)
        
    def train_LBFGS(self):
        
        f_pred,bv_pred,bv_x_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        loss1=self.criterion(f_pred,self.f)
        loss2=10*self.criterion(iv_pred,self.iv)+self.criterion(bv_pred,self.bv)+self.criterion(bv_x_pred,self.bv)
        def closure():
            if torch.is_grad_enabled():
                self.optimizer_LBFGS.zero_grad()
            f_pred,bv_pred,bv_x_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
            loss1=self.criterion(f_pred,self.f)
            loss2=self.criterion(iv_pred,self.iv)+self.criterion(bv_pred,self.bv)+self.criterion(bv_x_pred,self.bv)
            loss = loss1 + self.lam * loss2
            
            if loss.requires_grad:
                loss.backward()
            return loss
        self.optimizer_LBFGS.step(closure)
        loss=closure()
        
        return loss.item(), loss1.item(), loss2.item()
    
    def infos_LBFGS(self, epoch, loss, loss1, loss2):
        infos = 'LBFGS ' +             f'Epoch #{epoch:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.2e} = {loss1:.2e} + {self.lam:d} * {loss2:.2e} '
        print(infos)
        
    def validate(self, epoch):
        self.model.eval()
        f_pred,bv_pred,bv_x_pred,iv_pred=self.model(self.problem,self.p,self.p_bc,self.p_ic)
        loss1=self.criterion(f_pred,self.f)
        loss2=10*self.criterion(iv_pred,self.iv)+self.criterion(bv_pred,self.bv)+self.criterion(bv_x_pred,self.bv)
        loss=loss1+self.lam*loss2
        infos = 'Valid ' +             f'Epoch #{epoch+1:5d}/{self.epochs_Adam+self.epochs_LBFGS} ' +             f'Loss: {loss:.4e} '
        print(infos)
        self.model.train()
        return loss.item()


# In[ ]:


# 使用Trainer_AC进行训练
Problem=Problem_AC()
args=Options_AC().parse()
args.problem=Problem_AC()
args.model = PINN_AC(dim_in=2,
                  dim_out=1,
                  dim_hidden=args.dim_hidden,
                  hidden_layers=args.hidden_layers,
                  act_name='sigmoid')
if args.sample_method == 'uniform':
    args.trainset = Trainset_AC(args.problem, args.n_t, args.n_x, method='uniform')
elif args.sample_method == 'lhs':
    args.trainset = Trainset_AC(args.problem, args.n, args.n_bc,args.n_ic, method='lhs')
    
trainer_AC = Trainer_AC(args)
trainer_AC.train()


# In[33]:


class Tester_AC(object):
    def __init__(self,args):
        self.device  = args.device
        self.problem = args.problem
        self.criterion = nn.MSELoss()
        self.model = args.model
        model_name = self.model.__class__.__name__
        model_path = os.path.join('checkpointsAC',
                                  model_name,
                                  'best_modelAC.pth.tar')
        best_model = torch.load(model_path)
        self.model.load_state_dict(best_model['state_dict'])        
        self.model.to(self.device)
        
        self.p,self.t,self.x=args.testset(verbose='tensor')
    def predict(self):
        self.model.eval()
        u_pred=self.model(self.problem,self.p)
        u_pred=u_pred.detach().cpu().numpy()
        u_pred=u_pred.reshape(self.t.shape)
        
        plt.figure(figsize=(10,3),frameon=False)
        plt.contourf(self.t,self.x,u_pred,levels=1000,cmap='seismic')
        plt.show


# ### 3.6、测试过程

# In[34]:


# 使用Tester_AC进行预测
args=Options_AC().parse()
args.problem=Problem_AC()

args.model = PINN_AC(dim_in=2,
                  dim_out=1,
                  dim_hidden=args.dim_hidden,
                  hidden_layers=args.hidden_layers,
                  act_name='sigmoid')
args.testset = Testset_AC(args.problem, 100, 100)  
tester = Tester_AC(args)
tester.predict()

