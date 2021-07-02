#%%
"""
Created on Thu May  6 23:22:32 2021

@author: klepy
"""
import math
import torch
import random
from torch.nn.modules.module import T
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import utils
from data import IRSDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


#%% Functions
def model_MSE(test_loader):
    criterion = nn.MSELoss(reduction='sum')
    valid_loss_d = 0
    valid_loss_c = 0
    for x_i, y_i in test_loader:
        x_i =x_i.float()
        y_i = y_i.float()
        y_i_d = y_i[:,:,:,0]
        y_i_c = y_i[:,:,:,1::]
        pred = model(x_i)
        pred_d = pred[:,:,:,0]
        pred_c = pred[:,:,:,1::]
        loss_d = criterion(pred_d,y_i_d)  / (x_i.shape[0])
        loss_c = criterion(pred_c,y_i_c)  / (x_i.shape[0])
        valid_loss_d += float(loss_d)
        valid_loss_c += float(loss_c)
           
    valid_loss_d = valid_loss_d / len(test_loader)
    valid_loss_c = valid_loss_c / len(test_loader)

    MSE_d = 10*math.log10(valid_loss_d)
    MSE_c = 10*math.log10(valid_loss_c)
    # print("-Model- ")
    # print("Direct MSE: %f" % (MSE_d))
    # print("Cascaded MSE: %f" % (MSE_c))
    return MSE_d, MSE_c

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def LS(test_loader_ls):
    criterion = nn.MSELoss(reduction='sum')
    valid_loss_d = 0
    valid_loss_c = 0
    for data, target in test_loader_ls:
        data =data.float()
        x_i_d = data[:,:,:,0]
        x_i_c = data[:,:,:,1::]
        target =target.float()
        y_i_d = target[:,:,:,0]
        y_i_c = target[:,:,:,1::]
        loss_d = criterion(x_i_d,y_i_d)  / (data.shape[0])
        loss_c = criterion(x_i_c,y_i_c)  / (data.shape[0])
        valid_loss_d += float(loss_d)
        valid_loss_c += float(loss_c)
           
    valid_loss_d = valid_loss_d / len(test_loader_ls)
    valid_loss_c = valid_loss_c / len(test_loader_ls)

    MSE_d = 10*math.log10(valid_loss_d)
    MSE_c = 10*math.log10(valid_loss_c)
    # print("-LS- ")
    # print("Direct MSE: %f" % (MSE_d))
    # print("Cascaded MSE: %f" % (MSE_c))
    return MSE_d,MSE_c

def reverse(pred,ans):
    x = torch.zeros_like(pred)
    x.cuda()
    for i in range(pred.shape[0]):
        r_max = torch.max(ans[i,0,:,:])
        r_min = torch.min(ans[i,0,:,:])
        i_max = torch.max(ans[i,1,:,:])
        i_min = torch.min(ans[i,1,:,:])
        x[i,0,:,:] = pred[i,0,:,:] * (r_max-r_min) + r_min
        x[i,1,:,:] = pred[i,1,:,:] * (i_max-i_min) + i_min
    return x

def test_min(test_loader):
    criterion = nn.MSELoss(reduction='sum')
    valid_loss_d = 0
    valid_loss_c = 0
    for x_i, y_i in test_loader:
        x_i =x_i.float()
        y_i_d = y_i[:,:,:,0]
        y_i_c = y_i[:,:,:,1::]
        pred = model(x_i)
        pred_rescale = reverse(pred,y_i)    # y_i original
        pred_d = pred_rescale[:,:,:,0]
        pred_c = pred_rescale[:,:,:,1::]
        loss_d = criterion(pred_d,y_i_d)  / (x_i.shape[0])
        loss_c = criterion(pred_c,y_i_c)  / (x_i.shape[0])
        valid_loss_d += float(loss_d)
        valid_loss_c += float(loss_c)
            
    valid_loss_d = valid_loss_d / len(test_loader)
    valid_loss_c = valid_loss_c / len(test_loader)

    MSE_d = 10*math.log10(valid_loss_d)
    MSE_c = 10*math.log10(valid_loss_c)
    return MSE_d,MSE_c                          

def range_test(M,K,rho,scale_idx):
    mse_ls_d = []
    mse_ls_c = []
    mse_cnn_d = []
    mse_cnn_c =[]
    for snr in range(-10,6,1):
        print("SNR : %d dB"%(snr))
        test_real = pd.read_csv('data/test/DnCNN_real_%.1f_%d_%d_%d_dB.csv' % (rho,M,K,snr))
        test_imag = pd.read_csv('data/test/DnCNN_imag_%.1f_%d_%d_%d_dB.csv' % (rho,M,K,snr))
        x_test_,y_test_ = utils.prepare(test_real,test_imag)
        test_loader_ls = DataLoader(dataset=IRSDataset(x_test_,y_test_), batch_size=100, shuffle=False,num_workers=0)
        mse_LS_dir, mse_LS_cas = LS(test_loader_ls)
        if scale_idx == 2:
            x_test = utils.minmax(x_test_)
            test_loader = DataLoader(dataset=IRSDataset(x_test,y_test_), batch_size=100, shuffle=False,num_workers=0)
            mse_cnn_dir, mse_cnn_cas = test_min(test_loader)
        elif scale_idx == 0 :
            x_test = x_test_
            test_loader = DataLoader(dataset=IRSDataset(x_test,y_test_), batch_size=100, shuffle=False,num_workers=0)
            mse_cnn_dir, mse_cnn_cas = model_MSE(test_loader)
        mse_ls_d.append(mse_LS_dir)
        mse_ls_c.append(mse_LS_cas)
        mse_cnn_d.append(mse_cnn_dir)
        mse_cnn_c.append(mse_cnn_cas)
    return mse_ls_d,mse_ls_c,mse_cnn_d,mse_cnn_c

def feature_scale(norm_idx):
    if norm_idx ==1:
        print(">>Standarization ")
        m_r,m_i,std_r,std_i = utils.stat(x_train_) 
        x_train = utils.standard(x_train_,m_r,m_i,std_r,std_i)
        x_val = utils.standard(x_val_,m_r,m_i,std_r,std_i)
        y_train = y_train_
        y_val = y_val_
    elif norm_idx == 2:
        print(">>Minmax Scaling") 
        x_train = utils.minmax(x_train_)
        x_val = utils.minmax(x_val_)
        y_train = utils.minmax(y_train_)
        y_val = utils.minmax(y_val_)
    elif norm_idx ==3:
        print(">>Minmax column") 
        x_train = utils.minmax_col(x_train_)
        x_val = utils.minmax_col(x_val_)
        y_train = utils.minmax_col(y_train_)
        y_val = utils.minmax_col(y_val_)
    else:
        print(">>Raw Data") 
        x_train = x_train_
        y_train = y_train_
        x_val = x_val_  
        y_val = y_val_
    return x_train,y_train,x_val,y_val

def save_model():
    PATH = './model/'
    torch.save(model, PATH + 'model.pt' )  # 전체 모델 저장
    torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, PATH + 'all.tar')


#%% [1] Load Train Data
print("> Load CSV ")
#----System Model-------#
train_snr = 0
M = 10
K = 10
rho = 0.6
# Fixed SNR
train_real = pd.read_csv('data/train/DnCNN_real_%.1f_%d_%d_%d_dB.csv' % (rho,M,K,train_snr))
train_imag = pd.read_csv('data/train/DnCNN_imag_%.1f_%d_%d_%d_dB.csv'% (rho,M,K,train_snr))

val_real = pd.read_csv('data/valid/DnCNN_real_%.1f_%d_%d_%d_dB.csv' % (rho,M,K,train_snr))
val_imag = pd.read_csv('data/valid/DnCNN_imag_%.1f_%d_%d_%d_dB.csv'% (rho,M,K,train_snr))


#%% [2] Preprocessing & Dataloader
x_train_,y_train_ = utils.prepare(train_real,train_imag)
x_val_,y_val_ = utils.prepare(val_real,val_imag)

scale_idx = 0
x_train,y_train,x_val,y_val = feature_scale(scale_idx)

random_seed = 2
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

train_loader = DataLoader(dataset=IRSDataset(x_train,y_train), batch_size=100, shuffle=True,num_workers=0)
valid_loader = DataLoader(dataset=IRSDataset(x_val,y_val), batch_size=100, shuffle=True,num_workers=0)

#%% [3] Create Model and Start Train
from model import DnCNN
import matplotlib.pyplot as plt
import utils
from seqmodel import DNN

patience = 5
num_epochs = 20
cnt = 0
print("> Create Model...")

model = DnCNN(8,4)
model.cuda()
# model.apply(utils.weigths_init_kaiming)
optimizer = optim.Adam(model.parameters(),lr=0.001)
   

print("> Start Training...")
train_history, valid_history = [], []
criterion = nn.MSELoss(reduction='sum')
# criterion = nn.MSELoss(reduction='mean')
criterion.cuda()

for i in range(num_epochs):
    model.train()
    train_loss, valid_loss = 0, 0
    y_hat = []
    # train_batch start
    for x_i, y_i in train_loader:   
        x_i =x_i.float()
        y_i =y_i.float()
        pred = model(x_i)
        loss = criterion(y_i,pred)  / (x_i.shape[0])

        
        optimizer.zero_grad()
        loss.backward()

        # plot_grad_flow(model.named_parameters())

        optimizer.step()        
        train_loss += float(loss) # This is very important to prevent memory leak.
    train_loss = train_loss / len(train_loader)
    
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        
        for x_i_val, y_i_val in valid_loader:
            x_i_val =x_i_val.float()
            y_i_val =y_i_val.float()
            pred_val = model(x_i_val)
            loss = criterion(y_i_val,pred_val)  / (x_i.shape[0])

            valid_loss += float(loss)
            
            y_hat += [pred_val-y_i_val]
            
    valid_loss = valid_loss / len(valid_loader)
    
    train_history.append(train_loss)
    valid_history.append(valid_loss)
    print('Epoch %d: train loss=%.4e  valid_loss=%.4e' % (i, train_loss, valid_loss))

    # # Early Stop
    # if i>1:
    #     if valid_history[i] < min(valid_history):
    #         cnt +=1
    #         if cnt ==patience:
    #             print("--Early Stop Activated--")
    #             break    
    #         # else:
    #         #     cnt = 0
    #         #     continue

    
    # if valid_loss <= lowest_loss:
    #     lowest_loss = valid_loss
    #     lowest_epoch = i
    #     best_model = torch.clone(model.state_dict())

    # model.load_state_dict(best_model)
        
fig, loss_ax = plt.subplots()
loss_ax.plot(train_history, 'y', label='train loss')
loss_ax.plot(valid_history, 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.set_yscale('log')
loss_ax.legend(loc='upper right')
scaling = ['Raw','Standardization','Minmax','minmax col']
loss_ax.set_title('%s' %(scaling[scale_idx]))
plt.show()

#%% [4] Model Test & Plot

mse_ls_d,mse_ls_c,mse_cnn_d,mse_cnn_c = range_test(M,K,rho,scale_idx)

fig, loss_ax = plt.subplots()
loss_ax.plot(mse_ls_d, 'yo-', label='LS')
loss_ax.plot(mse_cnn_d, 'rs-', label='DnCNN')
loss_ax.set_xlabel('SNR')
loss_ax.set_ylabel('MSE(dB)')
loss_ax.legend(loc='upper right')
loss_ax.set_title('Direct Channel (K=%d)'% (K))
plt.show()

fig, loss_ax = plt.subplots()
loss_ax.plot(mse_ls_c, 'yo-', label='LS')
loss_ax.plot(mse_cnn_c, 'rs-', label='DnCNN')
loss_ax.set_xlabel('SNR')
loss_ax.set_ylabel('MSE(dB)')
loss_ax.legend(loc='upper right')
loss_ax.set_title('Cascaded Channel (K=%d)' % (K))
plt.show()


# %% Model Param Weigths
for param in model.parameters():
  print(param.data)

#%% [5] Load model and Test
PATH = './model/K_10/'

model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()

M=10
K=10
rho=0.9
mse_ls_d,mse_ls_c,mse_cnn_d,mse_cnn_c = range_test(M,K,rho,scale_idx)

fig, loss_ax = plt.subplots()
loss_ax.plot(mse_ls_d, 'yo-', label='LS')
loss_ax.plot(mse_cnn_d, 'rs-', label='DnCNN')
loss_ax.set_xlabel('SNR')
loss_ax.set_ylabel('MSE(dB)')
loss_ax.legend(loc='upper right')
loss_ax.set_title('Direct Channel (K=%d)'% (K))
plt.show()

fig, loss_ax = plt.subplots()
loss_ax.plot(mse_ls_c, 'yo-', label='LS')
loss_ax.plot(mse_cnn_c, 'rs-', label='DnCNN')
loss_ax.set_xlabel('SNR')
loss_ax.set_ylabel('MSE(dB)')
loss_ax.legend(loc='upper right')
loss_ax.set_title('Cascaded Channel (K=%d)' % (K))
plt.show()


