import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
import logging
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from net import CNNF
from torch.utils.tensorboard import SummaryWriter
from datasets import BasicDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

epochs = 100
lr = 0.01
val_percent = 0.05
BS = 1024
# using tensorboard: tensorboard --logdir=runs 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 'STFANPOINT', 'STFANPOLYG', 'STRUNOUTPOINT','STRUNOUTPOLYGON', 'STSOURCEPOINT', 'STSOURCEPOLYGON'
dir_checkpoint = r'checkpoints'
dir_data = r'csv\STRUNOUTPOLYGON.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = BasicDataset(dir_data)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
torch.manual_seed(0)  
# torch.cuda.manual_seed(1)  
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)

writer = SummaryWriter()
model = CNNF()
model.to(device=device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.003)
loss_func = torch.nn.MSELoss()  
Acc_best = 0.5
for epoch in range(epochs):
    train_loss, train_count = 0, 0
    model.train()
    for batch in train_loader:
        data = batch["data"]
        label = batch["label"]
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        pred = model(data)
        optimizer.zero_grad()
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_count += 1
    # writer.add_scalar("Loss/train", train_loss / train_count, epoch + 1)
    # writer.add_scalar('Acc/train', acc, train_step)
    writer.add_graph(model, data)
    logging.info(f"[Epoch {epoch + 1}/{epochs}] Train loss:{(train_loss/train_count):f}")
    

    val_loss, val_count = 0, 0
    preds, labels = [], []
    model.eval()
    for batch in val_loader:
        data = batch["data"]
        label_ = batch["label"]
        data = data.to(device=device, dtype=torch.float32)
        label = label_.to(device=device, dtype=torch.float32)
        pred = model(data)
        loss = loss_func(pred, label)
        val_loss += loss.item()
        val_count += 1
        
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
        labels.append(label_)

    preds = np.concatenate(preds,axis=0)
    labels = np.concatenate(labels,axis=0)
    FPR, TPR, thresholds = roc_curve(labels, preds)
    roc_auc = auc(FPR, TPR)
    print(roc_auc)

    logging.info(f"[Epoch {epoch + 1}/{epochs}] Test loss:{(val_loss/val_count):f}")

    if True:
        try:
            os.mkdir(dir_checkpoint)
            logging.info("Created checkpoint directory")
        except OSError:
            pass
        if roc_auc > Acc_best:
            Acc_best = roc_auc
            file_name = os.path.split(dir_data)[-1].split('.')[0]
            cp_name = file_name + "_" + repr(epoch+1)+'_auc_'+repr(int(roc_auc*10000))+'.pth'
            torch.save(model.state_dict(), os.path.join(dir_checkpoint, cp_name))

    