import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def load_data(start_point, end_point):
  
  cld = pd.read_csv("./CRU_data/cld.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  tmn = pd.read_csv("./CRU_data/tmn.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  tmp = pd.read_csv("./CRU_data/tmp.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  tmx = pd.read_csv("./CRU_data/tmx.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  vap = pd.read_csv("./CRU_data/vap.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  pet = pd.read_csv("./CRU_data/pet.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  pre = pd.read_csv("./CRU_data/pre.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')
  data = np.concatenate((cld.reshape(-1, 9, 1), tmn.reshape(-1, 9, 1), tmp.reshape(-1, 9, 1), tmx.reshape(-1, 9, 1), vap.reshape(-1, 9, 1), pet.reshape(-1, 9, 1), pre.reshape(-1, 9, 1)), axis = 2)
  GT = pd.read_csv("./CRU_data/spei.csv", header=None).iloc[start_point:end_point, :9].values.astype('float32')

  return data, GT
      
def r_square(y_true, y_pred):
  y_true = y_true.view(-1)
  y_pred = y_pred.view(-1)
  ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
  ss_residual = torch.sum((y_true - y_pred) ** 2)
  r2 = 1 - (ss_residual / ss_total)
  return r2

def Create_dataset(data, GT, num_sample):
  X, gt = [], []
  
  for i in range(num_sample):  
    feature = data[i:i+15, :9, :7]
    for m in range(9):
      for n in range(7):
        scaler = StandardScaler()
        scaler.fit(feature[:, m, n].reshape(-1, 1))
        feature[:, m, n] = scaler.transform(feature[:, m, n].reshape(-1, 1)).reshape(-1)
    X.append(feature)
    gt.append(GT[i+15, :9])

  return torch.tensor(X).transpose(1, 2).float(), torch.tensor(gt).float()